#!/usr/bin/env python3
"""Run CUAD distractor-quantity, similarity, and interference conditions, optionally with steering.

Each target question is answered from a context of contract paragraphs ("sections"):
the paragraph that holds the gold answer plus distractor paragraphs. The load part
varies how many paragraphs are shown, K, with random distractors (settings k1, k5, ...)
or distractors similar to the gold paragraph (high_k5, ...). The interference part keeps
a small random context and adds a secondary task (settings none, extra_questions,
summary). Answers are scored with SQuAD-style token F1 and exact match and appended
to a resumable JSONL file.

With --local a transformers model generates the answers; --els_mode also records the
prompt's activation features and final-token layer vectors for build_directions.py, and
--directions with --steer_alpha steers the model during prefill.

Options are parsed and the data are loaded at import time; the functions below read
the resulting module-level tables.

Usage:
    python run.py steer cuad --model gpt-5 --data-dir /path/to/cuad --part load --k_values 1,5,20,50 --distractors random --num_variants 200
"""
from __future__ import annotations
import os
import sys
import json
import time
import hashlib
import random
import asyncio
import argparse
import re
import string
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from steering.runtime import (  # noqa: E402
    API_ATTEMPTS, API_MAX_TOKENS, AXES, DISTRACTORS, INTERFERENCE_TYPES, MAX_INPUT_TOKENS, PARTS, api_client,
    generate_local, load_completed, load_local_model, load_setting, output_path, parse_answer, save_vectors,
    steering_setting, validate_options, vectors_dir,
)
from steering.activations import question_label  # noqa: E402


# --- Answer scoring (SQuAD v1.1 normalization, token F1, exact match) ---
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace, as in SQuAD v1.1."""
    def remove_articles(text: str) -> str:
        """Replace the articles a, an, and the with spaces."""
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        """Collapse runs of whitespace to single spaces."""
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        """Drop ASCII punctuation characters."""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        """Lowercase the text."""
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between normalized prediction and gold; 1.0 when both are empty."""
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(p)
    recall = n / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    """1.0 if the normalized prediction equals the normalized gold, else 0.0."""
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def score_prediction(pred: str, gold_spans: List[str]) -> Dict:
    """Return the max F1 and EM over the gold spans, and the span that gave the best F1."""
    pred = pred or ""
    if not gold_spans:
        return {"f1": 0.0, "em": 0.0, "best_gold": ""}

    # SQuAD-style: max F1 and EM over the annotated answer strings, taken separately.
    f1s = [(token_f1(pred, g), g) for g in gold_spans]
    best_f1, best_gold = max(f1s, key=lambda x: x[0])
    ems = [exact_match(pred, g) for g in gold_spans]
    return {
        "f1": float(best_f1),
        "em": float(max(ems)),
        "best_gold": best_gold,
    }


# --- Input files and distractor seeds ---
QAS_FILE = "cuad_qas_real.jsonl"                    # One record per question (see "Load questions and paragraph chunks").
PARAGRAPHS_FILE = "cuad_paragraphs.jsonl"           # One record per contract paragraph ("chunk").
NEIGHBORS_FILE = "paragraph_neighbors_top50.jsonl"  # Each chunk's nearest chunks by embedding similarity.
# Offset added to the variant seed when drawing each kind of distractor, so each kind
# draws from its own random stream. The values are fixed so that every run, on any
# Python version, draws the same distractors.
DISTRACTOR_SEED_OFFSETS = {"random": 54972, "similar": 69574}


# --- Command-line options ---
def parse_args():
    """Parse and validate options, and check that --data-dir holds the files the selected conditions need."""
    p = argparse.ArgumentParser(prog="run.py steer cuad", description="CUAD load and interference conditions with optional residual-stream steering")
    p.add_argument("--data-dir", required=True,
                   help=f"Directory with {QAS_FILE}, {PARAGRAPHS_FILE}, and (for similar "
                        f"distractors) {NEIGHBORS_FILE}")
    p.add_argument("--model", required=True,
                   help="API model (e.g. gpt-5, claude-sonnet-4-5) or, with --local, a Hugging Face model")
    p.add_argument("--part", default="all", choices=PARTS,
                   help="Conditions to run: load (distractor quantity and similarity), interference, or all (default: all)")
    p.add_argument("--num_variants", type=int, default=200, help="Number of target questions (default: 200)")
    p.add_argument("--seed", type=int, default=42, help="Seed for target and distractor selection (default: 42)")
    p.add_argument("--outdir", default="outputs/steering/cuad", help="Output directory (default: outputs/steering/cuad)")
    p.add_argument("--max_concurrent", type=int, default=30, help="Concurrent API requests (default: 30)")
    p.add_argument("--max_rpm", type=int, default=400, help="Requests per minute; zero disables limiting (default: 400)")

    # Local model (transformers) and activation feature extraction.
    p.add_argument("--local", action="store_true", help="Run a local transformers model instead of an API")
    p.add_argument("--max_new_tokens", type=int, default=512, help="Output token cap for local generation (default: 512)")
    p.add_argument("--max_input_tokens", type=int, default=MAX_INPUT_TOKENS,
                   help="Local prompts are cut to this many tokens (default: 32768)")
    p.add_argument("--els_mode", action="store_true",
                   help="Save prompt activation features and layer vectors for build_directions.py")

    # Steering (local models only).
    p.add_argument("--directions", default=None,
                   help="Direction archive (.npz from build_directions.py), applied at its saved layers")
    p.add_argument("--steer_alpha", type=float, default=0.0,
                   help="Steering strength: h += alpha*<h,v_hat>*v_hat, then rescaled to the original norm")
    p.add_argument("--steer_axis_filter", choices=AXES, default=None,
                   help="Evaluate only one axis's setting: quantity (k50), "
                        "similarity (high_k50), or interference (extra_questions)")
    p.add_argument("--steer_test_pair_keys_json", default=None,
                   help="JSON list of qa_ids to evaluate (e.g. the held-out test split)")

    # Load part: distractor quantity x similarity
    p.add_argument("--k_values", default="1,5,20,50",
                   help="Comma-separated numbers of chunks K shown in the load part, including the gold chunk (default: 1,5,20,50)")
    p.add_argument("--distractors", default="random,similar",
                   help="Comma-separated distractor kinds for the load part: random (settings k<K>), "
                        "similar (settings high_k<K>; never chunks that answer the same clause category) (default: random,similar)")

    # Interference
    p.add_argument("--interference_k", type=int, default=5,
                   help="Number of chunks shown in interference conditions, including the gold chunk (default: 5)")
    p.add_argument("--interference_types", default="none,extra_questions,summary",
                   help="Comma-separated secondary tasks: none, extra_questions (up to two more questions), "
                        "summary (a one-sentence summary) (default: all three)")

    p.add_argument("--dry_run", action="store_true", help="Print one prompt per setting without inference")
    parsed = p.parse_args()
    validate_options(p, parsed)
    parsed.data_dir = os.path.abspath(os.path.expanduser(parsed.data_dir))
    needed = [QAS_FILE, PARAGRAPHS_FILE]
    if parsed.part in ("load", "all") and "similar" in parsed.distractors:
        needed.append(NEIGHBORS_FILE)
    for name in needed:
        if not os.path.isfile(os.path.join(parsed.data_dir, name)):
            p.error(f"--data-dir has no {name}: {parsed.data_dir}")
    # Run under PYTHONHASHSEED=0, which run.py steer sets, as in the paper's runs.
    if os.environ.get("PYTHONHASHSEED") != "0":
        p.error("run through 'python run.py steer cuad ...' or set PYTHONHASHSEED=0, "
                "as in the paper's runs")
    return parsed


# --- Parse options at import time; the data tables below depend on them ---
args = parse_args()
USE_ANTHROPIC = args.model.startswith("claude")  # Claude models use the Anthropic API, others OpenAI.


# --- Load questions and paragraph chunks ---
# QAS records: qa_id, contract_id, contract_title, category (clause type), question,
# gold_spans (answer texts), gold_chunks_global (chunks holding the answer), is_impossible.
# PARAS records: global_chunk_id, contract_id, chunk_text, one per paragraph chunk in document order.
def _load_jsonl(path):
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


print("Loading CUAD prepared data...")
QAS = _load_jsonl(os.path.join(args.data_dir, QAS_FILE))
PARAS = _load_jsonl(os.path.join(args.data_dir, PARAGRAPHS_FILE))
print(f"  {len(QAS)} QAs")
print(f"  {len(PARAS)} paragraph-chunks")

# --- Index chunks and build distractor lookup tables ---
# Chunk lookup by global id.
PARA_BY_GLOBAL: Dict[int, dict] = {p["global_chunk_id"]: p for p in PARAS}

# Similarity neighbors, most similar first (similar distractors only).
NEIGHBORS: Dict[int, List[int]] = {}
if args.part in ("load", "all") and "similar" in args.distractors:
    for r in _load_jsonl(os.path.join(args.data_dir, NEIGHBORS_FILE)):
        NEIGHBORS[r["global_chunk_id"]] = r["neighbors"]
    print(f"  {len(NEIGHBORS)} chunks with similarity neighbors")

# Reverse index: gold_chunk_id -> list of QAs whose gold span lives in that chunk.
# Used by extra_questions to pick the extra questions about distractor chunks
# shown in context.
QAS_BY_GOLD_CHUNK: Dict[int, List[dict]] = defaultdict(list)
for qa in QAS:
    if qa.get("is_impossible"):
        continue
    for gid in qa.get("gold_chunks_global") or []:
        QAS_BY_GOLD_CHUNK[gid].append(qa)

# Per-category gold-chunk index, used to filter similar distractors:
# when picking similarity-based distractors for a question of category C,
# exclude any chunk that is a gold chunk for category C in any contract, so no
# similar distractor holds an annotated clause of the question's type.
GOLD_CHUNKS_BY_CATEGORY: Dict[str, set] = defaultdict(set)
for qa in QAS:
    if qa.get("is_impossible"):
        continue
    for gid in qa.get("gold_chunks_global") or []:
        GOLD_CHUNKS_BY_CATEGORY[qa["category"]].add(gid)
print(f"  category-gold index: {len(GOLD_CHUNKS_BY_CATEGORY)} categories")


# --- Distractor selection ---
# For each question and similarity mode, one "master" list of distractors is drawn
# once, and a context with K chunks uses its first K-1 entries, so the distractors at
# a smaller K are always a subset of those at a larger K. The gold chunk sits at the
# same relative position (a fraction u of the context) at every K. Both caches are
# keyed by question and variant seed (the master list also by distractor kind and
# length), so all conditions of one variant share them.
TARGET_POSITION_U: Dict[Tuple[str, int], float] = {}
MASTER_DISTRACTORS: Dict[Tuple[str, int, str, int], List[int]] = {}


def _get_target_u(qa_id: str, variant_seed: int) -> float:
    """Cached uniform(0,1) per (question, variant) that sets the gold chunk's relative position."""
    # The +777 offset gives target placement its own random stream.
    key = (qa_id, variant_seed)
    if key not in TARGET_POSITION_U:
        TARGET_POSITION_U[key] = random.Random(variant_seed + 777).random()
    return TARGET_POSITION_U[key]


def _get_master_distractors(qa: dict, sim: str, variant_seed: int, k_max: int) -> List[int]:
    """Build/retrieve a master distractor list of length up to (k_max - 1).

    Prefixes of this list are used for smaller K, guaranteeing nesting:
    K=5 distractors = master[:4]; K=20 = master[:19]; K=50 = master[:49].
    """
    key = (qa["qa_id"], variant_seed, sim, k_max)
    if key in MASTER_DISTRACTORS:
        return MASTER_DISTRACTORS[key]

    need = k_max - 1
    # Each distractor kind draws from its own seeded stream (see DISTRACTOR_SEED_OFFSETS).
    rng = random.Random(variant_seed + DISTRACTOR_SEED_OFFSETS[sim])
    master = (_master_random if sim == "random" else _master_similar)(qa, need, rng)
    MASTER_DISTRACTORS[key] = master
    return master


def _master_random(qa: dict, need: int, rng: random.Random) -> List[int]:
    """Random chunks from contracts other than the target's: the first `need` of a seeded shuffle.

    The shuffle does not depend on `need`, so a shorter request returns a prefix of a longer one.
    """
    target_cid = qa["contract_id"]
    other_chunk_ids = [p["global_chunk_id"] for p in PARAS if p["contract_id"] != target_cid]
    rng.shuffle(other_chunk_ids)
    return other_chunk_ids[:need]


def _master_similar(qa: dict, need: int, rng: random.Random) -> List[int]:
    """The gold chunk's nearest neighbors in similarity order, topped up with random other-contract chunks.

    Chunks that answer a question of the target's clause category are excluded, so a
    similar distractor never holds an annotated answer of the question's clause category.
    """
    gold_global = qa["gold_chunks_global"][0]
    same_cat_gold = GOLD_CHUNKS_BY_CATEGORY.get(qa["category"], set())
    nbrs = [g for g in NEIGHBORS.get(gold_global, []) if g not in same_cat_gold]
    # Too few neighbors: append chunks from a shuffle of other contracts' chunks, drawn
    # with this kind's stream, that are not already listed.
    if len(nbrs) < need:
        existing = set(nbrs)
        # Draw five times the needed count, leaving room for skipped chunks.
        topup_pool = _master_random(qa, need * 5, rng)
        for g in topup_pool:
            if g in existing or g in same_cat_gold:
                continue
            nbrs.append(g)
            if len(nbrs) >= need:
                break
    return list(nbrs[:need])


# --- Target questions ---
# Targets: answerable questions with exactly one gold chunk, so paired comparisons are well defined.
def select_targets(num_variants: int, seed: int) -> List[dict]:
    """Shuffle the eligible questions with `seed` and keep the first `num_variants`.

    A target's position in this list is its variant index; a larger `num_variants`
    extends the same list, so indices stay fixed across runs with the same seed.
    """
    rng = random.Random(seed)
    pool = [qa for qa in QAS
            if not qa.get("is_impossible") and len(qa.get("gold_chunks_global") or []) == 1]
    rng.shuffle(pool)
    return pool[:num_variants]


# Prompt builders read distractors through this wrapper around the master cache.
def get_distractors_for_K(qa: dict, sim: str, K: int, variant_seed: int, k_max: int) -> List[int]:
    """Return the first K - 1 entries of the master distractor list.

    Guarantees nesting: K=5 distractors = K=20 distractors[:4] = K=50 distractors[:4].
    """
    master = _get_master_distractors(qa, sim, variant_seed, k_max=k_max)
    return list(master[: max(0, K - 1)])


# --- Prompt rendering ---
# A prompt is: PROMPT_HEADER, a blank line, the numbered sections, the question(s),
# then the JSON answer instructions (PROMPT_FOOTER or an interference variant of it).
def render_chunks(chunk_ids_in_order: List[int]) -> str:
    """Render chunks in the given order as blocks headed "=== Section i ===", numbered from 1."""
    parts = []
    for i, gid in enumerate(chunk_ids_in_order):
        p = PARA_BY_GLOBAL[gid]
        parts.append(f"=== Section {i+1} ===\n{p['chunk_text']}")
    return "\n\n".join(parts)


PROMPT_HEADER = (
    "You will read sections from one or more commercial contracts.\n"
    "Locate the clause that answers each requested question and extract the relevant text directly from the sections.\n"
)
PROMPT_FOOTER = (
    'Respond in JSON: {"answer": "extracted clause text"}\n'
    'Extract each answer directly from the section text. Be concise.'
)

def build_single_answer_prompt(context: str, question: str) -> str:
    """Assemble header, rendered sections, one question, and the single-answer JSON footer."""
    return (
        PROMPT_HEADER + "\n" +
        context + "\n\n" +
        f"Question: {question}\n\n" +
        PROMPT_FOOTER
    )


# --- Prompt builders: each returns (prompt, metadata saved with the prediction) ---
def _ordered_chunks_for_K(target_gold: int, distractors: List[int], K: int,
                          qa_id: str, variant_seed: int) -> List[int]:
    """Insert the gold chunk at a position proportional to a cached u ~ U(0,1).

    This makes target proportional position consistent across K levels:
    if u=0.4, target sits at slot 2 in K=5, slot 8 in K=20, slot 20 in K=50 (0-based).
    """
    # distractors holds K-1 chunks; inserting the gold chunk gives K. The min() keeps
    # the slot within the context.
    ids = list(distractors)
    u = _get_target_u(qa_id, variant_seed)
    pos = min(int(u * K), K - 1)
    ids.insert(pos, target_gold)
    return ids[:K]


def build_load_prompt(qa: dict, settings: dict, variant_seed: int) -> Tuple[str, dict]:
    """Quantity × similarity. K=1 = gold only; K>1 = gold + (K-1) distractors.

    Uses cached master distractor list (nesting: K=5 distractors ⊂ K=20 ⊂ K=50)
    and cached target proportional position (consistent across K).
    """
    K = int(settings["num_chunks"])
    sim = settings["distractors"]  # "random" | "similar"
    target_gold = qa["gold_chunks_global"][0]

    if K == 1:
        chunk_ids = [target_gold]
    else:
        # Cache enough distractors for the largest requested K.
        distractors = get_distractors_for_K(qa, sim, K, variant_seed, k_max=max(args.k_values))
        chunk_ids = _ordered_chunks_for_K(target_gold, distractors, K,
                                          qa["qa_id"], variant_seed)

    body = render_chunks(chunk_ids)
    prompt = build_single_answer_prompt(body, qa["question"])

    meta = {
        "part": "load",
        "setting": load_setting(K, sim),
        "num_chunks": K,
        "distractors": sim,
        "qa_id": qa["qa_id"],
        "category": qa["category"],
        "contract_id": qa["contract_id"],
        "contract_title": qa["contract_title"],
        "question": qa["question"],
        "is_impossible": qa["is_impossible"],
        "gold_spans": [s["text"] for s in qa["gold_spans"]],
        "gold_chunk_global": target_gold,
        "rendered_chunk_ids": chunk_ids,
    }
    return prompt, meta


def _qas_for_distractor_chunks(target_qa: dict, distractor_chunk_ids: List[int],
                                count: int, rng: random.Random) -> List[dict]:
    """Pick up to `count` QAs whose single gold chunk is one of the shown distractor chunks.

    Picks differ from the target and each other in contract and category. Falls back to
    answerable QAs from distractor contracts, then from any other contract.
    """
    target_cat = target_qa["category"]
    target_qa_id = target_qa["qa_id"]
    distractor_set = set(distractor_chunk_ids)

    # Prefer questions answered by a displayed distractor chunk.
    direct_candidates: List[dict] = []
    for d in distractor_chunk_ids:
        for q in QAS_BY_GOLD_CHUNK.get(d, []):
            if (q["qa_id"] != target_qa_id
                    and q["category"] != target_cat
                    and len(q.get("gold_chunks_global") or []) == 1):
                direct_candidates.append(q)

    rng.shuffle(direct_candidates)

    picked: List[dict] = []
    used_contracts = {target_qa["contract_id"]}
    used_cats = {target_cat}
    for q in direct_candidates:
        if q["contract_id"] in used_contracts:
            continue
        if q["category"] in used_cats:
            continue
        picked.append(q)
        used_contracts.add(q["contract_id"])
        used_cats.add(q["category"])
        if len(picked) >= count:
            return picked

    # Then any answerable QA from a distractor contract.
    distractor_contract_ids = {PARA_BY_GLOBAL[d]["contract_id"] for d in distractor_set}
    distractor_contract_ids.discard(target_qa["contract_id"])
    fallback = [q for q in QAS
                if q["contract_id"] in distractor_contract_ids
                and q["category"] != target_cat
                and not q["is_impossible"]
                and len(q.get("gold_chunks_global") or []) == 1]
    rng.shuffle(fallback)
    for q in fallback:
        # Re-check inside the loop, since used_contracts and used_cats grow as questions are picked.
        if q["contract_id"] in used_contracts:
            continue
        if q["category"] in used_cats:
            continue
        picked.append(q)
        used_contracts.add(q["contract_id"])
        used_cats.add(q["category"])
        if len(picked) >= count:
            return picked

    # Last, any answerable QA from any other contract.
    catch_all = [q for q in QAS
                 if q["contract_id"] != target_qa["contract_id"]
                 and q["category"] != target_cat
                 and not q["is_impossible"]
                 and len(q.get("gold_chunks_global") or []) == 1]
    rng.shuffle(catch_all)
    for q in catch_all:
        if q["contract_id"] in used_contracts:
            continue
        if q["category"] in used_cats:
            continue
        picked.append(q)
        used_contracts.add(q["contract_id"])
        used_cats.add(q["category"])
        if len(picked) >= count:
            return picked
    return picked


def build_interference_prompt(qa: dict, settings: dict, variant_seed: int) -> Tuple[str, dict]:
    """Add a secondary task to the selected contract context.

    The context is K chunks with random distractors, drawn and placed exactly as in
    the load part, so `none` renders the same prompt as k<K>.
    extra_questions asks up to two more questions about other contracts alongside the
    target, in shuffled order, and reads the target's answer from its numbered key.
    summary also asks for a one-sentence summary of the sections' common theme.
    """
    K = int(settings["num_chunks"])
    i_type = settings["interference_type"]

    target_gold = qa["gold_chunks_global"][0]
    # A random master list is a prefix of one seeded shuffle, so k_max=K gives the same
    # distractors as the load part's longer list.
    if K > 1:
        distractors = get_distractors_for_K(qa, "random", K, variant_seed, k_max=K)
    else:
        distractors = []
    chunk_ids = _ordered_chunks_for_K(target_gold, distractors, K,
                                      qa["qa_id"], variant_seed)
    body = render_chunks(chunk_ids)

    if i_type == "none":
        prompt = build_single_answer_prompt(body, qa["question"])
    elif i_type == "extra_questions":
        # Separate seeded streams (+888 to pick the extra questions, +999 to order
        # all three) keep these draws independent of distractor selection.
        rng_q = random.Random(variant_seed + 888)
        # Prefer questions answerable from shown distractor chunks; the helper
        # can fall back to questions whose gold chunks are not shown.
        distractor_only = [c for c in chunk_ids if c != target_gold]
        other_qs = _qas_for_distractor_chunks(qa, distractor_only, 2, rng_q)
        extra_questions = [{
            "qa_id": q["qa_id"],
            "category": q["category"],
            "gold_spans": [s["text"] for s in q["gold_spans"]],
        } for q in other_qs]
        question_entries = [{"question": qa["question"], "is_target": True}]
        question_entries += [{"question": q["question"], "is_target": False} for q in other_qs]
        random.Random(variant_seed + 999).shuffle(question_entries)
        # Questions are listed as Q1..Qn; the target's answer is read from answer_<its position>.
        q_lines = [f"Q{i+1}: {entry['question']}" for i, entry in enumerate(question_entries)]
        target_question_position = next(i + 1 for i, entry in enumerate(question_entries) if entry["is_target"])
        # The schema always lists three answer keys, even when fewer extra questions were found.
        schema = {"answer_1": "answer to Q1", "answer_2": "answer to Q2", "answer_3": "answer to Q3"}
        prompt = (
            PROMPT_HEADER + "\n" +
            body + "\n\n" +
            "\n".join(q_lines) + "\n\n" +
            "Respond in JSON: " + json.dumps(schema) + "\n"
            "Extract each answer directly from the section text. Be concise."
        )
    elif i_type == "summary":
        schema = {"answer": "extracted clause text", "summary": "one sentence"}
        prompt = (
            PROMPT_HEADER + "\n" +
            body + "\n\n" +
            f"Question: {qa['question']}\n\n" +
            "Additionally, in one sentence (max 30 words), describe the common theme or topic "
            "across all the sections above. Output this under \"summary\" in your JSON response.\n"
            "\n"
            "Respond in JSON: " + json.dumps(schema) + "\n"
            "Extract each answer directly from the section text. Be concise."
        )
    else:
        raise ValueError(f"Unknown interference_type: {i_type}")

    meta = {
        "part": "interference",
        "setting": i_type,
        "interference_type": i_type,
        "num_chunks": K,
        "qa_id": qa["qa_id"],
        "category": qa["category"],
        "contract_id": qa["contract_id"],
        "contract_title": qa["contract_title"],
        "question": qa["question"],
        "is_impossible": qa["is_impossible"],
        "gold_spans": [s["text"] for s in qa["gold_spans"]],
        "gold_chunk_global": target_gold,
        "rendered_chunk_ids": chunk_ids,
    }
    if i_type == "extra_questions":
        meta["extra_questions"] = extra_questions
        meta["target_question_position"] = target_question_position
        meta["target_answer_key"] = f"answer_{target_question_position}"
    return prompt, meta


# --- Condition grid ---
PROMPT_BUILDERS = {
    "load": build_load_prompt,
    "interference": build_interference_prompt,
}


def build_conditions() -> List[Tuple[str, dict]]:
    """Return (setting, settings) pairs for the parts selected by --part."""
    # Settings follow a fixed kind and type order, whatever order the options list them in.
    conds: List[Tuple[str, dict]] = []
    if args.part in ("load", "all"):
        for K in args.k_values:
            # K=1 shows no distractors, so it is one setting whatever --distractors lists.
            for sim in (["random"] if K == 1 else [d for d in DISTRACTORS if d in args.distractors]):
                conds.append((load_setting(K, sim), {"part": "load", "num_chunks": K, "distractors": sim}))
    if args.part in ("interference", "all"):
        for i_type in INTERFERENCE_TYPES:
            if i_type in args.interference_types:
                conds.append((i_type, {"part": "interference", "num_chunks": args.interference_k,
                                       "interference_type": i_type}))
    return conds


# --- Inference backends ---
# Local model and tokenizer, loaded in main() when --local is set.
local_model = None
local_tokenizer = None

# Async API client, concurrency limit, and request times of the last minute for rate limiting.
async_client = None
sem = None
rpm_lock = asyncio.Lock()
rpm_timestamps: List[float] = []


def init_async_client():
    """Create the Anthropic (Claude models) or OpenAI async client and the concurrency semaphore."""
    global async_client, sem
    async_client = api_client(USE_ANTHROPIC)
    sem = asyncio.Semaphore(args.max_concurrent)
    print(f"{'Anthropic' if USE_ANTHROPIC else 'OpenAI'} async client ready (model={args.model}, "
          f"{args.max_rpm} rpm, {args.max_concurrent} concurrent)")


async def rate_limit_wait():
    """Wait until fewer than --max_rpm requests were sent in the last 60 seconds, then record this one."""
    if args.max_rpm == 0:
        return
    while True:
        async with rpm_lock:
            now = time.time()
            cutoff = now - 60.0
            while rpm_timestamps and rpm_timestamps[0] < cutoff:
                rpm_timestamps.pop(0)
            if len(rpm_timestamps) < args.max_rpm:
                rpm_timestamps.append(now)
                return
        # Over the limit: release the lock and check again in half a second.
        await asyncio.sleep(0.5)


async def call_api_async(prompt: str) -> str:
    """Send one prompt and return the raw reply text.

    Empty replies and errors are retried with exponential backoff, up to API_ATTEMPTS
    attempts in all; the last failure raises RuntimeError.
    """
    async with sem:
        for attempt in range(API_ATTEMPTS):
            await rate_limit_wait()
            try:
                if USE_ANTHROPIC:
                    resp = await async_client.messages.create(
                        model=args.model,
                        max_tokens=API_MAX_TOKENS,
                        temperature=0.0,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content = resp.content[0].text if resp.content else ""
                else:
                    resp = await async_client.chat.completions.create(
                        model=args.model,
                        max_completion_tokens=API_MAX_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content = resp.choices[0].message.content or ""
                if content.strip():
                    return content
                raise RuntimeError("Empty API response")
            except Exception as e:
                if attempt == API_ATTEMPTS - 1:
                    print(f"  API error (final attempt): {e}")
                    raise RuntimeError("CUAD inference failed: API retries exhausted") from e
                await asyncio.sleep(2 ** attempt)


# --- Worker: generate, score, and build one output record ---
def task_hash(setting: str, variant_idx: int, qa_id: str) -> str:
    """Resume key for one task: a short MD5 of setting, variant index, and question id."""
    key = f"{setting}|{variant_idx}|{qa_id}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


async def eval_one(setting: str, variant_idx: int, prompt: str,
                   meta: dict) -> Tuple[str, int, float, dict]:
    """Generate and score one prompt; return (setting, variant, F1, output record)."""
    features = vectors = None
    if args.local:
        raw, features, vectors = generate_local(
            local_model, local_tokenizer, prompt, model_name=args.model,
            max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
            q_label=question_label(setting) if args.els_mode else None)
    else:
        raw = await call_api_async(prompt)
    # Multi-question prompts name the target's answer key; others use "answer".
    pred = parse_answer(raw, key=meta.get("target_answer_key", "answer"))
    sc = score_prediction(pred, meta.get("gold_spans") or [])
    # The record is the builder's metadata plus the output, its scores, and the resume key.
    # The raw output and parsed answer are kept to their first 1000 and 500 characters.
    record = {
        **meta,
        "variant_index": variant_idx,
        "raw_output": raw[:1000],
        "pred_answer": pred[:500],
        "score": sc["f1"],
        "f1": sc["f1"],
        "em": sc["em"],
        "best_gold": sc["best_gold"],
        "task_hash": task_hash(setting, variant_idx, meta["qa_id"]),
    }
    if args.directions:
        record["steer_alpha"] = float(args.steer_alpha)
        record["steer_axis"] = args.steer_axis_filter
        record["steer_directions"] = os.path.basename(args.directions)
    if features is not None:
        # --els_mode: activation features go into the record; final-token layer vectors go to a .pt file.
        record["neural_dynamics"] = features["neural_dynamics"]
        record["activation_metrics"] = features["activation_metrics"]
        save_vectors(vectors_dir(args), setting, variant_idx, vectors, score=sc["f1"])
    return setting, variant_idx, sc["f1"], record


# --- Main ---
async def main():
    """Build pending prompts, run them, append records to the output JSONL, and print per-condition F1."""
    global local_model, local_tokenizer
    # Resume: tasks whose task_hash is already in the output file are skipped.
    out_path = output_path(args, "cuad")
    done = load_completed(out_path, args.directions)
    if done:
        print(f"Resume: {len(done)} prior tasks already complete (skipping)")

    # Conditions, optionally narrowed to one steering axis's setting and a list of question ids.
    conds = build_conditions()
    print(f"\nConditions to run ({len(conds)}):")
    for setting, _ in conds:
        print(f"  {setting}")
    axis_setting = steering_setting("cuad", args.steer_axis_filter, [setting for setting, _ in conds])
    if axis_setting:
        print(f"[STEERING] Restricting to setting={axis_setting} (axis={args.steer_axis_filter})")

    keys_filter = None
    if args.steer_test_pair_keys_json:
        with open(args.steer_test_pair_keys_json, encoding="utf-8") as f:
            keys_filter = set(json.load(f))
        print(f"[STEERING] Restricting to {len(keys_filter)} listed qa_ids")

    # Build every pending prompt before any inference, so a builder error stops the run early.
    targets = select_targets(args.num_variants, args.seed)
    if keys_filter is not None:
        # Listed IDs outside the first --num_variants targets cannot be run, so warn about them.
        missing = keys_filter - {qa["qa_id"] for qa in targets}
        if missing:
            print(f"[STEERING] Warning: {len(missing)} listed qa_ids are not among the "
                  f"selected targets; increase --num_variants to include them")
    tasks: List[Tuple[str, int, str, dict]] = []
    for setting, settings in conds:
        if axis_setting and setting != axis_setting:
            continue
        builder = PROMPT_BUILDERS[settings["part"]]
        for vi, qa in enumerate(targets):
            if keys_filter is not None and qa["qa_id"] not in keys_filter:
                continue
            # Completed and excluded tasks need no prompt reconstruction.
            if task_hash(setting, vi, qa["qa_id"]) in done:
                continue
            # The variant seed depends only on --seed and the target's index, so all
            # conditions of a target share its random draws and reruns repeat them. The
            # factor keeps seeds from different --seed values apart for up to 100000 targets.
            variant_seed = args.seed * 100000 + vi
            try:
                prompt, meta = builder(qa, settings, variant_seed)
                if meta["setting"] != setting:
                    raise ValueError(f"Builder returned setting {meta['setting']!r}, expected {setting!r}")
            except Exception as error:
                raise RuntimeError(
                    f"Could not build CUAD prompt {setting} v{vi} ({qa['qa_id']}): {error}"
                ) from error
            tasks.append((setting, vi, prompt, meta))

    print(f"\n{len(tasks)} tasks to run")
    # --dry_run: show the start and end of one prompt per setting, then stop.
    if args.dry_run:
        seen_settings = set()
        for setting, vi, prompt, meta in tasks:
            if setting in seen_settings:
                continue
            seen_settings.add(setting)
            print("=" * 70)
            print(f"  setting={setting}  qa_id={meta['qa_id']}")
            print("=" * 70)
            print(prompt[:1500])
            print("...")
            print(prompt[-500:])
            print()
        return

    if not tasks:
        print("No pending tasks; nothing to run.")
        return

    # Load the model (installing steering hooks for a nonzero alpha) or the API client.
    if args.local:
        local_model, local_tokenizer = load_local_model(args.model, args.directions, args.steer_alpha)
    else:
        init_async_client()

    completed = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cond_score_sum: Dict[str, float] = defaultdict(float)
    cond_count: Dict[str, int] = defaultdict(int)

    # Append each record as soon as it finishes, so an interrupted run can resume.
    with open(out_path, "a", encoding="utf-8") as out_f:
        async def _runner(setting, vi, prompt, meta):
            """Run one task and write its record; a failure is printed and counted at the end."""
            nonlocal completed
            try:
                s, v, sc, rec = await eval_one(setting, vi, prompt, meta)
                out_f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                out_f.flush()
                cond_score_sum[s] += sc
                cond_count[s] += 1
                completed += 1
                if completed % 25 == 0:
                    print(f"  {completed}/{len(tasks)} ...")
            except Exception as e:
                print(f"  Error: {e}")

        if args.local:
            # Local generation is GPU-bound, so tasks run one at a time.
            for task in tasks:
                await _runner(*task)
        else:
            await asyncio.gather(*[_runner(*t) for t in tasks])

    print(f"\nDone. {completed} tasks written to {out_path}")
    print("\nPer-condition F1:")
    for setting in sorted(cond_score_sum):
        n = cond_count[setting]
        if n > 0:
            print(f"  {setting:40s} F1={cond_score_sum[setting]/n:.3f}  n={n}")

    # Failed tasks wrote no record, so a rerun retries exactly those.
    if completed != len(tasks):
        raise RuntimeError(f"{len(tasks) - completed} CUAD tasks failed; rerun to resume incomplete tasks.")


if __name__ == "__main__":
    asyncio.run(main())
