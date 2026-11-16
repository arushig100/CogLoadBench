#!/usr/bin/env python3
"""Run QASPER distractor-quantity, similarity, and interference conditions, optionally with steering.

Each target question about a paper is answered from a context of full papers: the
target paper plus distractor papers. The load part varies how many papers are
shown, K, and how the distractors are chosen: random dissimilar papers without the
target's answer section type (settings k1, k3, ...) or the most similar papers by
title-and-abstract embedding, again without that section type (high_k3, ...). The
interference part shows dissimilar papers and adds a secondary task (settings none,
extra_questions, summary). Answers are scored with token F1 against the gold
extractive spans and appended to a resumable JSONL file.

With --local a transformers model generates the answers; --els_mode also records the
prompt's activation features and final-token layer vectors for build_directions.py, and
--directions with --steer_alpha steers the model during prefill.

Options are parsed and the data are loaded at import time; the functions below read
the resulting module-level tables.

Usage:
    python run.py steer qasper --model gpt-5 --part all --test_file /path/to/qasper-test-v0.3.json
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import hashlib
import random
import asyncio
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from steering.qasper_evaluator import token_f1_score as _qasper_token_f1_score  # noqa: E402
from steering.runtime import (  # noqa: E402
    API_ATTEMPTS, API_MAX_TOKENS, AXES, DISTRACTORS, INTERFERENCE_TYPES, MAX_INPUT_TOKENS, PARTS, api_client,
    generate_local, load_completed, load_local_model, load_setting, output_path, parse_answer, save_vectors,
    steering_setting, validate_options, vectors_dir,
)
from steering.activations import question_label  # noqa: E402


# --- Command-line options ---
def parse_args():
    """Parse and validate options, and check that distractor embeddings are cached or can be computed."""
    p = argparse.ArgumentParser(prog="run.py steer qasper", description="QASPER load and interference conditions with optional residual-stream steering")

    p.add_argument("--model", required=True,
                   help="API model (e.g. gpt-5, claude-sonnet-4-5) or, with --local, a Hugging Face model")
    p.add_argument("--part", default="all", choices=PARTS,
                   help="Conditions to run: load (distractor quantity and similarity), interference, or all (default: all)")
    p.add_argument("--num_variants", type=int, default=200,
                   help="Number of questions to evaluate per condition (default: 200)")
    p.add_argument("--outdir", default="outputs/steering/qasper", help="Output directory (default: outputs/steering/qasper)")
    p.add_argument("--embedding-cache-dir", default=None,
                   help="Paper embedding cache (embeddings.npy, paper_ids.json); defaults to <outdir>/embeddings. "
                        "Missing embeddings are computed with OpenAI text-embedding-3-small.")
    p.add_argument("--seed", type=int, default=42, help="Seed for target and distractor selection (default: 42)")
    p.add_argument("--test_file", required=True, help="Official QASPER test JSON (qasper-test-v0.3.json)")

    # Condition settings
    p.add_argument("--k_values", default="1,3,5,7",
                   help="Comma-separated numbers of papers K shown in the load part, including the target paper; "
                        "distractor pools are sized for the largest (default: 1,3,5,7)")
    p.add_argument("--distractors", default="random,similar",
                   help="Comma-separated distractor kinds for the load part: random (settings k<K>, random "
                        "dissimilar papers), similar (settings high_k<K>, the most similar papers); both use "
                        "the paper embeddings (default: random,similar)")
    p.add_argument("--interference_k", type=int, default=5,
                   help="Number of papers shown in interference conditions, including the target paper (default: 5)")
    p.add_argument("--interference_types", default="none,extra_questions,summary",
                   help="Comma-separated secondary tasks: none, extra_questions (up to two more questions), "
                        "summary (a one-sentence summary) (default: all three)")

    p.add_argument("--max_rpm", type=int, default=500, help="Requests per minute; zero disables limiting (default: 500)")
    p.add_argument("--max_concurrent", type=int, default=50, help="Concurrent API requests (default: 50)")

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
                   help="Evaluate only one axis's setting: quantity (k5), "
                        "similarity (high_k5), or interference (extra_questions)")
    p.add_argument("--steer_test_pair_keys_json", default=None,
                   help="JSON list of question_ids to evaluate (e.g. the held-out test split)")

    p.add_argument("--dry_run", action="store_true",
                   help="Print one prompt per setting without inference; needs cached embeddings")

    parsed = p.parse_args()
    validate_options(p, parsed)
    parsed.test_file = os.path.abspath(os.path.expanduser(parsed.test_file))
    if not os.path.isfile(parsed.test_file):
        p.error(f"--test_file: no such file: {parsed.test_file}")
    parsed.embedding_cache_dir = os.path.abspath(os.path.expanduser(
        parsed.embedding_cache_dir or os.path.join(parsed.outdir, "embeddings")))
    # Every prompt except k1 draws distractors by embedding similarity.
    needs_embeddings = parsed.part in ("interference", "all") or max(parsed.k_values) > 1
    cached = all(os.path.isfile(os.path.join(parsed.embedding_cache_dir, name))
                 for name in ("embeddings.npy", "paper_ids.json"))
    if needs_embeddings and not cached and (parsed.dry_run or not os.environ.get("OPENAI_API_KEY")):
        p.error(f"Distractors need embeddings.npy and paper_ids.json in {parsed.embedding_cache_dir}"
                + ("" if parsed.dry_run else ", or OPENAI_API_KEY to compute them"))
    # Set iteration order (hence distractor choice) follows the string hash, which depends
    # on PYTHONHASHSEED and the Python version; the paper's runs used Python 3.11 with
    # PYTHONHASHSEED=0, which run.py steer sets.
    if os.environ.get("PYTHONHASHSEED") != "0":
        p.error("run through 'python run.py steer qasper ...' or set PYTHONHASHSEED=0, "
                "so the distractors match the paper's runs")
    if sys.version_info < (3, 11):
        p.error("use Python 3.11 or later, so the distractors match the paper's runs")
    return parsed


# --- Parse options at import time; the data tables below depend on them ---
args = parse_args()
USE_ANTHROPIC = args.model.startswith("claude")  # Claude models use the Anthropic API, others OpenAI.


# --- Load the QASPER test set ---
# ALL_PAPERS maps paper id -> {title, abstract, full_text: [{section_name, paragraphs}], qas}.
print("Loading QASPER test data...")
with open(args.test_file, "r", encoding="utf-8") as f:
    ALL_PAPERS = json.load(f)

PAPER_IDS = list(ALL_PAPERS.keys())
print(f"  {len(PAPER_IDS)} papers loaded")


# --- Section classification ---
# Coarse section types, used to keep distractors from sharing the section type that
# holds the target's answer. Headings not listed map to "Other".
SECTION_TYPE_MAP = {
    "introduction": "Introduction",
    "related work": "Related Work",
    "related works": "Related Work",
    "background": "Related Work",
    "methods": "Methods",
    "method": "Methods",
    "methodology": "Methods",
    "model": "Methods",
    "approach": "Methods",
    "proposed method": "Methods",
    "experiments": "Experiments",
    "experimental setup": "Experiments",
    "experiment": "Experiments",
    "evaluation": "Experiments",
    "results": "Results",
    "results and discussion": "Results",
    "results and analysis": "Results",
    "analysis": "Results",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "conclusion and future work": "Conclusion",
    "conclusions and future work": "Conclusion",
    "future work": "Conclusion",
}


def classify_section(name: str) -> str:
    """Map a section heading, lowercased and without leading numbering, to its coarse type."""
    if not name:
        return "Other"
    key = name.strip().lower()
    # Drop one leading number and the dots or spaces after it, e.g. "4. results" -> "results".
    key = re.sub(r'^\d+[\.\s]+', '', key).strip()
    return SECTION_TYPE_MAP.get(key, "Other")


def get_paper_section_types(paper: dict) -> set:
    """Return the set of coarse section types in a paper."""
    return {classify_section(sec["section_name"]) for sec in paper["full_text"]}


# --- Question selection ---
def get_qa_info(qa: dict) -> Optional[dict]:
    """Return question, id, answer type, and gold spans for an extractive question, else None.

    Each annotator's answer is typed as unanswerable, extractive, yes/no, or free form;
    a question is kept when the most common type is extractive. The gold spans pool
    every annotator's extractive spans, without duplicates.
    """
    question = qa["question"]
    question_id = qa.get("question_id", hashlib.md5(question.encode()).hexdigest())

    answer_types = []
    extractive_spans = []

    for ann in qa["answers"]:
        a = ann["answer"]
        if a.get("unanswerable", False):
            answer_types.append("unanswerable")
        elif a.get("extractive_spans") and len(a["extractive_spans"]) > 0:
            answer_types.append("extractive")
            extractive_spans.extend(a["extractive_spans"])
        elif a.get("yes_no") is not None:
            answer_types.append("yes_no")
        elif a.get("free_form_answer", "").strip():
            answer_types.append("free_form")

    if not answer_types:
        return None

    type_counts = Counter(answer_types)
    answer_type = type_counts.most_common(1)[0][0]

    if answer_type != "extractive":
        return None

    # The order of a set of strings follows the string hash, which PYTHONHASHSEED fixes
    # (see parse_args); find_answer_section_type uses this order.
    gold = {"extractive_spans": list(set(extractive_spans))}

    return {
        "question": question,
        "question_id": question_id,
        "answer_type": answer_type,
        "gold": gold,
    }


# Index the extractive questions, in file order, overall and by paper.
print("Building QA index...")
ALL_QAS = []  # list of (paper_id, qa_info)
# Questions per paper, from which extra_questions draws its extra questions.
PAPER_QAS: Dict[str, List[dict]] = defaultdict(list)

for pid in PAPER_IDS:
    paper = ALL_PAPERS[pid]
    for qa in paper["qas"]:
        info = get_qa_info(qa)
        if info is not None:
            ALL_QAS.append((pid, info))
            PAPER_QAS[pid].append(info)

print(f"  {len(ALL_QAS)} extractive questions indexed")
print(f"  {len(PAPER_QAS)} papers with questions")


# --- Paper embeddings for distractor similarity ---
# One embedding of "title. abstract" per paper, cached as embeddings.npy with the
# matching paper_ids.json, and loaded only when a pending prompt needs distractors.
EMBEDDINGS_PATH = os.path.join(args.embedding_cache_dir, "embeddings.npy")
PAPER_IDS_PATH = os.path.join(args.embedding_cache_dir, "paper_ids.json")


def compute_embeddings():
    """Compute paper embeddings using OpenAI text-embedding-3-small."""
    from openai import OpenAI
    client = OpenAI()

    texts = []
    ids = []
    for pid in PAPER_IDS:
        paper = ALL_PAPERS[pid]
        text = f"{paper['title']}. {paper['abstract']}"
        texts.append(text)
        ids.append(pid)

    print(f"Computing embeddings for {len(texts)} papers...")
    all_embeddings = []
    batch_size = 512  # Texts per embeddings request.
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        for item in resp.data:
            all_embeddings.append(item.embedding)
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    emb_array = np.array(all_embeddings, dtype=np.float32)
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.save(EMBEDDINGS_PATH, emb_array)
    with open(PAPER_IDS_PATH, "w") as f:
        json.dump(ids, f)

    print(f"  Saved embeddings to {EMBEDDINGS_PATH}")
    return emb_array, ids


def load_embeddings():
    """Return (embeddings, paper ids) from the cache, computing and caching them if absent."""
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(PAPER_IDS_PATH):
        print("Loading cached embeddings...")
        emb = np.load(EMBEDDINGS_PATH, allow_pickle=False)
        with open(PAPER_IDS_PATH) as f:
            ids = json.load(f)
        print(f"  Loaded {len(ids)} embeddings")
        return emb, ids
    return compute_embeddings()


# Unit-normalized embeddings (rows in EMB_PAPER_IDS order), filled by ensure_embeddings().
EMBEDDINGS_NORM = None
EMB_PAPER_IDS = []
PID_TO_IDX = {}


def ensure_embeddings():
    """Load embeddings only when pending prompts actually need distractors."""
    global EMBEDDINGS_NORM, EMB_PAPER_IDS, PID_TO_IDX
    if EMBEDDINGS_NORM is not None:
        return
    embeddings, ids = load_embeddings()
    if (not isinstance(ids, list) or len(ids) != len(set(ids))
            or set(ids) != set(PAPER_IDS) or embeddings.ndim != 2
            or len(embeddings) != len(ids) or not np.isfinite(embeddings).all()):
        raise ValueError("Embedding cache must contain one finite vector for each paper in --test_file.")
    # Unit rows, so a matrix-vector product gives cosine similarities; 1e-9 avoids division by zero.
    EMBEDDINGS_NORM = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    EMB_PAPER_IDS = ids
    PID_TO_IDX = {pid: i for i, pid in enumerate(ids)}


def get_most_similar_papers(target_pid: str, count: int) -> List[str]:
    """The count papers most similar to the target by embedding cosine similarity."""
    if count <= 0:
        return []
    ensure_embeddings()
    idx = PID_TO_IDX[target_pid]
    sims = EMBEDDINGS_NORM @ EMBEDDINGS_NORM[idx]
    sims[idx] = -1  # Exclude the target itself.
    top_indices = np.argsort(sims)[::-1][:count]
    return [EMB_PAPER_IDS[i] for i in top_indices]


def get_dissimilar_papers(target_pid: str, count: int, rng: random.Random) -> List[str]:
    """Random papers from the least similar 10% of papers (at least 5 * count of them)."""
    if count <= 0:
        return []
    ensure_embeddings()
    idx = PID_TO_IDX[target_pid]
    sims = EMBEDDINGS_NORM @ EMBEDDINGS_NORM[idx]
    sims[idx] = 2  # Above any cosine, so the target sorts last.
    n_bottom = max(count * 5, len(sims) // 10)
    bottom_indices = np.argsort(sims)[:n_bottom]
    candidates = [EMB_PAPER_IDS[i] for i in bottom_indices]
    rng.shuffle(candidates)
    return candidates[:count]


# --- Section type that holds the answer ---
def find_answer_section_type(paper: dict, qa_info: dict) -> str:
    """Section type of the paper section that contains the answer.

    Gold spans are tried in list order, and the first section whose text contains
    one (case-insensitively) gives the type; "Results" if no span is found.
    """
    for span in qa_info["gold"]["extractive_spans"]:
        span_lower = span.strip().lower()
        for sec in paper["full_text"]:
            full_sec_text = " ".join(sec["paragraphs"]).lower()
            if span_lower in full_sec_text:
                return classify_section(sec["section_name"])
    return "Results"


# --- Distractor selection: random and similar ---
def get_random_distractors(target_pid: str, target_sec_type: str,
                           count: int, rng: random.Random) -> List[str]:
    """Papers dissimilar to the target in both topic and answer section type.

    Random papers from the least similar 20% by embedding (at least 10 * count) that
    have no section of the target's answer-section type. If too few qualify, the
    section-type filter is dropped.
    """
    if count <= 0:
        return []
    ensure_embeddings()
    idx = PID_TO_IDX[target_pid]
    sims = EMBEDDINGS_NORM @ EMBEDDINGS_NORM[idx]
    sims[idx] = 2  # Above any cosine, so the target sorts last.
    n_bottom = max(count * 10, len(sims) // 5)
    bottom_indices = np.argsort(sims)[:n_bottom]
    candidates = [EMB_PAPER_IDS[i] for i in bottom_indices]

    # Keep candidates with no section of the target's answer-section type.
    filtered = []
    for pid in candidates:
        if pid == target_pid:
            continue
        paper_sec_types = get_paper_section_types(ALL_PAPERS[pid])
        # Accept only papers without the target's answer section type.
        if target_sec_type not in paper_sec_types:
            filtered.append(pid)

    # If too few remain after filtering, use every candidate except the target.
    if len(filtered) < count:
        filtered = [p for p in candidates if p != target_pid]

    rng.shuffle(filtered)
    return filtered[:count]


def get_similar_distractors(target_pid: str, target_sec_type: str, count: int) -> List[str]:
    """High semantic similarity without structural overlap.

    The nearest embedding neighbors that have no section of the target's answer-section
    type, so a similar distractor is topically close but unlikely to hold the answer.
    """
    # Walk the nearest papers in similarity order, skipping any that have a section
    # of the target's answer-section type.
    # Fetch more neighbors than needed, since the section-type filter skips some.
    candidates = get_most_similar_papers(target_pid, count * 5 + 20)
    out = []
    for pid in candidates:
        if pid == target_pid:
            continue
        ptypes = get_paper_section_types(ALL_PAPERS[pid])
        if target_sec_type not in ptypes:
            out.append(pid)
        if len(out) >= count:
            break
    if len(out) < count:
        # Fall back to distractors with the same section type.
        for pid in candidates:
            if pid in out or pid == target_pid:
                continue
            out.append(pid)
            if len(out) >= count:
                break
    return out[:count]


# --- Master distractor cache (load part) ---
# For each target and distractor kind, one master list of distractor papers is drawn
# once, sized for the largest K in --k_values; a context with K papers uses its first
# K-1 entries, so smaller contexts are subsets of larger ones. The target paper sits at
# the same relative position (a fraction u of the context) at every K. The cache key is
# (paper, kind, answer section type, variant seed); it omits K because every load
# prompt sizes the list for max(--k_values).
MASTER_DISTRACTORS: Dict[Tuple[str, str, str, int], List[str]] = {}
TARGET_POSITION_U: Dict[Tuple[str, int], float] = {}
# Offset added to the variant seed for the random distractor draw, giving it its own
# random stream; similar distractors involve no random draw. A constant offset gives
# the same draw on every Python version.
DISTRACTOR_SEED_OFFSETS = {"random": 4972}


def _get_target_u(target_pid: str, variant_seed: int) -> float:
    """Cached uniform(0,1) per (paper, variant) that sets the target's relative position."""
    # The +777 offset gives target placement its own random stream.
    key = (target_pid, variant_seed)
    if key not in TARGET_POSITION_U:
        rng = random.Random(variant_seed + 777)
        TARGET_POSITION_U[key] = rng.random()
    return TARGET_POSITION_U[key]


def _get_master_distractors(target_pid: str, sim: str, target_sec_type: str,
                             variant_seed: int, k_max: int) -> List[str]:
    """Build or retrieve the master distractor list for one distractor kind."""
    key = (target_pid, sim, target_sec_type, variant_seed)
    if key in MASTER_DISTRACTORS:
        return MASTER_DISTRACTORS[key]

    need = k_max - 1
    if sim == "random":
        rng = random.Random(variant_seed + DISTRACTOR_SEED_OFFSETS["random"])
        candidates = get_random_distractors(target_pid, target_sec_type, need, rng)
    else:
        candidates = get_similar_distractors(target_pid, target_sec_type, need)
    MASTER_DISTRACTORS[key] = candidates
    return candidates


# --- Prompt rendering ---
# A prompt is: PROMPT_HEADER, a blank line, the papers, the question(s), then the JSON
# answer instructions (PROMPT_FOOTER or an interference variant of it).
PROMPT_HEADER = (
    'You are given scientific article sections and question(s). Answer each requested question as concisely as you can, using a single phrase or sentence if possible. If a question cannot be answered based on the information in the article sections, write "unanswerable". If a question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n'
)

PROMPT_FOOTER = (
    'Respond in JSON only: {"answer": "<your concise answer>"}\nBe as concise as possible.'
)

def render_paper(paper: dict, paper_idx: int) -> str:
    """Render one paper as "=== Paper i ===", its title, then each "## section" with its paragraphs."""
    parts = [f"=== Paper {paper_idx} ==="]
    parts.append(f"Title: {paper['title']}")
    for sec in paper["full_text"]:
        parts.append(f"\n## {sec['section_name']}")
        parts.extend(sec["paragraphs"])
    return "\n\n".join(parts)


def build_papers_text(pids: List[str]) -> str:
    """Render the papers in order, numbered from 1."""
    return "\n\n".join(render_paper(ALL_PAPERS[pid], i + 1) for i, pid in enumerate(pids))


# --- Prompt builders: each returns (prompt, metadata saved with the prediction) ---
def build_load_prompt(target_pid: str, qa_info: dict, settings: dict, variant_seed: int):
    """Load part: distractor quantity x kind (random/similar).

    Distractor lists are nested (K=3 is a prefix of K=5, which is a prefix of K=7),
    and the cached U value keeps the target position the same across K.
    """
    K = int(settings["num_papers"])
    sim = settings["distractors"]

    paper = ALL_PAPERS[target_pid]
    target_sec_type = find_answer_section_type(paper, qa_info)

    if K == 1:
        papers_text = render_paper(paper, 1)
    else:
        # Cache enough distractors for the largest requested K.
        master = _get_master_distractors(target_pid, sim, target_sec_type, variant_seed,
                                         max(args.k_values))
        distractor_pids = master[:K - 1]

        # Place the target at the same relative position for every K.
        u = _get_target_u(target_pid, variant_seed)
        target_pos = min(int(u * K), K - 1)

        # K-1 distractors plus the target give K papers.
        all_pids = list(distractor_pids)
        all_pids.insert(target_pos, target_pid)
        all_pids = all_pids[:K]
        papers_text = build_papers_text(all_pids)

    prompt = (
        PROMPT_HEADER + "\n" +
        papers_text + "\n\n" +
        f"Question: {qa_info['question']}\n\n" +
        PROMPT_FOOTER
    )

    meta = {
        "part": "load",
        "setting": load_setting(K, sim),
        "num_papers": K,
        "distractors": sim,
        "target_section_type": target_sec_type,
        "target_paper_id": target_pid,
        "question_id": qa_info["question_id"],
        "question": qa_info["question"],
        "answer_type": qa_info["answer_type"],
        "gold": qa_info["gold"],
    }
    return prompt, meta


# --- Interference part ---
def get_questions_for_distractor_papers(distractor_pids: List[str], count: int,
                                        rng: random.Random) -> List[Tuple[str, dict]]:
    """Find questions for distractor papers in the context.

    Returns list of (paper_pid, qa_info); the caller assigns context positions.
    Takes one random question from each paper in context order that has one,
    until `count` are found; fewer are returned if the papers have too few.
    """
    results = []
    for pid in distractor_pids:
        qs = PAPER_QAS.get(pid, [])
        if qs:
            results.append((pid, rng.choice(qs)))
            if len(results) >= count:
                break
    return results


def build_interference_prompt(target_pid: str, qa_info: dict, settings: dict, variant_seed: int):
    """Interference part: the target question plus a secondary task.

    - none: just the primary question
    - extra_questions: primary + up to two questions about other papers in context, in shuffled order
    - summary: primary + one-sentence summary of all passages

    All three share one context: K-1 random dissimilar papers and a random target
    position, both drawn from a stream seeded by the variant seed alone.
    """
    K = int(settings["num_papers"])
    i_type = settings["interference_type"]
    rng = random.Random(variant_seed)

    distractor_pids = get_dissimilar_papers(target_pid, K - 1, rng)

    # randint is inclusive, so the target can take any of the K slots.
    all_pids = list(distractor_pids)
    target_pos = rng.randint(0, max(0, K - 1))
    all_pids.insert(target_pos, target_pid)
    all_pids = all_pids[:K]
    papers_text = build_papers_text(all_pids)

    if i_type == "none":
        prompt = (
            PROMPT_HEADER + "\n" +
            papers_text + "\n\n" +
            f"Question: {qa_info['question']}\n\n" +
            PROMPT_FOOTER
        )

    elif i_type == "extra_questions":
        # Find 2 questions about distractor papers. Separate seeded streams (+888 to
        # pick them, +999 to order all three) keep these draws independent.
        rng_q = random.Random(variant_seed + 888)
        extra_qs = get_questions_for_distractor_papers(
            [p for p in all_pids if p != target_pid], count=2, rng=rng_q)

        extra_questions = [{
            "question": qinfo["question"],
            "paper_pid": qpid,
            "paper_index": all_pids.index(qpid) + 1,
            "gold": qinfo["gold"],
        } for qpid, qinfo in extra_qs]
        question_entries = [{"question": qa_info["question"], "is_target": True}]
        question_entries += [{"question": qinfo["question"], "is_target": False} for _, qinfo in extra_qs]

        random.Random(variant_seed + 999).shuffle(question_entries)
        # Questions are listed as Q1..Qn; the target's answer is read from answer_<its position>.
        q_lines = [f"Q{i+1}: {entry['question']}" for i, entry in enumerate(question_entries)]
        target_question_position = next(i + 1 for i, entry in enumerate(question_entries) if entry["is_target"])

        # The schema always lists three answer keys, even when fewer extra questions were found.
        schema = {"answer_1": "answer to Q1", "answer_2": "answer to Q2", "answer_3": "answer to Q3"}

        prompt = (
            PROMPT_HEADER + "\n" +
            papers_text + "\n\n" +
            "\n".join(q_lines) + "\n\n" +
            "Respond in JSON only: " + json.dumps(schema) + "\n"
            "Be as concise as possible."
        )

    elif i_type == "summary":
        schema = {"answer": "your answer here", "summary": "one sentence"}

        prompt = (
            PROMPT_HEADER + "\n" +
            papers_text + "\n\n" +
            f"Question: {qa_info['question']}\n\n" +
            "Additionally, in one sentence (max 30 words), describe the common theme or topic "
            "across all the passages above. Output this under \"summary\" in your JSON response.\n"
            "\n"
            "Respond in JSON only: " + json.dumps(schema) + "\n"
            "Be as concise as possible."
        )

    else:
        raise ValueError(f"Unknown interference type: {i_type}")

    meta = {
        "part": "interference",
        "setting": i_type,
        "interference_type": i_type,
        "num_papers": K,
        "target_paper_id": target_pid,
        "question_id": qa_info["question_id"],
        "question": qa_info["question"],
        "answer_type": qa_info["answer_type"],
        "gold": qa_info["gold"],
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


def build_conditions():
    """Return (setting, settings) pairs for the parts selected by --part."""
    # Settings follow a fixed kind and type order, whatever order the options list them in.
    conds = []
    if args.part in ("load", "all"):
        for K in args.k_values:
            # K=1 shows no distractors, so it is one setting whatever --distractors lists.
            for sim in (["random"] if K == 1 else [d for d in DISTRACTORS if d in args.distractors]):
                conds.append((load_setting(K, sim), {"part": "load", "num_papers": K, "distractors": sim}))
    if args.part in ("interference", "all"):
        for i_type in INTERFERENCE_TYPES:
            if i_type in args.interference_types:
                conds.append((i_type, {"part": "interference", "num_papers": args.interference_k,
                                       "interference_type": i_type}))
    return conds


# --- Answer scoring ---
def evaluate_answer(pred: str, qa_info: dict) -> float:
    """Token F1 against the target's combined gold extractive spans.

    The pooled spans are joined into one reference string and scored with the
    official evaluator's token F1.
    """
    spans = qa_info.get("gold", {}).get("extractive_spans", [])
    joined = ", ".join(spans) if spans else ""
    return _qasper_token_f1_score(pred or "", joined)


# --- Target questions and resume keys ---
def select_targets(num_variants: int, seed: int) -> List[Tuple[str, dict]]:
    """Shuffle all extractive questions with `seed` and keep the first `num_variants` (paper id, info) pairs.

    A target's position in this list is its variant index; a larger `num_variants`
    extends the same list, so indices stay fixed across runs with the same seed.
    """
    rng = random.Random(seed)
    candidates = list(ALL_QAS)
    rng.shuffle(candidates)
    return candidates[:num_variants]


def task_hash(setting: str, variant_idx: int, question_id: str) -> str:
    """Resume key for one task: a short MD5 of setting, variant index, and question id."""
    key = f"{setting}|{variant_idx}|{question_id}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


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
    """Sleep while --max_rpm requests were sent in the last 60 seconds, then record this one."""
    if args.max_rpm == 0:
        return
    # The lock is held while sleeping, so waiting requests proceed one at a time.
    async with rpm_lock:
        now = time.time()
        while rpm_timestamps and (now - rpm_timestamps[0]) > 60:
            rpm_timestamps.pop(0)
        if len(rpm_timestamps) >= args.max_rpm:
            # Sleep until the oldest request leaves the window, plus a 50 ms margin.
            sleep_for = 60 - (now - rpm_timestamps[0]) + 0.05
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        rpm_timestamps.append(time.time())


async def call_api_async(prompt: str) -> str:
    """Send one prompt and return the stripped reply text.

    Empty replies are retried at once and errors after an exponential backoff, up to
    API_ATTEMPTS attempts in all; then RuntimeError is raised.
    """
    for attempt in range(API_ATTEMPTS):
        async with sem:
            await rate_limit_wait()
            try:
                # The completion budget includes reasoning tokens where applicable.
                if USE_ANTHROPIC:
                    resp = await async_client.messages.create(
                        model=args.model,
                        max_tokens=API_MAX_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    content = resp.content[0].text
                else:
                    resp = await async_client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=API_MAX_TOKENS,
                    )
                    content = resp.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                print(f"  Empty response (attempt {attempt+1}/{API_ATTEMPTS}), retrying...")
            except Exception as e:
                print(f"  API error (attempt {attempt+1}/{API_ATTEMPTS}): {e}")
                if attempt < API_ATTEMPTS - 1:
                    await asyncio.sleep(2 ** attempt)
    raise RuntimeError("QASPER inference failed: API retries exhausted")


# --- Worker: generate, score, and build one output record ---
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
    score = evaluate_answer(pred, meta)

    # The record is the builder's metadata plus the output, its score, and the resume key.
    # The raw output and parsed answer are kept to their first 1000 and 500 characters.
    record = {
        **meta,
        "variant_index": variant_idx,
        "raw_output": raw[:1000],
        "pred_answer": pred[:500],
        "score": score,
        "task_hash": task_hash(setting, variant_idx, meta["question_id"]),
    }
    if args.directions:
        record["steer_alpha"] = float(args.steer_alpha)
        record["steer_axis"] = args.steer_axis_filter
        record["steer_directions"] = os.path.basename(args.directions)
    if features is not None:
        # --els_mode: activation features go into the record; final-token layer vectors go to a .pt file.
        record["neural_dynamics"] = features["neural_dynamics"]
        record["activation_metrics"] = features["activation_metrics"]
        save_vectors(vectors_dir(args), setting, variant_idx, vectors, score=score)
    return setting, variant_idx, score, record


# --- Main ---
async def main():
    """Build pending prompts, run them, append records to the output JSONL, and print per-condition F1."""
    global local_model, local_tokenizer
    # Resume: tasks whose task_hash is already in the output file are skipped.
    out_path = output_path(args, "qasper")
    done = load_completed(out_path, args.directions)
    if done:
        print(f"Resume: {len(done)} prior tasks already complete (skipping)")

    # Conditions, optionally narrowed to one steering axis's setting and a list of question ids.
    conds = build_conditions()
    print(f"\nConditions to run ({len(conds)}):")
    for setting, _ in conds:
        print(f"  {setting}")
    axis_setting = steering_setting("qasper", args.steer_axis_filter, [setting for setting, _ in conds])
    if axis_setting:
        print(f"[STEERING] Restricting to setting={axis_setting} (axis={args.steer_axis_filter})")

    keys_filter = None
    if args.steer_test_pair_keys_json:
        with open(args.steer_test_pair_keys_json) as f:
            keys_filter = set(json.load(f))
        print(f"[STEERING] Restricting to {len(keys_filter)} listed question_ids")

    # Build every pending prompt before any inference, so a builder error stops the run early.
    targets = select_targets(args.num_variants, args.seed)
    if keys_filter is not None:
        # Listed IDs outside the first --num_variants targets cannot be run, so warn about them.
        missing = keys_filter - {qa_info["question_id"] for _, qa_info in targets}
        if missing:
            print(f"[STEERING] Warning: {len(missing)} listed question_ids are not among the "
                  f"selected targets; increase --num_variants to include them")
    tasks: List[Tuple[str, int, str, dict]] = []
    for setting, settings in conds:
        if axis_setting and setting != axis_setting:
            continue
        builder = PROMPT_BUILDERS[settings["part"]]
        for vi, (pid, qa_info) in enumerate(targets):
            if keys_filter is not None and qa_info["question_id"] not in keys_filter:
                continue
            # Completed and excluded tasks need no prompt reconstruction.
            if task_hash(setting, vi, qa_info["question_id"]) in done:
                continue
            # The variant seed depends only on --seed and the target's index, so all
            # conditions of a target share its random draws and reruns repeat them. The
            # factor keeps seeds from different --seed values apart for up to 100000 targets.
            variant_seed = args.seed * 100000 + vi
            try:
                prompt, meta = builder(pid, qa_info, settings, variant_seed)
                if meta["setting"] != setting:
                    raise ValueError(f"Builder returned setting {meta['setting']!r}, expected {setting!r}")
            except Exception as error:
                raise RuntimeError(
                    f"Could not build QASPER prompt {setting} v{vi} ({qa_info['question_id']}): {error}"
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
            print(f"  setting={setting}  question_id={meta['question_id']}")
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
        raise RuntimeError(f"{len(tasks) - completed} QASPER tasks failed; rerun to resume incomplete tasks.")


if __name__ == "__main__":
    asyncio.run(main())
