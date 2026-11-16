"""Generate the Context underspecification pairs: one middle-step subject becomes a referring expression.

Each pair keeps a base chain and asks for the entity transformed in one middle step.
In the baseline record that step names its subject; in the underspecified record the
subject is a referring expression such as "It" or "The result", which points back to
the previous step's product.
The runners use the baseline file as the problem pool for both conditions and swap in
the underspecified record as the target.

Writes underspecification.jsonl and its explicit-subject baseline, underspecification_baseline.jsonl:
    python -m benchmark.context.generate_underspecification --output-dir outputs/generated/context/variants

The defaults reproduce benchmark/context/variants/.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Directory of this file; the default input file lives under it.
HERE = Path(__file__).resolve().parent

# Make benchmark/shared importable as "shared" when this file is run as a script.
sys.path.insert(0, str(HERE.parent))
from shared.common import read_jsonl  # noqa: E402

# Referring expressions that replace the subject, in sentence-initial form.
PRONOUNS_INITIAL = ["It", "The latter", "That substance", "This material", "The result"]


def meta(prob: Dict[str, Any]) -> Dict[str, Any]:
    """Return the decoded meta field of a base problem."""
    m = prob.get("meta") or {}
    return json.loads(m) if isinstance(m, str) else m


def _chain_names(prob: Dict[str, Any]) -> List[str]:
    """Return the N+1 entity names for an N-step chain."""
    return meta(prob).get("names", [])


def build_underspec(prob: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Choose a middle step whose subject becomes a referring expression, and the question about that subject."""
    names = _chain_names(prob)
    n_steps = int(prob["chain_length"])
    if n_steps < 5 or len(names) < 6:
        raise ValueError("underspecification needs a chain with at least 5 steps")

    # Step 1 has no antecedent, and the question names the chain by its final entity.
    coref_step = rng.choice(list(range(2, n_steps)))

    # Step i turns names[i - 1] into names[i]; the answer is the entity the chosen step transforms.
    subject = names[coref_step - 1]
    next_entity = names[coref_step]
    final_entity = names[-1]

    pronoun = rng.choice(PRONOUNS_INITIAL)

    return {
        "coref_step": coref_step,
        "pronoun": pronoun,
        "subject": subject,
        "next_entity": next_entity,
        "final_entity": final_entity,
        "n_steps": n_steps,
        "answer": subject,
        "question_text": (
            f"In the chain that ultimately produces {final_entity}, "
            f"what substance becomes {next_entity}?"
        ),
    }


def _replace_subject_with_pronoun(sentence: str, subject: str, pronoun: str) -> str:
    """Replace the first subject mention with the referring expression, lowercased unless it starts the sentence."""
    if sentence.startswith(subject + " ") or sentence.startswith(subject + ","):
        return pronoun + sentence[len(subject):]
    return sentence.replace(subject, pronoun.lower(), 1)


def render_underspec_chain(prob: Dict[str, Any], struct: Dict[str, Any], vague: bool) -> List[str]:
    """Render the chain sentences, replacing one subject with the referring expression when vague is true."""
    n = struct["n_steps"]
    lines: List[str] = []
    for i in range(1, n + 1):
        s = prob.get(f"s{i}", "").strip()
        if vague and i == struct["coref_step"]:
            s = _replace_subject_with_pronoun(s, struct["subject"], struct["pronoun"])
        lines.append(s)
    return lines


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    """Write records as JSON lines, creating the parent directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_record(
    base_prob: Dict[str, Any],
    variant_kind: str,
    answer: str,
    question_text: str,
    lines: List[str],
    extra_meta: Dict[str, Any],
    seq_idx: int,
) -> Dict[str, Any]:
    """Build a record in the format of the Context base problems."""
    base_id = base_prob.get("base_id") or base_prob.get("id")
    s_fields = {f"s{i+1}": ln for i, ln in enumerate(lines)}

    # meta is stored as a JSON string: the base meta, the variant kind, and the pair details.
    merged_meta = {
        **meta(base_prob),
        "variant_kind": variant_kind,
        **extra_meta,
        "chain_length_rendered": len(lines),
    }

    return {
        # The pair index and variant kind keep the IDs of the two records of a pair parallel.
        "id": f"{base_id}__{seq_idx:04d}__{variant_kind}",
        "base_id": base_id,
        "variant": variant_kind,
        "variant_kind": variant_kind,
        "domain": base_prob.get("domain", "unknown"),
        "structure": base_prob.get("structure", "unknown"),
        "chain_length": len(lines),
        **s_fields,
        "question_only": question_text,
        "answer": str(answer),
        "meta": json.dumps(merged_meta, ensure_ascii=False),
    }


def gen_underspec(
    baselines: List[Dict[str, Any]],
    rng: random.Random,
    n: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build up to n underspecified variants and their explicit-subject baselines, one per base problem."""
    # Shuffle so the n pairs are a random subset of the base problems.
    rng.shuffle(baselines)
    out_baseline: List[Dict[str, Any]] = []
    out_condition: List[Dict[str, Any]] = []
    seen = 0
    seen_base_ids: set = set()
    for prob in baselines:
        if seen >= n:
            break
        bid = prob.get("base_id") or prob.get("id")
        if bid in seen_base_ids:
            continue
        # Chains with fewer than five steps or six entity names are skipped.
        try:
            struct = build_underspec(prob, rng)
        except ValueError:
            continue

        baseline_lines = render_underspec_chain(prob, struct, vague=False)
        condition_lines = render_underspec_chain(prob, struct, vague=True)

        extra_meta = {
            "coref_step": struct["coref_step"],
            "pronoun": struct["pronoun"],
            "subject_resolved": struct["subject"],
            "next_entity": struct["next_entity"],
            "final_entity": struct["final_entity"],
        }

        out_baseline.append(_build_record(
            prob, "underspec_baseline",
            answer=struct["answer"],
            question_text=struct["question_text"],
            lines=baseline_lines,
            extra_meta=extra_meta,
            seq_idx=seen,
        ))
        out_condition.append(_build_record(
            prob, "underspec",
            answer=struct["answer"],
            question_text=struct["question_text"],
            lines=condition_lines,
            extra_meta=extra_meta,
            seq_idx=seen,
        ))
        seen_base_ids.add(bid)
        seen += 1
    return out_baseline, out_condition


def main() -> None:
    """Write the underspecification pair files from the Context base problems."""
    ap = argparse.ArgumentParser(description="Generate the Context underspecification pairs.")
    ap.add_argument("--baseline", default=str(HERE / "baselines/baseline.jsonl"), help="Context base problems")
    ap.add_argument("--output-dir", default=str(HERE.parents[1] / "outputs/generated/context/variants"),
                    help="Directory for the variant files")
    ap.add_argument("--num_pairs", type=int, default=300, help="Number of pairs (default: 300)")
    ap.add_argument("--seed", type=int, default=1893, help="Sampling seed (default: 1893)")
    args = ap.parse_args()

    if args.num_pairs < 1:
        ap.error('--num_pairs must be positive')
    # Refuse output paths that resolve to the input file.
    baseline = Path(args.baseline).resolve()
    for name in ['underspecification_baseline.jsonl', 'underspecification.jsonl']:
        output = Path(args.output_dir) / name
        if output.resolve() == baseline or (output.exists() and baseline.exists() and output.samefile(baseline)):
            ap.error('Generated outputs must not overwrite --baseline')

    baselines = read_jsonl(args.baseline)
    print(f"Loaded {len(baselines)} baseline problems from {args.baseline}", flush=True)

    print(f"\nGenerating underspecification pairs (target n={args.num_pairs})...", flush=True)
    out_bsl, out_var = gen_underspec(list(baselines), random.Random(args.seed), args.num_pairs)
    bsl_path = os.path.join(args.output_dir, "underspecification_baseline.jsonl")
    var_path = os.path.join(args.output_dir, "underspecification.jsonl")
    write_jsonl(out_bsl, bsl_path)
    write_jsonl(out_var, var_path)
    print(f"  → {bsl_path}: {len(out_bsl)} records")
    print(f"  → {var_path}: {len(out_var)} records")


if __name__ == "__main__":
    main()
