"""Generate the Math underspecification pairs.

Both conditions introduce two people; the underspecified condition refers to the
target person in one step as "the first person mentioned".

Each base problem yields one baseline record and one underspecified record that
differ only in that step. The runners use the baseline file as the problem pool
for both conditions and swap in the underspecified record as the target.

    python -m benchmark.math.generate_underspecification --output-dir outputs/generated/math/variants

The defaults reproduce benchmark/math/variants/.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# Code directory, the default input file, and the default output directory.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = PACKAGE_ROOT / "benchmark/math/baselines/baseline.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs/generated/math/variants"

# Label stored in the id and variant fields of the shipped records.
VARIANT_ID = "underspec"

# Domain -> roles of the target person and the second person.
DOMAIN_ROLES = {
    "kitchen": ("chef", "baker"),
    "garden": ("gardener", "farmer"),
    "classroom": ("teacher", "principal"),
    "zoo": ("keeper", "trainer"),
    "sports": ("coach", "athlete"),
}

# The second person's name is drawn from one of two fixed lists: SECOND_NAMES_B for
# targets named in USE_LIST_B, SECOND_NAMES_A otherwise. It only has to differ from
# the problem's own name, since the underspecified step says "the first person mentioned".
SECOND_NAMES_A = ["Maya", "Sara", "Nia", "Cleo", "Luna", "Aiko", "Riya", "Tess", "Indi", "Wren"]
SECOND_NAMES_B = ["Marco", "Liam", "Ethan", "Zane", "Aiden", "Noah", "Theo", "Owen", "Kai", "Jude"]
USE_LIST_B = {"Yara", "Luna", "Maya", "Cleo", "Sara", "Fiona", "Grace"}


def replace_actor(step_text: str, primary_name: str) -> str:
    """Replace the first actor mention, preserving sentence capitalization."""
    replacement = "the first person mentioned"
    if step_text.startswith(primary_name):
        return replacement.capitalize() + step_text[len(primary_name):]
    return step_text.replace(primary_name, replacement, 1)


def generate_variant(records: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Return the baseline and underspecified records."""
    rng = random.Random(seed)
    baseline_out, condition_out = [], []
    for rec in records:
        primary_name = rec["entity_name"]
        primary_role, distractor_role = DOMAIN_ROLES[rec["domain"]]
        pool = SECOND_NAMES_B if primary_name in USE_LIST_B else SECOND_NAMES_A
        distractor_name = rng.choice([name for name in pool if name != primary_name])

        # The new first line introduces both people: the target with the starting count
        # of s1, and the second person with a count 20 to 60 lower, or 20 to 60 higher
        # when the lower count would fall below 10.
        # The starting count is the first whole number in s1.
        primary_n = int(re.search(r"\b(\d+)\b", rec["s1"]).group(1))
        distractor_n = primary_n - rng.randint(20, 60)
        if distractor_n < 10:
            distractor_n = primary_n + rng.randint(20, 60)
        introduction = (
            f"{primary_name}, the {primary_role}, has {primary_n} {rec['item']}. "
            f"{distractor_name}, the {distractor_role}, has {distractor_n} {rec['item']}."
        )

        # One later step, drawn once per base problem, is the step the underspecified
        # record rewrites; the baseline record keeps it.
        # s1 is the introduction and s<num_steps + 1> is the question, so the step is one of
        # s2 ... s<num_steps>.
        step_keys = [f"s{i}" for i in range(2, rec["num_steps"] + 1)]
        modified_step = rng.choice(step_keys)
        for kind, output in [("baseline", baseline_out), ("condition", condition_out)]:
            row = dict(rec)
            row["s1"] = introduction
            if kind == "condition":
                row[modified_step] = replace_actor(rec[modified_step], primary_name)
            row["variant"] = f"{VARIANT_ID}_{kind}"
            row["variant_kind"] = f"{VARIANT_ID}_{kind}"
            row["distractor_name"] = distractor_name
            row["modified_step"] = modified_step
            row["id"] = f"{rec['base_id']}__{VARIANT_ID}_{kind}"
            output.append(row)
    return baseline_out, condition_out


def main():
    """Write underspecification_baseline.jsonl and underspecification.jsonl."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH, help="Math base problems")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for the variant files")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the second person's name and count")
    args = parser.parse_args()
    with args.baseline.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    baseline, condition = generate_variant(records, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("underspecification_baseline.jsonl", baseline), ("underspecification.jsonl", condition)]:
        path = args.output_dir / name
        with path.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row) + "\n")
        print(f"Wrote {len(rows)} records to {path}")


if __name__ == "__main__":
    main()
