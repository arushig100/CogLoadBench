#!/usr/bin/env python3
"""Run the CogLoadBench Math task.

Each prompt shows several short arithmetic word problems and repeats one target
question at the end; the answer is a number. Prompt construction is shared with
the Context task (shared/problems.py); this file supplies the Math header, line
rendering, answer format, and scoring.

    python run.py benchmark math --model Qwen/Qwen3-4B --axis all
    python run.py benchmark math --dry_run --num_variants 200
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional

# Make benchmark/shared importable as "shared" when this file is run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from shared import runner  # noqa: E402
from shared.problems import ProblemPrompts, TaskFormat  # noqa: E402
from shared.response_parsing import normalize_answer  # noqa: E402

# Directory of this file; the default input files live under it.
HERE = os.path.dirname(os.path.abspath(__file__))

# Answer format at the end of every prompt; interference replaces it with a template
# that also has the secondary-task fields.
FORMAT_BLOCK = (
    "\nAnswer format (return JSON only). Do not write anything before or after the JSON.\n"
    "Begin your response on the next line with '{' and output a SINGLE JSON object with two fields: "
    '"reasoning" (a brief explanation string) and "answer" (the final numerical answer as a string).\n'
    "For example, if the correct answer is 115, your output should look exactly like:\n"
    '{"reasoning": "Started with 100, added 20 to get 120, then subtracted 5 to get 115.", "answer": "115"}\n'
    "Replace the example reasoning and answer with your own — do not output the placeholder text."
)


def header(lines: int) -> str:
    """Return the prompt header for problems of the given number of lines."""
    # Example labels for problems 1 and 2, e.g. "[1.1], [1.2], [1.3]".
    labels = ", ".join(f"[1.{i}]" for i in range(1, lines + 1))
    labels2 = ", ".join(f"[2.{i}]" for i in range(1, lines + 1))
    return (
        f"You will read a series of short math problems. "
        f"Each problem has {lines} lines, labeled like {labels} for problem 1; "
        f"{labels2} for problem 2; and so on.\n\n"
        "IMPORTANT:\n"
        "• Read all problems, but answer ONLY the question that we repeat at the very end.\n"
        "• Ignore any other questions that appear earlier in the context.\n\n"
        "Return your answer in JSON ONLY (no extra text)."
    )


def render(row: Dict[str, Any], pid: int, target: Optional[Dict[str, Any]]) -> List[str]:
    """Render a problem as labeled lines; distractors take the target's name and drop their question."""
    lines = []
    i = 1
    while f"s{i}" in row:
        text = row[f"s{i}"]
        # Distractors use the target's person name, so problems differ in items and
        # numbers but not in who they are about.
        name = row.get("entity_name", "")
        # Whole words only, so a longer name that contains this one is left as it is.
        if target is not None and target.get("entity_name", "") and name and name in text:
            text = re.sub(rf"\b{re.escape(name)}\b", lambda _: target["entity_name"], text)
        lines.append(f"[{pid}.{i}] {text}")
        i += 1
    # Only the target keeps its question line.
    if target is not None and lines and lines[-1].split("] ", 1)[-1].strip().endswith("?"):
        lines = lines[:-1]
    return lines


def _number(answer: str) -> Optional[float]:
    """Return the first number in an answer (thousands separators removed), or None."""
    match = re.search(r"-?\d+(?:\.\d+)?", (answer or "").replace(",", ""))
    return float(match.group(0)) if match else None


def answers_match(pred: str, gold: str) -> bool:
    """Compare the first numbers in both answers, or the normalized text when either has none."""
    pred_number, gold_number = _number(pred), _number(gold)
    if pred_number is not None and gold_number is not None:
        return abs(pred_number - gold_number) < 1e-6
    return normalize_answer(pred) == normalize_answer(gold)


# Prompt pieces for shared/problems.py; the last argument names the text of the counting task.
TASK_FORMAT = TaskFormat(header, render, FORMAT_BLOCK, "the math problems and the final question")


def add_arguments(parser) -> None:
    """Add the Math input files."""
    parser.add_argument("--baseline_file", default=os.path.join(HERE, "baselines", "baseline.jsonl"),
                        help="Base problems for every axis except underspecification")
    parser.add_argument("--underspec_baseline_file",
                        default=os.path.join(HERE, "variants", "underspecification_baseline.jsonl"),
                        help="Base problems for both underspecification conditions")
    parser.add_argument("--underspec_variant_file", default=os.path.join(HERE, "variants", "underspecification.jsonl"),
                        help="Underspecified targets, matched to base problems by base_id")


def main(argv=None) -> None:
    """Run the Math task from the command line."""
    runner.main("math", lambda args: ProblemPrompts(TASK_FORMAT, args).build, answers_match, argv, add_arguments)


if __name__ == "__main__":
    main()
