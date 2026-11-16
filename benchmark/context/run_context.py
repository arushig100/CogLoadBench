#!/usr/bin/env python3
"""Run the CogLoadBench Context task (multi-step reasoning chains).

Each prompt shows several chains of transformations between invented entities and
repeats one target question at the end; the answer is an entity name. Prompt
construction is shared with the Math task (shared/problems.py); this file supplies
the Context header, line rendering, answer format, and scoring.

    python run.py benchmark context --model Qwen/Qwen3-4B --axis all
    python run.py benchmark context --dry_run --num_variants 200
"""
from __future__ import annotations

import os
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
    "Begin your response on the next line with '{' and output a SINGLE JSON object:\n"
    '{"reasoning": "<brief explanation of reasoning behind answer>", "answer": "<final answer to the repeated question>"}'
)


def header(steps: int) -> str:
    """Return the prompt header for chains of the given number of steps."""
    # Example labels for chains 1 and 2, e.g. "[1.1], [1.2], [1.3]".
    labels = ", ".join(f"[1.{i}]" for i in range(1, steps + 1))
    labels2 = ", ".join(f"[2.{i}]" for i in range(1, steps + 1))
    return (
        f"You will read a series of short {steps}-step reasoning chains. "
        f"Each chain has {steps} lines, labeled like {labels} for chain 1; {labels2} for chain 2; and so on.\n\n"
        "IMPORTANT:\n"
        "• Read all chains, but answer ONLY the question that we repeat at the very end.\n"
        "• Ignore any other questions that appear earlier in the context.\n\n"
        "Return your answer in JSON ONLY (no extra text)."
    )


def render(row: Dict[str, Any], pid: int, target: Optional[Dict[str, Any]]) -> List[str]:
    """Render a chain as labeled lines '[pid.step] text'; distractors are rendered as stored."""
    lines = []
    i = 1
    while f"s{i}" in row:
        lines.append(f"[{pid}.{i}] {row[f's{i}']}")
        i += 1
    return lines


def answers_match(pred: str, gold: str) -> bool:
    """Compare answers ignoring case; runs of commas and whitespace count as one space."""
    return normalize_answer(pred) == normalize_answer(gold)


# Prompt pieces for shared/problems.py; the last argument names the text of the counting task.
TASK_FORMAT = TaskFormat(header, render, FORMAT_BLOCK, "the reasoning chains and the final question")


def add_arguments(parser) -> None:
    """Add the Context input files."""
    parser.add_argument("--baseline_file", default=os.path.join(HERE, "baselines", "baseline.jsonl"),
                        help="Base chains for every axis except underspecification")
    parser.add_argument("--underspec_baseline_file",
                        default=os.path.join(HERE, "variants", "underspecification_baseline.jsonl"),
                        help="Base chains for both underspecification conditions")
    parser.add_argument("--underspec_variant_file", default=os.path.join(HERE, "variants", "underspecification.jsonl"),
                        help="Underspecified targets, matched to base chains by base_id")


def main(argv=None) -> None:
    """Run the Context task from the command line."""
    runner.main("context", lambda args: ProblemPrompts(TASK_FORMAT, args).build, answers_match, argv, add_arguments)


if __name__ == "__main__":
    main()
