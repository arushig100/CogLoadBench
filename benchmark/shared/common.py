"""Input reading, variant seeds, line shuffling, and the secondary tasks shared by all task runners.

The Math, Context, and Analogy runners use these helpers for two load axes:
discontinuity (``discontinuity_lines`` reorders the rendered problem lines) and
interference (``add_interference`` adds a secondary task and its answer fields).
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List

# Opening paragraph of every interference prompt, placed before the secondary-task instruction.
INTERFERENCE_HEADER = (
    "You will now perform a short secondary mental task while also solving the MAIN REASONING task.\n"
    "Always focus primarily on answering the final question correctly. The interference task should not be\n"
    "explicitly discussed in your reasoning text, but you must still carry it out and report its result in the JSON.\n\n"
)
# Placeholder of the "answer" field in the interference JSON template; Analogy passes its own.
DEFAULT_ANSWER_HINT = "<final answer to the repeated question>"
# Math and Context items and run records also store the target problem's ID and the
# ID of its base problem, which groups prompts for the ELS fits.
ITEM_FIELDS = ("target_id", "target_base_id")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSON objects, one per line, and reject malformed lines."""
    rows = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def variant_seed(seed: int, variant: int) -> int:
    """Return the seed of one variant index."""
    # Seeds stay distinct across base seeds as long as the variant index is below 100000.
    return int(seed) * 100000 + int(variant)


def discontinuity_lines(blocks: List[List[str]], condition: str, rng: random.Random) -> List[str]:
    """Shuffle the block order, then shuffle lines within blocks (local) or across blocks (global)."""
    # Every condition, including "ordered", first shuffles which problem comes first;
    # the conditions differ only in how lines move afterwards.
    blocks = blocks[:]
    rng.shuffle(blocks)
    if condition == "local_shuffle":
        out: List[str] = []
        for block in blocks:
            lines = block[:]
            rng.shuffle(lines)
            out.extend(lines)
        return out
    # "ordered" keeps each problem's lines together and in order; "global_shuffle"
    # interleaves lines from all problems.
    lines = [line for block in blocks for line in block]
    if condition == "global_shuffle":
        rng.shuffle(lines)
    return lines


def secondary_task(condition: str, material: str) -> tuple[str, Dict[str, Any]]:
    """Return the instruction and extra JSON answer fields of a secondary task.

    ``material`` names the text the counting task reads (for example "the math problems and the
    final question"). The field values are the placeholders shown in the JSON template.
    """
    if condition == "memory":
        return ("Before you begin, silently remember the sequence: TIGER, RED, JULY. "
                "At the end, output them under \"memory_sequence\" in the JSON response.",
                {"memory_sequence": ["TIGER", "RED", "JULY"]})
    if condition == "counting":
        return (f"While reading {material}, count how many times the word 'the' appears in that text. "
                "At the end, output only the total under \"word_count\" in the JSON.",
                {"word_count": 0})
    if condition == "arithmetic":
        return ("In your head, compute 34 × 26. Do not show or explain this intermediate computation. "
                "At the end, output only the result under \"aux_product\" in the JSON.",
                {"aux_product": 0})
    raise ValueError(f"Unknown interference condition: {condition}")


def json_template(extra_fields: Dict[str, Any], answer_hint: str) -> str:
    """Return the one-line JSON answer template with the secondary-task fields."""
    fields = {
        "reasoning": "<brief explanation of reasoning behind answer (do not discuss interference task)>",
        "answer": answer_hint,
        **extra_fields,
    }

    def render(value):
        """Render a placeholder value as JSON-like text: quoted strings, lists of quoted strings, bare numbers."""
        if isinstance(value, list):
            return "[" + ", ".join(f"\"{x}\"" for x in value) + "]"
        if isinstance(value, str):
            return f"\"{value}\""
        return str(value)

    return "{ " + ", ".join(f"\"{k}\": {render(v)}" for k, v in fields.items()) + " }"


def add_interference(prompt: str, condition: str, format_block: str, material: str,
                     answer_hint: str = DEFAULT_ANSWER_HINT) -> str:
    """Prepend a secondary task to a prompt and replace its answer format; 'none' keeps the prompt."""
    if condition == "none":
        return prompt
    instruction, extra_fields = secondary_task(condition, material)
    # Drop the task's own answer format from the end of the prompt; the replacement
    # format below adds the secondary-task fields to the JSON template.
    end = prompt.rfind(format_block)
    if end == -1:
        raise ValueError("The prompt does not contain the given answer format")
    body = prompt[:end].rstrip()
    answer_format = (
        "\nAnswer format (return JSON only). Do not write anything before or after the JSON.\n"
        "Begin your response on the next line with '{' and output a SINGLE JSON object:\n"
        f"{json_template(extra_fields, answer_hint)}"
    )
    # Layout: interference header, secondary-task instruction, task body, new answer format.
    return f"{INTERFERENCE_HEADER}{instruction}\n\n{body}\n{answer_format}"
