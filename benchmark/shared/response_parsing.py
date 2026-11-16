"""Recover JSON answer objects from model responses.

The runner stores each response and scores the "answer" field returned by
``parse_json_output``; the other fields (reasoning, secondary-task results) are
kept in the saved record.
"""
from __future__ import annotations

import json
import re


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers that models commonly emit."""
    text = text.strip()
    # Drop the opening fence line, including any language tag, and a closing fence.
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text.strip()


def extract_json_object(text: str) -> str | None:
    """Extract the first balanced {...} from text, handling nested braces."""
    start = text.find("{")
    if start == -1:
        return None
    # Track string literals and escapes so braces inside quoted text do not change the depth.
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def answer_object(text: str) -> str | None:
    """Return the first balanced {...} that has an answer field, else the first balanced {...}."""
    # Try every "{" in order of position as the start of an object.
    first = None
    for start in [i for i, c in enumerate(text) if c == "{"]:
        obj = extract_json_object(text[start:])
        if obj is None:
            continue
        if re.search(r'["\']answer["\']\s*:', obj):
            return obj
        first = first or obj
    return first


def parse_json_output(raw):
    """Recover an answer object, falling back to plain text when needed.

    Try strict JSON, single-quote repair, a balanced object, then an answer-field
    regex. Preserve additional fields when a complete object can be decoded.
    """
    if raw is None:
        return {"reasoning": "", "answer": ""}
    text = _strip_markdown_fences(str(raw).strip())
    # Candidates in order: the whole response, then its first balanced {...} with an
    # answer field (or else its first balanced {...}); each as written and with single
    # quotes turned into double quotes.
    obj_str = answer_object(text)
    for candidate in [text] + ([obj_str] if obj_str else []):
        for attempt in (candidate, candidate.replace("'", '"')):
            try:
                parsed = json.loads(attempt)
            except (ValueError, RecursionError):
                continue
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
    # Nothing decodes to an object with an answer field: take the last quoted
    # "answer" value, or else the whole response as the answer.
    # The pattern matches "answer": "value" with single or double quotes; the value has no quotes.
    m = re.findall(r'["\']answer["\']\s*:\s*["\']([^"\']*)["\']', text)
    if m:
        return {"reasoning": "", "answer": m[-1].strip()}
    return {"reasoning": "", "answer": text.strip()}


def normalize_answer(answer: str) -> str:
    """Lowercase an answer and collapse commas and whitespace to single spaces."""
    return re.sub(r"[,\s]+", " ", (answer or "").strip().lower()).strip()
