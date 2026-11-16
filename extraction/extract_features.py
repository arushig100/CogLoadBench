"""Extract per-prompt activation features from benchmark run records or cogloadbench items for one task and axis."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from benchmark.shared.axes import AXES
from benchmark.shared.common import ITEM_FIELDS
from extraction.feature_extraction import (
    FORMAT_LABEL,
    QUERY_LABELS,
    forward_with_hidden,
    generation_trace_metrics,
    prompt_feature_metrics,
)


def parse_args(argv=None):
    """Parse the command-line options."""
    p = argparse.ArgumentParser(prog="run.py extract", description=__doc__)
    p.add_argument("--model", required=True, help="Hugging Face model ID (e.g. Qwen/Qwen3-4B)")
    p.add_argument("--task", required=True, choices=sorted(QUERY_LABELS), help="Benchmark task of the input")
    p.add_argument("--axis", required=True, choices=AXES, help="Load axis of the input")
    p.add_argument("--input", required=True,
                   help="JSONL of benchmark run records or cogloadbench items with condition, variant and prompt")
    p.add_argument("--output", required=True, help="Output JSONL of feature records")
    p.add_argument("--condition", default="", help="Only extract records with this condition")
    p.add_argument("--q-label", default=None,
                   help="Marker between the context and question regions (default: the task's marker; "
                        "backslash escapes decoded)")
    p.add_argument("--fmt-label", default=None,
                   help="Marker that ends the question region (default: the answer-format line; "
                        "backslash escapes decoded)")
    p.add_argument("--max-len", type=int, default=8192, help="Maximum prompt length in tokens (default: 8192)")
    p.add_argument("--rank-token-cap", type=int, default=512,
                   help="Maximum tokens per region for the effective rank (default: 512)")
    p.add_argument("--gen-trace-tokens", type=int, default=50,
                   help="Greedy generation-trace length for entropy and margin; 0 disables (default: 50)")
    p.add_argument("--local-files-only", action="store_true", help="Load the model from the local cache only")
    p.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    p.add_argument("--resume", action="store_true", help="Continue an interrupted run")
    return p.parse_args(argv)


def load_model(model_id: str, local_files_only: bool):
    """Load a causal language model and its tokenizer for inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        local_files_only=local_files_only,
    ).eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def read_records(args) -> list[dict]:
    """Read and check the input records selected by --condition."""
    records, seen = [], set()
    with open(args.input, encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            location = f"{args.input}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON") from error
            if not isinstance(record, dict) or not {"condition", "variant", "prompt"} <= record.keys():
                raise ValueError(f"{location}: expected an object with condition, variant and prompt")
            if (record.get("task", args.task), record.get("axis", args.axis)) != (args.task, args.axis):
                raise ValueError(f"{location}: task or axis differs from --task/--axis")
            if type(record["variant"]) is not int or not isinstance(record["prompt"], str) or not record["prompt"].strip():
                raise ValueError(f"{location}: expected an integer variant and a nonempty prompt")
            if args.task != "analogy" and not all(isinstance(record.get(field), str) and record[field]
                                                  for field in ITEM_FIELDS):
                raise ValueError(f"{location}: Math and Context records need {' and '.join(ITEM_FIELDS)}, "
                                 "as in cogloadbench items and benchmark run records")
            if record.get("correct") not in (None, 0, 1):
                raise ValueError(f"{location}: correct must be 0 or 1")
            if args.condition and record["condition"] != args.condition:
                continue
            key = (record["condition"], record["variant"])
            if key in seen:
                raise ValueError(f"{location}: duplicate condition and variant {key}")
            seen.add(key)
            records.append(record)
    if not records:
        raise ValueError(f"No records in {args.input}" + (f" with condition {args.condition}" if args.condition else ""))
    return records


def read_resume_records(path: Path) -> tuple[list[str], set]:
    """Return the complete saved lines and their (condition, variant) keys, dropping a truncated final line."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    good, done = [], set()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            # Only the last nonblank line may be unreadable (a write cut off by an interruption).
            if any(rest.strip() for rest in lines[index + 1:]):
                raise ValueError(f"{path}:{index + 1}: invalid JSON before the final line") from error
            break
        good.append(line if line.endswith("\n") else line + "\n")
        done.add((record["condition"], record["variant"]))
    return good, done


def check_paths(args) -> None:
    """Check the input and output paths before loading the model."""
    source, output = Path(args.input), Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(f"Input is not a file: {source}")
    if output.resolve() == source.resolve():
        raise ValueError("--output must differ from --input")
    if args.force and args.resume:
        raise ValueError("Choose either --force or --resume")
    if output.exists() and not (args.force or args.resume):
        raise ValueError(f"{output} exists; choose a new output, --force, or --resume")


def decode(label: str | None, default: str) -> str:
    """Decode backslash escapes in a marker option, or return the default marker."""
    # Turns a typed '\n' into a newline; characters outside Latin-1 are kept as escapes
    # first, so unicode_escape restores them unchanged.
    return label.encode("latin-1", "backslashreplace").decode("unicode_escape") if label else default


def main(argv=None):
    """Extract one feature record per selected prompt and append it to --output."""
    # Check inputs before loading the model.
    args = parse_args(argv)
    check_paths(args)
    records = read_records(args)
    model, tokenizer = load_model(args.model, args.local_files_only)
    # Qwen3 chat templates are applied with thinking disabled.
    is_qwen3 = "qwen3" in args.model.lower()
    # Markers that split each prompt into context and question regions.
    q_label = decode(args.q_label, QUERY_LABELS[args.task])
    fmt_label = decode(args.fmt_label, FORMAT_LABEL)

    # Resume: keep the complete records already written and skip their prompts.
    out_path = Path(args.output)
    done = set()
    if args.resume and out_path.exists():
        good, done = read_resume_records(out_path)
        out_path.write_text("".join(good), encoding="utf-8")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Append after resumed records, otherwise start a new file; each record is flushed
    # as soon as it is written, so an interrupted run can resume.
    n_done, t_start = 0, time.time()
    with open(out_path, "a" if done else "w", encoding="utf-8") as f_out:
        for rec in records:
            condition, variant = rec["condition"], rec["variant"]
            if (condition, variant) in done:
                continue
            # One forward pass gives every layer's hidden states and the region spans.
            try:
                hidden_states, spans = forward_with_hidden(
                    model, tokenizer, rec["prompt"], q_label=q_label, fmt_label=fmt_label,
                    is_qwen3=is_qwen3, max_length=args.max_len)
            except ValueError as error:
                raise ValueError(f"{args.input}: {condition} variant {variant}: {error}") from error
            if spans["question"][1] <= spans["question"][0]:
                raise ValueError(f"{args.input}: {condition} variant {variant} has an empty question span; "
                                 "check --q-label and --fmt-label")
            # A short greedy continuation gives output entropy and logit margins.
            gen_trace = {}
            if args.gen_trace_tokens > 0:
                gen_trace = generation_trace_metrics(
                    model, tokenizer, rec["prompt"], is_qwen3=is_qwen3,
                    n_steps=args.gen_trace_tokens, max_len=args.max_len)
            # Per-layer region statistics and region dynamics across every layer.
            features = prompt_feature_metrics(hidden_states, spans, rank_token_cap=args.rank_token_cap,
                                              gen_trace=gen_trace)
            record = {
                "task": args.task,
                "axis": args.axis,
                "condition": condition,
                "variant": variant,
                # Math and Context inputs carry the target problem's IDs.
                **{field: rec[field] for field in ITEM_FIELDS if field in rec},
                "correct": rec.get("correct"),
                **features,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            # Free cached GPU memory and unreferenced objects before the next prompt.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            n_done += 1
            if n_done % 25 == 0:
                print(f"  [{n_done}] {(time.time() - t_start) / n_done:.1f} s/prompt", flush=True)

    print(f"Wrote {n_done} feature records to {out_path}")


if __name__ == "__main__":
    main()
