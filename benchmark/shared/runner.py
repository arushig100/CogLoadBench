"""Command line, generation, and the run loop shared by the Math, Context, and Analogy runners.

Each task runner supplies a ``build(condition, variant)`` function that returns a
prompt, its gold answer, and any extra item fields, and a ``score(pred, gold)``
function. With --dry_run the prompts are written to <outdir>/<axis>.jsonl;
otherwise a model answers every prompt and one record per prompt is appended to
<outdir>/<axis>/<model>.jsonl.
With the paper's seed and the task's input files (the defaults), each prompt is
read from cogloadbench/, the set evaluated in the paper; other seeds or input
files, and variants beyond those files, are built by the task's ``build``. An
interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .axes import AXES, SIMILARITY_SWEEP_K, Condition, conditions
from .common import ITEM_FIELDS, read_jsonl
from .response_parsing import answer_object, parse_json_output

# The prompts and answers evaluated in the paper, as <task>/<axis>.jsonl, and the
# seed of the paper's prompts.
SHIPPED_PROMPTS = Path(__file__).resolve().parents[2] / "cogloadbench"
PAPER_SEED = 42

# build(condition, variant) -> (prompt, gold answer, extra item fields such as the
# Math and Context target IDs); score(pred, gold) -> correct.
Build = Callable[[Condition, int], Tuple[str, str, Dict[str, str]]]
Score = Callable[[str, str], bool]


# A saved record is identified by (condition name, variant index).
Key = Tuple[str, int]


def prompt_hash(prompt: str) -> str:
    """Return the MD5 hex digest of a prompt."""
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def load_saved(path: str | Path) -> Dict[Key, str]:
    """Return the prompt hash of every saved (condition, variant), rejecting malformed or repeated records."""
    hashes: Dict[Key, str] = {}
    with open(path, encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            location = f"{path}:{number}"
            # Records are written whole with a trailing newline, so a line without one
            # was cut off when a run stopped.
            if not line.endswith("\n"):
                raise ValueError(f"{location}: unterminated record; remove the incomplete last line to resume")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON") from error
            if not isinstance(record, dict) or not {"condition", "variant", "prompt", "correct"} <= record.keys():
                raise ValueError(f"{location}: expected a result record")
            if type(record["correct"]) is not int or record["correct"] not in (0, 1):
                raise ValueError(f"{location}: correct must be 0 or 1")
            key = (record["condition"], record["variant"])
            if key in hashes:
                raise ValueError(f"{location}: a second record for {key[0]} variant {key[1]}")
            hashes[key] = prompt_hash(record["prompt"])
    return hashes


def with_shipped_prompts(task: str, build: Build) -> Build:
    """Return a build that reads each item from cogloadbench/ when the file has it."""
    # Axis -> (condition, variant) -> (prompt, answer, item fields); each axis file is read on first use.
    shipped: Dict[str, Dict[Key, Tuple[str, str, Dict[str, str]]]] = {}

    def lookup(condition: Condition, variant: int) -> Tuple[str, str, Dict[str, str]]:
        """Return the shipped prompt, answer and item fields of a condition and variant, or build them."""
        if condition.axis not in shipped:
            path = SHIPPED_PROMPTS / task / f"{condition.axis}.jsonl"
            rows = read_jsonl(str(path)) if path.is_file() else []
            shipped[condition.axis] = {
                (row["condition"], row["variant"]): (row["prompt"], row["answer"],
                                                     {field: row[field] for field in ITEM_FIELDS if field in row})
                for row in rows}
        return shipped[condition.axis].get((condition.name, variant)) or build(condition, variant)
    return lookup


def parse_args(task: str, argv: Optional[List[str]] = None,
               add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None) -> argparse.Namespace:
    """Parse the command line of a task runner."""
    sweep = ",".join(map(str, SIMILARITY_SWEEP_K[task]))
    parser = argparse.ArgumentParser(
        prog=f"run.py benchmark {task}",
        description=f"Run the CogLoadBench {task} prompts through a model, or write them with --dry_run.")
    parser.add_argument("--model", help="Hugging Face model ID; not needed with --dry_run")
    parser.add_argument("--axis", default="all", choices=[*AXES, "all"], help="Load axis to run (default: all)")
    parser.add_argument("--num_variants", type=int, default=200, help="Prompts per condition (default: 200)")
    parser.add_argument("--similarity_k", default="",
                        help=f"Comma-separated K values for extra high-similarity conditions high_k<K> (choose from {sweep}; default: none)")
    parser.add_argument("--seed", type=int, default=PAPER_SEED,
                        help=f"Prompt seed (default: {PAPER_SEED}); with the default input files, "
                             "this seed uses the paper's prompts in cogloadbench/")
    parser.add_argument("--outdir", default=f"outputs/benchmark/{task}",
                        help=f"Output directory (default: outputs/benchmark/{task})")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Generation budget (default: 4096)")
    parser.add_argument("--local_files_only", action="store_true", help="Load the model from the local cache only")
    parser.add_argument("--dry_run", action="store_true", help="Write prompts and answers without running a model")
    # Options of every task; add_arguments adds the task's input files.
    # Parsing an empty command line lists the names of these options.
    shared_options = vars(parser.parse_args([])).keys()
    if add_arguments:
        add_arguments(parser)
    args = parser.parse_args(argv)
    if not args.dry_run and not args.model:
        parser.error("--model is required unless --dry_run is set")
    if args.num_variants < 1:
        parser.error("--num_variants must be positive")
    try:
        ks = [int(k) for k in args.similarity_k.split(",") if k.strip()]
    except ValueError:
        parser.error(f"--similarity_k must be comma-separated values from {sweep}")
    # Each --similarity_k value adds a high_k<K> condition to the similarity axis.
    if any(k not in SIMILARITY_SWEEP_K[task] for k in ks) or len(set(ks)) != len(ks):
        parser.error(f"--similarity_k must be distinct values from {sweep}")
    if ks and args.axis not in ("similarity", "all"):
        parser.error("--similarity_k requires --axis similarity or all")
    args.similarity_k = tuple(ks)
    # cogloadbench/ holds the prompts of the paper's seed and the task's default input files.
    def is_default(option):
        """Whether an option has its default value; input files are compared by resolved path."""
        value, default = getattr(args, option), parser.get_default(option)
        if isinstance(value, str) and isinstance(default, str) and value and default:
            return os.path.realpath(value) == os.path.realpath(default)
        return value == default
    # Every task-specific option (the input files) must have its default.
    args.shipped_prompts = args.seed == PAPER_SEED and all(map(is_default, vars(args).keys() - shared_options))
    return args


class Generator:
    """Greedy generation with a local Hugging Face model."""

    def __init__(self, model_id: str, max_new_tokens: int, local_files_only: bool):
        """Load the model and tokenizer."""
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        # bfloat16 on GPU, the default dtype on CPU; layers are placed across available devices.
        kwargs = dict(torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                      device_map="auto", local_files_only=local_files_only)
        # Use FlashAttention 2 when it is installed.
        if importlib.util.find_spec("flash_attn"):
            kwargs["attn_implementation"] = "flash_attention_2"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        # generate() needs a pad token; reuse EOS when the tokenizer has none.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Context length in tokens, when the model config states one.
        self.max_length = getattr(self.model.config, "max_position_embeddings", None)
        # Qwen3 chat templates add a thinking block unless it is disabled.
        self.template_kwargs = {"enable_thinking": False} if "qwen3" in model_id.lower() else {}

    def chat_text(self, prompt: str) -> str:
        """Apply the tokenizer's chat template to a single user message."""
        text = prompt.strip()
        # Tokenizers without a chat template (base models) receive the plain prompt text;
        # a template that fails to render raises rather than silently falling back.
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                      add_generation_prompt=True, **self.template_kwargs)
        return text

    def __call__(self, prompt: str) -> str:
        """Return the response's first JSON object with an answer field, else its first JSON object, else the full response."""
        # A chat template already writes any BOS token, so no special tokens are added.
        templated = bool(getattr(self.tokenizer, "chat_template", None))
        inputs = self.tokenizer(self.chat_text(prompt), return_tensors="pt",
                                add_special_tokens=not templated).to(self.model.device)
        length = inputs["input_ids"].shape[-1]
        # Refuse prompts longer than the model's context instead of letting it truncate them.
        if self.max_length and length > self.max_length:
            raise ValueError(f"Prompt has {length} tokens; {self.model_id} accepts {self.max_length}")
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id)
        # Decode only the generated tokens after the prompt.
        text = self.tokenizer.decode(output[0][length:], skip_special_tokens=True).strip()
        return answer_object(text) or text


def dry_run(task: str, args: argparse.Namespace, build: Build) -> None:
    """Write the prompts and gold answers of the selected axes, one file per axis."""
    os.makedirs(args.outdir, exist_ok=True)
    total = 0
    for axis in (AXES if args.axis == "all" else (args.axis,)):
        with open(os.path.join(args.outdir, f"{axis}.jsonl"), "w", encoding="utf-8") as out:
            for condition in conditions(task, axis, args.similarity_k):
                for variant in range(args.num_variants):
                    prompt, answer, fields = build(condition, variant)
                    # The ID encodes task, axis, condition, and zero-padded variant index.
                    record = {"id": f"{task}-{axis}-{condition.name}-{variant:03d}", "task": task, "axis": axis,
                              "condition": condition.name, "variant": variant, "prompt": prompt, "answer": answer,
                              **fields}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
    print(f"Wrote {total} prompts to {args.outdir}")


def evaluate(task: str, args: argparse.Namespace, build: Build, score: Score) -> None:
    """Run the model on every prompt without a saved record and append one record per prompt."""
    # Results are saved under the last path component of the model ID (org/Model -> Model).
    model_name = args.model.rstrip("/").split("/")[-1]
    # Rebuild every prompt first, check it against saved records, and queue the ones
    # without a record; a changed prompt stops the run before the model loads.
    pending = []
    for axis in (AXES if args.axis == "all" else (args.axis,)):
        path = os.path.join(args.outdir, axis, f"{model_name}.jsonl")
        hashes = load_saved(path) if os.path.exists(path) else {}
        for condition in conditions(task, axis, args.similarity_k):
            for variant in range(args.num_variants):
                prompt, gold, fields = build(condition, variant)
                key = (condition.name, variant)
                if key in hashes and hashes[key] != prompt_hash(prompt):
                    raise ValueError(f"{path}: the saved prompt for {condition.name} variant {variant} differs "
                                     "from the current prompt; use a new --outdir")
                if key not in hashes:
                    pending.append((path, condition, variant, prompt, gold, fields))
    # The model is loaded only when there is something left to run.
    generate = Generator(args.model, args.max_new_tokens, args.local_files_only) if pending else None
    scored_count = correct_count = 0
    for index, (path, condition, variant, prompt, gold, fields) in enumerate(pending):
        raw = generate(prompt)
        parsed = parse_json_output(raw)
        pred = str(parsed.get("answer", "")).strip()
        correct = int(score(pred, gold))
        # raw_model_text holds the response as returned by the generator: its first JSON object when it has one.
        record = {"condition": condition.name, "variant": variant, **fields, "prompt": prompt, "raw_model_text": raw,
                  "parsed": parsed, "pred": pred, "gold": gold, "correct": correct}
        # Append and sync one record at a time so an interrupted run loses at most the current prompt.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        scored_count += 1
        correct_count += correct
        # Print the accuracy of this run's prompts at the end of each condition.
        if index + 1 == len(pending) or pending[index + 1][1] != condition:
            print(f"{condition.axis} {condition.name}: {correct_count}/{scored_count} correct")
            scored_count = correct_count = 0
    print(f"Wrote results to {args.outdir}" if pending else f"Nothing to run: {args.outdir} has every result")


def main(task: str, make_build: Callable[[argparse.Namespace], Build], score: Score,
         argv: Optional[List[str]] = None,
         add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None) -> None:
    """Parse the command line, then write prompts (--dry_run) or evaluate a model."""
    args = parse_args(task, argv, add_arguments)
    build = make_build(args)
    if args.shipped_prompts:
        build = with_shipped_prompts(task, build)
    if args.dry_run:
        dry_run(task, args, build)
    else:
        evaluate(task, args, build, score)
