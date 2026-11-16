"""Helpers shared by the CUAD and QASPER runners: option checks, outputs, parsing, and local generation.

Torch and transformers are imported inside the functions that need them, so API-only
runs do not require them.
"""
import json
import math
import os
import re

# --- Steering axes and condition names ---
AXES = ("quantity", "similarity", "interference")  # Load axes a direction can be evaluated on.
# The setting each steering axis evaluates.
AXIS_SETTINGS = {
    "cuad": {"quantity": "k50", "similarity": "high_k50", "interference": "extra_questions"},
    "qasper": {"quantity": "k5", "similarity": "high_k5", "interference": "extra_questions"},
}
# Condition parts (--part), distractor kinds (--distractors) and secondary tasks (--interference_types).
PARTS = ("load", "interference", "all")
DISTRACTORS = ("random", "similar")
INTERFERENCE_TYPES = ("none", "extra_questions", "summary")
API_MAX_TOKENS = 4096  # Output cap for API models, including reasoning tokens.
API_ATTEMPTS = 4  # API calls per prompt before the task fails.
# Local prompts are cut to this many tokens, Qwen3's native context length.
MAX_INPUT_TOKENS = 32768


def load_setting(k, distractors):
    """Name a load setting: k<K> with random distractors (K=1 has none), high_k<K> with similar ones."""
    return f"high_k{k}" if distractors == "similar" and k > 1 else f"k{k}"


# --- Option checks ---
def parse_list(value, name, parser, allowed=None, cast=str):
    """Split a comma-separated option into distinct values, failing through parser.error."""
    try:
        items = [cast(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError:
        parser.error(f"--{name} must be a comma-separated list" + (" of integers" if cast is int else ""))
    if not items:
        parser.error(f"--{name} must not be empty")
    if len(items) != len(set(items)):
        parser.error(f"--{name} must list distinct values")
    if allowed is not None and not set(items) <= set(allowed):
        parser.error(f"--{name} must contain values from {list(allowed)}")
    return items


def validate_options(parser, args):
    """Reject unusable settings and normalize paths before loading data or models."""
    for name in ("num_variants", "max_concurrent", "max_input_tokens", "max_new_tokens",
                 "interference_k"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_rpm < 0:
        parser.error("--max_rpm must be nonnegative; zero disables rate limiting")
    if not math.isfinite(args.steer_alpha):
        parser.error("--steer_alpha must be finite")
    # Output files are named by alpha at two decimals.
    if round(args.steer_alpha, 2) != args.steer_alpha:
        parser.error("--steer_alpha must have at most two decimal places")
    if (args.directions or args.els_mode) and not args.local:
        parser.error("--directions and --els_mode require --local")
    # --els_mode records unsteered states for direction building; a steered run would overwrite them.
    if args.directions and args.els_mode:
        parser.error("--els_mode records unsteered runs; omit --directions")
    if args.steer_alpha != 0 and not args.directions:
        parser.error("--steer_alpha requires --directions")
    for name in ("directions", "steer_test_pair_keys_json"):
        value = getattr(args, name)
        if value:
            value = os.path.abspath(os.path.expanduser(value))
            if not os.path.isfile(value):
                parser.error(f"--{name}: no such file: {value}")
            setattr(args, name, value)
    args.k_values = parse_list(args.k_values, "k_values", parser, cast=int)
    if min(args.k_values) <= 0:
        parser.error("--k_values must contain positive integers")
    args.distractors = parse_list(args.distractors, "distractors", parser, DISTRACTORS)
    args.interference_types = parse_list(args.interference_types, "interference_types", parser, INTERFERENCE_TYPES)
    args.outdir = os.path.abspath(os.path.expanduser(args.outdir))


def steering_setting(dataset, axis, conditions):
    """Return the setting an axis evaluates, or None; raise if the selected conditions omit it."""
    if axis is None:
        return None
    setting = AXIS_SETTINGS[dataset][axis]
    if setting not in conditions:
        raise ValueError(f"--steer_axis_filter {axis} evaluates {setting}, which the selected conditions (--part, --k_values, --distractors, --interference_types) do not include")
    return setting


# --- Output locations and resume ---
def model_slug(model):
    """Filesystem-safe model name, e.g. Qwen/Qwen3-14B -> Qwen-Qwen3-14B."""
    return model.replace("/", "-").replace(".", "-")


def output_path(args, dataset):
    """Return the prediction JSONL for this run.

    Steered runs are named by axis and alpha, plus the stem of the
    --steer_test_pair_keys_json file when one is given, e.g.
    <outdir>/Qwen-Qwen3-14B/cuad_steer_quantity_alpha-0.50_cuad_Qwen-Qwen3-14B_test_keys.jsonl.
    """
    slug = model_slug(args.model)
    if args.directions:
        name = f"{dataset}_steer_{args.steer_axis_filter or 'all'}_alpha{args.steer_alpha:+.2f}"
        if args.steer_test_pair_keys_json:
            name += "_" + os.path.splitext(os.path.basename(args.steer_test_pair_keys_json))[0]
        name += ".jsonl"
    else:
        name = f"{dataset}_predictions_{slug}.jsonl"
    return os.path.join(args.outdir, slug, name)


def vectors_dir(args):
    """Directory for the per-prompt layer vectors that build_directions.py reads."""
    return os.path.join(args.outdir, "els_vectors", model_slug(args.model))


def save_vectors(directory, setting, variant, layers, **scores):
    """Save one prompt's final-token layer vectors with its scores as els__<setting>__v<variant>.pt."""
    import torch
    os.makedirs(directory, exist_ok=True)
    torch.save({"setting": setting, "variant": variant, **scores, "layers": layers},
               os.path.join(directory, f"els__{setting}__v{variant:04d}.pt"))


def load_completed(path, directions=None):
    """Return task hashes already in a resume file; reject damaged JSONL instead of appending to it.

    A steered run resumes only records written with a direction archive of the same file
    name (`directions`), so results from another archive are never reused.
    """
    archive = os.path.basename(directions) if directions else None
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid resume record; repair the file before resuming") from error
            if not isinstance(record, dict) or not isinstance(record.get("task_hash"), str) or not record["task_hash"]:
                raise ValueError(f"{path}:{line_number}: resume record requires a nonempty task_hash")
            if not line.endswith("\n"):
                raise ValueError(f"{path}:{line_number}: resume record lacks a final newline; repair the file before resuming")
            if record.get("steer_directions") != archive:
                raise ValueError(f"{path}:{line_number}: record was written with directions {record.get('steer_directions')!r}, "
                                 f"not {archive!r}; use another --outdir")
            done.add(record["task_hash"])
    return done


# --- API client ---
def api_client(anthropic):
    """Return an async Anthropic client if `anthropic`, else an OpenAI one; the API key must be set."""
    variable = "ANTHROPIC_API_KEY" if anthropic else "OPENAI_API_KEY"
    if not os.environ.get(variable):
        raise RuntimeError(f"Set {variable} to run API models")
    if anthropic:
        from anthropic import AsyncAnthropic
        return AsyncAnthropic()
    from openai import AsyncOpenAI
    return AsyncOpenAI()


# --- Answer parsing ---
def parse_answer(raw, key="answer"):
    """Return the `key` (else "answer") field of JSON-like model output, else the first 500 characters."""
    if not raw:
        return ""
    raw = raw.strip()
    # Try, in order: the whole reply as JSON, the first flat {...} object in it, and a
    # "key": "value" pattern (which also matches unterminated JSON).
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            if key in obj:
                return str(obj[key]).strip()
            if "answer" in obj:
                return str(obj["answer"]).strip()
    except Exception:
        pass
    # The first brace pair with no braces inside it.
    m = re.search(r'\{[^{}]+\}', raw)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                if key in obj:
                    return str(obj[key]).strip()
                if "answer" in obj:
                    return str(obj["answer"]).strip()
        except Exception:
            pass
    # "key": "value" or "answer": "value"; the value ends at its first double quote.
    m = re.search(r'"(?:' + re.escape(key) + r'|answer)"\s*:\s*"([^"]*)"', raw)
    if m:
        return m.group(1).strip()
    return raw[:500]


# --- Local generation ---
def load_local_model(name, directions=None, alpha=0.0):
    """Load a float16 model and tokenizer; install steering hooks when alpha is nonzero."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading local model: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    if directions and alpha != 0:
        from steering.activations import install_steering_hooks
        install_steering_hooks(model, directions, alpha)
    return model, tokenizer


def generate_local(model, tokenizer, prompt, *, model_name, max_input_tokens, max_new_tokens,
                   q_label=None):
    """Generate a greedy answer; with q_label (--els_mode), also return prompt features and final-token layer vectors.

    Returns (text, features, vectors); features and vectors are None without q_label.
    q_label is the marker that opens the prompt's question region (see
    activations.prompt_spans).
    """
    import torch
    from steering.activations import layer_vectors, prompt_features, prompt_spans

    # The prompt is sent as a single user turn in the model's chat template and cut to
    # its first max_input_tokens tokens.
    template_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "qwen3" in model_name.lower():
        template_kwargs["enable_thinking"] = False  # Avoid long reasoning output.
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **template_kwargs)
    # The chat template already writes any BOS token, so no special tokens are added.
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                       return_offsets_mapping=q_label is not None)
    # truncated is measured before the cut and decides how prompt_spans finds the question.
    truncated = int(inputs["input_ids"].shape[1]) > max_input_tokens
    inputs = {k: v[:, :max_input_tokens] for k, v in inputs.items()}
    offsets = inputs.pop("offset_mapping")[0].tolist() if q_label is not None else None
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Features come from a separate forward pass of the base transformer over the
    # prompt and are computed on the device of its last hidden state. The base
    # transformer (model.model) has no LM head, so this pass computes no logits.
    features = vectors = None
    if q_label is not None:
        spans = prompt_spans(text, inputs["input_ids"][0].tolist(), offsets, tokenizer, q_label,
                             truncated=truncated)
        with torch.no_grad():
            forward = getattr(model, "model", model)(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True, use_cache=False)
            hidden_states = forward.hidden_states
            del forward
            features = prompt_features(hidden_states, spans)
            vectors = layer_vectors(hidden_states)
            # Release the [1, seq_len, dim] per-layer states before generation.
            del hidden_states
            torch.cuda.empty_cache()

    # Greedy decoding; temperature and top_p are unset since nothing is sampled. Only
    # the newly generated tokens are decoded.
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), features, vectors
