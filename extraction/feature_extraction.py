"""Compute prompt-time activation metrics, neural dynamics and a greedy generation trace for one prompt."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# Prompt wrapping + region (context / question) span detection
# Marker strings the benchmark prompts embed between the context block and the
# repeated question, and before the JSON answer-format block. Token positions
# of these markers define the context/question regions.
QUERY_LABELS = {
    "math":    "\nQuestion to answer (repeated):\n",
    "context": "\nQuestion to answer (repeated):\n",
    "analogy": "Now complete a new analogy using the same pattern:\n",
}
FORMAT_LABEL = "Answer format (return JSON only)"


def wrap_prompt_for_tokenization(tokenizer, prompt: str, is_qwen3: bool = False) -> str:
    """Apply the model's chat template as a user turn (Qwen3 with thinking disabled), if it has one."""
    text = str(prompt).strip()
    if not getattr(tokenizer, "chat_template", None):
        return text
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if is_qwen3:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], **kwargs)


def find_subseq(haystack, needle):
    """Index of the first occurrence of needle in haystack, or -1."""
    if len(needle) == 0 or len(haystack) < len(needle):
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def _find_label_token_pos(tok_prompt: str, full_ids: list, offsets: list, label: str, tokenizer,
                          last: bool = False) -> int:
    """Find a label by exact tokens, stripped tokens, then character offsets.

    offsets are the character offsets of full_ids in tok_prompt. The token searches
    return the first occurrence; the character search uses the last occurrence of
    the stripped label. last=True skips the token searches, so the last occurrence
    is returned. Returns -1 if the label is absent.
    """
    # A label's tokens can differ inside the prompt, because its edge characters (such as
    # newlines) may merge with neighboring text; the stripped label and then character
    # offsets are the fallbacks.
    stripped = label.strip()
    if not last:
        ids = tokenizer(label, add_special_tokens=False).input_ids
        p = find_subseq(full_ids, ids)
        if p != -1:
            return p
        if stripped != label:
            ids = tokenizer(stripped, add_special_tokens=False).input_ids
            p = find_subseq(full_ids, ids)
            if p != -1:
                return p
    char_pos = tok_prompt.rfind(stripped)
    if char_pos < 0:
        return -1
    # First token that starts at or after the label's first character.
    for i, (s, _) in enumerate(offsets):
        if s >= char_pos:
            return i
    return -1


def forward_with_hidden(model, tokenizer, prompt: str, *,
                        q_label: str, fmt_label: str = FORMAT_LABEL,
                        is_qwen3: bool = False, max_length: int = 8192):
    """One forward pass with output_hidden_states; returns (hidden_states, spans).

    hidden_states is a list of [seq_len, dim] fp32 CPU tensors (embeddings at
    index 0, then one per layer); spans marks the context / question token
    ranges located via the q_label / fmt_label markers.
    """
    tok_prompt = wrap_prompt_for_tokenization(tokenizer, prompt, is_qwen3=is_qwen3)
    # A chat template already writes any BOS token, so no special tokens are added.
    templated = bool(getattr(tokenizer, "chat_template", None))
    full = tokenizer(tok_prompt, return_tensors="pt", truncation=True, max_length=max_length,
                     add_special_tokens=not templated,
                     return_offsets_mapping=True)
    # offsets: (start, end) character offsets of each token in tok_prompt; the model
    # does not accept them as input.
    offsets = full.pop("offset_mapping")[0].tolist()
    full = full.to(model.device)
    full_ids = full["input_ids"][0].tolist()

    q_pos = _find_label_token_pos(tok_prompt, full_ids, offsets, q_label, tokenizer)
    fmt_pos = _find_label_token_pos(tok_prompt, full_ids, offsets, fmt_label, tokenizer)

    # Token spans: context = [start, q_label); question = [q_label, fmt_label).
    if q_pos == -1 or fmt_pos == -1:
        raise ValueError("question or answer-format marker not found in the tokenized prompt; check "
                         "--q-label and --fmt-label, or raise --max-len if the prompt was truncated")
    context_start, context_end = 0, q_pos
    question_start, question_end = q_pos, fmt_pos

    with torch.no_grad():
        # Use the base transformer to avoid allocating unused full-sequence logits.
        outputs = getattr(model, "model", model)(
            **full,
            output_attentions=False,
            output_hidden_states=True,
            use_cache=False,
        )

    # Drop the batch dimension: each state becomes [seq_len, dim] fp32 on the CPU, with
    # any NaN or infinite values replaced by finite numbers.
    hidden_states = [torch.nan_to_num(h[0].float().cpu()) for h in outputs.hidden_states]

    spans = {
        "context": (context_start, context_end),
        "question": (question_start, question_end),
    }
    return hidden_states, spans


# Core activation statistics
def gini_coefficient(x: torch.Tensor) -> float:
    """Gini coefficient of |x| over every element (activation sparsity)."""
    # The small offset keeps the denominator nonzero for an all-zero region.
    x = x.abs().flatten() + 1e-12
    x_sorted, _ = torch.sort(x)
    n = x_sorted.numel()
    idx = torch.arange(1, n + 1, dtype=torch.float32, device=x_sorted.device)
    # Gini over ascending values x_1..x_n: 2 * sum(i * x_i) / (n * sum(x)) - (n + 1) / n;
    # 0 for equal magnitudes, near 1 when one element holds all the mass.
    g = (2 * (idx * x_sorted).sum() / (n * x_sorted.sum())) - (n + 1) / n
    return float(g.item())


def _token_norm_stats(H: torch.Tensor) -> Dict[str, float]:
    """Mean of the region's token L2 norms (0 for an empty region)."""
    if H is None or H.numel() == 0:
        return {"mean": 0.0}
    return {"mean": float(H.norm(dim=-1).mean().item())}


def _effective_rank_participation_ratio(H: torch.Tensor, token_cap: int = 512) -> float:
    """Participation-ratio effective rank of a region's centered token matrix; 0 for an empty region, NaN if the SVD fails."""
    if H is None or H.numel() == 0:
        return 0.0
    X = H
    T = X.shape[0]
    # Subsample to at most token_cap rows (evenly spaced) to bound the SVD cost.
    if token_cap is not None and T > token_cap:
        idx = torch.linspace(0, T - 1, steps=token_cap).long().to(X.device)
        X = X.index_select(0, idx)
    # Center each hidden dimension over the region's tokens.
    X = (X - X.mean(dim=0, keepdim=True)).float()
    try:
        s = torch.linalg.svdvals(X)
    except torch.linalg.LinAlgError:
        return float("nan")
    s = s.clamp_min(1e-12)
    # Participation ratio over the singular values s: (sum s)^2 / sum s^2.
    num = (s.sum() ** 2).item()
    den = (s.pow(2).sum()).item()
    return float(num / (den + 1e-12))


def _layer_steps(means: List[torch.Tensor]) -> Tuple[List[float], List[float], List[float]]:
    """Drift (1 - cos), angle in degrees and L2 step norm between consecutive mean vectors."""
    drift, angle, step = [], [], []
    for a, b in zip(means, means[1:]):
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
        # Clamp rounding error so acos stays in its domain.
        cos = max(-1.0, min(1.0, cos))
        drift.append(1.0 - cos)
        angle.append(math.degrees(math.acos(cos)))
        step.append(torch.norm(b - a, p=2).item())
    return drift, angle, step


# Prompt-time per-layer metrics (activation_metrics.prompt_time)
def activation_metrics_from_hidden(
    hidden_states: List[torch.Tensor],
    spans: Dict[str, Tuple[int, int]],
    rank_token_cap: int = 512
) -> Dict[str, Any]:
    """Return token counts and, for every decoder layer, context/question norms, Gini, effective rank and cosine."""
    (cs, ce) = spans["context"]
    (qs, qe) = spans["question"]
    seq_len = int(hidden_states[0].shape[0])

    layers_out: Dict[int, Any] = {}
    for layer_id in range(len(hidden_states) - 1):
        H = hidden_states[layer_id + 1]  # hidden_states[0] is the embedding layer
        Hc = H[cs:ce]   # context-region tokens
        Hq = H[qs:qe]   # question-region tokens

        # Region mean vectors; cos_cq = context/question alignment at this layer. An empty
        # region's mean is the zero vector, which gives a cosine of 0.
        c_mean = Hc.mean(dim=0) if Hc.numel() > 0 else torch.zeros(H.shape[-1], device=H.device)
        q_mean = Hq.mean(dim=0) if Hq.numel() > 0 else torch.zeros(H.shape[-1], device=H.device)
        cos_cq = F.cosine_similarity(c_mean.unsqueeze(0), q_mean.unsqueeze(0), dim=-1).item()

        layers_out[layer_id] = {
            "context": {
                "token_norm": _token_norm_stats(Hc),
                "gini": gini_coefficient(Hc) if Hc.numel() > 0 else 0.0,
                "effective_rank_pr": _effective_rank_participation_ratio(Hc, token_cap=rank_token_cap),
            },
            "question": {
                "token_norm": _token_norm_stats(Hq),
                "gini": gini_coefficient(Hq) if Hq.numel() > 0 else 0.0,
                "effective_rank_pr": _effective_rank_participation_ratio(Hq, token_cap=rank_token_cap),
            },
            "context_question_cos": float(cos_cq),
        }

    return {
        "token_counts": {
            "context": int(max(0, ce - cs)),
            "question": int(max(0, qe - qs)),
            "full": int(seq_len),
        },
        "layers": layers_out,
    }


# Per-layer magnitude and layer-to-layer dynamics per region (neural_dynamics)
def neural_dynamics_from_hidden(hidden_states, spans):
    """Return per-region, per-layer magnitude, drift, angle and step norm.

    The regions are context, question and full (the whole prompt); the arrays cover
    the decoder layers, hidden_states[1:].
    """
    regions = {
        "context": spans["context"],
        "question": spans["question"],
        "full": (0, hidden_states[0].shape[0])
    }
    # Per region and decoder layer: the mean token vector and the mean token L2 norm.
    means = {r: [] for r in regions}
    mags  = {r: [] for r in regions}

    # hidden_states[0] is the embedding output, so the decoder layers start at index 1.
    for layer in range(1, len(hidden_states)):
        h = hidden_states[layer]
        for r, (s, e) in regions.items():
            H = h[s:e]
            if H.numel() == 0:
                means[r].append(torch.zeros(h.shape[-1], device=h.device))
                mags[r].append(0.0)
                continue
            mags[r].append(float(H.norm(dim=-1).mean().item()))
            means[r].append(H.mean(0))

    per_layer = {}
    for r in regions:
        drift, angles, steps = _layer_steps(means[r])
        per_layer[r] = {
            "activation_magnitude": mags[r],
            "drift_to_next_layer": drift,
            "angle_to_next_layer_deg": angles,
            "M_to_next_layer": steps,
        }
    return {"per_layer": per_layer}


def prompt_feature_metrics(hidden_states, spans, rank_token_cap: int = 512, gen_trace=None) -> Dict[str, Any]:
    """Return the activation_metrics and neural_dynamics fields of a feature record.

    gen_trace is the generation_trace_metrics output ({} when no trace is taken).
    """
    return {
        "activation_metrics": {
            "prompt_time": activation_metrics_from_hidden(hidden_states, spans, rank_token_cap=rank_token_cap),
            "generation_trace": gen_trace if gen_trace is not None else {},
        },
        "neural_dynamics": neural_dynamics_from_hidden(hidden_states, spans),
    }


class Fp32HiddenStates:
    """Model hidden states converted one layer at a time as forward_with_hidden converts them.

    Indexing returns torch.nan_to_num(h[0].float()) of that layer on `device`, so the
    feature functions above can run where the states are (e.g. on the GPU) without
    holding an fp32 copy of every layer at once.
    """

    def __init__(self, hidden_states, device):
        """Wrap a tuple of [1, seq_len, dim] hidden states (embeddings first)."""
        self._states = hidden_states
        self._device = device

    def __len__(self):
        """Number of hidden states, embeddings included."""
        return len(self._states)

    def __getitem__(self, index):
        """The [seq_len, dim] fp32 state at index."""
        return torch.nan_to_num(self._states[index][0].to(self._device).float())

    def __iter__(self):
        """The fp32 states in order."""
        return (self[i] for i in range(len(self)))


# Greedy generation trace (activation_metrics.generation_trace)
def _entropy_from_logits_cpu(logits_1d_cpu: torch.Tensor) -> float:
    """Entropy in nats of the softmax of a logit vector."""
    log_probs = logits_1d_cpu - torch.logsumexp(logits_1d_cpu, dim=-1)
    probs = torch.exp(log_probs)
    ent = -(probs * log_probs).sum()
    return float(ent.item())


def generation_trace_metrics(model, tokenizer, prompt: str, *,
                             is_qwen3: bool = False,
                             n_steps: int = 50, max_len: int = 8192) -> Dict[str, Any]:
    """Greedy-decode n_steps tokens; return the mean entropy and mean top-1/top-2 logit margin over the steps."""
    tok_prompt = wrap_prompt_for_tokenization(tokenizer, prompt, is_qwen3=is_qwen3)
    templated = bool(getattr(tokenizer, "chat_template", None))
    enc = tokenizer(tok_prompt, return_tensors="pt", truncation=True, max_length=max_len,
                    add_special_tokens=not templated).to(model.device)

    with torch.no_grad():
        gen_out = model.generate(
            **enc,
            max_new_tokens=int(n_steps),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    entropies: List[float] = []
    margins: List[float] = []
    scores = list(gen_out.scores) if getattr(gen_out, "scores", None) is not None else []
    # scores holds one [1, vocab] logit tensor per generated token.
    for step_scores in scores:
        logits = step_scores[0].detach().float().cpu()
        entropies.append(_entropy_from_logits_cpu(logits))
        # Decision margin: top-1 minus top-2 logit gap (larger = more decisive).
        top2 = torch.topk(logits, k=2).values
        margins.append(float((top2[0] - top2[1]).item()))

    return {"summary": {
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_logit_margin_top1_top2": float(np.mean(margins)) if margins else 0.0,
    }}
