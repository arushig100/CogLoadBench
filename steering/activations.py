"""Prompt regions and activation features, final-token vectors, and projective steering hooks.

The features are computed by extraction.feature_extraction, as in benchmark feature
extraction. Torch is imported lazily so API-only runs do not need model dependencies.
"""
from __future__ import annotations

from typing import List

import numpy as np

# Prompt markers. The context region runs from the start of the templated prompt to
# the question marker, and the question region from the question marker up to the
# answer-instruction marker, as the benchmark's question and FORMAT_LABEL markers do.
# Single-question prompts mark the question with "Question:". In extra_questions
# prompts the question region starts at "Q1:" and holds all listed questions (the
# target and the extra ones), since the target's position among them is shuffled.
# Every CUAD and QASPER prompt starts its answer instructions with "Respond in JSON".
# In summary prompts the question region ends at the summary request, so the
# secondary task lies outside it, as in the benchmark.
QUESTION_LABEL = "Question:"
MULTI_QUESTION_LABEL = "Q1:"
ANSWER_INSTRUCTIONS_LABEL = "Respond in JSON"
SUMMARY_REQUEST_LABEL = "Additionally, in one sentence"


def question_label(setting: str) -> str:
    """The marker that opens the question region of a setting's prompts."""
    return MULTI_QUESTION_LABEL if setting == "extra_questions" else QUESTION_LABEL


# --- Prompt regions and features ---
def prompt_spans(text: str, input_ids: list, offsets: list, tokenizer, q_label: str,
                 truncated: bool = False) -> dict:
    """Token spans of the context and question regions of a templated prompt.

    offsets are the character offsets of input_ids in text. Each marker is located at
    its last occurrence by feature_extraction's label search, so marker text inside
    the context is skipped. In a truncated prompt, the last 20% of tokens is the
    question region. Returns {"context": (start, end), "question":
    (start, end)} as half-open token index ranges. Raises ValueError if a marker is
    missing or the question region is empty.
    """
    if truncated:
        # max(1, ...) keeps at least one token in the context region.
        q_pos = max(1, int(0.8 * len(input_ids)))
        return {"context": (0, q_pos), "question": (q_pos, len(input_ids))}
    from extraction.feature_extraction import _find_label_token_pos
    q_pos = _find_label_token_pos(text, input_ids, offsets, q_label, tokenizer, last=True)
    fmt_pos = _find_label_token_pos(text, input_ids, offsets, ANSWER_INSTRUCTIONS_LABEL, tokenizer, last=True)
    if q_pos == -1 or fmt_pos == -1:
        raise ValueError(f"{q_label!r} or {ANSWER_INSTRUCTIONS_LABEL!r} not found in the tokenized prompt")
    if not q_pos < fmt_pos:
        raise ValueError(f"Empty question region between {q_label!r} and {ANSWER_INSTRUCTIONS_LABEL!r}")
    # A summary request between the question and the answer instructions ends the region.
    # The label search returns -1 when the prompt has no summary request, which fails the
    # range test below.
    summary_pos = _find_label_token_pos(text, input_ids, offsets, SUMMARY_REQUEST_LABEL, tokenizer, last=True)
    end = summary_pos if q_pos < summary_pos < fmt_pos else fmt_pos
    return {"context": (0, q_pos), "question": (q_pos, end)}


def prompt_features(hidden_states, spans: dict) -> dict:
    """Feature-record fields for one prompt, computed as benchmark extraction computes them.

    hidden_states is the model's tuple of [1, seq_len, dim] states (embeddings first).
    Every decoder layer is used, with a 512-token effective-rank cap and no generation
    trace (the defaults of extract_features apart from --gen-trace-tokens 0). Each
    layer is cast to fp32 on the device of the last state, one layer at a time.
    """
    from extraction.feature_extraction import Fp32HiddenStates, prompt_feature_metrics
    states = Fp32HiddenStates(hidden_states, hidden_states[-1].device)
    return prompt_feature_metrics(states, spans, rank_token_cap=512, gen_trace={})


def layer_vectors(hidden_states) -> dict:
    """Return the final prompt token's hidden state (query_last) for each decoder layer."""
    def compact(t):
        """Detached, contiguous CPU copy of a vector."""
        # Clone so torch.save stores only the vector, not the hidden-state storage it views.
        return t.detach().cpu().clone().contiguous()

    # hidden_states[0] is the embedding output, so li indexes decoder layers from 0.
    # Each entry is {"query_last": [hidden]} for the last prompt token.
    vectors = {}
    for li, h in enumerate(hidden_states[1:]):
        h = h.squeeze(0)
        vectors[li] = {"query_last": compact(h[-1])}
    return vectors


# --- Steering hooks ---
def find_decoder_layers(model):
    """Return the ModuleList of decoder blocks, model.model.layers in Qwen3 and Llama models."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("Steering hooks need decoder blocks at model.model.layers (Qwen3- and Llama-style models)")


def make_projective_hook(v_hat_np_layer, alpha):
    """Return a forward hook that adds alpha * <h, v_hat> v_hat to every prefill position except the last, then restores each token's norm."""
    import torch
    # v_hat_np_layer is the layer's unit direction, a float32 [hidden] array.
    cache = {}  # v_hat as a tensor, per (device, dtype).
    alpha_f = float(alpha)

    def fn(module, inputs, output):
        """Steer the block's [batch, seq, hidden] output states; other outputs pass through unchanged."""
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = None
        if not torch.is_tensor(h):
            return output
        # A single position is a cached decoding step: generated tokens are not steered.
        if h.shape[-2] <= 1:
            return output
        key = (h.device, h.dtype)
        v_hat = cache.get(key)
        if v_hat is None:
            v_hat = torch.tensor(v_hat_np_layer, dtype=h.dtype, device=h.device)
            cache[key] = v_hat
        # Scale each token's component along v_hat by (1 + alpha); alpha in (-1, 0)
        # shrinks it, alpha = -1 removes it, and alpha < -1 reverses it. coef is
        # [batch, seq, 1]; delta has the shape of h.
        coef = (h * v_hat).sum(dim=-1, keepdim=True)
        delta = alpha_f * coef * v_hat
        # Prefill: steer every position except the last prompt token.
        new_h = h.clone()
        new_h[..., :-1, :] = new_h[..., :-1, :] + delta[..., :-1, :]
        # Rescale each token back to its original L2 norm, so steering changes only direction.
        # Norms are taken in fp32; the 1e-6 floor guards against division by zero.
        orig_norm = h.float().norm(dim=-1, keepdim=True)
        new_norm = new_h.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        new_h = new_h * (orig_norm / new_norm).to(dtype=new_h.dtype, device=new_h.device)
        if rest is not None:
            return (new_h,) + rest
        return new_h

    return fn


def install_steering_hooks(model, directions_path: str, alpha: float) -> List:
    """Register projective steering hooks at the layers saved in a direction archive; return the handles."""
    # directions holds one direction per decoder layer; steering_layers lists the layers to steer.
    npz = np.load(directions_path)
    directions = npz["directions"].astype(np.float32)
    layers = [int(L) for L in npz["steering_layers"]]
    decoder = find_decoder_layers(model)
    if directions.shape[0] != len(decoder):
        raise ValueError(f"{directions_path} has {directions.shape[0]} layer directions; the model has {len(decoder)} decoder layers")

    handles = []
    for L in layers:
        if not 0 <= L < len(decoder):
            raise ValueError(f"Steering layer {L} is outside the model's {len(decoder)} decoder layers")
        v_np = directions[L]
        v_norm = float(np.linalg.norm(v_np))
        if v_norm < 1e-8:
            raise ValueError(f"Steering direction at layer {L} has zero norm")
        v_hat = (v_np / v_norm).astype(np.float32)
        handles.append(decoder[L].register_forward_hook(make_projective_hook(v_hat, alpha)))
    print(f"Steering alpha={alpha:+.4f} at layers {layers} from {directions_path}")
    return handles
