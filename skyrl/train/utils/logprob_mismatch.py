"""Diagnostics for large rollout-vs-train logprob mismatches.

``policy/rollout_train_logprobs_abs_diff_max`` sitting orders of magnitude above the mean says
*some* tokens disagree badly between the inference engine and the training forward pass, but not
which ones. This dumps the top-k offending tokens with enough context to separate the usual
suspects:

- **Chat-template / retokenization drift** concentrates at loss-mask transitions (the first
  trainable token after an observation, or the last before one) and on template glue tokens
  (role markers, newlines, EOS).
- **Engine numerics** (dtype, kernels, MoE expert-routing divergence between vLLM and the trainer)
  spread over interior tokens with no positional structure, typically on low-probability tokens.

The summary line reports what fraction of the top pool sits on mask boundaries: high means look at
the template, low means look at the engines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from loguru import logger

# How many positions feed the boundary-share summary; the per-token dump stays at ``top_k``.
_POOL_SIZE = 128


@torch.no_grad()
def log_top_logprob_mismatches(
    training_input: Any,
    action_log_probs: torch.Tensor,
    tokenizer: Any,
    top_k: int,
    all_metrics: Optional[Dict[str, float]] = None,
    context_tokens: int = 10,
) -> None:
    """Log the ``top_k`` largest |rollout - train| logprob gaps with decoded context.

    Driver-side only: needs the tokenizer and the full sequences, neither of which the training
    workers hold. Never raises — a diagnostic must not be able to kill a run.
    """
    try:
        _log_top_logprob_mismatches(training_input, action_log_probs, tokenizer, top_k, all_metrics, context_tokens)
    except Exception as error:  # noqa: BLE001 — diagnostics are best-effort
        logger.warning(f"logprob-mismatch diagnostic failed: {type(error).__name__}: {error}")


def _log_top_logprob_mismatches(
    training_input: Any,
    action_log_probs: torch.Tensor,
    tokenizer: Any,
    top_k: int,
    all_metrics: Optional[Dict[str, float]],
    context_tokens: int,
) -> None:
    rollout_logprobs: torch.Tensor = training_input["rollout_logprobs"]
    loss_mask: torch.Tensor = training_input["loss_mask"]
    sequences: torch.Tensor = training_input["sequences"]
    response_length = action_log_probs.shape[1]
    # Response tensors are right-aligned: the last ``response_length`` columns of ``sequences``
    # are the tokens the logprob columns describe.
    response_ids = sequences[:, -response_length:]

    diff = (rollout_logprobs - action_log_probs).abs() * (loss_mask > 0)
    flat = diff.flatten()
    if flat.numel() == 0 or float(flat.max()) == 0.0:
        return

    pool = min(_POOL_SIZE, flat.numel())
    top_values, top_indices = torch.topk(flat, k=pool)
    keep = top_values > 0
    top_values, top_indices = top_values[keep], top_indices[keep]
    if top_values.numel() == 0:
        return

    rows = top_indices // response_length
    cols = top_indices % response_length

    mask = loss_mask > 0
    boundary_hits = 0
    lines = []
    for rank in range(top_values.numel()):
        row, col = int(rows[rank]), int(cols[rank])
        at_start = col == 0 or not bool(mask[row, col - 1])
        at_end = col == response_length - 1 or not bool(mask[row, col + 1])
        if at_start or at_end:
            boundary_hits += 1
        if rank >= top_k:
            continue

        token_id = int(response_ids[row, col])
        token = tokenizer.decode([token_id])
        lo = max(0, col - context_tokens)
        hi = min(response_length, col + context_tokens + 1)
        context = tokenizer.decode(response_ids[row, lo:hi].tolist())
        response_len_row = int(mask[row].sum())
        position_in_row = int(mask[row, : col + 1].sum())
        lines.append(
            f"  #{rank + 1} diff={float(top_values[rank]):.3f} "
            f"rollout={float(rollout_logprobs[row, col]):+.3f} train={float(action_log_probs[row, col]):+.3f} "
            f"| row={row} col={col} (trainable token {position_in_row}/{response_len_row}) "
            f"| id={token_id} token={token!r} "
            f"| {'MASK-START ' if at_start else ''}{'MASK-END ' if at_end else ''}"
            f"| ctx={context!r}"
        )

    boundary_share = boundary_hits / top_values.numel()
    logger.info(
        f"top logprob mismatches (pool={top_values.numel()}, boundary share={boundary_share:.0%} — "
        "high points at template/retokenization, low at engine numerics):\n" + "\n".join(lines)
    )
    if all_metrics is not None:
        all_metrics["policy/logprob_mismatch_boundary_share"] = boundary_share
