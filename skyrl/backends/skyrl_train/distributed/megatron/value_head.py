"""Same-position 2-class value head for Megatron value-model training.

The FSDP path uses ``get_llm_for_sequence_regression`` (``Linear(H, 2)`` on
last hidden). Megatron has no critic worker; this module attaches an equivalent
replicated head on the last PP stage and scores packed / unpacked hidden
states without the vocab-parallel LM head.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def attach_value_head(
    model_or_models: Union[nn.Module, List[nn.Module]],
    num_labels: int = 2,
) -> Union[nn.Module, List[nn.Module]]:
    """Add a ``value_head`` Linear on last-PP-stage chunks (pre-DDP hook).

    Intermediate pipeline stages have ``post_process=False`` and are left
    unchanged. The head is replicated across TP (not vocab-parallel); all TP
    ranks see the same data so grads stay in sync.
    """
    models = model_or_models if isinstance(model_or_models, list) else [model_or_models]
    for model in models:
        if not getattr(model, "post_process", False):
            continue
        if hasattr(model, "value_head"):
            continue
        hidden_size = int(model.config.hidden_size)
        head = nn.Linear(hidden_size, num_labels, bias=False)
        head.weight.data.normal_(mean=0.0, std=1.0 / (hidden_size + 1))
        output_weight = getattr(getattr(model, "output_layer", None), "weight", None)
        if output_weight is not None:
            head = head.to(device=output_weight.device, dtype=output_weight.dtype)
        model.add_module("value_head", head)
    return model_or_models


def align_reward_labels(rewards: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
    """Broadcast response-window labels onto the full left-padded sequence.

    ``rewards`` is ``[B, S]`` (already same-position) or ``[B, A]`` (last
    ``A`` tokens). Class ids stay integer.
    """
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be [B, S] or [B, A], got shape {tuple(rewards.shape)}")
    if rewards.shape[0] != sequences.shape[0]:
        raise ValueError(
            f"rewards batch {rewards.shape[0]} does not match sequences batch {sequences.shape[0]}"
        )
    if rewards.shape[-1] == sequences.shape[-1]:
        return rewards.long()
    if rewards.shape[-1] > sequences.shape[-1]:
        raise ValueError(
            f"rewards width {rewards.shape[-1]} exceeds sequence length {sequences.shape[-1]}"
        )
    aligned = torch.zeros(
        sequences.shape[0],
        sequences.shape[1],
        dtype=torch.long,
        device=rewards.device,
    )
    aligned[:, -rewards.shape[-1] :] = rewards.long()
    return aligned


def value_head_logprobs_unpacked(
    hidden: torch.Tensor,
    value_head: nn.Module,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Same-position class log-probs. ``hidden`` / ``labels`` are ``[B, S, H]`` / ``[B, S]``."""
    logits = value_head(hidden)
    if temperature != 1.0:
        logits = logits / temperature
    return F.log_softmax(logits.float(), dim=-1).gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)


def value_head_logprobs_packed(
    hidden: torch.Tensor,
    value_head: nn.Module,
    packed_labels: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    unpacked_seqlen: int,
    attention_mask: Optional[torch.Tensor] = None,
    sub_seq_lengths: Optional[list[list[int]]] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Same-position class log-probs for THD-packed (and CP-sharded) hidden states.

    ``hidden`` is ``[1, T//CP, H]``. ``packed_labels`` is the full packed
    ``[1, T]`` label vector (not CP-sharded), matching ``_build_packed_targets``.
    Returns ``[B, unpacked_seqlen]`` — last-token included, unlike NTP helpers
    that emit ``[B, S-1]``.
    """
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        _packed_cp_rank_and_local_indices,
        _packed_sequence_indices,
        _packed_subseq_row_indices_offsets_and_lens,
        allgather_cp_sharded_packed_tensor,
    )

    hidden = hidden.squeeze(0)
    labels = packed_labels.squeeze(0)
    batch_size = len(sub_seq_lengths) if sub_seq_lengths is not None else cu_seqlens_padded.shape[0] - 1
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else torch.distributed.get_rank(cp_group)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=labels.device, dtype=torch.bool)

    cu_seqlens_padded, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
        cu_seqlens_padded, labels.shape[0], labels.device
    )
    same_pos_labels = labels[cu_seqlens_padded[seq_indices] + seq_offsets]
    if cp_size > 1:
        cp_rank_for_token, local_indices = _packed_cp_rank_and_local_indices(
            cu_seqlens_padded, seq_indices, seq_offsets, seq_lens_padded, cp_size
        )
        local_labels = torch.empty(labels.shape[0] // cp_size, dtype=labels.dtype, device=labels.device)
        current_rank_mask = cp_rank_for_token == cp_rank
        local_labels[local_indices[current_rank_mask]] = same_pos_labels[current_rank_mask]
    else:
        local_labels = same_pos_labels

    logits = value_head(hidden)
    if temperature != 1.0:
        logits = logits / temperature
    probs = F.log_softmax(logits.float(), dim=-1).gather(-1, local_labels.long().unsqueeze(-1)).squeeze(-1)
    if probs.dim() != 1:
        raise ValueError(f"Expected packed class log-probs to be 1D, got shape {tuple(probs.shape)}")

    if cp_size > 1:
        probs = allgather_cp_sharded_packed_tensor(probs, cu_seqlens_padded, cp_group)

    out_logprobs = torch.zeros((batch_size, unpacked_seqlen), dtype=probs.dtype, device=probs.device)
    _, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
        cu_seqlens_padded, probs.shape[0], probs.device
    )

    if sub_seq_lengths is not None:
        row_indices, row_offsets, seq_lens = _packed_subseq_row_indices_offsets_and_lens(
            cu_seqlens_padded, sub_seq_lengths, probs.device
        )
        packed_mask = seq_offsets < seq_lens[seq_indices]
        output_cols = row_offsets[seq_indices[packed_mask]] + seq_offsets[packed_mask]
        output_rows = row_indices[seq_indices[packed_mask]]
        output_in_bounds = output_cols < unpacked_seqlen
        out_logprobs[output_rows[output_in_bounds], output_cols[output_in_bounds]] = probs[packed_mask][
            output_in_bounds
        ]
        return out_logprobs

    if attention_mask is not None:
        seq_lens = attention_mask.sum(dim=1, dtype=torch.long)
        packed_mask = seq_offsets < seq_lens[seq_indices]
        out_logprobs[attention_mask] = probs[packed_mask]
        return out_logprobs

    packed_mask = (seq_offsets < seq_lens_padded[seq_indices]) & (seq_offsets < unpacked_seqlen)
    out_logprobs[seq_indices[packed_mask], seq_offsets[packed_mask]] = probs[packed_mask]
    return out_logprobs
