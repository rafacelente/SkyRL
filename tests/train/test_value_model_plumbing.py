"""CPU tests for value-model freeze + same-position head helpers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from skyrl.backends.skyrl_train.distributed.megatron.value_head import (
    align_reward_labels,
    value_head_logprobs_unpacked,
)
from skyrl.backends.skyrl_train.workers.worker_utils import apply_freeze_modules


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Linear(4, 4, bias=False)
        self.value_head = nn.Linear(4, 2, bias=False)


def test_apply_freeze_modules_substring_match():
    net = _TinyNet()
    frozen = apply_freeze_modules(net, ["self_attn"])
    assert frozen == 1
    assert not net.self_attn.weight.requires_grad
    assert net.mlp.weight.requires_grad
    assert net.value_head.weight.requires_grad


def test_apply_freeze_modules_empty_is_noop():
    net = _TinyNet()
    assert apply_freeze_modules(net, None) == 0
    assert apply_freeze_modules(net, []) == 0
    assert net.self_attn.weight.requires_grad


def test_apply_freeze_modules_list_of_modules():
    nets = [_TinyNet(), _TinyNet()]
    frozen = apply_freeze_modules(nets, ["mlp"])
    assert frozen == 2
    assert all(not n.mlp.weight.requires_grad for n in nets)


def test_align_reward_labels_full_sequence():
    sequences = torch.zeros(2, 5, dtype=torch.long)
    rewards = torch.tensor([[0, 1, 0, 1, 1], [1, 1, 0, 0, 1]])
    aligned = align_reward_labels(rewards, sequences)
    assert torch.equal(aligned, rewards.long())


def test_align_reward_labels_response_window():
    sequences = torch.zeros(2, 5, dtype=torch.long)
    rewards = torch.tensor([[1, 0, 1], [0, 0, 1]])
    aligned = align_reward_labels(rewards, sequences)
    assert aligned.shape == (2, 5)
    assert torch.equal(aligned[:, :2], torch.zeros(2, 2, dtype=torch.long))
    assert torch.equal(aligned[:, -3:], rewards.long())


def test_value_head_logprobs_are_same_position():
    torch.manual_seed(0)
    hidden = torch.randn(2, 6, 8)
    head = nn.Linear(8, 2, bias=False)
    labels = torch.randint(0, 2, (2, 6))
    got = value_head_logprobs_unpacked(hidden, head, labels)
    expected = F.log_softmax(head(hidden).float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(got, expected)
    # NTP would score labels[:, 1:] against hidden[:, :-1]; those must differ here.
    ntp = F.log_softmax(head(hidden[:, :-1]).float(), dim=-1).gather(-1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
    assert got.shape == labels.shape
    assert ntp.shape != got.shape or not torch.allclose(got[:, 1:], ntp)
