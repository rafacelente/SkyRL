"""The logprob-mismatch diagnostic: top-k selection, boundary classification, never raising."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

from loguru import logger as loguru_logger  # noqa: E402

from skyrl.train.utils.logprob_mismatch import log_top_logprob_mismatches  # noqa: E402


@contextmanager
def capture_loguru(level="INFO"):
    """pytest's caplog only sees stdlib logging; loguru needs its own sink."""
    messages = []
    handler_id = loguru_logger.add(lambda m: messages.append(str(m)), level=level)
    try:
        yield messages
    finally:
        loguru_logger.remove(handler_id)


class FakeTokenizer:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in (ids if isinstance(ids, list) else ids))


def batch(rollout, train, mask, prompt_len=2):
    rollout = torch.tensor(rollout, dtype=torch.float32)
    train = torch.tensor(train, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.int64)
    n, response_length = rollout.shape
    sequences = torch.arange(n * (prompt_len + response_length)).reshape(n, -1)
    return (
        {"rollout_logprobs": rollout, "loss_mask": mask, "sequences": sequences},
        train,
    )


def test_finds_the_planted_worst_token_and_flags_boundaries():
    # Row 0: interior mismatch of 3.0 at col 2. Row 1: boundary mismatch of 2.0 at col 0.
    training_input, train = batch(
        rollout=[[-1.0, -1.0, -4.0, -1.0], [-3.0, -1.0, -1.0, -1.0]],
        train=[[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]],
        mask=[[1, 1, 1, 1], [1, 1, 1, 0]],
    )
    metrics = {}
    with capture_loguru() as messages:
        log_top_logprob_mismatches(training_input, train, FakeTokenizer(), top_k=2, all_metrics=metrics)
    text = "".join(messages)
    assert "diff=3.000" in text and "diff=2.000" in text
    assert "MASK-START" in text, "row 1 col 0 is the first trainable token"
    assert "policy/logprob_mismatch_boundary_share" in metrics
    # Pool = 7 nonzero-mask positions... only positions with diff>0 count: two of them, one on a boundary.
    assert metrics["policy/logprob_mismatch_boundary_share"] == pytest.approx(0.5)


def test_masked_positions_never_selected():
    training_input, train = batch(
        rollout=[[-9.0, -1.0]],
        train=[[-1.0, -1.0]],
        mask=[[0, 1]],  # the huge gap is loss-masked
    )
    with capture_loguru() as messages:
        log_top_logprob_mismatches(training_input, train, FakeTokenizer(), top_k=3)
    assert "diff=8.000" not in "".join(messages)


def test_all_masked_or_zero_diff_is_silent():
    training_input, train = batch(rollout=[[-1.0, -1.0]], train=[[-1.0, -1.0]], mask=[[1, 1]])
    with capture_loguru() as messages:
        log_top_logprob_mismatches(training_input, train, FakeTokenizer(), top_k=3)
    assert "top logprob mismatches" not in "".join(messages)


def test_diagnostic_never_raises():
    """A broken tokenizer must degrade to a warning, not kill the training step."""

    class ExplodingTokenizer:
        def decode(self, ids):
            raise RuntimeError("boom")

    training_input, train = batch(rollout=[[-4.0, -1.0]], train=[[-1.0, -1.0]], mask=[[1, 1]])
    with capture_loguru(level="WARNING") as messages:
        log_top_logprob_mismatches(training_input, train, ExplodingTokenizer(), top_k=1)
    assert "diagnostic failed" in "".join(messages)
