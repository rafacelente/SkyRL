"""NaN step weights must reproduce stock GRPO bit-exactly for exactly the affected rows.

This is the graceful-degradation guarantee the Jev scorer relies on: any failure path emits NaN,
and the trainer maps NaN to the identity — not 0 (which erases the row) and not +1 (which flips
failed-trajectory rows positive).

Run:
    uv run --extra dev pytest tests/train/step_wise/test_step_weight_nan_identity.py
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from examples.train.step_wise.mock_grpo_step import (  # noqa: E402
    STEP_WEIGHTS_KEY,
    _cfg,
    _generator_output,
    _trainer,
)


def _weighted_advantages(step_weights):
    """Drive the mock's exact pipeline (postprocess -> convert -> forward -> advantages)."""
    trainer = _trainer(_cfg())
    generator_output = _generator_output()
    generator_output[STEP_WEIGHTS_KEY] = list(step_weights)
    uids = [tid.instance_id for tid in generator_output["trajectory_ids"]]
    generator_output, uids = trainer.postprocess_generator_output(generator_output, uids)
    training_input = trainer.convert_to_training_input(generator_output, uids)

    def mock_forward(model, data, key, mini_batch_boundaries=None):
        return torch.full((data["sequences"].shape[0], data.metadata["response_length"]), -1.0)

    trainer._execute_forward_pass = mock_forward
    training_input = trainer.fwd_logprobs_values_reward(training_input)
    return trainer.compute_advantages_and_returns(training_input)


def test_nan_rows_reproduce_unweighted_advantages_bit_exactly():
    weights = [-0.5, float("nan"), 0.2, 0.9, float("nan"), -0.3, 0.1, float("nan")]
    data = _weighted_advantages(weights)
    adv = data["advantages"]
    unweighted = data["advantages_unweighted"]
    for row, weight in enumerate(weights):
        if math.isnan(weight):
            assert torch.equal(adv[row], unweighted[row]), f"NaN row {row} must train as stock GRPO"
        else:
            assert not torch.equal(adv[row], unweighted[row]) or weight == 1.0 or unweighted[row].abs().sum() == 0


def test_all_nan_is_exactly_stock_grpo():
    data = _weighted_advantages([float("nan")] * 8)
    assert torch.equal(data["advantages"], data["advantages_unweighted"])
    assert data.metadata["metrics"]["step_weight_nan_rate"] == pytest.approx(1.0)


def test_real_weights_scale_magnitude_and_zero_erases():
    weights = [1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0, 1.0]
    data = _weighted_advantages(weights)
    adv, unweighted = data["advantages"], data["advantages_unweighted"]
    mask = data["response_mask"].bool()
    # w=1 -> |A|: same magnitude everywhere.
    assert torch.allclose(adv[0].abs()[mask[0]], unweighted[0].abs()[mask[0]])
    # w=0.5 -> half magnitude; w=0 -> erased.
    assert torch.allclose(adv[1].abs()[mask[1]], 0.5 * unweighted[1].abs()[mask[1]])
    assert adv[2][mask[2]].abs().sum() == 0
    # Positive weight pushes up regardless of the trajectory's own sign: row 4 belongs to the
    # losing trajectory (negative GRPO advantage) but w=+1 must make its advantage positive.
    losing_rows = adv[4][mask[4]]
    assert (losing_rows >= 0).all() and losing_rows.abs().sum() > 0


def test_nan_metrics_are_nan_safe():
    data = _weighted_advantages([0.5, float("nan")] * 4)
    metrics = data.metadata["metrics"]
    assert metrics["step_weight_nan_rate"] == pytest.approx(0.5)
    assert metrics["avg_step_weight"] == pytest.approx(0.5)
    assert not math.isnan(metrics["avg_weighted_advantages"])
