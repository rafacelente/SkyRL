"""CPU tests for Jev step weights on the Harbor path.

Run:
    uv run --extra dev pytest tests/train/step_wise/test_jev_weights.py

No network, no `harbor` extra, no typesafe-sdk: the generator module imports harbor lazily and the
scorer tests inject a fake client.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from examples.train_integrations.harbor.harbor_generator import (
    HarborTrajectoryOutput,
    build_step_wise_generator_output,
)
from examples.train_integrations.harbor.jev_weights.config import JevWeightsConfig
from examples.train_integrations.harbor.jev_weights.scorer import (
    JevStepScorer,
    _step_specificity_icc,
)
from examples.train_integrations.harbor.jev_weights.state import (
    CONTEXT_MODES,
    Turn,
    build_state,
    clean_observation,
    middle_truncate,
    split_turns,
)
from skyrl.train.generators.base import TrajectoryID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_traj(instance: str, rep: int, n_turns: int, reward: float, step_weights=None, stop="complete"):
    """A minimal HarborTrajectoryOutput with token-in-token-out rollout details."""
    prompts, completions, logprobs = [], [], []
    prompt = [1, 2, 3]
    for t in range(n_turns):
        completion = [100 + 10 * t, 101 + 10 * t]
        obs = [200 + t]
        prompts.append(list(prompt))
        completions.append(completion)
        logprobs.append([-0.1] * len(completion))
        prompt = prompt + completion + obs
    detail = {"prompt_token_ids": prompts, "completion_token_ids": completions, "logprobs": logprobs}
    return HarborTrajectoryOutput(
        trajectory_id=TrajectoryID(instance_id=instance, repetition_id=rep),
        rollout_details=None if stop in ("error",) else [detail],
        reward=reward,
        num_turns=n_turns,
        stop_reason=stop,
        step_weights=step_weights,
    )


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{i}" for i in ids)


class FakeClient:
    """Stands in for AsyncTypeSafeClient; returns canned probabilities, or raises."""

    def __init__(self, probabilities=None, error: Exception | None = None, delay: float = 0.0):
        self.probabilities = probabilities or {"0": 0.1, "1": 0.2, "2": 0.7}
        self.error = error
        self.delay = delay
        self.calls = 0

    async def system_one(self, state, questions):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return {"answers": {"contribution": {"type": "score", "probabilities": dict(self.probabilities)}}}


def make_scorer(tmp_path=None, client=None, **overrides) -> JevStepScorer:
    cfg = JevWeightsConfig(
        enabled=True,
        cache_dir=str(tmp_path / "cache") if tmp_path else None,
        collect_deadline_s=overrides.pop("collect_deadline_s", 5.0),
        **overrides,
    )
    return JevStepScorer(cfg, FakeTokenizer(), client=client or FakeClient())


# ---------------------------------------------------------------------------
# Alignment: the one silent-failure risk
# ---------------------------------------------------------------------------


def test_step_weights_align_one_per_row_in_input_order():
    trajs = [
        make_traj("a", 0, n_turns=3, reward=1.0, step_weights=[0.1, 0.2, 0.3]),
        make_traj("a", 1, n_turns=2, reward=0.0, step_weights=[-0.5, -0.6]),
        make_traj("b", 0, n_turns=4, reward=1.0, step_weights=[0.0, 0.1, 0.2, 0.9]),
    ]
    out = build_step_wise_generator_output(trajs, overlong_filtering=False)
    assert len(out["step_weights"]) == len(out["response_ids"]) == 9
    assert out["step_weights"] == [0.1, 0.2, 0.3, -0.5, -0.6, 0.0, 0.1, 0.2, 0.9]
    # Each trajectory's rows are contiguous and its last row is flagged.
    assert out["is_last_step"] == [False, False, True, False, True, False, False, False, True]


def test_unscored_trajectory_rows_are_nan_and_masked_instances_zero():
    trajs = [
        make_traj("a", 0, n_turns=2, reward=1.0, step_weights=[0.4, 0.5]),
        make_traj("a", 1, n_turns=2, reward=0.0, step_weights=None),  # scoring missed the deadline
        make_traj("c", 0, n_turns=2, reward=0.0, stop="error"),  # masked instance -> placeholder row
    ]
    out = build_step_wise_generator_output(trajs, overlong_filtering=False)
    weights = out["step_weights"]
    assert len(weights) == len(out["response_ids"])
    assert weights[0:2] == [0.4, 0.5]
    assert all(math.isnan(w) for w in weights[2:4]), "unscored turns must be NaN (stock-GRPO identity)"
    assert weights[4] == 0.0 and out["loss_masks"][4] == [0], "placeholder rows are inert either way"


def test_short_weight_list_pads_with_nan_never_misaligns():
    trajs = [make_traj("a", 0, n_turns=3, reward=1.0, step_weights=[0.7])]  # scorer returned too few
    out = build_step_wise_generator_output(trajs, overlong_filtering=False)
    assert out["step_weights"][0] == 0.7
    assert all(math.isnan(w) for w in out["step_weights"][1:])


def test_no_scoring_leaves_the_key_absent():
    trajs = [make_traj("a", 0, n_turns=2, reward=1.0)]
    out = build_step_wise_generator_output(trajs, overlong_filtering=False)
    assert "step_weights" not in out


# ---------------------------------------------------------------------------
# State building
# ---------------------------------------------------------------------------


def test_split_turns_recovers_observations_by_token_slice():
    traj = make_traj("a", 0, n_turns=3, reward=1.0)
    turns = split_turns(traj.rollout_details[0], FakeTokenizer())
    assert len(turns) == 3
    assert turns[0].action == "tok100 tok101"
    assert turns[0].observation == "tok200", "observation is the slice of the next prompt"
    assert turns[2].observation == "", "last turn has no next prompt to diff"


def test_split_turns_falls_back_to_text_diff_when_prefix_breaks():
    traj = make_traj("a", 0, n_turns=2, reward=1.0)
    detail = traj.rollout_details[0]
    detail["prompt_token_ids"][1] = [9, 9, 9] + detail["prompt_token_ids"][1]  # re-templated prompt
    turns = split_turns(detail, FakeTokenizer())
    assert len(turns) == 2
    assert turns[0].observation != "", "fallback still produces observation text"


def test_rl_online_state_is_causal():
    turns = [Turn(action=f"cmd {i}", observation=f"out {i}") for i in range(10)]
    material = {"instruction": "do the task", "tests": "assert x"}
    state = build_state(turns, focal=4, task_material=material, mode=CONTEXT_MODES["rl_online"])
    assert "later_steps" not in state
    assert "total_steps" not in state["trajectory_position"]
    assert "cmd 5" not in str(state) and "cmd 9" not in str(state)
    full = build_state(turns, focal=4, task_material=material, mode=CONTEXT_MODES["window4"])
    assert "later_steps" in full and full["trajectory_position"]["total_steps"] == 10


def test_window4_summarizes_distant_turns():
    turns = [Turn(action=f"command number {i}", observation=f"output {i}") for i in range(12)]
    material = {"instruction": "task", "tests": ""}
    state = build_state(turns, focal=8, task_material=material, mode=CONTEXT_MODES["window4"])
    earlier = state["earlier_steps"]
    assert "output 7" in earlier, "near turns keep their results"
    assert "output 0" not in earlier, "distant turns collapse to one line"
    assert "[step 0]" in earlier, "but are still named"


def test_shrink_scale_reduces_state_size():
    turns = [Turn(action="x" * 40_000, observation="y" * 40_000) for _ in range(3)]
    material = {"instruction": "i" * 40_000, "tests": "t" * 40_000}
    big = len(str(build_state(turns, 1, material, CONTEXT_MODES["window4"], scale=1.0)))
    small = len(str(build_state(turns, 1, material, CONTEXT_MODES["window4"], scale=0.25)))
    assert small < big * 0.5


def test_clean_observation_strips_terminal_echo():
    raw = "root@box:/app# cat > f.py\n> import os\n> print(1)\nactual output"
    cleaned = clean_observation(raw)
    assert "import os" not in cleaned and "actual output" in cleaned


def test_middle_truncate_keeps_both_ends():
    text = "A" * 100 + "B" * 100 + "C" * 100
    out = middle_truncate(text, 120)
    assert out.startswith("A") and out.endswith("C") and "omitted" in out


# ---------------------------------------------------------------------------
# Scorer behavior
# ---------------------------------------------------------------------------


def test_weight_is_the_signed_readout(tmp_path):
    scorer = make_scorer(tmp_path, client=FakeClient({"0": 0.25, "1": 0.15, "2": 0.60}))
    traj = make_traj("a", 0, n_turns=2, reward=1.0)
    weights = asyncio.run(scorer.score_trajectory(traj.rollout_details[0], task_path=str(tmp_path)))
    assert weights == pytest.approx([0.35, 0.35])  # P(pos) - P(det)


def test_api_failure_yields_nan_not_crash(tmp_path):
    scorer = make_scorer(tmp_path, client=FakeClient(error=RuntimeError("boom 500")))
    traj = make_traj("a", 0, n_turns=3, reward=1.0)
    weights = asyncio.run(scorer.score_trajectory(traj.rollout_details[0], task_path=str(tmp_path)))
    assert len(weights) == 3 and all(math.isnan(w) for w in weights)


def test_circuit_breaker_opens_and_stops_calling(tmp_path):
    client = FakeClient(error=RuntimeError("timeout"))
    scorer = make_scorer(tmp_path, client=client, circuit_breaker_failures=2)
    traj = make_traj("a", 0, n_turns=6, reward=1.0)
    asyncio.run(scorer.score_trajectory(traj.rollout_details[0], task_path=str(tmp_path)))
    calls_after_first = client.calls
    asyncio.run(scorer.score_trajectory(traj.rollout_details[0], task_path=str(tmp_path)))
    assert scorer.breaker.open
    assert client.calls == calls_after_first, "an open breaker makes no further API calls"


def test_402_opens_the_breaker_immediately(tmp_path):
    scorer = make_scorer(tmp_path, client=FakeClient(error=RuntimeError("402 no credits")))
    traj = make_traj("a", 0, n_turns=1, reward=1.0)
    asyncio.run(scorer.score_trajectory(traj.rollout_details[0], task_path=str(tmp_path)))
    assert scorer.breaker.open and "402" in scorer.breaker.reason


def test_cache_round_trip_skips_the_api(tmp_path):
    client = FakeClient()
    scorer = make_scorer(tmp_path, client=client)
    traj = make_traj("a", 0, n_turns=2, reward=1.0)
    first = asyncio.run(scorer.score_trajectory(traj.rollout_details[0], task_path=str(tmp_path)))
    calls = client.calls
    second = asyncio.run(scorer.score_trajectory(traj.rollout_details[0], task_path=str(tmp_path)))
    assert second == first and client.calls == calls, "identical states must be served from cache"


def test_collect_deadline_returns_empty_for_late_tasks(tmp_path):
    scorer = make_scorer(tmp_path, client=FakeClient(delay=5.0), collect_deadline_s=0.2)
    traj = make_traj("a", 0, n_turns=2, reward=1.0)

    async def run():
        task = asyncio.create_task(scorer.score_trajectory(traj.rollout_details[0], task_path=str(tmp_path)))
        return await scorer.collect({0: task})

    out = asyncio.run(run())
    assert out[0] == [], "late scoring falls back to unweighted rows, never blocks"


def test_metrics_report_icc_and_fallbacks(tmp_path):
    scorer = make_scorer(tmp_path)
    traj = make_traj("a", 0, n_turns=3, reward=1.0)
    weights = asyncio.run(scorer.score_trajectory(traj.rollout_details[0], task_path=str(tmp_path)))
    metrics = scorer.pop_metrics(weights_by_trajectory=[weights])
    assert metrics["jev/scored_steps"] == 3.0
    assert metrics["jev/label_positive"] == 1.0
    assert "jev/latency_p50_s" in metrics


def test_icc_flags_trajectory_level_collapse():
    per_step = [[0.9, -0.5, 0.1, 0.7], [-0.2, 0.8, -0.6, 0.3]]
    constant = [[0.9, 0.9, 0.9, 0.9], [-0.5, -0.5, -0.5, -0.5]]
    assert _step_specificity_icc(constant) > 0.95
    assert _step_specificity_icc(per_step) < 0.5
    assert math.isnan(_step_specificity_icc([[float("nan")] * 3]))


def test_jev_weights_with_merged_output_is_rejected_loudly():
    """The yaml footgun: merge_stepwise_output=true collapses the rows the weights align to."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from examples.train_integrations.harbor.harbor_generator import HarborGenerator

    generator_cfg = SimpleNamespace(
        inference_engine=SimpleNamespace(served_model_name="m"),
        step_wise_trajectories=True,
        merge_stepwise_output=True,
        rate_limit=None,
        jev_weights=JevWeightsConfig(enabled=True),
        use_cache_salt=False,
    )
    client = MagicMock()
    client.get_endpoint_url.return_value = "http://localhost:8000"
    with pytest.raises(ValueError, match="merge_stepwise_output=false"):
        HarborGenerator(
            generator_cfg=generator_cfg,
            harbor_cfg={"agent": {"kwargs": {}}},
            inference_engine_client=client,
            tokenizer=FakeTokenizer(),
            max_seq_len=1024,
        )
