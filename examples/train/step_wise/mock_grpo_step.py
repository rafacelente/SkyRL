"""Walk one step-wise GRPO training step with a mocked model (CPU, no Ray/GPU).

Builds a 1-prompt / 2-sample / 4-turn batch, then calls the real trainer methods
so you can see last-token reward placement, right-aligned padding, last-step GRPO,
the advantage broadcast onto every turn, and finally a per-step advantage
weighting layered on top (``StepWeightedGRPOTrainer``).

Step weighting rule (see ``StepWeightedGRPOTrainer.compute_advantages_and_returns``):
    adv_row = grpo_adv_traj * w_row * sign(grpo_adv_traj)  ==  |grpo_adv_traj| * w_row
so a positive step weight always pushes that turn up and a negative one always pushes
it down, regardless of whether the trajectory as a whole beat its group.

Run:
    uv run --isolated --extra dev --extra skyrl-train python examples/train/step_wise/mock_grpo_step.py
"""

from __future__ import annotations

from typing import Dict, List
from unittest.mock import MagicMock

import torch

from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry
from skyrl.train.config import (
    AlgorithmConfig,
    GeneratorConfig,
    InferenceEngineConfig,
    SamplingParams,
    SkyRLTrainConfig,
    TrainerConfig,
)
from skyrl.train.generators.base import GeneratorOutput, TrajectoryID
from skyrl.train.trainer import RayPPOTrainer

# ---------------------------------------------------------------------------
# Trainer subclass: GRPO on the outcome reward, then per-step advantage weights
# ---------------------------------------------------------------------------

from examples.train_integrations.harbor.step_weighted_trainer import (
    STEP_WEIGHTS_KEY,
    StepWeightedGRPOTrainer,
)

# ---------------------------------------------------------------------------
# Mock batch: 1 prompt, 2 GRPO samples, 4 turns each
# ---------------------------------------------------------------------------

OBS_TOKENS = [30, 31, 32]  # environment/tool observation appended after each turn
INITIAL_PROMPT = [10, 11]


def _build_trajectory(
    name: str, repetition_id: int, responses: List[List[int]], step_weights: List[float], reward: float
) -> List[Dict]:
    """Linearly appending multi-turn trajectory: prompt_t = prompt_{t-1} + resp_{t-1} + obs_{t-1}."""
    assert len(responses) == len(step_weights)
    rows = []
    prompt = list(INITIAL_PROMPT)
    n = len(responses)
    for t, (resp, w) in enumerate(zip(responses, step_weights)):
        is_last = t == n - 1
        rows.append(
            {
                "name": f"{name}/turn{t}",
                "prompt": list(prompt),
                "response": list(resp),
                # Outcome reward is only known at task completion -> last turn only.
                "reward": reward if is_last else 0.0,
                "step_weight": w,
                "is_last_step": is_last,
                "tid": TrajectoryID(instance_id="code_task_0", repetition_id=repetition_id),
            }
        )
        if not is_last:
            prompt = prompt + resp + [OBS_TOKENS[t]]
    return rows


# Sample A: task solved (reward 1.0). Turn 0 was harmful, turn 1 neutral, turns 2-3 useful.
# Sample B: task failed (reward 0.0). Turn 0 and 2 were reasonable, turns 1 and 3 hurt.
#
# GRPO (no std): last-step scores {A: 1.0, B: 0.0}, group mean 0.5
#   adv_A = +0.5, adv_B = -0.5
# Step weighting (|adv| * w):
#   A rows: 0.5 * [-0.5, 0.0, 0.2, 0.9] = [-0.25, 0.00, 0.10, 0.45]
#   B rows: 0.5 * [ 0.7,-0.3, 0.1,-0.8] = [ 0.35,-0.15, 0.05,-0.40]
ROWS: List[Dict] = _build_trajectory(
    "A",
    repetition_id=0,
    responses=[[20, 21, 22], [23, 24], [25, 26, 27, 28], [29]],
    step_weights=[-0.5, 0.0, 0.2, 0.9],
    reward=1.0,
) + _build_trajectory(
    "B",
    repetition_id=1,
    responses=[[40, 41], [42, 43, 44], [45], [46, 47]],
    step_weights=[0.7, -0.3, 0.1, -0.8],
    reward=0.0,
)


def _cfg() -> SkyRLTrainConfig:
    trainer_cfg = TrainerConfig(
        project_name="mock-grpo-step",
        run_name="local",
        logger="tensorboard",
        micro_train_batch_size_per_gpu=2,
        train_batch_size=1,
        eval_batch_size=1,
        policy_mini_batch_size=1,
        update_epochs_per_batch=1,
        epochs=1,
        max_prompt_length=32,
        seed=42,
        resume_mode="none",
        enable_ray_gpu_monitor=False,
        algorithm=AlgorithmConfig(
            advantage_estimator="grpo",
            grpo_norm_by_std=False,
            use_kl_loss=False,
            use_kl_in_reward=False,
            advantage_batch_normalize=False,
            loss_reduction="token_mean",
            policy_loss_type="regular",
        ),
    )
    generator_cfg = GeneratorConfig(
        sampling_params=SamplingParams(max_generate_length=20),
        n_samples_per_prompt=2,
        batched=False,
        max_turns=4,
        step_wise_trajectories=True,
        merge_stepwise_output=False,  # per-row step weights assume one row per turn
        inference_engine=InferenceEngineConfig(enable_ray_prometheus_stats=False),
    )
    return SkyRLTrainConfig(trainer=trainer_cfg, generator=generator_cfg)


def _trainer(cfg: SkyRLTrainConfig) -> StepWeightedGRPOTrainer:
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    trainer = StepWeightedGRPOTrainer(
        cfg=cfg,
        tracker=None,
        tokenizer=tokenizer,
        train_dataset=None,
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )
    trainer.dispatch = MagicMock()
    trainer.dispatch.get_lcm_dp_size.return_value = 1
    return trainer


def _generator_output() -> GeneratorOutput:
    return {
        "prompt_token_ids": [row["prompt"] for row in ROWS],
        "response_ids": [row["response"] for row in ROWS],
        "rewards": [row["reward"] for row in ROWS],
        "loss_masks": [[1] * len(row["response"]) for row in ROWS],
        "stop_reasons": ["stop"] * len(ROWS),
        "rollout_metrics": None,
        "rollout_logprobs": [[-1.2] * len(row["response"]) for row in ROWS],
        "trajectory_ids": [row["tid"] for row in ROWS],
        "is_last_step": [row["is_last_step"] for row in ROWS],
        # Extra key: survives concatenate/slice helpers, consumed by StepWeightedGRPOTrainer.
        STEP_WEIGHTS_KEY: [row["step_weight"] for row in ROWS],
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _fmt_tensor(t: torch.Tensor, ndigits: int = 2) -> List[str]:
    if t.dtype in (torch.int32, torch.int64, torch.bool):
        return ["[" + ", ".join(str(int(x)) for x in row.tolist()) + "]" for row in t]
    return ["[" + ", ".join(f"{float(x):+.{ndigits}f}" for x in row.tolist()) + "]" for row in t]


def _fmt_vec(t: torch.Tensor, ndigits: int = 2) -> List[str]:
    return [f"{float(x):+.{ndigits}f}" for x in t.tolist()]


def _print_rows(title: str, columns: List[tuple[str, List[str]]]) -> None:
    widths = [max(len(name), *(len(v) for v in vals)) for name, vals in columns]
    print(f"\n=== {title} ===")
    print("  ".join(name.ljust(w) for (name, _), w in zip(columns, widths)))
    n = len(columns[0][1])
    for i in range(n):
        print("  ".join(columns[j][1][i].ljust(widths[j]) for j in range(len(columns))))


def _names() -> List[str]:
    return [row["name"] for row in ROWS]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    trainer = _trainer(cfg)
    names = _names()

    # Same remap the train loop does after generate() when step_wise_trajectories=True.
    generator_output = _generator_output()
    uids = [tid.instance_id for tid in generator_output["trajectory_ids"]]

    print("Mocked one step-wise GRPO step: 1 prompt, 2 samples, 4 turns each.")
    print("No GPU / Ray / real weights — generate() is replaced by a fixed GeneratorOutput.")
    print(f"uids (instance_id per row): {uids}")
    print(f"is_last_step: {generator_output['is_last_step']}")
    print(f"sequence-level rewards (outcome on last turn only): {generator_output['rewards']}")
    print(f"step weights: {generator_output[STEP_WEIGHTS_KEY]}")
    print("Expected GRPO (no std): last-step scores 1.0 and 0.0, mean 0.5 -> adv +0.5 (A) / -0.5 (B)")
    print("Expected weighted: A = 0.5*w_A = [-0.25, 0, 0.10, 0.45]; B = 0.5*w_B = [0.35, -0.15, 0.05, -0.40]")

    # 1. Sequence-level reward -> last token of that row.
    generator_output, uids = trainer.postprocess_generator_output(generator_output, uids)
    _print_rows(
        "1. postprocess_generator_output  (outcome reward on last token of last turn only)",
        [
            ("row", names),
            ("last", [str(x) for x in generator_output["is_last_step"]]),
            ("w", [f"{w:+.1f}" for w in generator_output[STEP_WEIGHTS_KEY]]),
            ("per-token rewards", [str(r) for r in generator_output["rewards"]]),
        ],
    )

    # 2. Pad + right-align response tensors. Step weights ride along as a [batch] tensor.
    training_input = trainer.convert_to_training_input(generator_output, uids)
    _print_rows(
        "2. convert_to_training_input  (left-pad sequences, right-align response tensors)",
        [
            ("row", names),
            ("sequences", _fmt_tensor(training_input["sequences"], 0)),
            ("resp_mask", _fmt_tensor(training_input["response_mask"], 0)),
            ("rewards", _fmt_tensor(training_input["rewards"])),
            ("step_w", _fmt_vec(training_input[STEP_WEIGHTS_KEY], 1)),
        ],
    )
    print(f"response_length (padded): {training_input.metadata['response_length']}")
    print(f"policy_mini_batch_boundaries: {training_input.metadata['policy_mini_batch_boundaries']}")

    # 3. Mocked policy forward: constant old logprobs on the response window.
    def mock_execute_forward_pass(model, data_fwd_pass, key, mini_batch_boundaries=None):
        batch = data_fwd_pass["sequences"].shape[0]
        resp_len = data_fwd_pass.metadata["response_length"]
        print(f"\n[mock {model}] {key} shape=({batch}, {resp_len}), all -1.0")
        return torch.full((batch, resp_len), -1.0)

    trainer._execute_forward_pass = mock_execute_forward_pass
    training_input = trainer.fwd_logprobs_values_reward(training_input)

    # 4. Stock last-step GRPO + broadcast, then per-step weighting (inside the subclass).
    training_input = trainer.compute_advantages_and_returns(training_input)
    last = training_input.metadata["is_last_step"]
    _print_rows(
        "4a. stock step-wise GRPO  (last-step GRPO scalar broadcast x response_mask)",
        [
            ("row", names),
            ("last", [str(x) for x in last]),
            ("resp_mask", _fmt_tensor(training_input["response_mask"], 0)),
            ("adv (unweighted)", _fmt_tensor(training_input["advantages_unweighted"])),
        ],
    )
    _print_rows(
        "4b. step weighting  (adv * w * sign(traj_adv)  ==  |traj_adv| * w)",
        [
            ("row", names),
            ("w", _fmt_vec(training_input[STEP_WEIGHTS_KEY], 1)),
            ("sign(traj_adv)", _fmt_vec(torch.sign(training_input["advantages_unweighted"].sum(-1)), 0)),
            ("w_eff", _fmt_vec(training_input["step_weights_effective"], 1)),
            ("adv (weighted)", _fmt_tensor(training_input["advantages"])),
        ],
    )
    metrics = training_input.metadata["metrics"]
    print("\nmetrics (stock):    ", {k: round(v, 4) for k, v in metrics.items() if "weight" not in k})
    print("metrics (weighted): ", {k: round(v, 4) for k, v in metrics.items() if "weight" in k})
    print("trainer.all_metrics:", {k: round(v, 4) for k, v in trainer.all_metrics.items() if k.startswith("policy/")})

    # 5. token_mean scales advantages by 1 / n_loss_tokens. This is what the loss sees.
    training_input = trainer._normalize_advantages(
        training_input,
        training_input.metadata["policy_mini_batch_boundaries"],
        training_input.metadata.get("policy_prompt_boundaries"),
    )
    n_loss = int(training_input["loss_mask"].sum().item())
    _print_rows(
        f"5. _normalize_advantages  (token_mean: divide by {n_loss} loss tokens)",
        [
            ("row", names),
            ("adv (weighted, norm)", _fmt_tensor(training_input["advantages"], 4)),
        ],
    )

    # 6. Regular PPO clip loss with a fake "new" policy: every turn slightly more likely than
    # under the old policy (ratio > 1). Positive weighted advantage -> reward that, negative -> penalize.
    new_logprobs = training_input["action_log_probs"] + 0.2
    loss_fn = PolicyLossRegistry.get("regular")
    loss, loss_metrics = loss_fn(
        log_probs=new_logprobs,
        old_log_probs=training_input["action_log_probs"],
        advantages=training_input["advantages"],
        config=cfg.trainer.algorithm,
        loss_mask=training_input["loss_mask"],
        rollout_logprobs=training_input["rollout_logprobs"],
    )
    ratio = torch.exp(new_logprobs - training_input["action_log_probs"])
    per_token_surrogate = -ratio * training_input["advantages"] * training_input["loss_mask"]
    _print_rows(
        "6. ppo_policy_loss  (fake new logprobs = old + 0.2 everywhere, ratio ~1.22)",
        [
            ("row", names),
            ("ratio", _fmt_tensor(ratio)),
            ("adv (weighted, norm)", _fmt_tensor(training_input["advantages"], 4)),
            ("per-token -ratio*adv", _fmt_tensor(per_token_surrogate, 4)),
            ("row loss contrib", _fmt_vec(per_token_surrogate.sum(-1), 4)),
        ],
    )
    print(f"policy_loss={float(loss):.6f}  metrics={loss_metrics}")
    print(
        "\nDone. Rows with negative weighted advantage (A/turn0, B/turn1, B/turn3) contribute positive loss "
        "under ratio>1, i.e. the update pushes those turns down; positively weighted turns are pushed up, "
        "regardless of which trajectory won the group."
    )


if __name__ == "__main__":
    main()
