"""Step-weighted GRPO trainer: stock last-step GRPO, then per-turn multiplicative weights.

Contract (walkthrough with printed tensors: ``examples/train/step_wise/mock_grpo_step.py``):

- ``generator_output["step_weights"]``: one float per emitted step row, in [-1, 1].
- ``adv_row = |traj_adv| * w_row`` — a positive weight always pushes the turn up, a negative one
  always pushes it down, regardless of whether the trajectory beat its group.
- ``NaN`` weight = "the scorer had no opinion": the row trains as stock GRPO, bit-exactly.

Requires ``generator.step_wise_trajectories=true`` and ``generator.merge_stepwise_output=false``
(one row per turn; the prefix-aware merge cannot carry a per-row scalar).
"""

from __future__ import annotations

from typing import List, Optional

import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.trainer import RayPPOTrainer

STEP_WEIGHTS_KEY = "step_weights"


class StepWeightedGRPOTrainer(RayPPOTrainer):
    """Step-wise GRPO with a per-turn multiplicative weight on the broadcast advantage.

    Expects an optional ``generator_output["step_weights"]``: one float per emitted
    step (row), in ``[-1, 1]``. Positive means the turn helped, negative means it hurt.

    Pipeline:
      1. ``convert_to_training_input`` carries the weights into the batch as a
         ``[batch]`` tensor (extra GeneratorOutput keys are otherwise dropped there).
      2. ``compute_advantages_and_returns`` runs the stock last-step GRPO + broadcast,
         then multiplies each row by ``w_row * sign(traj_adv)``. Flipping the sign for
         trajectories with negative GRPO advantage makes "good step" always mean "push
         up" and "bad step" always mean "push down".
      3. Weighted-advantage metrics are logged alongside the stock ones.
    """

    def convert_to_training_input(self, generator_output: GeneratorOutput, uids: List[str]) -> TrainingInputBatch:
        step_weights: Optional[List[float]] = generator_output.get(STEP_WEIGHTS_KEY)
        batch = super().convert_to_training_input(generator_output, uids)
        if step_weights is None:
            return batch
        assert len(step_weights) == len(generator_output["response_ids"]), (
            f"{STEP_WEIGHTS_KEY} must have one entry per step, got {len(step_weights)} "
            f"for {len(generator_output['response_ids'])} steps"
        )
        w = torch.tensor(step_weights, dtype=torch.float32)
        pad_size = batch.metadata.get("pad_size", 0)
        if pad_size:
            # Padded rows are loss-masked; a zero weight keeps them inert either way.
            w = torch.cat([w, torch.zeros(pad_size, dtype=w.dtype)])
        batch[STEP_WEIGHTS_KEY] = w
        return batch

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        data = super().compute_advantages_and_returns(data)
        w = data.get(STEP_WEIGHTS_KEY)
        if w is None:
            return data

        adv = data["advantages"]
        resp_mask = data["response_mask"].to(adv.dtype)

        # Stock step-wise GRPO paints one scalar per trajectory onto every response token of
        # every row of that trajectory, so the masked row mean recovers that scalar.
        traj_adv = (adv * resp_mask).sum(dim=-1) / resp_mask.sum(dim=-1).clamp(min=1)
        traj_sign = torch.sign(traj_adv)

        # NaN means "the scorer had no opinion" (timeout, API failure, circuit breaker). The
        # identity there is stock GRPO: effective weight sign(traj_adv), so |A| * sign(A) = A and
        # the row trains exactly as if unweighted. Zero would erase the row and +1 would flip
        # failed-trajectory rows positive — both are wrong fallbacks.
        w_typed = w.to(adv.dtype)
        w_typed = torch.where(torch.isnan(w_typed), traj_sign, w_typed)
        effective_w = w_typed * traj_sign
        data["step_weights_effective"] = effective_w
        data["advantages_unweighted"] = adv.clone()
        data["advantages"] = adv * effective_w[:, None]

        # Metrics over real (non-padded) rows, response tokens only.
        pad_size = data.metadata.get("pad_size", 0)
        n = adv.shape[0] - pad_size
        valid = torch.masked_select(data["advantages"][:n], data["response_mask"][:n].bool())
        weighted_metrics = {
            "avg_weighted_advantages": valid.mean().item(),
            "avg_weighted_advantages_abs": valid.abs().mean().item(),
            "avg_step_weight": w[:n].nanmean().item(),
            "avg_step_weight_abs": w[:n].abs().nanmean().item(),
            "step_weight_nan_rate": w[:n].isnan().float().mean().item(),
        }
        data.metadata.setdefault("metrics", {}).update(weighted_metrics)
        self.all_metrics.update({f"policy/{k}": v for k, v in weighted_metrics.items()})
        return data
