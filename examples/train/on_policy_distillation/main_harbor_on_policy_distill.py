"""
On-policy distillation on Harbor tasks.

Composes the Harbor training integration (rollouts through HarborGenerator /
HarborTaskDataset in Harbor environments) with the on-policy-distillation
trainer from main_on_policy_distill.py: the teacher is loaded as the ref model
(`trainer.ref.model.path`) and per-token rewards are the reverse KL between
student and teacher on the student's own rollouts. The verifier reward is
intentionally discarded (apply_reward_kl_penalty overwrites `rewards`).

Usage mirrors main_harbor.py (data.train_data = Harbor dataset dirs,
harbor_trial_config.* overrides), plus the OPD flags:

    trainer.ref.model.path=<teacher, same tokenizer as the student>
    trainer.algorithm.advantage_estimator=no_op
    trainer.algorithm.policy_loss_type=importance_sampling
    trainer.algorithm.use_kl_in_reward=true
    trainer.algorithm.use_kl_loss=false
"""

import sys

import ray
import torch
import yaml
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
)
from skyrl.train.entrypoints.main_base import validate_cfg
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray

from examples.train_integrations.harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborExp,
    HarborSkyRLConfig,
    _deep_merge,
)


class OnPolicyDistillationTrainer(RayPPOTrainer):
    """
    Custom trainer for On Policy Distillation.

    Overrides the apply_reward_kl_penalty method to set the rewards just to the kl penalty
    """

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Computes the KL penalty and sets the rewards to the KL penalty"""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        teacher_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        # set rewards to the KL penalty
        # note: tinker seems to use k1 or k2: https://github.com/thinking-machines-lab/tinker-cookbook/blob/3dd0463472dda5847efee80010b50514fa3068ef/tinker_cookbook/rl/metrics.py#L40
        rewards = -(action_log_probs - teacher_action_log_probs) * loss_masks_all
        data["rewards"] = rewards
        return data


# Using the decorator
@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    # just pass through the rewards
    return token_level_rewards, token_level_rewards


class HarborOnPolicyDistillationExp(HarborExp):
    """HarborExp (generator + datasets) with the OPD trainer."""

    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: HarborSkyRLConfig):
    # make sure that the training loop is not run on the head node.
    exp = HarborOnPolicyDistillationExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Harbor training; "
            "it is required to truncate responses to the maximum allowed length."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
