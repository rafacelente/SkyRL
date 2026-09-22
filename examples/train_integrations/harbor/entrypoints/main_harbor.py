"""
Main entrypoint for training on Harbor tasks.
"""

import sys

import ray
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.config import SkyRLTrainConfig, GeneratorConfig, get_config_as_yaml_str
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl.train.utils.rate_limiter import RateLimiterConfig
from ..harbor_generator import HarborGenerator
from ..dataset import HarborTaskDataset
from ..jev_weights import JevWeightsConfig
from ..step_weighted_trainer import StepWeightedGRPOTrainer

# NOTE (sumanthrh): We use a YAML to store the defaults for the Harbor trial configuration
# TODO: Convert to a dataclass
HARBOR_DEFAULT_CONFIG = Path(__file__).parent.parent / "harbor_trial_config" / "default.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class HarborGeneratorConfig(GeneratorConfig):
    """GeneratorConfig with Harbor-specific rate limiting and optional Jev step weights."""

    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)
    # Per-step credit weights from Jev (generator.jev_weights.enabled=true). See
    # examples/train_integrations/harbor/jev_weights/ and the StepWeightedGRPOTrainer contract.
    jev_weights: JevWeightsConfig = field(default_factory=JevWeightsConfig)


@dataclass
class HarborSkyRLConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig with Harbor trial configuration."""

    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)


class HarborExp(BasePPOExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        """Stock trainer unless Jev step weights are enabled, then the step-weighted one.

        The weighted trainer is a strict superset: with no ``step_weights`` key in the generator
        output it behaves exactly like ``RayPPOTrainer``, so gating on the same flag that turns
        scoring on keeps one switch for the whole feature.
        """
        if not cfg.generator.jev_weights.enabled:
            return super().get_trainer(
                cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg
            )
        return StepWeightedGRPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the HarborGenerator.
        """
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,  # Pass harbor config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            HarborTaskDataset: The training dataset.
        """
        prompts_dataset = HarborTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = HarborExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    # Load harbor defaults and merge CLI overrides on top
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
