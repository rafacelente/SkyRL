"""
CLI entrypoint for value-model (token-level binary classifier) training.

Parses CLI arguments, validates the config, initializes Ray on the head
node, then dispatches training to a remote Ray task.

Usage::

    python -m skyrl.train.main_value_model strategy=fsdp model.path=Qwen/Qwen3-0.6B \\
        train_datasets="['/path/to/data.jsonl']" train_dataset_splits="['train']"
    python -m skyrl.train.main_value_model strategy=megatron language_model_only=true \\
        freeze_modules="['self_attention','linear_qkv']" ...
"""

import sys

import ray
from loguru import logger

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.sft_config import (
    SFTConfig,
    build_skyrl_config_for_sft,
    validate_sft_cfg,
)
from skyrl.train.utils.utils import initialize_ray
from skyrl.train.value_model_trainer import ValueModelTrainer


@ray.remote(num_cpus=1)
def value_model_entrypoint(cfg: SFTConfig, skyrl_cfg: SkyRLTrainConfig):
    """Run value-model training as a Ray task (off the head node)."""
    trainer = ValueModelTrainer(cfg, skyrl_cfg=skyrl_cfg)
    try:
        trainer.setup()
        trainer.train()
        trainer.shutdown()
    except Exception as e:
        if trainer.tracker is not None:
            trainer.tracker.log_exception(e, step=trainer.global_step)
        else:
            logger.error(f"Value-model setup failed before tracker was initialized:\n{e}")
        raise


def main():
    """CLI entrypoint for value-model training."""
    cfg = SFTConfig.from_cli_overrides(sys.argv[1:])
    cfg.value_model_training = True
    validate_sft_cfg(cfg)
    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    skyrl_cfg.trainer.value_model_training = True
    initialize_ray(skyrl_cfg)
    ray.get(value_model_entrypoint.remote(cfg, skyrl_cfg))


if __name__ == "__main__":
    main()
