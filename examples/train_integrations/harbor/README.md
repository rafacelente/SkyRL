## Harbor Integration

RL training with [Harbor](https://github.com/laude-institute/harbor) as the environment and reward source. See the [full documentation](https://docs.skyrl.ai/docs/harbor) for details.

### Structure

```
examples/train_integrations/harbor/
  harbor_generator.py              # HarborGenerator: bridges SkyRL <-> Harbor
  dataset.py                       # HarborTaskDataset: loads task directory paths
  prepare_harbor_dataset.py        # Downloads + extracts datasets from HuggingFace
  harbor_trial_config/
    default.yaml                   # Harbor TrialConfig template
  entrypoints/
    main_harbor.py                 # Full training entrypoint
    main_harbor_generate.py        # Generation-only debug entrypoint
  run_codecontest.sh               # Code contest training (Qwen3-8B)
  run_harbor_gen.sh                # Debug generation-only
```

### Quick Start

```bash
cd SkyRL

# 1. Set credentials
export WANDB_API_KEY=your_wandb_api_key
# Pick your sandbox provider:
export DAYTONA_API_KEY=your_daytona_api_key
# export MODAL_TOKEN_ID=your_modal_token_id
# export MODAL_TOKEN_SECRET=your_modal_token_secret

# 2. Prepare dataset
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/CodeContests
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/OpenThoughts-TB-dev

# 3. Launch training
bash examples/train_integrations/harbor/run_codecontest.sh
```


## Jev step weights (optional)

Per-step credit weights from the Jev classifier, consumed by the step-weighted GRPO trainer
(`examples/train/step_wise/mock_grpo_step.py` documents the contract: one weight in [-1, 1] per
flattened step row, `adv = |traj_adv| * w`, NaN = stock GRPO for that row).

```bash
export TYPESAFE_API_KEY=...   # required when enabled

... entrypoints.main_harbor \
  generator.jev_weights.enabled=true \
  generator.jev_weights.context=window4 \        # window4 | full | rl_online
  generator.jev_weights.model=jev-latest \         # only jev-latest / jev-preview exist; release date logged at startup
  generator.jev_weights.cache_dir=~/.cache/jev \
  generator.jev_weights.dump_dir=~/jev_dumps     # (state, probs) audit JSONL every N batches
```

How it behaves:

- Scoring runs per trajectory at completion, as background asyncio tasks inside `generate()`,
  so it overlaps generation of slower trajectories. A batch deadline
  (`jev_weights.collect_deadline_s`) bounds the wait after the last trajectory.
- Every failure path (timeout, API error, deadline, open circuit breaker, 402) emits NaN, and the
  trainer maps NaN to exactly stock GRPO for those rows. The scorer being down never blocks or
  corrupts a run.
- Watch `jev/icc` in the rollout metrics: near 1.0 means the scorer has collapsed to
  trajectory-level judgment (its weights just re-derive the outcome). Healthy is ~0.36. Also
  `jev/fallback_rate`, `jev/latency_p95_s`, and the label split.
- Eval batches are never scored.

Design, offline validation (rho 0.625 vs reference labels, calibration, throughput math):
`post-training/jev-like-plr/docs/skyrl-jev-integration.md`. The rubric in `jev_weights/questions.py`
is vendored from that repo — change it there, re-validate, then re-vendor.
