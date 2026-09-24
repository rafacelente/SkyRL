"""Configuration for Jev step weights.

Attached to the generator config as ``generator.jev_weights`` (see ``HarborGeneratorConfig`` in
``entrypoints/main_harbor.py``), so every knob is a hydra CLI override:

    generator.jev_weights.enabled=true generator.jev_weights.context=window4
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class JevWeightsConfig:
    enabled: bool = False

    # What the scorer sees around each step. All three were validated offline
    # (post-training/jev-like-plr, docs/skyrl-jev-integration.md):
    #   window4:  ±4 detailed turns, one-liners beyond, sees the future.  rho 0.625 vs reference.
    #   full:     every turn compact, sees the future.                    rho 0.613, ~2x tokens.
    #   rl_online: 4 turns back, nothing forward, no trajectory length.   rho 0.603, causal.
    context: str = "window4"

    # The API offers only rolling aliases ("jev-latest", "jev-preview") — there are no pinned
    # version ids, so alias drift mid-run cannot be ruled out by config. The scorer logs each
    # alias's release date at startup so every run records which release it actually got. An
    # invalid name here fails every request with 400 "Unknown model" (breaker opens immediately).
    model: str = "jev-latest"

    # ~14 concurrent requests saturates the documented 1,200 req/min account cap.
    concurrency: int = 14

    # Client-side pacing, requests per minute (0 disables). The account policy is 1,200/min, but
    # exceeding it from a datacenter IP is answered by the EDGE (403, no request id), not by a
    # polite 429 — observed live on 2026-09-23. Default sits safely under the policy.
    requests_per_minute: float = 900.0

    # A 403 edge block is transient (observed clearing within minutes). After the breaker opens
    # for a transient cause it re-probes after this cooldown; 402 and unknown-model stay open
    # for the rest of the run. 0 disables re-probing.
    breaker_cooldown_s: float = 300.0
    request_timeout_s: float = 60.0
    max_retries: int = 3

    # How long generate() waits for outstanding scores after the last trajectory finishes.
    # Late scores become NaN, which the trainer maps to stock GRPO for those rows. Sized with
    # pacing in mind: at requests_per_minute=900 a burst of ~2.5k step scores needs ~170s, and
    # trials themselves take minutes, so a generous deadline costs little and saves coverage.
    collect_deadline_s: float = 600.0

    # After this many consecutive API failures (or any 402) the breaker opens: no further
    # calls this run, every weight NaN, training degrades to plain GRPO.
    circuit_breaker_failures: int = 25

    # Content-addressed response cache. Cheap insurance for crash-restart idempotency;
    # hit rate during normal training is low since states rarely repeat. None disables.
    cache_dir: Optional[str] = None

    # Every N batches, dump (state, probabilities, trajectory_id, step) rows as JSONL —
    # the audit trail and the future critic-distillation set. None disables.
    dump_dir: Optional[str] = None
    dump_every_n_batches: int = 10
