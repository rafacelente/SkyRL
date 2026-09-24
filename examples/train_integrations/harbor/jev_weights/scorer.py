"""Async Jev scoring for Harbor trajectories.

One ``score_trajectory`` call per finished trajectory, submitted as a background task by the
generator's rollout worker so scoring overlaps generation of slower trajectories. ``collect``
gathers everything with a deadline. Every failure path yields NaN, which the trainer maps to the
stock-GRPO identity for exactly the affected rows — the scorer can be slow, down, or out of credit
and the run degrades to plain GRPO rather than blocking or corrupting.

The TypeSafe SDK import is lazy: the module (and its tests) work without the dependency, and tests
inject a fake client.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .config import JevWeightsConfig
from .questions import CONTRIBUTION_LEVELS, CONTRIBUTION_QUESTION
from .state import CONTEXT_MODES, build_state, load_task_material, split_turns

NAN = float("nan")
SHRINK_SCALES = (1.0, 0.5, 0.25, 0.1)


class _CircuitBreaker:
    """Opens after N consecutive failures, or immediately on a fatal response.

    Two kinds of open. Permanent (402 out-of-credits, unknown model): nothing heals mid-run, stay
    open. Transient (failure streaks — e.g. a 403 edge block, observed clearing within minutes):
    re-probe after ``cooldown_s``; if the API still refuses, the threshold re-opens it for another
    cooldown, so a dead API costs at most ``threshold`` requests per cooldown window.
    """

    def __init__(self, threshold: int, cooldown_s: float = 300.0):
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.consecutive = 0
        self.open = False
        self.permanent = False
        self.reason = ""
        self.reopens = 0
        self._reprobe_at = 0.0
        self._announced = False

    def is_open(self) -> bool:
        if not self.open:
            return False
        if self.permanent or self.cooldown_s <= 0:
            return True
        if time.monotonic() >= self._reprobe_at:
            self.open = False
            self.consecutive = 0
            self._announced = False
            self.reopens += 1
            logger.warning(f"Jev circuit breaker cooldown elapsed ({self.reason}); probing the API again.")
            return False
        return True

    def record_success(self) -> None:
        self.consecutive = 0

    def record_failure(self, error: Exception) -> None:
        self.consecutive += 1
        text = str(error)
        if "402" in text or "credit" in text.lower():
            self.open, self.permanent, self.reason = True, True, "API returned 402 (out of credits)"
        elif "Unknown model" in text:
            # A config typo fails every request identically; retrying only triggers the API
            # edge protection (403 storms). One loud line beats 25 identical failures.
            self.open, self.permanent, self.reason = True, True, f"invalid jev_weights.model ({text[:120]})"
        elif self.consecutive >= self.threshold:
            self.open, self.reason = True, f"{self.consecutive} consecutive failures"
            self._reprobe_at = time.monotonic() + self.cooldown_s
        if self.open and not self._announced:
            self._announced = True
            recovery = "permanently" if self.permanent else f"re-probing in {self.cooldown_s:.0f}s"
            logger.error(
                f"Jev circuit breaker OPEN ({self.reason}, {recovery}); last error: {text[:300]} — "
                "training continues as plain GRPO meanwhile."
            )


class JevStepScorer:
    """Scores every turn of a trajectory; returns one weight in [-1, 1] (or NaN) per turn."""

    def __init__(self, cfg: JevWeightsConfig, tokenizer, client: Optional[Any] = None):
        if cfg.context not in CONTEXT_MODES:
            raise ValueError(f"jev_weights.context must be one of {sorted(CONTEXT_MODES)}, got {cfg.context!r}")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.mode = CONTEXT_MODES[cfg.context]
        self.breaker = _CircuitBreaker(cfg.circuit_breaker_failures, cooldown_s=cfg.breaker_cooldown_s)
        self._semaphore = asyncio.Semaphore(cfg.concurrency)
        self._pace_interval = 60.0 / cfg.requests_per_minute if cfg.requests_per_minute > 0 else 0.0
        self._pace_lock = asyncio.Lock()
        self._next_slot = 0.0
        self._cache_dir = Path(cfg.cache_dir).expanduser() if cfg.cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._dump_dir = Path(cfg.dump_dir).expanduser() if cfg.dump_dir else None
        if self._dump_dir:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
        self._batch_counter = 0
        self._provenance_logged = False
        self._metrics_reset()
        self._client = client  # tests inject a fake; real client built lazily below

    def _build_client(self):
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy  # deferred: optional dependency

        return AsyncTypeSafeClient(
            model=self.cfg.model,
            retry=RetryPolicy(max_retries=self.cfg.max_retries),
            timeout=self.cfg.request_timeout_s,
        )

    @property
    def client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # ------------------------------------------------------------------ scoring

    def _log_failure(self, error: Exception) -> None:
        """Surface the first few distinct error bodies per batch; identical repeats stay quiet."""
        key = str(error)[:80]
        if key not in self._m["error_kinds"]:
            self._m["error_kinds"].add(key)
            if len(self._m["error_kinds"]) <= 3:
                logger.warning(f"Jev scoring failure ({len(self._m['error_kinds'])}): {str(error)[:400]}")

    async def _log_model_provenance_once(self) -> None:
        """Record which release the model alias resolves to; aliases drift and runs must be attributable."""
        if self._provenance_logged:
            return
        self._provenance_logged = True
        try:
            listing = await self.client.models.list()
            available = {m.name: m.release_date for m in listing.models}
            logger.info(f"Jev models available to this key: {available}; using {self.cfg.model!r}")
            if self.cfg.model not in available:
                logger.error(
                    f"jev_weights.model={self.cfg.model!r} is not offered by the API "
                    f"(valid: {sorted(available)}); every request will 400."
                )
        except Exception as error:  # noqa: BLE001 — provenance is best-effort
            logger.warning(f"could not list Jev models for provenance: {str(error)[:200]}")

    async def score_trajectory(self, rollout_detail: Dict[str, Any], task_path: str) -> List[float]:
        """One weight per turn. Any per-turn failure is NaN; the list length always matches."""
        await self._log_model_provenance_once()
        turns = split_turns(rollout_detail, self.tokenizer)
        if not turns:
            return []
        material = load_task_material(str(task_path))
        results = await asyncio.gather(*(self._score_turn(turns, t, material, task_path) for t in range(len(turns))))
        return list(results)

    async def _pace(self) -> None:
        """Space requests to ``requests_per_minute``. The edge answers over-rate with 403, not 429."""
        if self._pace_interval <= 0:
            return
        async with self._pace_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._pace_interval
        if wait > 0:
            self._m["pace_wait_s"] += wait
            await asyncio.sleep(wait)

    async def _score_turn(self, turns, focal: int, material: Dict[str, str], task_path: str) -> float:
        if self.breaker.is_open():
            self._m["fallbacks"] += 1
            return NAN
        for scale in SHRINK_SCALES:
            state = build_state(turns, focal, material, self.mode, scale=scale)
            cached = self._cache_get(state)
            if cached is not None:
                self._m["cache_hits"] += 1
                return self._weight_from_answers(cached, task_path, focal, state)
            start = time.monotonic()
            try:
                async with self._semaphore:
                    await self._pace()
                    if self.breaker.is_open():
                        self._m["fallbacks"] += 1
                        return NAN
                    response = await self.client.system_one(
                        state=state, questions={"contribution": CONTRIBUTION_QUESTION}
                    )
            except Exception as error:  # noqa: BLE001 — every failure becomes NaN, never a crash
                if "max_tokens" in str(error) and scale != SHRINK_SCALES[-1]:
                    self._m["shrinks"] += 1
                    continue  # state too long for the API's tokenizer: shrink and retry
                self._log_failure(error)
                self.breaker.record_failure(error)
                self._m["errors"] += 1
                self._m["fallbacks"] += 1
                return NAN
            self.breaker.record_success()
            self._m["latencies"].append(time.monotonic() - start)
            payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
            self._cache_put(state, payload)
            return self._weight_from_answers(payload, task_path, focal, state)
        self._m["fallbacks"] += 1
        return NAN

    def _weight_from_answers(self, payload: Dict[str, Any], task_path: str, focal: int, state) -> float:
        answer = (payload.get("answers") or {}).get("contribution") or {}
        raw = answer.get("probabilities") or {}
        probs = {CONTRIBUTION_LEVELS[int(k)]: float(v) for k, v in raw.items() if int(k) < 3}
        if not probs:
            self._m["fallbacks"] += 1
            return NAN
        # The (-1, 0, +1) readout: neutral is exactly silent, uncertainty self-attenuates.
        weight = probs.get("positive", 0.0) - probs.get("detrimental", 0.0)
        self._m["weights"].append(weight)
        self._m["labels"][max(probs, key=probs.get)] += 1
        self._record_dump(task_path, focal, probs, weight)
        return weight

    async def collect(
        self, tasks: Dict[int, "asyncio.Task[List[float]]"], deadline_s: Optional[float] = None
    ) -> Dict[int, List[float]]:
        """Await outstanding scoring tasks; anything late or failed becomes an empty result."""
        if not tasks:
            return {}
        deadline = self.cfg.collect_deadline_s if deadline_s is None else deadline_s
        done, pending = await asyncio.wait(tasks.values(), timeout=deadline)
        for task in pending:
            task.cancel()
        if pending:
            self._m["deadline_cancels"] += len(pending)
            logger.warning(
                f"Jev collect deadline ({deadline:.0f}s) hit; {len(pending)} trajectories fall back "
                "to plain GRPO. If jev/pace_wait_s is high, raise jev_weights.collect_deadline_s or "
                "requests_per_minute."
            )
        out: Dict[int, List[float]] = {}
        for idx, task in tasks.items():
            if task in pending or task.cancelled() or task.exception() is not None:
                out[idx] = []
            else:
                out[idx] = task.result()
        return out

    # ------------------------------------------------------------------ cache & dump

    def _cache_key(self, state: Dict[str, Any]) -> str:
        payload = json.dumps(
            {"model": self.cfg.model, "state": state, "question": CONTRIBUTION_QUESTION},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{self._cache_key(state)}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_put(self, state: Dict[str, Any], payload: Dict[str, Any]) -> None:
        if self._cache_dir is None:
            return
        try:
            (self._cache_dir / f"{self._cache_key(state)}.json").write_text(json.dumps(payload))
        except OSError:
            pass  # a failed cache write is not a scoring failure

    def _record_dump(self, task_path: str, focal: int, probs: Dict[str, float], weight: float) -> None:
        if self._dump_dir is None or self._batch_counter % max(1, self.cfg.dump_every_n_batches) != 0:
            return
        row = {"task": str(task_path), "step": focal, "probs": probs, "weight": weight, "t": time.time()}
        try:
            with (self._dump_dir / f"batch_{self._batch_counter:06d}.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ metrics

    def _metrics_reset(self) -> None:
        self._m: Dict[str, Any] = {
            "weights": [],
            "labels": {level: 0 for level in CONTRIBUTION_LEVELS},
            "latencies": [],
            "cache_hits": 0,
            "errors": 0,
            "fallbacks": 0,
            "deadline_cancels": 0,
            "shrinks": 0,
            "pace_wait_s": 0.0,
            "error_kinds": set(),
        }

    def pop_metrics(self, weights_by_trajectory: Optional[List[List[float]]] = None) -> Dict[str, float]:
        """Per-batch metrics for rollout logging; resets the accumulator and advances the batch."""
        m = self._m
        weights = m["weights"]
        scored = len(weights)
        out: Dict[str, float] = {
            "jev/scored_steps": float(scored),
            "jev/fallback_rate": m["fallbacks"] / max(1, scored + m["fallbacks"]),
            "jev/cache_hit_rate": m["cache_hits"] / max(1, scored),
            "jev/api_errors": float(m["errors"]),
            "jev/deadline_cancels": float(m["deadline_cancels"]),
            "jev/shrink_retries": float(m["shrinks"]),
            "jev/pace_wait_s": float(m["pace_wait_s"]),
            "jev/breaker_reopens": float(self.breaker.reopens),
            "jev/circuit_open": float(self.breaker.is_open()),
        }
        if weights:
            out["jev/weight_mean"] = sum(weights) / scored
            out["jev/weight_abs_mean"] = sum(abs(w) for w in weights) / scored
            for level, count in m["labels"].items():
                out[f"jev/label_{level}"] = count / scored
        if m["latencies"]:
            ordered = sorted(m["latencies"])
            out["jev/latency_p50_s"] = ordered[len(ordered) // 2]
            out["jev/latency_p95_s"] = ordered[int(len(ordered) * 0.95)]
        if weights_by_trajectory:
            icc = _step_specificity_icc(weights_by_trajectory)
            if icc == icc:  # not NaN
                out["jev/icc"] = icc
                # ICC near 1 means the scorer is emitting one number per trajectory — it has
                # collapsed to outcome-guessing and the weights re-derive the reward. Healthy
                # per-step judgment measured offline sits near 0.36.
                if icc > 0.75:
                    logger.warning(f"Jev step weights ICC={icc:.3f} (>0.75): trajectory-level collapse.")
        self._metrics_reset()
        self._batch_counter += 1
        return out


def _step_specificity_icc(groups: List[List[float]]) -> float:
    """Between-trajectory share of weight variance (the offline degeneracy alarm), NaN-safe."""
    cleaned = [[w for w in g if w == w] for g in groups]
    cleaned = [g for g in cleaned if len(g) > 1]
    if not cleaned:
        return NAN
    total = sum(len(g) for g in cleaned)
    grand = sum(sum(g) for g in cleaned) / total
    between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in cleaned) / total
    within = sum(sum((w - sum(g) / len(g)) ** 2 for w in g) for g in cleaned) / total
    denominator = between + within
    return between / denominator if denominator > 0 else NAN


def maybe_build_scorer(generator_cfg, tokenizer) -> Optional[JevStepScorer]:
    """Construct the scorer when ``generator.jev_weights.enabled`` is set; otherwise None."""
    cfg = getattr(generator_cfg, "jev_weights", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return None
    if not isinstance(cfg, JevWeightsConfig):
        cfg = JevWeightsConfig(**dict(cfg))
    logger.info(
        f"Jev step weights enabled: context={cfg.context} model={cfg.model} "
        f"concurrency={cfg.concurrency} deadline={cfg.collect_deadline_s}s"
    )
    return JevStepScorer(cfg, tokenizer)


__all__ = ["JevStepScorer", "maybe_build_scorer", "NAN"]
