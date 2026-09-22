"""Building the per-step state Jev scores, from Harbor rollout details.

Text recovery works on token ids alone. ``rollout_details`` gives, per turn t, the full prompt
``prompt_token_ids[t]`` and the completion ``completion_token_ids[t]``. Token-in-token-out means

    prompt[t + 1] == prompt[t] + completion[t] + observation[t]

so each turn's observation is a token slice of the next turn's prompt. The property is asserted per
trajectory; where a re-templated prompt breaks it, we fall back to decoding whole prompts and
diffing text — degraded, never silently wrong.

State shaping is vendored from jev-like-plr (``src/jev_like_plr/context.py`` and ``atif/render.py``)
with char budgets in place of token counts; Jev-side overflow is handled by the scorer's
shrink-retry. Offline validation for the three context modes lives in that repo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# One terminal observation line echoing back what the agent just typed ("> " continuations,
# sometimes with a stray redraw character in front). Dropping the echo is lossless for judging
# the step and roughly halves large tmux dumps.
_ECHO_LINE = re.compile(r"^.{0,2}>\s")
_BLANK_RUN = re.compile(r"\n{3,}")

TRUNCATION_MARK = "\n[... {dropped} chars omitted ...]\n"

# Char caps per section (~4 chars/token). Sized so a window4 state lands near the 7.2k tokens
# validated offline, comfortably under Jev's 32k state limit even with tokenizer slack.
INSTRUCTION_CAP = 24_000
TESTS_CAP = 24_000
FOCAL_ACTION_CAP = 16_000
FOCAL_RESULT_CAP = 20_000
CONTEXT_ACTION_CAP = 2_400
CONTEXT_RESULT_CAP = 2_400
ONELINE_CAP = 360

TASK_FILE_SUFFIXES = (".py", ".sh", ".txt", ".toml", ".json", ".yaml", ".yml")


@dataclass(frozen=True)
class ContextMode:
    """Which turns the scorer sees, and whether it may know how long the trajectory ran."""

    before: Optional[int]  # detailed turns before the focal one; None = all
    after: Optional[int]  # detailed turns after; 0 = strictly causal; None = all
    disclose_total_steps: bool


CONTEXT_MODES: Dict[str, ContextMode] = {
    "window4": ContextMode(before=4, after=4, disclose_total_steps=True),
    "full": ContextMode(before=None, after=None, disclose_total_steps=True),
    # Causal: no future turns, and no trajectory length — length alone predicts the verifier
    # outcome (AUC 0.63 offline), so disclosing it leaks the future.
    "rl_online": ContextMode(before=4, after=0, disclose_total_steps=False),
}


@dataclass
class Turn:
    """One decoded agent turn: what it emitted and what the environment answered."""

    action: str
    observation: str


def middle_truncate(text: str, limit: int) -> str:
    """Keep head and tail of a long block; both ends carry signal, the middle rarely does."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    dropped = len(text) - head - tail
    return text[:head] + TRUNCATION_MARK.format(dropped=dropped) + text[len(text) - tail :]


def clean_observation(text: str) -> str:
    """Drop the terminal's echo of what the agent typed, collapse blank runs."""
    if not text:
        return ""
    kept = [line for line in text.split("\n") if not _ECHO_LINE.match(line)]
    return _BLANK_RUN.sub("\n\n", "\n".join(kept)).strip()


def split_turns(rollout_detail: Dict[str, Any], tokenizer) -> List[Turn]:
    """Decode per-turn (action, observation) text from a Harbor rollout detail."""
    prompts: List[List[int]] = rollout_detail["prompt_token_ids"]
    completions: List[List[int]] = rollout_detail["completion_token_ids"]
    n = len(completions)
    turns: List[Turn] = []
    for t in range(n):
        action = tokenizer.decode(completions[t], skip_special_tokens=True)
        observation = ""
        if t + 1 < n:
            boundary = len(prompts[t]) + len(completions[t])
            next_prompt = prompts[t + 1]
            if next_prompt[: len(prompts[t])] == prompts[t] and len(next_prompt) >= boundary:
                observation = tokenizer.decode(next_prompt[boundary:], skip_special_tokens=True)
            else:
                # Prefix property broken (re-templated prompt): diff decoded text instead.
                whole_next = tokenizer.decode(next_prompt, skip_special_tokens=True)
                whole_here = tokenizer.decode(prompts[t] + completions[t], skip_special_tokens=True)
                observation = whole_next[len(whole_here) :] if whole_next.startswith(whole_here) else whole_next
        turns.append(Turn(action=action.strip(), observation=clean_observation(observation)))
    return turns


@lru_cache(maxsize=256)
def load_task_material(task_path: str) -> Dict[str, str]:
    """Instruction and test text from a Harbor task directory (same layout as the offline corpus)."""
    root = Path(task_path)
    instruction = ""
    instruction_path = root / "instruction.md"
    if instruction_path.exists():
        instruction = instruction_path.read_text(encoding="utf-8", errors="replace").strip()
    tests: List[str] = []
    tests_dir = root / "tests"
    if tests_dir.is_dir():
        for path in sorted(tests_dir.rglob("*")):
            if path.is_file() and path.suffix in TASK_FILE_SUFFIXES:
                body = path.read_text(encoding="utf-8", errors="replace").strip()
                tests.append(f"### {path.relative_to(tests_dir)}\n{body}")
    return {
        "instruction": middle_truncate(instruction, INSTRUCTION_CAP),
        "tests": middle_truncate("\n\n".join(tests), TESTS_CAP),
    }


def _render_turn(index: int, turn: Turn, action_cap: int, result_cap: int) -> str:
    parts = [f"[step {index}] agent turn"]
    if turn.action:
        parts.append("action:\n" + middle_truncate(turn.action, action_cap))
    if turn.observation and result_cap > 0:
        parts.append("result:\n" + middle_truncate(turn.observation, result_cap))
    if len(parts) == 1:
        parts.append("(no action or result recorded)")
    return "\n".join(parts)


def _render_oneline(index: int, turn: Turn) -> str:
    gist = re.sub(r"\s+", " ", turn.action).strip()[:ONELINE_CAP] or "(empty turn)"
    return f"[step {index}] {gist}"


def build_state(
    turns: List[Turn],
    focal: int,
    task_material: Dict[str, str],
    mode: ContextMode,
    scale: float = 1.0,
) -> Dict[str, Any]:
    """The state object for scoring turn ``focal``.

    ``scale`` shrinks every char cap; the scorer walks it down (1.0 -> 0.5 -> 0.25) when the API
    rejects a state as over its token limit, so an oversized step degrades instead of failing.
    """

    def cap(value: int) -> int:
        return max(200, int(value * scale))

    def section(indices: List[int], window: Optional[int], keep_tail: bool) -> str:
        if not indices:
            return ""
        near = indices if window is None else (indices[-window:] if keep_tail else indices[:window])
        near_set = set(near)
        rendered = [
            (
                _render_turn(i, turns[i], cap(CONTEXT_ACTION_CAP), cap(CONTEXT_RESULT_CAP))
                if i in near_set
                else _render_oneline(i, turns[i])
            )
            for i in indices
        ]
        return "\n\n".join(rendered)

    position: Dict[str, Any] = {"step_under_review": focal}
    if mode.disclose_total_steps:
        position["total_steps"] = len(turns)

    state: Dict[str, Any] = {
        "task_instruction": middle_truncate(task_material["instruction"], cap(INSTRUCTION_CAP)),
        "trajectory_position": position,
    }
    if task_material["tests"]:
        state["verification_tests"] = middle_truncate(task_material["tests"], cap(TESTS_CAP))

    before = section(list(range(focal)), mode.before, keep_tail=True)
    if before:
        state["earlier_steps"] = before
    state["step_under_review"] = _render_turn(focal, turns[focal], cap(FOCAL_ACTION_CAP), cap(FOCAL_RESULT_CAP))
    if mode.after != 0:
        after = section(list(range(focal + 1, len(turns))), mode.after, keep_tail=False)
        if after:
            state["later_steps"] = after
    return state
