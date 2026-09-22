"""The scoring question, vendored verbatim from jev-like-plr (rubric v2).

Source of truth and offline validation: post-training/jev-like-plr,
``src/jev_like_plr/scoring/questions.py`` (_CONTRIBUTION_CRITERIA_V2). Measured there against
Claude Opus 5 reference labels on 2,023 terminal-bench steps: rho +0.625, accuracy 0.714,
ECE 0.017 under the window4 context. Change it there first, re-validate, then re-vendor.

The question is a plain dict (the TypeSafe SDK accepts dict questions), so this module imports
without the SDK installed.
"""

SHARED_FRAME = (
    "`step_under_review` is one step of an agent's attempt at the task in `task_instruction`. "
    "Judge only that step. The attempt as a whole may have succeeded or failed; that outcome does "
    "not decide this step's grade. A step in a failed attempt can still be a good step, and a step "
    "in a successful attempt can still be wasted or harmful."
)

CONTRIBUTION_LEVELS = ("detrimental", "neutral", "positive")

CONTRIBUTION_QUESTION = {
    "type": "score",
    "instructions": (
        f"{SHARED_FRAME} Rate what `step_under_review` did to the attempt's position. Default to "
        "the middle level. Pick the top level only when you can name what the step changed, and "
        "the bottom level only when you can name what it cost."
    ),
    "criteria": [
        {
            "level": "set the attempt back",
            "definition": (
                "After this step the attempt is worse off than before it. The step broke, deleted or "
                "overwrote something that was already correct; or it committed to an approach that "
                "earlier results in this trajectory had already shown does not work; or it drew a wrong "
                "conclusion from a result and acted on it; or it spent the attempt's remaining effort on "
                "work that cannot make `verification_tests` pass."
            ),
            "examples": [
                "overwrites a working file with a version that no longer runs",
                "re-runs a command that already failed, without changing anything about it",
                "reads an error message as success and moves on",
                "starts rewriting a component the task never asked about",
            ],
        },
        {
            "level": "left the attempt where it was (the default)",
            "definition": (
                "Choose this level unless the step clearly belongs in one of the other two. After this "
                "step the attempt is neither closer to nor further from passing `verification_tests`. "
                "This includes a step whose command failed or was rejected by the environment, unless "
                "the failure itself told the agent something it then used. It also includes a step that "
                "re-gathered information already visible earlier, restated a plan without acting, "
                "produced output no later step used, or did routine bookkeeping."
            ),
            "examples": [
                "an install command that errors out and changes nothing",
                "prints a file the agent had already read",
                "describes what it intends to do next without running anything",
                "writes a file that a later step discards or overwrites entirely",
                "checks a version number that does not affect the approach",
            ],
        },
        {
            "level": "moved the attempt forward",
            "definition": (
                "After this step the attempt is measurably closer to passing `verification_tests`, and "
                "you can point to what changed. The step wrote or corrected part of the solution that "
                "survives into the final state; or it produced information a later step in the "
                "trajectory demonstrably used; or it fixed a real defect; or it ruled out a wrong "
                "approach on evidence. A step that only attempted these things, or whose command failed, "
                "is not this level."
            ),
            "examples": [
                "writes the function the tests exercise, correctly",
                "finds the actual cause of a failing test",
                "inspects the input data and gets the format the solution depends on",
                "establishes that a candidate approach cannot work, on evidence",
            ],
        },
    ],
}
