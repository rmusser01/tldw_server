# MCTS events after deferred transport initialization

Tracking: TASK-13263.2. Carried by sync PR #2971 and release candidate #2972.
Published 0.1.42 remains unchanged.

Mixed collection of Media tests (which import the application) and Prompt Studio
reproduced missing optimization lifecycle events. A boundary probe confirmed the
optimizer's cached manager was `None` while the live endpoint manager existed.
The optional import had encountered a startup cycle and was never retried.
Isolated MCTS modules passed, demonstrating the import-order dependency.

The optimizer retries the existing optional import when an identified optimization
reaches broadcasting. It retains an existing manager and preserves nonfatal behavior
when the transport remains unavailable. The recovered global also supplies the
engine's existing post-persistence completion broadcaster.

The deterministic deferred-manager regression fails before the change. Afterward:

- 69 Media/MCTS integration and unit tests pass.
- The original mixed collection with seed 328249554 passes all 11 selected event
  and unavailable-transport cases, with 1,209 other cases deselected.
- Scoped Bandit: 1,160 baseline / 1,152 current, no new findings. Ruff: 65 / 35,
  no new diagnostics. The two new pytest assertions use the existing B101 convention.
- Independent read-only review found no actionable issues: recovery, cached-manager
  behavior, engine completion, absent transport and monkeypatch restoration checked.

A broader 1,214-case run was interrupted after these missing-event failures for
root-cause investigation. These focused checks do not claim the entire broader
suite passed. Existing event/persistence assertions were retained, and an added
no-transport case confirms optimization still completes and persists its results.

Final sync validation: 161 directly affected Prompt Studio tests and 19 workflow/
collection contracts both exit successfully with publication-matching pytest
plugins. An earlier combined 180-case run reported all cases passed but needed
interruption during session cleanup; it is not claimed as a clean-exit pass.
The candidate separately passes 82 Media/MCTS/licensing cases, including all 13
protected-source checks, with Bandit21/21 and Ruff0/0.
