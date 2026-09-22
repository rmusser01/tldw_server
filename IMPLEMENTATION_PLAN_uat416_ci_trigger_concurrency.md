# UAT416 — Preserve PR CI when license follow-up admission skips

Tasks: TASK-13260.278.14 and TASK-13260.278.15 (UAT417). PR: https://github.com/rmusser01/tldw_server/pull/2979.

## Cause and repair design

Direct pull_request and inert workflow_run events share one per-PR cancellation group in 27 workflows. GitHub cancels the active PR run before evaluating the later run's job admission. PR2979 frontend-required run35690741339 explicitly reports that collision; follow-up runs35690768072/35690813461 are entirely skipped. Include the trigger event in the existing group. Newer runs within the same trigger family still supersede older ones; skipped or non-admitted trigger families cannot cancel other families. Keep admission, permissions, checkouts and the separate license-first rollout unchanged. Default-branch workflow definitions require their normal rollout, but the PR's event-qualified group also separates it from the current default branch's legacy group.

## Related UAT417 registration gap

Broader validation exposed one independent failure: the existing exhaustive shard contract finds test_character_chat_images.py absent from all five Character integration matrices. Add that file once to each existing chat-character-integration-chat shard, then rerun the unchanged exhaustive contract and the actual SQLite/PostgreSQL image suite. No new shard or relaxed assertion.

## Stage 1: Causal contract
**Goal**: Capture the required trigger isolation in existing workflow contracts.
**Success Criteria**: The concurrency assertion fails on the current workflows.
**Tests**: Selected license-first workflow context/concurrency contract.
**Status**: Complete

## Stage 2: Shared configuration repair
**Goal**: Add event identity to all 27 affected workflow groups.
**Success Criteria**: Contract passes; other admission and trust contracts stay intact.
**Tests**: Focused CI workflow/admission suites, actionlint and Bandit on touched Python tests (existing assertion findings distinguished).
**Status**: Complete

## Stage 3: Review and remote confirmation
**Goal**: Push the repair to PR2979 and verify active PR jobs survive license-audit completion.
**Success Criteria**: No replacement-event cancellation of the new head; actual CI failures remain visible and are addressed individually.
**Tests**: PR check/run metadata and annotations, diff review.
**Status**: In Progress

Verification: original concurrency contract1failure; expanded scope initially145pass/1missing Character image registration. After both repairs146pass, actionlint0, Bandit168unchanged/0new. Actual96-case image execution and remote cancellation proof remain in progress.
