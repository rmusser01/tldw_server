## Stage 1: Scope And Existing Evidence
**Goal**: Anchor TASK-535 to the current rail and sidepanel proof without duplicating TASK-534.
**Success Criteria**: Existing real-server /chat tests, evidence files, and open residual risks are inspected.
**Tests**: Read-only inspection of the real-server spec, review doc, and evidence artifacts.
**Status**: Complete

## Stage 2: Real-Server Green-Path Regression
**Goal**: Extend the existing /chat real-server suite to prove the remaining deterministic workflow gap: streaming/stop/regenerate alongside provider, model, Web search, and assistant coverage.
**Success Criteria**: A focused test asserts visible streaming controls and a deterministic regenerate affordance after a stopped or completed response, without requiring third-party Web search.
**Tests**: Focused Playwright run for the new/updated real-server chat test.
**Status**: Complete

## Stage 3: Evidence Refresh
**Goal**: Refresh post-rebase screenshots and structured evidence for first-time, desktop, mobile, and conversation states.
**Success Criteria**: Evidence JSON and review artifacts point at current TASK-535 output, with stale caveats narrowed or removed.
**Tests**: JSON parse check and screenshot existence check.
**Status**: Complete

## Stage 4: Documentation And Backlog Closeout
**Goal**: Update the rebaseline review and TASK-535 with verification, skips, and residual follow-ups.
**Success Criteria**: Review doc reflects current proof, TASK-535 AC/DoD are checked, and no unrelated work is staged.
**Tests**: `git diff --check`; focused test commands recorded; Bandit applicability recorded.
**Status**: Complete
