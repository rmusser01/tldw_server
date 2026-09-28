# TASK-13013.7.47 — PR 2869 integration

## Stage 1: Integrate latest dev
**Goal**: Preserve recovery refs and rebase onto latest fetched dev; now 3c9d97c56b29abc4c0396274b9560859aee06959 after the September 27 update.
**Success Criteria**: No unresolved conflicts; upstream changes and reviewed local behavior preserved.
**Tests**: Range-diff, targeted conflict review, lock and workflow contracts.
**Status**: Complete

## Stage 2: Review and repair
**Goal**: Reconcile Qodo feedback and confirmed integration/CI defects.
**Success Criteria**: Every actionable review item has a tested fix or evidence-backed disposition.
**Tests**: Focused regressions, scoped lint/Bandit, affected frontend suites.
**Status**: Complete

## Stage 3: Validate and merge
**Goal**: Push rebased head, obtain current review/check results, merge into dev when prerequisites hold.
**Success Criteria**: Current-head CI and review complete; accurate human Change summary and valid policy approvals.
**Tests**: GitHub required checks and review thread audit; exact-head merge verification.
**Status**: In Progress
