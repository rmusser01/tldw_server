# Post-PR2969 UAT repair checkpoint

Backlog: TASK13260.270. Branch: `codex/uat295-postgres-notes-20260919`.

## Stage 1: Finish targeted acceptance
**Goal**: Close remaining application findings, retaining UAT261 by user direction.
**Success Criteria**: UAT290 normal/Character tab, draft, recovery and account controls pass; storage-failure review finding repaired.
**Tests**: Causal session-storage regressions, adjacent composer/account tests, immutable native PostgreSQL acceptance.
**Status**: In Progress

## Stage 2: Verify combined repairs
**Goal**: Review and validate the complete change against latest dev.
**Success Criteria**: Affected frontend/backend tests pass with real PostgreSQL; no new static or Bandit findings; independent review clear.
**Tests**: Changed and adjacent test suites; ESLint/TypeScript baseline comparison; touched Python Bandit.
**Status**: Not Started

## Stage 3: Publish and review
**Goal**: Publish the checkpoint PR and address Qodo/CI feedback.
**Success Criteria**: Latest dev included, generated captures excluded, actionable comments resolved and required checks green.
**Tests**: PR check results and focused regressions for any review fixes.
**Status**: Not Started

## Stage 4: Merge and resume
**Goal**: Merge normally and resume the fresh four-cell A/B/C matrix.
**Success Criteria**: Requester-written Change summary satisfies the repository gate; merge verified; next matrix starts from the integrated revision.
**Tests**: Merge ancestry and fresh profile/source manifests.
**Status**: Not Started
