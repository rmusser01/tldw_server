# Workspace R5/R6 Remediation

Tracked by TASK-12020.50. Continue the accepted findings in
`Docs/Development/workspace-unmerged-review-2026-09-20.md` without enabling the
owned route or rebasing the existing dirty worktree.

## Stage 1: Current-Target Note Durability
**Goal**: Allow note creation when its own recovery marker is durable, even when
another target has an unsaved draft. Preserve aggregate warnings and reject
non-durable or conflicted current markers.
**Success Criteria**: Store and actual QuickNotes dispatch regressions pass.
**Tests**: Unrelated failed write, current-target quota failure, conflict, reload.
**Status**: Complete (235 tests passed; independent review clear)

## Stage 2: Canonical Session Readiness
**Goal**: Use the canonical identity profile for bearer and cookie readiness.
**Success Criteria**: Valid sessions work with legacy endpoints disabled; auth
denials remain disconnected without public-health fallback.
**Tests**: Connection store transport matrix and isolated real backend checks.
**Status**: Complete (50 tests and 18 real HTTP requests passed; review clear)

## Stage 3: Verification And Record
**Goal**: Run scoped regression, type/lint checks and review; update findings and
acceptance evidence without claiming complete workspace certification.
**Success Criteria**: Findings have measured dispositions and remaining limits.
**Tests**: Broader workspace regression, TypeScript, ESLint, diff check; Bandit
when Python production code is changed.
**Status**: Complete (1521-test regression; final focused 285 tests; TypeScript
and ESLint verified; findings and acceptance records updated)

Evidence: `Docs/Development/workspace-unmerged-review-2026-09-20.md`.
TASK-12020.50 stays In Progress for remaining integration/acceptance. No Python
production edits in this follow-up; Bandit is not applicable to TypeScript.
