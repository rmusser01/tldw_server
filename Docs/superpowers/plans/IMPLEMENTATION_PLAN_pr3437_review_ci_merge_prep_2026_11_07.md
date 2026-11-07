# PR3437 Review and CI Merge Preparation (TASK-15477)

ADR assessment: accepted ADR-097 and ADR-021 govern existing behavior. Correcting
implementation, tests and integration gates requires no new durable decision.
No merge or auto-merge is authorized by this request.

## Stage 1: Triage and Independent Review
**Goal**: Inventory every PR comment, failed CI job and independent code finding.
**Success Criteria**: Complete paginated comments/threads and exact job logs;
an independent requesting-code-review assessment of the current full PR.
**Tests**: Inspect base/head SHAs, actual failed steps and existing task history.
**Status**: Complete

## Stage 2: Correct and Verify
**Goal**: Address verified findings and CI root causes without weakening gates.
**Success Criteria**: Regression-first fixes where applicable, focused green
checks, touched compilation/lint/Bandit and independent fix review.
**Tests**: Exact failed commands, relevant regression suites and security scans;
preserve all budgets, ownership, privacy and fail-closed caller contracts.
**Status**: Complete

## Stage 3: Publish and Prepare for Merge
**Goal**: Push corrections, respond to comments and report real merge gates.
**Success Criteria**: PR updated, every finding disposition recorded, required
CI genuinely passing or exact external blockers stated; human Change summary
remains mandatory before merge. Preserve worktree and never forge statuses.
**Tests**: Latest remote PR head, reviews, checks, conflicts and clean worktree.
**Status**: In Progress
