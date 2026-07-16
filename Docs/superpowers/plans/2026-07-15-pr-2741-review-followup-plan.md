# PR 2741 review follow-up implementation plan

## Stage 1: Rebase and inventory
**Goal**: Rebase the isolated PR branch onto current `origin/dev` and enumerate every review thread and CI check.
**Success Criteria**: The intended diff is preserved; all inline, review, and issue comments are classified; CI failures are distinguished from queued jobs.
**Tests**: `git diff --check origin/dev...HEAD`; GitHub review-thread and check-rollup queries.
**Status**: Complete

## Stage 2: Reproduce and fix valid findings
**Goal**: Add regression coverage for each valid correctness or security finding and implement the smallest safe fix.
**Success Criteria**: Each behavioral regression fails before its fix and passes afterward; false positives receive evidence-backed explanations; no rollout scope expands.
**Tests**: Focused Python and Node test selections covering inventory validation, discovery contracts/planning/adapters/execution, DNS, and the one-hop transport.
**Status**: Complete

## Stage 3: Integrated verification
**Goal**: Validate the rebased and reviewed branch as one unit.
**Success Criteria**: Focused and complete Research suites, inventory validators, lint/format, compatibility parsing, diff hygiene, and Bandit pass for touched scope.
**Tests**: Commands and exact outcomes recorded in `TASK-12968.8`.
**Status**: Complete

## Stage 4: Publish and close review
**Goal**: Publish the rewritten branch safely and close every addressed review thread.
**Success Criteria**: One `--force-with-lease` push succeeds; replies identify fixes or verified non-actions; threads are resolved; PR mergeability and CI state are rechecked.
**Tests**: GitHub PR head/base/mergeability, unresolved-thread count, and check-rollup queries.
**Status**: In Progress
