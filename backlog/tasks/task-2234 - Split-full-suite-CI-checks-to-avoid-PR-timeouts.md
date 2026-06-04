---
id: TASK-2234
title: Split full-suite CI checks to avoid PR timeouts
status: Done
labels:
- ci
- github-actions
- testing
priority: high
modified_files:
- .github/workflows/ci.yml
- Docs/Plans/2026-06-03-ci-full-suite-sharding-implementation-plan.md
- backlog/tasks/task-2234 - Split-full-suite-CI-checks-to-avoid-PR-timeouts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restructure the GitHub Actions CI full-suite jobs so PRs do not run all slow test modules serially in one runner. Keep full Linux coverage for Python 3.12 and 3.13, keep full macOS/Windows Python 3.12 coverage on PRs through shards, and run expanded macOS/Windows Python 3.13 coverage only for non-PR release/main/manual contexts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] PR full-suite checks no longer run every backend module serially in one runner for each Python/OS combination.
- [x] Python 3.12 and Python 3.13 full testing runs through Ubuntu shard jobs.
- [x] Python 3.11 runs compatibility smoke coverage.
- [x] PR macOS/Windows Python 3.12 checks are backed by full shard coverage, not smoke-only subsets.
- [x] Expanded macOS/Windows Python 3.13 shard coverage runs only for non-PR release/main/manual contexts.
- [x] macOS/Windows full shard jobs skip Postgres fixture Docker auto-start quickly when no explicit Postgres service is provided.
- [x] Workflow validation evidence is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create an isolated worktree from origin/dev. 2. Replace the serial full-suite matrix jobs with reusable shard jobs and smoke/release variants. 3. Preserve check-name compatibility via summary jobs when practical. 4. Validate workflow YAML/action syntax and commit the scoped changes. 5. Push branch and open a PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Opened PR https://github.com/rmusser01/tldw_server/pull/2258 from codex/ci-full-suite-shards into dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR review feedback. PR macOS/Windows Python 3.12 checks now use full shard jobs plus per-OS summary checks retaining the old required check names, instead of smoke-only subsets. The non-PR OS expansion now covers Python 3.13 release/main/manual shards. All macOS/Windows full shard jobs set TLDW_TEST_NO_DOCKER=1 so Postgres-backed suites skip quickly when no explicit Postgres service is provided, avoiding fixture Docker auto-start flakiness on non-Linux runners. Verification rerun locally: git diff --check passed; PyYAML parsed .github/workflows/ci.yml; a structural check verified all needs targets, shard paths, OS summary check names, and OS Docker-skip envs resolve; grep found no smoke-only OS PR job and no remaining invalid needs.<hyphenated-job-id> expressions. actionlint is not installed locally. Bandit remains skipped because this change only touches CI YAML and task/plan documentation.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
