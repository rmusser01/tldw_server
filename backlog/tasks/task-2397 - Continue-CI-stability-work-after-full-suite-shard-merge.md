---
id: TASK-2397
title: Continue CI stability work after full-suite shard merge
status: In Progress
labels:
- ci
- github-actions
- stability
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track follow-up work after PR #2258 merged the full-suite shard split into dev. Initial scope: monitor the merged workflow behavior on dev/main/release triggers, inspect any remaining queued or stale workflow-run behavior, and address residual CI runtime/stability issues without reintroducing early pytest maxfail behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Monitor the merged CI shard workflow behavior on `dev`, `main`, and release/manual triggers.
- [ ] Inspect any remaining queued, cancelled, or failing GitHub Actions jobs before pushing follow-up fixes.
- [ ] Address residual CI runtime or stability issues without reintroducing early pytest `--maxfail` behavior.
- [ ] Verify and address still-valid unresolved PR #2258 review comments, including inline and outside-diff CodeRabbit/Gemini/Qodo findings.
- [ ] Record local and GitHub verification evidence before merging follow-up work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- PR #2258 merged into `dev` at 2026-06-23T05:55:28Z with merge commit `c16aada8dd7e3061eb89c3c312ffcc1986fafa35`.
- Fresh PR #2258 checks before merge: 755 passed, 4 skipped, 0 pending, 0 failed, 0 cancelled.
- One unrelated stale queued CodeQL run for PR #1985 returned HTTP 500 when cancellation was attempted; it did not block #2258 completion.
- Follow-up scope expanded to cover unresolved PR #2258 review feedback after merge: 1 Gemini inline thread, 26 CodeRabbit inline comments, 3 CodeRabbit outside-diff comments, and Qodo's top-level CI coverage/security concerns.
- Implementation plan: `Docs/Plans/2026-06-23-pr2258-review-followups.md`.
- PR #2258 review follow-up fixes added to PR #2431 include CI/config/doc fixes, runtime lifecycle/correctness fixes, and test isolation/assertion hardening. Qodo's DB URL exposure concern is handled by masking PostgreSQL passwords and assembled URLs before writing CI env vars.
- Qodo's prior PR full-suite and OS Postgres auto-start concerns are obsolete in this branch: macOS/Windows full-suite shards run for PRs with backend changes and set `TLDW_TEST_NO_DOCKER=1`.
- Local verification for the review follow-up: py_compile on touched Python passed; CI contract tests 36 passed; focused backend regression suite 81 passed, 1 skipped; APKG/media guard/embeddings suite 64 passed, 14 skipped; frontend quickstart networking Vitest 11 passed; `git diff --check` passed; Bandit exited 0 for touched production Python and touched tests.
- Known skip: real HuggingFace embedding tests now require explicit `RUN_REAL_HF_EMBEDDING_TESTS=true` to avoid accidental local/network-dependent runs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added PR #2258 review follow-up fixes to PR #2431: CI/config hardening, runtime correctness/lifecycle fixes, test isolation improvements, and local verification evidence. The broader task remains open until the pushed PR #2431 run is observed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
