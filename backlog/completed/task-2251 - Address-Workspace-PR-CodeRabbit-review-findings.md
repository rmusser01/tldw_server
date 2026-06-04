---
id: TASK-2251
title: Address Workspace PR CodeRabbit review findings
status: Done
labels:
- workspaces
- pr-review
- coderabbit
priority: high
documentation:
- https://github.com/rmusser01/tldw_server/pull/2252
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address current CodeRabbit review findings for Workspace core/project root/file inventory changes after rebasing PR #2252 onto latest dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review each current actionable CodeRabbit finding against current code.
- [x] #2 Fix still-valid findings with minimal scoped changes.
- [x] #3 Add or update regression tests for behavioral fixes.
- [x] #4 Run focused Workspace/service tests plus compile, Bandit, and diff hygiene.
- [x] #5 Push rebased PR branch and resolve addressed review threads where possible.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added regression coverage for relative path-hint redaction, sanitized partial dependency errors, root-relative diagnostic hints, recursive glob handling, ignore-before-bounds scanning, sandbox resolver failure wrapping, missing-workspace 404 service errors, startup worker registration, and DB fixture cleanup.
- Updated Workspace context and API root path hints to collapse relative multi-segment paths to their final segment.
- Updated partial dependency errors to preserve scope while returning a stable non-sensitive code/message.
- Preserved unsupported `**` ignore patterns and matched parent paths so ignored directories cover descendants without unsafe down-conversion.
- Moved scanner ignore decisions before bounds and metadata checks to prevent ignored trees from producing partial diagnostics.
- Converted sandbox attach/mount resolver exceptions into sanitized service/failure contracts.
- Confirmed the Config README already has the requested blank line before `## Workspace Project Roots`; no docs edit was needed there.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2252 onto latest `origin/dev` and addressed the current CodeRabbit review findings.

Verification:
- RED focused review tests: `9 failed, 1 passed, 6 warnings`.
- Focused green review tests: `13 passed, 6 warnings`.
- Broad Workspace/startup suite: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py -q --tb=short --disable-warnings` -> `275 passed, 8 warnings`.
- Compile: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m compileall ...` -> exit 0.
- Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r ... -f json -o /tmp/bandit_workspace_coderabbit_review.json` -> `0 results, 0 errors, 0 skipped`.
- Diff hygiene: `git diff --check` -> exit 0.
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
