---
id: TASK-2372
title: 'Task 3: Add local-only CATS fuzz environment builder and OpenAPI export'
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-27 17:58'
labels:
  - cats
  - fuzzing
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 for the CATS API fuzzing harness: tests for local-only environment safety and OpenAPI export command construction; add Helper_Scripts/cats_fuzz/env.py and openapi_export.py; verify focused pytest, black, and Bandit; commit with the requested message.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused env/OpenAPI tests cover sensitive detection, credential rejection, local sentinels, child env blanking, and OpenAPI command construction.
- [x] #2 env.py creates a local single-user runtime environment and rejects real credentials by default.
- [x] #3 openapi_export.py builds the module command and exports deterministic sorted OpenAPI JSON with a SHA-256 digest.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD flow: added test_cats_fuzz_env.py first, then captured the initial red pytest collection failure before implementing env.py and openapi_export.py. Verification after implementation: focused pytest passed 6 tests; Black check passed; Bandit reported zero findings; OpenAPI export CLI smoke run exited 0 with digest 50653ad9c3414b434eba48de360b9e18c2e3b5d4820b0b831a011577656e0b2c.

Review fix: split broad sensitive detection from default blocking so known local/tool variables (SINGLE_USER_API_KEY, SINGLE_USER_TEST_API_KEY, GITHUB_TOKEN, GH_TOKEN, NPM_TOKEN, CODEX_API_KEY) are sanitized or overwritten instead of rejecting normal developer shells. Added fake-FastAPI unit coverage for export_openapi cache clearing, deterministic bytes/digest, parent directory creation, and main() digest printing. Verification for fix: focused pytest passed 10 tests; Black check passed; Bandit reported zero findings in /tmp/bandit_cats_fuzz_env_export_fix.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 3 implemented and verified. Initial red focused pytest failed during collection before env.py/openapi_export.py existed. Final focused pytest passed 6 tests. Black check passed. Bandit completed with zero findings. OpenAPI export CLI smoke run exited 0 and printed digest 50653ad9c3414b434eba48de360b9e18c2e3b5d4820b0b831a011577656e0b2c.
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
