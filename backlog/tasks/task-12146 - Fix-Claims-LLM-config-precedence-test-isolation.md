---
id: TASK-12146
title: Fix Claims LLM config precedence test isolation
status: Done
assignee: []
created_date: '2026-07-04 17:48'
updated_date: '2026-07-04 19:33'
labels:
  - tests
  - claims
  - config
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The broad mid-slice pytest run fails `test_claims_llm_config_prefers_claims_specific_settings` with provider resolving to config-file `ollama` instead of test override `groq`. The test passes in isolation, indicating an order-dependent stale settings reference after prior module reload/config refresh behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Claims LLM config precedence tests remain deterministic when run after preceding Chat/Chunking/ChromaDB tests.
- [x] #2 Focused Claims config precedence test file passes.
- [x] #3 Relevant broader slice resumes past this Claims blocker.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the order-dependent failure and confirm root cause.
2. Update the test to mutate the current config module settings object instead of a stale imported settings binding.
3. Verify focused tests, relevant slice, diff check, and Bandit on the touched test file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Updated the Claims config precedence test to read the current config module settings object at runtime instead of retaining a stale imported settings binding after reloads.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the order-dependent Claims LLM config precedence test by resolving settings through the config module on each snapshot, mutation, and restore. Verification: focused touched-scope command passed (44 passed); Chat_NEW through Claims slice passed (1362 passed, 15 skipped, 15 xfailed, 2 xpassed); Discord-to-Jobs slice passed (3247 passed, 156 skipped); git diff --check passed; Bandit on touched tests reported no findings.
<!-- SECTION:FINAL_SUMMARY:END -->

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
