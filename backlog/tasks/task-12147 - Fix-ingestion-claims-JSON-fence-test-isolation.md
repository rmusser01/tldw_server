---
id: TASK-12147
title: Fix ingestion claims JSON-fence test isolation
status: Done
assignee: []
created_date: '2026-07-04 18:18'
updated_date: '2026-07-04 19:33'
labels:
  - tests
  - claims
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The broad mid-slice now passes the Claims LLM config precedence test and stops at `test_ingestion_llm_extractor_parses_json_in_fenced_block`. The test passes in isolation but returns no claims in the broad process, consistent with a stale imported `extract_claims_for_chunks` binding after module reload while the test patches the current `ingestion_claims` module.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The ingestion LLM JSON-fence parsing test patches and invokes the same module instance.
- [x] #2 Focused JSON-fence parsing test passes.
- [x] #3 The broad mid-slice progresses past this Claims ingestion blocker.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the focused test passes and the broad process fails, proving order dependence.
2. Update the test to import and invoke `ingestion_claims` at runtime instead of using a stale imported function binding.
3. Verify the focused test and broad mid-slice, then run Bandit/diff checks for touched files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Updated the JSON-fence ingestion claims test so it patches and invokes the same ingestion_claims module instance in broad runs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the order-dependent ingestion claims JSON-fence parsing test by removing the stale top-level function import and invoking extract_claims_for_chunks through the patched module. Verification: focused touched-scope command passed (44 passed); Chat_NEW through Claims slice passed (1362 passed, 15 skipped, 15 xfailed, 2 xpassed); Discord-to-Jobs slice passed (3247 passed, 156 skipped); git diff --check passed; Bandit on touched tests reported no findings.
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
