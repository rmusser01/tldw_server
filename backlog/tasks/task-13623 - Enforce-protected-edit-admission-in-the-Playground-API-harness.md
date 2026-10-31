---
id: TASK-13623
title: Enforce protected edit admission in the Playground API harness
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3229 message edit journeys currently fail because the fake API neither advertises nor enforces the deployed protected edit target. Restore schema and atomic original-target/prefix checks before fake provider activity and writes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Original edited-send, held-reply and replacement-draft journeys pass while canonical originals remain unchanged
- [x] #2 Schema advertises only enforced edit support; relevant harness regressions, both client types, bounded review and hooks pass
- [x] #3 Invalid original edit targets refuse before provider admission; changed/deleted admitted inputs refuse held-response settlement
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR; emulate the existing native protected edit contract and original-row preservation. Plan: Docs/Plans/IMPLEMENTATION_PLAN_chat_fake_protected_edit_13623.md. PR: https://github.com/rmusser01/tldw_server/pull/3229.
Refined the held-response criterion to match the real committed-admission contract: validate/check+append synchronously before provider work; later original-target changes do not invalidate an already admitted input, while changed/deleted admitted inputs refuse settlement. Lazy canonical helper loading preserves harness memory-DB binding. Original6 journeys plus17 direct admission/refusal controls pass; unsupported-capability UI control and wider regressions/types are next.
Final qualification: 50 tests pass across direct protected-edit, saved message-actions, harness and interrupted-reply suites. Both WebUI and extension normal TypeScript checks exit 0. Independent frontend review cleared the conversation-retirement finding after captured object identity and admitted-input settlement checks; originals may lawfully change after atomic admission. Review: Docs/Reviews/CHAT_FAKE_PROTECTED_EDIT_13623_2026_10_12.md. Bandit N/A: TypeScript test-only scope. Normal exact-file hooks pending.
Normal exact-file pre-commit hooks all passed. Fifty regression tests and both client type checks passed. Remaining whole-PR CI/UX/H2 work and final review/merge remain open; this bounded harness unit is complete.
Completed owned plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_fake_protected_edit_13623_completed.md; durable scope and evidence are in the review document.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored strict protected-edit support in the fake API, with atomic original-target/prefix admission, conversation incarnation checks across helper loading, and admitted-input validation at held reply settlement. Canonical originals remain intact. Fifty relevant tests, both client type checks, independent bounded review and normal hooks pass. TypeScript test-only scope; Bandit N/A.
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
