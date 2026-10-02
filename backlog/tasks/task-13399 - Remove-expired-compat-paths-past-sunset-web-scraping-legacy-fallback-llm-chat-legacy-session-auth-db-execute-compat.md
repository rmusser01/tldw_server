---
id: TASK-13399
title: Remove expired compat paths past sunset (web_scraping_legacy_fallback, llm_chat_legacy_session,
  auth_db_execute_compat)
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 1 Stage 1. Plan: Docs/plans/2026-10-01-due-debt-sweep-implementation-plan.md. All three sunsets (2026-06-30/07-15/08-01) past due as of 2026-10-01. Registry: tldw_Server_API/app/core/deprecations/runtime_registry.py; call sites chat_calls.py:79,128; auth_service.py:65,80; web_scraping_service.py:364. Includes deprecated /me endpoints (users.py:544,589) and USER_DB_BASE alias decision gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
