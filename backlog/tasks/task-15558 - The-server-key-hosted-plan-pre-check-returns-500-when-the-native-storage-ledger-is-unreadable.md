---
id: TASK-15558
title: The server-key hosted-plan pre-check returns 500 when the native storage ledger
  is unreadable
status: To Do
labels:
- quotas
- billing
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Priority: medium. Follow-up of TASK-15481 (PR #3492), from its final review. _enter_the_hosted_plans_enforcer (endpoints/chat.py, moved verbatim from dev by TASK-15481) catches only _CHAT_ENDPOINT_NONCRITICAL_EXCEPTIONS, so a NativeQuotaUnavailableError from _with_native_storage (Billing/enforcement.py, not governed by BILLING_ENFORCEMENT_FAILURE_MODE) fails a server-key /chat/completions with 500. It should follow BILLING_ENFORCEMENT_FAILURE_MODE or fail open as its docstring says.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A server-key /chat/completions request whose hosted-plan pre-check raises NativeQuotaUnavailableError no longer returns 500: it follows BILLING_ENFORCEMENT_FAILURE_MODE (or fails open as the docstring says), with a test that pins the real error
- [ ] #2 ADR-083 and Docs/Operations/Usage_Quotas.md state the behaviour, with the Docs/Published mirror refreshed
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
