---
id: TASK-13194
title: Provide conversational responses through the Persona Buddy live session
status: To Do
created_date: 2026-09-05 21:29
labels:
- persona
- buddy
- uat
priority: high
references:
- Docs/Reviews/MIGU_BUDDY_MERGED_LIVE_UAT_2026_09_05.md
- TASK-13180
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-merge Migu UAT on dev 220bf544b7 reproduced a usability gap: a simple greeting/request for an exact reply returns a rag_search tool plan and no conversational response. The Buddy transport and review feedback work, but ordinary conversation cannot yet pass provider-response acceptance. Preserve explicit tool approval and the existing Persona Chat/provider ownership boundaries; determine and record the relevant ADR before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A simple conversational prompt through the real Buddy live session returns a correlated provider-backed answer without requiring an unrelated RAG tool approval.
- [ ] #2 Stop cancels active generation and a subsequent send can complete in the same session without a late response replacing current feedback.
- [ ] #3 Tool-requiring requests retain explicit review and existing authentication, scope and approval policy.
- [ ] #4 Real provider UAT and targeted regression evidence identify the exact tested revision and provider configuration without exposing secrets.
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
