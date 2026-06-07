---
id: TASK-2310
title: Backfill LLM provider integration ADR
status: To Do
assignee: []
created_date: '2026-06-07 16:52'
labels:
  - docs
  - process
  - adr
  - llm
  - providers
dependencies:
  - TASK-2309
references:
  - Docs/ADR/inventory/2026-06-04-llm-provider-integration-confirmation-audit.md
  - tldw_Server_API/app/core/LLM_Calls/README.md
  - tldw_Server_API/app/core/Chat/chat_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a bounded accepted ADR for INV-027 after TASK-2309 aligned the local provider endpoint override policy with the documented behavior. Scope the ADR to adapter-registry routing, OpenAI-compatible response and SSE normalization, strict local payload filtering, trusted allowlisted base_url overrides, and config-only local provider endpoint URLs with request-level api_url/*_api_url rejection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Add a new immutable ADR under Docs/ADR/ with the next available ADR number and one bounded LLM provider integration decision.
- [ ] #2 Link the ADR from Docs/ADR/README.md and update Docs/ADR/inventory/2026-06-03-decision-inventory.md so INV-027 points to the accepted ADR.
- [ ] #3 Keep caveats explicit: request-level local endpoint rejection is enforced at the Chat adapter-request boundary, local adapters may still accept config-derived URLs internally, provider-specific response preservation is an extension, and future local provider URL policy changes need a separate decision.
- [ ] #4 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
