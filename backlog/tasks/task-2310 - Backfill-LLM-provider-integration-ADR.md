---
id: TASK-2310
title: Backfill LLM provider integration ADR
status: Done
assignee: []
created_date: '2026-06-07 16:52'
updated_date: '2026-06-07 17:32'
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
- [x] #1 Add a new immutable ADR under Docs/ADR/ with the next available ADR number and one bounded LLM provider integration decision.
- [x] #2 Link the ADR from Docs/ADR/README.md and update Docs/ADR/inventory/2026-06-03-decision-inventory.md so INV-027 points to the accepted ADR.
- [x] #3 Keep caveats explicit: request-level local endpoint rejection is enforced at the Chat adapter-request boundary, local adapters may still accept config-derived URLs internally, provider-specific response preservation is an extension, and future local provider URL policy changes need a separate decision.
- [x] #4 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-025 for the bounded LLM provider adapter routing and override decision. Updated the ADR index, INV-027 inventory row, provider/integration owner-review handoff, and the LLM provider confirmation audit to point to ADR-025.

Verification recorded on 2026-06-07:
- git diff --check: pass.
- ADR/link reference scan: ADR-025 is present in the ADR index, new ADR, inventory row, and confirmation audit.
- Portability artifact scan: no developer-machine absolute paths or temporary Bandit report artifact names found in touched docs/task files.

Bandit: not run because this branch only touches Markdown ADR, inventory, audit, and Backlog task records; no Python/source files were changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added ADR-025 as the accepted bounded LLM provider integration decision. Linked it from the ADR index, updated INV-027 to point at ADR-025, refreshed the related confirmation audit, and recorded docs-only verification plus Bandit non-applicability.
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
