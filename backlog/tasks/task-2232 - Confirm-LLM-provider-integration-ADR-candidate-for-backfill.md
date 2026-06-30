---
id: TASK-2232
title: Confirm LLM provider integration ADR candidate for backfill
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-04 01:31'
labels:
  - docs
  - process
  - adr
  - llm
  - providers
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit INV-027 from the ADR decision inventory against the current LLM provider implementation, docs, schemas, and tests before promoting it to an accepted ADR. Confirm whether adapter registry routing, OpenAI-compatible normalization/SSE behavior, trusted base URL override policy, and request-level local provider URL rejection are current governing behavior. Create a confirmation audit under Docs/ADR/inventory/, update the decision inventory with the bounded disposition, and create a follow-up backfill task only if the decision is current. Do not create accepted ADRs during this confirmation audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Confirm INV-027 against current LLM provider docs, implementation, schemas, and tests with concrete file-path evidence.
- [x] #2 Create Docs/ADR/inventory/2026-06-04-llm-provider-integration-confirmation-audit.md with disposition, evidence, caveats, and next action.
- [x] #3 Update Docs/ADR/inventory/2026-06-03-decision-inventory.md so INV-027 and the provider/integration slice reflect the confirmation result.
- [x] #4 Create a bounded follow-up Backlog task only if INV-027 is confirmed current enough for accepted ADR backfill.
- [x] #5 Record docs-only verification and Bandit applicability in TASK-2232.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review INV-027 source docs and current LLM provider code paths, provider schemas, endpoint behavior, and focused tests. Record concrete evidence and caveats in a new confirmation audit. Update the decision inventory row and recommended slice status. If current, create a bounded follow-up Backlog task for accepted ADR backfill; otherwise leave inventory-only with rationale. Run docs-only verification and record Bandit skip if no Python/source files are touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-06-04: Confirmed INV-027 evidence. Registry routing, OpenAI-compatible response/SSE normalization, strict local payload filtering, and trusted allowlisted base_url behavior are current. Found code/doc mismatch for request-level local api_url handling: ChatCompletionRequest allows extras, build_call_params_from_request and _build_adapter_request_from_chat_args do not strip api_url, and several local adapters pass request api_url to provider helpers. No accepted ADR or follow-up backfill task was created because INV-027 remains Needs owner review. Verification: git diff --check passed. Bandit: not applicable; docs-only changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the LLM provider integration confirmation audit and downgraded INV-027 to Needs owner review. No accepted ADR or follow-up backfill task was created because the local provider request-level api_url rejection claim is not confirmed by current code. Verification: git diff --check passed; Bandit not applicable for docs-only changes.
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
