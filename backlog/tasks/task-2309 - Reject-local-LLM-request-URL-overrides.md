---
id: TASK-2309
title: Reject local LLM request URL overrides
status: Done
assignee: []
created_date: '2026-06-07 16:49'
updated_date: '2026-06-07 17:06'
labels:
  - llm
  - providers
  - security
  - adr
  - chat
dependencies: []
references:
  - Docs/ADR/inventory/2026-06-04-llm-provider-integration-confirmation-audit.md
  - Docs/ADR/inventory/2026-06-03-decision-inventory.md
  - backlog/tasks/task-2310 - Backfill-LLM-provider-integration-ADR.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align INV-027 with the documented LLM provider integration policy by rejecting request-level local provider endpoint URL overrides before local adapters can forward them. Add focused regression tests for api_url and provider-specific *_api_url request keys on local providers, keep trusted allowlisted base_url behavior intact for supported providers, update the LLM provider confirmation audit/inventory as needed, and record verification/Bandit results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add failing regression coverage showing local providers reject request-level api_url and provider-specific *_api_url overrides before adapter dispatch.
- [x] #2 Implement the minimal chat-service/local-provider guard so local endpoint URL override keys are rejected while trusted allowlisted base_url behavior remains unchanged.
- [x] #3 Update tests/docs/inventory notes if the INV-027 disposition changes or if a follow-up ADR backfill task becomes appropriate.
- [x] #4 Run focused Chat/LLM tests plus Bandit on touched Python paths and record results.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started in isolated worktree .worktrees/llm-local-url-override-policy from origin/dev. Root cause from TASK-2232 audit: Chat request extras can carry api_url/provider-specific *_api_url keys into local adapters, contradicting the documented config-only local endpoint policy. Plan: write failing tests for local override rejection, implement the minimal chat-service guard, update ADR inventory/audit notes if now backfillable, then run focused tests and Bandit.

Implemented the local provider request URL override guard in chat_service before adapter dispatch. RED verification: the new local-provider override tests failed before the guard with 3 failed and 4 passed because ChatBadRequestError was not raised. GREEN verification: focused Chat/LLM regression suite passed with 22 passed and 5 warnings. Bandit on the touched Python files exited 0 with zero findings after annotating test assertions with nosec B101. Additional checks: git diff --check exited 0; reference scan across touched files found no absolute developer-machine paths or temporary Bandit report artifact names. Follow-up TASK-2310 was created for bounded INV-027 ADR backfill.

Alias hardening follow-up: provider names are canonicalized through the LLM adapter registry before the local-provider URL override guard runs, so local aliases such as llama-cpp and tabby_api are covered. Updated verification after this follow-up: focused Chat/LLM regression suite passed with 24 passed and 5 warnings; Bandit on touched Python files exited 0 with zero findings; git diff --check exited 0; reference scan across touched files found no absolute developer-machine paths or temporary report artifact names.

Rebased PR branch on latest origin/dev and addressed Gemini review threads by allowing None in _resolve_chat_provider_name typing and matching the local override helper signatures to dict[str, Any]. Post-review verification: focused Chat/LLM regression suite passed with 24 passed and 5 warnings; security scan on touched Python files exited 0 with zero findings; git diff --check exited 0; touched-file reference scan found no workstation-specific paths or temporary report artifact names.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned INV-027 with the documented local provider endpoint policy. Local providers now reject request-level api_url and provider-specific *_api_url keys at the Chat adapter-request boundary before local adapters can forward them. Added regression coverage, updated the LLM provider confirmation audit and decision inventory, and created TASK-2310 for the follow-up ADR backfill.
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
