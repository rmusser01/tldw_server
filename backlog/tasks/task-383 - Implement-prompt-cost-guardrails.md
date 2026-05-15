---
id: TASK-383
title: Implement prompt cost guardrails
status: Done
assignee: []
created_date: '2026-05-15 15:42'
updated_date: '2026-05-15 15:53'
labels:
  - chat
  - cost-control
  - llm-cache
  - implementation
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-chat-worldbook-cache-cost-control-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add shared pre-dispatch prompt/cost guardrail decisions for chat and character-chat prompt assembly. Guardrails should detect surprising prompt growth, cache-churn risk, high output caps, high choice counts, and reasoning-effort risk without exposing prompt text.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Warn-only remains the default behavior unless hard caps are explicitly configured.
- [x] #2 Shared guardrail logic returns bounded warning/block metadata without prompt text.
- [x] #3 Non-stream and streaming chat evaluate the same guardrail decision before provider dispatch.
- [x] #4 Tests cover warn thresholds, hard block thresholds, fingerprint churn, high max-token caps, high choice counts, and reasoning-effort risk.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented prompt_cost_guardrails.py with prompt-safe decisions, env/config loading, warn/block thresholds, fingerprint churn warnings, output cap/choice/reasoning risk warnings, and bounded metadata. Wired chat streaming/non-streaming before provider dispatch and added character-chat guardrail evaluation after world-book insertion.

Verification: red import failure confirmed for missing prompt_cost_guardrails module; focused guardrail tests and chat token estimate tests passed; broader chat_service_content and streaming structured tests passed; character chat error-mapping unit tests passed; py_compile passed for modified Python files; git diff --check passed; Bandit on touched app Python returned zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added prompt-safe pre-dispatch cost guardrails for chat and character chat. Default config remains disabled/warn-only; configured hard caps block with 413 before provider dispatch, and warnings are exposed through non-stream chat metadata, logs, and persisted character-chat message metadata when available. Documented the new config keys, including vLLM/llama.cpp as local runtime/cache-efficiency diagnostics rather than provider billing claims.
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
