---
id: TASK-379
title: Implement prompt cost envelope primitives
status: Done
assignee: []
created_date: '2026-05-15 14:53'
updated_date: '2026-05-15 14:59'
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
Implement Stage 1 of the approved chat/world-book cache cost-control plan. This slice creates deterministic prompt-cost envelope primitives for final provider-bound chat messages without changing provider request payloads, prompt layout, usage persistence, or cache behavior. Keep the work measurement-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Final outbound chat messages can be converted into a PromptCostEnvelope with bounded diagnostics and no prompt text persistence.
- [x] #2 Stable versioned fingerprints are produced from canonicalized provider-bound content, including order-sensitive aggregate fingerprints.
- [x] #3 System/static, world-book, retrieval/tool, history, and user-turn segments can be accounted separately when supplied.
- [x] #4 Token estimates are conservative, deterministic, non-negative, and do not add a tokenizer dependency.
- [x] #5 Focused prompt-cost envelope tests are written with failing red runs recorded before implementation and passing green runs recorded after implementation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused prompt-cost envelope unit tests first and verify the red failure.
2. Implement the minimal prompt_cost_envelope module without wiring it into provider dispatch.
3. Run the focused new test file and the existing chat token-estimate baseline test.
4. Run git diff --check and Bandit on the touched Chat module scope.
5. Update TASK-379 with verification results, complete it, then commit the slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline before implementation: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_token_estimates.py -q passed (1 test).

TDD red runs recorded: new test file initially failed during collection because prompt_cost_envelope did not exist; additional unknown-content-part test failed until unsupported content parts were changed to bounded markers.
Green verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py tldw_Server_API/tests/Chat/unit/test_chat_service_token_estimates.py -q passed (7 tests).
Security/format verification: git diff --check passed. Bandit command passed with zero findings: python -m bandit -r tldw_Server_API/app/core/Chat/prompt_cost_envelope.py -f json -o /tmp/bandit_task379.json.
Known notes: no provider dispatch, prompt layout, usage persistence, or cache behavior was wired in this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added prompt-cost envelope primitives for final provider-bound chat messages. The new helper canonicalizes messages, produces versioned SHA-256 prompt fingerprints, estimates segment tokens with a local 4 chars/token heuristic, separates static/world-book/retrieval/history/user-turn segment totals, and exposes bounded diagnostics without raw prompt text. Added focused unit coverage and marked Stage 1 complete in the implementation plan.
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
