---
id: TASK-385
title: Implement local prompt cache diagnostics
status: Done
assignee: []
created_date: '2026-05-15 16:16'
updated_date: '2026-05-15 16:27'
labels:
  - llm-cache
  - local-llm
  - cost-control
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
Add local inference prompt/prefix cache diagnostics for vLLM and llama.cpp without treating runtime cache reuse as paid-provider billing savings. Diagnostics should describe request-shape stability and known llama.cpp cache flags while preserving strict OpenAI-compatible filtering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 vLLM diagnostics describe prefix-cache compatibility and unstable request-shape risks without inventing billing-cache savings.
- [x] #2 llama.cpp diagnostics surface known prompt-cache startup flags and request-level cache extension use when available.
- [x] #3 Local providers remain cost-neutral unless authoritative provider usage metadata proves cache token effects.
- [x] #4 Strict-filter tests protect local OpenAI-compatible payload compatibility.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented local cache diagnostics with a separate inference_prefix_cache_intent path for vLLM and llama.cpp. Diagnostics are attached as tldw_local_cache_diagnostics on non-streaming local responses only when cache-related signal exists; strict payload filtering still removes non-OpenAI cache hint keys before sending to local OpenAI-compatible servers.

Verification: focused red run failed on missing local_cache_diagnostics module; focused Stage 7 pack passed 13 tests; expanded pack passed 61 tests; py_compile passed for touched Python files; git diff --check passed; Bandit wrote /tmp/bandit_local_cache_stage7.json with results: [].
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added cost-neutral local prompt/prefix cache diagnostics for vLLM and llama.cpp. The implementation reports vLLM request-shape stability, sanitized llama.cpp prompt-cache runtime flags and request extension keys, preserves strict OpenAI-compatible outbound filtering, and keeps local runtime cache reuse separate from paid-provider billing cache usage.
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
