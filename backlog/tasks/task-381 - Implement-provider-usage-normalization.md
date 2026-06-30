---
id: TASK-381
title: Implement provider usage normalization
status: Done
assignee: []
created_date: '2026-05-15 15:16'
updated_date: '2026-05-15 15:28'
labels:
  - usage
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
Implement Stage 3 of the approved chat/world-book cache cost-control plan. This slice normalizes provider usage metadata into bounded, provider-agnostic cache/cost fields before persistence. Keep the work measurement-only: do not add database columns, do not change cost calculation persistence, and do not change provider request payloads or cache behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 OpenAI, Anthropic, Gemini, OpenRouter, local OpenAI-compatible, and malformed/unknown usage payloads normalize into a stable NormalizedLLMUsage shape.
- [x] #2 Cached, cache-write, cache-read, billable input, output, total, reasoning, and choice-count fields are represented without changing existing prompt/completion/total token semantics.
- [x] #3 Raw provider usage metadata is bounded, redacted, and never includes known secret/header/prompt-like fields.
- [x] #4 Streaming fallback and missing-usage paths record an explicit estimate_source.
- [x] #5 Focused usage-normalizer tests are written with failing red runs recorded before implementation and passing green runs recorded after implementation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused red tests for provider usage normalization across OpenAI, Anthropic, Gemini, OpenRouter/local OpenAI-compatible, malformed raw payloads, and missing/estimated usage paths.
2. Implement a bounded provider-agnostic NormalizedLLMUsage helper under core Usage.
3. Wire chat usage logging to normalize provider usage metadata while preserving existing prompt/completion/total token semantics and persistence.
4. Run focused usage/chat tests, diff checks, Bandit on touched Python scope, then update the plan/task and commit the slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: focused usage/chat/accounting tests passed (18 passed); git diff --check passed; Bandit zero findings for touched Usage/Chat files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 3 provider usage normalization. Added a bounded NormalizedLLMUsage helper covering OpenAI, Anthropic, Gemini, OpenRouter, vLLM, llama.cpp, local OpenAI-compatible, and malformed/missing usage payloads. Wired chat usage logging to use normalized provider token fields and explicit estimate_source values while keeping persistence schema and legacy token columns unchanged.
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
