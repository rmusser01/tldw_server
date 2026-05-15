---
id: TASK-377
title: Design cache-aware cost controls for chat and world-book prompts
status: Done
assignee: []
created_date: '2026-05-15 07:00'
updated_date: '2026-05-15 07:03'
labels:
  - design
  - chat
  - world-books
  - cost-control
  - llm-cache
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved design spec for a cache-aware cost-control layer around the chat pipeline and world-book prompt injection. The design must distinguish paid provider prompt caches from local inference prefix/KV caches, include OpenAI/Anthropic/Gemini/OpenRouter plus vLLM and llama.cpp, and keep implementation staged so usage measurement and guardrails precede provider-specific behavior changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A repository design spec documents current chat/world-book prompt assembly seams and provider/local cache risks.
- [x] #2 The spec separates billing prompt-cache concerns from local inference prefix/KV cache concerns for vLLM and llama.cpp.
- [x] #3 The spec defines proposed diagnostics, usage accounting fields, guardrails, and staged implementation boundaries.
- [x] #4 The spec includes testing and verification expectations for prompt determinism, world-book token budgets, provider usage normalization, and streaming/non-streaming accounting parity.
- [x] #5 The Backlog task records the spec path and final verification notes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Approved design-spec task only. 1. Create a Backlog task before repo edits. 2. Write a repository design spec under Docs/superpowers/specs for chat/world-book cache-aware cost controls. 3. Ensure the spec separates paid provider billing prompt caches from local vLLM/llama.cpp inference prefix/KV caches. 4. Include diagnostics, usage accounting, guardrails, staged implementation boundaries, and testing expectations. 5. Verify the docs change with git diff checks and targeted content checks, then record final notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created design spec at Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md. The spec incorporates the user-approved revision separating billing prompt caches from local inference caches and explicitly includes vLLM and llama.cpp. Verification run so far: git diff --check on the spec, targeted rg coverage for required providers/cache concepts, and ASCII scan with rg -nP "[^\\x00-\\x7F]". Bandit is not applicable because this task only adds documentation.

Final verification: git diff --no-index --check returned no whitespace output for the new spec and TASK-377 Backlog file; targeted rg confirmed OpenAI, Anthropic, Gemini, OpenRouter, vLLM, llama.cpp, BillingPromptCacheIntent, InferencePrefixCacheIntent, world-book, and streaming coverage; ASCII scan returned no matches. No code was changed, so Bandit was documented as not applicable.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md to capture the approved cache-aware cost-control design for chat and world-book prompt injection. The spec separates paid provider prompt-cache accounting from local vLLM/llama.cpp inference prefix/KV cache diagnostics, defines prompt envelopes and usage normalization, adds world-book diagnostics and guardrails, and outlines staged implementation boundaries plus testing expectations. Verification: git diff --no-index --check on the new spec and Backlog task file, targeted rg coverage for required providers/cache concepts, ASCII scan with rg -nP "[^\\x00-\\x7F]". Bandit skipped as not applicable because the change is documentation and Backlog metadata only.
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
