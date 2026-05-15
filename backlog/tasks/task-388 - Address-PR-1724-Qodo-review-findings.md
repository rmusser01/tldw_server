---
id: TASK-388
title: Address PR 1724 Qodo review findings
status: Done
assignee: []
created_date: '2026-05-15 19:23'
updated_date: '2026-05-15 19:41'
labels:
  - llm-cache
  - pr-review
  - cost-control
  - coderabbit
dependencies: []
documentation:
  - 'https://github.com/rmusser01/tldw_server/pull/1724'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Handle the active PR #1724 review threads after the cache review-fix push. Implement concrete correctness, migration, schema, documentation, and maintainability fixes; verify policy-only findings against codebase reality; and resolve/comment on all review threads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 usage_tracker estimate_source fallback reserves provider_usage for real provider usage metadata
- [x] #2 chat_service forwards inference_prefix_cache_intent to provider call params
- [x] #3 new cache modules have module docstrings and touched review test helpers/tests are type annotated
- [x] #4 dynamic admin usage SQL finding is either fixed or answered with verified technical rationale
- [x] #5 All current PR review threads are replied to and resolved after verification
- [x] #6 CodeRabbit review threads are addressed with fixes or verified technical replies
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Expanded from Qodo-only follow-up to all unresolved PR #1724 review threads after live review refresh. Implemented valid CodeRabbit findings for character-chat guardrails/schema forwarding, migration 088 fail-fast behavior, world-book missing dependency handling, prompt envelope system-message accounting, data URI sanitization, local llama.cpp cache mode detection, cache-cost clamping, legacy estimated flag propagation, PostgreSQL summary fallback response shape, README guardrail docs, and Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all active PR #1724 review threads found during live refresh. Fixed cache estimate-source classification and estimated-flag propagation, cache cost clamping, local llama.cpp cache mode detection, prompt/system-message cost envelope accounting, character-chat prompt guardrail fail-closed behavior, character-chat local cache intent schema/forwarding, world-book no-db fallback, migration 088 fail-fast/idempotent column handling, PostgreSQL admin summary fallback shape, module docstrings, test type annotations, guardrail README docs, and Backlog metadata. Verified with py_compile, git diff --check, Bandit on touched source with zero findings, and the 66-test focused review suite.
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
