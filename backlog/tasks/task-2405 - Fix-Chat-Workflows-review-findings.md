---
id: TASK-2405
title: Fix Chat Workflows review findings
status: Done
assignee: []
created_date: '2026-06-23 18:11'
updated_date: '2026-06-24 00:45'
labels:
  - chat-workflows
  - review-fix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address review findings in tldw_Server_API/app/core/Chat_Workflows: dialogue round idempotent replays, llm_phrased renderer behavior, prompt safety/bounds, and sanitized renderer fallback metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dialogue round retries with the same idempotency key and body replay successfully after continue/finish/completion.
- [x] #2 Different dialogue round bodies with the same idempotency key are rejected as conflicts.
- [x] #3 llm_phrased either performs model-backed phrasing or reports an explicit sanitized fallback when unavailable.
- [x] #4 Dialogue prompt assembly separates untrusted content from fixed system instructions and applies bounded serialization.
- [x] #5 Renderer fallback metadata and logs do not expose raw exception text.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See IMPLEMENTATION_PLAN_chat_workflows_review_fixes_2405.md. Stages: tracking/scope, regression tests, implementation, verification/closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started implementation for Chat Workflows review fixes. No repository code edits made before creating TASK-2405.

Created IMPLEMENTATION_PLAN_chat_workflows_review_fixes_2405.md for this fix set.

Red test run: focused Chat Workflows tests produced 7 expected failures covering renderer fallback metadata, default llm_phrased fallback, dialogue round replay after continue/completion, conflict on mismatched replay, prompt isolation, and prompt bounds.

Verification in isolated worktree codex/chat-workflows-review-fixes after rebasing onto origin/dev: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:unraisableexception tldw_Server_API/tests/Chat_Workflows -q => 47 passed, 107 warnings in 28.17s. Bandit touched scope => 0 results, 0 errors. git diff --check => clean. Standard pytest cleanup hit a long pytest unraisable-exception GC shutdown path during iteration, so final pytest verification used -p no:unraisableexception; test behavior coverage is unchanged.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Chat Workflows review findings: dialogue round idempotency now replays completed rounds after step/run advancement and rejects mismatched key reuse; llm_phrased fallback now reports unavailable/error states explicitly; renderer fallback metadata/logs avoid raw exception text; dialogue prompts keep context/prior rounds out of system messages and bound serialized prompt payloads. Added regression coverage for each behavior and verified with focused Chat Workflows tests plus Bandit.
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
