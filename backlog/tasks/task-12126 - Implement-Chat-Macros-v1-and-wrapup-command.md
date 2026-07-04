---
id: TASK-12126
title: Implement Chat Macros v1 and wrapup command
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-04 00:03
labels: []
dependencies: []
documentation:
- Docs/superpowers/specs/2026-07-03-chat-macros-design.md
- Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md
priority: medium
modified_files:
- Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md
- tldw_Server_API/app/core/Chat_Macros/storage.py
- tldw_Server_API/app/core/Chat_Macros/settings.py
- tldw_Server_API/app/core/Chat_Macros/output_profiles.py
- tldw_Server_API/app/core/Chat_Macros/service.py
- tldw_Server_API/app/core/Chat_Macros/repository.py
- tldw_Server_API/tests/Chat_Macros/unit/test_macro_storage.py
- tldw_Server_API/tests/Chat_Macros/unit/test_macro_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Chat Macros v1 system and built-in /wrapup command according to the design spec and implementation plan. Scope includes backend macro definitions/storage/run records, Jobs execution, chat slash routing, minimal frontend settings/status UI, tests, docs, Bandit, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backend Chat_Macros module supports built-in /wrapup, user macro storage, settings/output profiles, run records, branch records, and validation.
- [ ] #2 Macro invocation works from chat/workspace surfaces with chat-native branch execution, background Jobs mode, cancellation, final result persistence, and idempotent post-back.
- [ ] #3 WebUI exposes minimal macro settings/manager controls and renders macro status/final output/run detail states.
- [ ] #4 Focused backend/frontend tests pass and Bandit is run on touched backend scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation should follow Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md. The plan was produced from the approved design spec and reviewed in three subagent passes; blocking review comments were folded into the plan. Start implementation with the plan Task 1 and keep commits task-sized.

Implementation started in worktree .worktrees/chat-macros-v1 on branch codex/chat-macros-v1.

Task 1 complete: added Chat_Macros parser/model foundation, built-in /wrapup MACRO.yaml, README, and parser tests. Commits: 331e6276fb (initial parser), 8dc047de8b (skill permission test), c1f4e6eb95 (parser hardening). Verification: clean baseline command-router suite passed before implementation (26 passed); focused parser tests passed at Task 1 HEAD (11 passed, 3 warnings); Bandit on tldw_Server_API/app/core/Chat_Macros reported no findings. Reviews: spec compliance passed after adding skills permission test; code-quality review passed after hardening defaults, alias collisions, duplicate non-repeated args, and prompt step support. Minor non-blocking note: parse_macro_args still trusts directly supplied hand-built arg_specs for alias uniqueness; intended call path uses validated MacroDefinition args.

Task 2 complete: added ChaChaNotes v52 chat macro tables/repository in commits 6ea0ac7e2d and hardening follow-ups a4feb40ee8, 7f7820395d. Review cycle found and fixed run-scoped idempotency, guarded status/final-post updates, PostgreSQL v52 drift repair, datetime row mapping, and malformed JSON wrapping. Verification: repository tests 15 passed; parser tests 11 passed; git diff --check clean; Bandit /tmp/bandit_chat_macros_task2_fix2.json results empty. Spec re-review and final code-quality re-review found no Task 2 blockers.

Task 3 complete: added file-backed user macro storage, chat macro settings/output profile helpers, and ChatMacrosService with built-in /wrapup listing/disable/clone, validate/create/update/delete flows, registry sync, collision checks, macro-local output profile bounds, and rejection of future permissions. Commits: 80eeb8c9e1 (initial storage/service), a2f7ca82ee (review hardening), 8f48bd6bc8 (atomic supporting-file replacement), c826582720 (symlink validation preservation), 9972f2b768 (YAML-only support-file preflight). Review cycle fixed validate-only macro name checks, stale registry rows after rename/delete, built-in name reuse, unsupported multiple_messages output format, duplicate TASK-12127, partial update commits, and symlink preflight regressions. Verification: storage/service tests 14 passed, 3 warnings; parser/repository regressions 26 passed, 3 warnings; git diff --check clean; Bandit JSON outputs /tmp/bandit_chat_macros_task3_atomic.json, /tmp/bandit_chat_macros_task3_symlink.json, and /tmp/bandit_chat_macros_task3_yaml_only.json had errors: [] and results: []. Spec and code-quality re-reviews found no findings at 9972f2b768.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation should follow Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md. The plan was produced from the approved design spec and reviewed in three subagent passes; blocking review comments were folded into the plan. Start implementation with the plan Task 1 and keep commits task-sized.

Implementation started in worktree .worktrees/chat-macros-v1 on branch codex/chat-macros-v1.

Task 1 complete: added Chat_Macros parser/model foundation, built-in /wrapup MACRO.yaml, README, and parser tests. Commits: 331e6276fb (initial parser), 8dc047de8b (skill permission test), c1f4e6eb95 (parser hardening). Verification: clean baseline command-router suite passed before implementation (26 passed); focused parser tests passed at Task 1 HEAD (11 passed, 3 warnings); Bandit on tldw_Server_API/app/core/Chat_Macros reported no findings. Reviews: spec compliance passed after adding skills permission test; code-quality review passed after hardening defaults, alias collisions, duplicate non-repeated args, and prompt step support. Minor non-blocking note: parse_macro_args still trusts directly supplied hand-built arg_specs for alias uniqueness; intended call path uses validated MacroDefinition args.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Implementation plan followed or deviations documented
- [ ] #3 Focused backend tests passing
- [ ] #4 Focused frontend tests passing
- [ ] #5 Bandit run for touched backend scope and new findings fixed
- [ ] #6 Documentation updated
- [ ] #7 Final summary added
- [ ] #8 Known skips or blockers documented
<!-- DOD:END -->
