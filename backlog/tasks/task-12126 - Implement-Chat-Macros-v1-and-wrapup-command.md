---
id: TASK-12126
title: Implement Chat Macros v1 and wrapup command
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-08-23 17:32
labels: []
dependencies: []
documentation:
- Docs/superpowers/specs/2026-07-03-chat-macros-design.md
- Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md
priority: medium
references:
- https://github.com/rmusser01/tldw_server/pull/2618
modified_files:
- Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md
- apps/packages/ui/src/components/Option/ChatWorkspace/MacroRunDetailDrawer.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/MacroStatusCard.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx
- apps/packages/ui/src/components/Option/Settings/ChatMacrosSettings.tsx
- apps/packages/ui/src/utils/server-error-message.ts
- tldw_Server_API/app/api/v1/endpoints/chat.py
- tldw_Server_API/app/api/v1/endpoints/chat_macros.py
- tldw_Server_API/app/api/v1/schemas/chat_macros.py
- tldw_Server_API/app/core/Chat_Macros
- tldw_Server_API/app/services/chat_macros_jobs_worker.py
- tldw_Server_API/app/services/startup_content_jobs_pollers.py
- tldw_Server_API/tests/Chat_Macros
- tldw_Server_API/tests/Chat_NEW/integration/test_chat_completions_api.py
- tldw_Server_API/tests/Services/test_chat_macros_jobs_worker_startup.py
- tldw_Server_API/tests/chat_macros_test_helpers.py
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation should follow Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md. The plan was produced from the approved design spec and reviewed in three subagent passes; blocking review comments were folded into the plan. Start implementation with the plan Task 1 and keep commits task-sized.

Implementation started in worktree .worktrees/chat-macros-v1 on branch codex/chat-macros-v1.

Task 1 complete: added Chat_Macros parser/model foundation, built-in /wrapup MACRO.yaml, README, and parser tests. Commits: 331e6276fb (initial parser), 8dc047de8b (skill permission test), c1f4e6eb95 (parser hardening). Verification: clean baseline command-router suite passed before implementation (26 passed); focused parser tests passed at Task 1 HEAD (11 passed, 3 warnings); Bandit on tldw_Server_API/app/core/Chat_Macros reported no findings. Reviews: spec compliance passed after adding skills permission test; code-quality review passed after hardening defaults, alias collisions, duplicate non-repeated args, and prompt step support. Minor non-blocking note: parse_macro_args still trusts directly supplied hand-built arg_specs for alias uniqueness; intended call path uses validated MacroDefinition args.

Task 4 complete: added Chat Macros REST schemas, service dependency, CRUD/settings/validate/run/detail/cancel endpoints, core router registration, minimal-test router registration, and integration tests. Root-cause fix during implementation: the core router was defined but absent from MINIMAL_REQUIRED_ROUTER_NAMES, so /api/v1/chat/macros returned 404 under MINIMAL_TEST_APP until router_groups/minimal.py included chat_macros. Verification: chat macro API tests 3 passed; router minimal contract slice 10 passed/166 deselected; full tldw_Server_API/tests/Chat_Macros suite 43 passed; py_compile passed for new API modules; git diff --check passed; Bandit /tmp/bandit_chat_macros_task4_api.json had errors: [] and results: []. Touched files: tldw_Server_API/app/api/v1/schemas/chat_macros.py, tldw_Server_API/app/api/v1/API_Deps/Chat_Macros_Deps.py, tldw_Server_API/app/api/v1/endpoints/chat_macros.py, tldw_Server_API/app/api/v1/router_groups/core.py, tldw_Server_API/app/api/v1/router_groups/minimal.py, tldw_Server_API/tests/Chat_Macros/integration/test_chat_macros_api.py, Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md.

Task 4 review follow-up complete: reviewer found two issues, narrow error redaction and unresolved output_profile persistence. Added regression coverage for bearer/header/JSON-style secret redaction and missing-profile fallback; observed the new test fail on missing_profile persistence before patching. Endpoint now stores the resolved profile name and redacts bearer tokens, header-style x-api-key/api_key/token/password/secret values, quoted JSON secrets, and OpenAI-style sk-* keys. Post-fix verification: targeted regression test passed; API tests 3 passed; router minimal contract slice 10 passed/166 deselected; full tldw_Server_API/tests/Chat_Macros suite 43 passed; py_compile passed; Bandit /tmp/bandit_chat_macros_task4_api_reviewfix.json had errors: [] and results: [].

Task 4 implementation committed as 874ef6112d (feat: expose chat macros api).

Task 5 complete: added command_router.extract_slash_candidate() and reserved_core_command_names(), then wired /chat/completions to resolve enabled chat macros after core slash commands. /wrapup now creates a pending macro run and returns a chat-shaped assistant status message with chat_macro metadata, invalid macro args return a chat-visible validation error without creating a run, unknown slash candidates still call the provider, and stream=true macro starts return the same non-streaming JSON status response. Storage errors are reported with a generic public message. Touched files: tldw_Server_API/app/core/Chat/command_router.py, tldw_Server_API/app/api/v1/endpoints/chat.py, tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py, tldw_Server_API/tests/Chat_NEW/integration/test_chat_completions_api.py, Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md. Verification: initial focused red test failed as expected; focused macro tests passed; full Task 5 suite passed (43 passed, 9 skipped, 4 warnings); full tldw_Server_API/tests/Chat_Macros suite passed (43 passed, 4 warnings); py_compile passed for command_router.py and chat.py; git diff --check passed; Bandit /tmp/bandit_chat_macros_task5.json had errors: [] and results: [].

Task 5 review follow-up complete: reviewer found storage/discovery failures and early macro returns could fall through to the provider or bypass shared validation/rate/billing gates. Fixed by deferring macro run creation until after post-command validation, normal chat rate/RG handling, and billing precheck; macro storage/discovery/create failures now produce a generic chat-visible storage_error response without calling the provider, including unwrapped discovery exceptions. Added regressions for discovery failure no-provider-fallthrough and rate-limit denial no-run-created; the discovery regression now raises RuntimeError with a secret-like string to verify redaction/fail-closed behavior. Final reviewer re-check found no findings. Final verification after the last fix: review regressions 2 passed, 4 warnings; full Task 5 command/chat suite 45 passed, 9 skipped, 4 warnings; full tldw_Server_API/tests/Chat_Macros suite 43 passed, 4 warnings; py_compile passed for command_router.py and chat.py; git diff --check passed; Bandit /tmp/bandit_chat_macros_task5_final2.json had errors: [] and results: [].

Task 5 implementation committed as 969c771070 (feat: route chat macro slash commands).

Task 6 complete: added redacted macro context snapshots, ACP branch capability/fallback decisions, branch runner seams, and the ChatMacroExecutor for preset/custom-question planning, sync/include-branches enforcement, fail-early caps, branch concurrency/retry/cancellation, redacted failure persistence, merge rendering, final output persistence, and idempotent post-back claim metadata. Touched files: tldw_Server_API/app/core/Chat_Macros/context_snapshot.py, tldw_Server_API/app/core/Chat_Macros/acp_adapter.py, tldw_Server_API/app/core/Chat_Macros/branch_runner.py, tldw_Server_API/app/core/Chat_Macros/executor.py, tldw_Server_API/app/core/Chat_Macros/models.py, tldw_Server_API/app/core/Chat_Macros/output_profiles.py, tldw_Server_API/app/core/Chat_Macros/builtin/wrapup/MACRO.yaml, tldw_Server_API/tests/Chat_Macros/unit/test_macro_executor.py, tldw_Server_API/tests/Chat_Macros/unit/test_acp_adapter.py, Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md. Reviewer re-review found no findings; residual risk is real Jobs/LLM/post-back integration in later slices. Verification: focused executor/ACP suite 29 passed, 3 warnings; full Chat_Macros suite 72 passed, 4 warnings; command/chat regression slice 50 passed, 9 skipped, 5 warnings; compileall exit 0; git diff --check exit 0; Bandit /tmp/bandit_chat_macros_task6_final.json errors/results empty.

Task 7 complete: added Chat Macros Jobs enqueue/handler/cancellation/post-back integration, chat_macros_jobs_worker service, startup poller registration, and API/chat /wrapup enqueue wiring. Direct REST runs and /wrapup now require a Jobs manager instead of leaving unexecutable pending runs; cancellation handling tolerates incomplete job payloads while still finalizing Jobs cancellation. Touched files: tldw_Server_API/app/core/Chat_Macros/jobs.py, tldw_Server_API/app/services/chat_macros_jobs_worker.py, tldw_Server_API/app/services/startup_content_jobs_pollers.py, tldw_Server_API/app/api/v1/endpoints/chat_macros.py, tldw_Server_API/app/api/v1/endpoints/chat.py, tests/Chat_Macros/unit/test_macro_jobs.py, tests/Services/test_chat_macros_jobs_worker_startup.py, tests/Services/test_startup_content_jobs_pollers.py, tests/Chat_Macros/integration/test_chat_macros_api.py, tests/Chat_NEW/integration/test_chat_completions_api.py, Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md. Verification: focused enqueue regression 2 passed/5 warnings; missing Jobs and malformed cancellation regressions 3 passed/5 warnings; full Chat_Macros plus /wrapup Chat_NEW slice 85 passed/5 warnings; startup/service worker suite 37 passed/3 warnings; compileall exit 0; git diff --check exit 0; Bandit /tmp/bandit_chat_macros_task7.json errors/results empty. Residual risk: worker uses a conservative unavailable branch runner until real LLM/ACP branch execution is wired into the Jobs runtime.

Task 8 complete: added the minimal chat macro frontend service, settings manager at /settings/chat-macros, settings navigation entry, status card, lazy run-detail drawer, workspace macro status rendering, OpenAPI path guard entries, and tests. Added backend enabled-only update support for PUT /api/v1/chat/macros/{name} so the UI toggle works for built-in and user macros without full YAML replacement. Verification: frontend macro/nav suite 6 files / 44 tests passed; backend chat macro API integration suite 6 tests passed; bun run verify:openapi passed with existing reviewed exceptions only; compileall passed for touched backend files; git diff --check passed; Bandit /tmp/bandit_chat_macros_task8.json errors/results empty.

Task 9 verification/docs complete: updated tldw_Server_API/app/core/Chat_Macros/README.md for current v1 macro definitions, /wrapup options, API/settings, Jobs execution, UI behavior, and security notes. Final verification passed: focused backend suite 133 passed / 9 skipped / 5 warnings; frontend macro/nav suite 6 files / 44 tests passed; OpenAPI/config smoke 4 passed / 3 warnings; bun run verify:openapi passed with existing reviewed exceptions only; compileall passed for touched backend Chat_Macros/API/jobs scope; git diff --check passed; Bandit /tmp/bandit_chat_macros.json errors/results empty. Manual local-server smoke was not run because it needs a configured Jobs manager plus real LLM/ACP branch runner; automated TestClient/executor/Jobs/frontend tests cover the implemented contract and the conservative unavailable branch runner is documented.
2026-08-22: Rebase and review follow-up requested for PR #2618. Preserve pre-rebase head ef4e58d6029a796529d77660ebadf26925f5bacb, rebase onto latest origin/dev, evaluate all Qodo and existing inline review findings against current code, rerun focused backend/frontend verification and Bandit, then merge only after required checks and review threads are clean.
2026-08-23 PR review follow-up: addressed all existing Qodo comments plus independent review findings. Added route rate limiting, core-owned structured arg normalization, async offloading for macro filesystem/DB endpoint work, contextual exception logging, centralized exceptions, endpoint/schema docstrings, pytest markers/type hints, OpenAI-compatible SSE framing, slash discovery fallthrough, enforced branch timeouts, no-conversation post-back, registry no-op reads, production chat-native LLM branch runner, bounded/redacted chat and REST snapshots, per-branch context token accounting, and explicit background-only v1 execution semantics. Latest verification before final rebase: Chat Macros unit 84 passed; property/API 17 passed; slash integration 8 passed; frontend 34 passed; Ruff passed; frontend ESLint zero errors; Bandit must be rerun after final rebase. origin/dev advanced during review with PRs #2799/#2804, so a second rebase and post-rebase verification remain before push/merge.
Final post-rebase local verification (2026-08-23): 189 passed, 9 skipped across Chat_Macros, chat command/completions integration, and worker startup suites; frontend Chat Macros Vitest suite 37 passed; PR-owned Ruff scope passed; Bandit reported no findings; ESLint reported 0 errors and 5 pre-existing warnings in WorkspaceChatPanel/test; git diff --check passed. Rebase fixture fixes: corrected the Chat Macros startup worker parametrize row and updated sync token-threshold test for per-branch context accounting.
2026-08-23 final review-fix verification: addressed current Qodo, Gemini, and CodeRabbit findings across parser/model validation, context redaction and bounds, output rendering, user settings/storage recovery, async DB/filesystem execution, branch failure/retry/cancellation semantics, idempotent final post-back repair, Jobs worker lifecycle, API error behavior, and frontend error/cancel/settings states. Verification: backend affected suite 186 passed, 9 skipped; frontend affected suite 42 passed; Ruff passed; ESLint passed with no changed-file warnings; Bandit `/tmp/bandit_task_12126.json` reported no findings; git diff --check passed. Frontend typecheck still fails only on known baseline files outside this PR's changed scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
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
