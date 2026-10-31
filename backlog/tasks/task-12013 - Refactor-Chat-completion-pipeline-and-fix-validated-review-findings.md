---
id: TASK-12013
title: Refactor Chat completion pipeline and fix validated review findings
status: Done
created_date: 2026-06-24 04:50
labels:
- chat
- refactor
- security
- review-fix
priority: High
references:
- tldw_Server_API/app/core/Chat
- tldw_Server_API/app/core/Chat/chat_service.py
- tldw_Server_API/app/core/Chat/README.md
- tldw_Server_API/app/core/Chat/REFACTORING_PLAN.md
- https://github.com/rmusser01/tldw_server/pull/2516
documentation:
- Docs/Design/2026-06-24-chat-completion-pipeline-refactor-design.md
- Docs/superpowers/plans/2026-06-24-chat-completion-pipeline-refactor.md
modified_files:
- Docs/Design/2026-06-24-chat-completion-pipeline-refactor-design.md
- Docs/superpowers/plans/2026-06-24-chat-completion-pipeline-refactor.md
- apps/tldw-frontend/lib/api/openapi.fingerprint.json
- tldw_Server_API/app/core/Chat/README.md
- tldw_Server_API/app/core/Chat/REFACTORING_PLAN.md
- tldw_Server_API/app/core/Chat/response_processor.py
- tldw_Server_API/app/core/Chat/moderation_pipeline.py
- tldw_Server_API/app/core/Chat/tool_execution_service.py
- tldw_Server_API/app/core/Chat/persistence_service.py
- tldw_Server_API/app/core/Chat/chat_logging.py
- tldw_Server_API/app/core/Chat/streaming_pipeline.py
- tldw_Server_API/app/core/Chat/completion_pipeline.py
- tldw_Server_API/app/core/Chat/chat_service.py
- tldw_Server_API/app/core/Chat/chat_orchestrator.py
- tldw_Server_API/app/core/Chat/command_authorization.py
- tldw_Server_API/app/core/Chat/command_router.py
- tldw_Server_API/app/core/Chat/document_generator.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_content.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_system_messages.py
- tldw_Server_API/tests/Chat/unit/test_streaming_utils.py
- tldw_Server_API/tests/Chat/unit/test_document_generator.py
- tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py
- tldw_Server_API/tests/Chat_NEW/integration/test_chat_command_audit.py
- tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py
updated_date: 2026-08-24 06:20
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a broad, compatibility-preserving refactor of the Chat completion pipeline. The first phase fixes validated review findings in `tldw_Server_API/app/core/Chat`: non-streaming multi-choice safety, sensitive logging, document prompt versioning, command authorization fail-closed behavior, and legacy history replacement safety. Later phases extract focused services from `chat_service.py` while preserving `/api/v1/chat/completions` API and response shapes except for intentional safety fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec is written, reviewed, and linked from this task before implementation planning begins.
- [x] #2 Validated findings have failing tests before production-code fixes.
- [x] #3 Non-streaming responses process moderation/redaction/structured validation across all returned choices or reject unsupported modes before provider calls.
- [x] #4 Chat logs no longer include raw user messages, system prompts, custom prompts, tool arguments, API keys, or assistant content.
- [x] #5 Document prompt saves support repeated versions while preserving exactly one active prompt per document type.
- [x] #6 Slash command dispatch enforces declared permissions fail-closed while preserving single-user owner/admin behavior.
- [x] #7 Legacy history replacement preserves the exported wrapper signature and avoids deleting existing messages before replacement can safely complete.
- [x] #8 `chat_service.py` remains a compatibility facade while focused modules own response processing, moderation, persistence, streaming orchestration, tool execution, command authorization, and safe logging.
- [x] #9 Public chat API response shapes and SSE event shapes remain stable except for intentional safety rejections.
- [x] #10 Targeted tests, relevant Chat regression tests, and Bandit over touched Chat scope are recorded before finalization.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write and commit the design spec in `Docs/Design/` after brainstorming approval.
2. After user review, create a detailed implementation plan before production edits.
3. Implement behavior fixes with TDD first.
4. Extract focused services behind stable `chat_service.py` facade.
5. Update Chat architecture documentation and Backlog task notes.
6. Run targeted tests, broader Chat regression checks as feasible, and Bandit on touched scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec drafted at `Docs/Design/2026-06-24-chat-completion-pipeline-refactor-design.md`. It captures the approved broad integrated refactor approach, public API compatibility constraint, validated findings, module boundaries, data flow, error handling, rollout stages, and test-first requirements.
Spec review refinements applied: locked `ChatCompletionPipeline` as the orchestration name, added a normalized command authorization context contract, clarified document prompt migration ownership as an idempotent local prompt-store repair unless implementation planning finds an existing owner, made missing-usage multi-choice accounting an intentional correction, and sharpened legacy history replacement transaction expectations.
Final spec clarification pass applied: non-streaming response processing order is safety/redaction before structured validation and persistence, user-facing command listings now use the same authorization decision by default, and logging hygiene explicitly excludes tool outputs and tool execution error details.
Implementation plan saved at `Docs/superpowers/plans/2026-06-24-chat-completion-pipeline-refactor.md`. The plan implements the approved integrated refactor in TDD stages: lock multi-choice safety with failing tests, extract response processing, moderation, persistence, tool execution, streaming, command authorization, and safe logging services, repair document prompt versioning, make legacy history replacement atomic, update Chat docs, then run targeted Chat tests and Bandit.
Progress update 2026-06-24: Tasks 1-4 of the Chat completion pipeline plan are complete. Implemented and reviewed multi-choice response processing, non-stream output moderation extraction, and local tool auto-exec multi-choice guards across non-stream/streaming direct, queued, and fallback paths. Latest Task 4 commits: `6f838f726` and `08fed328f`. Verification recorded locally: `test_chat_service_tool_autoexec.py` + `test_chat_service_streaming_tool_autoexec.py` = 40 passed; `test_chat_service_fallback.py` = 8 passed; Bandit on `chat_service.py` and `tool_execution_service.py` reported zero findings; `git diff --check` clean for the Task 4 scoped range.
Task 6 safe logging work started: scoped to chat_logging.py, chat_service.py, chat_orchestrator.py, and Chat system-message tests. Following TDD for summary helpers and preserving public API behavior.
Task 5 and Task 6 progress update 2026-06-24: First-choice persistence is extracted into `persistence_service.py` while preserving the existing persisted assistant/tool payload shape. Safe logging is extracted into `chat_logging.py` and wired through `chat_service.py` and `chat_orchestrator.py`; review follow-up removed raw exception metric labels, raw mapping-key summaries, and the `logging.exception` traceback leak. Verification recorded locally: `test_chat_service_system_messages.py` = 7 passed; focused Chat regression slice across system messages, content, tool autoexec, streaming tool autoexec, and fallback = 73 passed; Bandit on touched Chat logging/service/orchestrator scope reported zero findings. Task 6 spec and quality reviewers passed after follow-up commit `31dc9d163`.
Task 8 document prompt versioning completed in commit a2f598003. Added regression coverage for repeated prompt saves and legacy schema repair; replaced the legacy UNIQUE(document_type, is_active) table constraint with a partial unique index on active prompts only, preserving inactive history while enforcing one active prompt per document_type. Verification: targeted red test failed before fix with sqlite3.IntegrityError; focused prompt tests passed (2 passed); full document generator unit file passed (23 passed); Bandit on tldw_Server_API/app/core/Chat/document_generator.py reported 0 findings; focused review approved with no P0-P3 findings.
Task 7 command authorization completed across commits 3e961e15d, 601295701, 5893b0f92, 00cefdf1d, and e72e2901f. Centralized slash command authorization in command_authorization.py, wired both command dispatch and /chat/commands listing through the same fail-closed authorization decision, preserved direct orchestrator slash command compatibility in single-user owner mode, and updated docs/config references so CHAT_COMMANDS_REQUIRE_PERMISSIONS / require_permissions are documented as deprecated compatibility flags rather than security toggles. Verification: command router + command endpoint + injection focused tests passed earlier (35 passed); Bandit on touched command authorization scope reported 0 findings; final focused review approved with no P0-P3 findings after the guide snippet follow-up.
Task 9 legacy history replacement completed across commits 4c2c07489, 93a0ad866, and a74fd93dd. Added transaction-recording regression coverage, moved existing-conversation validation plus active message fetch/delete into the same transaction as replacement insertion, added rollback coverage for insert failure after soft-delete, and preserved empty-history target validation for existing conversations. Verification: initial regression failed with delete transaction [1] versus insert transaction [2]; focused tests passed after fixes; full chat history multi-image unit file passed (4 passed); Bandit on tldw_Server_API/app/core/Chat/chat_history.py reported 0 findings; final focused review approved with no P0-P3 findings.
Task 10 streaming pipeline extraction completed across commits 417f6e5611 and 698a17b415. Added `streaming_pipeline.py` as a thin assembly boundary around the existing streaming handler, routed the `chat_service.py` streaming call through it while preserving the `chat_service.create_streaming_response_with_timeout` monkeypatch surface, and added forwarding/default-preservation tests. Review finding addressed: unset optional wrapper fields now omit kwargs instead of overriding underlying factory defaults with `None`. Verification: default-preservation test failed before fix and passed after; `tldw_Server_API/tests/Chat/unit/test_streaming_utils.py` + `tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py` passed (43 passed, 1 skipped); Bandit on `tldw_Server_API/app/core/Chat/chat_service.py` and `tldw_Server_API/app/core/Chat/streaming_pipeline.py` reported 0 findings.
Task 11 completion pipeline coordinator completed in commit f44bfee1ad. Added `completion_pipeline.py` with `ChatCompletionPipeline`, renamed the non-stream body to `_execute_non_stream_call_impl`, kept the public `execute_non_stream_call` facade as a pipeline delegator, and routed streaming response assembly through the same default pipeline while preserving the existing stream factory monkeypatch path. Verification: new coordinator tests failed before implementation (missing module and facade bypass), then passed; final Task 11 checks passed for `test_chat_service_content.py`, `test_chat_service_tool_autoexec.py`, `test_streaming_utils.py`, and `test_chat_service_fallback.py` (91 passed, 1 skipped); Bandit on `completion_pipeline.py`, `chat_service.py`, and `streaming_pipeline.py` reported 0 findings.
Task 12 documentation update completed. Updated the Chat README module map and completion pipeline ownership section so `chat_service.py` is documented as the compatibility facade and focused modules own response processing, moderation, persistence, streaming assembly, tool execution, command authorization, and safe logging. Updated `REFACTORING_PLAN.md` with the 2026-06-24 integrated refactor status, validated findings fixed, and the compatibility note for intentional multi-choice local tool auto-execution rejection.
Task 13 verification update 2026-06-24: Wide Chat regression rerun after fixing two stale test harness assumptions. `test_chat_command_rbac_enforcement` now installs a non-admin request user before exercising deny/allow command audit events; root cause was that the default auth fixture is admin and correctly bypasses `_user_has_permission`. `test_chat_endpoint_auto_routing_uses_post_validation_tool_capabilities` now patches the endpoint's async skill-tool helper when present, while remaining compatible with the committed sync helper. Verification: command/auto-routing slice passed (`43 passed, 472 warnings`); full Chat regression passed except for the sandbox-only localhost mock-server setup errors (`1077 passed, 42 skipped, 9431 warnings, 2 errors`), and the two socket-backed tests passed when rerun with localhost bind permission (`2 passed, 115 warnings`). Bandit final scan over `app/core/Chat`, `endpoints/chat.py`, and `schemas/chat_commands_schemas.py` wrote `/tmp/bandit_chat_completion_pipeline_final.json`; it reported the same 7 low-severity historical findings in untouched files (`chat_exceptions.py`, `chat_helpers.py`, `tool_auto_exec.py`) and zero findings in the refactored modules/endpoints touched by this task.
2026-06-25 PR #2516 review follow-up: rebased branch on latest origin/dev (bbd7ffada2) and addressed validated Gemini/Qodo comments. Added user_prompt_configs table + round-trip tests, restored inspect.isawaitable command handler support with custom awaitable coverage, added new-module docstrings, made dict content extraction text-only, logged fail-closed permission backend failures with safe summaries, and replaced chat_orchestrator's Loguru-as-logging alias with direct logger calls. Verification: touched unit files 71 passed; broader PR slice 87 passed; command-router rerun 27 passed; Bandit over touched Chat files reported 0 findings.
2026-06-26 latest-dev follow-up: dev advanced again to ddf233a90e during CI polling. Rebased PR branch cleanly, adapted the new upstream context-integrity command-router test to use the authorized /skill context required by this branch's fail-closed command RBAC, reran command-router tests (28 passed), reran the full PR regression slice on latest dev (88 passed), and reran Bandit on touched Chat files (0 findings).
2026-07-09 latest-dev and PR review follow-up: rebased the three PR commits onto origin/dev at 20d96055e8. Resolved upstream overlap in chat_service.py by retaining current provider/model normalization and applying safe exception summaries; retained the current async skill-tool endpoint test behavior; combined upstream structured SSE parsing with the branch's multi-choice streaming guards. Re-queried PR #2516 and found no review or issue comments newer than the prior 2026-06-26 responses. Fresh verification on the rebased tree: primary PR regression slice 88 passed; streaming/provider conflict-sensitive slice 75 passed, 1 skipped; compileall over changed Chat/API production files passed; Bandit over changed Chat/API production files reported 0 findings in /tmp/bandit_chat_completion_pipeline_latest_dev.json.
2026-07-09 independent completion-review follow-up: review found that the original refactor commit had replaced current-dev bulk_generate behavior with an older implementation and reduced its regression test to a no-op. Restored the current-dev bulk generation contract, including provider/model/API-key/app-config fallback and overrides, DocumentType normalization, asyncio.to_thread execution, and llm_config initialization. TDD evidence: restored regression test failed before the fix with a missing api_key TypeError, then passed after the fix. Verification after remediation: single regression 1 passed; full document generator file 23 passed; primary PR regression slice 88 passed; compileall passed; Bandit on document_generator.py reported 0 findings in /tmp/bandit_chat_document_generator_followup.json.
2026-07-14 final latest-dev refresh: origin/dev advanced to 38bc70fd02 with no intervening Chat changes. Rebased all five PR commits cleanly onto that base. Fresh combined verification on the final rebased tree: 163 passed, 1 skipped across the primary PR, streaming, and provider/model conflict-sensitive tests; compileall over changed Chat/API production files passed; Bandit over changed Chat/API production files reported 0 findings in /tmp/bandit_chat_completion_pipeline_final_rebase.json.
2026-07-14 final base correction and refresh: the prior rebase note named ancestor 38bc70fd02, while the rebase reflog confirms the actual checkout tip was its descendant 83428eff33. origin/dev subsequently advanced to f05fe296db with no new Chat changes, and all six PR commits were replayed cleanly onto f05fe296db. Fresh final verification: 163 passed, 1 skipped across the combined PR/conflict-sensitive slice; compileall passed; Bandit over changed Chat/API production files reported 0 findings in /tmp/bandit_chat_completion_pipeline_f05_rebase.json.
2026-08-23 latest-dev rebase and compatibility repair: fetched origin/dev at 2c3589fa09, rebased all seven PR commits, and resolved Chat endpoint/service/test conflicts by retaining current-dev provider error sanitization, cancellation accounting, macro behavior, and replay-certified fallback rules while preserving the extracted completion services. Focused debugging found that the extracted multi-choice local tool guard was being normalized as an untrusted provider failure after the rebase. Added a canonical local exception plus an allowlisted private-provenance SSE frame so direct, fallback, and queued streaming paths return the intended bounded 400/error code without allowing provider-controlled frames to bypass sanitization. Updated stale tests to current-dev malformed-provider and replay-safety contracts and added utility-boundary tests for the trusted local frame allowlist/terminal behavior. Verification before publish: conflict-sensitive Chat suite 643 passed, 1 skipped; new boundary tests 2 passed; compileall passed; Bandit reported zero findings in all production files touched by the PR. A broad Chat scan retained the same seven historical low-severity findings in untouched chat_exceptions.py, chat_helpers.py, and tool_auto_exec.py.
2026-08-23 post-review verification: Ruff passed on all files modified during the latest-dev repair. The final conflict-sensitive suite, including the new trusted-frame boundary coverage, passed with 645 passed and 1 skipped. `git diff --check` remained clean; generated watchlist templates created by test startup were removed from the worktree.
2026-08-23 CI follow-up: GitHub's required `backend-required` aggregate failed only at the OpenAPI contract drift gate (checked-in 2012 paths/2941 schemas versus generated 2021/2958). Reproducing the check on the then-current origin/dev tip showed the base branch itself was stale; dev subsequently advanced to 4958cfed65 with additional scheduled-task APIs and still failed locally at 2028 paths/2965 schemas. Rebased the PR's eight commits cleanly onto 4958cfed65. This task will refresh the single committed OpenAPI fingerprint and regenerate/review the ignored frontend type artifacts so the required gate represents the actual latest-dev contract.
2026-08-23 final latest-dev remediation and pre-merge verification: rebased cleanly onto origin/dev 4958cfed65, regenerated the OpenAPI fingerprint with Python 3.12 to 2028 paths/2965 schemas (sha256 76dd350a574288b456c7e680a0130e8aaa770d20f8023ba4e91fbebd4ec0567c), regenerated the ignored frontend schema types, and passed the exact fingerprint drift check. Broad Ruff comparison against the same dev base identified four PR-only warnings: a dead stream-error assignment, an intentionally broad fail-closed authorization catch needing an explicit trust-boundary suppression, one import-spacing issue, and a missing test `Any` import; all were corrected and the differential lint result is now clean. Latest checks: compileall passed; Bandit reported zero findings in every production file changed by the PR; `git diff --check` passed; the post-fix 12-file Chat regression passed with 292 passed/1 skipped, following the earlier full conflict-sensitive run of 645 passed/1 skipped on this rebase. Implementation is ready for final GitHub CI, Qodo review of the published head, and merge.
2026-08-23 final merge-window base refresh: origin/dev advanced twice after the prior verification, first to 2ebf14c145 (Web Scraping phase 4C) and then to 1885fe8ee2 (Claims analytics exports). Both rebases were clean for Chat code. The Claims merge overlapped the OpenAPI fingerprint, so the conflict was resolved by regenerating the combined latest-dev contract under Python 3.12 instead of choosing either snapshot. Final combined contract remains 2028 paths/2965 schemas with sha256 8aa01ac97bebf2b3ae48467a4aedc3a10ee1da424413609f9a3d3eb5275b5de3; ignored frontend schema types were regenerated and the exact fingerprint drift check passed. Chat regression on the immediately preceding Web Scraping base remained 292 passed/1 skipped and touched-scope Bandit remained 0 findings; the subsequent Claims delta does not touch Chat code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the integrated Chat completion pipeline refactor and validated review fixes. `chat_service.py` now acts as a compatibility facade over focused services for response processing, moderation, persistence, streaming assembly, tool execution, command authorization, and safe logging. Validated findings for multi-choice safety, sensitive logging, document prompt versioning, slash command authorization, and legacy history replacement were fixed with regression coverage and architecture docs. Final verification recorded targeted Chat slices, the wide Chat regression, localhost-bind reruns for sandbox-limited mock-server tests, and Bandit results with only pre-existing low-severity findings in untouched files.
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
