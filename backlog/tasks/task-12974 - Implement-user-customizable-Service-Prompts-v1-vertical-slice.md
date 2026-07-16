---
id: TASK-12974
title: Implement user-customizable Service Prompts v1 vertical slice
status: In Progress
labels:
- service-prompts
- implementation
references:
- TASK-12973
documentation:
- Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
- Docs/superpowers/plans/2026-07-15-user-customizable-service-prompts-v1.md
priority: high
modified_files:
- apps/packages/ui/src/utils/__fixtures__/service-prompt-rendering.json
- tldw_Server_API/app/core/Prompt_Management/service_prompts.py
- tldw_Server_API/tests/Prompt_Management/test_service_prompts.py
- tldw_Server_API/app/core/DB_Management/Prompts_DB.py
- tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py
- tldw_Server_API/app/api/v1/schemas/service_prompt_schemas.py
- tldw_Server_API/app/api/v1/endpoints/service_prompts.py
- tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py
- tldw_Server_API/app/api/v1/endpoints/translate.py
- tldw_Server_API/tests/Translation/test_translate_service_prompt.py
- tldw_Server_API/tests/Translation/test_translate_endpoint_error_mapping.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/router_groups/minimal.py
- tldw_Server_API/tests/Services/test_router_groups_contract.py
- apps/tldw-frontend/lib/api/openapi.fingerprint.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved four-definition Service Prompts v1 plan across the backend, shared WebUI/browser-extension Settings surface, legacy migration, and named runtime consumers using TDD and staged review gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A static four-definition registry, strict renderer, one-table owner-scoped override store, and four Service Prompt API operations implement the approved contract.
- [ ] #2 Workflow prompts Settings is shared by WebUI and extension, supports scoped migration/save/reset/conflict/corrupt states, and keeps Prompt Library navigation and backup disclosure accurate.
- [ ] #3 All named RAG, rewrite, web-search, Compare, legacy Sidepanel, and synchronous Translation consumers use immutable server-backed request snapshots with preserved no-override semantics.
- [ ] #4 Focused backend/frontend/E2E verification, OpenAPI checks, lint/type/build checks, Bandit, and git hygiene pass; CI shards remain deliberately unchanged.
- [ ] #5 Each implementation task passes independent specification and code-quality review, followed by a final whole-feature review.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Tasks 1-7 from Docs/superpowers/plans/2026-07-15-user-customizable-service-prompts-v1.md sequentially. Use red-green-refactor for each behavior, commit each dependency stage, and complete spec-compliance then code-quality review before advancing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Execution started in isolated worktree /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/service-prompts-v1-plan on branch codex/service-prompts-v1-plan. .worktrees is gitignored; linked the existing project .venv and installed the existing Bun workspace dependencies without tracked lockfile changes. Focused pre-change baseline passed: 196 backend tests across Prompts DB, Translation error mapping, and router-group contracts; 10 frontend tests across Settings routing/index and chat-pipeline error recovery. CI shard edits remain explicitly out of scope per requester.
Task 1 registry/validator/renderer/resolver TDD evidence (2026-07-16): RED command `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Prompt_Management/test_service_prompts.py -v` failed during collection with the expected missing `Prompt_Management.service_prompts` ModuleNotFoundError. The same command is GREEN after implementation: 47 passed, 2 existing warnings, 0 failures. Independent review identified ignored-fixture staging and two JSON decoder corruption edges; the fixture is force-staged, and regression tests first reproduced plain ValueError (over-limit integer) and RecursionError (excessive nesting) before both were normalized to revision-only ServicePromptCorruptOverride errors. Canonical default byte lengths/SHA-256 values match the five source strings. Ruff check/format checks pass, and Bandit reports zero findings for service_prompts.py. Task 1 deliberately calls the narrow get_service_prompt_override interface that Task 2 will add; no direct SQL or Prompts_DB.py changes were made.
Task 1 review gates: independent specification review approved the actual a117c3a6a6 diff with no missing or extra requirements. Separate code-quality review found no Critical, Important, or Minor issues and approved correctness, Python 3.10 compatibility, security/error behavior, test quality, maintainability, and the absence of removable speculative complexity.
Task 2 v6 Service Prompt override persistence TDD/review evidence (2026-07-16): schema RED used the exact focused command `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py -k "schema or service_prompt" -v`; before production changes it produced 2 failed/1 passed (fresh DB remained schema 5, and reopening an actual v5 DB at target 6 failed with `Migration needed from 5 to 6, but no path defined`). Store RED with all behavior tests then produced 16 failed/1 passed, with get/save/reset cases failing on the missing narrow methods and the same schema gaps. Minimal implementation adds only the per-user `ServicePromptOverrides` v6 table, frozen raw row, revision conflict, raw read, atomic BEGIN IMMEDIATE save/CAS, and content-independent reset; no history/events/approval/second DB/repository abstraction or CI shard change. Independent review reproduced two boundary bugs: commit failures escaped raw sqlite errors (follow-up RED 2/2 failed with synthetic OperationalError), and reset materialized corrupt authored text. Transaction entry/body/commit are now wrapped in content-free DatabaseError, transaction/rollback logs are type-only, and reset reads only definition_id/revision. The approved-design safety invariant deliberately resolves the plan sketch reset return annotation to `None`: an undecodable TEXT regression first failed under DELETE RETURNING, then passed after conditional DELETE stopped reading parts_json. Final exact focused GREEN: 20 passed/17 deselected/2 warnings. Full DB file: 37 passed/2 warnings. Task 1 resolver regression: 47 passed/2 warnings. Ruff check passes on both Python files; every changed range passes Ruff format. Whole-file Ruff format remains pre-existing legacy debt on both files (the untouched HEAD versions also exit 1), so no unrelated mass reformat was applied. Bandit on Prompts_DB.py reports 0 findings; git diff --check passes. Independent re-review: READY, no Critical or Important findings; the same-connection defensive insert-race proxy fidelity note is Minor/non-blocking and accepted because BEGIN IMMEDIATE makes a faithful external race test artificial and flaky.
Task 2 spec-review coverage fix (2026-07-16): added deterministic BEGIN IMMEDIATE failure coverage for save/reset and actual Loguru capture for the trigger-driven statement failure. The new targeted command `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py -k "begin_immediate_failure or failed_save_rolls_back" -v` passed 3/3 against unchanged production code, confirming transaction-entry failures already map to fixed content-free DatabaseError messages, preserve the stored row, and emit type-only rollback logs without prompt/body/DB-path sentinels. Fresh gates: focused schema/service_prompt 22 passed/17 deselected/2 warnings; full DB file 39 passed/2 warnings; Task 1 file 47 passed/2 warnings; Ruff clean; Bandit 0 findings; git diff --check clean. Independent gap-fix review found no correctness/spec issues. Prompts_DB.py was not changed.
Task 2 code-quality atomicity fix (2026-07-16): strict file-backed RED `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py -k "rollback_failure_retires_connection" -v` produced 2 failed/39 deselected: after commit and rollback both raised OperationalError, the cached connection remained reusable and a later commit persisted the rejected save or finalized the rejected reset. Root cause: transaction() logged rollback failure but left the still-open captured connection in self._local.conn; its SELECT 1 health probe succeeded, so later callers reused the poisoned transaction. The minimal fix clears ownership of that exact connection, closes the captured handle, preserves detachment if close fails, and logs exception types only. Targeted GREEN: 2 passed/39 deselected. Fresh gates: focused schema/service_prompt 24 passed/17 deselected/2 warnings; full DB file 41 passed/2 warnings; Task 1 file 47 passed/2 warnings; Ruff clean; Bandit 0 findings; git diff --check clean. Atomicity re-review requested.
Task 2 rollback-failure atomicity re-review: APPROVED with no remaining issue.
Task 2 corrupt-override revision safety fix (2026-07-16): SQLite behavior inspection confirmed selecting undecodable TEXT raises OperationalError before the row/revision reaches Python, while the same atomic row selected with CAST(parts_json AS BLOB) returns raw bytes plus revision. json.loads accepts valid UTF-8 bytes and invalid bytes raise UnicodeDecodeError, already covered by the resolver's ValueError normalization. Strict file-backed RED `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Prompt_Management/test_prompts_db_v2.py -k "undecodable_text_preserves_revision_for_resolver_and_reset" -v` produced 1 failed/41 deselected at the raw getter with content-free generic DatabaseError and no discoverable revision. Minimal fix uses one SELECT for definition_id, BLOB-cast content, and revision; valid UTF-8 decodes back to str, invalid UTF-8 remains bytes, and ServicePromptOverrideRow.parts_json is honestly typed str | bytes. The resolver required no change. Targeted GREEN: 1 passed/41 deselected/2 warnings, with real getter -> ServicePromptCorruptOverride(revision) -> conditional reset and persisted deletion. Fresh gates: focused schema/service_prompt 25 passed/17 deselected/2 warnings; full DB file 42 passed/2 warnings; Task 1 file 47 passed/2 warnings; Ruff clean; Bandit 0 findings; git diff --check clean.
Task 2 final controller review gates: after the committed rollback-failure and corrupt-revision fixes, the same independent specification reviewer approved the complete a117c3a6a6..90d83b9b31 range. The independent quality reviewer found no remaining Critical, Important, or Minor issues and approved atomicity, migration, CAS/reset semantics, invalid-UTF-8 revision recovery, security/log safety, deterministic tests, and the lean one-table design.
Task 3 API/Translation TDD evidence (2026-07-16): API RED `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py -v` failed 1/1 because GET /api/v1/service-prompts returned the expected pre-route 404. Router RED selected 111 cases and produced exactly 2 expected missing-/service-prompts failures while 109 passed. Translation RED produced 6/6 expected TypeErrors because translate_text had no PromptsDatabase dependency. The minimal implementation adds separate catalog/detail/update schemas, a thin authenticated current-user API with read/write API-key scopes, locked content-free domain envelopes, no-store responses, always-on router registration, raw conditional reset without a post-delete content read, and Translation resolution/rendering before the provider error boundary. Final focused backend slice: 289 passed, 23 existing warnings, 0 failures. OpenAPI generation completed and fingerprint check reports OK. Ruff passes all new/touched focused production and test files; the large existing router contract also passes F/I checks, with unrelated pre-existing whole-file B/C/E501 baseline left untouched. Bandit scanned all five touched production Python files with 0 findings. git diff --check passes; CI shards remain unchanged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
