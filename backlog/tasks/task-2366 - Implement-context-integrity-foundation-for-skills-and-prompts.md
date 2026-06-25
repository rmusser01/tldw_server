---
id: TASK-2366
title: Implement context integrity foundation for skills and prompts
status: In Progress
labels:
- security
- skills
- prompts
- implementation
priority: high
references:
- TASK-2363
- TASK-2365
modified_files:
- tldw_Server_API/app/services/startup_context_integrity.py
- tldw_Server_API/app/services/lifespan_startup_sequence.py
- tldw_Server_API/app/services/lifespan_shutdown_sequence.py
- tldw_Server_API/tests/Services/test_startup_context_integrity.py
- tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py
- tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py
- tldw_Server_API/app/core/Skills/skills_service.py
- tldw_Server_API/app/core/Skills/context_integration.py
- tldw_Server_API/app/core/Chat/command_router.py
- tldw_Server_API/app/api/v1/endpoints/skills.py
- tldw_Server_API/tests/Skills/unit/test_skills_service.py
- tldw_Server_API/tests/Skills/integration/test_skills_api.py
- tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py
- tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py
- tldw_Server_API/app/core/Utils/prompt_loader.py
- tldw_Server_API/tests/Utils/test_prompt_loader_paths.py
- tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py
- tldw_Server_API/app/api/v1/endpoints/admin/context_integrity.py
- tldw_Server_API/app/api/v1/endpoints/admin/__init__.py
- tldw_Server_API/app/api/v1/schemas/admin_schemas.py
- tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the reviewed context integrity foundation for skill and prompt files, including manifests, startup verification, resolver enforcement, integration chokepoints, admin reporting, tests, docs, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-25-context-integrity-foundation-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 5 slice implemented the startup verification producer and lifecycle wiring. Startup now inventories prompt files, env prompt overrides, and discovered/test-injected user skill roots through rich InventoryResult APIs; loads optional signed HMAC manifests from environment; distinguishes no manifest from valid empty manifests; attaches ContextIntegrityBootState and ContextIntegrityResolver to app state; sets the global resolver; registers context_integrity.* startup warnings; and clears app/global resolver state during lifespan shutdown.

Task 5 review follow-up fixed read-only skill-root discovery: Context Integrity now resolves the user database base path for discovery without calling DatabasePaths.get_user_db_base_dir(), avoiding the helper's directory creation side effect. A regression test verifies a missing USER_DB_BASE_DIR is not created during discovery.

Verification recorded for Task 5:
- RED run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_startup_context_integrity.py tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py -v` failed as expected before implementation with missing startup_context_integrity imports and shutdown resolver cleanup assertion failure.
- Focused Services suite after follow-up fix: `.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_startup_context_integrity.py tldw_Server_API/tests/Services/test_lifespan_startup_sequence.py tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py -q` passed with `13 passed, 6 warnings`.
- Context_Integrity unit suite: `.venv/bin/python -m pytest tldw_Server_API/tests/Context_Integrity/unit -q` passed with `116 passed, 6 warnings`.
- Bandit: `.venv/bin/python -m bandit -r tldw_Server_API/app/services/startup_context_integrity.py tldw_Server_API/app/services/lifespan_startup_sequence.py tldw_Server_API/app/services/lifespan_shutdown_sequence.py -f json -o /tmp/bandit_context_integrity_task5_manual_review.json` exited 0 with zero findings.
- Formatter/whitespace: `git diff --check HEAD~1..HEAD` clean.

Task 6 slice enforced Context Integrity in Skills runtime paths. SkillsService now uses the current global or injected resolver, rehashes prompt-bearing skill files at use time, blocks direct reads/exports/execution for quarantined or live-edited skills, filters blocked skills from model context/discovery, returns write-response snapshots without approving newly written skills, and rejects symlinked skill directories or prompt-bearing symlink paths. The REST API maps blocked get/export/execute requests to content-free HTTP 423 responses; MCP skill tool calls and chat slash-command execution return stable content-free integrity errors.

Task 6 manual review follow-up removed an unsafe degraded-resolver fallback. Default SkillsService construction now honors degraded global resolver state and fails closed; non-integrity skills API tests clear the global resolver explicitly in their fixtures.

Verification recorded for Task 6:
- RED run for review regressions: `.venv/bin/python -m pytest tldw_Server_API/tests/Skills/unit/test_skills_service.py::TestSkillsService::test_degraded_global_resolver_blocks_default_service tldw_Server_API/tests/Skills/unit/test_skills_service.py::TestSkillsService::test_symlinked_skill_directory_is_not_read_without_resolver -q` failed before the correction with both tests failing.
- Focused Task 6 suite: `.venv/bin/python -m pytest tldw_Server_API/tests/Skills/unit/test_skills_service.py tldw_Server_API/tests/Skills/integration/test_skills_api.py tldw_Server_API/tests/Skills/integration/test_skill_mcp_integration.py tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py -q` passed with `133 passed, 6 warnings`.
- Bandit: `.venv/bin/python -m bandit -r tldw_Server_API/app/core/Skills/skills_service.py tldw_Server_API/app/core/Skills/context_integration.py tldw_Server_API/app/core/Chat/command_router.py tldw_Server_API/app/api/v1/endpoints/skills.py -f json -o /tmp/bandit_context_integrity_task6.json` exited 0 with zero findings.
- Formatter/whitespace: `.venv/bin/python -m black ...` completed; `git diff --check` clean.

Task 7 slice enforced Context Integrity in the prompt loader. Markdown, YAML, JSON, and `TLDW_PROMPT_FILE_*` override reads now flow through `_read_prompt_file_text()`, which reads bytes once, computes the canonical digest over those exact bytes using the same asset IDs and metadata as inventory, asks the global resolver, and only then decodes the same bytes for parsing. Quarantined files, unapproved files under enforcement, live edits after boot, invalid UTF-8, and symlinked prompt files fail closed without logging prompt content.

Verification recorded for Task 7:
- RED run: `.venv/bin/python -m pytest tldw_Server_API/tests/Utils/test_prompt_loader_paths.py -q` failed before implementation with the three new prompt-loader integrity tests failing.
- Focused Task 7 suite: `.venv/bin/python -m pytest tldw_Server_API/tests/Utils/test_prompt_loader_paths.py tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py -q` passed with `10 passed, 6 warnings`.
- Bandit: `.venv/bin/python -m bandit -r tldw_Server_API/app/core/Utils/prompt_loader.py -f json -o /tmp/bandit_context_integrity_task7.json` exited 0 with zero findings.
- Formatter/whitespace: `.venv/bin/python -m black ...` completed; `git diff --check` clean.

Task 8 slice added admin inspection for current-process Context Integrity state. The admin router now exposes `GET /api/v1/admin/context-integrity`, returning boot mode, degraded state, manifest sequence/digest, and content-free finding metadata from `app.state.context_integrity_boot_state`; non-admin callers receive the existing admin-router 403. Response schemas live beside startup-warning admin schemas.

Verification recorded for Task 8:
- RED run: `.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py -q` failed before implementation with 404 responses.
- Focused Task 8 suite: `.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ_SQLite/test_admin_context_integrity_sqlite.py tldw_Server_API/tests/AuthNZ_SQLite/test_admin_startup_warnings_sqlite.py -q` passed with `5 passed, 7 warnings`.
- Bandit: `.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/admin/context_integrity.py tldw_Server_API/app/api/v1/schemas/admin_schemas.py -f json -o /tmp/bandit_context_integrity_task8.json` exited 0 with zero findings.
- Formatter/whitespace: `.venv/bin/python -m black ...` completed for new endpoint/test files; unrelated schema formatting was reverted; `git diff --check` clean.
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
