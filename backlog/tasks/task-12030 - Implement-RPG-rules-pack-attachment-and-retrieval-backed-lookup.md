---
id: TASK-12030
title: Implement RPG rules-pack attachment and retrieval-backed lookup
status: In Progress
created_date: 2026-06-25 23:29
dependencies:
- TASK-12029
labels:
- rpg
- ttrpg
- rag
- backend
- implementation
priority: high
documentation:
- Docs/superpowers/specs/2026-06-25-rpg-rules-pack-attachment-retrieval-design.md
- Docs/superpowers/plans/2026-06-25-rpg-rules-pack-attachment-retrieval-implementation-plan.md
updated_date: 2026-06-26 03:02
modified_files:
- tldw_Server_API/app/core/RPG/rules/refs.py
- tldw_Server_API/app/core/DB_Management/RPG_DB.py
- tldw_Server_API/app/core/RPG/service.py
- tldw_Server_API/tests/RPG/test_rpg_rules_refs.py
- tldw_Server_API/tests/RPG/test_rpg_db.py
- tldw_Server_API/tests/RPG/test_rpg_service.py
- tldw_Server_API/app/api/v1/schemas/rpg_schemas.py
- tldw_Server_API/app/api/v1/endpoints/rpg.py
- tldw_Server_API/tests/RPG/test_rpg_api.py
- tldw_Server_API/Config_Files/privilege_catalog.yaml
- tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json
- Docs/superpowers/plans/2026-06-25-rpg-rules-pack-attachment-retrieval-implementation-plan.md
- tldw_Server_API/app/core/RPG/rules/retrieval.py
- tldw_Server_API/app/core/RPG/rules/source_validation.py
- tldw_Server_API/app/core/RPG/rules/content_packs.py
- tldw_Server_API/app/core/RPG/rules/lookup.py
- tldw_Server_API/app/core/RPG/context.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py
- tldw_Server_API/tests/RPG/test_rpg_rules_retrieval.py
- tldw_Server_API/tests/RPG/test_rpg_rules_context.py
- tldw_Server_API/tests/RPG/test_rpg_mcp_module.py
- tldw_Server_API/app/core/RPG/rules/answering.py
- tldw_Server_API/tests/RPG/test_rpg_rules_answering.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved RPG rules-pack attachment feature from TASK-12029. Campaigns and sessions should attach user-owned media items or media collections as rules references, then use scoped retrieval to augment RPG rules lookup and context building with optional grounded answer mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Campaigns and sessions can list and replace normalized media_item/media_collection rules-pack refs with whole-list writes, expected_version checks, idempotency replay, and server-owned timestamps.
- [x] #2 New REST endpoints expose campaign/session rules-pack ref list and replace operations with RPG permissions plus media.read requirements.
- [x] #3 Session creation copies campaign refs by default while explicit session refs can diverge from campaign refs.
- [x] #4 Rules lookup blends user-provided scoped retrieval snippets with bundled citation-only references, reports diagnostics, and never falls back to broad RAG or web search.
- [x] #5 Answer mode generates grounded answers only from retrieved snippets using the existing async chat service and returns citation IDs limited to lookup evidence.
- [x] #6 Session context building includes lookup-mode evidence within existing bounds and never invokes answer generation.
- [ ] #7 MCP tools expose the same ref-management and lookup semantics as REST.
- [x] #8 Focused RPG, API, MCP, privilege catalog, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-25: Began subagent-driven implementation from Docs/superpowers/plans/2026-06-25-rpg-rules-pack-attachment-retrieval-implementation-plan.md. Worktree verified clean on branch codex/rpg-runtime before runtime code edits.
2026-06-25: Completed implementation plan Task 1. Commits: 607ec4fe5c (rules-pack ref model and repository replacement) and 1b3ffa4cdb (strict enabled validation plus session idempotency mismatch coverage). RED checks showed missing refs module/repository methods, then strict enabled failures; GREEN verification: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_refs.py tldw_Server_API/tests/RPG/test_rpg_db.py -v` -> 38 passed, 88 existing warnings. Spec compliance reviewer approved Task 1. Code quality reviewer found one minor enabled-coercion hardening issue; follow-up fixed it and re-review approved. Worker-reported Bandit on touched Task 1 scope had no findings.
2026-06-25: Completed implementation plan Task 2. Commits: d86b0b6156 (service source validation and session copy semantics), 6b5d4e10be (replacement/create-session replay-before-validation fixes), and 62278af3e0 (explicit session rules refs replay-before-validation). GREEN verification: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_refs.py tldw_Server_API/tests/RPG/test_rpg_db.py tldw_Server_API/tests/RPG/test_rpg_service.py -q` -> 57 passed, 126 existing warnings. Spec compliance approved after final fix. Code quality review initially found idempotency replay issues; follow-up fixes resolved them and final review approved. Worker-reported Bandit on touched Task 2 scope had no findings.
2026-06-25: Completed implementation plan Task 3. Added REST schemas for rules-pack refs and lookup mode/provider/model options; added campaign/session rules-pack list and whole-list replace REST endpoints; wired REST RPGService construction with authenticated user's Media DB and Collections DB plus endpoint-layer rules source validator; added media.read to RPG lookup/ref endpoint permission contracts; regenerated privilege route registry snapshot with Helper_Scripts/update_privilege_registry_snapshot.py. RED verification: focused API/catalog pytest failed before implementation for missing constants/routes and lookup mode schema. GREEN verification: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_api.py tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v` -> 21 passed, 64 existing warnings after snapshot regeneration. Additional manual validator check: enabled missing media ref returned 400 `rules_pack_source_unreadable`. Bandit: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/rpg.py tldw_Server_API/app/api/v1/schemas/rpg_schemas.py tldw_Server_API/tests/RPG/test_rpg_api.py -f json -o /tmp/bandit_task12030_task3.json` -> zero findings.
2026-06-25: Task 3 post-review hardening completed. Spec re-review initially found collection refs could return ready media IDs without re-checking unreadable/deleted/trash/missing rows; fixed REST validator to re-read ready media IDs through Media DB and owner/client guard. Code quality review found broad service media DB dependency, permissive enabled coercion, and limited runtime media.read denial coverage; split base RPG service from rules-source service, changed RPGRulesPackRefInput.enabled to StrictBool, added non-boolean enabled rejection and rules-pack missing media.read 403 tests. Final focused verification: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_api.py tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v` -> 27 passed, 76 existing warnings. Bandit: `/tmp/bandit_task12030_task3.json` -> zero findings. Final spec re-review passed; quality re-review had no P0/P1 blockers after fixes, with remaining batching concern deferred to retrieval implementation where batched media reads belong.
2026-06-25: Completed implementation plan Task 4. Added scoped RPG rules retrieval adapter over RulesPackSourceValidator plus the existing RAG retrieval executor with media_db-only sources and allowed_media_ids; extended lookup result dataclasses for user_provided and bundled_citation items; wired async lookup through RPGService, REST lookup/context handlers, and existing MCP async call sites without adding Task 6 tools. Lookup keeps bundled citations, puts retrieved user snippets first, reports linked/enabled/ready/retrieved/bundled/skipped/no-fallback diagnostics, and does not implement grounded answer generation. RED verification: focused retrieval/context pytest failed during collection because rules.retrieval and RuleLookupCitation were missing. GREEN verification: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_retrieval.py tldw_Server_API/tests/RPG/test_rpg_rules_context.py tldw_Server_API/tests/RPG/test_rpg_service.py -v` -> 35 passed, 82 existing warnings. API async call-site verification: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_api.py tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v` -> 27 passed, 76 existing warnings. Bandit: `/tmp/bandit_task12030_task4.json` -> 0 findings.
2026-06-25: Task 4 post-review hardening completed. Addressed quality review findings by sharing rules source validation between REST and MCP, wiring MCP rules lookup/context to attached media refs when `context.db_paths.media` is present, filtering retrieval executor results back to validated ready_media_ids, and redacting unexpected retrieval exception details from both diagnostics and warning logs. Added regression coverage for out-of-scope retrieval documents, public/log redaction, MCP attached-media lookup, MCP answer-status behavior, and non-rules MCP tools avoiding media DB opens. Spec re-review previously approved after answer-status fix; code quality re-review approved after lazy MCP media binding. Final verification: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_retrieval.py tldw_Server_API/tests/RPG/test_rpg_rules_context.py tldw_Server_API/tests/RPG/test_rpg_api.py tldw_Server_API/tests/RPG/test_rpg_mcp_module.py tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v` -> 64 passed, 176 existing warnings. Bandit: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/RPG tldw_Server_API/app/api/v1/endpoints/rpg.py tldw_Server_API/app/api/v1/schemas/rpg_schemas.py tldw_Server_API/app/core/MCP_unified/modules/implementations/rpg_module.py -f json -o /tmp/bandit_task12030_task4.json` -> 0 findings, 0 errors.
2026-06-25: Completed implementation plan Task 5. Added grounded answer generation through the existing async chat call path, lookup wiring for `mode="answer"`, context diagnostics/fallback behavior, and REST answer option wiring. Post-review hardening requires `chat.completions` permission plus token-scope, LLM-budget, and chat rate-limit guard checks before REST answer generation; malformed/non-JSON model output now returns `generation_error` without fabricated citations; unexpected answer-generator failures are sanitized to `generation_error` while preserving lookup evidence. RED checks covered missing answer module, missing context diagnostics/fallback, malformed answer grounding, unexpected generator exceptions, and missing answer-mode chat permission. GREEN verification: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_answering.py tldw_Server_API/tests/RPG/test_rpg_rules_context.py tldw_Server_API/tests/RPG/test_rpg_api.py -v` -> 54 passed, 133 existing warnings. Broader regression: `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/RPG/test_rpg_rules_answering.py tldw_Server_API/tests/RPG/test_rpg_rules_retrieval.py tldw_Server_API/tests/RPG/test_rpg_rules_context.py tldw_Server_API/tests/RPG/test_rpg_api.py tldw_Server_API/tests/RPG/test_rpg_mcp_module.py tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py -v` -> 81 passed, 213 existing warnings. Bandit: `/tmp/bandit_task12030_task5.json` -> 0 findings, 0 errors. Spec and quality re-reviews reported no blockers after hardening.
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
- [ ] #7 Implementation plan tasks completed or consciously split into follow-up tasks with links.
- [ ] #8 Focused pytest commands recorded with results.
- [ ] #9 Privilege route catalog check recorded when endpoint metadata changes.
- [ ] #10 Bandit JSON report path and result recorded for touched Python scope.
- [ ] #11 Final summary explains what changed and why the chosen integration boundaries were used.
<!-- DOD:END -->
