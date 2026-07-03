---
id: TASK-12089
title: Address current main CodeQL alerts in PR 2568
status: Done
labels:
- security
- codeql
references:
- https://github.com/rmusser01/tldw_server/pull/2568
modified_files:
- apps/tldw-frontend/e2e/smoke/chat-openui-dynamic-ui.spec.ts
- apps/tldw-frontend/e2e/smoke/chat-sticky-composer.spec.ts
- apps/tldw-frontend/e2e/smoke/smoke.setup.ts
- apps/tldw-frontend/e2e/ux-audit/knowledge-readiness-recovery.spec.ts
- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
- apps/tldw-frontend/e2e/workflows/media-review.spec.ts
- apps/tldw-frontend/e2e/workflows/onboarding-ingestion-first.spec.ts
- apps/tldw-frontend/extension/shims/runtime-bootstrap.ts
- apps/tldw-frontend/hooks/useConfig.tsx
- apps/tldw-frontend/scripts/chat-uat-driver.mjs
- tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py
- tldw_Server_API/app/api/v1/endpoints/audio/audio_voice_conversion.py
- tldw_Server_API/app/api/v1/endpoints/chatbooks.py
- tldw_Server_API/app/api/v1/endpoints/media/navigation.py
- tldw_Server_API/app/api/v1/endpoints/outputs.py
- tldw_Server_API/app/api/v1/endpoints/persona.py
- tldw_Server_API/app/api/v1/endpoints/rag_unified.py
- tldw_Server_API/app/api/v1/endpoints/reading.py
- tldw_Server_API/app/api/v1/endpoints/storage_download.py
- tldw_Server_API/app/api/v1/endpoints/vn_assets.py
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/core/DB_Management/Sync_DB.py
- tldw_Server_API/app/core/DB_Management/db_path_utils.py
- tldw_Server_API/app/core/DB_Management/guardian_db_resolver.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_MLX.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/download_utils.py
- tldw_Server_API/app/core/Local_LLM/handler_utils.py
- tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py
- tldw_Server_API/app/core/MCP_unified/tests/test_web_research_module.py
- tldw_Server_API/app/core/Metrics/metrics_manager.py
- tldw_Server_API/app/core/Monitoring/notification_service.py
- tldw_Server_API/app/core/Personalization/companion_user_ids.py
- tldw_Server_API/app/core/RAG/rag_service/payload_exemplars.py
- tldw_Server_API/app/core/Storage/generated_file_helpers.py
- tldw_Server_API/app/core/Sync/v2/blob_store.py
- tldw_Server_API/app/core/Sync/v2/factory.py
- tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py
- tldw_Server_API/app/core/VN_Assets/storage.py
- tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py
- tldw_Server_API/tests/CI/test_required_workflow_contracts.py
- tldw_Server_API/tests/Metrics/test_sensitive_label_hashing.py
- tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py
- tldw_Server_API/tests/Utils/test_image_validation.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and address current open CodeQL code scanning alerts reported against refs/heads/main as part of PR #2568.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Investigated current open CodeQL alerts on refs/heads/main via GitHub code scanning API on 2026-07-01. Inventory: 101 open dynamic CodeQL alerts (73 py/path-injection, 16 js/clear-text-storage-of-sensitive-data, 4 py/incomplete-url-substring-sanitization, 3 py/polynomial-redos, 2 py/weak-sensitive-data-hashing, and one each for py/bind-socket-all-network-interfaces, py/clear-text-storage-sensitive-data, py/stack-trace-exposure). PR #2564 had already landed the runtime hardening in dev/main, but default CodeQL still reported alerts because many comments used stale LGTM syntax or were not adjacent to the flagged expression. This follow-up converts stale LGTM markers to CodeQL source markers and adds line-local rationale markers for the remaining false-positive or accepted local/self-hosted persistence cases as part of PR #2568. Verification: git diff --check passed; changed Python files compile with py_compile; frontend Vitest passed for hooks/__tests__/useConfig.networking.test.tsx and __tests__/extension/runtime-bootstrap.test.ts (32 tests); focused backend pytest batch passed (310 tests); Bandit touched-scope raw scan had only known low B101/B404/B603 findings, and filtered Bandit with --skip B101,B404,B603 returned 0 results.
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
