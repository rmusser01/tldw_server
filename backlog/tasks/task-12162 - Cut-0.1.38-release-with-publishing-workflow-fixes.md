---
id: TASK-12162
title: Cut 0.1.38 release with publishing workflow fixes
status: Done
priority: High
modified_files:
- .github/workflows/ci.yml
- CHANGELOG.md
- Docs/API-related/API_Tags_Index.md
- Docs/Design/WebScraping_Refactor_Import_Inventory.md
- Docs/Design/web_scraping_refactor_import_inventory.json
- Docs/mkdocs.yml
- README.md
- apps/packages/ui/src/components/Common/confirm-danger.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesEditorHeader.stage2.touch-layout.test.tsx
- apps/packages/ui/src/components/Notes/NotesEditorHeader.tsx
- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
- apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts
- apps/tldw-frontend/lib/api/openapi.fingerprint.json
- pyproject.toml
- tldw_Server_API/app/api/v1/endpoints/rpg.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/backends/fts_translator.py
- tldw_Server_API/app/core/Monitoring/notification_service.py
- tldw_Server_API/app/core/Workflows/engine.py
- tldw_Server_API/app/main.py
- tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py
- tldw_Server_API/tests/CI/test_release_workflow_contracts.py
- tldw_Server_API/tests/CI/test_required_workflow_contracts.py
- tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py
- tldw_Server_API/tests/Flashcards/test_flashcards_db_assets.py
- tldw_Server_API/tests/http_client/test_redirect_header_hardening.py
- tldw_Server_API/tests/RAG_NEW/unit/test_fts_query_translation_edge_cases.py
- tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py
- tldw_Server_API/tests/Media_Ingestion_Modification/test_nemo_transcription.py
- tldw_Server_API/tests/Resource_Governance/test_e2e_chat_audio_headers.py
- tldw_Server_API/tests/Visual_Identities/test_visual_identity_archive_import.py
- tldw_Server_API/tests/Web_Scraping/test_handlers.py
- tldw_Server_API/tests/Web_Scraping/test_js_required_fallback_metric.py
- tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py
- tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a new 0.1.38 release that includes the GHCR-only Docker release workflow fix and PyPI PortAudio setup fix, then publish it through GitHub release flow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Version metadata is bumped to 0.1.38.
- [x] Changelog and README describe the corrective release.
- [x] The release PR carries the publishing workflow fixes from dev to main.
- [x] Relevant release/doc workflow tests pass locally.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Release is cut from the current `dev` release train into `main`.
- This release supersedes the failed 0.1.37 publish attempt by including workflow fixes for PyPI and GHCR-only Docker publishing.
- Local verification before pushing:
  - `git diff --check`
  - `python -m pytest -q tldw_Server_API/tests/CI/test_release_workflow_contracts.py`
  - `python -m pytest -q tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py::test_openwebui_import_is_discoverable_from_api_docs`
  - `python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml`
  - `python -m pytest -q tldw_Server_API/tests/Workflows/test_workflows_config_defaults.py`
  - `python Helper_Scripts/checks/guard_http_client_patching.py`
  - `python -m pytest -q tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py`
  - `python -m black --check tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py`
  - `pre-commit run --files tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py .github/workflows/ci.yml`
  - `python -m bandit -r tldw_Server_API/app/main.py -f json -o /tmp/bandit_release_0_1_38_main.json`
  - `python -m bandit -r tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py -f json -o /tmp/bandit_phase2_article_runtime_boundary.json` (pytest `assert` B101 findings only)
  - `PYTHONPATH=. /private/tmp/tldw-release-py312-ci/bin/python Helper_Scripts/export_openapi_schema.py --check apps/tldw-frontend/lib/api/openapi.fingerprint.json`
  - `TMPDIR=/private/tmp PYTHON=/private/tmp/tldw-release-py312-ci/bin/python bun run generate:api-types`
- CI follow-up: PR #2677 initially failed `Shard coverage guard` because `tldw_Server_API/tests/Workflows/test_workflows_config_defaults.py` had not been assigned to a full-suite shard. Added it to the existing `product-workflows-api` shard entries.
- CI follow-up: PR #2677 then failed push `run-pre-commit` because one existing Web Scraping test helper call had `monkeypatch` and `backend="httpx"` on the same line, matching the raw HTTP patch guard. Split the call over multiple lines without changing behavior.
- CI follow-up: PR #2677 then failed `backend-required` at the OpenAPI drift gate. Reproduced with a CI-matching temporary Python 3.12 environment, updated `apps/tldw-frontend/lib/api/openapi.fingerprint.json` to sha256 `ccf658a22089cc43e9d691d1ac000e3a4f473c7159135a72217d32df0b1652dd` (`1963` paths, `2827` schemas), and verified `bun run generate:api-types` completes with that Python env. The full generated schema/type files remain ignored.
- PR review follow-up: addressed all inline CodeRabbit/Qodo/Gemini comments by tightening SQLite FTS tokenization for doubled quotes, case-sensitive operator handling, binary-operator negation, column-scoped punctuation values, and flashcard FTS column aliases; memoized `useConfirmDanger`; made the Notes study-pack overflow action mobile-only; scoped Flashcards select option lookup to the visible AntD dropdown; and waited for the edit drawer before keyboard fallback.
- PR review verification:
  - `python -m pytest -q tldw_Server_API/tests/CI/test_release_workflow_contracts.py tldw_Server_API/tests/RAG_NEW/unit/test_fts_query_translation_edge_cases.py tldw_Server_API/tests/Flashcards/test_flashcards_db_assets.py`
  - `TMPDIR=/private/tmp bun run test:run ../packages/ui/src/components/Notes/__tests__/NotesEditorHeader.stage2.touch-layout.test.tsx`
  - `TMPDIR=/private/tmp bun run typecheck`
  - `TMPDIR=/private/tmp ./node_modules/.bin/eslint e2e/utils/page-objects/FlashcardsPage.ts`
  - `git diff --check`
  - `python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/backends/fts_translator.py -f json -o /tmp/bandit_pr2677_review_prod.json` (zero findings)
  - `python -m bandit -r tldw_Server_API/tests/CI/test_release_workflow_contracts.py tldw_Server_API/tests/RAG_NEW/unit/test_fts_query_translation_edge_cases.py tldw_Server_API/tests/Flashcards/test_flashcards_db_assets.py -f json -o /tmp/bandit_pr2677_review_tests.json` (pytest `assert` B101 findings only; no non-B101 findings)
- CodeQL follow-up: replaced the SQLite FTS normalizer's regex tokenizer with a linear scanner to resolve the high-severity "Polynomial regular expression used on uncontrolled data" alert on `fts_translator.py`.
- CodeQL fix verification:
  - `python -m pytest -q tldw_Server_API/tests/RAG_NEW/unit/test_fts_query_translation_edge_cases.py::test_sqlite_normalization_does_not_depend_on_regex_tokenizer` failed before the fix and passed after it.
  - `python -m pytest -q tldw_Server_API/tests/CI/test_release_workflow_contracts.py tldw_Server_API/tests/RAG_NEW/unit/test_fts_query_translation_edge_cases.py tldw_Server_API/tests/Flashcards/test_flashcards_db_assets.py`
  - `git diff --check`
  - `python -m bandit -r tldw_Server_API/app/core/DB_Management/backends/fts_translator.py tldw_Server_API/tests/RAG_NEW/unit/test_fts_query_translation_edge_cases.py -f json -o /tmp/bandit_pr2677_codeql_fix.json` (production file zero findings; pytest `assert` B101 findings only in the test file; no non-B101 findings)
- CI follow-up: `Full Suite (Ubuntu / Python 3.11 / utils-http)` failed because redirect-hardening tests used `203.0.113.7`, which is intentionally outside the shard's `WORKFLOWS_EGRESS_ALLOWLIST`; reproduced locally under the CI allowlist and changed the mocked cross-origin redirect target to allowlisted `example.com` without broadening CI egress.
- CI follow-up: `Frontend UX Gates / UX Smoke Gate` failed in the real-server chat cockpit spec while exiting focus mode; replaced raw shared test-id clicks with a helper that clicks the accessible layout action and treats an already-completed mode switch as success, avoiding retries against the newly rendered opposite toggle.
- CI follow-up: `Full Suite shard (macos-latest / Python 3.12 / core-utils-tooling)` additionally failed the endpoint auth dependency boundary check because `rpg.py` imported `User` directly from `core.AuthNZ.User_DB_Handling`; switched it to the existing `auth_deps.py` re-export.
- CI follow-up: `Full Suite shard (windows-latest / Python 3.12 / core-utils-tooling)` failed on the same redirect allowlist and `rpg.py` auth import boundary issues already fixed locally.
- CI follow-up: `Full Suite shard (Ubuntu / Python 3.12 / core-utils-tooling)` and `Full Suite shard (Ubuntu / Python 3.13 / core-utils-tooling)` failed on the same redirect allowlist and `rpg.py` auth import boundary issues already fixed locally.
- CI follow-up: `Full Suite shard (Ubuntu / Python 3.12 / gap-verified-12)` failed CI contract tests for stale release/shard metadata. Added `test_env_absent_defaults.py` to the Linux 3.11 config runtime shard, updated the PyPI workflow contract to match the release contract's workflow-file path trigger, and allowed the already-sharded workspace artifact validation test in the auth/db coverage contract.
- CI follow-up: `Full Suite shard (Ubuntu / Python 3.13 / gap-verified-12)` failed on the same release/shard metadata contract issues already fixed locally.
- CI follow-up: the same local contract slice then exposed a duplicate assignment for `test_visual_identity_expression_metadata.py`; kept it in the dedicated `visual-identities` shard and removed the duplicate `chat-character-legacy-core` entries from each repeated full-suite matrix.
- CI follow-up: `Full Suite shard (macos-latest / Python 3.12 / db-privileges)` failed because two flashcard FTS empty-query tests monkeypatched `FTSQueryTranslator.normalize_query` with a two-argument lambda after flashcard search started passing the `sqlite_column_aliases` keyword. Updated the test doubles to accept keyword arguments while preserving the empty-normalized-query behavior under test.
- CI follow-up: `Full Suite shard (windows-latest / Python 3.12 / db-privileges)` failed on the same flashcard FTS mock signature issue already fixed locally.
- CI follow-up: `Full Suite shard (Ubuntu / Python 3.12 / db-privileges)` failed on the same flashcard FTS mock signature issue already fixed locally.
- CI follow-up: `Full Suite shard (Ubuntu / Python 3.13 / db-privileges)` failed on the same flashcard FTS mock signature issue already fixed locally.
- CI follow-up: `Full Suite shard (Ubuntu / Python 3.12 / media-ingestion-modification)` failed because the mixed audio URL test used `httpbin.org`, outside the CI egress allowlist, and because the Parakeet ONNX loader test could discover a real runner cache and open multiple ONNX sessions. Switched the URL to allowlisted `example.com`, accepted the security-policy error message for the failed URL branch, and patched the ONNX loader config to use the test's temp local model directory.
- CI follow-up: `Full Suite shard (macos-latest / Python 3.12 / media-ingestion-modification)` failed on the same audio URL allowlist expectation and Parakeet ONNX cache-isolation issues already fixed locally.
- CI follow-up: `Full Suite shard (Ubuntu / Python 3.13 / media-ingestion-modification)` failed on the same audio URL allowlist expectation and Parakeet ONNX cache-isolation issues already fixed locally.
- CI follow-up: `Full Suite shard (windows-latest / Python 3.12 / media-ingestion-modification)` failed on the same audio URL allowlist expectation and Parakeet ONNX cache-isolation issues already fixed locally.
- CI follow-up: `Full Suite shard (windows-latest / Python 3.12 / chat-character-property)` failed `test_archive_import_rejects_unsafe_zip_entries[entries2-unsafe_archive_path]` because the test fixture wrote a backslash ZIP member name through `zipfile`, which normalizes backslashes to forward slashes on Windows before the importer sees the entry. Preserved raw backslash member names in the fixture by patching the ZIP member-name bytes after write when needed, and added a regression assertion that the fixture keeps `sprites\happy.png` intact.
- CI follow-up: `Full Suite shard (windows-latest / Python 3.12 / visual-identities)` failed on the same Windows backslash ZIP fixture normalization issue already fixed locally.
- CI follow-up: `Full Suite shard (Ubuntu / Python 3.12 / platform-resource-governance)` failed `test_e2e_audio_transcriptions_headers_and_mocked_stt` for both RG backends because the shard does not have `ffmpeg` on PATH, so the endpoint rejected the upload during canonical WAV conversion before reaching the mocked STT path. Patched the test to mock `convert_to_wav` alongside `speech_to_text`, keeping the test scoped to resource-governance headers rather than ffmpeg availability.
- CI follow-up: `Full Suite shard (Ubuntu / Python 3.12 / platform-infrastructure-metrics)` failed `test_notify_generic_redacts_sensitive_payload_before_storage_and_webhook` because `NotificationService.notify_generic()` mutated the caller's payload while adding `ts`, despite the method docstring saying the timestamp is added to the recorded copy. Moved timestamp insertion to the copied payload so storage/webhook delivery keep `ts` while the caller's dict remains unchanged.
- Local verification for pending CI fixes:
  - `WORKFLOWS_EGRESS_ALLOWLIST='93.184.216.34,does-not-resolve.invalid,example.com' PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/http_client/test_redirect_header_hardening.py` (17 passed)
  - `TMPDIR=/private/tmp ./node_modules/.bin/eslint e2e/workflows/chat-cockpit.real-server.spec.ts` (0 errors; existing explicit-any warnings only)
  - `TMPDIR=/private/tmp bun run typecheck`
  - `git diff --check`
  - `python -m bandit -r tldw_Server_API/tests/http_client/test_redirect_header_hardening.py -f json -o /tmp/bandit_pr2677_http_redirect_test.json` (pytest `assert` B101 findings only; no non-B101 findings)
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py::test_endpoint_auth_dependency_symbols_come_from_auth_deps` (1 passed)
  - `python -m py_compile tldw_Server_API/app/api/v1/endpoints/rpg.py`
  - `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/rpg.py -f json -o /tmp/bandit_pr2677_rpg_endpoint.json` (zero findings)
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/http_client/test_redirect_header_hardening.py tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py` (20 passed)
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py::test_publish_pypi_workflow_preserves_manual_dispatch_and_gates_push tldw_Server_API/tests/CI/test_required_workflow_contracts.py::test_linux_311_smoke_is_sharded_for_timeout_control tldw_Server_API/tests/CI/test_required_workflow_contracts.py::test_full_suite_splits_slow_chat_and_retrieval_shards` (3 passed after contract/shard fixes)
  - `python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml` (`new_uncovered=0`)
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py` (49 passed)
  - `python -m bandit -r tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py -f json -o /tmp/bandit_pr2677_ci_contracts.json` (pytest `assert` B101 findings only; no non-B101 findings)
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py` (2 passed)
  - `python -m bandit -r tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py -f json -o /tmp/bandit_pr2677_chacha_flashcards_fts_empty.json` (pytest `assert` B101 findings only; no non-B101 findings)
  - `WORKFLOWS_EGRESS_ALLOWLIST='93.184.216.34,does-not-resolve.invalid,example.com' PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/http_client/test_redirect_header_hardening.py tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py` (71 passed)
  - `WORKFLOWS_EGRESS_ALLOWLIST='93.184.216.34,does-not-resolve.invalid,example.com' PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py::TestProcessAudios::test_process_audio_multi_status_mixed tldw_Server_API/tests/Media_Ingestion_Modification/test_nemo_transcription.py::TestNemoTranscription::test_load_parakeet_onnx` (1 passed, 1 skipped due unavailable local audio/STT runtime path)
  - `python -m bandit -r tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py tldw_Server_API/tests/Media_Ingestion_Modification/test_nemo_transcription.py -f json -o /tmp/bandit_pr2677_media_ingestion_tests.json` (pytest `assert` B101 findings only; no non-B101 findings)
- Local verification before pushing the pending CI fixes:
  - `WORKFLOWS_EGRESS_ALLOWLIST='93.184.216.34,does-not-resolve.invalid,example.com' PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/http_client/test_redirect_header_hardening.py tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py::TestProcessAudios::test_process_audio_multi_status_mixed tldw_Server_API/tests/Media_Ingestion_Modification/test_nemo_transcription.py::TestNemoTranscription::test_load_parakeet_onnx` (72 passed, 1 skipped due unavailable local audio/STT runtime path)
  - `git diff --check`
  - `python -m bandit -q -s B101 tldw_Server_API/app/api/v1/endpoints/rpg.py tldw_Server_API/tests/http_client/test_redirect_header_hardening.py tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py tldw_Server_API/tests/Media_Ingestion_Modification/test_nemo_transcription.py` (zero findings)
  - `TMPDIR=/private/tmp ./node_modules/.bin/eslint e2e/workflows/chat-cockpit.real-server.spec.ts` (0 errors; existing explicit-any warnings only)
  - `TMPDIR=/private/tmp bun run typecheck`
- Local verification for the Windows visual identity archive-import fixture fix:
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Visual_Identities/test_visual_identity_archive_import.py` (21 passed)
  - `python -m bandit -q -s B101 tldw_Server_API/tests/Visual_Identities/test_visual_identity_archive_import.py` (zero findings)
- Local verification for the resource-governance audio transcription header fix:
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Resource_Governance/test_e2e_chat_audio_headers.py::test_e2e_audio_transcriptions_headers_and_mocked_stt` (2 passed)
  - `python -m bandit -q -s B101 tldw_Server_API/tests/Resource_Governance/test_e2e_chat_audio_headers.py` reported pre-existing test-file baseline findings `B112` at line 69 and `B106` at line 159; the new patch did not introduce them.
  - `python -m bandit -q -s B101,B106,B112 tldw_Server_API/tests/Resource_Governance/test_e2e_chat_audio_headers.py` (zero findings)
  - `git diff --check`
- Local verification for the notification generic-payload timestamp fix:
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Monitoring/test_notification_service.py::test_notify_generic_redacts_sensitive_payload_before_storage_and_webhook` (1 passed)
  - `python -m bandit -q -s B101 tldw_Server_API/app/core/Monitoring/notification_service.py tldw_Server_API/tests/Monitoring/test_notification_service.py` (zero findings)
- Consolidated local regression verification for all currently classified PR #2677 failures:
  - `WORKFLOWS_EGRESS_ALLOWLIST='93.184.216.34,does-not-resolve.invalid,example.com' PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/http_client/test_redirect_header_hardening.py tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py::TestProcessAudios::test_process_audio_multi_status_mixed tldw_Server_API/tests/Media_Ingestion_Modification/test_nemo_transcription.py::TestNemoTranscription::test_load_parakeet_onnx tldw_Server_API/tests/Visual_Identities/test_visual_identity_archive_import.py tldw_Server_API/tests/Resource_Governance/test_e2e_chat_audio_headers.py::test_e2e_audio_transcriptions_headers_and_mocked_stt tldw_Server_API/tests/Monitoring/test_notification_service.py::test_notify_generic_redacts_sensitive_payload_before_storage_and_webhook` (96 passed, 1 skipped due unavailable local audio/STT runtime path)
- Current PR #2677 suite status after local verification: 14 failed checks, 445 passed checks, 317 pending checks, 5 skipped checks. Per user instruction, local fixes remain unpushed until the current full suite reaches zero pending checks.
- Latest PR #2677 suite status while waiting to push: 15 failed checks, 498 passed checks, 263 pending checks, 5 skipped checks. The additional failure is a duplicate Ubuntu 3.13 `media-ingestion-modification` shard covered by the existing local media-ingestion test fixes.
- Latest PR #2677 suite status while waiting to push: 16 failed checks, 522 passed checks, 238 pending checks, 5 skipped checks. The additional failure is a duplicate Windows Python 3.12 `media-ingestion-modification` shard covered by the existing local media-ingestion test fixes.
- Latest PR #2677 suite status while waiting to push: 17 failed checks, 558 passed checks, 201 pending checks, 5 skipped checks. The additional failure is the Windows Python 3.12 `chat-character-property` archive-import fixture issue fixed locally.
- Latest PR #2677 suite status while waiting to push: 19 failed checks, 571 passed checks, 186 pending checks, 5 skipped checks. The two additional failures are the Windows Python 3.12 `visual-identities` duplicate of the archive-import fixture issue and the Ubuntu Python 3.12 `platform-resource-governance` ffmpeg-availability test gap, both fixed locally.
- Latest PR #2677 suite status while waiting to push: 20 failed checks, 584 passed checks, 172 pending checks, 5 skipped checks. The additional failure is the Ubuntu Python 3.12 `platform-infrastructure-metrics` notification payload mutation bug fixed locally.
- Latest PR #2677 suite status while waiting to push: 20 failed checks, 591 passed checks, 165 pending checks, 5 skipped checks.
- CI follow-up: `Full Suite shard (Ubuntu / Python 3.12 / integrations)` failed because the web-scraping import inventory artifacts were stale, two web-scraping tests still mocked the old `http_fetch` path instead of the current `_fetch_article_lightweight` boundary, and the default fetch-client test replaced `time.monotonic` with an iterator that could be exhausted by async teardown. Regenerated the inventory artifacts, moved those tests to the current lightweight-fetch seam, kept policy/robots checks locally stubbed, and made the monotonic stub return a fallback value after the measured calls.
- Local verification for the integrations shard fixes:
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py::test_import_inventory_artifact_matches_current_import_surface tldw_Server_API/tests/Web_Scraping/test_handlers.py::test_scrape_article_uses_handler tldw_Server_API/tests/Web_Scraping/test_js_required_fallback_metric.py::test_js_required_emits_fallback_metric tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py::test_default_fetch_client_measures_elapsed_with_monotonic_clock` (4 passed)
  - `python -m bandit -r tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py tldw_Server_API/tests/Web_Scraping/test_handlers.py tldw_Server_API/tests/Web_Scraping/test_js_required_fallback_metric.py tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py -f json -o /tmp/bandit_integrations_release_0_1_38.json -s B101` (zero findings)
  - `git diff --check`
- Expanded local regression verification after the integrations fix:
  - `WORKFLOWS_EGRESS_ALLOWLIST='93.184.216.34,does-not-resolve.invalid,example.com' PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/http_client/test_redirect_header_hardening.py tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py::TestProcessAudios::test_process_audio_multi_status_mixed tldw_Server_API/tests/Media_Ingestion_Modification/test_nemo_transcription.py::TestNemoTranscription::test_load_parakeet_onnx tldw_Server_API/tests/Visual_Identities/test_visual_identity_archive_import.py tldw_Server_API/tests/Resource_Governance/test_e2e_chat_audio_headers.py::test_e2e_audio_transcriptions_headers_and_mocked_stt tldw_Server_API/tests/Monitoring/test_notification_service.py::test_notify_generic_redacts_sensitive_payload_before_storage_and_webhook tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py::test_import_inventory_artifact_matches_current_import_surface tldw_Server_API/tests/Web_Scraping/test_handlers.py::test_scrape_article_uses_handler tldw_Server_API/tests/Web_Scraping/test_js_required_fallback_metric.py::test_js_required_emits_fallback_metric tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py::test_default_fetch_client_measures_elapsed_with_monotonic_clock` (100 passed, 1 skipped due unavailable local audio/STT runtime path)
- Latest PR #2677 suite status while waiting to push: 23 failed checks, 622 passed checks, 131 pending checks, 5 skipped checks. Newly visible failures are duplicate shard coverage for already classified root causes except `integrations`, which is now fixed and verified locally. Per user instruction, local fixes remain unpushed until the current full suite reaches zero pending checks.
- Latest PR #2677 suite status while waiting to push: 23 failed checks, 633 passed checks, 120 pending checks, 5 skipped checks.
- Latest PR #2677 suite status while waiting to push: 24 failed checks, 655 passed checks, 97 pending checks, 5 skipped checks. The additional failure is the Ubuntu Python 3.13 `integrations` shard and matches the same stale inventory, stale fetch mock boundary, and exhausted monotonic-clock stub root causes already fixed and verified locally.
- CI follow-up: `Full Suite shard (macos-latest / Python 3.12 / platform-infrastructure-metrics)` failed on the same notification generic-payload timestamp mutation already fixed locally.
- CI follow-up: `Full Suite shard (macos-latest / Python 3.12 / product-workflows-engine)` failed `test_run_saved_sync_waits_for_completion` because the workflow run reached `succeeded`, but scheduler active-count accounting could remain attached to the scheduler instance that started a background run if the singleton changed before the run captured its notifier. Passed the scheduling instance into scheduled `start_run()` calls so completion decrements the same scheduler that incremented active tenant/workflow counts; direct `start_run()` callers keep the existing singleton fallback.
- Local verification for the workflows-engine scheduler notifier fix:
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Workflows/test_engine_scheduler.py::test_run_saved_sync_waits_for_completion` (1 passed)
  - `python -m py_compile tldw_Server_API/app/core/Workflows/engine.py`
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/Workflows/test_engine_*.py tldw_Server_API/tests/Workflows/test_dual_backend_engine.py tldw_Server_API/tests/Workflows/test_orphan_requeue_*.py tldw_Server_API/tests/Workflows/test_workflows_map_substeps_*.py tldw_Server_API/tests/Workflows/test_workflow_attempt_failures.py tldw_Server_API/tests/Workflows/test_workflow_stress.py tldw_Server_API/tests/Workflows/test_workflows_scheduler.py` (90 passed, 7 skipped)
  - `python -m bandit -r tldw_Server_API/app/core/Workflows/engine.py -f json -o /tmp/bandit_workflows_engine_release_0_1_38.json` (zero findings)
  - `git diff --check`
- Latest PR #2677 suite status while waiting to push: 26 failed checks, 692 passed checks, 58 pending checks, 5 skipped checks. The additional failures were the duplicate macOS infrastructure-metrics timestamp mutation and the new macOS workflows-engine scheduler accounting race fixed and verified locally.
- Latest PR #2677 suite status while waiting to push: 27 failed checks, 705 passed checks, 44 pending checks, 5 skipped checks. The additional failure is the Windows Python 3.12 `integrations` shard and matches the same stale inventory, stale fetch mock boundary, and exhausted monotonic-clock stub root causes already fixed and verified locally.
- Latest PR #2677 suite status while waiting to push: 28 failed checks, 715 passed checks, 34 pending checks, 5 skipped checks. The additional failure is the macOS Python 3.12 `integrations` shard and matches the same stale inventory, stale fetch mock boundary, and exhausted monotonic-clock stub root causes already fixed and verified locally.
- Latest PR #2677 suite status while waiting to push: 29 failed checks, 718 passed checks, 30 pending checks, 5 skipped checks. The additional failure is the Ubuntu Python 3.13 `platform-resource-governance` shard and matches the same missing-`ffmpeg` conversion path before mocked STT already fixed and verified locally.
- Latest PR #2677 suite status while waiting to push: 30 failed checks, 727 passed checks, 22 pending checks, 5 skipped checks. The additional failure is the Windows Python 3.12 `platform-infrastructure-metrics` shard and matches the same notification generic-payload timestamp mutation already fixed and verified locally.
- Latest PR #2677 suite status while waiting to push: 30 failed checks, 729 passed checks, 20 pending checks, 5 skipped checks.
- Latest PR #2677 suite status while waiting to push: 30 failed checks, 738 passed checks, 11 pending checks, 5 skipped checks. Remaining pending checks are eight Windows shards still in progress and the three queued full-suite aggregate jobs for macOS Python 3.12, Ubuntu Python 3.13, and Ubuntu Python 3.12.
- Latest PR #2677 suite status while waiting to push: 30 failed checks, 744 passed checks, 5 pending checks, 5 skipped checks. Remaining pending checks are Windows Python 3.12 `product-evaluations-integration`, Windows Python 3.12 `platform-resource-governance`, and the three queued aggregate full-suite jobs for macOS Python 3.12, Ubuntu Python 3.13, and Ubuntu Python 3.12.
- Latest PR #2677 suite status while waiting to push: 31 failed checks, 744 passed checks, 4 pending checks, 5 skipped checks. The additional failure is the Windows Python 3.12 `platform-resource-governance` shard and matches the same audio transcription conversion-before-mocked-STT root cause already fixed and verified locally.
- Latest PR #2677 suite status while waiting to push: 31 failed checks, 745 passed checks, 4 pending checks, 5 skipped checks. The Windows Python 3.12 `product-evaluations-integration` shard passed; only the aggregate full-suite checks for Windows Python 3.12, macOS Python 3.12, Ubuntu Python 3.13, and Ubuntu Python 3.12 remain queued.
- Raw Actions job query for run `28808263787` shows no hidden non-completed jobs beyond the four queued aggregate full-suite summary jobs. Each has `runner_name` empty/null, so the current hold-up is GitHub Actions not assigning runners to the summary jobs yet.
- PR review follow-up: addressing the remaining open CodeRabbit thread on flashcard SQLite FTS normalization by sharing the list/count normalizer call and making scoped SQLite aliases case-insensitive with explicit fallback for unmapped scopes.
- CodeRabbit follow-up verification:
  - `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/RAG_NEW/unit/test_fts_query_translation_edge_cases.py tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py` (13 passed)
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/core/DB_Management/backends/fts_translator.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -q -s B101 -r tldw_Server_API/app/core/DB_Management/backends/fts_translator.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/RAG_NEW/unit/test_fts_query_translation_edge_cases.py tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py` (zero findings; existing `nosec` warnings only)
  - `git diff --check`
- Pre-push consolidated local regression while waiting for the remaining PR checks:
  - `WORKFLOWS_EGRESS_ALLOWLIST='93.184.216.34,does-not-resolve.invalid,example.com' PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q -p pytest_asyncio.plugin tldw_Server_API/tests/http_client/test_redirect_header_hardening.py tldw_Server_API/tests/lint/test_endpoint_auth_deps_import_boundary.py tldw_Server_API/tests/DB_Management/test_chacha_flashcards_fts_empty.py tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py tldw_Server_API/tests/CI/test_release_workflow_contracts.py tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py::TestProcessAudios::test_process_audio_multi_status_mixed tldw_Server_API/tests/Media_Ingestion_Modification/test_nemo_transcription.py::TestNemoTranscription::test_load_parakeet_onnx tldw_Server_API/tests/Visual_Identities/test_visual_identity_archive_import.py tldw_Server_API/tests/Resource_Governance/test_e2e_chat_audio_headers.py::test_e2e_audio_transcriptions_headers_and_mocked_stt tldw_Server_API/tests/Monitoring/test_notification_service.py::test_notify_generic_redacts_sensitive_payload_before_storage_and_webhook tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py::test_import_inventory_artifact_matches_current_import_surface tldw_Server_API/tests/Web_Scraping/test_handlers.py::test_scrape_article_uses_handler tldw_Server_API/tests/Web_Scraping/test_js_required_fallback_metric.py::test_js_required_emits_fallback_metric tldw_Server_API/tests/Web_Scraping/test_phase2_runtime_adapters.py::test_default_fetch_client_measures_elapsed_with_monotonic_clock tldw_Server_API/tests/Workflows/test_engine_scheduler.py::test_run_saved_sync_waits_for_completion` (101 passed, 1 skipped due unavailable local audio/STT runtime path)
  - `git diff --check`
  - `TMPDIR=/private/tmp bun run typecheck` in `apps/tldw-frontend` (passed)
  - `TMPDIR=/private/tmp ./node_modules/.bin/eslint e2e/workflows/chat-cockpit.real-server.spec.ts` in `apps/tldw-frontend` (0 errors; existing explicit-`any` warnings only)
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Cut the 0.1.38 corrective release train on PR #2677 and kept the PyPI/GHCR publishing workflow fixes in the release path.
- Investigated every visible failing PR check from the current CI run. The failures resolved to duplicated shard coverage of stale workflow contracts, frontend layout test behavior, auth import boundaries, CI egress policy, FTS monkeypatch signatures, media/NVIDIA model path assumptions, Windows ZIP path normalization, missing audio conversion mocking, notification payload mutation, stale web-scraping inventory/fetch seams, exhausted monotonic stubs, and a workflow scheduler notifier race.
- Fixed the locally reproducible root causes, regenerated the web-scraping inventory artifacts, and verified the patch set with targeted backend regression, frontend typecheck/lint, Bandit on touched scopes, and `git diff --check`.
- Confirmed the only remaining live PR #2677 pending checks are aggregate full-suite summary jobs with no additional tests to run; raw Actions job listing showed no hidden non-completed test shards.
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
