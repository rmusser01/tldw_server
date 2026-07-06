---
id: TASK-12162
title: Cut 0.1.38 release with publishing workflow fixes
status: In Progress
priority: High
modified_files:
- .github/workflows/ci.yml
- CHANGELOG.md
- Docs/API-related/API_Tags_Index.md
- Docs/mkdocs.yml
- README.md
- apps/packages/ui/src/components/Common/confirm-danger.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesEditorHeader.stage2.touch-layout.test.tsx
- apps/packages/ui/src/components/Notes/NotesEditorHeader.tsx
- apps/tldw-frontend/e2e/utils/page-objects/FlashcardsPage.ts
- apps/tldw-frontend/lib/api/openapi.fingerprint.json
- pyproject.toml
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/backends/fts_translator.py
- tldw_Server_API/app/main.py
- tldw_Server_API/tests/CI/test_release_workflow_contracts.py
- tldw_Server_API/tests/Flashcards/test_flashcards_db_assets.py
- tldw_Server_API/tests/RAG_NEW/unit/test_fts_query_translation_edge_cases.py
- tldw_Server_API/tests/Web_Scraping/test_phase2_article_runtime_boundary.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a new 0.1.38 release that includes the GHCR-only Docker release workflow fix and PyPI PortAudio setup fix, then publish it through GitHub release flow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Version metadata is bumped to 0.1.38.
- [ ] Changelog and README describe the corrective release.
- [ ] The release PR carries the publishing workflow fixes from dev to main.
- [ ] Relevant release/doc workflow tests pass locally.
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
