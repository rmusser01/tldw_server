---
id: TASK-2334
title: Implement Chatbook format v1.1 rollout
status: Done
labels:
- chatbooks
- implementation
documentation:
- Docs/Product/Chatbooks_Format_v1_1_SPEC.md
- Docs/superpowers/plans/2026-06-18-chatbooks-format-v1-1-implementation-plan.md
modified_files:
- Docs/Schemas/chatbooks_manifest_v1_1.json
- Docs/API-related/Chatbook_API_Documentation.md
- Docs/Code_Documentation/Chatbook_Developer_Guide.md
- Docs/Product/Chatbooks_Format_v1_1_SPEC.md
- tldw_Server_API/app/core/Chatbooks/chatbook_models.py
- tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py
- tldw_Server_API/app/core/Chatbooks/chatbook_service.py
- tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py
- tldw_Server_API/app/core/Chatbooks/README.md
- tldw_Server_API/app/services/core_jobs_worker.py
- tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py
- tldw_Server_API/app/api/v1/endpoints/chatbooks.py
- tldw_Server_API/app/core/Explainer/chatbook_adapter.py
- tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py
- tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_file_inventory.py
- tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_preview.py
- tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py
- tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py
- tldw_Server_API/tests/Chatbooks/test_chatbook_service_preview_import_safety.py
- tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Chatbook v1.1 format rollout from the implementation plan: schema, helper module, opt-in export versioning, v1.1 manifest metadata and file inventory, Explainer envelopes, preview report, import integrity enforcement, docs, tests, and Bandit verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 5 slice: added v1.1 Explainer content envelope export support without changing v1 export behavior. Changed chatbook_format_v1_1.py, chatbook_service.py, and test_explainer_session_content_type.py. Red test first failed with KeyError: 'envelope'; green and verification passed. Commit: d5f399d28c feat: add v1.1 explainer chatbook envelopes. Verification: Explainer suites 16 passed; manifest/file-inventory suites 24 passed; Bandit on touched app files reported 0 results.

Task 6 slice: added v1.1 preview report API support while preserving the existing preview_chatbook two-tuple and endpoint fallback for legacy service doubles. Changed chatbook_schemas.py, chatbooks.py, chatbook_service.py, chatbook_format_v1_1.py, and test_chatbooks_v1_1_preview.py. Red test first failed with KeyError: 'compatibility'; focused green test passed. Verification: service preview safety/API mapping suite 19 passed; Bandit on touched app files reported 0 results. Required combined preview API command is blocked by an existing full-app TestClient teardown hang in test_chatbooks_api_preview.py after its first test, reproduced when that file is run alone.

Task 6 review fix: hardened preview report handling for malformed-but-parseable features_used and file_inventory manifest data. Added regression tests for non-string feature tokens, non-list inventory, and non-string inventory paths. RED failed with TypeError for all three malformed inputs; GREEN passed focused v1.1 preview suite (4 passed), service preview safety/API mapping suite (19 passed), and Bandit on chatbook_format_v1_1.py reported 0 results.

Task 7 slice: enforced v1.1 pre-import validation immediately after manifest parsing and before content selections/import writes. Added validate_v1_1_before_import(), checksum mismatch rejection, reject_import unknown-feature handling, and warning propagation for non-reject unknown-feature policy. RED targeted tests failed because import ignored checksum/feature policy and returned normal import messages; GREEN targeted tests passed (3 passed). Required verification passed: test_chatbooks_import_validation.py + test_chatbook_service_preview_import_safety.py (12 passed), test_chatbooks_v1_1_preview.py (4 passed), Bandit on chatbook_format_v1_1.py and chatbook_service.py reported 0 results/errors. Full git diff --check is blocked by unrelated trailing whitespace in Docs/Design/Agents.md:175; task-scoped git diff --check passed.

Task 7 review fix: required v1.1 import inventory coverage for bundled content item file_path values before writes. Added regressions for empty file_inventory and inventory missing the note payload entry. RED failed with normal import fallback messages; GREEN targeted tests passed (2 passed). Required verification passed: test_chatbooks_import_validation.py + test_chatbook_service_preview_import_safety.py (14 passed), test_chatbooks_v1_1_preview.py (4 passed), Bandit on chatbook_format_v1_1.py and chatbook_service.py reported 0 results/errors, and task-scoped git diff --check passed.

Task 7 re-review fix: validation now derives required v1.1 import inventory paths from current importer fallback paths, including nullable file_path cases. Added regression for a note with file_path null and missing fallback inventory. RED failed with normal import fallback message; GREEN targeted test passed. Required verification passed: test_chatbooks_import_validation.py + test_chatbook_service_preview_import_safety.py (15 passed), test_chatbooks_v1_1_preview.py (4 passed), Bandit on chatbook_format_v1_1.py and chatbook_service.py reported 0 results/errors, and task-scoped git diff --check passed.

Task 7 conversation attachment review fix: validation now reads verified conversation payloads during v1.1 import validation and requires inventory coverage for bundled image attachment file_path values. Added regression for a conversation archive with valid primary inventory but missing attachment inventory. RED failed with normal import fallback message; GREEN targeted test passed. Required verification passed: test_chatbooks_import_validation.py + test_chatbook_service_preview_import_safety.py (16 passed), test_chatbooks_v1_1_preview.py (4 passed), Bandit on chatbook_format_v1_1.py and chatbook_service.py reported 0 results/errors, and task-scoped git diff --check passed.

Task 8 slice: updated Chatbook API docs, developer guide, module README, and v1.1 product spec for the opt-in format_version path, v1.1 preview report fields, import validation behavior, helper module responsibilities, future content-type extension points, and implementation status. Verification: docs-scoped git diff --check on touched docs/backlog files exited 0 with no output; rg confirmed format_version, preview report, file_inventory, validate_v1_1_before_import, and chatbook_format_v1_1.py in the expected docs. Bandit skipped because this slice is docs-only.

Task 8 review fix: corrected Chatbook import API examples so they no longer advertise `import_media=true`, and documented that media/embedding import requests are currently rejected. Verification: `rg` found no remaining `import_media=true` or `"import_media": true` examples in the Chatbook API docs; docs-scoped `git diff --check` exited 0 with no output.

Task 9 final verification: focused v1.1 Chatbook test command passed with 46 tests and 6 warnings:
`source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_file_inventory.py tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_preview.py tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py -v`.
Endpoint mapping coverage passed with 14 tests and 7 warnings:
`source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py -v`.
Bandit touched-scope command wrote `/tmp/bandit_chatbook_v1_1_final.json` with 0 results and 0 errors:
`source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Chatbooks/chatbook_models.py tldw_Server_API/app/core/Chatbooks/chatbook_format_v1_1.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py tldw_Server_API/app/services/core_jobs_worker.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/core/Explainer/chatbook_adapter.py -f json -o /tmp/bandit_chatbook_v1_1_final.json`.
Schema parse passed:
`source .venv/bin/activate && python -m json.tool Docs/Schemas/chatbooks_manifest_v1_1.json >/tmp/chatbooks_manifest_v1_1.pretty.json`.
Task-scoped `git diff --check` on touched Chatbook files exited 0 with no output.
Known blocker: the combined endpoint preview command
`source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py -v`
hung after `test_preview_manifest_version_coercion_legacy` passed and made no progress on `test_preview_manifest_version_ok`; it was interrupted after 101.76 seconds. A reduced run of `test_chatbooks_api_preview.py -k 'not test_preview_manifest_version_ok' --timeout=60 -v` also timed out while setting up the next TestClient. This matches the pre-existing full-app TestClient lifecycle hang documented during Task 6.
Known unrelated workspace blocker: full `git diff --check` exits 2 because `Docs/Design/Agents.md:175` has trailing whitespace outside the touched Chatbook scope.
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
