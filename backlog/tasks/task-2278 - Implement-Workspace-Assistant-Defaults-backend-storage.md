---
id: TASK-2278
title: Implement Workspace Assistant Defaults backend storage
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-08 00:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first Workspace Assistant Defaults V1 slice from #1911: backend storage/schema support for a Persona-only, reference-backed assistant_defaults_json Workspace field in ChaChaNotes DB, with migration coverage and no Persona snapshots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ChaChaNotes workspace schema includes `assistant_defaults_json` for new SQLite/Postgres databases and v48-to-v49 migrations.
- [x] #2 Workspace create/read/update paths preserve and normalize `assistant_defaults_json` as a dict while storing JSON text internally.
- [x] #3 Storage rejects or clears malformed/unsupported assistant default payloads without persisting Persona display snapshots.
- [x] #4 Focused backend tests cover migration/new DB behavior, create/read/update round trips, malformed JSON handling, and no snapshot fields.
- [x] #5 Verification commands and Bandit results are recorded before commit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added Workspace Assistant Defaults Pydantic schema types with Persona-only `assistant_kind`, read-only/read-write memory modes, and explicit null-only deferred fields.
- Added ChaChaNotes schema v49 with `assistant_defaults_json` storage for SQLite and PostgreSQL creation/migration paths.
- Centralized workspace row normalization so DB callers receive `assistant_defaults_json` as `dict | None` while DB rows store compact JSON text.
- Added focused tests in `tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py`.
- Verification:
  - Red test: `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py -q --tb=short --disable-warnings` failed on missing `WorkspaceAssistantDefaults` import before implementation.
  - `python -m py_compile tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` passed.
  - `python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_assistant_defaults_db.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py -q --tb=short --disable-warnings` passed: 31 passed, 6 warnings.
  - `git diff --check` passed.
  - Production Bandit passed with 0 findings for `workspace_schemas.py` and `ChaChaNotes_DB.py`; full touched-scope Bandit only reported pytest `assert` warnings in the new test file.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

- PR review follow-up after rebase onto origin/dev: mapped API assistant_defaults patches to assistant_defaults_json, consumed confirmation-only request metadata, enforced read_write confirmation in WorkspacePatchRequest, added effective assistant default status validation, made malformed write payloads raise InputError while logging malformed stored DB values, and rejected unknown-only DB updates.
- Review verification: focused regression pytest passed (11 passed); broader workspace/DB pytest passed (111 passed); py_compile passed for touched app files; git diff --check passed; Bandit passed on touched app files with 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the backend storage foundation for Workspace Assistant Defaults V1: schema contracts, v49 migrations, JSON normalization helpers, and focused DB tests. The slice remains Persona-only and reference-backed, with no Persona display snapshots stored.
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
