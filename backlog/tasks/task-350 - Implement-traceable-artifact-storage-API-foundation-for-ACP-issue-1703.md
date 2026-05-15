---
id: TASK-350
title: Implement traceable artifact storage/API foundation for ACP issue 1703
status: Done
assignee:
  - codex
created_date: '2026-05-15 01:15'
updated_date: '2026-05-15 01:34'
labels:
  - acp
  - artifacts
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1703'
  - 'https://github.com/rmusser01/tldw_server/issues/1532'
  - 'https://github.com/rmusser01/tldw_server/issues/1525'
  - 'https://github.com/rmusser01/tldw_server/issues/1538'
documentation:
  - Docs/Product/Traceable_Work_Product_Artifact_Contract.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend-owned storage/API foundation for traceable work-product artifacts using the golden-path ACP-generated source-grounded workspace brief. Reuse existing file/output artifact primitives where practical, but add typed metadata or schemas needed by the artifact contract: workspace ownership, version chain, review state, source lineage, redaction posture, export references, and ACP producer references. Keep this slice backend-focused so Workspace UI, ACP promotion, export adapters, and verification can build on a stable contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Storage/API supports creating and reading the golden-path traceable work-product artifact.
- [x] #2 Versioning, review state, source-lineage, and ACP producer metadata are represented in backend response schemas.
- [x] #3 Redaction posture and support-safe view behavior are explicit at the API boundary.
- [x] #4 Focused backend tests cover create/read/version/review-state/source-lineage behavior.
- [x] #5 Docs link this slice back to issues #1525, #1538, #1532, and #1703.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reuse the existing workspace_artifacts surface as the traceable work-product API foundation instead of adding a parallel ACP-only artifact subsystem.
2. Extend ChaChaNotes workspace artifact storage with contract fields for review_state, owner_scope/owner_id, root_artifact_id, artifact_version_id, previous_version_id, producer metadata, source_lineage, review metadata, version metadata, export references, redaction posture, and schema_version. Preserve existing title/status/content behavior for compatibility.
3. Add a workspace_artifact_versions history table or equivalent version records so create/update operations produce stable version-chain entries instead of relying only on optimistic-lock version numbers.
4. Extend workspace Pydantic request/response schemas and endpoint mapping to expose stable contract groups while keeping older clients functional.
5. Add focused DB and API tests around creating, reading, updating/versioning, review-state mapping, source-lineage/provenance metadata, and support-safe redaction posture.
6. Update the traceable artifact and ACP PRD docs to link implementation issue #1703 and document the backend contract now available.
7. Verify with focused Workspaces/ChaChaNotes tests, git diff --check, py_compile for touched backend files, and Bandit on touched Python scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Planning survey: existing workspace artifact endpoints live in tldw_Server_API/app/api/v1/endpoints/workspaces.py and schemas in workspace_schemas.py. Persistence is in CharactersRAGDB workspace_artifacts. File/output artifacts are export representations and should be referenced later rather than used as the durable work-product record for this slice.

Baseline verification before implementation: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q passed with 56 passed and 5 warnings in 60.37s.

Implementation update: extended existing workspace_artifacts storage/API with traceable artifact contract fields, review-state validation, per-version rows, source-lineage/review/version/export/redaction JSON metadata, and API response exposure. Added DB/API regression coverage for create/read/versioning and documented #1703 as the backend foundation slice.

Verification: focused red tests failed before implementation; focused contract tests now pass. Full focused suite passed with 59 passed and 5 warnings. git diff --check passed. py_compile passed for touched backend modules. Bandit passed for touched backend scope with no errors and no results in /tmp/bandit_acp_artifact_storage_api_1703.json.

Post self-review update: added delete/recreate regression coverage for stable artifact version IDs. The regression failed before cleanup, then passed after hard delete began removing workspace_artifact_versions rows for the artifact. Final focused suite rerun passed with 60 passed and 5 warnings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented #1703 as the backend-owned storage/API foundation for traceable ACP-adjacent workspace artifacts. The existing workspace_artifacts surface now carries contract fields for ownership, review state, stable artifact/version IDs, source lineage, ACP producer metadata, review/version metadata, export references, redaction posture, and schema version. Artifact create/update writes version-history rows, and hard delete clears version rows so recreating an artifact ID remains compatible. Workspace artifact request/response schemas and endpoint mapping expose the new contract fields while preserving existing title/status/content behavior. Docs now mark #1703 as the storage/API foundation and keep UI detail, ACP promotion, export adapters, and broader signoff verification as follow-up slices. Verification recorded: 60 focused workspace DB/API tests passed, git diff --check passed, py_compile passed for touched backend modules, and Bandit reported no errors/results for the touched backend scope.
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
