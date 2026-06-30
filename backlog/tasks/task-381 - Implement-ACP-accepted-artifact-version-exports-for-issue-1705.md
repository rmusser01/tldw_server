---
id: TASK-381
title: Implement ACP accepted artifact version exports for issue 1705
status: Done
assignee: []
created_date: '2026-05-15 15:25'
updated_date: '2026-05-15 19:31'
labels:
  - acp
  - artifacts
  - backend
dependencies:
  - TASK-350
  - TASK-357
  - TASK-369
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1705'
  - 'https://github.com/rmusser01/tldw_server/issues/1532'
  - 'https://github.com/rmusser01/tldw_server/issues/1703'
  - 'https://github.com/rmusser01/tldw_server/issues/1706'
  - 'https://github.com/rmusser01/tldw_server/issues/1707'
documentation:
  - Docs/Product/Traceable_Work_Product_Artifact_Contract.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1705: export accepted traceable work-product artifact versions without losing artifact identity, version identity, source lineage, review state, ACP producer references, or export timestamp. Start with Markdown, HTML, and JSON exports for the golden-path workspace brief. Reuse the existing workspace_artifacts storage/API foundation and file/output artifact export patterns where practical; do not create an ACP-only artifact model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Accepted artifact versions can be exported through a backend contract for Markdown, HTML, and JSON.
- [x] #2 Export metadata preserves artifact id, artifact version id, workspace id, source lineage, review state, producer references, export format, and export timestamp.
- [x] #3 Non-accepted artifact states fail closed with a contextual error instead of exporting silently.
- [x] #4 Export references are recorded back onto the workspace artifact without losing existing refs.
- [x] #5 Focused tests cover Markdown, HTML, JSON, non-accepted states, and metadata identity round-tripping.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add workspace artifact export request/response schemas and route contract for Markdown, HTML, and JSON accepted-version exports. 2. Implement a focused export helper/service that renders from the exact accepted artifact version and preserves artifact/version/workspace/source/review/producer metadata. 3. Persist export_refs back to the workspace artifact while preserving existing references. 4. Update traceable artifact docs and verify with focused pytest, ruff/py_compile, Bandit, and git diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline before implementation: python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q passed with 64 passed and 5 warnings. Plan file: Docs/superpowers/plans/2026-05-15-acp-artifact-exports-1705-plan.md.

Implemented accepted-version export schemas, renderer, POST endpoint, and export_refs persistence without content version bumps. Red run: workspace_artifact_export tests failed with 404 before route implementation. Green runs: export slice passed, then workspace API plus ChaChaNotes DB slice passed with 66 passed.

PR #1726 review sweep: addressing Qodo docstring/logging/version-row/markdown-marker/concurrency findings plus Gemini direct-version lookup and HTML export comment.

PR #1726 review sweep addressed direct artifact version lookup, helper docstrings, contextual export error logging, base64 Markdown metadata marker, rendered HTML export bodies, transactional export_refs persistence, and missing version snapshot conflict handling. Verification: export slice passed (3 selected); broader workspace DB/API suite passed (67 passed, 5 warnings); Ruff passed on non-legacy touched files; compileall passed; Bandit touched scope reported 0 findings; git diff --check passed. Full ChaChaNotes_DB.py Ruff still reports existing baseline findings outside this patch.

Late CodeRabbit review sweep addressed workspace lookup error mapping by moving _require_workspace into the export try block and centralized WorkspaceArtifactExportStateError in app/core/exceptions.py. Existing race and missing-version comments were already addressed by the transactional CAS-style export_refs update and loud missing-version conflict.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented ACP accepted artifact version exports for issue #1705. Added Markdown, HTML, and JSON export contract for accepted workspace artifact versions, fail-closed non-accepted state handling, traceability-preserving export payloads, and append-only export reference persistence. Updated the artifact contract and ACP PRD docs; focused pytest, ruff on changed non-DB Python files, compileall, Bandit, and git diff checks passed.

PR #1726 review feedback addressed with direct version fetching, base64-safe Markdown metadata, rendered HTML export bodies, contextual logging, transactional export_ref persistence, missing-version conflict handling, and regression coverage for comment-boundary metadata plus missing artifact-version snapshots.

Late CodeRabbit review feedback was also addressed by mapping workspace lookup failures through the export endpoint error block and moving the export-state exception into app/core/exceptions.py. Fresh verification after these changes passed: export slice 3 passed, broader workspace DB/API suite 67 passed, Ruff on non-legacy touched files, compileall, Bandit 0 findings, and git diff --check.
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
