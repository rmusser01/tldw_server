---
id: TASK-467
title: Bootstrap server-backed active Research Workspace and source rows
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 22:16'
labels:
  - frontend
  - backend
  - api
  - research-workspace
  - workspaces
  - sync
  - trust
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure `/research-workspace` creates/upserts the active backend workspace and mirrors visible local source rows before trust/status fetches, so first-run users and migrated/local-cache users do not see unavailable trust state for a workspace that exists locally. Keep this slice one-way and best-effort: upsert the workspace, add missing visible source rows, skip invalid media IDs, and continue status/capability fetching when sync fails. Do not add `/workspace-playground` aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-research-workspace-server-bootstrap-sync-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a Research Workspace server reconciliation helper that upserts the active workspace, lists backend source rows, adds missing valid local sources, skips invalid/duplicate source rows, and returns bounded errors without throwing.
- Wired reconciliation into the Research Workspace trust refresh path before source status and capability fetches. Failures surface as bounded trust warnings while status/capability calls still run.
- Made duplicate workspace source POSTs idempotent in ChaChaNotesDB by returning the existing row for the same workspace/source id. This avoids dev StrictMode/racing bootstrap requests becoming 500s.
- Verified with Playwright/CDP against a live FastAPI backend and Next WebUI: final route stayed `/research-workspace`, no `/workspace-playground` requests occurred, workspace/source bootstrap ran before status/capabilities, and no workspace API 500s were observed. Screenshot: `/tmp/research-workspace-server-bootstrap-cdp.png`.
- Validation commands: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`; `.venv/bin/python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py::test_workspace_source_endpoints_happy_path tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py -q`; `node /private/tmp/research-workspace-cdp-validation.cjs`; `.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_task467.json`.

- Focused code review found one P2 issue: reconciliation errors were not actually bounded for repeated source-add failures. Added a red/green regression test and capped returned error metadata at five messages with a single omission summary while preserving continued add attempts.
- Post-review verification: focused Vitest now passes 28 tests, backend workspace pytest passes 4 tests, live CDP route/API-order validation passes, Bandit reports 0 findings, and diff checks report no whitespace diagnostics.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Research Workspace now bootstraps its active backend workspace and visible source rows before trust/status projection calls. The backend source add path is idempotent for duplicate workspace/source ids, which keeps first-run and dev-mode repeated bootstrap requests from producing server errors. Post-review, reconciliation diagnostics are bounded while source-add attempts continue. Focused frontend tests, backend workspace tests, live CDP validation, Bandit, and diff checks completed successfully.
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
