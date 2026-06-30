---
id: TASK-465
title: Implement Research Workspace source status and capability projection API
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 20:26'
labels:
  - backend
  - research-workspace
  - workspaces
  - jobs
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-23-research-workspace-source-status-capabilities-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase A backend slice for the Research Workspace replacement roadmap. Add read-only workspace endpoints for source ingestion/indexing readiness and workspace capabilities, grounded in existing workspace sources and conservative fail-closed defaults. Planned endpoints: GET /api/v1/workspaces/{workspace_id}/sources/status and GET /api/v1/workspaces/{workspace_id}/capabilities. Include schemas, focused API tests, docs/plan linkage, and verification. Reference: Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET /api/v1/workspaces/{workspace_id}/sources/status returns per-source lifecycle, readiness dimensions, progress, optional job detail, and summary counts.
- [x] #2 GET /api/v1/workspaces/{workspace_id}/capabilities returns research_workspace capability gates, source summary, workspace services, and allowed action reason codes.
- [x] #3 MCP, ACP, and sandbox are represented as core workspace services with fail-closed defaults until bindings exist.
- [x] #4 Focused API regression coverage is added and existing workspace API tests still pass.
- [x] #5 Touched backend scope passes Bandit with no findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-research-workspace-source-status-capabilities-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a read-computed status projector backed by workspace source membership, optional Media DB readiness, and optional recent media_ingest Jobs. Jobs own in-flight ingestion/extraction/chunking/indexing progress; Media DB owns text/index readiness; capability gates fail closed for MCP/ACP/sandbox until workspace bindings exist. Live route validation used uvicorn with lifespan disabled to isolate API behavior from unrelated startup services.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added first-class backend API contracts for Research Workspace source readiness and workspace capabilities. Source status now distinguishes missing media, extraction/chunking/indexing progress, queryable, partially queryable, failed, and retrying states. Capability response exposes source summary, conservative allowed actions, and service gates for migration, sharing, MCP, ACP, sandbox, and provider readiness. Focused API tests cover readiness/missing media, fail-closed empty workspace capabilities, and active media-ingest Job progress. Verification: focused source-status API tests passed, existing workspace API tests passed, touched backend Bandit scan reported no findings, and live uvicorn HTTP validation succeeded for both endpoints with lifespan disabled to isolate route behavior from unrelated startup services.
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
