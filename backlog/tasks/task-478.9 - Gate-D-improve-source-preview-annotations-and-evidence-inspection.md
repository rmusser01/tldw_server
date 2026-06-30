---
id: TASK-478.9
title: 'Gate D: improve source preview, annotations, and evidence inspection'
status: Done
labels:
- research-workspace
- uat
- gate-d
- source-preview
- annotations
- citations
priority: Medium
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
documentation:
- Docs/superpowers/plans/2026-05-25-research-workspace-source-preview-context-plan.md
modified_files:
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
- tldw_Server_API/app/api/v1/schemas/workspace_schemas.py
- tldw_Server_API/app/core/Workspaces/status_projection.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_preview_context_api.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py
- apps/packages/ui/src/services/tldw/domains/workspace-api.ts
- apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts
- apps/packages/ui/src/services/tldw/openapi-guard.ts
- apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible gap: opening a source/annotation modal showed metadata and annotation fields but no useful ingested source-content preview. Annotation creation worked, but the user could not inspect what content was actually captured or citeable from that source.

User goal: inspect a source, verify captured text/snippets, add notes/annotations, and connect those notes to later evidence/citation workflows.

Scope:
- Add source-content preview for ingested text or extracted chunks with clear loading/error/empty states.
- Show citation/evidence snippets or chunk metadata when available, respecting readiness/status from TASK-478.3.
- Validate annotation create/edit/delete/read behavior and persistence if those actions exist or should exist.
- Ensure preview handles large sources with pagination, search-within-source, or bounded snippets rather than dumping unbounded content.
- Add tests for preview available, preview pending, extraction failed, large source, and annotation persistence paths.

Acceptance criteria:
- A user can open a workspace source and verify at least representative captured content or a precise reason content is unavailable.
- Annotation controls do not hide or replace source inspection.
- Evidence/citation snippets are linked to source identity and readiness where supported.
- Live CDP/Playwright validation covers preview and annotation behavior.

Depends on: TASK-478.3 for readiness semantics.
Parallelization: can run in parallel with acquisition/layout/onboarding once status fields are stable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design review completed before implementation. Decision: add one canonical workspace page-context endpoint for shell/readiness/capability/service status, while keeping source preview content in a bounded source-detail endpoint to avoid oversized payloads and preserve failure isolation. Plan saved at docs/superpowers/plans/2026-05-25-research-workspace-source-preview-context-plan.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented source preview and evidence inspection for Research Workspace Gate D.

Design review and fixes:
- Validated against a live backend and WebUI using Playwright/CDP-style automation.
- Kept the context design split: one workspace context envelope for page status/readiness/capabilities, and one bounded source preview endpoint for content/evidence to avoid oversized page payloads.
- Fixed a design trust gap where context partial_errors were stored but invisible by surfacing them as a compact Sources-pane warning icon instead of adding another full-width banner.
- Localized preview-unavailable copy.
- Fixed stale active workspace_source_ingest jobs masking already-extracted media by making status projection prefer extracted/queryable media readiness over stale lifecycle jobs.

User-visible behavior:
- Source preview modal now shows captured content, citation-ready state, bounded chunk evidence snippets, loading/error/unavailable states, retry, and browser-local annotations.
- Local annotations are explicitly labeled as browser-local workspace state and persist across modal remounts.

Live validation:
- Uploaded /private/tmp/task4789-live-source.md through /research-workspace.
- media/add returned 200; workspace source creation returned 201; context and preview endpoints returned 200.
- Source row reached READY; modal showed captured content, chunk snippets, and persisted annotation.
- Screenshots: /private/tmp/task4789-research-workspace-initial.png, /private/tmp/task4789-add-source-modal.png, /private/tmp/task4789-upload-tab.png, /private/tmp/task4789-source-preview-live.png.

Verification:
- Backend focused tests: 13 passed.
- Frontend focused tests: 55 passed.
- OpenAPI verification: passed with existing reviewed exceptions.
- Bandit touched backend scope: no findings.
- git diff --check: clean.
- TypeScript full check remains blocked by unrelated pre-existing WatchlistsPlaygroundPage.tsx syntax errors.
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
