---
id: TASK-211
title: Design MCP Hub workflow-first control panel UX
status: Done
assignee:
  - Codex
created_date: '2026-05-10 02:39'
updated_date: '2026-05-10 02:45'
labels:
  - ux
  - mcp-hub
  - webui
  - extension
  - design
dependencies: []
references:
  - apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx
  - apps/packages/ui/src/components/Option/MCPHub
  - apps/packages/ui/src/services/tldw/mcp-hub.ts
documentation:
  - docs/superpowers/specs/2026-03-27-mcp-hub-navigation-parity-design.md
  - Docs/MCP/mcp_hub_management.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an implementation-ready design spec for replacing the current object-centric MCP Hub tab row with a workflow-first WebUI/extension control panel. Scope is design documentation only: preserve current implementation behavior while defining the future IA, component contract, readiness summaries, deep-link handling, audit drilldown mapping, and verification plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec is written under docs/superpowers/specs with the approved workflow-first IA and staged implementation scope.
- [x] #2 Spec explicitly maps existing MCP Hub child views into Setup, Access, Workspaces, Governance, and Audit workflows.
- [x] #3 Spec covers URL/deep-link compatibility, audit drilldown behavior, readiness/status summaries, and frontend-first aggregation boundaries.
- [x] #4 Spec includes implementation sequencing and focused verification plan for shared WebUI/extension behavior.
- [x] #5 Spec review loop is completed and issues are either resolved or documented.
- [x] #6 Only the design spec and associated Backlog task changes are committed; unrelated dirty worktree files remain untouched.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Approved design-doc plan:
1. Write an implementation-ready design spec for a workflow-first MCP Hub control panel in docs/superpowers/specs/2026-05-10-mcp-hub-workflow-first-control-panel-design.md.
2. Preserve current implementation scope in the spec: no code changes yet, reuse existing MCP Hub child components in the first implementation slice, and avoid backend changes unless frontend aggregation proves insufficient.
3. Specify the top-level workflow IA: Setup, Access, Workspaces, Governance, Audit.
4. Map current child views into workflow groups and define URL/deep-link behavior plus audit drilldown mapping.
5. Define readiness/status summaries, empty-state behavior, and first-use guidance for the shared WebUI/extension surface.
6. Include implementation sequencing, risks, non-goals, and focused verification requirements.
7. Run the brainstorming spec-document-reviewer loop and resolve any blocking review findings.
8. Commit only the spec and Backlog task updates, leaving unrelated dirty files untouched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created for the approved workflow-first MCP Hub UX design. User selected the heavier redesign direction over a staged control-center overlay or light copy/empty-state polish.

Spec review loop completed after three passes. First pass approved with advisory clarifications; second pass found planning ambiguities in readiness staging and binding inputs; third pass approved after those were fixed. Placeholder scan found no TODO/TBD/FIXME/PLACEHOLDER markers. Bandit is not applicable because only Markdown design/task files were changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created docs/superpowers/specs/2026-05-10-mcp-hub-workflow-first-control-panel-design.md for the approved workflow-first MCP Hub redesign.

What changed:
- Added an implementation-ready design spec that maps existing MCP Hub object views into Setup, Access, Workspaces, Governance, and Audit workflows.
- Defined route/query state, legacy internal view-key compatibility, audit drilldown behavior, frontend-first readiness summary boundaries, accessibility/responsive requirements, implementation staging, and focused WebUI/extension verification.
- Clarified after review that Stage 1 is the workflow shell plus URL/drilldown behavior, while Stage 2 owns readiness/status aggregation and richer first-use guidance.

Why:
- The current 11-tab MCP Hub layout exposes backend object types as peer navigation and hides the real user workflow. The spec preserves current capabilities while planning a workflow-first control panel that better supports first-time setup and governance/admin work.

Verification:
- Placeholder scan: no TODO/TBD/FIXME/PLACEHOLDER markers.
- Spec review loop: approved on the third pass after resolving readiness-staging and credential-binding input ambiguities.
- git diff --cached --check: passed.
- Bandit skipped because this task touched only Markdown design/task files.

Scope:
- No implementation code changed.
- Unrelated existing dirty files and other untracked backlog tasks were left untouched.
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
