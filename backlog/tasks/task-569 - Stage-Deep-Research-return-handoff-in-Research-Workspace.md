---
id: TASK-569
title: Stage Deep Research return handoff in Research Workspace
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-30 23:55'
labels:
  - research-workspace
  - deep-research
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2178'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Parse Deep Research return parameters on /research-workspace and surface a bounded handoff in Research Workspace Studio so completed Deep Research runs have a visible return target before full bundle import lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /research-workspace parses research_run_id and source artifact return parameters without mutating unrelated workspace state.
- [x] #2 A return URL whose source_workspace_id matches the current workspace opens/focuses Studio and displays the source artifact label plus run ID.
- [x] #3 Return parameters for a different workspace are ignored so they do not create or display a handoff in the current workspace.
- [x] #4 Focused route tests cover valid return context and mismatched workspace handling.
- [x] #5 Verification records focused UI tests and Bandit rationale for frontend-only TypeScript changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review fix: replace Research Workspace URL query reads with React Router location.search so HashRouter URLs are supported. Add regression coverage for a hash-style /research-workspace?... URL before changing production code, then rerun focused tests and TypeScript.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added bounded Deep Research return URL parsing for Research Workspace, matched return contexts to the active workspace, focused Studio for valid returns, and surfaced a compact handoff banner with the source artifact label and research run ID. Review fix for PR #2178: added HashRouter query extraction so mobile tab state, shared-workspace parsing, and Deep Research return context work when route params live under window.location.hash. Added route-level regression coverage for matching/mismatched returns and HashRouter tab/return URLs. PR: https://github.com/rmusser01/tldw_server/pull/2178. Verification: bunx vitest run src/components/Option/ResearchWorkspace/__tests__/research-workspace-route-state.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx (25 tests passed); NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit -p tsconfig.json (passed); git diff --check (passed). Bandit was not run because the touched production/test files are frontend TypeScript/TSX only.
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
