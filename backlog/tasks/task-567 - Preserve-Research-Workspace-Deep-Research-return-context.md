---
id: TASK-567
title: Preserve Research Workspace Deep Research return context
status: Done
documentation:
- Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md
- Docs/superpowers/plans/2026-05-30-research-workspace-literature-workproducts-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next narrow Deep Research bridge slice after the literature work-products follow-up: carry Research Workspace origin metadata (workspace/artifact/template/title) through Deep Research launch URLs and /research run creation so later bundle import/display work can target the correct Research Workspace artifact without guessing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deep Research launch paths from Research Workspace literature artifacts include bounded origin metadata for workspace ID, artifact ID, artifact template, and artifact title when available.
- [x] #2 The /research console parses Research Workspace origin metadata, preserves it after manual or autorun run creation, and exposes a return link/context for the selected run.
- [x] #3 Focused route, Research Workspace launch, and research console tests cover the handoff metadata and pass.
- [x] #4 Bandit is skipped with rationale if only frontend TypeScript files are touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added bounded Research Workspace source metadata to Deep Research launch URLs and return URLs.
- Passed focused Vitest coverage after rebasing onto latest origin/dev: package route helper, Research Workspace StudioPane literature work products, and /research run console.
- Addressed Qodo review feedback by trimming bounded route params after slicing so launch/return URL builders match the /research parser normalization.
- Bandit skipped: touched files are frontend TypeScript/TSX tests and UI code only; no Python execution path changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Research Workspace literature artifacts now launch Deep Research with workspace/artifact/template/title origin metadata. The /research console parses that context, preserves it after run creation, and exposes a Back to Research Workspace link with the source artifact label.
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
