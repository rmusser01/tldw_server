---
id: TASK-442
title: Document Persona PRD reconciliation scope split
status: Done
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1902
- Docs/Product/Persona_Agent_Design.md
- Docs/Plans/2026-03-08-persona-garden-design.md
modified_files:
- Docs/superpowers/specs/2026-05-21-persona-prd-reconciliation-design.md
- backlog/tasks/task-442 - Document-Persona-PRD-reconciliation-scope-split.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a design/spec artifact that reconciles the original Persona module PRD into current completion scope and future PRD tracks, linking the future work tracker issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents current Persona module completion scope.
- [x] #2 Spec lists moved-out future PRD tracks and links GitHub issue #1902.
- [x] #3 Spec keeps ordinary chat, workspace defaults, scheduled work, rich avatar, cross-app personalization, tool administration, and multi-agent collaboration out of current completion scope.
- [x] #4 No design-system backlog tasks are touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed and hardened the Persona PRD reconciliation design. Added risk notes and constraints for transcript export redaction/ownership, non-escalating Scopes/Policies editing, safe MCP/tool discovery, evidence-backed shipped-status claims, and a suggested PRD patch sequence. Verification: git diff --check passed on the updated spec. Bandit skipped because this remains documentation-only.
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
