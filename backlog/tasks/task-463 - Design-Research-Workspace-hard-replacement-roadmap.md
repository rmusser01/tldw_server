---
id: TASK-463
title: Design Research Workspace hard replacement roadmap
status: In Progress
labels:
- design
- webui
- research-workspace
- workspace
priority: High
documentation:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
modified_files:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create umbrella design spec for replacing Workspace Playground with server-backed /research-workspace, aligned with unified Workspace model, sharing, MCP/ACP/sandbox, migration safety, source status, and phased roadmap A→D→B→C.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design/spec phase only. Umbrella roadmap created for hard replacing Workspace Playground with server-backed Research Workspace at /research-workspace, aligned with unified Workspace model, sharing, MCP, ACP, sandbox, migration safety, source status, and phased roadmap A -> D -> B -> C.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec review loop completed. First review found three planning blockers: Phase A capability fields too open, migration legacy storage inventory unresolved, and access enforcement too broad. Spec was patched to add the Phase A minimum capability contract, legacy store inventory/schema-mapping gate, and action-specific enforcement. Second review approved. Advisory clarifications applied for unknown capability states, UI-only legacy key classification, and tombstone/local UI preference scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created reviewer-approved umbrella design spec for replacing Workspace Playground with server-backed /research-workspace, aligned with unified Workspace model including sharing, MCP, ACP, sandbox, migration safety, source status, and phased roadmap A -> D -> B -> C. No code implementation yet; this is design/spec output only.
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
