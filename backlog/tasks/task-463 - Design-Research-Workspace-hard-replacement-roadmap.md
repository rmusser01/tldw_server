---
id: TASK-463
title: Design Research Workspace hard replacement roadmap
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 19:10'
labels:
  - design
  - webui
  - research-workspace
  - workspace
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create umbrella design spec for replacing Workspace Playground with server-backed /research-workspace, aligned with unified Workspace model, sharing, MCP/ACP/sandbox, migration safety, source status, and phased roadmap A→D→B→C.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reviewer-approved design spec captures the hard /research-workspace replacement roadmap and user-requested review clarifications.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design/spec phase only. Umbrella roadmap created for hard replacing Workspace Playground with server-backed Research Workspace at /research-workspace, aligned with unified Workspace model, sharing, MCP, ACP, sandbox, migration safety, source status, and phased roadmap A -> D -> B -> C.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec review loop completed. First review found three planning blockers: Phase A capability fields too open, migration legacy storage inventory unresolved, and access enforcement too broad. Spec was patched to add the Phase A minimum capability contract, legacy store inventory/schema-mapping gate, and action-specific enforcement. Second review approved. Advisory clarifications applied for unknown capability states, UI-only legacy key classification, and tombstone/local UI preference scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

User-review follow-up completed. Added clarifications for phased implementation planning boundaries, capability mapping, fail-closed unknown governance state, all-payload-class migration deletion eligibility, chunk/object integrity validation, auditable user-acknowledged discard, concrete workspace picker contract, and local-only metrics/export semantics. Re-ran spec-document-reviewer loop; final result: Approved.

Verification recorded: git diff --check passed for the spec and Backlog task; final spec-document-reviewer result was Approved. Bandit and code tests are not applicable because this patch changes documentation/task records only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created reviewer-approved umbrella design spec for replacing Workspace Playground with server-backed /research-workspace, aligned with unified Workspace model including sharing, MCP, ACP, sandbox, migration safety, source status, and phased roadmap A -> D -> B -> C. No code implementation yet; this is design/spec output only.
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
