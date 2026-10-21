---
id: TASK-45.44.16
title: Design token, color, radius, and layout drift guards
status: In Progress
assignee: []
created_date: 2026-05-14 03:20
labels:
- design-system
- webui
- extension
- governance
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/1673
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub governance issue. Closure requires a durable guard, documented policy, CI path, component ownership decision, documentation artifact, or visual QA checklist as specified by the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The linked GitHub issue owns public status.
- [ ] #2 Backlog notes record PR links and verification evidence.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-20: User approved repo-wide UI drift reporting with no blocking. Writing a design-only spec for a standalone verifier covering shared UI, WebUI, extension, and admin UI; existing and new findings stay visible. ADR check: ADR required: yes; proposed Docs/ADR/079-advisory-ui-design-drift-reporting.md because this establishes lasting UI conventions and nonrequired reporting policy. No runtime, verifier, or CI changes in this design slice.
Design artifact: Docs/superpowers/specs/2026-10-20-design-system-drift-reporting-design.md; proposed ADR-079 linked from ADR index and main design-system contract. Validation: 78 relative Markdown links resolve across four documents; no TODO/TBD/FIXME placeholders; backlog-py normalize --check passes; git diff --check passes. Bandit and runtime/browser tests skipped because this slice changes only Markdown. Written-spec approval and implementation planning remain pending; issue #1673 remains open.
Read-only independent review found no substantive contradictions or scope issues in the spec and ADR. Parser feasibility and actual CI behavior remain implementation-plan verification work; neither is claimed complete.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
