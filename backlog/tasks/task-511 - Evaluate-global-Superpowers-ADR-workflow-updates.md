---
id: TASK-511
title: Evaluate global Superpowers ADR workflow updates
status: Done
labels:
- docs
- process
- adr
documentation:
- Docs/superpowers/specs/2026-06-03-global-superpowers-adr-workflow-design.md
modified_files:
- Docs/superpowers/specs/2026-06-03-global-superpowers-adr-workflow-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After repo-local ADR workflow validation, decide whether to update global Superpowers skills so brainstorming, writing-plans, and verification workflows consider ADR assessment across repositories.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review repo-local ADR workflow outcomes before proposing global skill edits.
- [x] #2 Identify which skill files would change and what trigger wording they need.
- [x] #3 Produce a separate design/spec before modifying global Superpowers files.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan created: Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md. Depends on repo-local ADR workflow outcomes from TASK-509/TASK-510 before global Superpowers edits are proposed.

Evidence reviewed:
- ADR-001 established the repo-local ADR workflow and ADR assessment requirement.
- TASK-509 produced the decision inventory, classification rules, and concrete owner-review defaults.
- TASK-510 required one completed owner-reviewed child backfill slice before global Superpowers evaluation.
- TASK-514 completed the Workspace/WebUI pilot and produced ADR-007, ADR-008, and ADR-009.

Candidate global skill files identified:
- `$CODEX_HOME/superpowers/skills/brainstorming/SKILL.md`
- `$CODEX_HOME/superpowers/skills/writing-plans/SKILL.md`
- `$CODEX_HOME/superpowers/skills/verification-before-completion/SKILL.md`

No global Superpowers files were modified. The recommendation is to create a separate global skill update task using `superpowers:writing-skills` after owner review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed repo-local ADR workflow outcomes from TASK-509, TASK-510, and TASK-514. Produced Docs/superpowers/specs/2026-06-03-global-superpowers-adr-workflow-design.md before modifying any global Superpowers files. The spec identifies candidate global updates for brainstorming, writing-plans, and verification-before-completion, with repo-agnostic trigger wording and owner-review defaults. Verification: rg confirmed the spec contains Candidate Skill Changes, brainstorming, writing-plans, verification-before-completion, and Recommendation sections. Bandit skipped: documentation-only task; no Python/code paths touched. Known blockers: none.
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
