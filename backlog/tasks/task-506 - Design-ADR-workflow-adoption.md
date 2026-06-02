---
id: TASK-506
title: Design ADR workflow adoption
status: In Progress
labels:
- docs
- process
- adr
modified_files:
- Docs/superpowers/specs/2026-06-02-adr-workflow-adoption-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a repo-local Architecture Decision Record workflow for tldw_server that integrates with AGENTS.md, Superpowers specs/plans, Backlog.md tracking, and a staged authoritative ADR backfill.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design spec captures ADR governance, template/index structure, workflow integration, staged migration plan, and scope guardrails.
- [ ] #2 Spec includes an ADR Assessment section for the adopted workflow.
- [ ] #3 Spec review loop is completed or documented if unavailable.
- [ ] #4 User review gate is reached before implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design spec written and reviewed. Spec review iteration 1 found scope ambiguity and backfill-status ambiguity. Spec was revised to split Stage 1 implementation from the broader migration program, clarify that backfill is metadata rather than status, and make Stage 1 seed ADRs deterministic. Spec review iteration 2 approved with no blocking issues. User then requested a cleanup review pass before implementation planning. Cleanup patch updated stale status text, required substantial implementation plans to include or reference ADR assessment, and named the default decision inventory path as Docs/ADR/inventory/YYYY-MM-DD-decision-inventory.md. Spec review iteration 3 approved with no blocking issues. Advisory implementation-plan notes: make 'substantial spec, plan, or PR' operational by pointing to the ADR trigger list, and name follow-up Backlog tasks for inventory/backfill and global Superpowers review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
