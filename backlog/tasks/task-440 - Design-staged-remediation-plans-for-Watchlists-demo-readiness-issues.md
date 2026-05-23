---
id: TASK-440
title: Design staged remediation plans for Watchlists demo-readiness issues
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-21 00:01'
labels:
- watchlists
- design
- demo-readiness
dependencies: []
priority: High
modified_files:
- Docs/superpowers/specs/2026-05-20-watchlists-demo-remediation-staged-plans-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a staged design/spec for addressing the Watchlists demo-readiness issues found in live WebUI/extension QA: template mismatch, audio enqueue failure, misleading health/status, first-time cadence gaps, review-state inconsistencies, and power-user/operator hardening needs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec covers a parallel-track staged remediation strategy for urgent demo rescue, product workflow completion, and power-user/operator hardening.
- [x] #2 Spec maps identified issues to stages, ownership, dependencies, gates, and verification expectations.
- [x] #3 Spec preserves existing news/OSINT/CTI Watchlists workflows and keeps the core MVP inside /watchlists.
- [x] #4 Spec is reviewed before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Metadata reconciliation only; no design/spec content was changed in this cleanup.
- The referenced remediation spec exists at `Docs/superpowers/specs/2026-05-20-watchlists-demo-remediation-staged-plans-design.md`.
- Successor implementation planning and execution records exist: `TASK-441`, `TASK-477`, and `TASK-478`.
- Later remediation addendum `TASK-476` superseded the remaining demo-rescue details and fed the P0 implementation plan.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed stale tracking status. The staged remediation spec exists and the follow-on Watchlists demo-rescue planning/implementation work proceeded through successor tasks, including `TASK-441`, `TASK-476`, `TASK-477`, and `TASK-478`.
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
