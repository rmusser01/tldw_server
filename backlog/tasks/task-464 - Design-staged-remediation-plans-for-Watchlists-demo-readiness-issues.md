---
id: TASK-464
title: Design staged remediation plans for Watchlists demo-readiness issues
status: Done
labels:
- watchlists
- design
- demo-readiness
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
Created the staged Watchlists demo-readiness remediation spec at `Docs/superpowers/specs/2026-05-20-watchlists-demo-remediation-staged-plans-design.md`.

The spec separates urgent demo rescue from first-time workflow completion, durable audio artifacts, operator recovery, and power-user throughput while preserving existing Watchlists news, OSINT, CTI, and advanced workflows inside `/watchlists`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and locally reviewed the staged remediation spec for verified Watchlists demo-readiness blockers. The spec maps each issue to staged ownership, dependencies, gates, and verification expectations, and it became the source document for the follow-on implementation plan.
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
