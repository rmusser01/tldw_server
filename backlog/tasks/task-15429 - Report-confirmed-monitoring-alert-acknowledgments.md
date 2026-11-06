---
id: TASK-15429
title: Report confirmed monitoring alert acknowledgments
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The dashboard and monitoring view need to distinguish reading an alert from acknowledging it, and report confirmed bulk write outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A read-only alert remains actionable until it has a confirmed acknowledgment.
- [x] #2 The dashboard interprets the API list response and reports failed or partial bulk results truthfully.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: exercise normalizer and dashboard with synthetic API responses, implement confirmed bulk results, run focused tests and review. ADR not required; the public monitoring API contract is already established.
The monitoring endpoint exposes separate read and acknowledgment fields. Bulk results validate the response status, ID and merged item timestamp, then refresh the list. Dashboard refresh now uses the shared in-flight guard. Reviewed implementation and scoped diff; no API or database changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
24 focused checks passed and Admin TypeScript passed. Wider monitoring/dashboard run: 116 passed, three existing failures reproduced on dev (two AlertsBanner assertions and the custom range labels). No live backend, full build or browser UAT. Bandit applies to Python changes; this change is TypeScript only. Pending requester-written change summary and merge review.
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
