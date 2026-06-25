---
id: TASK-10000
title: Harden Watchlists core review findings
status: Done
assignee: []
created_date: 2026-06-23 21:55
updated_date: 2026-06-25 00:02
labels:
- watchlists
- core
- security
- review-fix
dependencies: []
references:
- Docs/superpowers/plans/2026-06-23-watchlists-core-review-hardening.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix Watchlists core review findings around unsafe regex filters, tenant-aware egress policy checks, cancellation status handling, source ownership enforcement, malformed source scope cleanup, and output enrichment scheduling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Regex filters reject or time-bound unsafe patterns without blocking job evaluation.
- [x] #2 Watchlist jobs and scheduler payloads propagate tenant IDs to egress policy checks.
- [x] #3 Cancelled runs are not swallowed and scheduler status reflects the pipeline result.
- [x] #4 Source tags, groups, seen items, and deletes enforce source/user ownership in the core DB layer.
- [x] #5 Malformed source scopes do not leave runs stuck in running status.
- [x] #6 Output enrichment pending work is scheduled after output creation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Review remediation complete for PR #2482. Branch rebased onto latest origin/dev and Watchlists diff remains scoped. Addressed Qodo comments by documenting B608 suppressions, adding regex timeout exception compatibility, moving enrichment scheduling into core Watchlists code, and routing enrichment through the durable Scheduler with idempotency plus fallback. Verification: focused changed tests passed (6 passed), broader Watchlists focused suite passed (96 passed, 1 skipped), py_compile passed with existing endpoint SyntaxWarnings only, git diff --check passed, Bandit touched app scope reported 0 findings.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-24: Reopened to address PR #2482 review comments after rebasing onto latest origin/dev. Review items: justify new B608 suppressions, move enrichment scheduling out of endpoint, catch regex timeout variants, and route output enrichment through Scheduler with fallback only when unavailable.
2026-06-24: PR #2482 review remediation completed. Rebased branch onto origin/dev, dropping unrelated Claims commits from the PR diff. Added documented B608 rationale comments, regex timeout variant handling, Scheduler-backed output enrichment submission with in-process fallback, and watchlists_enrich_output Scheduler task coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
