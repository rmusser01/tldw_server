---
id: TASK-13139
title: Coordinate agent-native web research quality and provenance program
status: In Progress
created_date: 2026-08-28 00:56
labels:
- web-research
- chatbooks
- security
- planning
- donsetch-review
priority: High
milestone: Agent-Native Web Research Quality and Provenance
references:
- https://github.com/dondai44423/donsetch
- TASK-12964
- TASK-13100
- TASK-2354
- TASK-2359
- TASK-2360
documentation:
- Docs/superpowers/specs/2026-08-27-agent-native-web-research-quality-provenance-roadmap.md
- Docs/superpowers/plans/2026-08-27-web-retrieval-quality-baseline.md
- Docs/superpowers/plans/2026-08-27-browser-transport-safety-gate.md
updated_date: 2026-08-28 05:26
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Coordinate a current-dev improvement program informed by the DonSeTch comparison. Reuse the existing governed Web_Scraping pipeline, MCP web tools, Jobs, Media ingestion chokepoint, and Chatbooks source_refs contract. Own the cross-program roadmap and dependency map; each implementation remains a focused child or an existing authoritative task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The roadmap records the current-dev baseline and removes work already delivered by Web_Scraping Phase 3/4 and MCP web.fetch.
- [ ] #2 Every retained improvement is represented by one focused child task or an existing authoritative Backlog task, with parked experiments clearly separated from delivery work.
- [ ] #3 TASK-12964 owns Research HTML-to-Media handoff design and TASK-13100 owns cookie remediation; this program creates no duplicate ingestion, crawl-Jobs, or cookie-vault task.
- [ ] #4 Ownership is explicit: general Web_Scraping contracts own reusable search/retrieval primitives, MCP owns agent-facing composition, Media is the only durable web snapshot owner, and Chatbooks exports Media provenance.
- [ ] #5 Detailed implementation plans are written just in time for the next approved wave after prerequisites are current, rather than frozen for all waves up front.
- [ ] #6 Security release gates cover browser DNS rebinding, credential isolation, bounded output, and secret-free provenance before affected capabilities are broadened.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Publish the current-dev reconciliation roadmap and focused child backlog.
2. Update authoritative prerequisite tasks with the minimal cookie and Phase 2B ownership decisions.
3. After roadmap review, write only the Wave 0 implementation plan(s), then plan later waves just in time as dependencies complete.
4. Keep this parent current with verification, plan, and PR links as each reviewable unit lands.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created after auditing the earlier planning commit against current origin/dev. TASK-12125 is already the completed Chat Macros planning task and must not be reused. The earlier branch-local TASK-12125.* records are invalid and must not be merged. This planning slice changes Backlog records and documentation only; runtime tests and Bandit are not applicable.

2026-08-27 reconciliation result: rebuilt the program with 11 focused children (TASK-13139.1 through TASK-13139.11) and subsequently rebased the planning branch onto origin/dev 9fd2246157ce8a32ae6a6691a75efab788229f77. Reused and updated TASK-12964 and TASK-13100 rather than duplicating their ownership. The roadmap removes already-delivered browser interception and HTTP-to-Playwright escalation work, places fusion in general Web_Scraping contracts, makes Media the sole durable snapshot owner, limits MCP revalidation to public credentialless process-local caching, splits crawl/resilience work, and parks PDF/comparator/fuzz work.

Planning verification: all 12 new task IDs are unique across inspected repository worktrees and parse through Backlog CLI; the parent reports exactly 11 children with an acyclic dependency order; local documentation/code references exist; the roadmap and tasks contain no placeholders or stale pinned comparator release; git diff --check passes. Runtime tests and Bandit are not applicable because this slice changes Markdown Backlog/spec/plan records only.

2026-08-27 approval checkpoint: the user approved the reconciled roadmap. Wrote only the two just-in-time Wave 0 plans: Docs/superpowers/plans/2026-08-27-web-retrieval-quality-baseline.md and Docs/superpowers/plans/2026-08-27-browser-transport-safety-gate.md. Both child tasks remain To Do; no runtime implementation has started.
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
