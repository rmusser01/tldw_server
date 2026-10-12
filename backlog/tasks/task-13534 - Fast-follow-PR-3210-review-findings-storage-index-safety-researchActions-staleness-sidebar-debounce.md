id: TASK-13534
title: Fast-follow PR #3210 review findings (storage index safety, researchActions staleness, sidebar debounce)
status: Done
labels:
- frontend
- review-followup
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements the three Important findings (plus four Minor quick-hits) from the code review of merged PR #3210 (TASK-13511), requested via the requesting-code-review workflow on 2026-10-09.

1. db/models.ts ensureIndex cached a failed one-time migration as an empty index; a later persistRecords would write [] over the real index and permanently orphan every migrated model record. Fix: propagate migration failure (never cache []), log it; regression test makes storage set fail during migration and asserts loud failure + no index clobber + recovery on a fresh instance.
2. PlaygroundChat researchActions cache keyed on (messageId, metadataExtra, followUpAvailable) but the actions close over buildMessageResearchActions (linked research runs + attach callbacks), so rows could keep stale closures/labels. Fix: builder identity joins the cache validity check.
3. useMediaSearch.loadKeywordSuggestions (feeding MediaReviewFilterSidebar's per-keystroke onSearch) hit /api/v1/media/keywords on every key with no debounce — the surface the PR #3210 plan named but only fixed on ReviewPage/ViewMediaPage. Fix: 300ms debounce wrapper with superseded-input dropping; empty input and mount/results-sync stay immediate.

Minors: copilot stub comment now matches the actual frameId-in-payload delivery; chunk-load failure logs a breadcrumb; pollTrackedIngestJobs options require exactly one fetch strategy at compile time (union type).

ADR check (adr-assessment skill): ADR required: no — no durable architecture rule changes; these are correctness/robustness fixes within existing patterns (the storage/indexing and fetch-strategy shapes are already governed by PR #3210's design and TASK-13526 tracks the structural work).

NOTE: created manually (backlog CLI absent on this machine; see TASK-13511's record for the history).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Migration failure in ModelDb is never cached as an empty index; covered by a storage-failure regression test
- [x] #2 researchActions cache invalidates when the builder closure changes
- [x] #3 MediaReviewFilterSidebar keyword search is debounced with stale-input dropping
- [x] #4 Copilot stub comments match actual behavior; chunk-load failures log; orchestrator fetch strategy is compile-time required
- [x] #5 Touched test suites green
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
All three Important review findings and the four Minor quick-hits implemented and verified (models.storage 6/6 incl. failure-injection regression, PlaygroundChat 10/10, Review search 21/21, orchestrator 4/4, copilot stub guard 3/3). Merged via PR #3216 (merge commit 2a935383a) after the human change summary.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip (N/A: TypeScript only)
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
