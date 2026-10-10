id: TASK-13534
title: Fast-follow PR #3210 review findings (storage index safety, researchActions staleness, sidebar debounce)
status: In Progress
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
- [ ] #1 Migration failure in ModelDb is never cached as an empty index; covered by a storage-failure regression test
- [ ] #2 researchActions cache invalidates when the builder closure changes
- [ ] #3 MediaReviewFilterSidebar keyword search is debounced with stale-input dropping
- [ ] #4 Copilot stub comments match actual behavior; chunk-load failures log; orchestrator fetch strategy is compile-time required
- [ ] #5 Touched test suites green
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip (N/A: TypeScript only)
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
