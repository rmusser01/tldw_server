---
id: TASK-12988
title: Design Web_Scraping Phase 4 extraction and article orchestration
status: Done
created_date: 2026-07-26 23:29
labels:
- web-scraping
- refactor
- phase-4
- design
priority: High
references:
- Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md
- Docs/superpowers/specs/2026-07-14-web-scraping-phase-3-governed-preflight-package-design.md
- Docs/Design/WebScraping_Refactor_Import_Inventory.md
- https://github.com/rmusser01/tldw_server/pull/2752
documentation:
- Docs/superpowers/specs/2026-07-26-web-scraping-phase-4-extraction-article-orchestration-design.md
modified_files:
- Docs/superpowers/specs/2026-07-26-web-scraping-phase-4-extraction-article-orchestration-design.md
updated_date: 2026-07-26 23:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved Phase 4 design for moving article extraction and governed single-page orchestration out of Article_Extractor_Lib.py without losing preflight analyzers, compatibility contracts, enhanced-scraper behavior, or security controls. Define four reviewable delivery units (4A-4D) and explicitly defer crawl/jobs, search providers, and wrapper removal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design defines layered extraction and article-orchestration boundaries with dependency direction and no imports from new packages to legacy wrappers.
- [x] #2 The design preserves governed preflight, per-fetch egress enforcement, public compatibility contracts, strategy-specific fields, and enhanced-scraper behavior.
- [x] #3 The design specifies default regex enrichment, explicit strategy-order compatibility, cancellation/event-loop behavior, bounded regex execution, and the verified service keyword fix.
- [x] #4 The design defines Phase 4A-4D delivery units, test matrices, security gates, exact-base failure handling, and explicit out-of-scope work.
- [x] #5 The written spec passes placeholder, consistency, ambiguity, and scope self-review and is committed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Capture the user-approved architecture, data flow, compatibility rules, error/security behavior, migration slices, test strategy, scope exclusions, and completion criteria. Self-review the final spec before committing it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Phase 3 merged in PR #2752. Latest dev still contains multiple unrelated TASK-12970 records, so Phase 4 must use an independently allocated unique ID rather than extending the ambiguous Phase 3 parent.
The user approved the complete design after section-by-section review. Independent self-review identified five issues: cross-event-loop executor admission, browser interception/DNS guarantees, preflight payload wording, deterministic public error sanitization, and an over-broad final delivery unit. The spec now uses a process-wide BoundedSemaphore with cancellation-aware bounded admission, defines interception coverage and its URL-validation-only DNS guarantee, clarifies payload attachment, lists stable public error codes, and redistributes consumer migrations across 4A-4C. A final pass also documents Playwright DNS rebinding as a residual risk rather than inventing an undefined pinning requirement. Verification: Phase 1 contract baseline passed (19 tests); ASCII and placeholder scans were clean; the final staged diff is checked separately before commit. Bandit is not applicable because this task changes only design and Backlog documentation. No blockers remain; implementation requires a separate reviewed plan and child tasks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and reviewed the Phase 4 design for extracting shared leaf helpers, the canonical extraction package, and governed single-page article orchestration while retaining the pre-scrape analyzer, compatibility surfaces, enhanced-scraper behavior, per-dispatch egress controls, and deferred crawl/search scope. The design defines four sequential merge units, explicit approved behavior changes, deterministic security/error contracts, cancellation and cross-loop executor semantics, a full test matrix, and completion gates.
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
