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
updated_date: 2026-07-27 15:41
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
- [x] #6 The revised design explicitly allowlists guarded direct-browser routing, sanitizes moved observability fields, defines the direct-browser compatibility matrix, and bounds HTTP/browser acquisition.
- [x] #7 The revised design specifies executor generation synchronization, deterministic public failure mappings, predecessor-fixture provenance, and active-loop rejection for both sync entry points.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Capture the user-approved architecture, data flow, compatibility rules, error/security behavior, migration slices, test strategy, scope exclusions, and completion criteria. Self-review the final spec before committing it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Phase 3 merged in PR #2752. Latest dev still contains multiple unrelated TASK-12970 records, so Phase 4 must use an independently allocated unique ID rather than extending the ambiguous Phase 3 parent.
The user approved the complete design after section-by-section review. Independent self-review identified five issues: cross-event-loop executor admission, browser interception/DNS guarantees, preflight payload wording, deterministic public error sanitization, and an over-broad final delivery unit. The spec now uses a process-wide BoundedSemaphore with cancellation-aware bounded admission, defines interception coverage and its URL-validation-only DNS guarantee, clarifies payload attachment, lists stable public error codes, and redistributes consumer migrations across 4A-4C. A final pass also documents Playwright DNS rebinding as a residual risk rather than inventing an undefined pinning requirement. Verification: Phase 1 contract baseline passed (19 tests); ASCII and placeholder scans were clean; the final staged diff is checked separately before commit. Bandit is not applicable because this task changes only design and Backlog documentation. No blockers remain; implementation requires a separate reviewed plan and child tasks.
Written-spec review identified seven changes required before implementation planning: explicitly allowlist guarded direct-browser behavior, resolve URL-label observability contradictions, define browser compatibility fields and cross-origin credential behavior, bound acquired article content, define executor generation/reset synchronization, specify deterministic error and differential-baseline contracts, and name scrape_article_sync in the active-loop guard. The task is reopened to revise and recommit the design.
Revision implemented in the spec: approved behavior changes now total eleven; direct browser routing is correctly identified as new; moved URL/raw-error observability fields are explicitly removed; browser field parity is tabulated; HTTP, browser-transfer, and rendered-HTML limits are defined with optional bounded simple-fetch support; executor lifecycle uses locked generations and terminal shutdown; public error fields/codes and predecessor fixtures are deterministic; scrape_article_sync shares the before-side-effects loop guard. The document status is set to awaiting final review.
The user gave final written-spec approval on 2026-07-27. The specification status is now Approved for implementation planning; implementation planning will be tracked as a separate reviewable Backlog task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and user-approved the Phase 4 extraction and article-orchestration design. The final specification defines canonical extraction and orchestration boundaries, preserves governed preflight and public compatibility, records eleven explicitly approved behavior changes, and adds concrete contracts for browser egress controls, acquisition limits, executor lifecycle synchronization, sanitized observability, deterministic failures, predecessor fixtures, and active-loop rejection. Focused Phase 1 contracts passed (19 tests); document consistency, ASCII, placeholder, and diff checks were clean. Bandit was not applicable because only design and Backlog documentation changed. No known blockers remain before implementation planning.
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
