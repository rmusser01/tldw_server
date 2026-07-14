---
id: TASK-12968
title: Design Web_Scraping refactor Phase 3 governed preflight package
status: Done
created_date: 2026-07-14 15:40
labels:
- web-scraping
- design
- refactor
- preflight
references:
- Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md
- Docs/superpowers/specs/2026-07-04-web-scraping-phase-2-runtime-policy-boundary-design.md
- Docs/superpowers/plans/2026-07-05-web-scraping-phase-2-runtime-policy-boundary.md
- Docs/Design/WebScraping_Refactor_Import_Inventory.md
modified_files:
- Docs/superpowers/specs/2026-07-14-web-scraping-phase-3-governed-preflight-package-design.md
documentation:
- Docs/superpowers/specs/2026-07-14-web-scraping-phase-3-governed-preflight-package-design.md
updated_date: 2026-07-14 15:46
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and self-review the approved Phase 3 design spec for moving the governed pre-scrape analyzer into Web_Scraping/preflight. The design must preserve public behavior and analyzer functionality while centralizing both scrape consumers behind a typed facade, governed runtime adapters, temporary compatibility shims, deterministic scheduling, and fail-open analyzer behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The spec defines preflight package ownership, dependency direction, facade contracts, options, execution context, and compatibility shims.
- [x] #2 The spec preserves existing config keys, analyzer result keys, scoring, recommendations, advice, optional payload shape, and both scrape consumers.
- [x] #3 The spec defines policy checks, URL-bound authorization, budgets, timeouts, cancellation, browser interception, external-tool handling, cleanup, and redaction.
- [x] #4 The spec includes deterministic no-network testing, compatibility guardrails, optional browser smoke coverage, property invariants, and completion gates.
- [x] #5 The spec contains no placeholders, contradictions, ambiguous failure behavior, or unreviewed scope expansion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved decisions recorded: facade-led physical move into preflight; both scrape consumers migrate in Phase 3; existing PreflightResult/PreflightAdvice contracts are reused; legacy imports remain through temporary explicit shims; primary policy remains blocking while analyzer failures fail open; exact-target authorization is reused and redirects/subrequests receive governed checks; budget limits remain unbounded by default; analyzer order remains deterministic; unexpected analyzer failures are isolated; external-tool behavior uses the approved compatibility transition. Self-review verified package ownership, data flow, cancellation, browser interception, opaque external-tool limitations, atomic budgets, redaction, compatibility, and deterministic testing. Documentation-only task: Bandit is not applicable because no Python code changed. Verification included reference existence, unresolved-marker scan, and staged diff checks before commit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and self-reviewed the Phase 3 governed preflight package design. It defines the preflight facade, typed options and context, URL-bound policy flow, governed HTTP/browser/external-tool adapters, analyzer migration and compatibility shims, centralized advice and payload eligibility, failure/cancellation/redaction semantics, deterministic testing, rollout stages, and completion gates. The design preserves current successful analyzer and scrape behavior while making probe governance and ownership explicit.
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
