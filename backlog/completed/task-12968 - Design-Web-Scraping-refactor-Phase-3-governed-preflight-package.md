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
updated_date: 2026-07-15 00:08
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
User approved the follow-up design review corrections: define separate internal async analyzer implementations and legacy sync/async wrappers; split scrape-level policy from probe egress checks; block service workers, require Playwright 1.48, and document browser DNS pinning limits; specify bounded shielded cancellation cleanup and allowed compatibility/resolver thread bridges; add observability and a Phase 7 sunset for the external-tool legacy default. Reopening the design task while these amendments are written and re-reviewed.
Incorporated and self-reviewed the approved design amendments. The final spec now preserves each analyzer's historical signature and sync/async classification through isolated compatibility wrappers; separates scrape-level policy from per-dispatch ProbeEgressGuard checks; requires fresh guard decisions for HTTP, browser, and explicit external-tool launches; raises Playwright to 1.48 with service workers blocked and runtime capability fallback; documents browser and external-tool DNS/redirect limits; defines monotonic deadline precedence and one shared two-second shielded cleanup grace; and adds a concurrency-safe warning/metric plus Phase 7 sunset for the legacy external-tool default. Verification: unresolved-marker scan clean, internal terminology scan reviewed, and git diff check clean. Documentation-only amendment; Bandit remains not applicable.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and self-reviewed the Phase 3 governed preflight package design. It defines the preflight facade, typed options and context, URL-bound policy flow, governed HTTP/browser/external-tool adapters, analyzer migration and compatibility shims, centralized advice and payload eligibility, failure/cancellation/redaction semantics, deterministic testing, rollout stages, and completion gates. The design preserves current successful analyzer and scrape behavior while making probe governance and ownership explicit.
Amended the committed Phase 3 governed-preflight design after the final issue review. The revised design closes compatibility, policy-boundary, browser-routing, cancellation-cleanup, and external-tool migration ambiguities without expanding Phase 3 into a new analyzer engine. The spec remains ready for the written-spec approval gate before implementation planning.
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
