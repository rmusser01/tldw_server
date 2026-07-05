---
id: TASK-12025
title: Design Web_Scraping modular refactor preserving preflight analyzers
status: Done
created_date: 2026-07-03 23:59
labels:
- web-scraping
- design
- refactor
priority: High
references:
- /Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Web_Scraping
- /Users/appledev/Documents/GitHub/tldw_server/Docs/Design/WebScraping.md
- /Users/appledev/Documents/GitHub/tldw_server/Docs/superpowers/specs/2026-06-26-web-scraping-hardening-design.md
modified_files:
- Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md
updated_date: 2026-07-04 00:04
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a design spec for the larger Web_Scraping modular refactor. The design must improve maintainability, extensibility, and stability while preserving current functionality, especially governed pre-scrape analyzer behavior and existing compatibility entry points.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec is written under Docs/superpowers/specs with the approved compatibility-facade architecture.
- [x] #2 Spec covers package boundaries, data flow, error handling, compatibility, migration phases, and test strategy.
- [x] #3 Spec explicitly preserves governed preflight analyzer functionality and existing public dict-shaped contracts during migration.
- [x] #4 Spec self-review records placeholder, consistency, scope, and ambiguity checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Draft the modular refactor design spec from the approved brainstorming sections. 2. Self-review the spec for placeholders, contradictions, over-broad scope, and ambiguous requirements. 3. Record verification and ask the user to review before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-03: Created after user selected the full module architecture approach and approved the revised compatibility-facade design direction.
2026-07-03: Wrote the design spec at Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md. Self-review checks completed: non-ASCII scan passed after cleanup, scoped whitespace check passed, and placeholder scan only matched the self-review statement itself. Bandit is not applicable because this task changed documentation and Backlog metadata only.
2026-07-03: Reopened for a follow-up design review before implementation planning. The review found the spec should explicitly cover import inventory, broader compatibility APIs, resource-governance/rate-limit/dedup responsibilities, source import helpers, and guardrails against new code depending on legacy wrappers.
2026-07-03: Applied follow-up review amendments to the spec. Added routing/content/sources package homes, runtime rate-limit and resource-governance ownership, Phase 0 import inventory and dependency guardrails, broader compatibility entry points, and open decisions for duplicate WebSearch path/source helper placement. Verification after amendment: non-ASCII scan found no matches, placeholder/TODO scan found no matches, and scoped git diff whitespace check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and amended the Web_Scraping modular refactor design spec. The final design uses a compatibility-facade migration, preserves governed preflight analyzer behavior, keeps existing public dict-shaped contracts during migration, adds an import-inventory guardrail phase, and explicitly covers runtime, routing, content, source import, resource-governance, extraction, crawl/jobs, search provider, and wrapper cleanup responsibilities.
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
