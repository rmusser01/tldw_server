---
id: TASK-13525
title: 'WebUI perf batch W5: extension background and content scripts'
status: In Progress
created_date: 2026-10-07 04:14
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WebUI/extension perf program. Index: Docs/Plans/2026-10-06-webui-perf-remediation-coordination-index.md. Plan: Docs/Plans/2026-10-06-webui-perf-batch-5-extension-scripts-implementation-plan.md. Scope: Web-clipper tiered capture, copilot popup streaming appends, chunked base64, sidepanel-open guard, warm alarm hygiene. WP-56..63.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 1 (web-clipper tiered capture, WP-56/WP-57): overlap check found TASK-13511 (in dev) had made the parser lazily loaded, but capture still eagerly serialized outerHTML + ran the parser + forced innerText on every clip; requestedType never reached the extraction. Stage 1 threads requestedType into extractClipPageTextFromDocument: selection clips with a live selection return the selection immediately (no outerHTML serialize, no parser, no innerText forced layout — spy-pinned mechanism test), article tier next, fullPageText only as the final fallback for selection/article requests; other clip types keep the eager general flow (characterization trio pins selection-present / selection-absent / article-empty tiers; no captured-content regression). parser/default.ts per-attribute removeAttr loop replaced by one in-place attribs pass preserving the exact kept-set {href, src} (pinned via link title / image alt behavior). Tests: services/web-clipper/__tests__/content-extract.test.ts (6) + parser/__tests__/default.test.ts (1); scoped suites green (30 tests / 6 files incl. existing), apps/tldw-frontend bun run typecheck exit 0. ADR assessment: none required — scoped perf change under the approved plan (Docs/Plans/2026-10-06-webui-perf-batch-5-extension-scripts-implementation-plan.md); no new durable architecture rule.
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
