---
id: TASK-13500
title: Implement complete Media UX review fixes and enhancements
status: In Progress
labels:
- media
- ux
- frontend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement all five major findings, all nine additional issues and all potential enhancements from TASK-13450 on latest dev 75ab224081bf140ef52017c1a9b0a04f6878d488, preserving WebUI/extension parity. User authorized every item and delegated ordering. Existing audit solutions are approved design direction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reliable URL and file queue handoff, deduplication, eligible counts and overwrite semantics across WebUI and extension.
- [ ] #2 Failed-item retry preserves successful results, saved batches open directly in multi-review, and saved/search readiness is truthful.
- [ ] #3 Preview, selection, keyboard activation and selected-set navigation agree across pages with an explicit 30-item simultaneous reading cap.
- [ ] #4 Inspector selection persists across pages, deletion is recoverable, and reading/bulk actions remain accessible on mobile.
- [ ] #5 Recent imports, active-tab capture, compact batch summaries and contextual batch guidance are implemented using existing state.
- [ ] #6 Meaningful unit/integration and browser checks, build/type/security validation, source review, documentation and tracked incremental commits cover the full accepted scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Audit: .impeccable/critique/2026-10-04T23-52-07Z__i-src-components-review-viewmediapage-tsx-a9a598d1.md. ADR required: no new ADR; local UX repairs reuse existing shared UI, scoped ingestion session/runtime and server APIs. Governing ADR-022 media pipeline ownership, ADR-059 task editor cutover. No new dependencies or backend architecture changes planned.
Approved design: Docs/Design/2026-10-04-media-ux-complete-design.md. Five-stage plan: IMPLEMENTATION_PLAN_media_ux_complete_20261004.md. Sequential fresh implementation and review subagents; baseline 16 focused tests pass. Preflight interface scan and rulings recorded in plan-specific SDD ledger. All original audit findings/enhancements mapped to acceptance checks. Current dev requires backlog-py; prior audit record normalized through canonical CLI.
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
