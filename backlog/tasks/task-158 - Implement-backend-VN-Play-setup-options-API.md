---
id: TASK-158
title: Implement backend VN Play setup-options API
status: Done
assignee: []
created_date: '2026-05-09 05:38'
updated_date: '2026-05-09 05:53'
labels:
  - vn-play
  - api
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1407'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/pull/1409'
documentation:
  - Docs/superpowers/specs/2026-05-09-vn-play-setup-options-design.md
  - >-
    Docs/superpowers/plans/2026-05-09-vn-play-setup-options-backend-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend-first VN Play setup-options API described in the design spec so the API server can stand alone and custom frontends can create VN Play sessions without duplicating character, pack readiness, compatibility, trust, and warning logic. PR #1409 is frontend-only and should be treated as a UX/reference path rather than the architecture to copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend exposes a bounded GET /api/v1/vn-play/setup-options endpoint with selector-safe character data, selected_character preservation, asset pack readiness/compatibility/warning summaries, pagination metadata, defaults, scoped empty states, and no image bytes.
- [x] #2 Asset pack listing for setup options applies query, sort, limit, and offset at the repository/service boundary before readiness fanout; readiness is evaluated only for returned pack rows.
- [x] #3 Trust provenance and pack_untrusted_import warnings are derived from completed import journal metadata when available, and degrade safely when provenance cannot be classified.
- [x] #4 Focused backend tests cover endpoint contract, bounded readiness behavior, selected_character pagination behavior, warning severity/acknowledgement metadata, scoped empty states, and safe fallback when per-pack readiness fails.
- [x] #5 A follow-up frontend integration path is documented so PR #1409 or a successor can consume the backend contract instead of client-side duplicating setup rules.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Direction confirmed: prefer a backend setup-options API over PR #1409's frontend-only selector logic so custom frontends and standalone API deployments can reuse server-authoritative setup rules. Created backend implementation plan Docs/superpowers/plans/2026-05-09-vn-play-setup-options-backend-implementation-plan.md.

Implemented backend-first setup-options endpoint, bounded pack listing/provenance helpers, response schemas, composer, API docs, and focused backend coverage. Verification: pytest VN_Play+VN_Assets 24 passed; Bandit production scope exit 0; Bandit touched tests with B101 skipped exit 0; git diff --check exit 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added server-authoritative VN Play setup options so custom frontends can query selector-safe characters, selected-character preservation, bounded asset pack readiness, compatibility, trust provenance, warning summaries, defaults, pagination, and empty-state hints from the API. Pack setup listing now filters and paginates at the repository boundary before readiness fanout, import provenance is derived from latest completed journals, and lookup/readiness failures degrade safely. Documented the endpoint as the integration target for PR #1409 or a successor frontend slice.
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
