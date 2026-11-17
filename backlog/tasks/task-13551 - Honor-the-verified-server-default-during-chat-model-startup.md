---
id: TASK-13551
title: Honor the verified server default during chat model startup
status: Done
labels:
- frontend
- chat
- ux
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair CC-01 / #3111 at the shared catalog and startup selection root. Retain the existing model/favorite hydration fences and explicit selections, preserve provider identity when model IDs collide, and qualify behavior through the actual Playground send request. This continues the accepted all-ten chat UX remediation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A first run with no selected model or available favorite uses the selectable default model of the verified server default provider, regardless of catalog order.
- [x] #2 Explicit selections and available favorites retain precedence; hydration, unavailable defaults, absent defaults, provider aliases, and duplicate model IDs are covered without weakening catalog owner/invalidation fences.
- [x] #3 WebUI and extension type gates, affected model/catalog/send tests, authored lint/format checks, and two independent source reviews pass with evidence recorded.
- [x] #4 The accepted remediation design records this bounded repair and remaining program work honestly.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Existing ADR-049 owner-bound selection and ADR-059 canonical task tracking govern; this carries existing provider default metadata through the current shared catalog, changes no public API or persistence authority, and introduces no dependency. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_model_startup_13551.md. Dev freshly fetched at 2c5f19d0328360d45bce99b8bae75ce252cc312f and included in HEAD; highest task ID checked across origin/dev, all 53 open PRs and local backlog before creation.
Final positive matrix:369/369 across20 suites, no skips or expected-failure acceptance. Both client type gates pass; effective authored lint23 files zero errors/zero added diagnostics (672 inherited warnings), nine whole owned files plus nine owned formatting regions and diff checks pass. Two independent final source reviews clear the shared path. Evidence: /private/tmp/chat-model-startup-13551-{final-matrix,web-types-final,extension-types-final,static-final,format}.log and -lint.json. Same-tab reload preserves the chosen model and exact two server rows; -native.jpg/-native-messages.json. Native first-run provider matrix, real generation quality, packaged MV3/docking, physical/zoom/VoiceOver and participants are not claimed. Bandit N/A: TS/TSX/Markdown/task records only. All three bounded plan stages complete and this completed owned plan is retired; accepted remediation design records the root repairs and remaining all-ten work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Retain verified server defaults through the existing catalog and schema5 cache; preserve explicit/favorite preferences and canonical provider identity. Repair all four startup callers, case-sensitive IDs, bounded deployed provider aliases and both Research hydration boundaries.369 positive checks, both client types, authored static gates and two independent source reviews pass. Same-tab reload preserves selected model and exact saved rows. This bounded unit is Done; all-ten remediation and remaining native/external acceptance continue.
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
