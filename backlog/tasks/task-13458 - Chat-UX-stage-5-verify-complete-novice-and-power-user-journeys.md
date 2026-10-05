---
id: TASK-13458
title: 'Chat UX stage 5: verify complete novice and power-user journeys'
status: To Do
labels:
- chat
- ux
- webui
- extension
- nng
dependencies:
- TASK-13454
- TASK-13455
- TASK-13456
- TASK-13457
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate all ten findings and enhancements end to end, then apply bounded polish from evidence. Run a complete surface/state matrix with the real backend lifecycle and a deterministic provider, preserve owner/privacy gates, and update the audit with verified changes and remaining limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Start/name/leave/find/continue and choose/import card/greeting/two turns/reopen/change scene pass on WebUI and full extension, with compact sidepanel coverage and both handoffs where supported.
- [ ] #2 Provider error, midstream stop/interruption, offline/reconnect, save/load failure, conflicting histories, account/workspace change, custom context capability gates, and draft recovery have recorded outcomes without unacknowledged-save claims or silent replay.
- [ ] #3 More than five interleaved character sessions and a large seeded history, 320/390/desktop widths, 200 percent zoom, both themes, keyboard, and screen-reader checks have reproducible evidence.
- [ ] #4 Existing meaningful unit/integration/e2e, typecheck, lint, builds, and touched-scope security checks pass; test known-defect wrappers are converted only after independent confirmation the behavior is fixed.
- [ ] #5 Before/after screenshots and NN/G review cover all ten issues and approved enhancements, preserving Pro capability and temporary-chat privacy; all failures and unrun checks are explicit.
- [ ] #6 A concise novice/power-user study script defines uncoached tasks, completion, wrong-chat selections, assistance requests, post-task ease, and proposed two-minute first message / thirty-second reopen targets; participant results are reported only if actually collected.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Program: TASK-13453. Design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Base: origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488, verified 2026-10-04. User approved equal audience priority and all ten findings plus enhancements in five stages. Written-design approval and the requested critical review are recorded on 2026-10-04; application implementation has not begun. ADR required: no for the client remediation; ADR-049 and ADR-050 govern owner-bound history and native fork boundaries, ADR-059 governs canonical task editing. Reassess if implementation changes public APIs or persistence authority. Reuse existing shared UI, loaders, presets, and test harnesses; preserve temporary-chat privacy and account/workspace fences. All findings and named enhancements. Run the real server persistence/ownership lifecycle with a deterministic mock provider; do not treat mocked browser success as server proof. Reuse existing harnesses from #3139/#3137/#3136 when available. No new telemetry, study recruitment, or test framework.
Pre-implementation review adds integrated checks for lost creation/append acknowledgements and reload, greeting preview versus committed stored rows/context, settings local success/server failure/unknown and unresolved-only retries, outage-safe cached mirrors/local DB unavailable, and each transfer mechanism including destination conflicts and partial import. Source-derived scenarios are risks to test, not newly observed incidents. Report native-shell and participant checks only when actually performed.
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
