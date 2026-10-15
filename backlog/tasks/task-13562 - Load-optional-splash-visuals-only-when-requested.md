---
id: TASK-13562
title: Load optional splash visuals only when requested
status: Done
labels:
- chat
- ux
- frontend
- performance
dependencies:
- TASK-13561
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the eager optional splash visual payload from the shared WebUI shell. Preserve event-driven display, preferences, forced display and dismissal, and keep chat usable through a held or rejected optional chunk. Verify the emitted preview against the existing shared-shell budget and token gates without increasing limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hidden splash does not load visual code; requested display preserves the existing hook and event contract.
- [x] #2 Held or rejected optional visual loading leaves application children usable and records an actionable diagnostic without replacing chat.
- [x] #3 Focused controls, actual client types, authored static checks and emitted current preview build/token/bundle gates pass, with native and whole-program limitations explicit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Search found no duplicate splash-loading task. Highest task13561 confirmed across fresh origin/dev, 1694 open-PR paths and both local task directories; new task13562 allocated. Fresh dev97ea9cd5fa remains an ancestor. Installed Next16.1.4 dynamic/loadable code confirms ssr:false suppresses server preload and its existing loading path handles rejected chunks. Reuse the installed next/dynamic and existing default SplashOverlay leaf, preserving useSplashScreen/state/events/preferences. ADR assessment: no new ADR; no durable protocol, storage, authority, route, framework or dependency decision changes. Existing UI lazy-loading pattern and ADR059 task tracking govern. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_optional_splash_13562.md.
Qualified35/35 across9 actual Next/hook/settings/event and declared visual/provider controls; four causal pre-patch failures/two provider positives. Both final actual client compilers;2 TSX files0 introduced lint; matched WebUI3.8.1 runtime-equivalent formatting; diff and both independent source reviews CLEAR. Current Webpack preview compiles66s; exact dist token gate passes; shared595.1KB below unchanged600KB, heaviest791.9KB below900KB;799 assets present and splash absent from shared startup files. Build ID0YZ4gXAN46r63OcAR-H-D. Bandit N/A:no authored Python. Logs/master updated; completed plan copy /private/tmp/chat-optional-splash-13562-completed-plan.md retained and only owned plan retired. Standalone symlinked dependency tracing is not portable deployment acceptance. Prior native fetch/chunk root and extension AppShell eager import remain separate required follow-ups; no new native/packaged/physical/AT/participant acceptance claimed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Optional WebUI splash visuals now load only when requested, with actual Next failure isolation and preserved preferences/events/dismissal. Current emitted preview passes the unchanged bundle budget at595.1KB and token/asset checks; focused controls, compilers/static and independent source reviews qualify this bounded repair. Broader native runtime and all-ten acceptance continue under TASK13458.
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
