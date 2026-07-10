---
id: TASK-12106
title: Add explicit single-user API key device persistence and relaunch coverage
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-10 22:17'
labels: []
dependencies:
  - TASK-12108
references:
  - TASK-12108
  - TASK-12030
  - TASK-12127
  - 'https://github.com/rmusser01/tldw_server/issues/2590'
documentation:
  - >-
    Docs/superpowers/specs/2026-07-10-single-user-api-key-device-persistence-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add explicit device/session persistence for API keys only when users manually configure a remote single-user server in the WebUI or browser extension, with origin binding and browser/extension relaunch coverage. Same-origin runtime auth is handled without browser-readable keys by TASK-12108.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Manually configured remote single-user servers expose an explicit Remember on this device choice that defaults enabled for new setups.
- [ ] #2 Session-only choice survives hard reload but does not persist the API key across a full browser restart.
- [ ] #3 Remembered choice persists the origin-bound API key across a full browser restart until logout/reset.
- [ ] #4 Remote WebUI regression coverage includes save then hard reload and save then close/reopen the same profile.
- [ ] #5 Extension regression coverage includes save then close/reopen the same extension installation.
- [ ] #6 Same-origin cookie-session/runtime credentials are never copied into browser-readable manual key storage.
- [ ] #7 No browser password-manager behavior is required for API-key persistence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Scope revised after the hybrid architecture was approved. This task covers only manually configured remote WebUI and browser-extension API-key persistence. Same-origin runtime auth moves to TASK-12108 and must never persist or expose the runtime key. Prior spec review required no-inherited-auth candidate probes, ordered origin transitions, strict legacy ownership classification, and device → session → memory fallback. Design: Docs/superpowers/specs/2026-07-10-single-user-api-key-device-persistence-design.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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
