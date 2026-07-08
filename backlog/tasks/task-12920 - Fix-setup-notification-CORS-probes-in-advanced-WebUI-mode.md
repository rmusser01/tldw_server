---
id: TASK-12920
title: Fix setup notification CORS probes in advanced WebUI mode
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-08 14:22'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real-backend UAT on the clean dev-based worktree exposed console CORS errors from NotificationToastBridge probing notification endpoints on /setup before API auth is configured. Gate global notification startup on authenticated app state so setup/login do not issue credentialed cross-origin notification requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Setup and login routes do not start notification bootstrap before auth is configured.
- [x] #2 Hidden-header layouts do not poll notification unread count.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: after rebasing onto dev, setup mounted global notification clients before auth was configured. The toast bridge and header unread-count poll made credentialed cross-origin notification requests, which the backend CORS policy rejected in advanced local mode.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an auth-controlled AppProviders notification gate, passed it from _app only after auth resolves on non-setup routes, and skipped WebLayout unread-count polling while the header is hidden. Verified focused Vitest coverage and real-backend UAT: zero 429s, zero burst targets, no bad responses, and setup notification CORS errors removed.
<!-- SECTION:FINAL_SUMMARY:END -->

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
