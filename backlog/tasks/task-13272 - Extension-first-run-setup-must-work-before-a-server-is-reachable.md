---
id: TASK-13272
title: Extension first-run setup must work before a server is reachable
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - extension
  - onboarding
  - ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first thing a new extension user sees in Options is a red panel reading "Setup progress could not be loaded. The server may still be starting, or the connection details may be missing - the wizard works once the app can reach your tldw server." The wizard whose purpose is to establish that connection reports that it needs the connection to work. A setup path is still offered below, so it is not blocking, but the failure is the visual dominant of the first screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening first-run setup with no server configured shows no error state.
- [ ] #2 The wizard renders its setup paths as the primary content on first open.
- [ ] #3 Connection problems are reported only after the user supplies connection details.
- [ ] #4 Any retry affordance appears only once there is something to retry.
<!-- AC:END -->
