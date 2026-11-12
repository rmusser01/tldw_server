---
id: TASK-13268
title: Make Show technical details carry real diagnostics
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - error-handling
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The progressive-disclosure affordance on chat errors expands to three words, "Stream completion failed", growing the page by 21 characters. It carries no status code, endpoint, provider name or correlation id. The backend knew all of these and logged them. The affordance sets an expectation of diagnosability it does not meet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Expanding technical details reveals the HTTP status and the provider or endpoint involved.
- [ ] #2 A correlation or request identifier is shown when the backend supplies one.
- [ ] #3 The detail text is selectable and copyable in one action.
- [ ] #4 No secret or credential value appears in the disclosed detail.
<!-- AC:END -->
