---
id: TASK-13285
title: Audio transcriptions health probe returns 500 on every chat load
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - backend
  - chat
  - observability
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A health probe for audio transcriptions returns 500 on every cold load of the chat page. It is the only failing request on the page and is surfaced nowhere, so it is pure console noise that masks real errors while debugging. The dictation control now correctly reports itself unavailable, so the user-facing half of the original finding is fixed; the failing probe is not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The probe returns a success or a well-formed unavailable response.
- [ ] #2 A cold load of the chat page produces no failed requests.
- [ ] #3 When speech is not configured, the probe reports that state rather than failing.
<!-- AC:END -->
