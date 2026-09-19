---
id: TASK-13270
title: Mount the help modal host on the web app so the chat tour works
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - onboarding
  - ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Take a quick tour on the chat empty state is a no-op. Verified live: the DOM is byte-identical before and after the click, and dispatching the open event by hand also does nothing. The host that renders the help modal is mounted only when the layout is running headerless, which is false for a signed-in user on /chat. Three correctly-targeted chat tutorials and a mounted tutorial runner are therefore dead code on the web app. A unit test asserts the button works, against a mock of the store action. The button is also 118x16, below the accessibility target floor.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clicking Take a quick tour on /chat produces a visible result.
- [ ] #2 The help modal host is mounted for signed-in users on /chat.
- [ ] #3 The existing chat tutorials are reachable from the chat surface.
- [ ] #4 A test exercises the rendered outcome, not only that the store action was called.
- [ ] #5 The control meets the minimum target size.
<!-- AC:END -->
