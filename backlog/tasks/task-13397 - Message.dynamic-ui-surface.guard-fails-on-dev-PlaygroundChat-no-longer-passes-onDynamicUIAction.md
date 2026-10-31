---
id: TASK-13397
title: >-
  Message.dynamic-ui-surface.guard fails on dev: PlaygroundChat no longer passes
  onDynamicUIAction
status: To Do
assignee: []
created_date: '2026-09-30 01:47'
labels:
  - bug
  - chat
  - webui
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
apps/packages/ui/src/components/Common/Playground/__tests__/Message.dynamic-ui-surface.guard.test.ts > 'keeps compare cluster messages in the main /chat dynamic UI surface' fails on origin/dev (seen 2026-09-29 while verifying TASK-13389): PlaygroundChat.tsx no longer contains 'onDynamicUIAction={onDynamicUIAction}'. Not caught by the frontend ratchet (head-vs-base only). Last PlaygroundChat.tsx change: da0f1cd3a3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified (stale source guard vs dropped dynamic UI action wiring) and the case passes
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
