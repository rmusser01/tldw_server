---
id: TASK-288
title: Specify main /chat true cockpit control gaps
status: In Progress
assignee: []
created_date: '2026-05-12 04:21'
updated_date: '2026-05-12 04:30'
labels:
  - webui
  - chat
  - ux
  - frontend
  - spec
dependencies:
  - TASK-272
  - TASK-280
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1582'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design/spec artifact for the main WebUI /chat page that defines the missing true cockpit-control functionality compared with the intended goal. Scope is explicitly the main chat window only, not the browser-extension sidepanel/sidebar. The artifact should separate merge-blocking parity gaps from cockpit maturity work and define the test coverage needed before implementation/merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec states the target is the main /chat page only and excludes sidepanel/sidebar work.
- [ ] #2 Spec distinguishes existing /chat functionality parity from true cockpit-control design gaps.
- [ ] #3 Spec identifies missing or incomplete cockpit controls with evidence from current files/components where possible.
- [ ] #4 Spec defines practical test coverage needed for true cockpit controls and old /chat workflow parity.
- [ ] #5 Spec does not implement code changes or propose unrelated product features.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created design spec at Docs/superpowers/specs/2026-05-12-main-chat-cockpit-controls-gap-design.md in the chat-degraded-health worktree. Scope is main WebUI /chat only, excludes extension sidepanel/sidebar, separates merge-blocking parity from true cockpit-control maturity, and defines real-server/functionality test coverage. Verification so far: git diff --check passed for the worktree; docs-only change, no Bandit needed.

Design review pass found and addressed three spec risks: real-server requirements could be misread as allowing mocked browser data, cockpit scope was too broad for one implementation pass, and open questions lacked default decisions. The spec now adds a shared control contract, a narrow first implementation slice, explicit no page.route/synthetic payload language for merge-critical Playwright coverage, interim degraded-health classification, and default answers for persistence/runtime/submit-test scope.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
