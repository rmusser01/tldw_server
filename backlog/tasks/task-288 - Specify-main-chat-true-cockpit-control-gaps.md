---
id: TASK-288
title: Specify main /chat true cockpit control gaps
status: Done
assignee: []
created_date: '2026-05-12 04:21'
updated_date: '2026-05-23 12:47'
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
documentation:
  - Docs/superpowers/specs/2026-05-12-main-chat-cockpit-controls-gap-design.md
  - Docs/superpowers/specs/2026-05-13-main-chat-cockpit-rail-completion-design.md
  - Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
  - Docs/superpowers/specs/2026-05-16-main-chat-cockpit-merge-certification.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design/spec artifact for the main WebUI /chat page that defines the missing true cockpit-control functionality compared with the intended goal. Scope is explicitly the main chat window only, not the browser-extension sidepanel/sidebar. The artifact should separate merge-blocking parity gaps from cockpit maturity work and define the test coverage needed before implementation/merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec states the target is the main /chat page only and excludes sidepanel/sidebar work.
- [x] #2 Spec distinguishes existing /chat functionality parity from true cockpit-control design gaps.
- [x] #3 Spec identifies missing or incomplete cockpit controls with evidence from current files/components where possible.
- [x] #4 Spec defines practical test coverage needed for true cockpit controls and old /chat workflow parity.
- [x] #5 Spec does not implement code changes or propose unrelated product features.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created design spec at Docs/superpowers/specs/2026-05-12-main-chat-cockpit-controls-gap-design.md in the chat-degraded-health worktree. Scope is main WebUI /chat only, excludes extension sidepanel/sidebar, separates merge-blocking parity from true cockpit-control maturity, and defines real-server/functionality test coverage. Verification so far: git diff --check passed for the worktree; docs-only change, no Bandit needed.

Design review pass found and addressed three spec risks: real-server requirements could be misread as allowing mocked browser data, cockpit scope was too broad for one implementation pass, and open questions lacked default decisions. The spec now adds a shared control contract, a narrow first implementation slice, explicit no page.route/synthetic payload language for merge-critical Playwright coverage, interim degraded-health classification, and default answers for persistence/runtime/submit-test scope.

Clarified the spec is only the first implementation slice toward the longer-term main /chat cockpit target. Added a First Slice Boundary section with explicit in-scope and out-of-scope lists, renamed broader items to Cockpit Maturity Backlog, and added a boundary warning not to implement later cockpit-maturity items without explicit user approval.

Closeout review 2026-05-23: verified the design artifact at `Docs/superpowers/specs/2026-05-12-main-chat-cockpit-controls-gap-design.md` already satisfies the task acceptance criteria: it scopes to `apps/tldw-frontend/pages/chat/index.tsx` -> `apps/packages/ui/src/routes/option-chat.tsx` -> `Playground`, excludes browser-extension sidepanel/sidebar work, separates merge-blocking parity from true cockpit-control completion, names file/component evidence, and defines component plus real-server Playwright coverage. PR #1582 is merged into `dev` at `ef1390857fee0e322f26756f7f1da48115373272`; downstream TASK-290, TASK-291, TASK-295, TASK-319, TASK-390, and `Docs/superpowers/specs/2026-05-16-main-chat-cockpit-merge-certification.md` record the implementation, certification, and post-merge roadmap that followed from this spec. Bandit skipped for this closeout because only Markdown spec/task files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the main `/chat` cockpit-control gap specification task. The original spec is scoped to the main WebUI `/chat` page, excludes sidepanel/sidebar work, separates parity gaps from cockpit maturity, identifies missing controls with concrete file/component evidence, and defines practical component plus real-server verification coverage. No application code was changed in this closeout; the design has already fed the merged PR #1582 implementation and later maturity roadmap.
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
