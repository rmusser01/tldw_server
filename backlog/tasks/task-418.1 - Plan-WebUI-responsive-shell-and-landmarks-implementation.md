---
id: TASK-418.1
title: Plan WebUI responsive shell and landmarks implementation
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 18:29'
labels:
  - ux
  - design
  - webui
  - extension
  - planning
  - responsive
  - accessibility
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
  - >-
    Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
  - >-
    Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md
parent_task_id: TASK-418
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documentation-only child implementation plan for the approved WebUI/extension UX remediation program Task 4. Scope maps findings F2, F10 support, F11 support, F13, and F15 into a reviewable responsive shell and route landmark implementation plan without product code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Child implementation plan exists for the responsive app shell and page landmark slice.
- [x] #2 Plan stays documentation-only and does not modify product frontend or backend code.
- [x] #3 Plan maps the slice to F2, F10 support, F11 support, F13, and F15 from the approved UX remediation program.
- [x] #4 Plan enumerates all WP4 routes and defines expected 390px behavior plus semantic heading outcomes.
- [x] #5 Plan names shared shell, settings, chat, media, prompts, workspace, sources, MCP Hub, STT, TTS, chat-workspace files and tests.
- [x] #6 Planning verification commands and results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Created Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md.
- Reused the approved parent plan and remediation spec as the source of scope.
- Route rows covered by the plan are /chat, /media, /settings, /settings/model, /prompts, /workspace-playground, /setup, /sources, /mcp-hub, /stt, /tts, and /chat-workspace.
- The plan starts with Playwright heading and horizontal-overflow gates, then sequences shared shell constraints before settings, chat, media, and route-local fixes.
- Verification run for this planning artifact: placeholder-language scan exited 1 with no output; ASCII/trailing-whitespace scan exited 1 with no output; git diff check exited 0.
- Bandit was not run because this task changed only Markdown planning and Backlog task files.

- Node coverage check confirmed required route, finding, file, and test tokens are present.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the documentation-only child implementation plan for WP4 responsive app shell and page landmarks. The plan defines route behavior, file ownership, test ownership, browser QA gates, and staged implementation order without changing product code.
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
