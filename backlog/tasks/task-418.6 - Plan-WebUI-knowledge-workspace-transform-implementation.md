---
id: TASK-418.6
title: Plan WebUI knowledge workspace transform implementation
status: Done
labels:
- ux
- design
- webui
- extension
- planning
- knowledge
- workspace
priority: High
parent_task_id: TASK-418
documentation:
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-17-webui-knowledge-workspace-transform-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documentation-only child implementation plan for the approved WebUI/extension UX remediation program Task 9. Scope maps findings F14, F1 support, F2 support, and F15 support into a reviewable plan for knowledge, research, workspace, and transform route jobs without product code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Child plan saved at `Docs/superpowers/plans/2026-05-17-webui-knowledge-workspace-transform-implementation-plan.md`.
- [x] Plan covers `/knowledge`, `/search`, `/research`, `/workspace-playground`, `/chat-workspace`, `/document-workspace`, `/repo2txt`, `/model-playground`, `/writing-playground`, and `/presentation-studio`.
- [x] Plan maps findings `F14`, `F1 support`, `F2 support`, and `F15 support` to route-level implementation tasks.
- [x] Plan preserves `/knowledge` as direct cited Q&A and avoids converting it into a generic knowledge-management hub.
- [x] Plan keeps scope documentation-only for this task and does not modify product frontend or backend code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created a route-specific implementation plan for the Ask, Research, Workspace, and Transform product ladder.
- Included an explicit route-job contract so implementation can test canonical labels, aliases, primary jobs, output kinds, and boundaries before changing UI copy.
- Split implementation into small reviewable tasks: route contract, Knowledge Ask and Search alias, Research run console, Workspace Playground, Chat and Document Workspace states, Transform tools, and browser QA.
- Used code evidence from current route wrappers, Next pages, workspace components, research run page, Repo2Txt, Model Playground, Writing Playground, and Presentation Studio ownership.
- Verification run:
  - `rg -n "T[O]D[O]|T[B]D|F[I]XME|\\.\\.\\.|\\bm[a]ybe\\b|\\bpr[o]bably\\b|\\bshould c[o]nsider\\b" Docs/superpowers/plans/2026-05-17-webui-knowledge-workspace-transform-implementation-plan.md` exited 1 with no matches.
  - `rg -n "[[:blank:]]$|[^\\x00-\\x7F]" Docs/superpowers/plans/2026-05-17-webui-knowledge-workspace-transform-implementation-plan.md` exited 1 with no matches.
  - `git diff --check -- Docs/superpowers/plans/2026-05-17-webui-knowledge-workspace-transform-implementation-plan.md` exited 0.
  - Node route and file coverage check exited 0 with `coverage ok`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the documentation-only WP9 child implementation plan for the WebUI/extension UX remediation program. The plan defines route inventory, non-goals, scoped files, route-job metadata, TDD tasks, Playwright browser QA, and verification commands for improving the Ask, Research, Workspace, and Transform route families without changing product code in this task.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip: skipped because this task changed Markdown planning and Backlog documentation only.
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented: no product implementation or browser QA was run because this task only writes the implementation plan.
<!-- DOD:END -->
