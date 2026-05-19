---
id: TASK-418.4
title: Plan WebUI persona context and agents implementation
status: Done
labels:
- ux
- design
- webui
- extension
- planning
- persona
- agents
priority: High
parent_task_id: TASK-418
documentation:
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-17-webui-persona-context-agents-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documentation-only child implementation plan for the approved WebUI/extension UX remediation program Task 7. Scope maps findings F1 support, F9 support, F15 support, and F18 support into a reviewable plan for persona, character, companion, context asset, and agent route jobs without product code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan file created at `Docs/superpowers/plans/2026-05-17-webui-persona-context-agents-implementation-plan.md`.
- [x] #2 Plan maps findings F1 support, F9 support, F15 support, and F18 support to concrete route-family implementation tasks.
- [x] #3 Plan covers `/persona`, `/characters`, `/companion`, `/agents`, `/agent-tasks`, `/acp-playground`, `/chat-workflows`, `/dictionaries`, and `/world-books`.
- [x] #4 Plan names route/component ownership for Persona Garden, Characters, Companion, Chat Workflows, Dictionaries, World Books, Agent Registry, Agent Tasks, and ACP Playground.
- [x] #5 Plan defines verification commands for focused Vitest, Playwright persona and character journeys, document hygiene checks, and browser-observed route QA.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created a documentation-only child implementation plan for WP7. No product frontend or backend code was modified.
- Plan starts with route-job contract tests, then separates Persona Garden from Companion, clarifies character launch semantics, clarifies context asset activation, aligns Agent and ACP capability states with WP2, and finishes with route-family browser QA.
- Used code evidence from route registry, sidepanel registry, `sidepanel-persona`, `CharactersWorkspace`, `useCharacterQuickChat`, `CompanionHomeShell`, `ChatWorkflowsPage`, dictionary navigation, `WorldBookDetailPanel`, Agent Registry, Agent Tasks, and ACP Playground surfaces.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the WP7 persona/context/assets/agents child implementation plan and recorded the route, component, test, and browser QA scope for future implementation. This task is documentation-only and did not modify product code.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit skipped because this task changes only Markdown and Backlog task metadata
- [x] #5 Final summary added
- [x] #6 Known skip documented: no product tests were run because no product code changed
<!-- DOD:END -->
