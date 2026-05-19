---
id: TASK-390
title: Plan post-merge /chat cockpit UX maturity roadmap
status: Done
assignee: []
created_date: '2026-05-15 21:34'
updated_date: '2026-05-15 21:38'
labels:
  - webui
  - chat
  - ux
  - frontend
  - plan
dependencies:
  - TASK-288
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
  - Docs/superpowers/plans/2026-05-15-chat-cockpit-composition-preview-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a post-merge staged roadmap for bringing the main WebUI /chat cockpit from recertified parity to a fully mature cockpit experience. Scope is the main /chat page only, with PR 1 focused on Context Stack plus Prompt/Persona/Model Composition Preview.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Roadmap is saved under Docs/superpowers/specs with a unique dated filename and explicitly scopes to the main /chat page only.
- [x] #2 Roadmap covers staged PRs for the identified UX maturity areas and identifies PR 1 as Context Stack plus Prompt/Persona/Model Composition Preview.
- [x] #3 PR 1 implementation plan is saved under Docs/superpowers/plans with concrete files, tests, verification commands, and acceptance criteria.
- [x] #4 Plan distinguishes quick wins from larger redesign work and avoids sidepanel/sidebar scope.
- [x] #5 Planning task does not implement application code changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the post-merge main /chat cockpit maturity roadmap and the PR 1 Context Stack + Prompt/Persona/Model Composition Preview implementation plan. Scope remains main /chat only, with sidepanel/sidebar explicitly excluded.

Verification: git diff --check passed. Bandit skipped because this task only adds Markdown planning/task files and no Python/application code. No blockers.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the post-merge main /chat cockpit UX maturity roadmap and the detailed PR 1 implementation plan for Context Stack + Prompt/Persona/Model Composition Preview. The roadmap is scoped strictly to the main /chat page, starts with the requested composition-preview slice, separates quick wins from larger redesign opportunities, and records concrete files/tests/real-server verification expectations. No application code was changed.
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
