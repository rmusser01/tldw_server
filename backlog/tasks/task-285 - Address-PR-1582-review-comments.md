---
id: TASK-285
title: Address PR 1582 review comments
status: In Progress
assignee: []
created_date: '2026-05-12 03:35'
updated_date: '2026-05-12 03:48'
labels:
  - webui
  - chat
  - frontend
  - pr-review
dependencies:
  - TASK-280
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1582'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review feedback on PR #1582 for the /chat cockpit focus-mode branch. Scope is the existing PR branch codex/chat-degraded-health and should stay limited to comments/check failures directly tied to this PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #1582 is converted to draft while review fixes are in progress.
- [x] #2 All current actionable CodeRabbit, Qodo, Gemini, and GitHub inline review findings are inspected against current code and either fixed or documented as no longer applicable.
- [x] #3 Focused verification covers the changed frontend/backend files and any review-specific regression cases.
- [ ] #4 PR branch is committed and pushed with review fixes and the Backlog task records final evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR marked draft. Fixed reviewed items: cockpit i18n strings and status/session labels, model popover callback dependencies, provider-qualified model selection, provider configured usability fields, scoped setting canonicalization, direct backend metadata assignments, and line-length cleanup. Verification so far: focused Vitest cockpit/model tests pass; backend llm model filter tests pass; Bandit on llm_providers.py reports zero findings; git diff --check clean. Full UI package tsc still fails on existing unrelated baseline test/type errors outside the touched scope.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
