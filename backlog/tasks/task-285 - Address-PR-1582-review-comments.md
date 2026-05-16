---
id: TASK-285
title: Address PR 1582 review comments
status: Done
assignee: []
created_date: '2026-05-12 03:35'
updated_date: '2026-05-12 03:56'
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
- [x] #4 PR branch is committed and pushed with review fixes and the Backlog task records final evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR marked draft. Fixed reviewed items: cockpit i18n strings and status/session labels, model popover callback dependencies, provider-qualified model selection, provider configured usability fields, scoped setting canonicalization, direct backend metadata assignments, and line-length cleanup. Verification so far: focused Vitest cockpit/model tests pass; backend llm model filter tests pass; Bandit on llm_providers.py reports zero findings; git diff --check clean. Full UI package tsc still fails on existing unrelated baseline test/type errors outside the touched scope.

Follow-up provider-qualified routing fix added after rechecking Qodo/CodeRabbit ambiguous-provider comments: provider:model selections now parse into a request model id plus provider override at chat action and chat pipeline boundaries, with resolve-api-provider regression coverage.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #1582 was marked draft and pushed with review fixes in commit 52977e6e5. Addressed cockpit i18n/session-label findings, runtime/status pluralization through i18n count paths, model popover dependencies, provider-qualified model selection and settings scope canonicalization, provider configured metadata handling, and the Python test line-length issue. Verification: focused Vitest cockpit/model/store tests passed; backend llm model filter tests passed; Bandit on llm_providers.py reported zero findings; git diff --check passed. Full UI package tsc was attempted and still fails on pre-existing unrelated baseline test/type errors outside this task scope.

Additional pushed fix: provider-qualified selections are now stripped to request model IDs while preserving provider routing, with resolve-api-provider and chat pipeline focused tests covering the path.
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
