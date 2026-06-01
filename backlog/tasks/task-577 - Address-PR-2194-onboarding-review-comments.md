---
id: TASK-577
title: Address PR 2194 onboarding review comments
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-01 00:31'
labels:
  - onboarding
  - review-fix
  - webui
  - setup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evaluate and fix unresolved PR review comments for the unified first-run solo onboarding PR, covering backend setup recovery semantics, provider response sanitization, optional advanced path validation, frontend provider resume/save behavior, documentation nits, and focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All unresolved PR #2194 review comments are mapped to fixes or documented as non-actionable.
- [ ] #2 Backend setup/readiness fixes have regression coverage for sanitizer behavior, local provider hosts, setup recovery writes, completion ordering, first-run state doc/type requirements, and audio health logging.
- [ ] #3 Frontend onboarding fixes have regression coverage for provider resume state, multi-provider saves, first-chat completion errors, audio recommendation error surfacing, static test imports, and provider secret redaction.
- [ ] #4 Targeted pytest, Vitest, Bandit, ESLint, and diff hygiene checks are recorded before PR thread closeout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Inspect unresolved PR review comments and classify each as code, test, docs, or no-op.', 'Add focused regression tests for the behavioral issues before implementation.', 'Implement backend review fixes for setup state, validation, completion ordering, docstrings, and audio pack handling.', 'Implement frontend review fixes for provider resume/save behavior, secret redaction, and error handling.', 'Run targeted verification, update task notes, commit/push, and resolve addressed PR threads.']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented review fixes for PR #2194:

- Backend: tightened first-run step-data sanitizer key matching, allowed storage path values only under optional advanced storage keys, accepted common local provider host suffixes, moved InvalidFirstRunTransition to core exceptions, offloaded first-run store calls from async setup endpoints, split setup write guards for recovery, reordered completion persistence, removed dead audio pack import path handling, and switched TTS health logging to logger.exception.
- Frontend: restored provider selection from backend state, saved all selected/configured providers, split first-chat verification/completion errors, surfaced audio recommendation load failures, logged wizard catch blocks, replaced dynamic test imports, and redacted api_key fields from provider save responses.
- Task cleanup: corrected the task-499 end-to-end typo raised by review.

Verification recorded during implementation: backend RED tests failed before fixes; frontend RED tests failed before fixes; after fixes, targeted setup pytest, onboarding Vitest, first-run state pytest, setup suite pytest, Kokoro health tests, Bandit, ESLint, and git diff hygiene checks exited cleanly.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
