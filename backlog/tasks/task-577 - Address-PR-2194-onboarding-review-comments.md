---
id: TASK-577
title: Address PR 2194 onboarding review comments
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-01 00:52'
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
- [x] #1 All unresolved PR #2194 review comments are mapped to fixes or documented as non-actionable.
- [x] #2 Backend setup/readiness fixes have regression coverage for sanitizer behavior, local provider hosts, setup recovery writes, completion ordering, first-run state doc/type requirements, and audio health logging.
- [x] #3 Frontend onboarding fixes have regression coverage for provider resume state, multi-provider saves, first-chat completion errors, audio recommendation error surfacing, static test imports, and provider secret redaction.
- [x] #4 Targeted pytest, Vitest, Bandit, ESLint, and diff hygiene checks are recorded before PR thread closeout.
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

Follow-up: Cubic added five new review threads after commit 2ffcc1c50e. Reopened this task to address the new Layout shell override cleanup, quick ingest preset validation, option setup localization/shared gating, and post-onboarding readiness race comments.

Follow-up implementation: added owner-safe shell override cleanup in OptionLayout, strict own-property preset validation for first-source quick ingest metadata, shared setup status gating helper, /setup translation keys and English locale entries, and request sequencing for post-onboarding media readiness checks.

Follow-up RED/GREEN evidence: the new focused Vitest tests first failed on the inherited preset key and stale readiness overwrite, setup-status helper was missing, and the shell override cleanup test failed because cleanup called another owner setOverrides(null). After implementation, the focused follow-up/onboarding Vitest suite passed with 43 tests. Targeted ESLint exited 0 with the existing Next pages-directory warning, and git diff --check exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the original PR #2194 review threads in 2b1c568ecf and the post-push Cubic follow-up threads in b658b3b8d2. Backend fixes covered setup sanitizer false positives, optional advanced storage path handling, local provider hostname validation, centralized first-run transition errors, async first-run store offloading, setup recovery write guards, completion persistence order, audio pack import cleanup, and TTS health exception logging. Frontend fixes covered provider resume state, multi-provider saves, first-chat completion failure messaging, audio recommendation error feedback, wizard error logging, static imports in tests, api_key redaction, owner-safe shell override cleanup, strict quick-ingest preset validation, shared setup status gating, /setup localization keys, and stale readiness request protection.

Verification: python -m pytest tldw_Server_API/tests/Setup tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py tldw_Server_API/tests/TTS_NEW/integration/test_kokoro_runtime_health_envelope.py -q passed with 347 passed, 4 warnings; bunx vitest run targeted onboarding/setup tests passed with 19 passed; follow-up focused Vitest run passed with 43 passed; Bandit on touched Python implementation files exited 0 and wrote /tmp/bandit_pr2194_review_fixes.json; targeted ESLint exited 0 with the existing Next pages-directory warning; git diff --check exited 0.

PR thread closeout: replied to and resolved the original 24 review threads, then replied to and resolved the 5 post-push Cubic code threads. A final task-record cleanup removed the absolute local path from this summary. No known blockers remain.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
