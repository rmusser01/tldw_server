---
id: TASK-12002
title: Harden Slides module review findings
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the current Slides module review findings under `tldw_Server_API/app/core/Slides`, including script-safe export settings, atomic sync logging, bounded generation chunking, default asset caps, render duration/count caps, markdown sanitization fallback, malformed FTS handling, and schema initialization locking.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exported Reveal HTML does not embed raw script-closing settings JSON.
- [x] #2 Presentation create/update mutations and sync logs are atomic.
- [x] #3 Generation chunking, slide asset resolution, and video rendering enforce resource ceilings by default.
- [x] #4 Markdown sanitization continues to use bleach when CSS sanitizer support is unavailable.
- [x] #5 Malformed presentation search queries return controlled validation errors.
- [x] #6 Slides schema initialization is safe under concurrent first use.
- [x] #7 Focused Slides tests and Bandit pass for touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: IMPLEMENTATION_PLAN_slides_review_fixes.md.

RED verification: focused regression run failed before implementation because the new render limit constants did not exist, confirming the tests were exercising unfixed behavior.

Implemented script-safe inline settings JSON for Reveal exports, independent bleach markdown sanitization when CSS sanitizer support is unavailable, default slide asset byte caps with post-read verification, explicit render slide/duration/total ceilings, generation source/chunk fan-out ceilings, transactional sync-log writes, malformed FTS query validation, schema initialization locking, API search error mapping, API asset cap propagation, and Reveal settings enum validation.

GREEN verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Slides -q` passed with 171 passed and 369 warnings.

Security verification: Bandit on touched production files wrote /tmp/bandit_slides_review_fixes.json and reported 0 results, 0 errors, and zero high/medium/low severity findings. `git diff --check` on touched files exited 0.
PR follow-up 2026-06-24: rebased branch onto origin/dev at 2618c4b70. Qodo review reported four actionable issues: broad/silent optional import catches, silent generator setting fallback, negative chunk size bypass, and overbroad FTS OperationalError mapping. Reopening task for review-comment remediation.
PR follow-up completed 2026-06-24: addressed Qodo review comments after rebase onto origin/dev 2618c4b70. Fixed optional bleach/CSSSanitizer imports to only catch ImportError, made invalid generator integer settings visible via warnings while propagating unexpected settings backend failures, rejected non-positive chunk_size_tokens before LLM calls, and narrowed FTS OperationalError mapping so non-query database errors propagate. Added regression tests for each behavior and stabilized the Slides ordering property test health check.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Slides hardening PR rebased onto latest dev and review comments remediated. Verification: python -m pytest tldw_Server_API/tests/Slides -q -> 176 passed; python -m bandit -r touched Slides production files -f json -> 0 findings; git diff --check -> clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run for touched code when applicable or documented skip
- [x] #4 Final summary added
<!-- DOD:END -->
