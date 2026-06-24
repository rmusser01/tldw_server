---
id: TASK-2419
title: Harden Personalization module review findings
status: In Progress
assignee: []
created_date: '2026-06-23 19:00'
updated_date: '2026-06-24 04:45'
labels:
  - personalization
  - security
  - review-fix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and fix validated Personalization companion module review findings. Scope: consistent companion storage user ID resolution, bounded/sanitized activity metadata retention, reflection job type and cadence validation, and actionable logging for activity capture failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated review findings are fixed or explicitly documented as not applicable
- [x] #2 Focused regression tests cover each behavior change
- [x] #3 Touched Personalization/API dependency tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- Verify each review finding against current code and tests before making behavior changes.
- Add failing regression tests for validated behavior bugs.
- Implement the smallest compatible fixes in the Personalization module and API dependency boundary.
- Run focused tests plus Bandit on touched code and record results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Validated and fixed inconsistent text user ID storage resolution across companion activity capture, context loading, API dependency creation, reflection jobs, lifecycle-compatible helpers, and personalization consolidation.
- Validated and fixed raw free-text retention for reading item notes and reading highlight quote/note metadata by storing bounded previews, character counts, digests, and truncation flags.
- Validated and fixed reflection job cadence handling by normalizing supported cadences before slot/dedupe generation and rejecting unknown cadences.
- Validated and fixed reflection worker dispatch by rejecting unknown job types instead of treating them as reflection jobs.
- Validated and improved non-fatal activity capture logging with sanitized exception class reasons; exception messages and paths remain omitted.
- Addressed PR review feedback by adding structured activity-capture log context, sanitized traceback frame summaries, and event/user/dedupe references without logging raw exception messages or paths.
- Addressed PR review feedback by adding legacy text-user storage ID candidates and a read-only existing-DB resolver so older personalization DBs remain discoverable without creating new directories during lookup.
- Reviewed the broad adapter-file P3 concern and lifecycle no-op scopes as maintainability/iteration items rather than correctness or security defects for this fix. No broad split was done in this task.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest --confcutdir=tldw_Server_API/tests/Personalization tldw_Server_API/tests/Personalization/test_companion_activity_adapters.py tldw_Server_API/tests/Personalization/test_companion_activity_db.py tldw_Server_API/tests/Personalization/test_companion_context.py tldw_Server_API/tests/Personalization/test_companion_reflection_jobs.py tldw_Server_API/tests/Personalization/test_companion_user_ids.py -q` - 52 passed.
- `env SINGLE_USER_TEST_API_KEY=test-single-user-key SINGLE_USER_API_KEY=test-single-user-key /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest --confcutdir=tldw_Server_API/tests/Collections tldw_Server_API/tests/Collections/test_companion_reading_activity_bridge.py -q` - 3 passed in the isolated worktree.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest --confcutdir=tldw_Server_API/tests/Personalization tldw_Server_API/tests/Personalization/test_companion_derivations.py tldw_Server_API/tests/API_Deps/test_personalization_deps_sanitization.py tldw_Server_API/tests/Personalization/test_companion_lifecycle.py -q` - 8 passed.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Personalization tldw_Server_API/app/api/v1/API_Deps/personalization_deps.py tldw_Server_API/app/services/personalization_consolidation.py -f json -o /tmp/bandit_personalization_review_pr_comments.json` - 0 findings, 0 errors.
- Running the same focused tests without `--confcutdir` timed out before test execution in the global autouse fixture while importing the full app. The traceback pointed to `tldw_Server_API/tests/conftest.py` importing `tldw_Server_API.app.main`, not to the touched Personalization code.
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Personalization companion module by centralizing storage ID resolution, reducing retained free-text metadata, validating reflection job inputs, and preserving non-leaky but actionable capture logs. Added legacy DB fallback handling for text-user storage IDs, structured sanitized capture-failure logs, regression coverage for each behavior change, and updated the reading activity bridge expectation to the safer preview metadata contract.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run for touched code when applicable or document environment skip
- [x] #4 Final summary added
- [x] #5 Known skips or blockers documented
<!-- DOD:END -->
