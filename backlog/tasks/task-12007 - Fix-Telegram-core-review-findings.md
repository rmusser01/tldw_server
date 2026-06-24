---
id: TASK-12007
title: Fix Telegram core review findings
status: Done
assignee: []
created_date: '2026-06-24 00:00'
updated_date: '2026-06-24 05:10'
labels:
  - telegram
  - security
  - authnz
dependencies: []
references:
  - IMPLEMENTATION_PLAN_telegram_core_review_fixes.md
  - https://github.com/rmusser01/tldw_server/pull/2498
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address validated review findings in Telegram core/runtime code: scope-bind pairing-code consumption, avoid plaintext pairing-code storage for new codes, and canonicalize Telegram session mapper inputs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pairing codes can only be consumed in the scope where they were created.
- [x] #2 New pairing codes are stored hashed rather than as plaintext bearer credentials.
- [x] #3 Telegram session IDs are derived from canonical, scalar-safe inputs and reject unsafe values.
- [x] #4 Regression tests cover scope mismatch, hash-only storage, and canonical input validation.
- [x] #5 Targeted tests and Bandit pass for touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Backlog MCP tools were not exposed in this session. The Backlog CLI was available, but `search`, `list`, `view`, and `task create` hung repeatedly, including with browser-opening disabled. The user approved direct task-file creation as the temporary fallback.

Implemented:
- Scoped Telegram pairing-code consumption by passing `scope_type` and `scope_id` through the webhook link path and enforcing those values inside the runtime repository.
- Stored newly created Telegram pairing codes as HMAC-SHA256 digests using the shared AuthNZ HMAC key derivation, while preserving the existing raw one-time code return for the admin link-start response.
- Kept legacy plaintext pairing-code consumption as a scoped fallback for active rows created before this fix.
- Hardened Telegram session mapper inputs to reject non-scalar IDs and derive persona/character UUIDs from canonical JSON component boundaries.

Verification:
- `python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_telegram_runtime_repo.py tldw_Server_API/tests/Telegram/test_telegram_session_mapper.py tldw_Server_API/tests/Telegram/test_telegram_linking_and_policy.py::test_link_command_consumes_pairing_code_with_current_scope -q` passed: 13 passed, 57 warnings.
- `DATABASE_URL=sqlite:////tmp/tldw_tg_endpoint_task12007.db python -m pytest tldw_Server_API/tests/Telegram/test_telegram_linking_and_policy.py -q` passed: 7 passed, 324 warnings.
- `python -m bandit tldw_Server_API/app/core/Telegram/session_mapper.py tldw_Server_API/app/core/AuthNZ/repos/telegram_runtime_repo.py tldw_Server_API/app/api/v1/endpoints/telegram_support.py -f json -o /tmp/bandit_telegram_core_review.json` passed: no findings.
- PR: https://github.com/rmusser01/tldw_server/pull/2498

PR review/rebase follow-up:
- Replayed the PR changes onto latest `origin/dev` (`ab4ebc51ed0c958f3cfc12b3b7a3bf387aa4dd2a`) after sandbox restrictions blocked in-place `git rebase` metadata writes.
- Checked issue comments, review comments, formal reviews, and review threads on PR #2498. There were no inline/formal actionable review comments. Bot comments were status/summary only; Qodo suggested a future dedicated pairing-code hash column/migration as a follow-up, but did not flag it as a required fix for this PR.
- Addressed Qodo PR review comments after rebase by adding helper docstrings, caching Telegram pairing-code HMAC key derivation, adding pytest unit markers/return types to new tests, and splitting the link-command scope regression into separate wrong-scope and correct-scope tests.
- Re-verified after review fixes:
  - `python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_telegram_runtime_repo.py tldw_Server_API/tests/Telegram/test_telegram_session_mapper.py tldw_Server_API/tests/Telegram/test_telegram_linking_and_policy.py::test_link_command_rejects_pairing_code_for_wrong_scope tldw_Server_API/tests/Telegram/test_telegram_linking_and_policy.py::test_link_command_links_pairing_code_for_current_scope -q` passed: 14 passed, 58 warnings.
  - `DATABASE_URL=sqlite:////tmp/tldw_tg_endpoint_task12007_reviewfix.db python -m pytest tldw_Server_API/tests/Telegram/test_telegram_linking_and_policy.py -q` passed: 8 passed, 324 warnings.
  - `python -m bandit tldw_Server_API/app/core/Telegram/session_mapper.py tldw_Server_API/app/core/AuthNZ/repos/telegram_runtime_repo.py tldw_Server_API/app/api/v1/endpoints/telegram_support.py -f json -o /tmp/bandit_telegram_core_review_reviewfix.json` passed: no findings.

Local verification caveat:
- Full `tldw_Server_API/tests/Telegram/test_telegram_linking_and_policy.py` failed against the default workspace SQLite AuthNZ database before policy assertions in the admin bot seeding helper because the existing `org_provider_secrets` table lacks the `created_by` column (`sqlite3.OperationalError: table org_provider_secrets has no column named created_by`). The same suite passed against a fresh temporary AuthNZ database, so this is local schema drift rather than a Telegram runtime/session regression.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the Telegram review findings by scope-binding pairing code redemption, hashing newly stored pairing codes, and hardening Telegram session identity derivation against non-scalar inputs and delimiter collisions. Added focused regression coverage for repository storage/consumption, session mapper validation, and webhook link-command scope propagation. Targeted tests, the Telegram endpoint policy suite against a fresh temporary AuthNZ database, and Bandit passed for the touched scope; the default workspace AuthNZ SQLite database still has local schema drift noted above.
<!-- SECTION:FINAL_SUMMARY:END -->
