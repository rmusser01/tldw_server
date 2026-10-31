---
id: TASK-13447
title: Add tests for uncovered storage/oauth/OSCE endpoints and cookie_cloner crypto
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 1 Stage 4. Plan: Docs/plans/2026-10-01-due-debt-sweep-implementation-plan.md. Zero-test-reference endpoints: storage_trash, storage_user_files, storage_user_folders (data deletion - highest risk), quizzes_osce, discord_oauth_admin, slack_oauth_admin. Plus unit tests for cookie_scraping/cookie_cloner.py (PBKDF2/AES round-trip + negative cases). Verify zero-coverage claim per file via grep before writing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Executed 2026-10-03. Real gap was cookie_cloner only: 13 synthetic unit tests added (round-trips, wrong-key no-leak, malformed input, Safari parser). The other five surveyed modules verified already covered: storage trio via tests/Storage URLs; quizzes_osce via tests/Quizzes/test_osce_* (17 passed, 1 pre-existing failure at base); oauth admin impls via tests/Integrations. Survey's module-name grep was the wrong heuristic (URL/indirect coverage existed).
Review (Approved, 3 minors): attribution corrected - oauth admin impls are genuinely exercised by tests/Discord/test_discord_oauth_lifecycle.py, test_discord_policy_hardening.py, tests/Slack/test_slack_oauth_lifecycle.py, test_slack_policy_hardening.py (Integrations file only monkeypatches one impl). Bandit recorded: 11 findings all B101 (pytest asserts), zero Medium/High. Residual minor gap (follow-up optional): GET /files/{file_id} single-metadata route lacks direct HTTP-level test.
Renumbered 2026-10-04 from TASK-13402 when PR #3155 was rebased onto dev: the branch was cut from codex/post2970-uat-20260920, and dev had meanwhile given that id to an unrelated task. Branch commit messages and code references use the new id. The stage plan (Docs/Plans/2026-10-01-due-debt-sweep-implementation-plan.md) exists only on that unmerged branch.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
cookie_cloner crypto covered with 13 deterministic unit tests; endpoint-gap claims corrected with evidence.
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
