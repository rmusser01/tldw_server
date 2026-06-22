---
id: TASK-2395
title: Fix README and getting-started P1/P2 onboarding docs
status: Done
assignee: []
created_date: '2026-06-21 23:46'
updated_date: '2026-06-22 00:01'
labels:
  - docs
  - onboarding
dependencies: []
references:
  - README.md
  - Docs/Getting_Started/QUICKSTART.md
  - Docs/Getting_Started/Profile_Local_Single_User.md
  - Docs/Getting_Started/TROUBLESHOOTING.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address reviewed P1/P2 new-user documentation issues in README and Docs/Getting_Started. Scope: command guidance, manual local auth setup, local WebUI setup completeness, setup surface consistency, current Windows/no-make troubleshooting, and NEXT_PUBLIC_X_API_KEY wording.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 README no longer recommends make quickstart-prereqs as a first fresh-checkout Docker step that can fail before setup tooling exists.
- [x] #2 QUICKSTART manual local path includes single-user auth setup before starting uvicorn.
- [x] #3 NEXT_PUBLIC_X_API_KEY guidance consistently describes local single-user quickstart bootstrap behavior and production caution.
- [x] #4 Local profile includes or links complete WebUI dependency/setup commands before bun run dev.
- [x] #5 QUICKSTART setup wizard wording matches the canonical WebUI-first setup and /setup recovery/operator positioning.
- [x] #6 Windows/no-make troubleshooting uses current single-user compose files and PowerShell syntax.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented docs repairs for P1/P2 onboarding review: README preflight wording, QUICKSTART manual local auth init and /setup recovery wording, Local profile WebUI setup commands plus published mirror, TROUBLESHOOTING Windows/no-make compose update and NEXT_PUBLIC_X_API_KEY clarification, and focused docs regression coverage.

Verification: source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py tldw_Server_API/tests/Docs/test_onboarding_entrypoints.py tldw_Server_API/tests/Docs/test_onboarding_default_contract.py tldw_Server_API/tests/Docs/test_quickstart_same_origin_docs.py tldw_Server_API/tests/Docs/test_public_onboarding_profile_parity.py tldw_Server_API/tests/Docs/test_published_onboarding_parity.py => 30 passed. Local Markdown path check for edited files => MISSING_LINKS=0. Bandit: source .venv/bin/activate && python -m bandit -r tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py -f json -o /tmp/bandit_task_2395.json => 0 issues. Bandit not applicable to Markdown-only touched docs.

Known skips/blockers: no app/server runtime smoke was started because this change only edits documentation and a docs contract test. No subagent code-reviewer was spawned because the available subagent tool requires the user to explicitly request delegation; local verification was run instead.

PR: https://github.com/rmusser01/tldw_server/pull/2427
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the P1/P2 onboarding documentation issues from the README/Getting Started review. Updated README preflight positioning, QUICKSTART manual local auth setup and /setup recovery language, local WebUI setup steps plus published mirror, troubleshooting for Windows/no-make compose commands, and NEXT_PUBLIC_X_API_KEY guidance. Added a focused docs regression test for these contracts.
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
