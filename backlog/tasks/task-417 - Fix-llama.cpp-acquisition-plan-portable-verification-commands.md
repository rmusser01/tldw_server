---
id: TASK-417
title: Fix llama.cpp acquisition plan portable verification commands
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 03:11'
labels:
  - llamacpp
  - docs
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1810'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address review feedback after PR #1810 merged: remove developer-specific absolute virtualenv paths from the llama.cpp model acquisition/import workflow plan verification snippets so examples are copy/pasteable across machines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Verification snippets use portable python -m pytest / python -m bandit commands with a clear note to activate the project virtual environment first.
- [x] #2 Docs-only validation confirms no hardcoded local path remains in the plan and the diff is clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the finding on current origin/dev after PR #1810 merged: the plan contained seven absolute pytest invocations and one absolute Bandit invocation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced all developer-specific absolute Python invocations in the llama.cpp acquisition/import workflow plan with portable python -m pytest and python -m bandit commands, and added a command note to activate the project virtual environment from the repository root. Validation: rg found no local absolute path or absolute virtualenv invocation references in the plan/task, git diff --check passed, and the docs-only change skipped Bandit as not applicable.
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
