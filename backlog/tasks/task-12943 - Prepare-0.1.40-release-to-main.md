---
id: TASK-12943
title: Prepare 0.1.40 release to main
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-10 03:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare the 0.1.40 dev-to-main release branch after PR #2698 merged into dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Package, FastAPI, README, and MkDocs release metadata are bumped to 0.1.40
- [x] #2 CHANGELOG.md and release notes promote current Unreleased content into 0.1.40
- [x] #3 Release branch is verified, committed, pushed, and opened as a PR to main
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Created isolated worktree .worktrees/release-main-0.1.40 from origin/dev on branch codex/release-main-0.1.40. origin/main already carries 0.1.39, so this is the next patch release, 0.1.40.

Prepared release branch codex/release-main-0.1.40 from origin/dev for main. Bumped package, FastAPI, README, and MkDocs release metadata to 0.1.40. Promoted current unreleased Chatbooks, chat document processing, CodeQL/security, media ingest, Guardian/audio/CI, and quiz/design-doc follow-up notes into 0.1.40 release notes and reset Unreleased.

Verification: git diff --check passed; pyproject.toml and Docs/mkdocs.yml parsed successfully with project venv Python; python -m py_compile tldw_Server_API/app/main.py passed; Bandit on tldw_Server_API/app/main.py wrote /tmp/bandit_task_12943.json with 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared the 0.1.40 release branch for a dev-to-main PR. Updated release metadata, changelog, README, published release notes, and Backlog tracking; verification and Bandit passed.
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
