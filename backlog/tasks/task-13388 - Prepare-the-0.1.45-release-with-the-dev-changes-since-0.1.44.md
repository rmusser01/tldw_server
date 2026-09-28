---
id: TASK-13388
title: Prepare the 0.1.45 release with the dev changes since 0.1.44
status: In Progress
assignee: []
created_date: '2026-09-27 20:09'
updated_date: '2026-09-28 00:18'
labels:
  - release
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Patch release. Puts the license-gate publish fix (#3029, #3032; TASK-13361) on main. The gate runs on pull_request_target, which executes the default branch's workflow file, so the fix had no effect on PRs until it reached main. It also rolls up #3030 (SQLite users bootstrap under sqlglot 30.20) and the #3029 ledger changes.

Release branch release/0.1.45 is cut from main (v0.1.44 line) and merges origin/dev. The one conflict (profile_user_write_guard AUTOINCREMENT check) is resolved to 0.1.44's shipped backend-scoped version.

Protected source: `6b87adbec20d910f05f109c2d31d39167ef8f09a`.
Protected manifest SHA-256: `1739e25a6dd4bdcfc5d7385366e4b0d9c620f99bf4ae95d3b049cfba37d8a58e`.

The protected frontend is unchanged since 0.1.44: the manifest is byte-identical, 7370 files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 LICENSES/releases/0.1.45 record, countdown grant and manifest are consistent (test_licensing_policy passes, including TLDW_VERIFY_RELEASE_SOURCE=1)
- [x] #2 pyproject and app version are 0.1.45; CHANGELOG has the 0.1.45 entry
- [x] #3 Owner has reviewed the licence record in the release PR before merge
- [ ] #4 After merge: main synced back to dev; TASK-13361 AC #1/#4 verified live on a PR
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-27: released. #3034 merged to main at 96bf0996eb after owner approval, with all 7 required checks green and the Watchlists RC gates confirmed failing identically on main (pre-existing). Annotated tag v0.1.45; GitHub release published; PyPI publish, publish-ghcr-main and the Docker images all succeeded, and PyPI serves tldw-server 0.1.45. TASK-13361 verified live on the published gate. Remaining for AC #4: the main-to-dev sync #3035, held behind #3033 (the other session's 0.1.44 sync) and blocked by the pre-existing Playground coordinator failure (TASK-13391).
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
