---
id: TASK-13442
title: run-pre-commit checks a PR's last commit, or all of dev after update-branch,
  instead of the PR's diff
status: Done
labels:
- ci
- pre-commit
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
For pull_request events, .github/workflows/pre-commit.yml overrides FROM_REF with HEAD_SHA^. That was right when the job checked out GitHub's merge ref (HEAD^ = base tip); since 0b17bd6161 (2026-03-01) it checks out the PR head, so HEAD_SHA^ is the PR's previous commit. pre-commit then checks only the last commit's files, or, after gh pr update-branch, every file dev brought in (103 files instead of #3143's 8). That flags files the PR never touched (TASK-13434, end-of-file issues from dev) and misses earlier commits, which breaks ADR-059's claim that backlog-task-format catches a non-canonical task file on the PR that adds it. pre-commit already diffs FROM...TO from the merge base, so FROM_REF=base.sha is the PR's own diff, as the workflow_run path does (TASK-12986). Also migrates deprecated hook stage names (commit/push to pre-commit/pre-push).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The pull_request branch of run-pre-commit uses the PR base commit as FROM_REF, with no HEAD_SHA^ override, and the contract test asserts that
- [x] #2 Hook stages use pre-commit/pre-push names, so pre-commit no longer warns about deprecated stages
- [x] #3 Workflow contract tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified that pre-commit's get_changed_files diffs old...new (three-dot, from the merge base). On #3143's update-branch head 3c9038adbf, HEAD^..HEAD lists 103 files (all from dev); base...HEAD lists #3143's own 8. test_license_first_workflow_contracts.py plus the e2e-budget contract: 32 passed (the e2e-budget test needs 'python' on PATH; it fails identically on clean origin/dev without it). pre-commit validate-config: ok. Bandit: not applicable (YAML and a test assertion only).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
run-pre-commit now checks the PR's own diff (FROM_REF = PR base commit; pre-commit diffs from the merge base) instead of HEAD_SHA^, which since the 2026-03 checkout change meant the last commit only, or all of dev after update-branch. The contract test now forbids HEAD_SHA^ in the pull_request branch, matching the workflow_run branch. Hook stages migrated to pre-commit/pre-push. No skips.
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
