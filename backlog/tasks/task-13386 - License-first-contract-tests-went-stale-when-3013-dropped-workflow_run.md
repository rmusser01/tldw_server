---
id: TASK-13386
title: 'License-first contract tests went stale when #3013 dropped workflow_run'
status: To Do
assignee: []
created_date: '2026-09-27 17:24'
labels:
  - ci
  - tests
  - tech-debt
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3013 (merged) dropped the workflow_run trigger from the 28 license-first gates and added an await_license job on the pull_request path. tests/CI/test_license_first_workflow_contracts.py was not updated, and two contract tests fail on dev (verified 2026-09-27):

- test_all_ordinary_workflows_call_exact_inert_admission_gate: KeyError 'workflow_run' at :330. It still asserts the removed trigger.
- test_runner_roots_cannot_bypass_admission_and_checkouts_are_immutable: at :400, ORIGINAL_JOB_NAMES does not include the new await_license job.

These are the contract that keeps a runner root from bypassing license admission, so they must describe the new design, not be deleted. Open question: tests/CI is in the gap-verified-12 shard of ci.yml, yet #3013 merged with these red. Establish whether that shard runs on PRs; if it does not, the shard gap is the larger defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both tests assert the post-#3013 design (await_license on pull_request, no workflow_run) across all 28 gated workflows
- [ ] #2 A deliberate bypass (a gate job without await_license) makes the tests fail
- [ ] #3 Explained why gap-verified-12 did not block #3013, and fixed if it is a shard gap
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
