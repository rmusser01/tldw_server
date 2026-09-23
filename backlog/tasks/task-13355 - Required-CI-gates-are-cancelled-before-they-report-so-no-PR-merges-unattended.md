---
id: TASK-13355
title: 'Required CI gates are cancelled before they report, so no PR merges unattended'
status: To Do
assignee: []
created_date: '2026-09-23 04:52'
labels:
  - ci
  - tech-debt
  - blocked-merges
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every push cancels all CI, including all six required gates, so mergeStateStatus stays BLOCKED indefinitely until someone manually re-runs them. Measured 2026-09-22 across PRs #2981-#2984 and six pushes: 45-50 checks per PR went to CANCELLED within roughly 40 seconds of each push, with zero FAILURE. Docs/Development/CI_REQUIRED_GATES.md claims 'Each required gate always reports a status for deterministic branch protection behavior'; in practice none of them does.

MECHANISM, evidenced end to end:

1. A pull-request event fires Frontend License Gate Audit, which triggers on pull_request_target with types [opened, reopened, synchronize, ready_for_review, edited]. 'edited' is included, so editing a PR description is enough.
2. Its completion fires every workflow declaring workflow_run: [Frontend License Gate Audit]. Those runs are attributed to the default branch, which is why they are invisible when listing runs for the PR branch -- that is what made this hard to see.
3. A workflow_run run resolves its concurrency group through github.event.workflow_run.pull_requests[0].number, the same PR number the pull_request run used, so both share one group and cancel-in-progress: true kills the pull_request run. license-first-admission.yml:67 requires that array to hold exactly one valid number, confirming it is populated.
4. That workflow_run run's admission job requires vars.LICENSE_FIRST_CI_ENABLED == 'true'. 'gh variable list' returns nothing -- no repository variables are set -- so admission is SKIPPED.
5. Each gate job requires needs.admission.result == 'success' for workflow_run events, so it skips too. Directly observed: event=workflow_run runs of backend-required, coverage-required, frontend-required and pre-commit all at completed/skipped.

Net: the run that would have reported a status is cancelled by a run that then reports nothing.

WHY THIS IS NOT A ONE-LINE FIX. Four separate levers are pinned by contract tests, so changing any of them fails the suite deliberately:

- The gate job's admission clause is pinned byte-for-byte by tests/CI/test_license_first_workflow_contracts.py::test_runner_roots_cannot_bypass_admission_and_checkouts_are_immutable. Its name states the invariant.
- The concurrency group is pinned byte-for-byte by test_pr_context_and_base_diff_logic_are_workflow_run_safe.
- cancel-in-progress: true is pinned by the same test.
- The admission guard including vars.LICENSE_FIRST_CI_ENABLED == 'true' is pinned at test_license_first_workflow_contracts.py:338 AND recorded as evidence in Docs/Evidence/PR2761-codeql-actions.json.

I implemented a fix (11 conditions across the six gates, letting the workflow_run path proceed when admission was skipped because the feature is off, plus an 18-case contract test verified to catch the regression) and reverted it on discovering the above. The code is self-consistent with its design; the configuration is in a state the design does not handle.

TWO CANDIDATE RESOLUTIONS, both needing a decision rather than a patch:

A. Set LICENSE_FIRST_CI_ENABLED=true. One repository variable. The design then works as intended: admission runs, admits, the gates execute, and the cancellation becomes the intended supersede. Risk: if admission denies a PR the gates skip by design, so this opts every PR into license-first gating. Cheapest if license-first is meant to be on.

B. Accept that license-first is parked and make the gates work with it off. That means loosening the admission clause and deliberately updating all four pinned contracts plus the PR #2761 evidence file. Design-doc sized, because the pinned tests exist to stop exactly this.

OPERATIONAL WORKAROUND, documented in Docs/Development/CI_REQUIRED_GATES.md and used to land #2981-#2984: rebase onto current dev and push; wait for the audit on that exact head SHA to complete; re-run the six gates individually paced ~40s apart; merge before anything else lands. Do not edit the PR description after opening it. --admin does not help: dev-core-required-gates has no bypass actors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded on whether license-first CI is enabled or parked
- [ ] #2 A PR opened and left untouched reaches all six gates green without any manual re-run
- [ ] #3 If the pinned contracts are changed, each of the four is updated deliberately with the reason recorded, and Docs/Evidence/PR2761-codeql-actions.json is reconciled
- [ ] #4 Docs/Development/CI_REQUIRED_GATES.md no longer needs its manual landing procedure
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
