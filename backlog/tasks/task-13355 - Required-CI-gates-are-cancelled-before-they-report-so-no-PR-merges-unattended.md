---
id: TASK-13355
title: 'Required CI gates are cancelled before they report, so no PR merges unattended'
status: To Do
assignee: []
created_date: '2026-09-23 04:52'
updated_date: '2026-09-23 06:11'
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
- [x] #1 A decision is recorded on whether license-first CI is enabled or parked
- [ ] #2 A PR opened and left untouched reaches all six gates green without any manual re-run
- [ ] #3 If the pinned contracts are changed, each of the four is updated deliberately with the reason recorded, and Docs/Evidence/PR2761-codeql-actions.json is reconciled
- [ ] #4 Docs/Development/CI_REQUIRED_GATES.md no longer needs its manual landing procedure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DECISION TAKEN 2026-09-23: resolution A. LICENSE_FIRST_CI_ENABLED=true set on the repository at 05:14:57Z (gh variable set). Resolution B -- loosening the four pinned contracts -- is not pursued.

Effect NOT yet confirmed. Setting the variable makes admission run on workflow_run events instead of skipping, but no pull-request event has occurred since, so no evidence either way. The next PR through is the test. Two ways it can still fail:

- admission may DENY, in which case the gate jobs skip by design and the PR cannot merge. That would be worse than the previous state, and the remedy is 'gh variable delete LICENSE_FIRST_CI_ENABLED'.
- the pull_request run may still be cancelled while the admitted workflow_run run does the work. That is the intended design, but it only helps if the admitted run actually reports a status for each required check name.

Diagnostic for whoever picks this up, also recorded in Docs/Development/CI_REQUIRED_GATES.md:

  gh api "repos/rmusser01/tldw_server/actions/runs?event=workflow_run&per_page=20" --template '{{range .workflow_runs}}{{.name}} {{.status}}/{{.conclusion}}{{"\n"}}{{end}}' | grep -E "required|container-build-check"

completed/skipped on the required lanes means the admission path is still declining the work. As of the decision the most recent workflow_run runs still read completed/skipped, but all of them predate the variable being set.

RESOLUTION A TESTED AND INSUFFICIENT, 2026-09-23. Setting LICENSE_FIRST_CI_ENABLED=true does NOT unblock pull requests. Tested on PR #2988; I had called it working an hour earlier on partial evidence and that was wrong.

What it does change, confirmed:
- admission now succeeds instead of skipping. Run 35821927623 (event=workflow_run, created 05:19:24Z) shows 'admission / admission: completed/success'. Before the variable it was SKIPPED.
- the admitted run does the real work and passes -- completed/success on all three jobs including backend-required itself.

What it does not change:
- the status never reaches the pull request. That run's head_sha is c2bab8a5, the tip of main, not the PR head 116c9314. 'gh pr checks 2988' still lists exactly one check named backend-required and it is the CANCELLED one from the pull_request run.

ROOT CAUSE, and it is a design gap rather than a configuration gap. Permissions asymmetry:

  frontend-license-gate.yml   permissions: contents: read, statuses: write
                              -> posts explicitly against the PR head, which is why
                                 frontend-license-policy/trusted/dev DOES appear on PRs
  the six required workflows  permissions: contents: read
                              -> no explicit status; they rely on GitHub's implicit check
                                 run, which attaches to the run's own head_sha, and for a
                                 workflow_run event that is the default branch

So only the pull_request run can report to a pull request, and that is precisely the run the admitted one cancels. With the variable set, each PR now runs the gates twice and neither occurrence unblocks it -- strictly more compute for the same outcome.

WHAT THIS MEANS FOR THE OPTIONS.

Resolution A is not sufficient on its own. Either it needs pairing with one of the below, or the variable should be unset again to stop paying for the duplicate runs. Left set pending a decision; 'gh variable delete LICENSE_FIRST_CI_ENABLED' reverts it.

A new third option, and the smallest one that could work: make the concurrency group event-specific, e.g. group: <name>-${{ github.event_name }}-${{ ...pr number... }}, so a workflow_run run cannot cancel the pull_request run. The pull_request run then survives and reports against the correct SHA. This is contract-pinned by test_pr_context_and_base_diff_logic_are_workflow_run_safe, so it needs a deliberate contract update, but it touches the cancellation rather than the admission invariant that test_runner_roots_cannot_bypass_admission_and_checkouts_are_immutable exists to protect.

Resolution B alternative: give the six workflows statuses: write and have them post explicitly against needs.admission.outputs.head_sha, matching what frontend-license-gate.yml already does. Larger change, and it grants status-write to six more workflows.

Recommendation: the event-specific concurrency group. It is the narrowest change, it leaves admission untouched, and it fixes the actual failure (the reporting run being killed) rather than working around it.
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
