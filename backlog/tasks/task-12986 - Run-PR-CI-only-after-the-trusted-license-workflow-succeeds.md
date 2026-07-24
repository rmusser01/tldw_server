---
id: TASK-12986
title: Run PR CI only after the trusted license workflow succeeds
status: In Progress
assignee: []
created_date: '2026-07-24 04:08'
updated_date: '2026-07-24 14:39'
labels:
  - ci
  - github-actions
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2758'
documentation:
  - Docs/superpowers/specs/2026-07-24-workflow-run-license-first-ci-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the trusted frontend-license workflow the only workflow started by main/dev pull-request activity. Trigger ordinary PR CI from its successful workflow_run completion, validate the exact current PR head before running untrusted code, and use no PAT, GitHub App, label, or polling waiter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The trusted frontend-license workflow is the only workflow directly triggered by main/dev pull-request activity.
- [ ] #2 Ordinary PR CI starts only from a successful completed run of the trusted license workflow.
- [ ] #3 Every downstream workflow validates the upstream workflow identity and the current PR head/base before running PR code.
- [ ] #4 Downstream workflow permissions are explicit and no new secret, PAT, GitHub App, label, or polling waiter is introduced.
- [ ] #5 Existing non-PR triggers remain available and PR event expressions are translated safely.
- [ ] #6 Workflow contract tests, actionlint, focused CI tests, Bandit for touched Python, and diff checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Requester corrected the objective: the license check must complete successfully before any other PR CI begins. The status-bypass direction was a misunderstanding and will not be implemented; no live ruleset changes were made. Selected credential-free approach: workflow_run chaining from Frontend License Gate Audit.

Corrected workflow_run specification passed independent review on iteration 3. Iteration 1 fixed server-side non-success skipping, unsupported-base PR trigger preservation, exact workflow/PR payload field validation, job-scoped permissions and credentialless checkouts, CodeQL no-upload PR analysis, default-branch rollout, check association canarying, and path-filter edge behavior. Iteration 2 added the normative LICENSE_FIRST_CI_ENABLED cutover guard, !cancelled() semantics, and pre-admission workflow-level concurrency expressions. Iteration 3 approved with no blockers.

Pre-PR verification on 2026-07-24: source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/CI/test_frontend_license_gate_workflow.py tldw_Server_API/tests/CI/test_required_workflow_contracts.py -> 54 passed, 2 warnings in 12.81s. git diff --check origin/dev...HEAD passed. Final PR diff contains only the corrected workflow_run design and TASK-12986 record. Bandit is not applicable because the PR adds no executable code.

Draft PR #2758 opened against dev from codex/trusted-license-first-ci. It remains draft pending requester review and the repository-required requester-written Change summary.

Requester approved the strict license-first design on 2026-07-24. Independent re-review found and the spec now addresses: head-scoped stale-run semantics, base-branch advancement without admission deadlock, conservative fail-open path routing under conflicting GitHub file-limit documentation, event-loss-free cutover/rollback ordering, and main-as-canonical CI definition parity for dev-targeted PRs.

Independent specification re-review approved on 2026-07-24 after resolving base-advancement path-routing ambiguity and replacing Actions reruns with fresh supported PR activity during cutover.

Post-review verification on 2026-07-24: focused CI contract suites passed (54 passed, 2 warnings in 14.63s); git diff --check passed. This commit remains documentation/task-record only, so Bandit is not applicable.

State correction on 2026-07-24: GitHub reports PR #2758 is currently ready for review (not draft) at head a2f3338c575b; no merge or ruleset change was performed.
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
