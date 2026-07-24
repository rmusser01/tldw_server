---
id: TASK-12986
title: Run PR CI only after the trusted license workflow succeeds
status: In Progress
assignee: []
created_date: '2026-07-24 04:08'
updated_date: '2026-07-24 13:47'
labels:
  - ci
  - github-actions
dependencies: []
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
