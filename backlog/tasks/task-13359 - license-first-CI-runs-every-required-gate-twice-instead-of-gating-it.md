---
id: TASK-13359
title: license-first CI runs every required gate twice instead of gating it
status: To Do
assignee: []
created_date: '2026-09-23 15:20'
labels:
  - ci
  - cost
  - tech-debt
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`LICENSE_FIRST_CI_ENABLED` is meant to be an *ordering* control: run the cheap frontend license gate first, and admit the expensive required gates only once it passes. The implementation adds that path without suppressing the old one, so with the flag on each gate runs **twice** per PR.

Verified on dev (91e8bbf), 2026-09-23, using `backend-required.yml` (all six required gates plus codeql, container-build-check, pre-commit, e2e-smoke, pypi-package and the frontend UX gates share the shape):

- The workflow triggers on both `pull_request` and `workflow_run: [Frontend License Gate Audit]` (`:3-11`).
- `LICENSE_FIRST_CI_ENABLED` appears **once**, on the `admission` job (`:23`), which is additionally `github.event_name == 'workflow_run'`.
- The `changes` job admits either path: `(workflow_run && admission succeeded && should_run) || github.event_name != 'workflow_run'` (`:39-47`). The second clause is unconditional, so the direct `pull_request` path runs the full gate regardless of the flag.

**Effect:** the flag roughly doubles CI volume rather than reducing it. Measured while four PRs were open (#2992, #2994, #2996, #2997): ~15 `pull_request` runs per PR already dispatched, with the matching ~14 `workflow_run` runs still to spawn once each license audit completes — ~29 runs per PR where ~15 would do.

This became visible only after #2989. Before that fix the two paths shared a concurrency group and cancelled each other, so the duplication presented as gates that never reported (TASK-13355) rather than as queue pressure. With the paths correctly scoped by event, both now survive and both consume runners.

**Observed consequence:** with those four PRs plus one other active branch holding 15 queued runs, every required gate sat `queued` for 35+ minutes — including the single-job `Frontend License Gate Audit` that the `workflow_run` path waits on, so the admission path cannot even start. Runner starvation, not a defect in the gates.

**Owner decision on intent.** If license-first is meant to gate, the `pull_request` path must be suppressed when the flag is on — e.g. make the `changes` job's second clause `(github.event_name != 'workflow_run' && vars.LICENSE_FIRST_CI_ENABLED != 'true')`, applied consistently across every caller. If both paths are wanted deliberately, the flag's name and the admission job's existence are misleading and should say so. The cheap interim lever is setting `LICENSE_FIRST_CI_ENABLED=false`, which drops the `workflow_run` path (its `changes` job needs `admission.result == 'success'`, and a skipped job is not a success) and halves the load — but that also turns the ordering control off, so it is a policy choice, not a tuning knob.

Related: `Docs/Development/CI_REQUIRED_GATES.md` documents the admission path; whichever way this is resolved, that section needs updating.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each required gate executes once per PR under the intended configuration
- [ ] #2 The chosen behaviour is applied consistently across every workflow that calls license-first-admission
- [ ] #3 Docs/Development/CI_REQUIRED_GATES.md matches the resolved behaviour
- [ ] #4 Measured: run count per PR before and after, recorded on the task
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
