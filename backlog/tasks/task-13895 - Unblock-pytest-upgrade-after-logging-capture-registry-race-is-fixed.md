---
id: TASK-13895
title: Unblock pytest upgrade after logging capture registry race is fixed
status: To Do
labels:
- testing
- ci
- dependencies
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pytest 9.1.1 iterates the live logging logger registry while entering log capture. Background logger registration can raise RuntimeError: dictionary changed size during iteration, as previously seen by TASK-13820 and reproduced again while preparing PR3084. Retain pytest>=9.0.3,<9.1 until a verified fix permits a safe upgrade; then update the project dev extra and explicit in-process e2e CI requirement together.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Identify and link the upstream issue/fix and a released pytest version containing the logger-registry capture fix, or document why no fixed release is yet available.
- [ ] #2 Verify a deterministic concurrent-logger-registration regression against the real pytest capture implementation: current9.1.1 fails, the selected fixed release passes, and worker cleanup is bounded.
- [ ] #3 Retain ordinary and non-propagating logger capture, existing pytest plugins, failure diagnostics and assertions; do not disable logging capture or patch installed pytest to manufacture a passing result.
- [ ] #4 Validate the selected release with the focused logging/async/plugin checks and the in-process e2e workflow, including teardown, on supported Python versions.
- [ ] #5 After the repair passes, update pyproject.toml and .github/workflows/e2e-smoke.yml consistently and remove their temporary race-workaround comments; keep the current safe bound until then.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Related: PR3084 https://github.com/rmusser01/tldw_server/pull/3084; original workaround cc0a722c08 / TASK-13820. PyPI and upstream changelog checked2026-10-20: latest release9.1.1 has no documented fix. Fresh isolated Python3.12 probe: real _pytest.logging.catching_logs enters a non-propagating logger, a bounded worker registers another logger during that iteration, and9.1.1 raises RuntimeError dictionary changed size during iteration. Same control with9.0.3 succeeds because it does not traverse the mutable registry. A synchronization wrapper on that isolated logger only forces the real interleaving; no application runtime/source changes or vendor fix. The requester explicitly chose to retain pytest>=9.0.3,<9.1. Scope is future upstream-fix adoption and qualification, not a speculative dependency bump. ADR assessment: no new durable architecture decision; ADR059 governs this task record. ID13895 selected above current dev and all44open PRs after complete paginated file inventory; maximum observed13894.
References: https://pypi.org/project/pytest/9.1.1/ and https://docs.pytest.org/en/stable/changelog.html. Reproduction: (1) use an isolated Python process with the candidate pytest, (2) register a non-propagating seed logger, (3) synchronize its addHandler call with one worker creating another logger through logging.getLogger, (4) enter real _pytest.logging.catching_logs with a normal handler, (5) require the worker to finish within two seconds and join it during cleanup. On 9.1.1 the live loggerDict.values iterator raises RuntimeError; on 9.0.3 capture does not traverse that mutable registry and the control completes. This is tool-only causal evidence, not a repeated application/e2e run. No fix has been implemented.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
