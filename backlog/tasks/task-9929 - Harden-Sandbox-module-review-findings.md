---
id: TASK-9929
title: Harden Sandbox module review findings
status: Done
assignee: []
created_date: 2026-06-23 11:27
updated_date: 2026-06-26 06:41
labels:
- sandbox
- security
- review
dependencies: []
references:
- tldw_Server_API/app/core/Sandbox
- Docs/superpowers/plans/2026-06-23-sandbox-review-hardening.md
- https://github.com/rmusser01/tldw_server/pull/2509
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address validated findings from the current Sandbox module review. Scope is limited to `tldw_Server_API/app/core/Sandbox` and focused regression tests unless a validated fix requires a narrow adjacent test utility update.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated findings are reproduced with targeted tests before production changes.
- [x] #2 Confirmed findings are fixed in the Sandbox module.
- [x] #3 Non-applicable findings are documented with rationale.
- [x] #4 Focused tests and Bandit on touched scope are run before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-06-23-sandbox-review-hardening.md

Backlog CLI allocator reused an occupied TASK-2420 in this checkout because several current task files are untracked and not reflected in the allocator state. A manual TASK-2424 Sandbox record then collided with another untracked TASK-2424, so the Sandbox record was renumbered to TASK-9929.

Validated and fixed: snapshot restore symlink workspace root, Docker security-option fallback, Docker granular egress fail-open, Docker env-value log exposure, artifact storage symlink escapes, Lima/Firecracker non-zero status capture, and Lima/Firecracker shell env-file quoting/key validation.

Not treated as an actionable defect in this slice: SandboxService runtime dispatch duplication. It is a maintainability smell, but no behavior/security failure was reproduced; broad runtime-dispatch refactoring is deferred to a separate design/refactor task.

Verification:
- Focused red run before fixes: 11 new regression tests failed for the expected reviewed behaviors.
- Focused green run: `python -m pytest -q --confcutdir=tldw_Server_API/tests/sandbox tldw_Server_API/tests/sandbox/test_snapshot_manager_restore_security.py tldw_Server_API/tests/sandbox/test_docker_runner_hardening_defaults.py tldw_Server_API/tests/sandbox/test_docker_egress_enforcement.py tldw_Server_API/tests/sandbox/test_orchestrator_artifact_security.py tldw_Server_API/tests/sandbox/test_cross_runtime_cleanup_contracts.py` -> 34 passed, 1 skipped.
- Compile: `python -m compileall -q` on touched Sandbox production files -> passed.
- Diff check: `git diff --check` on touched files -> passed.
- Bandit: touched Sandbox production scope wrote `/tmp/bandit_sandbox_review_hardening.json`, errors=0, results=95. The new Docker readiness-gate subprocess finding was suppressed with a scoped `# nosec`; remaining results are the existing low-severity Sandbox subprocess baseline.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed.
- [x] #2 Tests or verification recorded.
- [x] #3 Documentation updated when relevant.
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip.
- [x] #5 Final summary added.
- [x] #6 Known skips or blockers documented.
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Sandbox snapshot restore, Docker runner, artifact storage, and Lima/Firecracker guest script/env handling for the validated review findings. Added focused regression coverage for each reproduced defect and documented the non-behavior runtime-dispatch concern as deferred refactor work. Focused Sandbox tests, compileall, and diff check passed; Bandit has no errors and only the existing low-severity subprocess baseline remains in the touched Sandbox scope.
<!-- SECTION:FINAL_SUMMARY:END -->

## PR Reference

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR opened: https://github.com/rmusser01/tldw_server/pull/2509
Rebased PR branch on latest origin/dev and addressed PR review feedback: Docker create failure messages now redact env values, Docker env redaction preserves --env names without inline values, Firecracker env files export variables to child commands, artifact writes use fd-relative openat-style traversal to close parent symlink race windows, artifact listing resolves root once, new artifact test module has docstrings, modified monkeypatch fixtures are typed, and Docker readiness-gate nosec is narrowed to explicit Bandit IDs. Verification after fixes: focused Sandbox suite 38 passed/1 skipped; compileall passed; git diff --check passed; Ruff passed on touched production files; Bandit JSON errors=0, high=0, medium=0, docker readiness findings=[] with existing low-severity subprocess baseline remaining.
Second PR review pass addressed newly surfaced CodeRabbit threads: renamed duplicate Backlog heading, cleaned partial fallback egress rules before raising, rejected symlinked artifact and snapshot ancestors, reset Docker ENTRYPOINT for granular allowlist runs, removed raw RunSpec env values from Docker debug logging, and added child-env propagation coverage for Lima plus existing Firecracker coverage. Verification after second pass: focused Sandbox suite 43 passed/1 skipped; compileall passed; git diff --check passed; Ruff passed on touched production files; Bandit JSON errors=0, high=0, medium=0, docker readiness findings=[] with existing low-severity subprocess baseline remaining.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
