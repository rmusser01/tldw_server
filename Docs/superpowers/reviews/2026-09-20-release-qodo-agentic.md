# Release candidate deep Qodo review

Tracking: TASK-13263.1. Review: [PR2972 agentic review](https://github.com/rmusser01/tldw_server/pull/2972#issuecomment-5752955730).

This is a separate set of 18 findings following the 23 original PR2761 findings and the later Notes reviews. A row is closed only after its fix or disposition is verified.

| # | Finding | Status and evidence |
|---|---|---|
| 1 | Release source disagreement | Reconciled current inventory, plan and Backlog authority with release.json; old checkpoints explicitly historical. New CI test fails on the old documents. Final source/manifest refresh follows all protected edits. |
| 2 | Missing chat loading state | Fix and regression in progress. |
| 3 | Media account authority | Confirmed independently; fix and account-transition regressions in progress. |
| 4 | User message fallback name | Fix and regression in progress. |
| 5 | Image-detail retry persistence | Fix and regression in progress. |
| 6 | OSCE endpoint helper docstrings | Fix in progress. |
| 7 | OSCE focus timers | Fix and regressions in progress. |
| 8 | Shared UI coverage omitted | Added separate bounded coverage report under apps/packages/ui, using its own Vitest setup/aliases. The WebUI exclusion prevents running UI under the wrong package configuration. Required package-owned UI unit shards already enforce results. Existing report-only coverage policy remains: AGENTS says aim for >80%, not a repository-wide blocking threshold; this repair does not claim 80% has been achieved or silently expand the separately scoped global frontend work. Workflow regression was red before this change, green afterward. |
| 9 | Two host test classifications | Rejected: integration is the sole test-type classification; vz_linux_host_failure_drill is a registered orthogonal manual-selection marker (pyproject.toml). skipif is a platform condition. Removing the selector would weaken operator selection; removing integration would omit the real scenario from category collection. |
| 10 | OSCE loose null checks | Fix in progress, preserving both null and undefined semantics. |
| 11 | Prompt focus timer | Fix and regression in progress. |
| 12 | OSCE schema helper docstrings | Fix in progress. |
| 13 | Real VM operations | Intentional approved integration acceptance, not a unit test. TASK-13243.3 explicitly requires real VSock missing-exec metadata and a negative control that actually executes when the guard is disabled. Separate E2E and fault-injection opt-ins, disposable bundles, explicit isolated helper, empty initial VM inventory, owned-resource cleanup and retained errors constrain the test. Portable fake-client tests cover cleanup failures and lost create replies. Replacing the live boundary with a fake would invalidate its purpose; existing live evidence is in Docs/Sandbox/vz-linux-prepared-host-evidence.md. No live VM operation was performed for this review. |
| 14 | OSCE untranslated guidance | Fix and regression in progress. |
| 15 | Flashcard loose null checks | Fix in progress, preserving session ownership semantics. |
| 16 | OSCE finite request limit | Boundary investigation and regression in progress. |
| 17 | Public setup model paths | Confirmed local anonymous exposure; remote setup already requires admin. Fix preserves model input validation/execution and authenticated configuration resume while redacting the public state. Independent security review pending. |
| 18 | Untracked host skips | Added TASK-13243.3 / GitHub issue1442 to each local prerequisite skip and documented intentional manual acceptance. Host eligibility is a prerequisite, not a disabled failing test. |

Parent verification: 14 workflow/portable host tests passed, one live integration deselected. Scoped Bandit (excluding test assertions) reports zero findings. Ruff has the same five pre-existing broad cleanup exception diagnostics as HEAD; those catches intentionally retain every cleanup error and do not swallow success. Current authority consistency test is separately validated. Final combined verification and review remain in progress.
