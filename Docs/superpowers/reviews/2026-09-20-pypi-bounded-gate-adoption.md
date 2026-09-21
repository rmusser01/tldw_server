# Approved bounded PyPI gate adoption

Requester decision: "yes adopt it", approving the bounded publication gate
proposed in PR #2956. Tracking: TASK-13013.3 and TASK-13263; proposal TASK-13257,
commit `f5c541db61c2cd14606d94533abe5569876c1548`.

Only the three workflow/test hunks are adopted. Newer worker image/import and
frontend hook contracts are preserved. The release job runs contract tests and
the existing scrubbed-environment startup smoke (five-minute step limits,
150-second probe); overall job limit is 30 minutes. Build still requires a
successful gate and runs `make pypi-check`. Detection, manual targets, upload and
trusted publisher permissions/environments are unchanged. Normal CI is unchanged.
Merging the workflow path onto main automatically attempts the absent 0.1.42
version; no publication occurred during this local adoption.

## Verification

- Recovery test-first result: three failures and 12 passes against the old workflow,
  proving the contracts detect the unbounded gate/missing replacement.
- Recovery after adoption: 15 contract tests pass with publication-matching pytest
  plugins. Startup returns canonical HTTP 200; valid 0.1.42 wheel and sdist,
  Twine checks and backend-only contents checks pass.
- Candidate: 28 workflow/licensing tests pass, including all 13 explicit protected
  source checks. Startup returns canonical HTTP 200.
- Both worktrees: Actionlint passes, Ruff zero diagnostics, Bandit baseline 66 to
  current 58, with no new findings using exact triggering-line comparison.
  The first comparison falsely classified unchanged EOF context as new; comparing
  the actual triggering line confirms that assertion is unchanged.
- Both independent reviews and diff checks pass. The current protected frontend
  bytes/manifest, published 0.1.42 legal record, package version and application
  source are unchanged by this workflow adoption.

Logs: `/tmp/pypi0142-adopt-gate-{red,green,startup,package}.log`,
`/tmp/release0143-adopt-gate-{green,startup}.log`, and matching scoped static JSON
reports. Verification uses local Python 3.11; final GitHub CI is still required.
No result claims that all 63,144 repository tests passed. The explicitly approved
publication scope change replaces that job's gate rather than resolving every
unverified full-suite failure.
