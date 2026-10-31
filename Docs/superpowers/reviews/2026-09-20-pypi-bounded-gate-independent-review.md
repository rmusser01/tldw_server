# Independent review: bounded PyPI release gate adoption

Result: no actionable findings in the reviewed change.

Reviewed the working-tree delta against HEAD `965798cd1dca6f4ff790ae73485e18d157aab5cb` in `/Users/macbook-dev/.codex/worktrees/pypi-0142-test-gate-recovery/tldw_server2`. The reviewed delta contains the publish workflow, two CI contract files, and the existing Backlog task notes. This was a read-only source/diff review; no tests, builds, publishing, GitHub comments, pushes, merges, or repository edits were performed by this reviewer.

## Evidence

- The workflow and `test_pypi_workflow_contracts.py` match approved commit `f5c541db61c2cd14606d94533abe5569876c1548` exactly. The release workflow contract change applies only the corresponding PyPI test adjustment and retains the newer worker container, isolated local-package import, backend-only publishing, and frontend shared-hook contracts.
- `release-gate` needs `detect-version`, has a 30-minute job timeout, runs the targeted PyPI contracts with a 5-minute step timeout, and invokes the real subprocess startup smoke with `--timeout 150` and a 5-minute step timeout.
- The smoke launches the actual application through uvicorn in a scrubbed, explicitly minimal environment and requires `/health` to return HTTP 200 with exactly `{"status": "ok"}`. Early server exit or timeout returns failure. This is an ultra-minimal startup/liveness check, not comprehensive endpoint or optional-feature coverage.
- No `continue-on-error`, `always()`, or equivalent failure bypass was introduced. The build needs both version detection and the release gate. Its conditional does not override the normal successful-dependency requirement. A failed, cancelled, or timed-out gate prevents the build and consequently prevents both publishing jobs.
- Build still executes `make pypi-check`: clean sdist/wheel build, Twine validation, and backend/API-only artifact-content validation. Both trusted-publishing jobs still need the successful build and download its artifacts.
- Version detection, push filters, manual target selection, concurrency, action pins, credential persistence controls, environments, and permissions are unchanged. PyPI lookup failure suppresses automatic publication. Manual dispatch still selects TestPyPI by default and can explicitly select PyPI.
- `pyproject.toml` still declares `0.1.42`. The change does not modify package source, build configuration, packaging checker, startup smoke, version metadata, or legal/license files relative to the stated HEAD. This establishes source/configuration preservation; it does not claim byte-identical rebuilt archives.

## Publication consequence and limits

Merging this workflow change to `main` matches the existing workflow-path push trigger. If version `0.1.42` remains absent from PyPI and version lookup succeeds, the merged workflow will automatically attempt production publication after the bounded release gate and package build/validation succeed. This is a publication-enabling merge, not merely dormant CI maintenance.

Replacing the 63,144-test serial release gate is intentional and explicitly approved. The bounded gate provides targeted workflow contracts, minimal application startup/liveness, and package artifact checks; it does not establish full-suite correctness or validate all optional runtime features. Runtime results and hosted Actions verification remain the parent task's responsibility. The supplied fact that `0.1.42` is missing was not independently queried by this read-only source reviewer.
