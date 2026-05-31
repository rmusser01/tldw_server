# Phase 4.1 Coverage Ratchet Measurement Packet

**Date:** 2026-04-25

**Status:** Measurement packet complete; fresh baseline not run in this dirty workspace.

## Purpose

Convert the Phase 4.1 coverage ratchet plan into an execution packet for the future clean measurement run. This packet records the current CI contract, what must be measured, where the ratchet is enforced, and why no threshold should change before a fresh baseline exists.

## Current CI Contract

Source: `.github/workflows/coverage-required.yml`.

Current behavior:

- The `coverage-required` job is path-conditional.
- If coverage is not required for the changed paths, the job exits through a no-op pass.
- If coverage is required, CI installs backend dependencies with `dev,multiplayer` extras on Python 3.12.
- The current backend scope is:
  - `tldw_Server_API/tests/unit`
  - `tldw_Server_API/tests/sanity_tests`
- The current marker filter is:
  - `-m "not jobs and not e2e"`
- The current global backend floor is:
  - `--cov-fail-under=4`

Contract test:

- `tldw_Server_API/tests/CI/test_required_workflow_contracts.py::test_coverage_required_uses_documented_global_floor`
- This test asserts `--cov-fail-under=4` remains present in the workflow.

Frontend signal:

- `apps/tldw-frontend/package.json` has `test:coverage` as `vitest run --coverage`.
- `apps/packages/ui/package.json` has focused Vitest scripts but no package-level coverage script today.

## Why Baseline Was Not Run Here

This workspace is intentionally dirty with unrelated user/other-agent work. A coverage baseline should be recorded from a clean accepted base because:

- coverage output writes artifacts such as coverage XML;
- unrelated dirty runtime files can change import paths, startup behavior, and measured coverage;
- Phase 4 remains deferred until Phase 2/3 closeout is stable;
- a baseline from this workspace would be noisy and hard to defend in CI policy.

## Backend Measurement Command

Run from a clean worktree:

```bash
source .venv/bin/activate
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
python3 -m pytest -q --maxfail=1 --disable-warnings -p pytest_cov -p pytest_asyncio.plugin \
  -m "not jobs and not e2e" \
  tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests \
  --cov=tldw_Server_API --cov-report=xml --cov-report=term-missing --cov-fail-under=0
```

Record:

- branch and commit SHA
- Python version
- dependency install method
- total coverage percentage
- failed tests, if any
- skipped test count, if visible
- slowest or most fragile files, if visible
- generated coverage artifact path

## Frontend Measurement Command

Run separately:

```bash
cd apps/tldw-frontend
bun run test:coverage
```

Record:

- branch and commit SHA
- Bun version
- whether `apps/packages/ui` code is included in the coverage output
- total line/branch/function coverage reported by Vitest
- failed tests, if any

Do not combine backend and frontend coverage percentages into one threshold.

## Ratchet Decision Rules

Recommended:

- Keep `--cov-fail-under=4` until a fresh backend baseline is recorded.
- Raise the backend floor only to a value below the measured baseline with a buffer.
- Update `.github/workflows/coverage-required.yml` and `tldw_Server_API/tests/CI/test_required_workflow_contracts.py` together.
- Keep threshold-only PRs separate from DB, endpoint, Phase 3 API contract, and docs work.
- Do not make frontend coverage required until the frontend scope and ownership are agreed.

Candidate backend ratchet:

- first raise: measured baseline minus 1 point, or current floor plus 1 point, whichever is lower-risk;
- later raises: 10%, 15%, 20%, then 25% after decomposition slices add real tests.

## Implementation Handoff

When maintainers approve Phase 4.1 execution:

1. Create a clean worktree from the accepted base.
2. Run backend measurement with `--cov-fail-under=0`.
3. Run frontend measurement separately.
4. Write a dated baseline note under `Docs/superpowers/reviews/phase4-parking-lot/`.
5. Propose a small backend floor bump only if the measured baseline supports it.
6. Update the workflow and CI contract test in the same PR.
7. Run the CI contract test:

```bash
source .venv/bin/activate
python3 -m pytest tldw_Server_API/tests/CI/test_required_workflow_contracts.py -v
```

## Do Not Do

- Do not raise directly to 25%.
- Do not change coverage thresholds from a dirty workspace.
- Do not combine backend and frontend thresholds.
- Do not include jobs, e2e, external API, or local model suites in the required global floor without an explicit policy update.
- Do not combine threshold changes with Phase 3 helper implementation.

## Handoff Checklist

- [ ] Clean accepted base chosen.
- [ ] Backend baseline measured with `--cov-fail-under=0`.
- [ ] Frontend baseline measured separately.
- [ ] Baseline note created with command output summary.
- [ ] Maintainers accept the first threshold bump.
- [ ] Workflow and CI contract test are updated together.
