# Phase 4.1 Coverage Ratchet Baseline Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. This plan is for measurement and ratchet design only. Do not raise CI coverage thresholds before recording a fresh baseline.

**Goal:** Establish a defensible path from the current low global coverage floor toward a 25% backend coverage ratchet without destabilizing unrelated PRs.

**Architecture:** Measure first, ratchet second. Keep backend and frontend coverage baselines separate. Use existing CI path classification and avoid mixing coverage threshold changes with endpoint, DB, or API contract migrations.

**Tech Stack:** pytest, pytest-cov, Vitest coverage, GitHub Actions

---

## Current Signals

- `pyproject.toml` includes `pytest-cov>=7.1.0` in the `dev` extra.
- `.github/workflows/coverage-required.yml` currently runs:
  - `pytest -q --maxfail=1 --disable-warnings -p pytest_cov -p pytest_asyncio.plugin`
  - marker filter: `-m "not jobs and not e2e"`
  - paths: `tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests`
  - coverage target: `--cov=tldw_Server_API --cov-report=xml --cov-report=term-missing --cov-fail-under=4`
- `tldw_Server_API/tests/CI/test_required_workflow_contracts.py` asserts the documented global floor is `--cov-fail-under=4`.
- `apps/tldw-frontend/package.json` has `test:coverage` using `vitest run --coverage`.
- `apps/packages/ui/package.json` has focused Vitest scripts but no package-level coverage script today.

Measurement handoff packet:

- `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-coverage-ratchet-measurement-packet.md`

## Stage 1: Measure Backend Baseline

**Goal:** Capture the current backend coverage baseline using the same scope as CI.

**Success Criteria:** A dated baseline note records command, environment, total coverage, slowest/failing areas, and whether skipped suites affect the number.

**Tests:** Existing coverage-required command only.

**Status:** Complete - measured 2026-05-03

- [x] Create a clean worktree from the accepted base.
- [x] Activate the virtual environment.
- [x] Run the current CI-equivalent command:

```bash
source .venv/bin/activate
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
python3 -m pytest -q --maxfail=1 --disable-warnings -p pytest_cov -p pytest_asyncio.plugin \
  -m "not jobs and not e2e" \
  tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests \
  --cov=tldw_Server_API --cov-report=xml --cov-report=term-missing --cov-fail-under=0
```

- [x] Record the coverage percentage.
- [x] Record failed or skipped behavior if the command does not complete.
- [x] Do not change the CI floor in this stage.

## Stage 2: Measure Frontend Baseline

**Goal:** Capture frontend coverage separately from backend coverage.

**Success Criteria:** A dated baseline note records WebUI coverage command, result, and whether shared UI package coverage is included.

**Tests:** Existing frontend coverage script.

**Status:** Complete with blockers - measured 2026-05-03

- [x] Run WebUI coverage:

```bash
cd apps/tldw-frontend
bun run test:coverage
```

- [x] Decide whether `apps/packages/ui` needs a separate coverage script.
- [x] Do not merge frontend and backend percentages into a single threshold.

Result: WebUI coverage required adding `@vitest/coverage-v8@4.0.18`; after that, the full suite exposed broad unrelated frontend failures and eventually hit a Node heap OOM before producing a reliable coverage percentage. Frontend ratcheting remains blocked on separate test-suite stabilization or sharding.

## Stage 3: Define Ratchet Policy

**Goal:** Decide how to move from the current backend floor to 25%.

**Success Criteria:** Maintainers accept an incremental ratchet policy.

**Tests:** None.

**Status:** Complete - first ratchet recorded 2026-05-03

Recommended policy:

- Keep current global backend CI floor at `4` until a fresh baseline is recorded.
- Increase by small increments only when the measured baseline exceeds the next threshold by a buffer.
- Start with backend-only global floor.
- Add touched-scope coverage expectations only after the global floor is not noisy.
- Do not require slow, jobs, e2e, external API, or local model suites for the global coverage floor.

Candidate ratchet:

- baseline + 1 point after first successful baseline PR
- 10% after repeated stable runs
- 15% after high-churn Phase 3 API contract work settles
- 20% after DB/endpoint decomposition begins adding focused tests
- 25% after decomposition slices add enough coverage to sustain it

## Stage 4: Update CI And Contract Tests

**Goal:** Change CI only after the ratchet policy is accepted.

**Success Criteria:** Workflow floor and contract test agree.

**Tests:**

```bash
source .venv/bin/activate
python3 -m pytest tldw_Server_API/tests/CI/test_required_workflow_contracts.py -v
```

**Status:** Complete - first backend floor update

- [x] Update `.github/workflows/coverage-required.yml`.
- [x] Update `test_coverage_required_uses_documented_global_floor`.
- [x] Add a dated note explaining why the new floor is reachable.
- [x] Keep threshold-only PRs separate from runtime refactors.

## Stage 5: Prepare 25% Follow-Up

**Goal:** Make 25% achievable through targeted tests rather than threshold churn.

**Success Criteria:** High-value modules have test-backlog entries tied to Phase 4.3 and 4.4 decomposition.

**Tests:** TBD per module.

**Status:** Not Started

- [ ] Link coverage gaps to DB hotspot inventory.
- [ ] Link coverage gaps to endpoint hotspot inventory.
- [ ] Prioritize tests for modules already being decomposed.
- [ ] Avoid writing shallow tests that only inflate coverage without behavior value.

## Out Of Scope

- Raising `--cov-fail-under` directly to 25 in one PR.
- Combining threshold changes with Phase 3 envelope/pagination/auth implementation.
- Combining threshold changes with large DB or endpoint decomposition.
- Treating frontend and backend coverage as the same threshold.
