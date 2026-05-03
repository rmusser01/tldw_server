# Phase 4.1 Coverage Baseline - 2026-05-03

## Backend Baseline

- Base at initial measurement time: `4f2dda1ab2 docs: add sandbox security policy matrix (#1218)`.
- Post-rebase verification base: `1016a3b056 refactor(flashcards): split import/export tab panels (#1217)`.
- Python: `Python 3.11.13` from the repo-root virtual environment.
- Command:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q \
  --maxfail=1 --disable-warnings -p pytest_cov -p pytest_asyncio.plugin \
  -m "not jobs and not e2e" \
  tldw_Server_API/tests/unit tldw_Server_API/tests/sanity_tests \
  --cov=tldw_Server_API --cov-report=xml --cov-report=term-missing --cov-fail-under=0
```

- Result: `215 passed, 8 warnings in 66.42s`.
- Coverage XML summary: `lines-valid=592952`, `lines-covered=37753`, `line-rate=0.06367`.
- Terminal total: `6%`.
- Post-rebase ratchet check with `--cov-fail-under=5`: `215 passed, 8 warnings in 64.91s`; `Required test coverage of 5% reached. Total coverage: 6.37%`.
- Generated artifacts: `.coverage` and `coverage.xml` are produced locally and ignored by git.

## First Ratchet

The measured backend total supports raising the required backend global floor from `4` to `5`.
This keeps a one-point buffer below the observed `6%` total and avoids any jump toward the Phase 4 target before additional decomposition coverage exists.

The workflow and contract test should move together:

- `.github/workflows/coverage-required.yml`: `--cov-fail-under=5`
- `tldw_Server_API/tests/CI/test_required_workflow_contracts.py`: expected floor `5`

## Frontend Baseline

- Initial command: `cd apps/tldw-frontend && bun run test:coverage`.
- Initial blocker: `MISSING DEPENDENCY Cannot find dependency '@vitest/coverage-v8'`.
- Dependency fix: add `@vitest/coverage-v8@4.0.18` to the WebUI dev dependencies.
- After dependency install, `bun run test:coverage` started successfully but did not produce a reliable coverage percentage.
- The full frontend run surfaced broad unrelated failures across existing UI tests, then hit Node heap OOM during the large `CharactersManager` test file and hung during Vitest worker termination.
- The run was terminated after the OOM with `error: script "test:coverage" exited with code 143`.

Representative frontend blockers observed before termination:

- stale mocks missing exports such as `createWorkspaceStorage`, `useUpdateDeckMutation`, `useAttemptRemediationConversionsQuery`, `useCreateFlashcardTemplateMutation`, and `Trans`
- router-context failures in tests rendering components that now call `useNavigate` or `Link`
- snapshot and route-registry drift in media, moderation, repo2txt, watchlists, companion, and family-wizard tests
- full-suite memory pressure culminating in `FATAL ERROR: Ineffective mark-compacts near heap limit Allocation failed - JavaScript heap out of memory`

No frontend coverage floor should be set from this run. Frontend coverage needs a separate stabilization pass that either fixes the stale test harnesses or introduces a smaller, shardable coverage command before any ratchet.

## Follow-Up

- Keep frontend coverage separate from backend coverage.
- Do not use the failed frontend full-suite coverage attempt as a numeric baseline.
- Consider adding a package-level UI coverage script only after the high-churn shared UI test failures are addressed or sharded.
