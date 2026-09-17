# UAT227 — TASK-13260.167

## Change and scope

The existing `_app` transport-test helper assigned a FakeAPI builder directly to the shared endpoint module and never restored it. Later real-database tests therefore received fake ready capabilities and empty suggestion pages. The original mixed run remains retained: `.tmp/fresh-uat-recovery-20260916/uat225-226-final-green.redacted.log` (38 FAIL / 111 PASS), with its matching command receipt. This was test-process pollution, not application behavior.

Only two test files change. `_app` now installs the same fake using `pytest.MonkeyPatch.context()` within an ordinary `asynccontextmanager` FastAPI lifespan. The fake starts when TestClient enters and the exact previous builder is restored when it exits. The helper's dependency overrides and all existing assertions remain unchanged. Creating the app or reading its OpenAPI has no shared-factory side effect.

Caller inventory: fourteen local transport call sites and the imported route-order caller all enter TestClient contexts. The remaining local caller only reads OpenAPI. No helper signature or caller rewrite, product file, database setup policy, runtime, browser, config, task, or git change is included.

## Causal and regression proof

The new isolation file tests:

1. App construction and OpenAPI preserve the real factory.
2. A normal client exit restores the exact previous factory after exercising its fake route.
3. An exceptional client-body exit restores it and preserves the raised exception.
4. Nested client exit restores the outer fake first, then the real builder on outer exit; request counts prove each client used the intended fake.

Before the helper repair, all four unchanged controls fail (1.10s). After repair, all four pass (0.92s). The original helper SHA256 is `0a471f094b1f459e567fe9ecc6426e997d4563ddeb38936ae05139e9a1c0abde`; final helper SHA256 is `2bd5fb7d3d26b540cc56cb458d99381c64be2d449b9d8ccdca383ebaf747a809`.

The broader suite includes all Stage A/226 actual PostgreSQL/SQLite cases, the existing endpoint/API tests, imported route-order tests and the four new isolation tests. Collection reordering is disabled with `--randomly-dont-reorganize`. Both explicit module orders are run against independent official disposable fixture databases with `TLDW_TEST_POSTGRES_REQUIRED=1`. **Forward156 PASS/0skip97.40s; reverse156 PASS/0skip96.82s.** Final outcomes are recorded in verification-final.json.

This fixes the leaking helper itself. The existing Stage A fixture's explicit real-factory pin remains a useful declaration of its actual-boundary intent; it is not the UAT227 repair or the isolation proof. All ten frozen Stage A/UAT226 paths remain unchanged.

## Verification commands

From repository root, activate `.venv` and use the existing official runner. Exact commands are retained in:

- `.tmp/fresh-uat-recovery-20260916/uat227-forward-green-command.json`
- `.tmp/fresh-uat-recovery-20260916/uat227-reverse-green-command.json`
- `.tmp/fresh-uat-recovery-20260916/uat227-fixture-isolation-red-command.json`
- `.tmp/fresh-uat-recovery-20260916/uat227-isolation-green-command.json`

For an independent short causal-scope check:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat227-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_endpoint_fixture_isolation.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_endpoints.py tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_route_order.py --randomly-dont-reorganize -q --tb=short
```

Ruff and format checks pass both files. Both compile. Bandit reports one unchanged baseline/current B106 false positive on AuthPrincipal token_type="access" (an enum value, not a credential), zero new findings and zero parse errors. Only assertion rule B101 is excluded for test code; the baseline finding remains visible. AST review confirms the only existing function/class changed is `_app`, plus the asynccontextmanager import; all prior test assertions remain. The lifespan patch assumes the existing sequential/LIFO test-client use and is not a new concurrent application-dependency mechanism.
