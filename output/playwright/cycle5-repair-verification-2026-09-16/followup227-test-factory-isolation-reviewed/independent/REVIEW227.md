# Independent UAT227 review — TASK13260.167

**CLEAR.** The two frozen test files match author manifest `756098217a82978de625720a330f089108759825f54dcc3872bb66fe35861098` and their snapshots. No production change or remaining actionable finding.

The existing `_app` helper now installs its fake factory only inside the FastAPI lifespan, using the ordinary pytest monkeypatch context. App construction and OpenAPI inspection do not change the endpoint module. TestClient exit restores the exact preceding factory, including exception exit and sequential nested clients. Source review confirms the original helper's only functional change is this lifetime; existing dependencies, fake responses and test assertions are preserved. The only added import is `asynccontextmanager`.

The new tests exercise a real request while each fake is installed and assert call counts as well as restoration identities. Their initial real-factory pin makes the precondition explicit; restoration is asserted before that fixture's own teardown, so the fixture cannot mask the leak being tested. The Stage A real-factory pin is separately retained and is not treated as the UAT227 repair.

## Verification

- Independent official-runner command: **58 passed, zero skipped, 2.54s**, five baseline warnings. This includes the four isolation cases, existing endpoint tests and imported route-order tests, with `--randomly-dont-reorganize`. These focused tests use transport fakes; using the required-PG runner does not turn them into real database integration tests.
- Author retained both explicit full module orders: **156 passed / zero skipped** forward and reverse, 97.40s and 96.82s. Their actual PostgreSQL/SQLite Stage A/226 coverage supplements the independent149 actual-boundary run recorded in `.tmp/uat225-226-independent-20260917/`. These author runs were inspected, not represented as independently rerun156 suites.
- Inspected causal four-test RED on the unchanged helper and the original mixed-suite38 failures/111 passes receipt. The failure mechanism is the un-restored global assignment removed by this change.
- Independent AST comparison: only existing `_app` changed; all prior test function/class ASTs match baseline. Both Python files compile without bytecode writes.
- Ruff: zero findings.
- Bandit: one identical baseline/current B106 diagnostic on the unchanged `AuthPrincipal(token_type="access")` enum value, zero new findings and zero parse errors. Only B101 assertions are excluded. An initial comparison mistakenly retained shifted snippet line numbers; the corrected comparison strips those numbers and matches the diagnostic and source snippet exactly. This comparison correction is recorded in `static-summary.json`.

Final source hashes:

- `test_suggestion_endpoints.py`: `2bd5fb7d3d26b540cc56cb458d99381c64be2d449b9d8ccdca383ebaf747a809`.
- `test_suggestion_endpoint_fixture_isolation.py`: `fa562459497c8b05cefab9c6a8e29f2b0c9d3fa9afaa8f135a9a32019b3d4e0b`.

The existing call sites use sequential/LIFO TestClient contexts; this helper does not promise independent simultaneous applications in one process. No native/browser, provider, product, runtime, task, tracker or git change was made. Only private review artifacts and the authorized test invocation were created.
