# UAT257/259 authenticated content-scope repair

## Status

Implementation complete; focused causal validation is green. Adjacent authentication validation is not fully green:95 tests pass and one virtual-key setup test fails before its authenticated request. Independent review and the root-owned original native Rowan QA / Delete–Trash–Restore acceptance remain required. No native acceptance or task closure is claimed.

Backlog: TASK13260.199 / TASK13260.201. Production baseline: `2787043410`.

## Root cause and bounded correction

Configured single-user key and cookie resolution constructed authenticated request state without activating the content authorization ContextVar. Cached principal/user returns reused identity but did not restore scope. Permission-first Media DELETE and RAG stream therefore reached restricted PostgreSQL with no authenticated content owner despite successful authentication.

`User_DB_Handling.activate_authenticated_content_scope` restores user-backed scope from the canonical AuthPrincipal. It compares owner, memberships, active organization/team selectors, and admin authority. An exactly matching scope retains its database session role; absent or different authority is replaced without copying an unrelated session role. Service/anonymous principals are outside this user-backed repair. Existing claim normalization/admin decisions remain authoritative.

Call sites are the resolver's cached-principal, configured-key, and cookie paths, plus `get_request_user`'s cached-user path. JWT/API-key verification, SQL, schema, RLS, database roles, retrieval settings, providers and timeouts are unchanged.

## Owned files

- `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_authenticated_content_scope.py` (new)
- `tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py` (prior untracked controls retained; causal tests appended)
- `tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py` (two actual-cookie scope cases appended)

No other production files, Git/Backlog records, native fixtures/profiles, runtime/model configuration, original archives, or credentials were modified by this author.

## Causal RED before production edits

Official runner label `authscope257259-causal-red-03`, exit 1: **3 failed, 5 passed, zero skips**.

- PostgreSQL configured X-API-KEY: real GET returned 200, real permission-first DELETE returned 404 instead of 204.
- PostgreSQL configured Bearer key: same GET200 / DELETE404 behavior.
- PostgreSQL RAG stream: real retrieval emitted no matching source for the persisted Rowan fixture.
- SQLite lifecycle/stream controls and true no-match queries passed.

These use actual FastAPI route dependency resolution and real MediaDatabase/retrieval code. Authentication is not overridden. Fixture content is inserted within a temporary setup scope that is reset before every request. The official PostgreSQL fixture supplies a separate test database, and the existing restricted-store fixture creates a NOSUPERUSER/NOBYPASSRLS role; final tests additionally assert these role flags. The database dependency supplies that real fixture handle with its real `get_request_user` dependency. Non-media databases are unused for media-only search; external generation and vector deletion are controlled. Endpoint authentication, lookup, mutation and retrieval are not mocked.

`authscope257259-unit-red-01`, exit 1: **8 failed, 4 passed**. Cached principal/user paths retained missing/unrelated identity instead of canonical owner/membership/admin context. Matching-scope controls passed.

Earlier test-harness runs are retained: `causal-red-01` had a wrong unused-resource dependency import; `causal-red-02` had a doubled RAG router prefix. Both were corrected only in tests before production edits. Neither is claimed as the final streaming causal RED.

## GREEN and controls

- `authscope257259-causal-green-01`, exit 0: **24 passed**, zero skips. First production implementation attempt passed the causal endpoint and cache tests.
- `authscope257259-causal-final-02`, exit 0: **38 passed**, zero skips. Includes configured key/bearer GET–DELETE–restore, actual streamed source retrieval, legitimate no-match, SQLite controls, absent/stale cached authority, matching session-role preservation, child-task propagation, parent-context isolation, and ordinary-user PostgreSQL denial after unrelated elevated scope.
- `authscope257259-cookie-final-02`, exit 0: **2 passed, 32 deselected**, zero skips. Real single-user session mint and cookie validation through principal-first and standalone `get_request_user`; test bypass explicitly disabled; invalid explicit API-key header is rejected rather than falling back to a valid cookie; caller scope remains absent.
- `authscope257259-adjacent-green-01`, exit1: **95 passed, 1 failed**, zero skips. Existing principal normalization, ordinary multi-user JWT/API-key happy paths, claim/membership/admin semantics, principal/state alignment, and cookie revocation/CSRF/header cases pass. `test_api_key_scopes_org_and_team_membership` fails at line285 in `APIKeyManager.create_virtual_key`, wrapped as PostgreSQL TransactionError/DatabaseError, before its first authenticated request.
- `authscope257259-adjacent-isolated-02`, exit1: the same virtual-key setup test fails in isolation (11.22s). The author did not alter that fixture/manager or broaden production scope. Baseline comparison is still needed before calling this pre-existing; root owns Git/baseline preparation. This adjacent failure is an explicit remaining verification limit.

The initial expanded `causal-final-01` recorded two failed SQLite assertions in an added cross-owner RLS control; its 38 other tests passed. SQLite uses separate owner databases and does not provide PostgreSQL RLS on a deliberately shared raw file. That specifically shared-store denial test was bounded to PostgreSQL by parametrization (no skipped tests and no production change). Same-owner and no-match SQLite endpoint controls remain. This report does not claim cross-owner SQLite isolation from that raw-handle control.

All test runs used the required command form with escalated local fixture-network access:

```sh
source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=authscope257259-LABEL node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs TEST_PATHS -q --tb=short
```

Exact test paths/arguments, safe summaries, and redacted log hashes are retained in `.tmp/uat-repairs-231-246/authscope257259/test-receipts.json`; official runner command receipts and full logs remain in its existing evidence directory. No PostgreSQL skip, replacement cluster, native-row alteration, or provider tuning was used.

## Static/security checks

- Scoped `python -m compileall -q`: exit 0.
- Scoped Ruff: exit 1 only for pre-existing `SIM110` at resolver line174, outside the changed lines. All touched test files and new code have no Ruff findings.
- Scoped Bandit: exit 1 for **8 unchanged production findings** and test **B101 assertions only (114)**. Before/after comparison found **zero new production findings**. No suppressions or unrelated security fixes were added.
- Before/after production hashes and final test hashes: `.tmp/uat-repairs-231-246/authscope257259/{before,after}.sha256`.
- Static reports: `.tmp/uat-repairs-231-246/authscope257259/{bandit-before,bandit-final,ruff-final,static-summary}.json`.

Final focused run emitted existing Chroma destructor logging noise during interpreter shutdown after the passing pytest summary; process exit remained0. Existing dependency/deprecation warnings are retained in logs.

## Limits and handoff

The causal tests inject the official real fixture MediaDatabase handle rather than invoking the deployment's full per-request factory. They prove the dependency-order failure and the actual PostgreSQL lookup/retrieval/mutation boundary, not native browser acceptance. Earlier preseeded retrieval probes are retained as controls and are not relabeled causal evidence. The original fixture row253, analysis238, profile and archives were not touched.

Root must resolve or baseline-classify the adjacent virtual-key setup failure, finish independent review, commit/reconcile Backlog/docs as appropriate, prepare immutable native copies, and rerun original native Rowan QA and Delete/Trash/Restore before closing either finding. One production implementation attempt was needed; no implementation retry or scope expansion occurred.
