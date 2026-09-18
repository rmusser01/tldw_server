# UAT257/259 independent code and boundary review

## Verdict

**APPROVE the bounded production correction.** No authority bypass, RLS weakening, endpoint-specific workaround, or broken authentication mock was found. The outstanding broad-auth-suite setup failure is correctly retained as unresolved and is not evidence that the source fix is complete outside its focused scope.

## Reviewed boundary

The two production edits restore the existing `ScopeContext` at the two canonical reuse boundaries that bypass the normal user-first authentication path:

- `get_auth_principal` activates scope when it returns a cached `AuthContext`, and when its configured single-user key or cookie-session paths construct that context.
- `get_request_user` activates scope before returning an already cached principal/user pair.

`activate_authenticated_content_scope` derives authority only from the canonical `AuthPrincipal`: user id, organization and team memberships, active selectors, and admin state. A full authority match preserves the existing database session role. Any absent or changed authority calls `set_scope` without a role, so a role from another owner or selector set cannot survive. Service and anonymous principals are intentionally excluded.

The unchanged normal JWT and ordinary API-key authenticators already establish scope before they cache their auth context. This correction therefore does not add a second authentication scheme or alter those flows.

## Test boundary review

The new causal endpoint tests use actual FastAPI dependency resolution and the official restricted PostgreSQL fixture. They do not override authentication or pre-seed scope for the request under test:

- configured `X-API-KEY` and Bearer-key requests perform Media GET, permission-first DELETE, and restore against a real restricted media database;
- the RAG stream uses the real request auth and actual retrieval path, with only generation/vector work controlled, and checks owned context plus a real no-match case;
- the restricted-store fixture creates a `NOSUPERUSER NOBYPASSRLS` PostgreSQL role and asserts both flags;
- cache tests cover missing and unrelated context, memberships, selectors, admin state, matching-session-role retention, child-task propagation, and caller-context restoration;
- the added cookie test mints and validates a real session in both dependency orders and proves an explicitly invalid header does not fall back to the cookie.

The direct-context RAG tests remain controls. They are distinct from the appended permission-first regressions and do not stand in for them.

## Independent validation

Executed with the required virtual environment and official PostgreSQL runner using escalated local fixture access:

```sh
source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=authscope257259-independent-review node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/RAG/test_restricted_postgres_media_retrieval.py tldw_Server_API/tests/AuthNZ_Unit/test_authenticated_content_scope.py tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py -q --tb=short
```

Result: exit 0, **72 passed**, zero skips. The runner emitted a known Chroma destructor message during interpreter shutdown after the passing pytest result; it did not change the exit status.

The author report also retains causal RED (3 failed / 5 controls passed under restricted PostgreSQL), focused GREEN (38 passed), and focused real-cookie GREEN (2 passed, 32 deselected). Those are consistent with the independent run.

## Outstanding adjacent result

The author’s broader adjacent authentication run finished **95 passed, 1 failed, zero skips**. Its isolated repeat fails in `APIKeyManager.create_virtual_key` before an authenticated request. It has not been classified as pre-existing; root is comparing the baseline with the same official runner. This is not a defect introduced by the reviewed production diff, but it must remain visible for integration/closure.

## Reviewed frozen hashes

```text
3eee904c6d103f0cfa6931f3167a8028b40e2eed90a2eb70cfb79c937732cfe0  auth_principal_resolver.py
efb8497693942be43a61fefbeb0ece1b38d9f57059e3c9f5c3a90009a502aacb  User_DB_Handling.py
817a9a2559d5afa861cb5afd582b9cdf5a374ed21de1eb1f3c4cf085af30f53e  test_authenticated_content_scope.py
004e14ba8ecc1a09c40edaa87b3f0f8d5ed93362df285e23cc141782fee72bd7  test_restricted_postgres_media_retrieval.py
e731880d63031672483f8c637362cc12a88ecb28651538b1f95755a73a510415  test_single_user_cookie_session.py
```

Author report: `.superpowers/sdd/IMPLEMENTATION_PLAN_fresh_matrix_231_246_repairs/task-257-259-report.md`.
