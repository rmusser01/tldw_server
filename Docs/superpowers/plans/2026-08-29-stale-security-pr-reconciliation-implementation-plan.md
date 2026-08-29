# Stale Security PR Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PRs #2610, #2614, #2622, #2623, and #2625 with one current-`dev`, test-proven security reconciliation.

**Architecture:** Keep the repository's current HTTP, RBAC, request-database, JWT, principal, and audit frameworks. Add only the missing enforcement at their existing boundaries, with five isolated boundary commits, bounded review corrections, and one replacement PR.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, Loguru, httpx, pytest, Ruff, Bandit, Backlog.md, GitHub CLI.

**Spec:** `Docs/superpowers/specs/2026-08-29-stale-security-pr-reconciliation-design.md`

## Global Constraints

- Start from `origin/dev` at `41bd5dda336c70259595ebf3ce3fb4a6a5b549db` in the isolated worktree.
- Do not merge, rebase, or copy whole stale branches.
- Write and run a failing regression test before each production behavior change.
- Do not install dependencies or modify system files.
- Keep the five security boundaries as separate commits in one replacement PR;
  keep review corrections narrow and separately reviewable.
- Close the five stale PRs only after the replacement PR merges.

---

### Task 1: Redact RAG query logs

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_convenience_endpoint_credentials.py`

**Interfaces:**
- Consumes: Loguru-compatible `logger.info` formatting.
- Produces: `_log_rag_search_request(label: str, query: object | None, *, user: str | None = None) -> None`.

- [x] **Step 1: Write failing log-redaction tests**

Extend the existing real endpoint tests for unified, simple, and advanced
search. Capture Loguru output and assert that the literal query is absent while
`query_len=18` is present.

```python
logs: list[str] = []
sink_id = logger.add(logs.append, format="{message}")
try:
    response = await rag_endpoint.advanced_search_endpoint(
        query="credential runtime",
        # existing request/user/database fixtures
    )
finally:
    logger.remove(sink_id)
assert "credential runtime" not in "".join(logs)
assert "query_len=18" in "".join(logs)
```

- [x] **Step 2: Verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py::test_standard_search_passes_ephemeral_runtime_and_closes_it tldw_Server_API/tests/RAG_NEW/integration/test_rag_convenience_endpoint_credentials.py::test_authenticated_convenience_endpoint_passes_one_real_runtime_and_closes -q
```

Expected: collection or assertion failure because the helper is absent and the
unified/advanced endpoints still log raw query text.

- [x] **Step 3: Implement the minimal helper and call it from three endpoints**

```python
def _log_rag_search_request(
    label: str,
    query: object | None,
    *,
    user: str | None = None,
) -> None:
    query_text = "" if query is None else str(query)
    if user is None:
        logger.info("{}: query_len={}", label, len(query_text))
    else:
        logger.info("{}: query_len={} user={}", label, len(query_text), user)
```

- [x] **Step 4: Verify GREEN and commit**

Run the new module, then:

```bash
git add tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_convenience_endpoint_credentials.py
git commit -m "fix(security): redact RAG query logs"
```

---

### Task 2: Route weather through controlled egress

**Files:**
- Modify: `tldw_Server_API/app/core/Integrations/weather_providers.py`
- Modify: `tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py`
- Modify: `Docs/Operations/Env_Vars.md`
- Modify: `tldw_Server_API/app/core/Integrations/README.md`

**Interfaces:**
- Consumes: `http_client.fetch`, `RetryPolicy`, central egress exceptions.
- Produces: one non-redirecting, single-attempt, sensitive synchronous provider request.

- [x] **Step 1: Write failing transport-policy tests**

Replace direct-client test doubles with a strict fake `fetch` that records its
literal keyword arguments. Add allow, deny, redirect, and secret-redaction cases.

```python
assert call["method"] == "GET"
assert call["url"] == "https://api.openweathermap.org/data/2.5/weather"
assert call["allow_redirects"] is False
assert call["sensitive_observability"] is True
assert call["retry"].attempts == 1
```

- [x] **Step 2: Verify RED**

Run the weather module and expect failures because `http_client_factory` is
still used.

- [x] **Step 3: Implement central fetch with sanitized exception mapping**

Import `fetch`, `RetryPolicy`, `EgressPolicyError`, `NetworkError`, and
`RetryExhaustedError`; remove the direct client seam; call:

```python
response = fetch(
    method="GET",
    url=self._BASE_URL,
    params=params,
    retry=RetryPolicy(attempts=1),
    timeout=self.timeout_seconds,
    allow_redirects=False,
    sensitive_observability=True,
)
```

- [x] **Step 4: Document the required allowlist**

State that enabling OpenWeather requires
`EGRESS_ALLOWLIST=api.openweathermap.org`, that redirects are disabled, and that
policy denial returns an unavailable result.

- [x] **Step 5: Verify GREEN and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py -q
git add Docs/Operations/Env_Vars.md tldw_Server_API/app/core/Integrations/README.md tldw_Server_API/app/core/Integrations/weather_providers.py tldw_Server_API/tests/Chat_NEW/unit/test_weather_providers.py
git commit -m "fix(security): enforce weather egress policy"
```

---

### Task 3: Require media.create on processing routes

**Files:**
- Create: `tldw_Server_API/app/api/v1/API_Deps/media_route_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_audios.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_code.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_documents.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_ebooks.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_emails.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_pdfs.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_videos.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py`
- Create: `tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py`

**Interfaces:**
- Consumes: `RequirePermission`, `rbac_rate_limit`, `MEDIA_CREATE`.
- Produces: `media_create_dependencies() -> list[Any]` for route decorators.

- [x] **Step 1: Write failing route and denial tests**

Use the real registered FastAPI routes to assert all target paths contain both
dependencies. Exercise one representative route with a principal lacking
`media.create` and assert HTTP 403.

- [x] **Step 2: Verify RED**

Expected: the seven routes other than video and web scraping lack the required
dependency pair.

- [x] **Step 3: Add the dependency factory and apply it**

```python
def media_create_dependencies() -> list[Any]:
    return [
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
    ]
```

Prepend `*media_create_dependencies()` without changing existing quota,
billing, or backpressure ordering.

- [x] **Step 4: Verify GREEN and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py -q
git add tldw_Server_API/app/api/v1/API_Deps/media_route_deps.py tldw_Server_API/app/api/v1/endpoints/media tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py
git commit -m "fix(security): authorize media processing routes"
```

---

### Task 4: Scope MediaWiki ingest to the request user

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py`
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py`
- Modify: `tldw_Server_API/tests/test_mediawiki_ephemeral_smoke.py`

**Interfaces:**
- Consumes: `get_media_db_for_user`, `get_request_user`, `get_media_repository`, `User.id_str`.
- Produces: optional `media_writer` and `vector_user_id` parameters threaded through the MediaWiki core.

- [x] **Step 1: Write failing request-scope tests**

Assert the ingest endpoint passes a literal request-owned writer and user ID to
the stream processor and never opens the legacy managed database. Assert vector
construction receives the literal request user.

- [x] **Step 2: Verify RED**

Expected: request-scoped parameters are absent and core falls back to singleton
identity.

- [x] **Step 3: Thread identity through the HTTP and core boundaries**

Add `media_writer: Any | None = None` and `vector_user_id: str | None = None`
only where required. At the ingest endpoint:

```python
media_writer = get_media_repository(db)
```

Use the supplied writer before opening a managed fallback and normalize a
non-empty supplied vector ID before using `SINGLE_USER_FIXED_ID`.

- [x] **Step 4: Verify fallback and ephemeral compatibility**

Run the three MediaWiki modules and retain tests proving trusted direct calls and
ephemeral processing preserve current behavior.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py tldw_Server_API/tests/test_mediawiki_ephemeral_smoke.py
git commit -m "fix(security): scope MediaWiki ingest identity"
```

- [x] **Step 6: Apply review corrections**

Namespace checkpoint state by a digest of request identity while preserving the
legacy direct-call filename. Remove the impossible `media_writer is None`
branch and its synthetic test because `get_media_repository` returns a writer
or raises.

---

### Task 5: Enforce audited admin impersonation

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/principal_model.py`
- Modify: `tldw_Server_API/app/services/admin_audit_service.py`
- Modify: `tldw_Server_API/tests/Admin/test_admin_account_audit_events.py`
- Modify: `tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py`

**Interfaces:**
- Consumes: `MandatoryAuditWriteError`, `emit_admin_account_audit_event`, JWT payload validation.
- Produces: optional token `expires_delta`, principal `impersonation` and `impersonated_by`, opt-in mandatory audit writes.

- [x] **Step 1: Write failing TTL and mandatory-audit tests**

Assert the endpoint passes `timedelta(minutes=15)`, persists one literal audit
payload, and returns sanitized 503 when the audit service raises
`MandatoryAuditWriteError`.

- [x] **Step 2: Write failing strict-claim and propagation tests**

Cover true + integer success, true + missing/malformed failure, orphan actor
failure, non-boolean flag failure, and propagation into `AuthPrincipal`.

- [x] **Step 3: Verify RED**

Run the four impersonation-related modules. Expected failures: no expiration
override, no mandatory audit, and no principal fields/strict pairing.

- [x] **Step 4: Implement minimal JWT and principal support**

Add `expires_delta: timedelta | None = None`, compute `issued_at` once, and use
the override only when supplied. Add defaulted principal fields and preserve
them through `_mapping_from_user_like` and the legacy adapter.

- [x] **Step 5: Implement strict claim extraction**

Reject inconsistent or malformed pairs with the existing credentials exception;
do not coerce strings through `bool()`.

- [x] **Step 6: Implement opt-in mandatory audit persistence**

Thread `raise_on_failure: bool = False` through the existing admin audit service.
Existing callers remain best effort. The impersonation endpoint opts in, emits
the literal action `admin.impersonation.token.create`, and maps
`MandatoryAuditWriteError` to sanitized 503.

- [x] **Step 7: Verify GREEN and commit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Admin/test_admin_account_audit_events.py tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py -q
git add tldw_Server_API/app/api/v1/API_Deps/auth_deps.py tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py tldw_Server_API/app/core/AuthNZ/jwt_service.py tldw_Server_API/app/core/AuthNZ/principal_model.py tldw_Server_API/app/services/admin_audit_service.py tldw_Server_API/tests/Admin/test_admin_account_audit_events.py tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py
git commit -m "fix(security): audit admin impersonation"
```

- [x] **Step 8: Apply issuer-identity review correction**

Before repository, token, or audit work, reject already-impersonated principals
and principals without an exact integer user actor. Reuse the validated actor
for the token claim, audit record, response, and log.

---

### Task 6: Integrated verification and replacement delivery

**Files:**
- Modify: `backlog/tasks/task-13013.4 - Resolve-the-stale-release-blocking-security-pull-requests.md` through Backlog CLI only.
- Modify: this plan's checkboxes as work completes.

**Interfaces:**
- Consumes: the five green security commits.
- Produces: focused evidence, replacement PR, and stale-PR supersession records.

- [x] **Step 1: Run the combined focused suite**

Run every touched test module in one pytest process and require zero failures.

- [x] **Step 2: Run static security checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m compileall -q tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/app/core/Integrations/weather_providers.py tldw_Server_API/app/api/v1/API_Deps/media_route_deps.py tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/ruff check <touched-python-paths>
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/bandit -q <touched-production-python-paths>
git diff --check origin/dev...HEAD
```

- [x] **Step 3: Perform a focused security self-review**

Review raw diffs for secret/PII logging, fail-open behavior, missing route coverage,
identity fallback at HTTP boundaries, audit error handling, and compatibility.
Record Critical/High/Medium/Low findings and fix any Critical or High via a new
RED-GREEN cycle.

The independent review found no Critical issue. Three Important issues were
closed by checkpoint identity isolation and fail-closed impersonation issuer
validation. Its one Minor issue was removed as dead code. The final combined
suite passed 127 tests; touched production Bandit reported zero Medium/High
findings; compile and range whitespace checks passed. Ruff reported only three
unchanged baseline lines, while every other touched Python file passed.

- [x] **Step 4: Update TASK-13013.4 and commit documentation**

Record exact commands/results, touched files, known skips, and the replacement
strategy using `backlog task edit`. Commit the spec, plan, and task file.

- [ ] **Step 5: Push and open one replacement PR**

The PR body maps each stale PR to its replacement commit and states that the
stale PR will be closed only after merge. Do not merge without a separate user
decision.

- [ ] **Step 6: Record supersession on stale PRs**

Comment readably on #2610, #2614, #2622, #2623, and #2625 with the replacement
PR URL. After replacement merge, close each unmerged and finish TASK-13013.4.
