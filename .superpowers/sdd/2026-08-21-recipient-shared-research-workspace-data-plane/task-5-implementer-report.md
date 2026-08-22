# Task 5 Implementer Report

## Status

DONE. Task 5 replaces the recipient read plane with strict bounded envelopes and keeps chat generation fail closed for Task 7.

Starting commit: `52e95b5bb4bae2efefea1acb5b5147e759cfd776`

Task commit: `feat(sharing): add bounded recipient workspace reads` (this report is included in that commit)

## Implementation

- Added recipient-only strict Pydantic request/response models with explicit identifier, text, timestamp, collection, score, and pagination bounds.
- Added `SharedWorkspaceRecipientRoute` only to the `/shared-with-me/{share_id}` subrouter. It maps dependency 401 and request validation failures to exact typed detail objects without global handlers.
- Extended `require_permissions` and `rbac_rate_limit` only through optional detail arguments. Existing defaults and `_tldw_rate_limit_resource` metadata are unchanged.
- Required `sharing.read` and the existing `sharing.read` rate resource on every recipient route, with distinct typed read/chat rate errors.
- Resolved `SharedWorkspaceAccessService` before opening owner media/ChaCha or recipient history storage.
- Replaced bootstrap and raw source reads with bounded bootstrap, source page, source-ID preview, and read-only recipient history envelopes.
- Added deterministic source order, query-before-projection, derived-state-before-pagination, 200-row page bounds, 50-source bootstrap, 30-message bootstrap, and eight-error caps.
- Projected only safe source fields. HTTP/HTTPS values become normalized origins; credentials, unsupported schemes, and non-reconstructable origins return at most a bounded host.
- Built allowed actions from a fresh policy copy plus retrieval and fail-closed generation readiness. Empty readable workspaces retain source inspection.
- Replaced claim-loop discovery with `list_active_shares_for_user` before owner-name enrichment while preserving the existing list response.
- Added a strict interim POST chat request solely for typed malformed-body mapping. Valid requests authorize first and then return typed 503; the route performs no retrieval, generation, persistence, or thread creation. The removed `{"query": ...}` contract is rejected with typed 422.
- Kept clone and all owner/token/admin routes separate and unchanged.

## TDD RED

Initial required RED:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py -q
```

Result: collection failed because `shared_workspace_recipient_schemas` did not exist.

Self-review URL fallback RED:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py::test_source_origin_sanitization_never_exposes_paths_or_credentials -q --timeout=30 -o log_cli=false
```

Result: `2 failed, 5 passed`; parseable FTP and scheme-relative URLs did not yet return the required bounded host-only fallback.

## GREEN Verification

Bounded exact Task 5 path matrix:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py tldw_Server_API/tests/Sharing/test_sharing_endpoints.py tldw_Server_API/tests/Sharing/test_cross_user_access.py -q -n 4 --dist=loadfile --timeout=45 -o log_cli=false
```

Result: `93 passed, 25 warnings in 53.94s`.

Final recipient and malformed-chat suite:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py tldw_Server_API/tests/Sharing/test_shared_workspace_chat_security.py -q --timeout=45 -o log_cli=false
```

Result: `31 passed, 4 warnings in 7.81s`.

Focused authentication, dependency metadata, route isolation, OpenAPI, malformed-body, and old-contract absence target:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_chat_security.py tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py -k 'openapi or optional or isolated or authentication or validation or malformed or old or legacy or unsafe' -q --timeout=45 -o log_cli=false
```

Result: `10 passed, 19 deselected, 4 warnings in 7.14s`.

URL fallback GREEN: `7 passed, 4 warnings in 6.97s`.

Ruff over all touched Python files: passed.

Bandit:

```text
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/API_Deps/auth_deps.py tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/schemas/sharing_schemas.py tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py tldw_Server_API/app/api/v1/utils/shared_workspace_recipient_route.py -f json -o /tmp/bandit_task_12020_40_task5.json
```

Result: zero findings, zero skipped tests, 4,505 production LOC scanned.

`git diff --check`: passed.

## Bounded-Run Diagnostic

The first serial exact-matrix attempt was interrupted after a bounded idle interval. A verbose rerun reached 14 passing tests before interruption. Cleanup logged one pending default-character executor future in the repository-wide autouse `_restore_user_db_env_and_chacha_cache` / `_reset_workflow_scheduler` path. The next test in deterministic order, `TestAdmin::test_admin_list_shares`, passed alone in 10.81 seconds under a 30-second timeout; it and its predecessor passed together in 7.46 seconds under a 20-second timeout. The bounded four-worker matrix later logged and drained the same single cleanup future and exited normally with all 93 tests passing. No Task 5 pytest process remained open after verification. Unrelated UI pytest processes from other workspaces were not touched.

## PostgreSQL State

Task 5 changed no PostgreSQL schema, policy, migration, fixture, or production query. PostgreSQL was not started or touched. The sharing fixture change is in-memory SQLite setup for authoritative organization/team membership only.

## Files

Production:

- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- `tldw_Server_API/app/api/v1/schemas/sharing_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py`
- `tldw_Server_API/app/api/v1/utils/shared_workspace_recipient_route.py`

Tests:

- `tldw_Server_API/tests/Sharing/conftest.py`
- `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_security.py`
- `tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py`
- `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`

Tracking/reporting:

- `backlog/tasks/task-12020.40 - Bind-recipient-shared-workspace-sources-and-chat-to-the-canonical-share.md`
- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/task-5-implementer-report.md`

## Self-Review

- Every recipient response is a dedicated extra-forbid model; no owner schema inheritance or raw list remains.
- Exact typed 401, 403, 422, 429, 404, and 503 bodies are covered. Recipient route mapping is isolated from clone/owner/token/admin routes.
- Access resolution precedes every owner media/ChaCha and recipient history open.
- No recipient model exposes media IDs, owner IDs, internal share scope, filesystem/database paths, credentials, endpoints, raw exceptions, prompts, queries, or internal excerpts.
- Owner display, workspace/source text, source type, host, URL, identifier, preview, message, citation, score, timestamp, page, bootstrap, and partial-error bounds are enforced.
- URL output contains no credentials, path, query, or fragment. Unsafe reconstruction emits at most a bounded host.
- Bootstrap returns at most 50 sources, latest 30 messages, and eight safe partial errors. History reads do not create a thread and empty history remains empty.
- Source search happens before status projection; deterministic order and derived state happen before bounded pagination.
- Preview resolves canonical source-ID membership before opening owner media and emits only recipient fields.
- Allowed actions are copied before overlay; source inspection remains allowed for an empty readable set, and asking requires both a retrieval-ready source and generation target.
- `generation_default` is bounded and unavailable by default without credential/provider diagnostics.
- Shared-with-me discovery is authoritative; clone behavior remains separate.
- The two unrelated untracked watchlist templates were not touched or staged.

## Concerns

No Task 5 implementation blocker. Repository-wide ChaCha default-character cleanup can add intermittent serial-suite latency; bounded verification passed and this task did not change that lifecycle code. Task 7 must replace the interim chat 503 with canonical scoped retrieval and generation before chat is usable.

## Fix Round 1

### Status

DONE. Review head `6be3d619b8f8818d61946161d296606ff86f3f1a`; all four accepted findings are fixed without restoring an unsafe recipient chat path.

### Implementation

- Source `q` matching now uses only canonical source ID, bounded projected title/type, and sanitized origin URL/host. Credentials, paths, queries, fragments, file URLs, unsupported schemes, and other raw URL text cannot affect membership or totals.
- Recipient preview projection now allocates one aggregate `max_chars` budget across focused chunk text, `text_preview`, and remaining chunks. Focus is prioritized, the duplicate content-excerpt projection is omitted, exact duplicate texts are suppressed, and `text_truncated` reflects omitted source text.
- History maps only the canonical store's `InputError` cursor rejection to exact `422 invalid_shared_workspace_request`; operational exceptions remain typed 503. The test executes the real `SharedWorkspaceChatStore` decoder before any DB transaction.
- Added strict `SharedWorkspaceErrorResponse` and declared typed 401/403/404/422/429/503 responses on every recipient operation. The interim POST chat retains `SharedWorkspaceChatRequest`, advertises no 200, is documented as 503, authorizes first, and remains generation-free.

### TDD RED

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py -k 'source_search_ignores_hidden_raw_url_content or source_search_matches_sanitized_origin' -q --timeout=45 -o log_cli=false
```

Result: `1 failed, 1 passed, 30 deselected`; hidden raw URL content changed source membership.

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py -k 'preview_text_uses_one_aggregate_budget_with_focus_first' -q --timeout=45 -o log_cli=false
```

Result: `2 failed, 32 deselected`; no aggregate recipient preview allocator existed.

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py -k 'history_rejects_cursor_from_canonical_store_decoder or history_store_failure_remains_unavailable' -q --timeout=45 -o log_cli=false
```

Result: `1 failed, 1 passed, 34 deselected`; canonical malformed cursor returned 503 instead of 422.

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py -k 'recipient_openapi_declares_only_typed_route_scoped_errors' -q --timeout=45 -o log_cli=false
```

Result: `1 failed, 36 deselected`; recipient operations did not declare the strict typed error wrapper and chat still advertised 200.

### GREEN Verification

- Exact Task 5 matrix from the brief, serial and bounded: `102 passed, 5 warnings in 266.86s`.
- Focused auth, dependency metadata, route isolation, OpenAPI, malformed body, and old-route absence target: `11 passed, 27 deselected, 4 warnings in 6.80s`.
- Final all-new-fix target: `7 passed, 30 deselected, 4 warnings in 9.16s`.
- Ruff on both touched production files and the recipient endpoint test: passed.
- Bandit on touched production: zero findings, zero skipped tests, 2,022 LOC scanned; JSON at `/tmp/bandit_task_12020_40_task5_fix_round_1.json`.
- `git diff --check`: passed.
- No pytest process was left open. The exact matrix continuously progressed and exited normally.

### Files

- `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- `tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py`
- `tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py`
- This report, the SDD progress ledger, and existing `TASK-12020.40` notes.

### Self-Review

- Search input cannot observe non-projected URL material; positive matching still works on the normalized safe origin.
- Preview text fields share one 1..12,000-character budget; focused text consumes that budget first and does not enlarge it.
- Access resolution still occurs before history storage or owner media access. No cursor codec was copied into the API layer.
- Every recipient operation documents the same strict bounded detail envelope; route-scoped runtime mapping remains isolated from clone, owner, token, and admin routes.
- No media/owner IDs, internal share scope, paths, secrets, raw errors, credentials, prompts, queries, or provider diagnostics were added.
- The two unrelated watchlist templates remain unmodified and unstaged.

### PostgreSQL State And Concerns

No PostgreSQL schema, policy, fixture, query, or runtime state was touched. No Fix Round 1 blocker remains. The known repository-wide default-character executor cleanup can make the serial sharing matrix slow, but it continued making progress and exited successfully. Task 7 still owns replacing the typed interim 503 chat route with canonical safe generation.
