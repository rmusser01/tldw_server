# UAT181 / TASK13260.118 — real HTTP lifetime RED

**Frozen: 9 expected failures,14 passing controls,zero skips (227.66s).** Only one new permanent test file and private design/evidence were written. Production remains unchanged; no candidate implementation exists.

## Causal result

Empty and populated real HTTP due-review→Buddy→Notes chains both return the expected endpoint data. Due/Buddy run on the same AnyIO worker, while actual Notes runs on the event loop. The worker connection remains INTRANS with source_review_occurrences, source_review_plans, buddy_attachments and buddy_profiles relation locks; the Notes connection is IDLE without locks. Both originals remain open while a real second CharactersRAGDB constructor fails ensuring PostgreSQL source-review storage. No endpoint/DB response or DDL is mocked.

The passing Notes-only HTTP control distinguishes this from the repaired Notes queries. The due-only response still retains its transaction. Sync handoff, joined child work, route failure and cancellation each retain the operation's read transaction. An undecided successful write remains uncommitted, as required, but its ended HTTP operation still holds the transaction. Two concurrent HTTP requests for the same user can observe the first request's uncommitted title; the first owner's write remains intact until its explicit rollback. This is same-user request isolation evidence, not a cross-account finding.

## Preserved controls

All10 implicit/raw-BEGIN/ChaCha/nested/external-backend commit/rollback decision cases pass through a real HTTP dependency. They invoke the real nested non-HTTP owner accessor, retain the pending value internally, keep it invisible to an independent backend reader, and persist or roll back only after the explicit decision. The no-auto-success-commit control passes. Ordinary Notes, Buddy list and Buddy attachment isolated reads finish cleanly.

The external-backend controls concern caller-managed transaction contexts inside a request. An explicit pre-existing external-session binding and detached-child lifetime are part of the proposed prototype design and still require tests against the selected new API; they are not claimed as already covered by these23 tests.

## Actual boundary and fixture isolation

The test mounts real routers and keeps actual get_chacha_db_for_user, runtime admission, cache health/default-character work, endpoint signatures, DB methods, backend execution and replacement schema initialization. An official disposable pg_database_config database is placed in an isolated cache; identity and rate limiting are fixture inputs. No AuthNZ test_db_pool, manual database creation, provider request, native browser or running API is used.

HTTP goes through httpx ASGITransport with a deterministic AnyIO worker limit. Observers record only endpoint path, OS thread and raw connection identity; they preserve each original handler's sync/async nature and dependency tree. Both sync-worker and async-event-loop connections stay alive through the replacement check.

The first two-case run had one instrumentation failure: health-check work inherited the request marker and polluted the endpoint thread set. That failed before replacement and is **not** causal RED. After narrowing observation to actual endpoint bodies, the same two cases produce one real replacement failure and one passing Notes control. The subsequent unchanged full23-case suite yields the nine expected failures and14 passing controls above.

## Reproduce

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-http-lifetime-review node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_http_operation_lifecycle.py -q --tb=short
```

The runner requires PostgreSQL and uses the existing owned cluster through official fixtures. Redacted logs and command receipts are copied alongside this report. `red-result.json` preserves failed nodes and the actual replacement-state diagnostics. `red-source/` contains the unchanged production baseline; `red-test.py` and `review-snapshot/` exactly match the permanent test. All three baseline production hashes still matched after the run, before the separately owned UAT192 changes.

## Design handoff and static checks

`DESIGN181-http-ownership.md` proposes a four-file HTTP/maintenance owner-boundary prototype and identifies why a DB-dependency finalizer alone misses direct accessor routes. It distinguishes full ASGI lifetime from early BaseHTTPMiddleware call_next completion, preserves existing migration semantics, and flags health-check transaction adoption as a must-test risk. `nonhttp-accessor-audit.json` retains30 loaded references across20 files, including worker/callback/WebSocket entry points and source hashes.

Ruff:0 findings. Formatter check and Python compilation pass. New-test Bandit:0 findings/errors with only pytest assertion ruleB101 excluded. Owned whitespace check passes. No claim of GREEN, native acceptance or application-wide lifetime repair. Parent owns review, any production release, runtime and integration.
