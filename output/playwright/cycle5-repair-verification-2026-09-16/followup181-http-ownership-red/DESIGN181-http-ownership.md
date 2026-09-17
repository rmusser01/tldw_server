# UAT181 / TASK13260.118 — owned HTTP operations, proposal only

Production remains unchanged. This is the reviewable prototype design following the dependency-wired causal RED; it is not an implemented session framework.

## Proven boundary

The first valid real HTTP regression produces the native tuple: due-review and Buddy execute on the same AnyIO worker and leave source_review_occurrences/source_review_plans/buddy_profiles/buddy_attachments locked in INTRANS; async Notes executes on the event loop and finishes IDLE. The real replacement constructor fails ensuring source-review storage while both original connections are alive. The Notes-only real HTTP control passes. The test retains actual dependency/cache/runtime/health/default-character work and only isolates fixture cache, identity and rate-limit inputs.

An initial recorder also counted the cache health-check thread. Its thread assertion failed before replacement. That instrumentation failure is retained separately; body-only observation corrected the recorder without changing endpoint/dependency behavior.

## Smallest complete HTTP scope proposed for approval

Use one lazy operation-owner context across the full ASGI HTTP lifetime. It contains per-CharactersRAGDB state (connection, pinned backend and transaction depth), acquired only on DB use. Cache initialized DB/schema/backend configuration as today. The operation must return its own captured checkouts after the downstream ASGI application finishes, including streaming/background response work, on success, failure or cancellation. It must not commit based on HTTP success.

A yielding DB dependency alone is incomplete: direct owner accessors exist in Chat sharing, shared-workspace helpers, RAG resume, Slides, VN debug and Flashcards job-result paths. They do not all pass through Depends(get_chacha_db_for_user). A single HTTP owner context also reaches these existing direct lookups without changing each endpoint. Do not attach lifetime cleanup to generic auth dependencies.

Proposed production files, pending review:

1. New `tldw_Server_API/app/core/DB_Management/chacha/operation_scope.py`: small explicit owner/state object, context manager and pure ASGI wrapper. No SQL parsing, per-domain registry, scheduler or configuration framework. Use existing pool return semantics. Capture owner/state objects for cleanup rather than querying the teardown thread's threading.local.
2. `ChaChaNotes_DB.py`: resolve the existing connection/pinned-backend/depth state through the active operation for PostgreSQL; leave legacy out-of-operation and SQLite behavior intact. Inspected touchpoints are BackendManagedTransaction entry/exit, _get_pinned_backend, _get_thread_connection, close_connection, and execute_query's existing depth guard. The existing state accesses are concentrated there. No SQL/default/read flags or migration method changes.
3. `ChaCha_Notes_DB_Deps.py`: establish independent owned maintenance scopes for initialization/publication, cache health checks and default-character executor work. **Health is material:** it currently runs db.transaction() in asyncio.to_thread. Blindly propagating a request lease into that helper could commit a caller's pending write. Its maintenance scope must explicitly borrow a separate checkout and never adopt the request transaction. Preserve the cached-DB async accessor signatures and existing initialization/admission error behavior.
4. `main.py`: register the pure ASGI owner wrapper exactly once around HTTP work in normal/minimal app wiring. The wrapper does nothing for requests that never touch ChaCha. It must wrap the complete ASGI response, not BaseHTTPMiddleware's early call_next return.

This is a proposed four-file plumbing unit; implementation approval is still required. No change to PostgreSQLBackend defaults/autocommit, request authentication, global pool cleanup, schema checks, RLS or migration/trigger rebuilding.

## Ownership contract and prototype gates

- Each new HTTP request creates an independent owner even if it runs on a reused worker or inherits another task's context. The same request's joined task and sync handoff resolve the same operation state when intended.
- Nested service calls and get_chacha_db_for_user_id within an active operation reuse that owner. Explicit/raw-BEGIN/nested/backend transaction controls retain their write and decide commit/rollback themselves. The finalizer never commits leftover work.
- A caller-supplied external connection must remain borrowed. The prototype needs an explicit external-owner binding when entering an operation around a pre-existing connection; INTRANS or a matching thread is not sufficient evidence of ownership. Never adopt a pre-existing legacy checkout implicitly. Independent HTTP cleanup must not close/settle another owner's active connection.
- Allocation/state updates must be safe across sync handoff. Do not hold a threading lock across await. Avoid hiding overlapping independently owned transaction scopes on one physical connection; shared joined work versus independent child operation must be explicit.
- A detached task cannot keep using a closed inherited owner. It must open a managed operation of its own or fail clearly. A joined child remains part of its parent until completion. Default-character and health work are independent maintenance, even when launched during a request.
- Record/check owner closed state so cancellation or repeated finalization cannot return a checkout twice. Use the existing backend cleanup behavior, with exception precedence preserved. No rollback of another owner and no global connection sweep.
- HTTP success with an undecided write rolls back only that operation's leftover transaction; existing explicitly committed writes remain durable. No automatic commit on successful response.
- Bootstrap/migrations retain their separate existing backend transaction. Entering an HTTP operation must not reconstruct CharactersRAGDB or trigger DDL on every request. The already-current DDL optimization remains a separate option, not part of this ownership patch.

## Test status and necessary integration follow-up

New permanent `test_chacha_postgres_http_operation_lifecycle.py` contains real dependency-wired due/Buddy/Notes replacement cases, isolated endpoint reads, joined child/sync handoff/failure/cancellation, pending-write explicit decisions, no-success-autocommit, owned teardown and concurrent request isolation. Its instrumentation preserves sync/async call kinds and the original dependency tree. All PostgreSQL work uses pg_database_config and the existing required-PG runner.

The minimal test app currently reproduces the real dependency boundary without full main-app middleware. If the ASGI option is approved, GREEN must install the actual production owner middleware in this fixture and separately prove normal/minimal main registration. Preserve the current RED source/test snapshot. Do not describe dependency-only GREEN as proof of production middleware installation. Additional detached-child/external-binding/stream completion and tenant-scope controls must be added against the selected actual owner API before production is considered review-ready.

## Non-HTTP audit and explicit limit

The accessor audit finds **30 loaded references in 20 files**, including callback and audio-shim references. This is an audit of the canonical owner accessors, not all standalone CharactersRAGDB constructors or dynamic plugin calls. Exact functions/lines/source hashes are retained in `nonhttp-accessor-audit.json`.

| Surface | Existing lifetime evidence | Proposed adoption |
| --- | --- | --- |
| Nine app/services files: study_pack_jobs_worker, study_suggestions_jobs_worker, visual_identity_jobs_worker, vn_asset_jobs_worker, workspace_file_inventory_jobs_worker, research_workspace_output_jobs_worker, writing_annotation_review_jobs_worker, notes_graph_suggestions_worker, notes_graph_suggestions_maintenance | They have explicit close/release helpers/finally paths. Some work crosses asyncio.to_thread; closing the event-loop thread's cached connection does not prove worker checkout cleanup. | Wrap the actual job/pass owner in the new operation context, preserving injected externally supplied DB ownership. Existing cleanup can delegate to that owner. Do not globally wrap every Jobs SDK handler as an implicit DB transaction. |
| core/Chat_Macros/jobs.py: run and cancellation paths | Finally invokes _close_worker_database; repository work crosses asyncio.to_thread. | One managed operation per acquired run/cancellation action; borrowed repository keeps its external owner. |
| core/Sharing/shared_workspace_clone_jobs_worker.py | Owner accessor is passed to access service and load_chacha_db callback; explicit Media session factory already exists. | Bind the actual clone operation and captured multiple-owner DB handles, not only the factory callback. Verify cancellation and lease/job completion cleanup. |
| Voice assistant and audio streaming WebSocket accessors | DB is acquired during session initialization; audio uses a deferred shim, so a call-only grep misses it. | Per command/turn managed operation. A whole WebSocket lifetime can last hours and is not a read-transaction cleanup boundary. Preserve intentional multi-message state as explicit service state, not an implicit DB transaction. |
| core/Chat/command_router.py skills runtime | Getter can be reached from more than one transport. | Reuse an existing operation when invoked by HTTP; otherwise caller owns a command operation. |
| HTTP-only direct lookups and callbacks in sharing, flashcards, rag_unified, chat, slides, vn_play | They can bypass the DB dependency. Some callbacks may be reused outside HTTP. | Covered by the ASGI operation when executed in HTTP; callbacks used in jobs still need the job scope above. |

Recommend reviewing the four-file HTTP/maintenance prototype first and keeping non-HTTP adoption as explicitly bounded follow-on work. Leaving legacy compatibility outside an operation is deliberate and must not be reported as a whole-application lifecycle guarantee. No workers, WebSockets, routes or middleware have been edited in this stage.

## Frozen RED outcome

Final required-PG run: 9 expected failures,14 passing controls,0 skips in227.66s. Both actual replacement failures reproduce the source-review lock chain. All10 explicit-decision controls and the no-auto-commit control pass. The cross-request failure concerns two requests for the same user; no cross-account leak is claimed. See RED181-http-lifetime.md and red-result.json.
