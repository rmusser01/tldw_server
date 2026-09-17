# UAT238: PostgreSQL ingest RLS failure — read-only diagnosis

Associated repair: parent-created TASK13260.180. Frozen matrix revision `8f8774e6c868b304a96d95ab82e28389c129a78b`, cell `pg-single`.

## Finding

**Native RLS denial is confirmed. The frozen source strongly identifies a missing content authorization scope at the worker/executor boundary, despite a correct job and row owner.** The job carries owner `1`; the worker constructs MediaDatabase with client `1`; the repository derives numeric `owner_user_id=1` and `visibility='personal'`. Those row values do not set PostgreSQL's separate `app.current_user_id` session setting. The executor persistence path does not install or carry the content scope that supplies that setting.

The policy is acting as an authorization guard. Do not disable/relax RLS, elevate the runtime role, mark the operation as admin, or exempt single-user mode. This is distinct from UAT204's incorrectly prefixed StudyPack client identity.

Confidence: high for the source-supported cause; exact native `current_setting` values and bound INSERT parameters were not captured. No tests, inference, database connection/query, browser, runtime, private credentials/config, or raw logs were used in this diagnosis. Only this report is written. The supplied redacted log excerpts are evidence inputs.

## Native evidence window

- Startup receipt **14:12:35.226 UTC** identifies this frozen revision, PostgreSQL single-user mode, completed initializer exit 0, and a direct-login runtime role with **superuser=false, bypassrls=false, inherit=false, createdb=false, createrole=false, replication=false, memberships=0**. No credential or role name is needed in this report.
- `POST /api/v1/media/ingest/jobs` returns **200 at 14:19:00.444**, creating job **1**, UUID **fca92acb-398f-41c8-8379-7376cc655615**. Job GET at **14:19:00.453** reports queued and `owner_user_id:'1'`; **14:19:01.681** reports processing/20%.
- Safe backend excerpt at **07:19:51.254 local / 14:19:51.254 UTC** shows construction under user-database path `/1/Media_DB_v2.db` with **Client ID: 1**. At **14:19:51.334**, `add_media_with_keywords` logs client **1**.
- PostgreSQL's supplied server excerpt at **14:19:51.337 UTC**, PID **95728**, states **“new row violates row-level security policy for table "media"”**, on the INSERT with `owner_user_id` and `client_id` columns. The excerpt contains placeholders, not their parameter values.
- Job GET at **14:19:51.373** returns HTTP 200 with outer **completed/100%**, nested result **Warning**, `media_id:null`, `media_uuid:null`, and a DB-error string. Thus “failed” describes persistence and the UI result, not the outer Jobs status. `ingest-observation.txt` at **14:20:09.450** shows 0 succeeded/1 failed; the catalogue snapshot at **14:21:33.815** shows zero results. This audit does not infer a direct authoritative row count from that UI.
- The nested result also contains provider-analysis warnings. They are independent observations; resolving RLS does not establish successful model analysis. The plaintext processing reaches persistence with Warning status, as designed.

## Exact ownership/session chain

| Boundary | Frozen behavior | Implication |
|---|---|---|
| API admission | `endpoints/media/ingest_jobs.py:397–419` creates the job with `owner_user_id=str(current_user.id)` | Persisted trusted job owner is available. |
| Jobs dispatch | `worker_sdk.py:952–955` awaits handler(job); Media worker handler at 708–709 forwards to `_handle_job` | No content `set_scope/scoped_context` is installed in either inspected dispatch path. Lease ownership is separate from content authorization. |
| Worker identity | `media_ingest_jobs_worker.py:296–304` resolves job owner before payload fallback; `_create_db` at 350–353 passes that identity as client ID | Native client `1` is correct; no StudyPack-style prefixed identity here. |
| Document pipeline | Worker 530–532 extracts db path/client; 586–598 passes numeric user ID plus client to processing | An integer processing parameter does not itself set a ContextVar. |
| Persistence handoff | `persistence.py:5554–5566` forwards db path/client into `persist_doc_item_and_children`; 5788–5848 defines `_db_worker` and runs it with `loop.run_in_executor(None, _db_worker)` | The synchronous thread callable creates a new DB session; no explicit scope binding or `copy_context().run` is present. Standard `run_in_executor` does not copy caller ContextVars. An async-only worker scope wrapper would therefore be insufficient. |
| New DB session | `_with_media_db_session`, `persistence.py:133–146`, creates the DB, invokes the operation, closes it in finally; API/runtime factories pass the client to the class | Client identity is data ownership/provenance, not authorization scope installation. |
| Row owner | `media_repository.py:100–107` derives `int(client_id)`; 690–708 chooses personal visibility and derived owner when none explicitly supplied; 709–739 builds INSERT | With the observed client `1` and these arguments, row owner is `1`, personal, not deleted. The bound values are inferred from source, not directly read from the server. |
| PostgreSQL scope | Pool checkout calls `_apply_scope_settings` (postgresql_backend.py:178–191, 225–232); Media transaction obtains that pool connection (media_database_impl.py:1930–1939) | Authorization is set when the persistence thread obtains its own connection. |
| Missing scope defaults | `scope_context.py:101` defines a ContextVar default None; backend 519–574 reads `get_scope`, defaults user/org/team to empty and is_admin to 0, then applies `app.current_user_id` and other settings | The worker/executor has no explicit request owner scope. A fresh unscoped executor operation therefore applies empty current-user rather than inferring it from `db.client_id`. |
| Policy check | `schema/features/postgres_rls.py:43–75` personal predicate compares `COALESCE(owner_user_id::TEXT,client_id)` to `current_setting('app.current_user_id',true)`; 148–155 installs the same predicate as WITH CHECK | A correctly labelled personal row still fails if the session user is empty. This matches the observed denial. |

HTTP authentication normally installs scope via `auth_deps.py:364–388`; it belongs to the request context. A queued worker is a separate execution lifetime, not a continuation of that authenticated request. Content RLS deliberately separates authorization context from a caller-supplied DB label.

## Existing patterns and coverage limits

- `scope_context.scoped_context` (141–148) is the existing token/reset helper. `shared_workspace_chat_service.py:602` demonstrates an explicitly authorized owner scope. `Web_Scraping/orchestration/executor.py:203` demonstrates explicit context copying for a thread executor; copying alone cannot invent the missing worker owner.
- UAT204 / TASK13260.142 corrected a cold StudyPack accessor from `study-pack-worker-2` to canonical owner `2`, with explicitly qualified restricted-role post-acquisition coverage. That repaired a stored/cache identity defect. This ingest path already uses numeric owner/client and instead lacks the authorization context at persistence. Do not reopen or claim equivalence to its narrower accepted coverage.
- `test_media_ingest_jobs_worker.py`'s owner-search case calls `_create_db('2')`, inserts directly, then scopes **reads**. It does not use an explicit required-PostgreSQL restricted role or the actual document executor insertion. Many other worker cases stub processing/database creation. The integration job suite explicitly selects SQLite Jobs storage; it is not evidence of this restricted content-RLS path.
- `test_media_db_postgres_rls_ops.py` validates SQL through fake backend calls; `test_media_db_request_scope_isolation.py` includes SQLite/fake session controls. Neither substitutes for a real restricted-role worker persistence regression. No existing test was executed or declared passing here.

## Bounded repair recommendation and causal controls

1. Bind the **trusted persisted job owner** explicitly for Media worker content work, preserving token reset on success, failure, and cancellation. Ensure the same intended authorization reaches the actual persistence executor callable and any shared child persistence callbacks used by this ingest path. A numeric `user_id` argument, constructor client ID, or async-only scoped block is not sufficient by itself.
2. Prefer the existing scoped-context helper and an explicit propagation/binding at the current worker/persistence thread boundary. Do not make every generic database client ID an authorization authority or change the global pool/RLS default. Define whether explicit org/team context is supported for these jobs; never manufacture memberships/admin privileges or trust arbitrary upload options. Preserve synchronous HTTP ingestion's existing authorized scope and SQLite device/provenance label semantics.
3. RED must execute the actual worker → document persistence → executor → MediaDatabase/repository path with official required-PostgreSQL fixtures and a direct NOSUPERUSER/NOBYPASSRLS role. Stub only extraction/model output if needed; do not mock the writer, scope function, INSERT or policy. Capture sanitized executor `get_scope().user_id` and the actual checked-out `app.current_user_id`/is_admin with row-owner assertions. Keep the observed native receipt separate from fixture reproduction.
4. GREEN: owner 1 and owner 2 persistence; correct owner readback and foreign denial; same thread/pooled checkout reused across jobs without authority leakage; absent/mismatched owner fails safely; scope restoration after success/error/cancellation; meaningful source/child write rollback; SQLite controls. Retain an unscoped/mismatched direct write rejection control to prove the guard remains active. Keep job lease/cancel and existing result/error semantics unchanged unless separately authorized.
5. Then rerun the ordinary native ingest on reviewed source with the same restricted runtime contract and verify persisted media/source/owner readback. Model-analysis failures and the outer completed/nested-warning presentation are not resolved or accepted by this diagnosis.

## Hash binding

All 21 selected frozen source/history files below match the original `copy-preparation/pg-single-archive-manifest.json` entries. This is selected-file verification, not a runtime tree walk.

Archive manifest SHA-256: `f9a6d30e6a8faef5635df40d5ee026e18ebc225d344bf29b62a8bcca7f2b2f4f`.

Source paths relative to `sources/pg-single`:

| Path | SHA-256 |
|---|---|
| `tldw_Server_API/app/services/media_ingest_jobs_worker.py` | `5878a1b81706d0c95a29ce4c52da0da4bd9fc3b2fff3c5f9af0b6303b916802f` |
| `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py` | `94f066f27f63a7c2adbbbf9737d4e682ac5e41e39956df3c34342c56a25ce680` |
| `tldw_Server_API/app/core/Jobs/worker_sdk.py` | `b0fefdee93d48bade931978466900b2fe0e383d7539f9715b5bb25920c0b0d47` |
| `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py` | `0527850094c24d2f776aa0b30d67aff52d7a53f06f3ad7b22d8fa77c0f600600` |
| `tldw_Server_API/app/core/DB_Management/media_db/api.py` | `12481e48b89cbf4e1b81b41b5f90fac47d2589f58da69f4ce67339ebae1887e8` |
| `tldw_Server_API/app/core/DB_Management/media_db/runtime/factory.py` | `606171475b2e433935391b27b4369fc2ab0de2a5a46179e794a232843f610a82` |
| `tldw_Server_API/app/core/DB_Management/media_db/runtime/connection_lifecycle.py` | `99449a6275b0bf26cbd0d6738ded987a9dd9355c43deedee83e81aeb0722f6b9` |
| `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py` | `1321396222b90ab55ab3062fbebe75791e6658fc2e1ab91cb3fc95c9d233e9ec` |
| `tldw_Server_API/app/core/DB_Management/media_db/repositories/media_repository.py` | `8dc7bb652903d2c64146ccc6edbc9975231664507505722269f0fc5c48f09499` |
| `tldw_Server_API/app/core/DB_Management/media_db/schema/features/postgres_rls.py` | `216678344788f4692a9ef8d6222d304f31156ba0574be821632ba39184cfb715` |
| `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py` | `6a85f94a1aaf7346a5d92ec237205238e72d20a47e61d118567dc771cdf9deb3` |
| `tldw_Server_API/app/core/DB_Management/scope_context.py` | `1a5b2c94d1afeb3b78528d2873b05d7bb0c8fbc169ef54f185f64e915a973ade` |
| `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py` | `1c2737edf2bc7e535f031ef2ea0a4c02f475972a44cea304770a29582e691149` |
| `tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py` | `f2a365648e4113092ba5da890125e913d77f52c2da0df0f79bbe7839fd0a6e9d` |
| `tldw_Server_API/app/core/Web_Scraping/orchestration/executor.py` | `57bdce454f0686f5fac9d33d8277a1e21ef2ba694c80370d6909122dc57b341d` |
| `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py` | `cc815970837783011b95e90c1105c52e357ca7723524c77c669f1bf997257e64` |
| `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs.py` | `2e2a929b6c265dfa5bbc1e7f078de539825c1c85a9a65dabaf96fc3c87170e7d` |
| `tldw_Server_API/tests/DB_Management/test_media_ingest_owner_backfill.py` | `afa494606c4253bc4d6c039b455a09e19ade0267929a87ebabefeb7f5ca9e99a` |
| `tldw_Server_API/tests/DB_Management/test_media_db_postgres_rls_ops.py` | `ca0a8a21e9139980540de14dfc05757c41912ba4286c6ecf89b30eb147b22b2c` |
| `tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py` | `44648dbb1fd97f6612c8d9138cc54a4147e04593e7e4cad4ee60d194ca97e771` |
| `backlog/tasks/task-13260.142 - Use-canonical-owner-identity-for-cold-Study-Pack-workers.md` | `5bbf479573bd5f8589aa82d886690c23cde2ba41878b420e4e94a8be8a0a32a8` |

Evidence paths relative to `native/pg-single`:

| Path | SHA-256 |
|---|---|
| `ingest-observation.txt` | `6836015c37a818bd949a8740329239f5be6cfc9ddbcc81c68ce52525ee09949c` |
| `ingest-database-error-log-excerpt.txt` | `8b6419eff3ac5f4266af2eac6a0f5ab770ed31e548d2c219afe823475aea7ff5` |
| `ingest-postgres-error-log.txt` | `e37ee4afa0c151ebc7704b605e16058e9b013e88acb2b8d60e3970da6e167073` |
| `startup-summary.json` | `d07c453a47c5512fed514665138d91b86a81f081b83ae7d4cd113ab937ee9093` |
| `ingest-failed-media-catalogue.txt` | `af51d827beb79b506099ca418a944ef9340806066162100e0d68de287e59c030` |
