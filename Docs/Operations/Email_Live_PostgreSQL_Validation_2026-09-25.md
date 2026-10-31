# Live PostgreSQL Email Validation — 2026-09-25

TASK-13364 exercised the full FastAPI app through Uvicorn on `127.0.0.1`,
with test-mode flags unset. AuthNZ and shared content used separate, freshly
created PostgreSQL 18 databases in the existing local test container. The
application role was a database owner but neither a superuser nor a
`BYPASSRLS` role. Ancillary scheduler and user-local data remained under
temporary SQLite paths; this test does not claim every subsystem used
PostgreSQL.

Two synthetic users had separate organizations, API keys, and an explicit
quota for the uploading organization. The probe disabled Gmail and connector
workers, analysis, claims, chunking, and embeddings. Socket guards blocked
non-loopback connections, and model/background-task guards recorded zero
attempts. No personal email or Gmail account was accessed. Raw probe and
logs stayed outside the repository at `/tmp/email_live_postgres_probe_13364.py`
and `/tmp/email_live_postgres_probe_13364_final.log`.
The temporary PostgreSQL databases and application role were removed after
the final probe.

## Final result

| Check | Observed result |
| --- | --- |
| Full app startup/shutdown | Clean exit; `/health` and `/internal/ready` HTTP 200 |
| Auth and core flags | `AUTH_MODE=multi_user`, `PROFILE=multi-user-postgres`, native persistence/operator search enabled, delegation `opt_in` |
| Optional sources | `EMAIL_GMAIL_CONNECTOR_ENABLED=false`, `CONNECTORS_WORKER_ENABLED=false` |
| Unauthenticated email search | HTTP 401 |
| Alice's synthetic EML upload | HTTP 200, one successful message |
| Alice's email operator search/detail | One expected result; detail HTTP 200 |
| Alice's media operator search | One expected email result |
| Bob's email search/detail | Zero results; Alice's detail HTTP 404 |
| Direct PostgreSQL RLS check | `media` RLS enabled and forced; Alice sees one row, Bob zero |
| Guard and SQL diagnostics | Non-loopback attempts 0; model attempts 0; failing PostgreSQL statements 0 |

The first PostgreSQL probes exposed three product defects and one separate
optional-module defect. All were fixed and the final probe was repeated on
fresh synthetic databases:

- TASK-13365 added the missing AuthNZ `storage_quotas` table and indexes to
  the runtime and packaged PostgreSQL bootstraps. A real PostgreSQL quota
  upsert/read regression passes; the final database has a scope check, two
  foreign keys, and both scoped unique indexes.
- TASK-13366 aligned startup validation with schema v26's current
  `media_visibility_access` policy. The validator still requires the
  `sync_log` scope policies.
- TASK-13367 made Collections schema backfills inspect PostgreSQL columns
  and disabled SQLite FTS5 writes for PostgreSQL adapters whose schema
  bootstrap was already cached. The PostgreSQL Collections round-trip and
  focused cache-path regression pass.
- TASK-13368 replaced the MCP media health check's SQLite-only
  `INSERT OR REPLACE` with a portable conflict upsert. The final startup
  no longer logs that health-check failure. The SQLite live probe was
  repeated after this change and still passed.

Two probe setup details were corrected without product changes: live
multi-user billing required a synthetic organization/quota, and PostgreSQL's
default RBAC roles were seeded during app startup, so the probe assigned
the synthetic users' roles after readiness.

This is a one-message loopback smoke test. It does not establish TLS or
reverse-proxy behavior, multi-worker behavior, production parity, or the
one-million-message performance target. The core rollout and scale gates
remain open.
