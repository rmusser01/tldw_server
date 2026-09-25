# Live SQLite Email Validation — 2026-09-25

TASK-13362 exercised the full FastAPI app through a real Uvicorn server on
`127.0.0.1`, with test-mode flags unset. The server reached both `/health`
and `/internal/ready` (HTTP 200), then shut down cleanly.

The probe used temporary SQLite AuthNZ, scheduler and per-user Media databases;
two synthetic users had separate organizations and API keys. Alice's organization
had an explicit storage quota. No personal email or Gmail account was connected.
The temporary probe and its raw log stayed outside the repository at
`/tmp/email_live_sqlite_probe_13362.py` and
`/tmp/email_live_sqlite_probe_13362_final.log`.

## Configuration and observed result

| Setting or check | Result |
| --- | --- |
| AuthNZ and content | `AUTH_MODE=multi_user`, `PROFILE=multi-user-sqlite`, `CONTENT_DB_MODE=sqlite`; isolated database paths |
| Core flags | Native persistence and operator search enabled; media delegation `opt_in` |
| Optional sources | `EMAIL_GMAIL_CONNECTOR_ENABLED=false`, `CONNECTORS_WORKER_ENABLED=false` |
| Outbound/model guard | Non-loopback socket attempts 0; intercepted model calls 0; upload analysis, claims, chunking and embeddings disabled |
| Unauthenticated email search | HTTP 401 |
| Alice EML upload | HTTP 200, one successful synthetic message |
| Alice email operator search/detail | One expected message; detail HTTP 200 |
| Alice media operator search | One expected email result |
| Bob email search/detail | Zero messages; Alice's message detail HTTP 404 |
| Storage | Separate Alice and Bob Media SQLite files under the temporary root |

The first probe received HTTP 403 because the live multi-user billing path
requires an active organization context. Creating synthetic memberships, adding
`X-TLDW-Org-Id`, and assigning Alice a quota made the probe reflect that
requirement; this did not require a product change. A later probe assertion
was corrected for macOS's `/var` → `/private/var` path alias. The final
run exited successfully.

One email observability defect surfaced: successful persistence logged
`Metric email_native_persist_total not registered`. Its counter therefore
cannot be relied on in the first run. TASK-13363 registered that counter with
`path_kind` and `outcome` labels. The focused metric test recorded a
`path_kind=primary, outcome=success` sample, and a second full-server synthetic
SQLite run passed with no unregistered-metric warning. Its raw log is
`/tmp/email_live_sqlite_probe_13363_metric.log`.

This is a loopback smoke test with one synthetic message. It does not establish
TLS/reverse-proxy behavior, multi-worker behavior, PostgreSQL tenant isolation,
production data parity, or the one-million-message performance target. The
core release gate remains open.
