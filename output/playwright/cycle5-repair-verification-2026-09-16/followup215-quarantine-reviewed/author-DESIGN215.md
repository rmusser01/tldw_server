# UAT215 / TASK13260.154 — Study Pack quarantine status

The real worker persists terminal `quarantined`; Study Pack serializer falls back to `queued` and the public error helper recognizes only `failed`. Reuse existing public `failed` state for quarantine and existing sanitized failure text. Preserve Jobs internal status, retry/quarantine policy, public schema, owner checks, queued/processing/completed/cancelled and pagination behavior. List status filters continue their existing raw Jobs status contract; this bounded repair does not redesign filtering.

Tests first: actual JobManager SQLite and official temporary PostgreSQL create/acquire/retryable fail with threshold1, assert raw quarantined, then real FastAPI detail and list responses must show failed; detail has safe generic error and no pack, never raw diagnostic text. Existing endpoint, Jobs and frontend terminal/polling controls verify normal paths. Independent review and original native job2 GET after reviewed runtime restart before closure. Preserve job2; no queue retry/mutation during diagnosis.

Stage1 cause and real runtime receipts complete. Stage2 RED/GREEN in progress. Stage3 independent review/native pending.
