# PostgreSQL CASE placeholder and ordinary ingress repair

Goal: correct the shared placeholder parser so ordinary Sync ingress has backend parity.

ADR required: no
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: preserve the existing SQL/backend and Sync authority contracts; no schema or policy change.

1. Add failing parameterized CASE-placeholder regressions to `tldw_Server_API/tests/DB_Management/unit/test_postgres_placeholder_prepare.py`. Cover simple/searched/nested CASE, result binds adjacent to THEN/ELSE/END, unchanged parameters, quoted text/identifiers, and JSONB operators within CASE.
2. Replace the raw envelope insert workaround in `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_ingress_repair.py` with the ordinary `SyncV2Store` insertion API. Verify the existing SQLite/PostgreSQL fixture reproduces the bind mismatch and add domain watermark non-regression checks.
3. Minimally correct CASE keyword recognition in `tldw_Server_API/app/core/DB_Management/backends/query_utils.py`; do not rewrite the Sync query or introduce a SQL parser dependency.
4. Run targeted placeholder/backend-utils and ingress tests with `TLDW_TEST_POSTGRES_REQUIRED=1`, using the shared isolated PostgreSQL fixture. Verify successive ordinary insertions and monotonic watermark behavior on both backends. Run affected-file static checks and self-review; record evidence and obtain independent review before closing the task.

Use the server root `.venv/bin/python` from this isolated worktree and verify source provenance. No full-suite sweep or ongoing-sync capability enablement.
