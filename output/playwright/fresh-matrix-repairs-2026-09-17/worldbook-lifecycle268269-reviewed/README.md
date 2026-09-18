# Reviewed World Book lifecycle repairs (UAT268/269)

Entry create/read/update/delete, book delete and Character detach use the existing database transaction boundary. Removing local commits keeps writes inside caller-owned transactions. PostgreSQL no longer treats its connection wrapper as a context manager. Ownership, conflict checks, IDs, return types, cache invalidation and schema remain unchanged.

Causal PostgreSQL failures and SQLite rollback failures are retained. Author checks: 57 actual SQLite/PostgreSQL cases. Independent root checks: 21 lifecycle/timestamp cases and 10 permission/negative cases, zero skips. A separate reviewer approved the frozen source and evidence. Bandit has no production finding and 34 test-only assertion findings. Native lifecycle acceptance remains pending.

Ten legacy mock failures reproduce on committed baseline0d7 and are separately tracked as UAT270/TASK13260.211. They are not claimed passing here. Safe evidence is retained as exact bytes or lossless gzip; private fixture logs/configuration are excluded. No full-matrix acceptance.
