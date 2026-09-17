# UAT247 / TASK13260.189 — bounded sequence maintenance

## Cause and retained evidence
Tenant-filtered MAX(id) is not a sequence high-water mark. The old helper resets even an empty tenant to setval(1,false), reusing another owner's ID, rolled-back allocations, and cached reservations. The final causal suite has six failures/four controls on unchanged production; interleaving allocates 21 real IDs after the final maintenance read and before setval. The original no-option ALTER probe failed (with an initially masked cleanup error); no production version used it.

## Selected boundary
Keep the exact UAT147 Media serial-column allowlist. Acquire each known serial's lock in deterministic catalog order by reasserting its existing catalog-proven OWNED BY table.column dependency. Actual official-PG proof under a non-superuser/non-BYPASSRLS schema owner confirms unchanged pg_sequence parameters, sequence state and pg_depend, and a concurrent nextval waiting on a real PostgreSQL Lock until caller rollback releases it.

Under that lock, read last_value/is_called and tenant-visible MAX, and call setval only to advance a genuinely behind positive maximum (including equal-but-uncalled). Empty or already-ahead state remains untouched. Keep the caller's transaction; do not commit, roll back, change RLS, change configuration, or use an elevated application role. This helper runs in the existing schema-maintenance transaction and requires the schema/sequence owner; it is not a general data-only grantee maintenance API.

Read-last-value then GREATEST/setval without a lock still races. Advisory locking only maintenance does not coordinate ordinary nextval. Repeated nextval would be monotonic but scales with explicit-ID gaps. Sequence locking preserves bounded work and existing import repair. PostgreSQL documentation: https://www.postgresql.org/docs/17/sql-altersequence.html and https://www.postgresql.org/docs/16/functions-sequence.html.

## Stages
1. RED and lock characterization: complete (6 product failures / 4 controls; lock proof pass).
2. Minimal helper implementation and focused/adjacent controls: complete; final117 passed, zero skips.
3. Static checks and four-file freeze: complete. Original worker replay passed48/zero skips; independent review is pending and native acceptance remains parent-owned.

## Tests / limits
Official fixture alone owns databases. Disposable restricted role owns its tables, FORCE RLS protects tenant rows, application scopes remain admin0. Cover foreign hidden IDs, called/uncalled high water, pristine/foreign sequence, explicit imports, rollback allocations, cache high water, real allocation interleaving and parameter-preserving lock release. Preserve existing foreign ChaCha lock controls. Retry owns original worker tests; replay only after source freeze. No native/browser/held-profile changes. Native both-owner queued upload acceptance remains parent-owned.

## Lock-lifetime reassessment (before second production revision)
The first implementation passed11 monotonic/caller controls but failed unchanged147 concurrent ChaCha/Media bootstrap (113pass/1fail total). A driver-only observer proves SQLSTATE40P01: table AccessExclusive versus sequence RowExclusive. Holding the temporary sequence lock until the outer constructor ends is too broad. Source candidate and failure retained.

Proposed lifetime correction: use existing psycopg `conn.transaction(force_rollback=True)` around each sequence's no-op ownership reassertion, reads and forward setval. Rolling back this savepoint releases only its acquired lock/catalog no-op; PostgreSQL setval is intentionally nontransactional and survives. No application row writes occur in this savepoint. Existing caller writes/depth/transaction remain owned by the caller; real proof passes with a pending caller row, allocation completing on another connection before outer close, later caller rollback and continued unique IDs. See https://www.psycopg.org/psycopg3/docs/basic/transactions.html and https://www.postgresql.org/docs/17/explicit-locking.html. Parent approved this second production revision after reading actual proof and official documentation. Final production-helper controls verify allocation completes before caller close, pending rows roll back, and real query errors propagate without settling caller work. The unchanged UAT147 concurrency test passes with the narrowed lifetime.
