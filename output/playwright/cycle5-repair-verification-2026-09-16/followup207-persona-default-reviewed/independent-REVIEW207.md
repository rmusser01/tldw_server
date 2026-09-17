# UAT207 independent root review — clear

TASK13260.145. Reviewed all three production diffs, core/endpoint callers and36 permanent cases against the unchanged default-ID and ownership contracts. No unresolved finding in this bounded repair.

Fresh required official PostgreSQL/SQLite run:36 passed,0skips,56.68s. Both actual cold profiles/catalog routes cover two owners and repeated reads; legacy and tombstone controls preserve IDs/content. Direct core materialization propagates the resolved ID. Ten positive/failed insertion cases cover implicit pending work, raw BEGIN and empty managed/backend/borrowed transactions. Profile uniqueness failures retain typed ConflictError and a usable connection; unexpected errors propagate.

The local psycopg transaction is intentionally bounded to profile insertion; public outer transaction helpers alone do not supply the required savepoint or preserve an empty caller-owned transaction. The explicit BEGIN is limited to IDLE connections with an existing owning depth. Active operation-use bookkeeping spans the savepoint. Generic backend/execute behavior is unchanged. Endpoint delegation removes the duplicate default bootstrap; policy seed uses the resolved profile ID.

Author additionally passed121 adjacent Persona/session/memory and26 lifetime controls,0skips; these were inspected, not claimed as independently rerun. Independent Bandit has0findings/0errors in all production paths and new test (B101 excluded only for assertions); diff whitespace clean. Author Ruff baseline/current five identical pre-existing findings, no additions. Frozen four-path hashes match after review and independent tests.

Native retry requires a later accepted backend restart; current native API still runsa130b8e550. This is not full Persona-domain, full fresh-matrix or clean-project validation.
