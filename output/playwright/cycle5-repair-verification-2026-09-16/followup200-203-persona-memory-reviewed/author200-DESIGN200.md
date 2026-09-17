# UAT200 / TASK13260.138 — persona memory PostgreSQL filtering

Primary native boundary: complete-v2 calls `_inject_character_memory_from_db`, whose optional-error catch continues after PersonaStateStore's `archived = 0` query fails against PostgreSQL BOOLEAN. The same connection is INERROR when WorldBookService initializes, causing the visible500 before a model call. Existing176 WorldBook transaction fix is preserved.

Three reference contracts: persona profile read methods already use execute_query(read_only=True); persona memory writes bind bool for PostgreSQL/int for SQLite; WorldBookService owns its initialization through db.transaction and does not settle a caller transaction. Shared memory filtering preserves owner, persona, scope/session, type, deleted/archive flags, order and pagination.

Stages:
1. Real official PostgreSQL + SQLite RED for actual memory list and actual injection→WorldBook chain, including empty memory and all four archive/deleted policies. No production edits before valid failure.
2. If proven, change only archived predicate to backend-compatible FALSE on PostgreSQL. Prove a real driver failure of a standalone optional list read and caller-owned pending-write behavior before adding that one read_only opt-in. Never broad rollback/commit or endpoint catch changes.
3. GREEN controls, adjacent tests, Ruff/Bandit/diff and frozen review packet. Parent owns native/runtime/task/tracker/git.

Adjacent count positional row access is outside the primary completion path. A real failure will be reported separately before editing it. No generic SQL rewrite or broad store refactor.
