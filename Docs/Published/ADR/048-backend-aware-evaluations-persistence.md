# ADR-048: Backend-Aware Evaluations Persistence

**Status:** Accepted
**Date:** 2026-09-25
**Backfilled from:** `Docs/Evals/Evals-Plan-1.md`, `tldw_Server_API/app/core/DB_Management/Evaluations_DB.py`
**Decision owner:** Human requester approved the bounded persistence backfill after current-code review
**Related task:** TASK-13374
**Related spec/plan:** `Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md`
**Evidence baseline:** `origin/dev` at `3f909e133b`

## Decision

Keep `EvaluationsDatabase` backend-aware for persisted evaluation definitions, runs, datasets, and unified results: use SQLite file storage by default or an explicitly configured/injected PostgreSQL backend, with structured fields stored as JSON `TEXT` in SQLite and `JSONB` in PostgreSQL and DB-owned value normalization for converted primary CRUD records rather than a uniform normalized-read guarantee.

## Context

The historical embedded Evaluations decisions described SQLite-only storage and JSON TEXT. That no longer covers the implemented persistence contract. `EvaluationsDatabase` initializes SQLite tables/migrations or bootstraps PostgreSQL tables, resolves the shared content backend when no backend is injected, and supports explicitly supplied backend instances.

Representative SQLite fields include `evaluations.eval_spec`, `evaluation_runs.config/progress/results`, and `datasets.samples`. The PostgreSQL schema declares their counterparts as JSONB. CRUD writers serialize structured inputs with `json.dumps`; primary evaluation/run/dataset row conversion uses `_json_maybe`, which parses stored JSON strings and accepts already-decoded JSON-compatible values returned by a PostgreSQL driver. Those converters own representation handling, but not every read uses them.

Default SQLite paths are derived through `DatabasePaths`, while callers can supply explicit paths and services can honor `EVALUATIONS_TEST_DB_PATH`. The manager's missing-path fallback uses the configured single-user path, with a legacy `Databases/evaluations.db` fallback on recognized path-resolution errors. PostgreSQL uses its configured backend target rather than the supplied SQLite filename.

[ADR-020](020-db-management-per-user-paths-and-content-backend.md) already governs the project's SQLite default and opt-in shared content backend. This record covers the narrower Evaluations storage/representation contract. Alternatives below are assessed for this backfill; they do not claim to reconstruct an undocumented historical deliberation.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep the SQLite-only embedded decision as the complete current rule | It omits the implemented PostgreSQL schema, backend selection, and driver-returned JSONB values. |
| Describe structured fields as JSON TEXT on every backend | PostgreSQL declares JSONB; a uniform TEXT claim would misdescribe storage and read behavior. |
| Require API callers to decode backend-specific JSON representations | It exposes driver details despite existing DB row conversion and would duplicate normalization outside the storage boundary. |
| Make PostgreSQL mandatory or claim it covers every Evaluations helper | That would change the local/self-hosted default and overstate the reach of this manager; other managers, adapters, connection pools, and A/B-test storage need their own evidence. |

## Consequences

- Primary CRUD consumers should use the manager's row-conversion boundary rather than duplicate SQLite/PostgreSQL representation handling.
- `get_unified_evaluation()` returns raw row dictionaries from the unified table and its legacy fallbacks without `_json_maybe`. Those reads can expose SQLite JSON strings or driver-decoded PostgreSQL JSONB; this ADR does not promise equivalent application values for that helper.
- Schema work must account for SQLite TEXT and PostgreSQL JSONB representations. This is not a claim that every schema change or migration is already equivalent across backends.
- Explicit path overrides, test paths, and the legacy missing-path fallback remain compatibility caveats; per-user defaults are not a blanket user-isolation or RLS guarantee.
- The manager's backend resolver catches recognized configuration/backend-resolution exceptions and can return the SQLite path. This ADR does not claim every manager construction is fail-closed for PostgreSQL; ADR-020's startup-validation rule remains separate.
- `_json_maybe` can return caller defaults for missing, malformed, unsupported, or selected falsey values. This is normalization for the current structured fields, not a lossless arbitrary-JSON-scalar guarantee.
- SQLite migration/fallback paths and PostgreSQL bootstrap differ. Existing dual-backend tests support representative behavior, not universal backend parity.
- Core-run async execution and Jobs-backed recipe execution remain a separate ownership decision under INV-014.

## Follow-up

Use this ADR as the covering record for INV-009 and INV-012. Keep the historical embedded SQLite-only records as provenance, not current complete persistence instructions. Future changes to backend selection, representation, or required isolation guarantees need a separate decision record.
