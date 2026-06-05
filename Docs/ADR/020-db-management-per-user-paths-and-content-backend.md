# ADR-020: DB Management Per-User Paths and Content Backend

**Status:** Accepted
**Date:** 2026-06-04
**Backfilled from:** `tldw_Server_API/app/core/DB_Management/README.md`
**Decision owner:** TASK-2253 confirmation and TASK-2254 backfill scope
**Related task:** TASK-2254
**Related spec/plan:** `Docs/ADR/inventory/2026-06-04-db-management-confirmation-audit.md`

## Decision

DB_Management owns per-user database path resolution through `DatabasePaths` under `USER_DB_BASE_DIR` by default, SQLite remains the default content-storage mode using per-user file paths by default, and PostgreSQL is an explicit shared content backend option that must pass startup validation when enabled.

## Context

The project stores content and user-scoped state across Media DB, ChaCha Notes, Prompts, Evaluations, Workflows, and related feature databases. Those stores need predictable local/self-hosted defaults, safe multi-user path separation, and an explicit path for deployments that want shared PostgreSQL-backed content storage.

TASK-2253 confirmed the current DB Management behavior that bounds this ADR:

- `DB_Management` is the module boundary for database path utilities, content backend selection, DB factories, migrations, and representative user-scoped stores.
- `DatabasePaths` resolves per-user directories from `USER_DB_BASE_DIR`, defaulting outside tests to `Databases/user_databases`, and derives Media, ChaCha Notes, and Prompts database paths from that base.
- Single-user mode maps missing storage IDs to the configured fixed single-user ID, while multi-user mode rejects missing user IDs for per-user storage.
- SQLite is the default content backend. In SQLite content mode, callers resolve per-user database files rather than receiving a shared content backend.
- PostgreSQL content mode is opt-in through content backend configuration. When enabled, DB Management creates and caches a shared PostgreSQL backend and Media/content runtime requires that backend.
- Startup validation calls PostgreSQL content backend validation and reraises runtime validation errors, so schema and required Media/sync RLS policy gaps stop startup when PostgreSQL content mode is configured.

This ADR is intentionally bounded. It does not include AuthNZ/users DB persistence, does not claim that every DB family is fully PostgreSQL-backed, and does not remove documented path overrides or compatibility aliases.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Let each DB caller choose its own default file path | Scatters path ownership, makes multi-user isolation inconsistent, and keeps legacy root-level Media DB patterns alive as normal architecture. |
| Make PostgreSQL the default content backend | Local/self-hosted single-user deployments still need a zero-service SQLite default, and PostgreSQL content mode requires explicit operational setup, schema readiness, and RLS validation. |
| Treat SQLite content mode as a shared backend abstraction | Current SQLite behavior intentionally uses per-user file paths instead of returning a shared content backend, which keeps file ownership and user isolation visible to callers. |
| Claim all DB families are PostgreSQL-backed | The confirmation evidence supports Media/content backend runtime plus representative backend-aware factories and dependencies, but some user-scoped stores still instantiate SQLite files directly. |
| Include AuthNZ `DATABASE_URL` and users DB persistence in this ADR | AuthNZ has separate persistence configuration and security semantics, so it needs a separate AuthNZ persistence decision if one is required. |
| Remove or ignore explicit SQLite path overrides and legacy aliases in the decision | Current behavior still supports explicit SQLite path overrides, test fallback directories, and selected deprecated aliases, so the ADR must preserve those caveats. |

## Consequences

New user-scoped database defaults should use `DatabasePaths` and should not introduce new root-level Media DB-style paths without a separate compatibility decision.

SQLite remains the default content-storage mode. In SQLite mode, DB factories and API dependencies should resolve per-user file paths by default, while still respecting documented explicit SQLite path overrides such as `TLDW_CONTENT_SQLITE_PATH` or `[Database].sqlite_path`.

PostgreSQL content mode remains opt-in. When a feature participates in PostgreSQL content mode, it must account for shared backend creation, schema readiness, and required RLS policy validation. Startup should fail on PostgreSQL content runtime validation errors rather than silently falling back to SQLite.

Test-mode path isolation, deprecated compatibility aliases, and historical explicit path overrides remain recognized caveats. Documentation and tests should distinguish default production/local paths from test fallback paths.

AuthNZ/users DB persistence and any future decision to make additional DB families fully PostgreSQL-backed remain separate decisions.

## Follow-up

- Use this ADR as the covering record for the DB Management portion of INV-030.
- Keep `Docs/ADR/inventory/2026-06-04-db-management-confirmation-audit.md` as the evidence record and caveat boundary for this backfill.
- Create separate ADRs if AuthNZ persistence policy or non-Media DB family PostgreSQL support becomes a durable architecture decision.
