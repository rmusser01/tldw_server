# UAT199 conversation update projection

Task13260.137. Bounded repair: remove unused `rowid` from the existing conversation-update preflight SELECT. The method never consumes this column; no SQL condition, write, transaction, exception, version, owner or search behavior changes. A dialect conditional or fake PostgreSQL row identifier would add unnecessary code.

Actual native Bob message creation returned201 while its conversation updates logged PostgreSQL undefined rowid. Native server remains on pre-repair source48f89447fc. Completion has a separate Persona Memory failure200. These observations are retained under `.tmp/uat193-031-native-20260917`.

New real-backend regressions reproduce six PostgreSQL failures with six SQLite controls passing (15.93s), before the production edit. After removing that column, the same12 plus10 adjacent ConversationStore tests pass22/0skip (17.46s). Each PostgreSQL case uses the official fixture with PostgreSQL required. Behaviors cover touch/version/identity, title+rating+memory-mode edit and real title search, stale-version rejection, missing/deleted conflicts, outer rollback and search restoration. SQLite FTS triggers remain intact. Test formatting changed after GREEN, with no assertion or behavior change; independent verification will use frozen formatted bytes.

Scoped Ruff reports six baseline/six current diagnostics, zero added/removed; new tests are clean. Production Bandit0 findings/errors; test Bandit0 excluding B101 assertions. Scoped whitespace check passes. Frozen identities are in source-manifest.json. No native acceptance, broader ownership audit, provider success or full UAT is claimed. Independent review remains required before integration.
