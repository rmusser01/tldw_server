# Independent root review: UAT181 enrichment callback ownership

Clear for integration. Independent mandatory official PostgreSQL and SQLite run: **32 passed, zero skips, 34.79 seconds**. Production and test hashes match the frozen author versions during and after execution. Native acceptance remains pending.

The production diff adds one import and an independent operation context inside each existing native tagging/clustering callback try block. Direct and pytest-inline execution retains its existing caller ownership. Existing exception handling also covers cleanup failure; no success auto-commit is added. The context releases only its owned checkouts after the callback completes, including the successful tagging path that creates another clustering thread. I inspected the scheduling branches, operation-scope contract, real thread/pool fixtures, lifecycle controls and diff.

Tests exercise actual PostgreSQL queries and observed real pool calls for early exits, missing conversations, successful child callbacks, repeated use, SQL failure, in-flight work after HTTP ownership ends, caller commit/rollback isolation and unfinished callback writes. The corrected lease ledger retires an exact generation before returning the connection and records success only after actual return. It never holds its ledger lock while awaiting a checkout. The suite does not assert unavailable cancellation behavior or generic ownership of every background task.

Independent Ruff is0 baseline/0 current using the real source path (an initial copied-file path applied different per-file rules; its diagnostic output is retained separately). Independent Bandit reports0 findings/0 errors for production and tests; tests exclude only B101 assertions. Both files parse. No code changes were made by this reviewer.

The retained native COUNT transactions provide causal context. This verification does not accept the actual overlapping backend replacement, account-switch bootstrap or StudyPack worker journeys; those remain parent-owned targeted UAT after integration and runtime replacement. UAT199 rowid and UAT200 persona filtering are separate repaired defects.
