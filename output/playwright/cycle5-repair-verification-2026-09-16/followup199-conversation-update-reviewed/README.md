# UAT199 remove unused SQLite-only rowid projection

Integrateddd5104799a. Exactone-column removal; no predicates/writes/transaction orversion changes. Causal6PGFAIL/6SQLitePASS; repaired and independent22PASS0skip. Realbackend controls cover update/search/conflicts/deletedrows/callerrollback. Ruff6baseline6current0added; Bandit0prod/test(B101excluded). Nativeupdate acceptance remains pending untilnewsource loaded.

Known runtime credentials and JWT/PEM patterns scanned, zero matches. Original evidence remains unchanged.
