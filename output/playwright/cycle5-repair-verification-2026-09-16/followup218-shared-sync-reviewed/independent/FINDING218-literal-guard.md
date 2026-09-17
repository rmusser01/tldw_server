# UAT218 independent review finding — exact operation CHECK guard

P2, confirmed on initial frozen manifest0503e25583006e78db2ef199129b31754c7cc12e54c567f611c9f2d38e30f43b. No production edits by reviewer.

`ensure_postgres_sync_log_contract` strips all whitespace from `pg_get_constraintdef` before comparing it with the two approved known definitions. This also removes spaces *inside quoted operation values*. A custom check allowing `cre ate` or `create ` is therefore mistaken for the old application check and replaced with the five-operation check.

Private actual-PG normal Media constructor→custom isolated fixture constraint→normal reopen:2 expected failures,0skips,0.96s. The errors are assertion failures because reopen succeeded; metadata receipts prove that the custom constraint definition changed. No initializer mock or native database was used. All eight initial source/test hashes matched before/after. Initial174 regression suite passed44.75s; this counterexample is additional coverage.

Minimal recommended correction: compare the canonical full definitions directly, preserving literal whitespace; no parser or broader compatibility policy is needed. Add these two negative cases permanently and prove unchanged custom schema on rejection. The author/root have been notified; final review remains pending corrected freeze.
