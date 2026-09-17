# TASK13260.170 / UAT229

Observed production SQLite transaction logging formats brace-bearing exception text before rollback. Actual probe masks ValueError as KeyError, leaves transaction active and pending write visible. Ordinary-text control rolls back.

Scope: parameterize exactly4 Loguru calls in existing TransactionContextManager.__exit__: outer-error, rollback-error, commit-error, rollback-after-commit-error. Preserve logger levels/extra fields, transaction decisions, exception identity/causes/wrapping. No PostgreSQL code, genericlogger wrapper or transaction framework change.

Permanent actual SQLite/PG plain/brace/nested rollback and normalcommit controls; real SQLite write with proxy-injected commit/rollback failures tests each diagnostic branch. Driver-failure injection does not replace database persistence.

Status: final causal RED6 failures/9 controls on15 cases retained. Parent released the production lease after168 commit. Focused GREEN15 and combined GREEN47 passed with zero skips. Frozen for independent review; parent owns tasks/git/runtime.
