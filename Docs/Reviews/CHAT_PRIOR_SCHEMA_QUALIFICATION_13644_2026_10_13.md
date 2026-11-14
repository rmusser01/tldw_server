# Prior schemas and incarnation preservation — TASK-13644

Four provenance fixtures now construct real historical SQLite76/77 and PostgreSQL80/81 schemas using the existing constructors and official backend fixtures. Temporary target versions are restored before upgrading to the actual current heads. Tests no longer rewind a current marker while retaining future columns. Original RLS, receipts and repeat-reopen assertions remain; original note content is also checked.

The PostgreSQL v61 preservation control checks every historical conversation field exactly, separately verifies a newly assigned nonempty incarnation and retains that same identity on the second rerun. The incarnation forgery test keeps SQLite's trigger-message assertion and checks PostgreSQL's proper sanitized backend error. Both engines additionally require all conversation and message rows to remain unchanged after the rejected update.

Causal qualification failed7 and passed34; the repaired complete suites pass41tests with no skips,12warnings,85.89s and process exit0 on the available official SQLite/PostgreSQL fixtures. Independent bounded review is clear. Black formatting preserves AST; one existing import block is normalized. Ruff is clean; scoped test Bandit has0findings/scanerrors after excluding intentional assertions. No production migration, duplicate-column tolerance or ownership policy changed.

ADR required:no; test qualification under existing migrations and [ADR-071](../ADR/071-conversation-source-incarnation.md). Normal hooks and publication are recorded in the task. Other whole-PR findings, final independent review, final remote-dev rebase and merge remain pending.

PR: https://github.com/rmusser01/tldw_server/pull/3229
