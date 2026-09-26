# PR2969 checkpoint review

PR: https://github.com/rmusser01/tldw_server/pull/2969

Tasks: TASK13260.242, TASK13260.242.1 (CI), TASK13260.242.2 (Qodo). This review is limited to the published checkpoint. The failed full UAT matrix and its 16 open findings remain unchanged.

## Initial hosted CI finding

The first head, c8613667d9, fails frontend unit shard 4/8 at the first Study Pack source-to-drawer handoff. The test's synchronous click starts asynchronous React work before a default one-second title lookup. The revised test awaits `act` around the user action. Assertions and timeouts are unchanged. An independent review found no product race; the local 26-case suite passed before the synchronization change. Final targeted frontend validation passes all 36 cases, including ten Media controls, on CI's Node20.20.2.

Scoped TypeScript validation also found 17 unsupported `exact` options on role queries in this PR's handoff tests. Removing those options retains the default exact name match. The final three touched frontend files have zero semantic diagnostics with the existing Vitest setup included. The initial check without that setup produced missing matcher declarations and is retained as a harness error, not a product failure.

## Qodo dispositions

Each number links to the original inline finding. Replies and resolution are recorded on the PR after the verified patch is published.

| Finding | Disposition and evidence |
| --- | --- |
| [4053603179](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603179): advisory SQL in AuthNZ repository | Moved bound advisory-lock SQL into `DB_Management/backends/postgresql_locks.acquire_schema_lock`. The caller still owns the active transaction, commit/rollback, and cancellation. All five real PostgreSQL blacklist tests pass. |
| [4053603183](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603183): fault setup connection ownership | Added a typed administrative fixture depending on the official AuthNZ `isolated_test_environment`. It verifies the database identity and closes its connection before the owning fixture tears down. Production guards remain active. Two attempted application-pool variants correctly rejected destructive DDL and are retained as failed validation; the final five cases pass without skips. |
| [4053603190](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603190): Prompt database lifecycle | Retained the official shared PostgreSQL plugin. Root `conftest.py` registers it; `pg_database_config` depends on `pg_temp_db`, which owns database creation/drop. The local content fixture closes only its adapters/pool. AuthNZ's separate application fixture is inappropriate for these content-only tests. Added a clarifying ownership comment; real SQLite/PostgreSQL sync and restricted-role controls pass. |
| [4053603195](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603195): retrieval diagnostics | Failed sources now log the enumerated source and exception class, both in structured context and the ordinary message. Aggregate failure also records operation, class, and source count. Existing request logging context is retained. Raw exception payloads/traceback locals remain excluded because they can expose SQL, credentials, and user queries. Partial success and terminal all-failed behavior remain covered. |
| [4053603199](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603199): streaming diagnostics | Standard retrieval failures record operation and exception class, visible in ordinary logs. Terminal error/fallback safety remains unchanged. Regression assertions require useful diagnostics and reject private path/query leakage. |
| [4053603203](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603203): manual BEGIN before savepoint | Retained the existing transaction logic. The ChaCha and PostgreSQL wrappers may increment transaction depth before sending BEGIN; psycopg `raw.transaction()` at IDLE would otherwise own and commit the caller's transaction. The conditional BEGIN materializes that caller-owned transaction, after which the context manager owns only a savepoint. Real tests cover idle reads, pending writes, lazy outer transactions, SQL failures, and outer rollback. A second independent review confirms this contract; replacing it with another lazy wrapper would break rollback ownership. |
| [4053603206](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603206): blacklist helper types | Added fixture, connection, async iterator, executor, parameter and return annotations, including literal scenario values. |
| [4053603211](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603211): blacklist helper documentation | Documented fixture ownership, gated DDL execution, cancellation propagation, transaction completion, and bounded overlap observation. |
| [4053603214](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603214): Prompt test types | Typed fixture, persisted event tuples, test arguments and returns. |
| [4053603220](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603220): Prompt test documentation | Documented durable project-event and prompt-revision scenarios. |
| [4053603232](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603232): Media handoff object identity | Fixed a real transfer rejection when keyword hydration replaces the selected object for the same item. Validate stable kind/ID while retaining current-media and account guards. Before the fix, the same-media case fails and two changed-item/account controls pass. All ten adjacent Media tests now pass. |
| [4053603228](https://github.com/rmusser01/tldw_server/pull/2969#discussion_r4053603228): private exemplar injection | Faults now enter through public `execute_query`, without matching generated search SQL or patching private statement preparation. The database still executes an invalid statement, so PostgreSQL's actual aborted-transaction and rollback behavior is tested. |

## Verification and limits

- 54 unique database cases pass: 49 SQLite/PostgreSQL exemplar and Prompt cases from the combined run, plus all five blacklist cases after fixture correction. Zero PostgreSQL skips. The earlier combined run's four fault-setup failures remain recorded.
- 109 retrieval/streaming tests pass. Both diagnostic regressions fail before correction; a second causal pair verifies context is visible in the default text message as well as structured records.
- 36 frontend tests pass on Node20.20.2; scoped TypeScript diagnostics are zero. Independent frontend review has no actionable findings.
- Ruff passes the touched Python scope. Bandit on four touched production Python files reports zero findings and zero parse errors. ESLint reports zero errors; 19 warnings are pre-existing (14 in ViewMediaPage, five on unchanged test lines), with no new warnings. The root invocation's missing Next pages directory warning is retained.
- Independent Python review caught the application-pool fault-setup rejection; the fixture correction was reviewed again and accepted. No application authorization guard was weakened.
- Evidence stays ignored under `.tmp/postmerge-uat-checkpoint/`; generated captures, private runtime settings, and unrelated files remain outside the PR.
- The requester supplied a new Change summary, published verbatim and verified by exact comparison. All seven enforced checks pass on46a98a552c: backend, security, coverage, frontend, end-to-end, container build and trusted-dev license. Hosted coverage runs report442 global and222 AuthNZ tests passing. All eight frontend shards and the WebUI/extension API-key and single-user cookie persistence lifecycles pass. The report-only frontend coverage step is not a coverage threshold certification.

## Merge verification

PR2969 merged normally into dev at2026-09-19T16:15:21Z as1dfdd819b6e7056c2e7721d4763c85d5ad07d85c. Immediately before merge, latest dev was3cff7962721a60b768464221c1f7fe2a8b25e4d5, all12 review threads were resolved and the human summary matched the supplied text exactly. After fetch, the merge is on origin/dev and its tree exactly matches CI-tested46a98a552c. No merge bypass or generated/private captures were used. The remaining16 UAT findings and failed frozen matrix remain open; this is not a new full UAT pass. The completed checkpoint plan is removed; its three stages (finish repair, publish/review, verified merge) are complete. Remaining repairs resume on a new branch from this merge.
