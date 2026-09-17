# Residual PostgreSQL read-transaction candidates

Read-only source investigation for parent TASK13260, 2026-09-17 UTC. No repair, test execution, database/runtime/browser action, task change or new issue classification. A fixed-to-fixed replacement failure has **not** been established by this investigation.

## Evidence boundary

[fixed-runtime-source.json](fixed-runtime-source.json) records API PID14540 and the four frozen181/182 production hashes. All four current files matched those hashes during inspection. Parent reports old API2996 exited. [after-old-api-interrupt.json](after-old-api-interrupt.json), captured at `2026-09-17T00:38:45.602051+00:00`, records these remaining sessions:83569/83604 idle in transaction with `character_cards` AccessShareLock;84146 idle in transaction with `buddy_profiles` and `buddy_attachments` AccessShareLock. Other listed sessions are idle.

The receipt identifies database sessions and a selected relation inventory, not application thread IDs or the first statement of each transaction. Its `transactionReadOnlyVerified` refers to the **observer connection**, not every observed transaction. [metadata-monitor.py](metadata-monitor.py) filters to selected public tables; absence of other relation names is not proof no other reads occurred.

## Strong candidate: request-driven default-character maintenance

1. [ChaCha_Notes_DB_Deps.py:841](../../tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py#L841) schedules `_ensure_default_character_async` after ordinary `get_chacha_db_for_user` resolution. [Line506](../../tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py#L506) executes `_ensure_default_character` on the dedicated ChaCha executor.
2. [Lines539–543](../../tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py#L539) call readiness, look up the default card, and return immediately when it exists.
3. [ChaChaNotes_DB.py:20039](../../tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py#L20039) runs `SELECT 1 FROM character_cards LIMIT 1` without `read_only` or an explicit transaction. The recovery-verification read at [20065](../../tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py#L20065) does the same.
4. [character_store.py:318](../../tldw_Server_API/app/core/DB_Management/chacha/character_store.py#L318) also performs the by-name SELECT without an owned read scope (recovery retry at328). The existing-card success path has no close/settle operation.
5. [ChaChaNotes_DB.py:7909](../../tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py#L7909) documents that PostgreSQL connections are cached per thread until released. Thus these reads can retain the executor connection's transaction after maintenance returns. This is consistent with the two character-only sessions, but their exact executing threads were not observed.

Multi-user [startup_chacha_warmup.py:28](../../tldw_Server_API/app/services/startup_chacha_warmup.py#L28) explicitly skips startup warmup. For this PG-multi receipt, ordinary request-driven maintenance is the relevant candidate; do not attribute it to the single-user startup path.

## Buddy-only session remains unexplained

The immediate [Buddy endpoint/service path](../../tldw_Server_API/app/api/v1/endpoints/buddies.py#L64) constructs no database reads before the service operation. [Buddy_DB.py:81](../../tldw_Server_API/app/core/DB_Management/Buddy_DB.py#L81) get/list/assets and [attachment:165](../../tldw_Server_API/app/core/DB_Management/Buddy_DB.py#L165) all already opt into `read_only=True`. [BuddyService.attachment:331](../../tldw_Server_API/app/core/Buddy/service.py#L331) returns immediately for an empty slot. New ChaCha connection client-scope setup [commits at7841](../../tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py#L7841), so that normal setup path is not an identified residual starter.

[execute_query:8206](../../tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py#L8206) owns a read only when the connection is IDLE and neither ChaCha nor backend managed depth is active. Preserving a pre-existing transaction is intentional; the39 opt-ins cannot safely settle unknown earlier work. A predecessor read could explain84146, but none was established in the immediate empty Buddy chain. The different PID and absent character lock prevent assigning the character-executor explanation to it.

One separately reachable shell read, [persona_state_store.py:1808](../../tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py#L1808), leaves `list_persona_profiles` unscoped; [persona.py:3818](../../tldw_Server_API/app/api/v1/endpoints/persona.py#L3818) calls it from GET `/persona/profiles`. This could precede other reads in a different sequence, but would retain `persona_profiles`, which the current receipt does not show. It is **not a demonstrated cause of84146**.

## Existing tests and the missing composition

- [test_chacha_postgres_study_read_lifecycle.py:64](../../tldw_Server_API/tests/DB_Management/test_chacha_postgres_study_read_lifecycle.py#L64) closes the seeded connection before standalone reads. [Its real Buddy/replacement test:268](../../tldw_Server_API/tests/DB_Management/test_chacha_postgres_study_read_lifecycle.py#L268) calls BuddyService directly; it does not exercise request dependency/default-character scheduling. Its caller-transaction controls intentionally retain pending work.
- [test_chacha_postgres_note_read_lifecycle.py](../../tldw_Server_API/tests/DB_Management/test_chacha_postgres_note_read_lifecycle.py) similarly exercises direct endpoint chains with an injected DB after seed cleanup.
- [test_chacha_notes_db_deps_postgres_health.py:42](../../tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_postgres_health.py#L42) exercises `_get_or_init_db_instance`, bypassing the wrapper that schedules default-character maintenance. Startup warmup tests cover scheduling/mode policy; default-character error-mapping tests use stubs.

If further reproduction is authorized, the bounded missing composition is actual default-character maintenance on an existing-card PostgreSQL instance, then observer transaction/lock inspection and actual replacement initialization. A separate preceding-statement trace is needed for the Buddy-only session. Do not fix it by committing all existing transactions, expanding every SELECT flag, or treating these candidates as a failed replacement without runtime evidence.
