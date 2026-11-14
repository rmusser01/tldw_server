# Native quota PostgreSQL allocation — TASK-13703

Task identity corrected on 2026-10-15: this PR's historical TASK-13641 is now TASK-13703; the old ID belongs to a separate task on dev. This report keeps its stable filename and original evidence.

Actual PostgreSQL retention allocation failed with the private type-only chain ProfileUserWriteRejected <- TokenError <- TokenError. The native reservation INSERT used compact question-mark placeholders that became adjacent PostgreSQL dollar parameters. The managed SQL parser could not tokenize that list and refused it. The fix spaces bound placeholders; original values, ownership, locking, reservation state and SQL guard remain unchanged.

A focused regression executes the real reservation repository, transaction-bound parameter adapter and managed guard on both engines. It failed on PostgreSQL and passed on SQLite before the fix; both then passed, followed by actual PostgreSQL retention success. The first full run passed65 and failed11, exposing eight similarly compact fixture setup INSERTs and an admin test attempting to patch a read-only asyncpg instance method. Fixture INSERTs now use spaced bound parameters. The admin outage targets the backend's actual query method on the original connection and still runs its missing-table query through the managed method, retaining503 behavior.

Final official-fixture qualification passes all76 cases on actual SQLite/PostgreSQL, with no skips and1837 runtime warnings (209.89s, process exit0). Cases cover replay/release, rejected identity/receipt authority, rollback, shared-budget races, held-delete rereads, strict retention byte refusals, interrupted preparation and fresh/prior bootstrap preservation. Independent bounded review is clear. Production Bandit has0 findings/scan errors; test scope has0 after excluding intentional test assertions. Ruff passes; Black validates all three touched Python files and the embedded runtime script parses.

ADR required: no; this corrects SQL emission under the existing owner/transaction/quota design, including proposed [ADR-070](../ADR/070-native-chat-retained-byte-quota-authority.md), with no new durable policy. No API schema or client behavior changed. Normal exact-file hooks and publication are recorded in the task. Whole-PR failures, final independent review, rebase and merge remain pending; this is not a whole-PR approval or installed-extension/AT acceptance.

PR: https://github.com/rmusser01/tldw_server/pull/3229
