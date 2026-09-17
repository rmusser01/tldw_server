# UAT203 / TASK13260.141 — PostgreSQL persona-memory count result

A real official-PG count probe with both archived/deleted inclusion enabled bypassed UAT200's bad predicate and reached a distinct failure: COUNT succeeds, but `row[0]` raises KeyError because PostgreSQL aggregate rows are mappings. SQLite positional access succeeds. Parent created this separate task before the count edit.

Exactly one expression changes in `count_persona_memory_entries`: use `row["count"]` for PostgreSQL and retain `row[0]` for SQLite, preserving the existing empty fallback. SQL, bound filters, query/transaction flags and count semantics are unchanged. The shared archive predicate is UAT200's separate prerequisite. `owned.patch` isolates this expression and the new test; the full-source snapshot also contains UAT200's two statements.

## Evidence

Permanent `test_persona_memory_count_backends.py` covers empty counts, all four archive/deleted policies, populated owner/persona/type boundaries, and another user's zero result for a specified persona on both real backends.

- Before count edit, with the UAT200 predicate/read fix already present: **6 PostgreSQL failures / 6 SQLite passes / 0 skips, 14.58 seconds**, all PostgreSQL failures at positional count access.
- Final combined suite: **103 passed / 0 skipped / 2 explicitly deselected, 63.84 seconds**, including all 12 count cases. The two excluded existing migration fixtures also fail with pre-UAT200 source and are separately tracked under reopened UAT183; see the UAT200 report, not a claimed passing migration suite.

`causal-red-command.json`, `causal-red.redacted.log`, `final-green-command.json` and `final-green.redacted.log` retain exact commands and outcomes. Focused rerun uses the official required-PG helper with `test_persona_memory_count_backends.py -q --tb=short` after activating `.venv`.

Ruff: 0 new findings; the two production findings are exactly baseline, and the new test is clean. Production Bandit 0 findings / 0 parse errors; tests 0 / 0 with B101 excluded only for assertions. Python AST and scoped whitespace checks pass. Full underlying static evidence is in the adjacent UAT200 packet; `static-summary.json` is copied here.

Frozen manifest SHA256: `c81575ad63db39d1ecaf28c846e3b75ea5cac676500b44b9dd11b68931546b43`.
Combined source SHA256: `52652fa347b2d945abcdf01674a16f2f76d997875bc6a56995457656dee9cfea`.

No native UI/runtime/provider action or task/tracker/git edit. This repairs the separately diagnosed store count contract; it is not an independent native completion acceptance claim. Independent review remains pending.
