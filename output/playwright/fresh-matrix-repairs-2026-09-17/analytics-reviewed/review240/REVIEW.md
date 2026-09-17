# Independent UAT240 review — CLEAR

**TASK13260.182. No findings or requested changes.** Independent actual SQLite/PostgreSQL verification passed **296 tests, zero skips**, four existing warnings, in 233.22 seconds. Native Hard/re-rate statistics acceptance remains pending.

## Scope and contract

Verified all four frozen hashes before and after testing against `source-freeze.json` SHA256 `15521a447de3aee657f0a505038a9cdf1e59c0660ba3561bc6eb4b571e572219`. Production SHA256 is `bf6e79fe0fc0c31262c8acbcc95e2cb0650e16bee6aeb366b1e2ced8aca0004d`. Full bindings are in `source-before.json`, `source-after.json`, and `REVIEW.json`.

Exactly three production methods change: daily metrics count persisted `was_lapse`; live and reconstructed session correctness require non-lapse and rating other than Again0. The rest of the production module AST is unchanged. Owner/deck/workspace/date predicates, transaction handling, scheduler transitions, API shape and writes remain intact.

The canonical field is non-null with a false default. Trusting it preserves intentionally recorded historical outcomes without inventing a rating fallback. Again in learning remains incorrect session recall without being a mature lapse. Both current schedulers accept API-only ratings1/4 as recall, consistent with the documented compatibility policy. The new tests assert actual stored review outcomes and saved card lapses before checking analytics.

## Evidence and fixture review

- Retained original RED: 28 failures/45 controls on 73 cases. All original RED test functions are AST-identical in the frozen final test; three additional functions add nine historical reconstruction/UTC/owner controls.
- Reviewed retained adjacent six-failure receipt. Four session fixtures now use actual Again0 to represent incorrect recall; their aggregate, repair, and race assertions are preserved. The explicit Hard2 reconstruction case correctly expects three recalls.
- The HTTP metric fixture first graduates the second card with actual Easy5, then records Again0 as a real lapse. It counts all three review events and retains the 3500ms mean over the two supplied durations. The change preserves the intended successful/failed review coverage.
- Independent same-three-suite run: analytics82 plus session/endpoint214, 296 passed/zero skipped. Uses the official required-PG runner and disposable fixture only; no native profile or service changed.

## Static checks and limits

Independent compile passes for all four files. Bandit reports zero findings/errors (test assertions alone excluded with B101). Ruff reports zero findings for production/analytics/sessions and the same 14 pre-existing endpoint-test diagnostics; baseline/current code/message pairs match exactly. The scoped diff and exact commands/results are retained here.

This is source, database, and actual test-router verification, not a native browser acceptance claim. No production/test edits were made by this reviewer. No history was rewritten or broad lint cleanup performed.
