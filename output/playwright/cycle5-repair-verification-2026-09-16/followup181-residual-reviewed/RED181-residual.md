# UAT181 residual permanent RED

TASK13260.118. Production unchanged at this checkpoint; all five source baselines and exact hashes are retained in red-source/ and red-source-manifest.json. Final test snapshot: test.before-green.py.

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-residual-final-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_shell_read_lifecycle.py -q --tb=short
```

**9 failed,22 passed,0 skipped,4 warnings**,45.70s. All PostgreSQL cases used official isolated fixture databases. Both SQLite controls pass. First run had the same9/22 result; final run narrows test exception diagnostics and retains executor relation-state context. No production change between these runs.

## Proven failures

- Four standalone starters retain INTRANS: character readiness, character by-name lookup, persona profile list, persona buddy projection list.
- Actual existing-default maintenance followed by real Buddy reads blocks actual replacement initialization in `_ensure_chacha_rls_postgres`; followed by actual Notes list it blocks note-folder initialization.
- Actual persona list endpoint (including buddy projection) followed by Buddy or Notes reads blocks actual replacement in `_ensure_recent_persona_schema_postgres`.
- Actual `_ensure_default_character_async` on its dedicated test-owned executor retains only character_cards; the replacement fails in `_ensure_chacha_rls_postgres`. This is stronger evidence than an idle-status assertion alone.
- Replacement constructors execute production SQL. Tests shorten the existing bootstrap lock budget and connection lock_timeout to100ms; no SQL or backend path is replaced.

## Passing controls

Both real Buddy/Notes replacement chains with no predecessor pass. Twelve caller-write cases retain pending writes for implicit, nested ChaCha and backend transaction owners until the caller commits/rolls back. Six first-read explicit-scope cases retain caller ownership. Both SQLite actual-chain/rollback cases pass. Total22 positive controls.

## Approved bounded response

Parent native overlap independently failed: API32260 still alive as38457 reached health200; actual Notes returns500 and observer sees old7234 holding persona_profiles while new12217 waits for AccessExclusiveLock. Parent retains `.tmp/uat181-native-20260917/bounded-replacement-notes-38457.json` and matching wire/UI evidence. This agent did not operate native runtime/browser/DB. Earlier nonoverlapping replacement remains distinct.

Parent now releases exactly six `read_only=True` SELECT sites across three files: readiness normal/recovery; get_character_card_by_name normal/recovery; list_persona_profiles; list_persona_buddies. Reuse existing owned-read semantics. No dependency scheduling, endpoint, global commit/rollback, backend helper or RLS change. Recovery SELECTs use the same pure-read contract; the real PG normal/executor chains establish the regression. Adjacent SQLite/store tests will cover existing behavior.

Ruff check and formatting pass. Bandit test report has zero findings/errors, B101 excluded for pytest assertions. GREEN, prior lifecycle and adjacent checks, independent review and parent native reacceptance are still required.
