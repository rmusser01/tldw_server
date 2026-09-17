# UAT181 residual shell-read composition

Existing TASK13260.118. Parent authorized permanent causal tests, then released exactly six SELECT flags after RED and an independently observed native overlap failure. This packet is separate from the completed39 read opt-ins and UAT187 schema repair.

## Evidence and bounded hypothesis

The latest native metadata records current API32260 character-only idle transactions and session7234 with notes, note_folders and persona_profiles. Old API14540 had exited before that replacement started: this is not evidence of fixed-to-fixed overlapping replacement failure. The parent owns that native experiment. No native DB/browser/runtime action is performed here.

Actual existing-default maintenance calls readiness and get-by-name without an owned read scope. The actual persona list endpoint calls both list_persona_profiles and list_persona_buddies; both SELECTs are unscoped. Earlier Buddy/Notes read flags intentionally preserve a transaction already begun by an earlier statement, so these predecessor reads can retain later relation locks. Character-only lingering transactions and a failed replacement constructor are distinct outcomes to measure.

## Stages

1. **RED (complete9 failures/22 controls,0 skips):** official isolated PostgreSQL fixture; individually exercise four existing read starters, the real default-maintenance executor wrapper, and real default/persona endpoint chains followed by Buddy or Notes reads and a second real CharactersRAGDB constructor. Include a no-predecessor control. Preserve exact fixture row identities; observe only transaction state and relation names. Use the existing short lock-timeout constructor pattern, not simulated DDL readiness.
2. **Ownership controls (all22 controls pass in RED):** implicit pending writes, nested ChaCha and backend transactions must remain unsettled until caller commit/rollback; first reads within explicit scopes must retain scope ownership. SQLite actual-chain controls remain readable and rollback-capable.
3. **GREEN complete; independent review pending:** exactly six approved pure SELECT sites use the existing read_only helper: readiness normal/recovery, by-name normal/recovery, persona list and persona buddy projection list. No broad commit, rollback, auto-close or global SQL classification. All184 new/prior lifecycle and adjacent cases pass with0 skips. Independent review and native acceptance remain required. See RED181-residual.md for actual failures and approved scope.

Owned test: `tldw_Server_API/tests/DB_Management/test_chacha_postgres_shell_read_lifecycle.py`. Existing reference tests: test_chacha_postgres_note_read_lifecycle.py, test_chacha_postgres_study_read_lifecycle.py, Chat default-maintenance scheduling/health tests. Endpoint calls inject the real fixture DB/user and enabled feature flag; they do not certify auth or browser transport. The dedicated executor is test-owned and closed only in fixture cleanup.
