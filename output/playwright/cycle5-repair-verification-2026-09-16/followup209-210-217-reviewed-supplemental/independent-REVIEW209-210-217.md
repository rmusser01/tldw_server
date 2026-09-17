# Independent review — UAT209, UAT210 residual, UAT217

## Verdict

**CLEAR within the approved scope; no remaining actionable defect found.** The frozen source is ready for the parent’s integration and bounded native acceptance. This review does not close native acceptance or certify the unrelated UAT216 method.

Independent required-PostgreSQL/SQLite run: **202 passed, 0 skipped, 2 warnings, 292.02 seconds**, exit 0. This comprises 156 selected-owner cases, 24 graph cases and 22 literal-search cases. All six current source/test hashes and their review snapshots matched the author manifest before and after the run. Scoped Ruff: 0 findings. Bandit: 0 findings / 0 parse errors across the three production files and three tests; only test assertion rule B101 was excluded.

## Reviewed boundary

Author packet: `.tmp/uat209-repair-20260917/`. Final manifest SHA-256: `18c31ef5d905b2916ebc38c3df5053a5937e85c5dd81dfdf4f12a04a097a956d`. Its change from the initial manifest is status prose; the six source/test bytes stayed fixed.

The complete `owned209.patch`, surrounding transaction/readback code, separate UAT210 and UAT217 patches, and all three permanent suites were inspected. Independent AST comparison confirms exactly the declared **84 changed methods** across ChaCha helpers/folders, NoteStore and KeywordStore, plus the separately owned `_ensure_study_pack_schema_postgres` method. Module and class nonmethod AST content is unchanged. The shared ChaCha hash includes UAT216; that method was excluded from this ownership review.

### UAT209 ownership and compatibility

- Owner clauses are PostgreSQL-only and parameter-bound. Relevant facade/store callers explicitly opt into the selected database owner; generic helper defaults and SQLite per-file/device-label semantics stay unchanged.
- Reads, optimistic-version lookups, writes, restores and zero-row status readbacks all apply the selected owner. Foreign rows remain absent rather than being reassigned by a later metadata update. The scoped initial read also rejects a foreign note before an otherwise successful no-op update.
- Keyword links and reverse/batch/count/search queries check both resource parents. The mapped keyword table is used consistently. Malformed pre-existing cross-owner links are excluded; merge only moves/deletes links within the selected owner’s resources.
- Message authority follows its live conversation, not its `client_id`, which can be a device label. The helper locks message then conversation and rejects deleted/foreign ancestors. `messages.conversation_id` is non-null in the schema. This preserves the real owned-device-label case while closing the misleading canonical-label case.
- Sync upsert/tombstone requires the canonical selected PostgreSQL client ID. UUID conflict updates and versioned updates include the owner guard. The existing idempotent ingestion postcondition also compares `client_id`; a foreign no-op cannot advance local graph/Studio state. SQLite’s independent-file UUID and device-label behavior is retained.
- Folder name/path lookup, restore and linked reads are owner-qualified. Mutating link operations validate parents inside the existing transaction. Global ingestion source-key collisions are rejected/preserved rather than transferred; no new uniqueness policy or migration is introduced.
- Tests use the real cold dependency/cache and actual HTTP Notes routes for two users on one shared PostgreSQL backend, plus separate SQLite files. Caller-owned raw BEGIN, nested and supplied-connection rollback, owned lifecycle, stale-owner mutation, same-name reuse, malformed links, and deletion controls pass.
- A real NOSUPERUSER/NOBYPASSRLS role is assumed inside an explicit fixture transaction after normal fixture bootstrap. It verifies selected-owner service create/read and existing Notes RLS. This is not a cold restricted-login bootstrap claim. It does not imply arbitrary SQL isolation for a bypass role.

Existing scoped Studio/attachment/modern graph and shared-owner loader policies were not rewritten. Private Notes do not become shared by supplying a workspace ID. The result is bounded application persistence protection, not a whole-database authorization audit.

### UAT210 residual

The only residual production correction resolves `get_note_tag_edges`’ keyword JOIN through the existing backend table mapper. The previous five-method mapping-row repair is already committed. The new empty-edge control still executes the query with a real note, so it cannot pass by taking an empty-input early return. Both empty and populated PostgreSQL/SQLite cases pass. Later owner predicates in the same method belong to UAT209.

### UAT217 literal search

The isolated production change is `ILIKE ?` → `ILIKE (?)`, disambiguating the bound operand for the existing SQL translator. Backslash, percent and underscore escaping remains unchanged; the simple-token FTS path, quote/empty rejection, owner filtering, ordering and bound limit are preserved. The actual database cases cover punctuation and decoys, combined backslash/bang ordering and limit. No shared translator rewrite or mixed-placeholder workaround is introduced.

The author’s failed escape-only candidate is correctly retained as **7 failures / 15 passes**, despite its historical `green` log label. It is not counted as validation. The causal and corrected-driver history remains explicitly qualified.

## Evidence

The independent command is recorded verbatim in `verification.json`. It used the existing credential-redacting required-PG runner and official DB_Management fixtures; no native databases or model calls were used.

- Independent test log: `required-pg-independent.redacted.log`, SHA-256 `891818919803bbd116a8a0d187eb2619b1967527f8c4d2e24e5171597bcdda17`.
- Source attribution: `before-source-hashes.json`, `after-source-hashes.json`, `independent-ast-review.json`.
- Static checks: `ruff-current.json`, `bandit-production.json`, `bandit-tests.json`, `static-summary.json`.
- Selected original causal receipts were inspected and their hashes/results retained in `retained-causal-receipts.json`, including six read failures, foreign Sync writes, actual cold HTTP failure, rejected message-label candidate, deleted ancestor, mapped JOIN and literal-search failures. This reviewer did not rerun the full RED matrix.
- Author adjacent 176 Notes/lifetime and 69 Sync passes were read as author evidence, not relabelled as independent reruns. They predate the final message-parent correction; the dedicated corrected controls are in the independently passing frozen 202 cases.

Bandit emitted existing `nosec` comment/parser warnings; its JSON contains zero findings and zero parse errors. Suppressed SQL sites were reviewed for closed table/alias fragments and bound values. This is not a warning-free stderr claim. The two pytest warnings are retained in the full redacted test receipt.

No production/test/task/git/browser/runtime edits were made by the reviewer. Only this private review packet and isolated test-fixture state were written. Native Notes, tag-edge and literal-search acceptance remains the parent’s gate.
