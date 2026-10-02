# UAT369: conversation search punctuation

Task: TASK13260.277.19. Root approved this bounded design on 2026-09-20.

Preserve valid raw SQLite FTS expressions. If SQLite rejects the MATCH text with a recognized query parse error, retry once with the complete input escaped as a literal phrase. Reuse identical owner, deleted, workspace, and other filters; propagate unrelated database errors. PostgreSQL behavior stays unchanged. Existing global normalization is unsuitable because it changes valid NEAR and quoted phrase-prefix searches.

## Stage 1: Causal database regressions
**Goal**: Reproduce the native marker failure through real database operations.
**Success Criteria**: Punctuation tests fail on current source; supported FTS and scope controls pass.
**Tests**: Real SQLite paginated/unpaginated punctuation, count/rank/page consistency, owner/deleted/workspace filtering, supported expressions, unrelated errors; official PostgreSQL fixture controls.
**Status**: Complete

## Stage 2: Bounded SQLite fallback
**Goal**: Retry recognized FTS parse failures with an escaped literal phrase.
**Success Criteria**: Count and rank/page use the same fallback text with unchanged predicates; only one retry; normal queries and PostgreSQL remain unchanged.
**Tests**: Stage 1 suite plus existing conversation search/store/scope tests.
**Status**: Complete

## Stage 3: Verification and review handoff
**Goal**: Freeze a reviewable source snapshot before native acceptance.
**Success Criteria**: Real SQLite and official-fixture PostgreSQL checks pass; scoped lint/type checks show no added diagnostics; Bandit on touched Python has no new findings; exact manifest and limitations recorded in task.
**Tests**: Focused and surrounding pytest suites, matched lint/types where supported, Bandit, whitespace review. Diagnose Sidebar failure presentation without editing frontend files.
**Status**: In Progress

## Evidence and limits

- Initial real SQLite probe: `/private/tmp/uat369-diagnosis/result.jsonl`.
- Causal run: 18 failures and 24 passing controls before repair (`/private/tmp/uat369-red.log`); all 42 passed after repair (`/private/tmp/uat369-green.log`).
- Real PostgreSQL control passed using the official `pg_database_config` / `pg_temp_db` fixture against existing localhost:55475. Docker startup was disabled. Initial sandbox socket denial was resolved with approved execution; native holder databases were not used.
- Added deleted-search and retry-error controls. The first expanded run had 82 passes and two empty-fixture failures: SQLite skipped MATCH evaluation on an empty table. Seeding the real row made both retry-error controls pass; no production change was needed.
- Matched Ruff diagnostics: 6 baseline / 6 current, zero additions. Matched mypy: 7 baseline / 7 current, zero additions after normalizing source line references. Existing production formatting drift was retained; added code and the new test file follow Ruff formatting.
- Bandit: zero production findings. Raw test scan reports only expected pytest-assert B101 findings; test scan excluding B101 has zero findings. Reports are `/private/tmp/uat369-bandit-production.json` and `/private/tmp/uat369-bandit-tests.json`.
- Sidebar diagnosis only: server search destructures data with an empty default and ignores error/loading; empty arrays render “No matches found.” This separate UI behavior is unchanged.
- Exact four-file manifest: `/private/tmp/uat369-owned-files.txt`; Python source hashes: `/private/tmp/uat369-source-sha256.json`.
- Final six-suite run: **84 passed**, including the new 47-case suite and its real official-fixture PostgreSQL control (`/private/tmp/uat369-final-verified.log`). Four existing warnings remain. Source is frozen for independent root review and native acceptance; stage 3 and the task remain In Progress for those gates.
- No shared runtime/configuration, native holder database, frozen candidate, or git changes authorized here.
- Task remains In Progress until root review and native acceptance.
