# Character Chat DB Recovery And Root-Cause Plan

> For implementation agents: use the repository superpowers workflow before editing code. This plan is evidence-first; do not overwrite the original user database during investigation.

**Goal:** Preserve the malformed default `ChaChaNotes.db`, recover usable data non-destructively, identify the most likely corruption path, and add startup/recovery guardrails so a single per-user DB failure does not block the whole WebUI without actionable diagnostics.

**Primary evidence:** `Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_AUDIT_2026_05_09.md`

**Likely surfaces:**
- `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- `tldw_Server_API/app/core/DB_Management/DB_Backups.py`
- `tldw_Server_API/app/core/DB_Management/sqlite_policy.py`
- `Docs/Operations/`
- `tldw_Server_API/tests/DB_Management/`
- `tldw_Server_API/tests/Chat/`

## Confirmed Facts

- The default DB path is `Databases/user_databases/1/ChaChaNotes.db`.
- Default backend startup fails with `database disk image is malformed`.
- Immutable `integrity_check`, `quick_check`, and `sqlite_master` inspection fail on the original DB.
- The DB header is valid SQLite and indicates WAL file format mode.
- No `ChaChaNotes.db-wal` or `ChaChaNotes.db-shm` sidecars exist beside the DB.
- `.recover` can emit SQL that imports into a clean DB.
- The recovered DB has schema `rag_char_chat_schema|44`.
- Recovered counts include `character_cards=451`, `conversations=915`, and `messages=2123`.
- Recovered `lost_and_found` rows map to root pages associated with `writing_themes`, its autoindex, and `writing_wordclouds`.
- Those writing tables are created by the v15/v16 writing migrations.

## Hypotheses To Test

1. **Interrupted write or checkpoint:** a process, OS, or machine interruption occurred while SQLite was writing btree pages for writing tables or while checkpointing WAL content.
2. **Missing sidecar WAL after unsafe copy/restore:** the DB file was copied, moved, backed up, or restored without its WAL sidecar or without a checkpoint.
3. **Migration-time damage around writing tables:** v15/v16 migration or later schema repair touched `writing_themes` or `writing_wordclouds` and left malformed pages.
4. **Concurrent access outside SQLite's intended process model:** two server/test processes or an external tool touched the same file during migration/write.
5. **Filesystem or external tooling interruption:** APFS, sync, backup, quarantine, or manual metadata operations affected the DB file.

Do not report any hypothesis as proven until it has supporting evidence.

## Stage 1: Preserve And Baseline The Original

**Goal:** Make the current corrupted state reproducible and immutable before deeper inspection.

**Success Criteria:**
- Original DB is copied to a dated forensic working directory without modifying the source.
- SHA-256, file size, file times, SQLite header bytes, sidecar-file state, and immutable check outputs are recorded.
- The original path remains unchanged.

**Tests:** Shell verification only; no production tests.

**Status:** Complete

Steps:

- Create a read-only investigation copy under `/private/tmp` or `tldw_DB_Backups/forensics/` if a repo-ignored location exists.
- Record `shasum -a 256`, `stat`, `file`, `xxd -l 128`, and sidecar listings.
- Run immutable `PRAGMA integrity_check`, `PRAGMA quick_check`, and basic schema inspection against the copy.
- Record all commands and outputs in a dated investigation note under `Docs/Operations/` or `Docs/Reviews/`.

## Stage 2: Validate Recovery Quality

**Goal:** Determine whether `.recover` output is safe enough to use as a candidate restored DB.

**Success Criteria:**
- Recovery SQL imports into a clean DB.
- `PRAGMA integrity_check` returns `ok` on the recovered DB.
- Critical tables have plausible counts and sample rows.
- `CharactersRAGDB` can initialize against the recovered copy without mutating the original.
- Any `lost_and_found` rows are classified by root page and likely table.

**Tests:** A one-off recovery smoke script or focused pytest around opening a copied recovered DB.

**Status:** Complete

Outcome note: `.recover` produced an integrity-clean candidate with substantial
character-chat data, but `CharactersRAGDB` rejected the candidate because
`flashcards_fts` was missing. Treat it as salvage data, not a direct restore.

Steps:

- Run `.recover` from the preserved copy, not the live DB.
- Import into a new DB.
- Check schema version, table counts, and sample rows for characters, conversations, messages, writing tables, and sync log.
- Compare recovered counts against any available backups or chatbook exports.
- Run the backend DB wrapper against the recovered DB in a temporary user DB base.

## Stage 3: Root-Cause Investigation

**Goal:** Identify the most probable corruption path and separate proven evidence from remaining uncertainty.

**Success Criteria:**
- Each hypothesis has supporting evidence, contradicting evidence, or a clear "not enough evidence" outcome.
- The investigation narrows where prevention should be added.

**Tests:** Focused reproductions where practical.

**Status:** Complete

Steps:

- Map `lost_and_found` root pages to current schema objects and migration versions.
- Review recent commits and local logs around writing migration paths, backup/restore code, DB startup, and shutdown.
- Check whether any backup or restore path could copy WAL-mode DB files without sidecars or without using SQLite backup APIs.
- Build a small temp DB reproduction for interrupted v15/v16 migration or malformed writing table recovery if feasible.
- Inspect whether `ChaChaNotes_DB.py` runs integrity checks before migrations or only fails after initialization opens the malformed DB.
- Look for multiple local server/test processes that may have shared `Databases/user_databases/1`.

## Stage 4: Startup And Recovery Guardrails

**Goal:** Prevent one corrupt per-user DB from producing an opaque global WebUI failure.

**Success Criteria:**
- Startup logs name the failing per-user DB and classify the failure as corruption.
- Health/setup endpoints can expose a safe degraded state without leaking sensitive paths beyond local/admin contexts.
- A documented doctor/recovery flow exists for backup, recover, validate, and restore.
- Restore remains explicit and backup-first.

**Tests:** Unit tests for error classification plus a startup/dependency test using a deliberately malformed temp SQLite file.

**Status:** Complete

Steps:

- Add corruption classification around `sqlite3.DatabaseError` / malformed errors in the ChaCha init path.
- Add a non-destructive diagnostic helper or command path that runs immutable checks and recovery validation.
- Make restore require an explicit backup of the current DB and an integrity-clean candidate.
- Document the recovery flow and known limitations.

## Risks

- `.recover` can salvage data while losing some rows or indexes. Treat it as candidate data, not automatic truth.
- Automatically quarantining user DBs can hide real failures if not surfaced clearly.
- A startup resilience change must not silently create a fresh empty DB over a user's corrupted profile.

## Handoff Notes

The original DB should remain untouched unless the user explicitly approves a restore. If a recovered DB is proposed for use, provide exact counts, integrity output, and a rollback path.
