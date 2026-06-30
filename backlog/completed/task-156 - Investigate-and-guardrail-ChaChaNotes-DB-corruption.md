---
id: TASK-156
title: Investigate and guardrail ChaChaNotes DB corruption
status: Done
assignee: []
created_date: '2026-05-09 05:17'
updated_date: '2026-05-09 05:31'
labels:
  - character-chat
  - db
  - sqlite
  - ux-audit
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-09-character-chat-db-recovery-root-cause-plan.md
  - Docs/superpowers/specs/2026-05-09-character-chat-ux-work-packages-design.md
  - Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_AUDIT_2026_05_09.md
  - Docs/Operations/ChaChaNotes_DB_Corruption_Recovery_Runbook_2026_05_09.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the first character-chat UX work package: preserve and baseline the malformed default ChaChaNotes database non-destructively, validate the recovered candidate, investigate plausible corruption paths, and add startup/recovery guardrails so a corrupt per-user DB yields actionable diagnostics rather than an opaque WebUI failure. Do not overwrite the original user database during investigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A dated forensic note records hash, file metadata, SQLite header, sidecar state, immutable integrity/quick-check outputs, recovery validation, and root-page mapping for the malformed default DB copy.
- [x] #2 Recovered candidate quality is documented with integrity output, critical table counts, sample sanity checks, schema version, and explicit limitations.
- [x] #3 Root-cause analysis distinguishes confirmed evidence, contradicted hypotheses, and unknowns across interrupted writes/checkpoints, missing WAL sidecars, migration-time writing-table damage, concurrent access, and filesystem/tooling interruption.
- [x] #4 Startup or dependency handling classifies SQLite malformed/corruption failures for ChaChaNotes DBs without silently creating or overwriting user data.
- [x] #5 A documented doctor/recovery flow explains backup-first recovery validation and explicit restore steps.
- [x] #6 Focused tests cover corruption classification and the chosen startup/dependency behavior using temporary malformed SQLite inputs.
- [x] #7 Relevant docs are updated and verification commands are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the malformed default ChaChaNotes DB and record immutable forensic evidence without modifying the source.
2. Generate and validate a `.recover` candidate, then test whether the app wrapper can initialize it.
3. Map recovered `lost_and_found` root pages and review likely root-cause paths across WAL sidecars, write/checkpoint interruption, migrations, backup/restore, concurrent access, and filesystem/tooling.
4. Add dependency-layer corruption classification so malformed ChaChaNotes SQLite files fail closed with safe API/health diagnostics before wrapper construction.
5. Document a backup-first manual doctor/recovery flow and verify with focused tests, Bandit, and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved the original malformed DB non-destructively at `/private/tmp/chachanotes-corruption-forensics-20260509/ChaChaNotes.original.db`; source and copy SHA-256 both `5c159820e8eb1954f1c04ed2ddb606371b3924d2a1993d36a13304d32bd5cb92`. Immutable quick/integrity checks fail with `database disk image is malformed`; source had no WAL/SHM sidecars at investigation time.

`.recover` produced an integrity-clean candidate with schema `rag_char_chat_schema|44` and recovered counts including `character_cards=451`, `conversations=915`, and `messages=2123`. The app wrapper smoke check rejected the candidate because `flashcards_fts` is missing, so it is salvage data rather than a direct restore candidate.

Root-page mapping shows `lost_and_found` rows concentrated on `writing_themes`, `sqlite_autoindex_writing_themes_1`, and `writing_wordclouds`. Most likely corruption classes are interrupted SQLite write/checkpoint or unsafe WAL-mode copy/restore; current backup code uses SQLite backup APIs and does not itself prove a raw-copy cause.

Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/API_Deps/test_chacha_notes_db_deps_error_mapping.py -q` passed with 6 tests; `git diff --check` passed; Bandit JSON scans for touched production code and touched test code completed with zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented ChaChaNotes SQLite corruption guardrails and documented the DB investigation/recovery path.

Changes:
- Added a read-only quick-check preflight before constructing `CharactersRAGDB` for an existing per-user ChaChaNotes DB.
- Classified SQLite malformed/corruption signatures as a safe `503` response with detail `ChaChaNotes DB corruption detected; repair or restore required`, avoiding path leaks while setting health `last_error` to `sqlite_corruption`.
- Logged the local user/path context when the corruption preflight fails.
- Added focused tests covering direct init failures, cached waiter failures, and malformed temp DB preflight behavior.
- Added a dated operations runbook with forensic evidence, recovered candidate quality, root-cause hypothesis outcomes, and a backup-first manual doctor/recovery flow.

Why:
- The original DB is genuinely malformed under SQLite checks, and a recovered candidate contains useful data but is not app-wrapper restore-ready because `flashcards_fts` is missing. The app should therefore fail closed with actionable diagnostics rather than silently creating, overwriting, or treating the failure as a generic startup error.

Verification:
- Focused pytest: 6 passed.
- `git diff --check`: passed.
- Bandit on touched production/test scope: zero findings.

Known limitation:
- This package documents the recovery flow and prevents opaque startup failure, but it does not implement an automatic FTS rebuild or one-click restore path for `.recover` candidates.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
