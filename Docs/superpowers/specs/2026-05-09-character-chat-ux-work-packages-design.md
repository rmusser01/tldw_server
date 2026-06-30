# Character Chat UX Work Packages Design

Date: 2026-05-09
Backlog task: TASK-154
Source audit: `Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_AUDIT_2026_05_09.md`

## Goal

Turn the character-chat UX audit into separate, executable work-package plans. The packages must be small enough for independent review, but coordinated enough that shared concepts such as character-chat readiness, selected-character intent, and DB startup health do not fragment across the WebUI and backend.

## Package Boundaries

The work is split into eight packages:

1. DB recovery and corruption root-cause investigation
2. Character-chat intent preservation
3. Route-aware first-run onboarding
4. Character-mode task sequencing
5. Model readiness and in-context blockers
6. Library clarity and quick-create polish
7. Terminology and taxonomy alignment
8. Post-implementation character-chat walkthrough re-audit

Each package gets its own implementation plan in `Docs/superpowers/plans/`. No production changes are included in this planning task.

## Sequencing

The first package should run before any work that assumes the default local profile can start. The audited default user database, `Databases/user_databases/1/ChaChaNotes.db`, is malformed and should not be used in place.

Model readiness should land before or alongside intent preservation, character-mode sequencing, and onboarding. Those UI flows all need the same answer to "can a character chat actually start right now?"

Intent preservation and character-mode sequencing should land before the re-audit. Library clarity, quick-create, and terminology work can be implemented in parallel once shared readiness and intent contracts are understood.

The post-implementation re-audit is last and should use the same user personas and Puppeteer/Chrome-driver evidence pattern as the original audit.

## Shared Product Rules

- A user who selects or creates a character should not lose that character when model setup is required.
- A row-level `Chat` action should either open the intended character chat or show an in-context blocker that preserves the selected character.
- Character chat should sequence as character selection, model readiness, optional scene setup, then first message.
- First-run onboarding should respect character-chat intent when the user arrives from `/characters` or selects a character-chat entry point.
- The Characters library should keep power-user density while giving first-time character-chat users a clear primary path.
- User-facing terms should distinguish `Character`, `Character chat`, `Scene`, and `Persona` only where the user needs to make a decision.

## DB Corruption Evidence To Preserve

Known facts from the audit and follow-up inspection:

- Default startup fails on `Databases/user_databases/1/ChaChaNotes.db` with `database disk image is malformed`.
- Immutable `PRAGMA integrity_check`, `PRAGMA quick_check`, and a simple `sqlite_master` query fail on the original file.
- The file has a valid SQLite header and is in WAL file format mode, but no `ChaChaNotes.db-wal` or `ChaChaNotes.db-shm` sidecars are present.
- `.recover` produced a SQL stream that imports into an integrity-clean recovered DB.
- The recovered DB reports `rag_char_chat_schema|44` and recovers 451 character cards, 915 conversations, and 2123 messages.
- The recovered DB contains `lost_and_found` rows associated with root pages 87, 88, and 89. In the recovered schema those pages map to `writing_themes`, `sqlite_autoindex_writing_themes_1`, and `writing_wordclouds`.
- `writing_themes` and `writing_wordclouds` are introduced around the v15/v16 writing migrations.

These facts do not prove a single cause. The DB package must test candidate causes separately: interrupted writes/checkpoints, missing WAL sidecars after an unsafe copy or restore, migration-time page damage around writing tables, concurrent process access, and external filesystem/tooling interruption.

## Verification Strategy

Each implementation package should have focused unit/component tests first, then the narrowest integration or browser verification that proves the user-facing workflow. The re-audit package should rerun a full Puppeteer walkthrough and produce a comparison report against the 2026-05-09 baseline.

Bandit applies only to backend production/test code touched by implementation packages. Planning-only docs should record Bandit as not applicable.
