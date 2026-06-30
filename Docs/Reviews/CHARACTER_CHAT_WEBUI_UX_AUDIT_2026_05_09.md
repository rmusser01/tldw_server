# Character Chat WebUI UX Audit

Date: 2026-05-09
Backlog task: TASK-146
Audience lens: PhD-level UX/HCI review using cognitive walkthrough, task analysis, heuristic evaluation, and error-recovery analysis.

## Scope

This audit walks the WebUI from two character-chat-centered personas:

- First-time user: wants to create or import a character and start a character chat quickly.
- Returning user: wants to find an existing character, resume or start a chat, and make small character edits.

The audit separates observed browser evidence from code/doc interpretation and terminal verification.

## Environment And Evidence

- Frontend: `apps/tldw-frontend`, `bun run dev -- -p 8080`, visited at `http://localhost:8080`.
- Browser driver: Puppeteer 24.36.0 with Chrome for Testing 144.0.7559.96.
- Backend default startup: failed against the default user database.
- Backend audit workaround: temporary config at `/private/tmp/tldw_character_chat_ux_config/config.txt` with `USER_DB_BASE_DIR=/private/tmp/tldw_character_chat_ux_user_databases`.
- Model/provider state: no LLM models were available in the audited server. Character creation and library flows were testable; final LLM response generation was not.
- Puppeteer artifacts: [assets/2026-05-09-character-chat-ux/puppeteer-states.json](assets/2026-05-09-character-chat-ux/puppeteer-states.json)

Key screenshots:

- [First-run connection state](assets/2026-05-09-character-chat-ux/03-home-before-connection.png)
- [Connected first-run home](assets/2026-05-09-character-chat-ux/04-connected-first-run-home.png)
- [Character library](assets/2026-05-09-character-chat-ux/05-connected-character-library.png)
- [New character modal](assets/2026-05-09-character-chat-ux/06-new-character-modal-empty.png)
- [Template applied](assets/2026-05-09-character-chat-ux/07-new-character-template-applied.png)
- [Character created](assets/2026-05-09-character-chat-ux/08-character-created.png)
- [Returning-user search](assets/2026-05-09-character-chat-ux/09-returning-user-search-character.png)
- [Returning-user edit](assets/2026-05-09-character-chat-ux/10-returning-user-edit-character.png)
- [Row chat action result](assets/2026-05-09-character-chat-ux/11-row-chat-action-result.png)
- [Chat empty state](assets/2026-05-09-character-chat-ux/12-chat-empty-state.png)
- [Character mode result](assets/2026-05-09-character-chat-ux/13-chat-character-mode-result.png)

## Database Corruption Verification

This is a verified technical blocker, not just a UX inference.

Observed:

- Default backend startup failed on `Databases/user_databases/1/ChaChaNotes.db` with `database disk image is malformed`.
- The file exists and has a SQLite header, but direct SQLite access fails:
  - `sqlite3 "file:/Users/macbook-dev/Documents/GitHub/tldw_server2/Databases/user_databases/1/ChaChaNotes.db?immutable=1" "PRAGMA integrity_check;"` returns `database disk image is malformed`.
  - `PRAGMA quick_check` and `SELECT name FROM sqlite_master LIMIT 5` fail the same way.
- There are no `ChaChaNotes.db-wal` or `ChaChaNotes.db-shm` files next to the default DB.
- No backup file was found under `Databases/user_databases/1`.
- `sqlite3 .recover` emitted a recoverable SQL stream:
  - `/private/tmp/chacha_notes_user1_recover.sql`
  - 29,892 lines, 22,642,636 bytes
- Importing that recovery stream into `/private/tmp/chacha_notes_user1_recovered_20260509.db` produced an integrity-clean DB:
  - `PRAGMA integrity_check` returned `ok`.
  - `db_schema_version` reports `rag_char_chat_schema|44`.
  - Recovered row counts: `character_cards=451`, `conversations=915`, `messages=2123`.

Implication:

The default user-1 character/chat/notes DB should not be used in place. A recovery path appears plausible, but it should be handled as a deliberate backup-and-restore operation, not by overwriting the original during an audit.

## Findings

### P0: Default Character/Chat DB Corruption Blocks The WebUI Backend

Evidence: Terminal verification above. The default backend cannot complete startup because `ChaChaNotes.db` is malformed.

Persona impact:

- First-time user: sees generic server failure or onboarding detours before reaching character chat.
- Returning user: cannot resume existing character chats from the default local profile.

HCI principle: Reliability, error recovery, and user control.

Improvement:

- Add a startup DB health check that identifies which per-user DB failed and why.
- Quarantine a corrupt per-user DB instead of failing the whole app when safe to do so.
- Provide a `doctor` or setup UI action for backup, `.recover`, validation, and restore.
- Add automatic pre-migration backups for user 1, matching backup behavior visible for other user IDs.

### P1: Row-Level "Chat As Character" Loses Task Context

Evidence: Puppeteer clicked `Chat as Puppeteer Creative Partner ...` from the Characters table. Result: URL changed to `/`, showing Companion Home and a generic "Configure an LLM provider to start chatting" message.

Persona impact:

- Returning user expects to enter a chat with that character.
- Instead, the user is displaced to a different product surface and must infer that model setup blocked the requested action.

HCI principle: Match between system and user goal; visibility of system status; error recovery.

Improvement:

- Keep the user in the character-chat task context.
- If no model is configured, show an in-context blocker: "Select or configure a model to chat as [character]."
- Provide direct actions: `Open Model Settings`, `Retry with this character`, `Create chat without sending`, and `Back to character`.
- Preserve the selected character through model setup and return to the interrupted flow.

### P1: First-Run Onboarding Is Not Route-Aware For Character Chat Intent

Evidence: First-run home after connection says "You're connected. Start by ingesting one source" and frames the first-value loop as `ingest -> verify -> ask`. Character chat is not offered as a first-run path, despite the persona explicitly starting from character-chat intent.

Persona impact:

- First-time character-chat users are pushed toward ingestion and media workflows.
- They can still find Characters from navigation, but the guided path does not align with their job.

HCI principle: User-centered task fit; progressive disclosure; recognition over recall.

Improvement:

- Make onboarding route-aware. If the user came from `/characters`, offer `Create a character`, `Import a card`, `Choose a model`, and `Start character chat`.
- Add a character-chat first-run branch alongside ingest/research.
- Make `Done` or setup completion return to the interrupted route.

### P1: Header "Character" Mode Opens Scene Setup Before Character Selection

Evidence: In Chat, selecting "Character chat" produced "Character mode starter selected. Choose a character before sending" and opened "Scene Director (Actor)." The visible next step is scene context, not character choice. No obvious character picker was visible in the captured state.

Persona impact:

- First-time user: scene setup is premature before choosing who they are speaking with.
- Returning user: mode activation does not quickly resume or select a known character.

HCI principle: Natural task sequencing; cognitive load; recognition over recall.

Improvement:

- Sequence character chat as: character -> model readiness -> optional scene director -> first message.
- Show recent characters and last-used character chats when entering character mode.
- Defer Scene Director unless the user asks for roleplay/scenario controls or selects an advanced path.

### P1: Model Readiness Is Fragmented Across Surfaces

Evidence:

- New character modal shows "No models available" because AI generation is unavailable.
- Chat empty state shows "No AI models available."
- Character row chat redirects to Home with "Configure an LLM provider."
- Characters library itself does not preflight whether chat can start.

Persona impact:

- Users can successfully create a character but cannot tell, at the point of action, why chat will not start.

HCI principle: Visibility of system status; prevention over correction.

Improvement:

- Add a compact character-chat readiness panel: `Server connected`, `Character available`, `Model configured`, `Chat ready`.
- If one item is missing, keep the user in place and provide a local fix.
- Make row chat buttons show disabled or warning state when chat cannot start.

### P2: Character Library Actions Are Visually Icon-Heavy

Evidence: The table uses icon-only row actions for chat, edit, delete, favorite, and more. Accessibility labels exist in the DOM, but the visual affordance is weak for first-time users.

Persona impact:

- Returning users may learn the icons.
- First-time users must infer which small icon starts chat.

HCI principle: Recognition over recall; affordance clarity.

Improvement:

- Make the primary row action a visible `Chat` text button or split button.
- Keep edit/delete/favorite as icons but expose persistent tooltips.
- In gallery view, make `Chat as...` the dominant card action.

### P2: Search Result Count Does Not Match Filtered Result

Evidence: Puppeteer searched for the unique created character. The table showed only that row, but status still read `4 characters found`.

Persona impact:

- Returning users receive contradictory feedback while filtering.
- Screen-reader users may be told the wrong result count.

HCI principle: Visibility of system status; accessibility.

Improvement:

- Update the count after search/filtering, for example `1 of 4 characters shown`.
- Announce result-count changes through the existing status region.

### P2: Creation Flow Is Strong But Too Dense For First-Time Character Chat

Evidence: The New Character modal has useful templates, preview, tags, prompt preset, avatar options, and AI-generation affordances. It also presents a long form and a "No models available" alert before the user has created anything.

Persona impact:

- Power users get control.
- First-time users may see provider setup, prompt preset, avatar mode, tags, and advanced fields before understanding the minimum viable character.

HCI principle: Progressive disclosure; cognitive load management.

Improvement:

- Split the drawer into "Quick character" and "Advanced character."
- Keep required fields and templates first.
- Move prompt preset, avatar generation, tags, and advanced fields behind secondary disclosure unless a template needs them.
- When models are unavailable, hide or disable AI-generation buttons with one clear explanation.

### P2: Terminology Competes Across Assistant, Persona, Character, Companion, And Scene

Evidence: The audited path exposed `tldw Assistant`, `Persona Garden`, `Characters`, `Character chat`, `Companion Home`, and `Scene Director (Actor)`.

Persona impact:

- A user seeking character chat must distinguish adjacent concepts before they have a stable mental model.

HCI principle: Consistency and standards; information scent.

Improvement:

- Establish a user-facing taxonomy:
  - `Characters`: reusable speaking identities.
  - `Character chat`: conversations using a selected character.
  - `Scene`: optional roleplay/context layer.
  - `Persona`: only if materially different from Character.
- Add short, local disambiguation only where the user must choose between concepts.

### P3: Power-User Features Are Present But Not Task Prioritized

Evidence: Characters has keyboard shortcut hints, table/gallery modes, filters, display options, trash, inline edits, favorites, and import/export-like controls.

Persona impact:

- Returning users benefit from density and shortcuts.
- First-time users need a clearer default path through create/import -> chat.

HCI principle: Flexibility and efficiency of use.

Improvement:

- Keep table density for regular users.
- Add a first-run empty-state/checklist focused on character chat.
- Add a recent character-chat section to the top of Characters or Chat.

## Persona Walkthrough Summary

### First-Time Character Chat User

Observed path:

1. Connects to server from Home.
2. First-run guidance recommends ingestion, not character chat.
3. Navigates to Characters manually.
4. Sees a useful but dense library table.
5. Opens New Character and can use templates.
6. Can create a character without a model configured.
7. Cannot start actual chat because model setup is missing, and the blocker is not kept in character context.

Main improvement:

Create a character-chat onboarding lane that preserves intent: `Connect server -> choose/create/import character -> configure model -> start chat`.

### Returning Character Chat User

Observed path:

1. Characters table supports search and shows recent activity.
2. Search finds the intended character but reports an incorrect total count.
3. Edit opens a full character maintenance drawer with version history, advanced prompt controls, metadata, preview, and generation settings.
4. Row-level `Chat as...` is discoverable to assistive tech but icon-only visually.
5. Clicking `Chat as...` lands on Home rather than in a character chat or local blocker.
6. Chat's character mode opens scene setup before character selection.

Main improvement:

Optimize for resumption: recent characters, last chats, a clear `Chat` primary action, and in-context model/readiness blockers.

## Recommended Work Packages

1. DB recovery and startup resilience
   - Add a documented recovery workflow for malformed `ChaChaNotes.db`.
   - Add backend startup checks that isolate per-user DB failures.
   - Add diagnostics language that distinguishes server unreachable from DB initialization failure.

2. Character-chat intent preservation
   - Preserve selected character across model setup and route redirects.
   - Make `Chat as...` produce either a chat or an in-context blocker.
   - Ensure the fallback action returns to the same character.

3. Route-aware first-run onboarding
   - Add a character-chat lane for users arriving at `/characters` or selecting `Character`.
   - Do not force the ingestion-first story for every new user.

4. Character mode task sequence
   - Character selection first.
   - Model readiness second.
   - Scene Director optional third.

5. Library clarity polish
   - Visible primary `Chat` action.
   - Accurate filtered counts.
   - Dialog-scoped template selection and simpler quick-create mode.

## Verification Notes

- Puppeteer walkthrough completed and wrote screenshots plus JSON to `Docs/Reviews/assets/2026-05-09-character-chat-ux/`.
- Default backend DB corruption was verified directly with SQLite.
- A temporary backend profile was used for live WebUI exploration because the default profile cannot start.
- Final message sending was not tested because no LLM model/provider was configured.
- Bandit is not applicable to this documentation-only audit; no production code was changed.
