# Character Card Worldbook And Dictionary UX Audit

Date: 2026-05-09
Backlog task: TASK-178
Audience lens: PhD-level UX/HCI review using cognitive walkthrough, task analysis, heuristic evaluation, information scent analysis, and reliability-oriented formative evaluation.

## Scope

This audit validates whether worldbooks and chat dictionaries work with character cards in chat and workspace-oriented work contexts for two personas:

- First-time character-chat user: wants to understand how to make a character remember setting/lore and terminology without reading implementation docs.
- Regular power user: already maintains characters and wants fast, inspectable, reliable context wiring across character chats, workspace chats, and repeated sessions.

The walkthrough used Puppeteer/Chrome against the live local WebUI. It also used API probes where the UI cannot directly expose model-context assembly. Model generation itself was not treated as evidence because the local server reported no configured LLM provider on the chat route.

## Environment

- Backend: `http://127.0.0.1:8000`, `AUTH_MODE=single_user`, `SINGLE_USER_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY`.
- Frontend: `http://127.0.0.1:8080`, `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`, Puppeteer/Chrome.
- Seeded character: `Archivist Mira UX 20260509191047`.
- Seeded worldbook: `Lumen City Field Notes UX 20260509191047`.
- Seeded dictionary: `Mira Terminology Normalizer UX 20260509191047`.
- Seeded global chat: `Mira global lore/dictionary check 20260509191047`.
- Seeded workspace chat: `Mira workspace lore/dictionary check 20260509191047`.

## Evidence

Artifacts:

- [API probe JSON](assets/2026-05-09-character-card-worldbooks-dictionaries/api-probe.json)
- [Puppeteer state capture JSON](assets/2026-05-09-character-card-worldbooks-dictionaries/puppeteer-states.json)
- [Worldbooks focused route](assets/2026-05-09-character-card-worldbooks-dictionaries/world-books-focused.png)
- [Worldbook entries detail](assets/2026-05-09-character-card-worldbooks-dictionaries/world-books-detail-entries.png)
- [Worldbook attachments detail](assets/2026-05-09-character-card-worldbooks-dictionaries/world-books-detail-attachments.png)
- [Character focused preview](assets/2026-05-09-character-card-worldbooks-dictionaries/characters-focused-preview.png)
- [Dictionaries list](assets/2026-05-09-character-card-worldbooks-dictionaries/dictionaries-list.png)
- [Dictionary quick assign modal](assets/2026-05-09-character-card-worldbooks-dictionaries/dictionaries-quick-assign-modal.png)
- [Workspace playground](assets/2026-05-09-character-card-worldbooks-dictionaries/workspace-playground.png)
- [Chat route no-provider state](assets/2026-05-09-character-card-worldbooks-dictionaries/chat-route.png)
- [First-time character focus route](assets/2026-05-09-character-card-worldbooks-dictionaries/first-time-route-gate-character-focus.png)

API reliability checks all passed except dictionary diagnostics in prompt preview:

- `worldBookAttachmentPersisted`: true
- `worldBookProcessInjectedEchoVault`: true
- `promptPreviewLorebookInjectedEchoVault`: true
- `promptPreviewLorebookHasDiagnostics`: true
- `dictionaryExplicitReplacedEV`: true
- `dictionaryActiveProcessingReplacedEV`: true
- `globalChatDictionarySettingsPersisted`: true
- `workspaceChatDictionarySettingsPersisted`: true
- `dictionaryUsageIncludesCreatedChat`: true
- `promptPreviewHasDictionaryDiagnostics`: false

## Executive Verdict

Worldbooks work with character cards in the core reliability sense. A worldbook can be attached to a character, survives reload, appears in the character preview, appears in the worldbook Attachments tab, and is injected into character prompt preview as a `lorebook` section with diagnostics.

Chat dictionaries work as chat-session text processing tools, including persisted assignment to global and workspace-scoped character chats. They do not work as character-card attachments. The UI and API make dictionaries a chat-level concern, and the character preview does not expose dictionary state. This is not inherently wrong, but it conflicts with a character-chat user's likely mental model: "Mira has a lorebook and a terminology dictionary."

Workspace reliability is mixed. A workspace-scoped character chat can carry dictionary settings once a real workspace exists, but the Workspace Playground does not expose worldbook or dictionary context controls. A raw workspace-scoped chat create with a nonexistent `workspace_id` failed with a 500 foreign-key error before the probe was corrected to create a workspace first.

## First-Time User Walkthrough

Goal: "I want Archivist Mira to know Lumen lore and understand EV as Echo Vault."

Observed path:

1. The character-focused route opens the character surface and preserves focus on the character preview.
2. The preview shows a `World Books` section, an attached worldbook chip, `Open workspace`, and `Back to World Books`.
3. The worldbook detail view, after row selection, shows the Echo Vault entry and the Attachments tab with Archivist Mira.
4. The Dictionaries page shows the terminology normalizer with `2 chats (2 active)`.
5. Quick Assign says "Choose chat sessions to link with this dictionary", not "characters" or "character cards".
6. The chat route is blocked by no-provider setup, so a first-time user cannot validate final assistant behavior in the UI.

First-time interpretation:

- Worldbook discoverability is now plausible from the character preview. The visible "World Books" section is the right conceptual anchor.
- Dictionary discoverability is weak for character-chat intent. A first-time user has to infer that dictionaries attach to chat sessions, not characters, and that assignment happens separately from the character card.
- The app gives no single "context health" answer for a character chat. The user can see lorebook attachment in one place, dictionary usage in another, and no combined context preview in chat.

## Regular Power User Walkthrough

Goal: "I maintain multiple characters and need fast, reliable context inspection."

Observed path:

1. API probe confirmed character-worldbook attachment persisted across reload.
2. API prompt preview inserted `World info` with the Echo Vault entry and returned matching diagnostics.
3. API dictionary processing replaced `EV` with `Echo Vault` explicitly by dictionary id.
4. API dictionary processing also replaced `EV` when run through active dictionaries with chat attribution.
5. Dictionary usage summary reported both the global and workspace chat.
6. The UI quick-assign modal only displayed global chats from the tested session, despite the usage count showing two active chats.
7. Workspace Playground showed general chat and source-grounded research controls but no visible worldbook or dictionary controls.

Power-user interpretation:

- The backend behavior is stronger than the UI verification surface.
- The power-user workflow lacks a compact "show me exactly what this character chat will send" view that includes both lorebook and dictionary transformations.
- Workspace chat scoping is under-surfaced. If a dictionary can be active for workspace-scoped chat, the assignment UI should make workspace scope legible and filterable.

## Findings

### P1: Dictionary State Is Not Character-Card Visible

Evidence:

- Character preview shows `World Books` and the attached worldbook.
- Character preview has no equivalent dictionary section.
- Dictionaries quick assign says "Choose chat sessions to link with this dictionary."
- API settings persist dictionary ids on chats, not on character cards.

Impact:

For character-chat users, the mental model is character-centered. Worldbooks match that model; dictionaries do not. Users may reasonably expect character cards to carry both lore and terminology context.

Recommendation:

Add a character-preview "Chat dictionaries" panel that distinguishes:

- Dictionaries linked to chats with this character.
- Dictionaries available globally/active by default.
- Whether dictionaries are inherited from a workspace chat scope.

Do not necessarily change the data model first. A read-only synthesized view would close most of the UX gap.

### P1: Prompt Preview Includes Lorebook Diagnostics But No Dictionary Diagnostics

Evidence:

- API prompt preview `sections` includes `lorebook` with Echo Vault content and diagnostics.
- API probe found `promptPreviewHasDictionaryDiagnostics: false`.
- Dictionary processing evidence exists only through `/api/v1/chat/dictionaries/process`.

Impact:

Power users cannot inspect the full context pipeline in one place. They can validate lorebook injection and dictionary substitution separately, but not the combined chat turn.

Recommendation:

Extend character prompt preview to include a dictionary section or diagnostic block:

- Active dictionary ids and names.
- Entries that would fire for the candidate user turn.
- Original vs transformed user text.
- Whether dictionary transforms happen before or after lorebook scanning.

### P1: Workspace-Scoped Dictionary Assignment Is Not Visible In Quick Assign

Evidence:

- API usage summary reported `used_by_chat_count: 2` and listed both global and workspace-scoped chats.
- Puppeteer quick-assign modal showed the global chat and an older global chat, but not the workspace chat created by the same probe.
- Workspace Playground did not expose dictionary controls.

Impact:

Workspace users cannot reliably audit or change dictionary assignment from the primary dictionary UI. This is a power-user reliability problem because usage count and assignable list disagree.

Recommendation:

Update quick assign to support workspace-scoped chats:

- Show scope badges: `Global`, `Workspace: <name>`.
- Include a scope filter.
- Include linked workspace chats in assignable and already-linked lists.
- Make `2 chats (2 active)` clickable to a complete usage detail view, not only a quick global chat path.

### P2: Worldbook Focus Links Do Not Fully Resolve To Detail State

Evidence:

- Direct `/world-books?from=characters&focusCharacterId=9&focusWorldBookId=2` showed the worldbook list and "Select a world book to view its entries and settings."
- Explicit row click was required before entries and attachments became visible.

Impact:

The character preview gives links that look like focused deep links, but the destination still requires the user to identify and click the row. This weakens cross-feature wayfinding.

Recommendation:

Honor `focusWorldBookId` on World Books route load by selecting the book, focusing the detail heading, and optionally opening the Attachments tab when `focusCharacterId` is present.

### P2: Workspace Chat Creation Returns 500 For Missing Workspace IDs

Evidence:

- Initial probe attempted workspace-scoped character chat creation with an arbitrary `workspace_id`.
- Backend failed with SQLite `FOREIGN KEY constraint failed` and HTTP 500.
- Creating the workspace first made workspace-scoped chat creation succeed.

Impact:

This is a reliability and error-recovery problem. A caller that supplies an invalid workspace id made a recoverable user/domain error look like server corruption or infrastructure failure.

Recommendation:

Validate `workspace_id` before insert and return 400 or 404 with a concrete message, for example: `Workspace ux-audit-... does not exist`.

### P2: Workspace Playground Does Not Surface Character Context Tools

Evidence:

- Workspace Playground shows general chat, RAG mode, sources, studio outputs, and character-related broad text.
- It does not expose visible worldbook, lorebook, dictionary, or prompt-context diagnostics in the captured state.

Impact:

For users who treat Workspace Playground as "work", character-card context behavior is opaque. They cannot see whether a workspace chat will use character lore or dictionary transforms.

Recommendation:

Add a compact context drawer in workspace chat:

- Character/persona identity.
- Attached worldbooks and matched entries for the draft turn.
- Active dictionaries and expected substitutions.
- Scope source: character, chat, workspace, global.

### P3: Worldbook And Dictionary Copy Mixes "Characters", "Chats", And "Activate"

Evidence:

- World Books header says knowledge bases that "characters and chats can reference."
- Character preview describes worldbooks as context during conversations with the character.
- Dictionary manager describes substitutions "across chats."

Impact:

The language is mostly accurate, but it makes the activation model harder to learn. First-time users need to know what attaches to a character card, what attaches to a chat, and what applies globally.

Recommendation:

Standardize labels around scope:

- `Character-attached lorebooks`
- `Chat-linked dictionaries`
- `Workspace chat scope`
- `Global active dictionaries`

### P3: Browser Console Noise Should Be Cleaned Up

Evidence:

- React unique key warning in WorldBookEntryManager/List detail path.
- Ant Design `useForm` disconnected warning in character preview flow.
- Ant Design `destroyOnClose` deprecation in Workspace Playground.
- Notification/event stream aborts during route navigation.

Impact:

No visible walkthrough failure, but noisy diagnostics reduce trust during future reliability work.

Recommendation:

Triage as cleanup after higher-risk context visibility issues.

## Reliability Matrix

| Capability | API result | UI result | Verdict |
| --- | --- | --- | --- |
| Attach worldbook to character card | Passed | Visible in character preview and attachments tab | Works |
| Worldbook entry matching | Passed | Entry visible in detail | Works |
| Lorebook prompt injection | Passed | Not visible in chat route, visible in prompt-preview API | Works, but needs UI diagnostics |
| Dictionary explicit processing | Passed | Dictionary entry and usage visible | Works |
| Dictionary chat assignment | Passed for global and workspace chat settings | Quick assign modal omitted workspace chat | Partially works |
| Dictionary prompt diagnostics | Not present in prompt preview | Not visible | Gap |
| Character card dictionary visibility | Not modeled | Not visible | Gap |
| Workspace chat context visibility | Settings persisted through API | Workspace Playground does not expose controls | Gap |
| Full model response with context | Not tested, no provider | No-provider blocker visible | Blocked |

## Recommended Work Packages

1. Character context summary surface:
   - Add read-only character preview context summary for attached worldbooks and linked/active dictionaries.

2. Unified prompt/context diagnostics:
   - Extend prompt preview to show worldbook and dictionary processing in one inspectable turn preview.

3. Workspace-scoped dictionary assignment:
   - Make quick assign include workspace chats, scope badges, and complete linked-chat detail.

4. Worldbook deep-link repair:
   - Honor `focusWorldBookId` and `focusCharacterId` by selecting the book and opening relevant detail state.

5. Workspace chat error hardening:
   - Return 400/404 for missing workspace ids instead of leaking SQLite FK failure as 500.

6. Context controls in Workspace Playground:
   - Add a compact context drawer for character, worldbook, and dictionary state in workspace chat.

7. Low-risk console cleanup:
   - Fix unique key, disconnected form, deprecated modal prop, and expected stream-abort noise.

## Verification

- API probe command: `node /private/tmp/tldw-worldbooks-dicts-audit-20260509/seed-and-probe.mjs`
- Puppeteer command: `node /private/tmp/tldw-worldbooks-dicts-audit-20260509/puppeteer-walkthrough.mjs`
- JSON validation target: `Docs/Reviews/assets/2026-05-09-character-card-worldbooks-dictionaries/api-probe.json`
- JSON validation target: `Docs/Reviews/assets/2026-05-09-character-card-worldbooks-dictionaries/puppeteer-states.json`
- Browser: Puppeteer/Chrome, not Computer Use.
- Full model response generation: not validated because the live chat route reported no LLM provider configured.
- Bandit: not applicable for this audit artifact because no Python files were changed.
