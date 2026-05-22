# PRD: First-Class Character Chat and Role-Play Workflow

Status: Draft, canonical phased PRD
Date: 2026-05-18
Last updated: 2026-05-20
Backlog: TASK-426
Post-Phase-6 update: TASK-455
Primary surface: `/chat`
Related surfaces: `/characters`, extension sidepanel chat, Persona Garden only where picker parity requires it
Evidence base: code inspection on `origin/dev` at `65430b962`, browser-backed UX audits in `Docs/Reviews`, existing terminology taxonomy in `Docs/Product/WebUI/Character_Chat_Terminology_Taxonomy_2026_05_09.md`, and post-Phase-6 real-backend review on `origin/dev` at `d2c0e5eaa`

## 1. Executive Summary

Character chat has a real backend and runtime path today, but the current `/chat` UI presents it as a generic Playground variant. Users can start character chat from starter cards, the header, the character library, and sidepanel context controls, but those paths do not converge into a first-class role-play workflow. The result is unnecessary ambiguity around character selection, prompt behavior, model readiness, parameter presets, scene setup, saved role-play setups, and conversation continuity.

This PRD restores Character Chat / Role-play as a first-class `/chat` experience while reusing the existing implementation seams:

- Existing character chat creation and streaming in `useChatActions` and `useMessage`.
- Existing assistant and character persistence through `useSelectedAssistant` and `useSelectedCharacter`.
- Existing server-chat metadata restoration through `useServerChatLoader`.
- Existing character-scoped history through `useServerChatHistory(filterMode: "character")`.
- Existing Role-play setup drawer, role-play state helpers, saved role-play setup bundles, and Characters page launch actions.

The product direction is not a separate competing chat system. It is a Character Chat mode inside `/chat`, with a dedicated IA, readiness model, session/history affordances, and extension parity.

As of the post-Phase-6 review, the foundation for Character Chat mode, role-play setup, saved setup safety, mobile behavior, session panel structure, and real-backend signoff has materially improved. This PRD now tracks the remaining post-Phase-6 remediation needed before the product should claim Character Chat / Role-play is first-class again.

## 2. Problem Statement

The original first-class Character Chat problem was that the product had the parts needed for character role-play, but they were spread across generic chat controls:

- `/chat` routed into generic `Playground`.
- The starter card opened the assistant picker but did not activate a durable role-play workspace.
- The header "Character" button could clear chat and open selection only when needed.
- The role-play setup surface was a drawer attached to composer controls.
- Character conversations could be filtered by server metadata, but `/chat` did not expose this as a first-class character-session rail.
- The extension sidepanel had character support, but it was buried under Conversation context rather than presented as a role-play mode.

Phase 0-6 work materially improved that baseline. The remaining problem is narrower and should not be treated as a greenfield role-play redesign:

- model/provider readiness can still disagree across status surfaces and allow invalid sends;
- WebUI character sessions can still receive extension-specific fallback titles;
- direct character-mode entry does not yet make last-used/session resume obvious enough;
- first-time users still lack create/import affordances at the point of character-chat intent;
- role-play setup is improved but still needs dependable primary desktop access;
- sidepanel character chat is supported but not yet first-viewport first-class;
- final confidence still requires real-backend signoff for the remaining failure and resume paths.

These issues keep character chat from feeling fully trustworthy even though the core mode, setup surface, sessions panel, and backend runtime now exist.

## 3. Goals

1. Make character role-play discoverable from `/chat` without requiring the user to understand Playground internals.
2. Provide a coherent first-time path: choose/create/import character, choose model, choose optional prompt/setup, type the first message.
3. Provide a fast returning-user path: resume recent character chats, switch character, switch model/preset, reuse saved setup, continue streaming.
4. Make Assistant, Prompt, Character, Persona, Scene, and Parameter Preset terminology understandable at the point of decision.
5. Preserve selected character, selected assistant, prompt, model, parameter preset, scene settings, and chat history across refresh, route changes, and session switches.
6. Keep extension behavior consistent with WebUI while respecting the sidepanel's narrow form factor.
7. Improve accessibility, keyboard navigation, screen-reader labels, mobile layout, error recovery, and destructive-action safety.
8. Require real backend verification for core workflows. Frontend mocks may support unit coverage, but signoff must use the running app against a running tldw backend.

## 4. Non-Goals

- Replacing the character backend, SillyTavern card import/export model, or character card schema.
- Redesigning the entire Playground, Characters page, Persona Garden, or global navigation.
- Turning Persona Garden into the same object as Characters. The taxonomy keeps them distinct.
- Building new LLM provider configuration flows beyond clearer handoff and recovery from model/provider failures.
- Adding marketplace, public sharing, or multi-user character collaboration.
- Removing existing power-user density from Playground. The goal is a dedicated role-play path, not simplification of every chat mode.
- Treating database recovery as a broad storage redesign. This PRD does include a narrow chat/character DB health release dependency because a corrupt per-user chat DB directly blocks `/chat` and character-chat continuity.

## 5. Evidence Summary

| Evidence | Current behavior | Product implication |
| --- | --- | --- |
| `apps/tldw-frontend/pages/chat/index.tsx` and `apps/packages/ui/src/routes/option-chat.tsx` | `/chat` dynamically loads generic `OptionChat`, which renders `Playground`. | First-class character chat must be added inside or directly around Playground, not as an unrelated route. |
| `PlaygroundEmpty.tsx` and `PlaygroundForm.tsx` | "Chat as a character" opens character selection and announces that the user should choose a character. | Starter path needs a durable Character Chat mode, not just a picker event. |
| `Header.tsx` and `ChatHeader.tsx` | Header Character action clears chat and opens selection only if no character is selected. | Header action should start or resume Character Chat mode and preserve intent. |
| `useChatActions.ts` and `useMessage.tsx` | Character chat creates server chats with `character_id`, persists greetings, and streams through `streamCharacterChatCompletion` with `include_character_context: true`. | Runtime is real and should be reused. The PRD should not invent a parallel runtime. |
| `useSelectedAssistant.ts`, `useSelectedCharacter.ts`, `useServerChatLoader.ts` | Assistant/character state persists and server chat metadata restores character/persona identity. | The new mode can rely on existing persistence, with explicit URL/session rules layered on top. |
| `useServerChatHistory.ts` | Supports `filterMode: "character"` and maps to server `character_scope`. | Character session/history rail can reuse existing APIs. |
| `RolePlaySetupDrawer.tsx` | Provides identity, behavior, generation style, scene notes, saved setups, and apply/save flows. | Use it as the basis for a first-class setup panel. |
| `startup-template-bundles.ts` and saved setup tests | Saved role-play setups exist as startup-template bundles. | Keep the bundle model, but expose setups separately from chat sessions. |
| `CharacterPreviewPopup.tsx`, `useCharacterCrud.tsx`, `useCharacterQuickChat.tsx` | `/characters` has stronger character entry points than `/chat`: Chat, Chat in new tab, quick test, conversations, model readiness blockers. | `/chat` should catch up and use the same readiness and selection concepts. |
| `ConversationContextPopover.tsx`, `CharacterSelect.tsx`, sidepanel chat route | Extension sidepanel supports character selection and backend character chat, but it sits under Conversation context. | Extension needs a compact first-class character state, not a hidden context-only affordance. |
| `Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_AUDIT_2026_05_09.md` | Earlier audit found a corrupt default per-user chat DB, generic onboarding, fragmented model readiness, row-chat context loss, terminology competition, dense creation flow, and misleading search counts. | This PRD must cover both existing residual issues and newer first-class `/chat` gaps. |
| `Docs/Reviews/CHARACTER_CHAT_WEBUI_UX_REAUDIT_2026_05_09.md` | P1 fixes improved onboarding and row-chat blockers, but search count, provider-backed send coverage, debugging noise, and final character-mode sequencing remained unresolved. | The phased path must include real backend E2E and no-provider/provider-failure recovery. |
| Post-Phase-6 real-backend review on `origin/dev` at `d2c0e5eaa` | `/chat` Character Chat mode is discoverable, character selection works, the greeting appears, and the mobile role-play setup drawer fits narrow width. The same run found contradictory model readiness, WebUI-created character sessions titled `Extension chat`, incomplete resume continuity, weak create/import affordance from the picker, fragile desktop setup access, sidepanel discoverability gaps, and insufficient send gating. | The remaining plan should extend the shipped Phase 0-6 track, not restart it. The next work must focus on truth, continuity, setup access, sidepanel parity, and real-backend release gates. |

## 6. User Journeys

### 6.1 First-Time Character Role-Play User

The user arrives at `/chat` with little knowledge of tldw. They see Character Chat as a visible mode, not a hidden card or toolbar option. Choosing it opens a readiness-oriented setup view:

1. Character: choose an existing character, create one, or import a character card.
2. Model: choose a chat-capable model or open model setup if none are available.
3. Prompt and behavior: choose an optional prompt or role-play behavior template.
4. Scene: optionally add setting, goals, or notes after a character is selected.
5. Composer: type the first message with clear status that the character, model, and persistence mode are ready.

If setup is incomplete, the user stays in context. The UI says exactly what is missing and offers local actions such as `Choose character`, `Open model settings`, `Retry with this character`, or `Continue without scene`.

### 6.2 Experienced Power User

The returning user enters `/chat`, toggles Character Chat, and can act quickly:

1. Resume the last character chat or pick from recent character sessions.
2. Switch character without losing the current session accidentally.
3. Apply a saved role-play setup, behavior template, model, prompt, or parameter preset.
4. Use keyboard shortcuts or command palette actions for frequent changes.
5. Keep streaming reliable and visible, with quick stop/retry/regenerate controls.
6. Recover from provider/model errors without losing the selected character, draft, or scene state.

The workflow should be denser for experienced users but predictable: the same controls stay in the same places, destructive state changes require confirmation or undo, and saved state is inspectable.

## 7. Product Requirements

R1-R11 remain the canonical product requirements for first-class Character Chat. Some are already partially or fully satisfied by Phase 0-6. Section 8 summarizes the shipped foundation, and Section 11 defines the remaining post-Phase-6 remediation work. Implementation planning should not reopen a completed requirement unless the latest branch reproduces a regression.

### R1. Add Character Chat As A First-Class `/chat` Mode

Character Chat mode must be visible from the `/chat` first viewport and header. It may be implemented as a mode within Playground, but it must behave like a primary workflow.

Requirements:

- Add a durable mode state for `standard`, `character`, and existing specialized chat modes where applicable.
- Support URL bootstrapping with a stable query contract, for example `/chat?mode=character&characterId=123`.
- When launched from `/characters`, preserve the selected character and route directly into Character Chat mode.
- When launched from the header or empty-state starter, open Character Chat mode before showing advanced scene controls.
- Do not clear an active chat without a confirmation or explicit new-session action.

Likely files:

- `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- `apps/packages/ui/src/components/Layouts/Header.tsx`
- `apps/packages/ui/src/components/Layouts/ChatHeader.tsx`
- `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterCrud.tsx`

### R2. Create A Character Chat Setup And Readiness Panel

Character Chat mode must show whether the session is ready and why not.

Readiness states:

- Server connection
- Character selected and available
- Model selected and available
- Prompt/system behavior loaded or intentionally empty
- Parameter preset applied or default
- Persistence mode: server saved, local, temporary, or blocked
- Optional scene state

Requirements:

- Use existing `buildCharacterChatReadiness` for the model/character/server baseline.
- Extend readiness UI to cover prompts, presets, scene, and persistence.
- Keep the user in character context when model/provider setup is missing.
- Show loading, empty, failed, unavailable, and ready states with actionable next steps.
- Make readiness messages screen-reader accessible with `role="status"` or `role="alert"` based on severity.

Likely files:

- `apps/packages/ui/src/utils/chat-model-availability.ts`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`
- `apps/packages/ui/src/components/Option/Characters/CharacterDialogs.tsx`

### R3. Promote Role-Play Setup From Drawer To Mode Surface

The existing Role-play setup drawer should become the source for a first-class setup surface. It can remain as a drawer on narrow screens or in compact mode, but Character Chat mode should not require hunting through toolbar overflow.

Requirements:

- Present identity, behavior, generation style, scene, saved setups, prompt, model, and parameter preset as a cohesive setup area.
- Sequence the default flow as Character first, Model second, Prompt/Behavior third, Scene optional.
- Retain the current advanced controls for scene notes, aspects, generation style, and template save/apply.
- Convert generation-style segmented choices to accessible radio semantics where needed.
- Show before/after preview for saved setup application.
- Add confirm or undo behavior for deleting saved role-play setups.

Likely files:

- `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`
- `apps/packages/ui/src/components/Option/Playground/SavedRolePlaySetupsPanel.tsx`
- `apps/packages/ui/src/components/Option/Playground/role-play-state.ts`
- `apps/packages/ui/src/components/Option/Playground/startup-template-bundles.ts`
- `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`

### R4. Clarify Assistant, Prompt, Character, Persona, Scene, And Parameter Preset

The mode must reduce concept switching at decision points.

Requirements:

- Follow `Character_Chat_Terminology_Taxonomy_2026_05_09.md`.
- Label character-backed conversations as `Character chat`.
- Label the picker as `Select character or persona` only when personas are present.
- Explain `Persona` only when the user can actually choose a persona.
- Frame `Scene` as optional context after a character is selected.
- Display Prompt and Parameter Preset as independent layers, not hidden advanced settings.
- Add empty/error states to Prompt selection and Assistant selection instead of silently hiding triggers or swallowing failures.

Likely files:

- `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
- `apps/packages/ui/src/components/Common/PromptSelect.tsx`
- `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx`

### R5. Add Character Chat Sessions And Continuity

Character sessions should be easy to resume and distinguish from generic chats.

Requirements:

- Add a Character Chat session rail or filtered history view using `useServerChatHistory(filterMode: "character")`.
- Show recent characters and recent character chats separately.
- Show selected character, chat title/topic, last updated, message count where available, and persistence state.
- On refresh or route return, restore server-chat character/persona identity from metadata before applying stored selected-character fallback.
- Keep saved role-play setups distinct from conversations.
- Provide safe actions: rename, duplicate/fork, archive, restore, delete, and open in new tab where current chat infrastructure supports them.

Likely files:

- `apps/packages/ui/src/hooks/useServerChatHistory.ts`
- `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`
- `apps/packages/ui/src/components/Common/ChatSidebar`
- `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
- `apps/packages/ui/src/components/Option/Characters/CharacterPreviewPopup.tsx`

### R6. Improve Composer Ergonomics For Role-Play

The composer must clearly show where to type and what identity/context will affect the response.

Requirements:

- In Character Chat mode, show a compact active-character chip near the composer.
- Show prompt, model, preset, and scene chips with clear remove/edit actions.
- Keep primary send, stop, retry, regenerate, continue-as-user, impersonate-user, and force-narrate actions accessible where supported.
- Add draft preservation across character switch confirmation flows.
- Prevent text/control overlap on narrow widths and when mobile keyboard is open.
- Keep power controls available in Pro mode without forcing first-time users through every knob.

Likely files:

- `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- `apps/packages/ui/src/components/Chat/composer/ChatComposer`
- `apps/packages/ui/src/hooks/chat/useChatActions.ts`

### R7. Improve Extension Parity

The extension sidepanel must reflect the same character state and terminology, scaled down for width.

Requirements:

- Add a visible Character Chat state/chip in sidepanel chat when a character or persona is active.
- Keep full Character Chat setup in WebUI, but expose compact choose/switch/clear/retry actions in sidepanel.
- Avoid hiding character selection only under Conversation context.
- Preserve sidepanel tab snapshots with selected character/persona and prompt/model/preset state.
- Make `Open full app` carry character chat intent and active conversation where possible.

Likely files:

- `apps/packages/ui/src/routes/sidepanel-chat.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/ConversationContextPopover.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx`
- `apps/tldw-frontend/extension/routes/sidepanel-chat.tsx`

### R8. Add Power-User Speed And Predictability

Requirements:

- Prioritize recents, focus order, resume, and quick-switch flows before adding new shortcuts.
- Add command palette actions for: switch character, open role-play setup, apply saved setup, resume last character chat, toggle scene, open prompt picker, open parameter preset picker where the command palette infrastructure already supports route-local actions.
- Add keyboard shortcuts or configurable shortcuts only after the target actions are stable and only if they do not collide with editor input.
- Add recents and favorites for characters and saved role-play setups.
- Add quick switcher search across character name, tags, description, and recent session titles.
- Preserve state across refreshes, route changes, and session switches with clear precedence rules.

Precedence rules:

1. Active server chat metadata is authoritative.
2. Explicit URL character intent applies only to new or empty chat state.
3. Stored selected assistant/character applies only when no active server chat identity exists.
4. Default character applies only to fresh chat state and only once per fresh session.

Likely files:

- `apps/packages/ui/src/hooks/keyboard`
- `apps/packages/ui/src/components/Common/CommandPalette`
- `apps/packages/ui/src/hooks/useSelectedAssistant.ts`
- `apps/packages/ui/src/hooks/useSelectedCharacter.ts`
- `apps/packages/ui/src/utils/default-character-preference.ts`

### R9. Improve Accessibility And Responsive Behavior

Requirements:

- Ensure mode switch, selectors, setup panel, saved setups, and session rail are keyboard navigable in logical order.
- Use semantic tabs, radio groups, menus, labels, and live regions.
- Ensure icon-only controls have accessible names and visible tooltips where needed.
- Avoid text truncation without title/tooltip on selected character, prompt, model, and preset labels.
- Support desktop, tablet, and narrow/mobile widths without horizontal overflow.
- Ensure setup and session panels collapse predictably on narrow widths.
- Verify color contrast for status chips, warnings, and disabled controls.

Likely files:

- Shared components under `apps/packages/ui/src/components/Common`
- Playground and sidepanel chat components listed above
- Design system state primitives where already available

### R10. Require Real Backend Verification

Requirements:

- Core acceptance must run against the real FastAPI backend and real frontend.
- Unit tests may mock components and API clients, but release signoff must include browser tests against a running backend.
- Message-send verification must use a real configured local/provider path wired through the backend provider path. It must not rely on frontend-only interception or simulated frontend responses as proof.
- If no real callable provider is available in the verification environment, mark successful-send signoff blocked and continue verifying no-provider, model-unavailable, provider-failure, and send-gating behavior against the real backend.
- Tests must verify that character context is included in the backend request/stream path.

Likely files:

- `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts`
- `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`

### R11. Track Chat/Character Database Health As A Release Dependency

Character Chat depends on the per-user chat/notes database. The earlier audit verified a malformed default `ChaChaNotes.db` that blocked backend startup. A first-class `/chat` workflow cannot be considered reliable if a corrupt per-user DB can make the whole app fail without clear recovery.

Requirements:

- Add or link a startup health check that identifies the affected per-user chat DB and failure reason. Backend gate: `/api/v1/health` exposes `checks.chacha_notes.last_failure` with a sanitized `affected_db`, `reason_code`, and recovery metadata when ChaChaNotes corruption is detected.
- Prefer quarantine or degraded startup where safe, so one corrupt per-user chat DB does not prevent reaching setup, diagnostics, or recovery UI. Backend gate: startup warm-up fails open and records a degraded ChaChaNotes health snapshot instead of aborting app startup.
- Provide a documented doctor/recovery path for backup, SQLite `.recover`, integrity validation, and restore. Recovery guide: `Docs/Operations/ChaChaNotes_DB_Recovery.md`.
- Surface user-facing recovery copy from setup or diagnostics without implying data was silently changed.
- Treat this as a release dependency for Character Chat GA, even if implemented in a separate backend-focused task.

Backend acceptance evidence:

- Corrupt existing per-user `ChaChaNotes.db` returns a sanitized 503 to character DB callers and records `last_failure.reason_code = sqlite_corruption`.
- `/api/v1/health` degrades to 206 with `checks.chacha_notes.last_failure.recovery.automatic_repair = false`.
- Health payloads do not expose absolute temp or host filesystem paths.
- Startup warm-up against a corrupt DB does not raise out of the warm-up path.

Likely files:

- `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
- `tldw_Server_API/app/api/v1/endpoints/health.py`
- `tldw_Server_API/app/services/startup_chacha_warmup.py`
- `Docs/Operations/ChaChaNotes_DB_Recovery.md`
- `tldw_Server_API/app/core/DB_Management`
- `tldw_Server_API/app/core/Character_Chat`
- `tldw_Server_API/app/main.py`
- `apps/packages/ui/src/components/Common/WorkspaceConnectionGate.tsx`
- `apps/packages/ui/src/routes/option-setup.tsx`

## 8. Completed Foundation Through Phase 6

The original Phase 0-6 track is treated as shipped foundation, not as a second active roadmap. Future work should not recreate these phases unless a regression proves that a shipped capability no longer works.

| Phase | Shipped foundation | Representative tracking |
| --- | --- | --- |
| Phase 0 | Baseline contracts and real-backend harness for character chat. | TASK-428 |
| Phase 1 | Durable Character Chat mode and route intent for `/chat`. | TASK-431 |
| Phase 2 | Readiness, errors, empty states, and selected-character recovery. | TASK-438 |
| Phase 3 | Role-play setup surface, saved setup workflows, delete safety, and generation-style accessibility. | TASK-446, TASK-447 |
| Phase 4 | Character sessions, recents, and continuity panel. | TASK-449 |
| Phase 5 | Power-user and extension parity foundations, including sidepanel handoff. | TASK-452 |
| Phase 6 | Accessibility, mobile behavior, scoped stream/persist fixes, and real-backend signoff. | TASK-454, PR #1892 |

The post-Phase-6 plan below extends this foundation. It should use Phase 7-13 numbering so future agents do not confuse new remediation work with already-merged Phase 0-6 implementation slices.

## 9. Post-Phase-6 Real-Backend Findings

The latest review ran against `origin/dev` at `d2c0e5eaa` with a real FastAPI backend and the real Next WebUI. The backend health check passed, default characters were created by the server, character selection created real server chats, and the send attempt used the real `/api/v1/chats/{id}/complete-v2` path.

Positive evidence:

- `/chat` now exposes Character Chat / role-play from the first viewport.
- Character mode opens a real character picker and can select server-backed characters.
- Selecting `Helpful AI Assistant` injects the greeting and creates a saved server chat.
- `RolePlaySetupDrawer` is usable on narrow/mobile width without horizontal overflow.
- `CharacterChatSessionsPanel` exists and separates character sessions from saved role-play setups.
- Sidepanel code has handoff support to full `/chat`.

Remaining findings:

| Severity | User type | Evidence | Why it matters | Requirement | Suggested acceptance test |
| --- | --- | --- | --- | --- | --- |
| P1 | First-time and power users | Readiness says choose a chat model while the model selector reports `OpenAI / gpt-4o Healthy`; SEND can still call real `/complete-v2` and receive `503`. | Users cannot trust setup state and are led into avoidable failure. | R2, R6, R10 | Browser: selected-character plus no usable model disables or gates SEND; no `/complete-v2` request is made; all status surfaces agree. |
| P1 | Power users | Real WebUI-created character sessions were titled `Extension chat`; fallback strings exist in persistence/title paths and locale defaults. | Recents and history are hard to reuse and look like the wrong surface created them. | R5 | Component and real-backend browser: WebUI character chat titles include character name or first prompt, never `Extension chat` unless true extension context is the only source. |
| P1 | Power users | Direct `/chat?mode=character` enters character mode but does not foreground last-used character/session. | Returning users must rebuild context instead of resuming. | R5, R8 | Browser: direct character mode shows `Resume last character chat` or restores explicit route/server chat identity using documented precedence. |
| P1 | First-time and power users | Desktop setup access did not reliably open the setup drawer after selection in the automated walkthrough; mobile setup did open. | Role-play setup remains discoverable but not dependable as a primary control. | R3, R9 | Playwright: after desktop character selection, one click on Role-play setup opens the drawer/panel, traps focus, and restores focus on close. |
| P2 | First-time users | Character picker exposes existing characters/personas but no visible create/import affordance in the role-play path. | A new role-play user cannot bring in a character at the moment of intent. | R1, R4 | Component/browser: picker footer and no-results state include `Create character` and `Import character card`, returning to `/chat?mode=character&characterId=...`. |
| P2 | First-time users | Character mode still competes with dense cockpit rails, model/runtime controls, and advanced setup concepts. | The first-time path requires too much architecture knowledge. | R2, R3, R6 | Browser: first-time character mode foregrounds Character, Model, Message, and optional Role-play setup before advanced rails. |
| P2 | First-time and power users | Generic server-error recovery appears after a model/provider setup failure. | Users retry impossible states instead of fixing configuration. | R2, R10 | Browser: provider/model setup failures show model-specific recovery, not generic retry guidance. |
| P2 | Extension users | Sidepanel first viewport does not present Character Chat as first-class unless state is already active. | Extension users do not see the same workflow as WebUI users. | R7 | Sidepanel QA: initial chat surface exposes compact Character Chat state/entry and full-app handoff. |
| P3 | Keyboard users | Character-mode focus order prioritizes general composer tools before character/setup/model/preset controls. | Power users need avoidable tabbing for core role-play changes. | R8, R9 | RTL/browser: from composer in character mode, core role-play controls are reachable before secondary attachment/tool controls. |

## 10. Model Usability Contract

The next implementation must not treat `selectedModel`, model catalog presence, provider label, provider health, and callable send readiness as interchangeable. Character Chat needs one model usability contract that feeds all status surfaces and send gating.

Required states:

| State | Meaning | Primary action |
| --- | --- | --- |
| `loading` | Model/provider data is still hydrating. | Wait or refresh. |
| `no_server` | The WebUI cannot reach tldw_server. | Connect to server. |
| `no_selection` | No chat model is selected. | Choose model. |
| `selected_missing` | Stored selected model is not present in the current model catalog. | Choose a different model. |
| `provider_unconfigured` | Model is known, but the provider lacks required configuration or API key. | Open model/provider settings. |
| `model_unavailable` | Model/provider is known but not currently callable. | Refresh, switch model, or inspect provider. |
| `ready` | Selected model is known and callable for character chat. | Enable send. |
| `degraded` | Model is callable but has a known limitation relevant to the current role-play mode. | Allow send with a specific notice. |

Contract requirements:

- The readiness panel, status strip, model selector chip, composer SEND gate, and error recovery must consume the same model usability result.
- Users can always type and preserve drafts. Only SEND is blocked or converted into a setup action when readiness is invalid.
- A blocked send must not call `/complete-v2`.
- Provider/model setup errors must not be presented as generic server failures when the client can classify them.
- The UI must never show a selected model as `Healthy` while Character Chat readiness says that no chat model is usable.

## 11. Post-Phase-6 Remediation Phases

### Phase 7: Model Usability, Readiness Truth, And SEND Gating

Goal: make setup state truthful before the user sends.

Scope:

- Add the model usability contract described above.
- Unify model/provider readiness across readiness panel, status strip, model selector, runtime inspector, and SEND eligibility.
- Gate character SEND when model usability is not `ready` or explicitly allowed `degraded`.
- Preserve drafts and selected character while blocking invalid sends.
- Classify provider/model setup failures into actionable recovery states.

Acceptance criteria:

- No-provider state never shows `Healthy` in any character-chat status surface.
- Character selected plus no usable model disables SEND or turns it into a setup action.
- Blocked SEND does not create a `/complete-v2` request.
- Error recovery names the missing/unconfigured model/provider state and offers a local fix.

Likely files:

- `apps/packages/ui/src/utils/chat-model-availability.ts`
- `apps/packages/ui/src/components/Option/Playground/CharacterChatReadinessPanel.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
- send-control and character chat action hooks

### Phase 8: Character Session Naming And Resume Continuity

Goal: make character chat history understandable and resumable.

Scope:

- Replace WebUI-created `Extension chat` fallback titles with character-aware WebUI titles.
- Audit all title fallback paths, including persistence, message branching, locale defaults, and sidepanel-specific creation.
- Show character name, chat title/topic, updated time, message count, and persistence state in recent character sessions.
- Add a primary `Resume last character chat` action when entering character mode with no explicit active chat.
- Preserve selected character/session intent across refresh and direct `/chat?mode=character`.

State precedence:

1. Active server chat metadata is authoritative.
2. Explicit URL `chatId` applies next.
3. Explicit URL `characterId` applies only to new or empty chat state.
4. User-selected last-used character/session applies only when no explicit route/server state exists.
5. Default character applies only to fresh chat state and only once per fresh session.

Acceptance criteria:

- WebUI character chats are not titled `Extension chat` unless they actually came from extension context and no better title exists.
- Recent sessions are distinguishable by character and topic or first user prompt.
- Returning to `/chat?mode=character` foregrounds last character chat or an explicit resume choice.
- Switching sessions does not leak prior character, prompt, scene, or generation style state.

Likely files:

- `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundPersistence.tsx`
- `apps/packages/ui/src/hooks/handlers/messageHandlers.ts`
- `apps/packages/ui/src/components/Option/Playground/CharacterChatSessionsPanel.tsx`
- `apps/packages/ui/src/hooks/useServerChatHistory.ts`
- `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`
- route/query state helpers and relevant locale defaults

### Phase 9: First-Time Create/Import Entry From `/chat`

Goal: let a new role-play user create, import, or select a character from the point of intent.

Scope:

- Add `Create character` and `Import character card` affordances to the character picker footer.
- Add the same actions to empty and no-results states.
- Reuse existing `/characters` create/import flows instead of redesigning the character editor.
- Return to `/chat?mode=character&characterId=...` after create/import.
- Keep first-time character mode focused on Character, Model, Message, and optional Role-play setup.

Acceptance criteria:

- A default-only or empty character library still gives the user an obvious way forward.
- A newly created/imported character returns to `/chat` selected and subject to the same model readiness checks.
- The create/import handoff preserves any draft when technically feasible, or explicitly warns before leaving.

Likely files:

- `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
- character create/import launch utilities
- `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterCrud.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`
- route-path helpers

### Phase 10: Role-Play Setup Access And Desktop Reliability

Goal: make Role-play setup a primary control, not a fragile toolbar detail.

Scope:

- Place `Role-play setup` beside the selected character/readiness area in character mode.
- Ensure the character picker closes after selection and cannot obstruct setup actions.
- Add deterministic desktop coverage that setup opens after character selection.
- Preserve current mobile drawer behavior that already fits narrow width.
- Ensure setup drawer/panel focus trap and focus restoration are tested.

Acceptance criteria:

- After selecting a character on desktop, one click opens Role-play setup.
- The setup surface opens from the same semantic action on desktop and mobile.
- Focus is trapped while open and restored to the invoking control on close.
- No character picker popover remains layered over setup controls after character selection.

Likely files:

- `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx`
- `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`
- `apps/packages/ui/src/components/Option/Playground/CharacterChatReadinessPanel.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- setup drawer tests and browser smoke tests

### Phase 11: Power-User Focus Order, Recents, And Persistence

Goal: reduce repeat-work friction for regular role-play users without burying first-time users in controls.

Scope:

- Improve keyboard order in character mode: character, role-play setup, model, prompt/behavior, generation style, then secondary attachment/tool controls.
- Persist recent character, prompt/template, model, generation style, and saved setup choices where existing storage supports it.
- Add quick switch/change actions for character, behavior template, generation style, and saved setup.
- Use recents before configurable shortcuts. Configurable shortcuts are optional unless command infrastructure already exists.
- Confirm or preserve state when switching would discard a draft or active session.

Acceptance criteria:

- From the composer in character mode, core role-play controls are reached before secondary tools.
- Frequent users can resume or switch without rebuilding setup from scratch.
- Drafts and selected character are not lost silently.
- Any new shortcuts avoid collisions with text entry and remain discoverable.

Likely files:

- composer toolbar and overflow controls
- `RolePlaySetupDrawer.tsx`
- saved setup helpers
- selected character/model/prompt stores
- keyboard/focus tests

### Phase 12: Extension Sidepanel First-Class Character State

Goal: make extension-side character chat visible and consistent without cloning the full WebUI setup.

Scope:

- Add compact Character Chat state/chip to the sidepanel first viewport.
- Expose choose, switch, clear, and retry actions without hiding them only under Conversation context.
- Ensure `Open full app` carries `mode=character`, selected character, and active conversation when possible.
- Keep terminology aligned with WebUI: Character Chat, Role-play setup, Scene, Generation style.
- Do not implement the full role-play setup drawer inside the sidepanel.

Acceptance criteria:

- A sidepanel user can discover character chat without already knowing it lives in Conversation context.
- Handoff to WebUI preserves role-play intent.
- Sidepanel remains compact and does not add a parallel setup workflow.
- Sidepanel tab snapshots preserve selected character/persona state.

Likely files:

- `apps/packages/ui/src/routes/sidepanel-chat.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/ConversationContextPopover.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx`
- sidepanel full-app route helpers

### Phase 13: Real-Backend Release Signoff Matrix

Goal: prevent another "passes in mocked tests, fails in the real app" loop.

Scope:

- Add a repeatable signoff matrix for real backend plus real WebUI.
- Required scenarios: backend health, no-provider, configured-provider where available, character selection, send gating, successful or explicitly blocked completion, provider failure, resume, refresh, mobile setup, and sidepanel handoff.
- Keep simulated/unit tests for coverage, but never use them as final proof.
- Store screenshots, request logs, or concise browser artifacts with each signoff PR.
- Sidepanel signoff should use the packaged extension when practical. The `/__debug__/sidepanel-chat` route is acceptable for component-level reproduction, but it is not sufficient by itself for final extension parity claims.

Acceptance criteria:

- Each phase records focused Vitest coverage plus real-browser evidence when behavior is visible.
- Final first-class signoff cannot pass unless `/api/v1/health`, `/chat`, and sidepanel flows are verified against a running backend.
- If no real callable provider is configured, the successful-send scenario is explicitly marked blocked, while no-provider/send-gating remains verified. Frontend interception is not accepted as proof of successful character completion.

Signoff matrix:

| Scenario | Required evidence |
| --- | --- |
| Backend health | `/api/v1/health` succeeds before testing. |
| No provider | UI shows one consistent unavailable state. |
| Character selected | Greeting appears and chat is saved with character metadata. |
| Send blocked | No `/complete-v2` request is made. |
| Successful completion | A real configured backend provider returns a character response, or this scenario is explicitly marked blocked because no real callable provider is available. |
| Provider failure | Error copy maps to model/provider recovery. |
| Resume | Recent session title is character-aware. |
| Refresh | Character/session intent is preserved or explicitly recoverable. |
| Mobile | No horizontal overflow; setup remains usable. |
| Sidepanel | Character entry is visible and full-app handoff preserves intent. |

## 12. Release Dependencies

The following items are release dependencies for declaring Character Chat first-class. They should not be mixed into the post-Phase-6 UX remediation sequence unless they directly fail the current `/chat` flow.

1. **ChaChaNotes DB health and recovery**: keep R11 as a backend release dependency with a dedicated owner and task. It is valid and important, but it should not block documenting or implementing the post-Phase-6 UX remediation phases unless the live backend health check fails.
2. **Real callable provider availability**: successful message-send signoff requires a real configured local/provider path. If none is available, document that single scenario as blocked and continue verifying no-provider, model-unavailable, provider-failure, and send-gating behavior.
3. **No parallel runtime**: all remediation must reuse existing Character Chat runtime, stream, persist, and history contracts.
4. **Sidepanel scope boundary**: extension parity means compact state and handoff, not a full sidepanel setup clone.

## 13. Quick Wins Under One Day

1. Add a model usability helper and wire it to one visible status surface plus SEND gating tests.
2. Change WebUI character-chat fallback titles away from `Extension chat`.
3. Add `Create character` and `Import character card` actions to the picker footer and no-results state.
4. Move or duplicate `Role-play setup` next to the selected-character/readiness area in character mode.
5. Add a desktop regression test proving setup opens after character selection.
6. Add a sidepanel first-viewport Character Chat chip or entry row when character state is inactive.
7. Add a focused tab-order test for character-mode core controls.

## 14. Larger Improvements

1. Model usability contract consumed by all chat readiness/status/send surfaces.
2. Character-aware session naming and resume flow across WebUI and sidepanel-created chats.
3. First-time create/import handoff from `/chat` into existing `/characters` flows and back.
4. Role-play setup access model that works identically on desktop and narrow/mobile layouts.
5. Power-user recents, quick switchers, and optional command-palette actions after continuity is reliable.
6. Real-backend signoff harness that can distinguish no-provider, provider-unconfigured, provider-failure, and success scenarios.

## 15. Telemetry And Success Metrics

No external telemetry is required. If local analytics or internal event logs exist, capture only local/self-hosted usage signals.

Suggested product metrics:

- First-time success: user reaches a ready or clearly blocked Character Chat composer from `/chat` without losing selected character intent.
- Time to first ready state: target under 90 seconds when a model is already configured.
- Returning-user resume speed: resume recent character chat in two clicks or fewer.
- Error recovery: missing model/provider states preserve selected character and draft in 100% of tested flows.
- Session clarity: recent character chats show character-aware titles in 100% of WebUI-created character sessions.
- Accessibility: zero known critical keyboard or screen-reader blockers in Character Chat mode.
- Reliability: real backend E2E passes create/select/gate/send-or-explicitly-block/resume workflow.

## 16. Open Questions

1. Should successful-send signoff standardize on a local OpenAI-compatible provider profile, or remain environment-dependent with explicit blocked notation when absent?
2. Should URL intent support `chatId` in addition to `mode=character&characterId=...` for resume links?
3. Should saved role-play setups remain local storage bundles, or eventually become server-backed assets?
4. Should "continue as user", "impersonate user", and "force narrate" be visible by default in Character Chat mode or remain in Pro/advanced controls?
5. Should character-specific parameter presets be supported later, or should generation styles remain global in this PRD?

## 17. Release Gates

- Product: Character Chat is visible and coherent as a first-class `/chat` workflow after the post-Phase-6 remediation findings are resolved or explicitly deferred.
- Engineering: implementation reuses existing backend/runtime seams and does not create a parallel character chat stack.
- QA: real backend signoff covers health, no-provider, selected character, send gating, provider error, resume, refresh, mobile setup, and sidepanel handoff.
- UX: first-time and power-user walkthroughs pass on desktop and narrow widths.
- Accessibility: keyboard and screen-reader checks pass for primary setup and send flow.
- Safety: destructive actions have confirm or undo, state-changing mode switches are explicit, and invalid sends do not hit `/complete-v2`.

## 18. Recommended First Fix

Fix Phase 7 first: model usability, readiness truth, and SEND gating.

Reason: the post-Phase-6 browser review found that the UI can simultaneously claim no usable model, show a model as healthy, and still allow a real failing `/complete-v2` send. That is the highest-trust defect. Session naming and resume should follow immediately after because they determine whether power users can reliably reuse the improved workflow.
