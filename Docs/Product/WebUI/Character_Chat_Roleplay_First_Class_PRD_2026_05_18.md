# PRD: First-Class Character Chat and Role-Play Workflow

Status: Draft
Date: 2026-05-18
Backlog: TASK-426
Primary surface: `/chat`
Related surfaces: `/characters`, extension sidepanel chat, Persona Garden only where picker parity requires it
Evidence base: code inspection on `origin/dev` at `65430b962`, browser-backed UX audits in `Docs/Reviews`, existing terminology taxonomy in `Docs/Product/WebUI/Character_Chat_Terminology_Taxonomy_2026_05_09.md`

## 1. Executive Summary

Character chat has a real backend and runtime path today, but the current `/chat` UI presents it as a generic Playground variant. Users can start character chat from starter cards, the header, the character library, and sidepanel context controls, but those paths do not converge into a first-class role-play workflow. The result is unnecessary ambiguity around character selection, prompt behavior, model readiness, parameter presets, scene setup, saved role-play setups, and conversation continuity.

This PRD restores Character Chat / Role-play as a first-class `/chat` experience while reusing the existing implementation seams:

- Existing character chat creation and streaming in `useChatActions` and `useMessage`.
- Existing assistant and character persistence through `useSelectedAssistant` and `useSelectedCharacter`.
- Existing server-chat metadata restoration through `useServerChatLoader`.
- Existing character-scoped history through `useServerChatHistory(filterMode: "character")`.
- Existing Role-play setup drawer, role-play state helpers, saved role-play setup bundles, and Characters page launch actions.

The product direction is not a separate competing chat system. It is a Character Chat mode inside `/chat`, with a dedicated IA, readiness model, session/history affordances, and extension parity.

## 2. Problem Statement

The current UI has the parts needed for character role-play, but they are spread across generic chat controls:

- `/chat` routes into generic `Playground`.
- The starter card opens the assistant picker but does not activate a durable role-play workspace.
- The header "Character" button clears chat and opens selection only when needed.
- The role-play setup surface is a drawer attached to composer controls.
- Character conversations can be filtered by server metadata, but `/chat` does not expose this as a first-class character-session rail.
- The extension sidepanel has character support, but it is buried under Conversation context rather than presented as a role-play mode.

This makes character chat feel secondary even though the backend and earlier product behavior treat it as a core feature.

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

- Add command palette actions for: switch character, open role-play setup, apply saved setup, resume last character chat, toggle scene, open prompt picker, open parameter preset picker.
- Add keyboard shortcuts or configurable shortcuts for the most common actions, with no collisions against editor input.
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
- Message-send verification must use either a configured local model provider or an OpenAI-compatible deterministic provider wired through the backend provider path. It must not rely on frontend-only interception as the only proof.
- Tests must verify that character context is included in the backend request/stream path.

Likely files:

- `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts`
- `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- `mock_openai_server`
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`

### R11. Track Chat/Character Database Health As A Release Dependency

Character Chat depends on the per-user chat/notes database. The earlier audit verified a malformed default `ChaChaNotes.db` that blocked backend startup. A first-class `/chat` workflow cannot be considered reliable if a corrupt per-user DB can make the whole app fail without clear recovery.

Requirements:

- Add or link a startup health check that identifies the affected per-user chat DB and failure reason.
- Prefer quarantine or degraded startup where safe, so one corrupt per-user chat DB does not prevent reaching setup, diagnostics, or recovery UI.
- Provide a documented doctor/recovery path for backup, SQLite `.recover`, integrity validation, and restore.
- Surface user-facing recovery copy from setup or diagnostics without implying data was silently changed.
- Treat this as a release dependency for Character Chat GA, even if implemented in a separate backend-focused task.

Likely files:

- `tldw_Server_API/app/core/DB_Management`
- `tldw_Server_API/app/core/Character_Chat`
- `tldw_Server_API/app/main.py`
- `apps/packages/ui/src/components/Common/WorkspaceConnectionGate.tsx`
- `apps/packages/ui/src/routes/option-setup.tsx`

## 8. Prioritized Issue And Requirement Matrix

| Severity | User type | Evidence | Why it matters | Requirement | Suggested acceptance test |
| --- | --- | --- | --- | --- | --- |
| P0 | First-time and returning users | Earlier audit verified malformed default `ChaChaNotes.db` blocking backend startup. | Character chat cannot be first-class if local chat data failure prevents reaching recovery. | R11 | Backend integration: corrupt per-user chat DB is identified and recoverable without silent overwrite. |
| P0 | First-time and power users | Real runtime exists, but `/chat` is generic Playground. | Character role-play feels secondary despite being a core capability. | R1 | Browser: `/chat` exposes Character Chat mode in first viewport and can enter setup without leaving `/chat`. |
| P0 | First-time users | Starter only opens picker event. | Setup sequence is incomplete and easy to lose. | R1, R2 | Vitest: starter sets Character Chat mode, opens character step, and returns focus predictably. |
| P0 | Power users | Character streaming path exists with `character_id` and `include_character_context`. | A PRD that invents new runtime would increase risk. | R10 | Real backend E2E: send one character message and verify backend stream path includes character context. |
| P1 | First-time users | Model/provider blockers appear across different surfaces. | Users do not know what to fix before typing. | R2 | Browser: no-provider state keeps selected character and shows model action, retry, and return controls. |
| P1 | First-time users | Role-play setup is a drawer attached to generic toolbar. | Scene and behavior setup feels advanced or hidden. | R3 | Browser: Character Chat mode shows setup/readiness without opening toolbar overflow first. |
| P1 | First-time users | Terminology mixes Assistant, Character, Persona, Companion, Scene, Actor. | Users must learn internal architecture before acting. | R4 | Text snapshot tests: user-facing labels follow taxonomy at picker/setup/runtime decision points. |
| P1 | Power users | Character-scoped history API exists but not surfaced as character sessions. | Resume workflows are slower than they need to be. | R5 | Vitest: character history rail queries `filterMode: "character"` and renders recent sessions. |
| P1 | Power users | Header Character action can clear current chat. | Hidden destructive state changes damage trust. | R1, R8 | Vitest: active chat character switch requires confirm or explicit new-session action. |
| P1 | Extension users | Sidepanel selection sits inside Conversation context. | Extension role-play does not match WebUI mental model. | R7 | Browser extension test: sidepanel shows active character chip and compact switch/clear action. |
| P2 | First-time users | PromptSelect can render nothing when prompt query has no data. AssistantSelect swallows catalog load errors in common path. | Missing controls look like unavailable features instead of recoverable states. | R4 | Component tests: prompt and assistant selectors expose loading, empty, and error states. |
| P2 | Power users | Saved role-play setup delete is immediate in panel. | Deleting reusable setup has weak recovery. | R3 | Component test: delete saved setup requires confirm or offers undo. |
| P2 | Power users | Search count issue remains from audit. | Status feedback becomes untrustworthy. | R5 | Characters test: filtered count reads `1 of N shown` and updates live region. |
| P2 | All users | Mobile wraps dense controls, current audit found no overflow but high density. | Usability degrades on narrow widths. | R6, R9 | Playwright: 390px viewport has no horizontal overflow and setup controls remain reachable. |
| P2 | Screen-reader and keyboard users | Some segmented options use pressed buttons, not radio semantics. | Assistive tech receives weak state semantics. | R3, R9 | RTL: generation style controls expose radiogroup/radio or equivalent accessible state. |
| P2 | Engineers and QA | No-provider environments prevent final send verification. | Role-play can regress without deterministic real-backend tests. | R10 | CI or local E2E profile runs with backend-wired deterministic provider. |
| P2 | Engineers and QA | Re-audit saw repeated form warnings and route-churn request abort noise. | Debugging signal is weaker during role-play UX verification. | R10 | Browser E2E logs expected navigation aborts separately and fails unexpected console errors in touched flows. |
| P3 | Power users | Shortcuts exist generally but no role-play-specific command set. | Frequent switch/setup workflows are slower than necessary. | R8 | Shortcut tests: command palette exposes switch character, saved setup, and role-play setup actions. |

## 9. Phased Implementation Plan

### Phase 0: Baseline Contracts And Test Harness

Goal: Freeze the existing behavior and prove the implementation will reuse real backend seams.

Scope:

- Add or update route/mode contract tests for `/chat`.
- Add fixture coverage for `useServerChatHistory(filterMode: "character")`.
- Document state precedence rules for server chat metadata, URL intent, stored selected assistant, and default character.
- Add deterministic real-backend E2E plan for character message send.
- Confirm whether R11 is covered by an existing backend recovery task or open a separate dependency before GA work starts.

Acceptance criteria:

- Tests verify current character chat creation and streaming payload shape at the service/hook level.
- A real-backend E2E profile is documented and runnable locally.
- The per-user chat DB corruption blocker has an owner, linked task, and release-gate decision.
- No production behavior changes except optional test IDs or diagnostics needed for reliable tests.

### Phase 1: First-Class Character Chat Mode

Goal: Make `/chat` visibly support Character Chat as a primary mode.

Scope:

- Add durable Character Chat mode state to Playground.
- Update starter card and header action to enter Character Chat mode.
- Add URL bootstrapping for `mode=character` and optional `characterId`.
- Update `/characters` Chat and Chat in new tab actions to open `/chat` with character intent where applicable.
- Prevent implicit clear on active chat unless confirmed.

Acceptance criteria:

- First-time user can open `/chat`, choose Character Chat, select/create/import a character, and see where to type.
- Returning user can launch a character from `/characters` into `/chat` without losing task context.
- Active chat is not cleared by Character Chat action without explicit confirmation.

### Phase 2: Readiness, Errors, And Empty States

Goal: Make incomplete setup, loading, no-provider, and failure states local and actionable.

Scope:

- Add Character Chat readiness panel.
- Extend model/provider readiness copy for selected character context.
- Add explicit PromptSelect and AssistantSelect loading/error/empty states.
- Add persistence state messaging for server/local/temporary chat.
- Handle deleted/missing character on restored chat with clear recovery options.

Acceptance criteria:

- Missing server, missing character, missing model, unavailable model, prompt load failure, and character catalog failure all have visible local states.
- Selecting a character before model setup preserves the character through settings handoff and retry.
- Screen readers receive setup status changes through appropriate live regions.

### Phase 3: Role-Play Setup And Saved Setup Workflow

Goal: Turn the existing Role-play setup drawer into a first-class setup experience.

Scope:

- Add setup panel variant for desktop Character Chat mode.
- Keep drawer behavior for compact/mobile surfaces.
- Expose character/persona identity, behavior template, prompt, generation style, parameter preset, scene, and saved setups in one coherent surface.
- Add saved setup preview, apply, rename, delete confirm/undo, and save-current flows.
- Clarify behavior between saved role-play setup and saved conversation.

Acceptance criteria:

- User can save current role-play setup, apply it to a fresh chat, preview changes, and undo or confirm destructive setup deletion.
- Scene controls are optional and appear after character selection.
- Generation style controls meet keyboard and screen-reader expectations.

### Phase 4: Character Sessions, Recents, And Continuity

Goal: Make continuing role-play sessions fast and safe.

Scope:

- Add Character Chat sessions rail or filtered history panel using character-scoped history.
- Show recent characters and recent character chats.
- Restore server chat character/persona identity before stored selected state.
- Add session actions: rename, duplicate/fork if supported, archive, restore, delete with existing safety patterns.
- Add state persistence tests across refresh, route change, and chat switch.

Acceptance criteria:

- Returning user can resume a recent character chat in two clicks or fewer from `/chat`.
- Refreshing a character-backed server chat restores character identity and composer context.
- Switching from one character chat to another does not leak old character/prompt/scene state.

### Phase 5: Power-User Controls And Extension Parity

Goal: Make frequent role-play work fast across WebUI and extension.

Scope:

- Add command palette actions for role-play operations.
- Add shortcut support where it does not conflict with text entry.
- Add favorites/recents for characters and role-play setups.
- Add compact sidepanel active-character chip and switch/clear/retry actions.
- Carry intent from sidepanel `Open full app` into `/chat`.

Acceptance criteria:

- Power user can switch character, apply saved setup, and resume last character chat without opening settings.
- Extension sidepanel and WebUI use the same visible labels for Character, Persona, Scene, and Character Chat.
- Sidepanel tab switching preserves selected character/persona state.

### Phase 6: Accessibility, Mobile, And Real Backend Signoff

Goal: Close the UX quality gates before release.

Scope:

- Keyboard focus pass for mode switch, setup panel, session rail, selectors, composer, and dialogs.
- Screen-reader label and live-region pass.
- Contrast and density pass for status chips and disabled states.
- Mobile/narrow responsive pass at 390px, 768px, and desktop.
- Real backend E2E with deterministic provider or configured local provider.
- Update user docs and release notes.

Acceptance criteria:

- Playwright captures pass for desktop and narrow layouts without horizontal overflow.
- Keyboard-only user can complete first-time character setup and send path.
- Real backend E2E verifies create/select/send/resume for character chat.
- Chat/character DB health dependency is either resolved or explicitly release-blocked with an owner.
- No known P0/P1 issues remain open.

## 10. Quick Wins Under One Day

1. Make the `/chat` empty-state Character card enter a persistent Character Chat mode flag before opening the picker.
2. Change header Character button behavior to avoid silent chat clear and show a confirm/new-session choice when needed.
3. Add visible loading/error/empty states to PromptSelect and AssistantSelect.
4. Add confirm or undo to saved role-play setup deletion.
5. Update labels to match the terminology taxonomy in the current picker/setup/header surfaces.
6. Add a first-pass character readiness summary using existing `buildCharacterChatReadiness`.
7. Add direct `Open Role-play setup` command palette action if command infrastructure already supports route-local actions.

## 11. Larger Improvements

1. Character Chat mode shell inside Playground with dedicated setup, sessions, and composer context.
2. Character-scoped session rail backed by `useServerChatHistory(filterMode: "character")`.
3. Saved role-play setup library distinct from chat history.
4. Real backend deterministic provider harness for character chat E2E.
5. Extension sidepanel parity pass with compact active-character state.
6. Full accessibility and responsive redesign of role-play setup controls.

## 12. Telemetry And Success Metrics

No external telemetry is required. If local analytics or internal event logs exist, capture only local/self-hosted usage signals.

Suggested product metrics:

- First-time success: user reaches a ready Character Chat composer from `/chat` without leaving the flow.
- Time to first ready state: target under 90 seconds when a model is already configured.
- Returning-user resume speed: resume recent character chat in two clicks or fewer.
- Error recovery: missing model/provider states preserve selected character and draft in 100% of tested flows.
- Accessibility: zero known critical keyboard or screen-reader blockers in Character Chat mode.
- Reliability: real backend E2E passes create/select/send/resume workflow.

## 13. Open Questions

1. Should Character Chat mode be visible as a segmented top-level mode in `/chat`, or as a prominent cockpit lane inside the existing Playground shell?
2. Should URL intent use `characterId` only, or also support `assistantKind=persona&assistantId=...` for persona parity?
3. Should saved role-play setups remain local storage bundles, or eventually become server-backed assets?
4. Should "continue as user", "impersonate user", and "force narrate" be visible by default in Character Chat mode or remain in Pro/advanced controls?
5. Should character-specific parameter presets be supported later, or should presets remain global in this PRD?

## 14. Release Gates

- Product: Character Chat is visible and coherent as a first-class `/chat` workflow.
- Engineering: implementation reuses existing backend/runtime seams and does not create a parallel character chat stack.
- QA: real backend E2E passes create/select/send/resume and no-provider recovery.
- UX: first-time and power-user walkthroughs pass on desktop and narrow widths.
- Accessibility: keyboard and screen-reader checks pass for primary setup and send flow.
- Safety: destructive actions have confirm or undo, and state-changing mode switches are explicit.

## 15. Recommended First Fix

Fix Phase 1 first: make `/chat` enter and preserve a visible Character Chat mode from the starter card, header action, and `/characters` launch paths.

Reason: this creates the product container that every other improvement needs. Readiness, sessions, saved setups, power controls, and extension parity all become simpler once Character Chat is a durable mode rather than a set of generic chat controls plus picker events.
