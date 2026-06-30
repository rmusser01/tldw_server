# Character Chat Mode Sequencing Plan

> For implementation agents: use the repository superpowers workflow before editing code.

**Goal:** Make Chat's character mode follow the natural task order: choose character, confirm model readiness, optionally configure scene, then send the first message.

**Primary evidence:** Selecting `Character chat` in Chat opened Scene Director before a visible character picker.

**Likely surfaces:**
- `apps/packages/ui/src/components/Layouts/ChatHeader.tsx`
- `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
- `apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts`
- `apps/packages/ui/src/hooks/useSelectedCharacter.ts`
- `apps/packages/ui/src/services/actor-settings.ts`
- `apps/packages/ui/src/store/chat-surface-coordinator.ts`
- `apps/packages/ui/src/components/Sidepanel/Chat/ModeToggle.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/empty.tsx`

## Stage 1: Trace Current Mode Transitions

**Goal:** Identify why character mode opens scene setup first.

**Success Criteria:**
- Current events, store updates, and modal/drawer triggers are mapped.
- A failing test captures the scene-first behavior.

**Tests:** Component tests around Chat header/mode selection and selected-character state.

**Status:** Complete

Steps:

- Trace the `Character` / `Character chat` click path.
- Identify the event that opens Scene Director.
- Determine whether recent/last character state already exists.

Outcome: The scene-first behavior came from two entry points that opened
`tldw:open-actor-settings` before character selection: the chat header
`Character chat` action when no character was selected, and the Playground empty
state's character starter.

## Stage 2: Implement Character-First Entry

**Goal:** Make character selection the first visible next step.

**Success Criteria:**
- Entering character mode without a selected character opens a character picker or recent-character chooser.
- Entering with a selected character shows that character as active.
- Scene Director remains reachable but optional.

**Tests:** Component tests for no-character, selected-character, and recent-character states.

**Status:** Complete

Steps:

- Reuse `AssistantSelect` or existing character picker primitives.
- Present recent characters and favorites if available.
- Do not open actor/scene controls unless the user selects that advanced path.

Outcome: Added a shared `tldw:open-assistant-select` event and reused the
existing `AssistantSelect` character tab. Header character chat and the
Playground character starter now open character selection first. The existing
Scene Director footer remains available from the picker as an optional advanced
path.

## Stage 3: Add Readiness And Scene Progression

**Goal:** Create a predictable progression after character selection.

**Success Criteria:**
- Missing model state appears after character selection and names the selected character.
- Scene Director is clearly labeled as optional context.
- First message send uses the selected character.

**Tests:** Chat-mode E2E with mocked model state; unit tests for mode state transitions.

**Status:** Complete

Steps:

- Consume the shared model-readiness contract.
- Gate composer send only when required readiness items are missing.
- Verify switching back to normal chat clears character-only state as intended.

Outcome: This package kept the model-readiness contract from the prior
readiness work in place. Regression coverage verifies that an already-selected
character remains active when starting a new character chat, while saved and
temporary chat starts clear character-specific state.

## Risks

- Existing power users may rely on fast actor settings access.
- Character and persona selection may overlap if assistant selection semantics are unclear.

## Handoff Notes

Do not rename user-facing terms in this package unless required for sequencing. Larger terminology cleanup belongs in the taxonomy package.
