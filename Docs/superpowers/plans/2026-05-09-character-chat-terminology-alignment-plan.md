# Character Chat Terminology Alignment Plan

> For implementation agents: use the repository superpowers workflow before editing code.

**Goal:** Reduce user confusion across `Assistant`, `Persona`, `Character`, `Companion`, and `Scene` by aligning labels and local explanatory text around the character-chat workflow.

**Primary evidence:** The audit path exposed `tldw Assistant`, `Persona Garden`, `Characters`, `Character chat`, `Companion Home`, and `Scene Director (Actor)` during one character-chat task.

**Likely surfaces:**
- `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- `apps/packages/ui/src/components/Layouts/ChatHeader.tsx`
- `apps/packages/ui/src/components/Common/AssistantSelect.tsx`
- `apps/packages/ui/src/components/Option/Characters/Manager.tsx`
- `apps/packages/ui/src/services/companion-home.ts`
- `apps/packages/ui/src/services/settings/ui-settings.ts`
- `Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md`
- `Docs/Product/WebUI/`

## Stage 1: Create A User-Facing Taxonomy

**Goal:** Define what each term means and where it should appear.

**Success Criteria:**
- Taxonomy distinguishes character, character chat, scene, persona, assistant, and companion.
- Terms are mapped to user decisions, not internal architecture.
- A review checklist identifies labels to change and labels to leave alone.

**Tests:** Documentation review and string-search verification.

**Status:** Not Started

Proposed baseline:

- `Character`: reusable speaking identity.
- `Character chat`: conversation using a selected character.
- `Scene`: optional context or roleplay layer.
- `Persona`: persistent user/assistant behavior profile only if distinct from character.
- `Assistant`: generic AI identity where no character is selected.
- `Companion Home`: product shell/home only if it remains broader than chat.

## Stage 2: Align Critical Workflow Labels

**Goal:** Update labels where users must choose a path.

**Success Criteria:**
- Character-chat entry points use the same language.
- Scene/Actor controls are clearly optional after character selection.
- Persona labels are not used as synonyms for characters.
- Existing docs and tests reflect the chosen taxonomy.

**Tests:** Component tests that depend on accessible names; snapshot/string checks where appropriate.

**Status:** Not Started

Steps:

- Audit UI strings in the affected surfaces.
- Update only labels that affect character-chat comprehension.
- Avoid broad rebranding unrelated to this flow.

## Stage 3: Add Local Disambiguation Where Needed

**Goal:** Help users choose between adjacent concepts without adding explanatory clutter.

**Success Criteria:**
- Short local helper text appears only where the user must choose between Character, Persona, or Scene.
- No in-app tutorial prose is added to surfaces that already have clear actions.

**Tests:** Component tests for the decision points and UX copy review.

**Status:** Not Started

Steps:

- Add concise helper text in assistant/character picker contexts if needed.
- Keep settings/docs more detailed than runtime UI.
- Verify mobile and desktop layouts do not overflow.

## Risks

- Renaming too much can break existing user habits and tests.
- Some terms may be meaningful API/domain terms and should not be changed globally.

## Handoff Notes

This package should follow the sequencing and intent packages so copy reflects actual behavior, not planned behavior.
