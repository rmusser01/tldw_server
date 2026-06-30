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

**Status:** Complete

Proposed baseline:

- `Character`: reusable speaking identity.
- `Character chat`: conversation using a selected character.
- `Scene`: optional context or roleplay layer.
- `Persona`: persistent user/assistant behavior profile only if distinct from character.
- `Assistant`: generic AI identity where no character is selected.
- `Companion Home`: product shell/home only if it remains broader than chat.

Executed outcome:

- Added `Docs/Product/WebUI/Character_Chat_Terminology_Taxonomy_2026_05_09.md` as the source of truth for Character, Character chat, Scene, Persona, Assistant, and Companion Home.
- Kept API/module terms unchanged; this package only changes user-facing decision labels.

## Stage 2: Align Critical Workflow Labels

**Goal:** Update labels where users must choose a path.

**Success Criteria:**
- Character-chat entry points use the same language.
- Scene/Actor controls are clearly optional after character selection.
- Persona labels are not used as synonyms for characters.
- Existing docs and tests reflect the chosen taxonomy.

**Tests:** Component tests that depend on accessible names; snapshot/string checks where appropriate.

**Status:** Complete

Steps:

- Audit UI strings in the affected surfaces.
- Update only labels that affect character-chat comprehension.
- Avoid broad rebranding unrelated to this flow.

Executed outcome:

- Updated the mixed identity picker from `Select assistant` to `Select character or persona`.
- Updated identity picker search from `Search assistants` to `Search characters and personas`.
- Updated the picker tablist label from `Assistant types` to `Character or persona`.
- Updated `Scene Director (Actor)` to `Optional scene context` in the character/persona picker and English locale entries.

## Stage 3: Add Local Disambiguation Where Needed

**Goal:** Help users choose between adjacent concepts without adding explanatory clutter.

**Success Criteria:**
- Short local helper text appears only where the user must choose between Character, Persona, or Scene.
- No in-app tutorial prose is added to surfaces that already have clear actions.

**Tests:** Component tests for the decision points and UX copy review.

**Status:** Complete

Steps:

- Add concise helper text in assistant/character picker contexts if needed.
- Keep settings/docs more detailed than runtime UI.
- Verify mobile and desktop layouts do not overflow.

Executed outcome:

- Avoided adding new runtime tutorial prose; the only local disambiguation is the shorter optional scene label at the exact decision point.
- Focused verification covered accessible names; full UI typecheck is recorded below.

## Verification

- RED: `bunx vitest run src/components/Common/__tests__/AssistantSelect.behavior.test.tsx --testTimeout=20000` failed on old `Select assistant`, `Search assistants`, and `Scene Director (Actor)` labels.
- GREEN: `bunx vitest run src/components/Common/__tests__/AssistantSelect.behavior.test.tsx --testTimeout=20000` passed, 7 tests.
- GREEN: `../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false` passed.
- GREEN: `git diff --check` passed.
- GREEN: string search found no old `Select assistant`, `Search assistants`, `Assistant types`, or `Scene Director (Actor)` labels in the touched picker/locale/test/doc scope.
- Bandit skipped: touched scope is frontend TypeScript/tests plus docs/backlog.

## Risks

- Renaming too much can break existing user habits and tests.
- Some terms may be meaningful API/domain terms and should not be changed globally.

## Handoff Notes

This package should follow the sequencing and intent packages so copy reflects actual behavior, not planned behavior.
