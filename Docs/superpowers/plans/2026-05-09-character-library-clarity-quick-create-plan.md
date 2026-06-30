# Character Library Clarity And Quick-Create Plan

> For implementation agents: use the repository superpowers workflow before editing code.

**Goal:** Improve the Characters library for first-time and returning character-chat users without removing power-user density.

**Primary evidence:** The audit found icon-heavy row actions, inaccurate filtered count text, and a strong but dense creation flow.

**Likely surfaces:**
- `apps/packages/ui/src/components/Option/Characters/Manager.tsx`
- `apps/packages/ui/src/components/Option/Characters/CharacterListContent.tsx`
- `apps/packages/ui/src/components/Option/Characters/CharacterGalleryCard.tsx`
- `apps/packages/ui/src/components/Option/Characters/CharacterListToolbar.tsx`
- `apps/packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx`
- `apps/packages/ui/src/components/Option/Characters/CharacterDialogs.tsx`
- `apps/packages/ui/src/components/Option/Characters/hooks/useCharacterFiltering.tsx`
- `apps/packages/ui/src/components/Option/Characters/__tests__/`

## Stage 1: Lock Current Library Defects

**Goal:** Capture the count mismatch and primary-action ambiguity in tests.

**Success Criteria:**
- Search result count test fails on current `4 characters found` behavior after filtering.
- Row/card primary chat action expectations are defined.
- Quick-create density problem is translated into concrete UI states to test.

**Tests:** Characters component tests and existing E2E character workflow.

**Status:** Not Started

Steps:

- Add a test for `1 of 4 characters shown` or agreed equivalent wording.
- Add a test that the primary chat action is text-visible in table and gallery contexts.
- Identify which create/edit fields should remain visible in quick mode.

## Stage 2: Improve Library Actions And Counts

**Goal:** Make the common returning-user action easy to scan.

**Success Criteria:**
- Table rows expose a visible `Chat` or `Chat as...` primary action.
- Secondary actions remain available with accessible names and tooltips.
- Search/filter count reflects filtered and total counts.
- Screen-reader status text updates when filters change.

**Tests:** Component and accessibility-focused tests.

**Status:** Not Started

Steps:

- Promote chat to the dominant action.
- Keep edit, favorite, delete, and more actions compact.
- Update result-count computation and status region.

## Stage 3: Add Quick Character Creation Mode

**Goal:** Let first-time users create a usable character without confronting all advanced controls.

**Success Criteria:**
- `Quick character` mode shows required fields and templates first.
- Advanced fields remain accessible behind disclosure or an explicit advanced mode.
- Optional AI-generation controls are hidden or clearly disabled when models are unavailable.
- Edit mode preserves full maintenance controls.

**Tests:** Modal/drawer component tests and screenshot/E2E smoke.

**Status:** Not Started

Steps:

- Split create flow into quick and advanced sections without duplicating form state.
- Keep import/template behavior intact.
- Ensure the quick flow can create a character and immediately hand off to chat readiness.

## Risks

- Over-simplifying the create form can hide important SillyTavern-compatible fields from users who expect them.
- Adding visible `Chat` actions can reduce table density if not designed carefully.

## Handoff Notes

Coordinate with intent preservation so the visible `Chat` action uses the same handoff contract as the existing icon action.
