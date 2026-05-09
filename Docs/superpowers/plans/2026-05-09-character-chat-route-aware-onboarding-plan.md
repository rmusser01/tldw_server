# Route-Aware Character Chat Onboarding Plan

> For implementation agents: use the repository superpowers workflow before editing code.

**Goal:** Make first-run onboarding respect character-chat intent when users arrive from `/characters` or select character-chat entry points, instead of forcing an ingestion-first path.

**Primary evidence:** After connection, the first-run home guided the user toward ingestion while the persona's goal was character chat.

**Likely surfaces:**
- `apps/packages/ui/src/services/companion-home.ts`
- `apps/packages/ui/src/store/companion-home-layout.ts`
- `apps/packages/ui/src/routes/option-home-resolver.tsx`
- `apps/packages/ui/src/routes/option-characters.tsx`
- `apps/packages/ui/src/components/Option/Characters/Manager.tsx`
- `apps/packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx`
- `apps/tldw-frontend/e2e/workflows/onboarding-ingestion-first.spec.ts`
- `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts`

## Stage 1: Inventory Existing Onboarding State

**Goal:** Understand how first-run, connection, and route origin state are currently represented.

**Success Criteria:**
- Existing first-run state sources are identified.
- The ingestion-first path remains documented as the default for users with no stated intent.
- A failing test captures the character-chat route losing intent.

**Tests:** Existing onboarding tests plus a new character-route first-run test.

**Status:** Not Started

Steps:

- Trace Home/Companion Home connection completion.
- Identify whether route origin is already available through resolver state, query params, or navigation history.
- Add test coverage for connecting from a `/characters` starting route.

## Stage 2: Add Character-Chat First-Run Lane

**Goal:** Offer first-run actions aligned with character chat.

**Success Criteria:**
- Users arriving from character-chat intent see `Create character`, `Import character`, `Choose model`, and `Start character chat`.
- Completion returns to the interrupted character-chat route.
- Users arriving with no specific intent still see the existing ingestion/research guidance.

**Tests:** Component tests for onboarding lane selection and E2E route-origin smoke.

**Status:** Not Started

Steps:

- Add a small intent classifier for onboarding entry points.
- Reuse existing character creation/import routes rather than creating a parallel wizard.
- Preserve source route and selected character, if present.

## Stage 3: Verify Recovery And Skip Paths

**Goal:** Make onboarding useful without trapping power users.

**Success Criteria:**
- Skip/done actions preserve or intentionally clear route intent.
- Returning users do not see first-run character guidance unnecessarily.
- Model setup blockers remain local and actionable.

**Tests:** First-run, skip, and returning-user tests.

**Status:** Not Started

Steps:

- Define when the character-chat lane is considered complete.
- Keep user controls for skipping onboarding.
- Verify with browser screenshots at desktop and mobile widths if layout changes.

## Risks

- Too many onboarding branches can obscure the main product model.
- Route-origin heuristics may be wrong if users deep-link or refresh.

## Handoff Notes

Coordinate with the intent-preservation and model-readiness packages. The onboarding lane should not duplicate their state models.
