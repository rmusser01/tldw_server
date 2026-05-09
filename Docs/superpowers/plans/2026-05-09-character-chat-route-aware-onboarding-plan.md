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

**Status:** Complete

Steps:

- Trace Home/Companion Home connection completion. `WorkspaceConnectionGate` sent first-run users to plain `/`; `OptionIndex` owned the first-run shell and `OnboardingConnectForm` owned the post-connect success screen.
- Identify whether route origin is already available through resolver state, query params, or navigation history. Route origin was available in router location; a guarded `intent=character-chat&returnTo=...` handoff now carries it into home onboarding.
- Add test coverage for connecting from a `/characters` starting route. Red coverage was added in `WorkspaceConnectionGate.test.tsx` and `core-route-identity.test.tsx`.

## Stage 2: Add Character-Chat First-Run Lane

**Goal:** Offer first-run actions aligned with character chat.

**Success Criteria:**
- Users arriving from character-chat intent see `Create character`, `Import character`, `Choose model`, and `Start character chat`.
- Completion returns to the interrupted character-chat route.
- Users arriving with no specific intent still see the existing ingestion/research guidance.

**Tests:** Component tests for onboarding lane selection and E2E route-origin smoke.

**Status:** Complete

Steps:

- Add a small intent classifier for onboarding entry points. Added `utils/onboarding-route-intent.ts`.
- Reuse existing character creation/import routes rather than creating a parallel wizard. Added shared `CharacterChatOnboardingLane` actions that navigate to Characters create/import, Settings > Model, and Chat.
- Preserve source route and selected character, if present. The first-run gate carries a sanitized `returnTo` path; create/import actions preserve the character route and normalize action params.

## Stage 3: Verify Recovery And Skip Paths

**Goal:** Make onboarding useful without trapping power users.

**Success Criteria:**
- Skip/done actions preserve or intentionally clear route intent.
- Returning users do not see first-run character guidance unnecessarily.
- Model setup blockers remain local and actionable.

**Tests:** First-run, skip, and returning-user tests.

**Status:** Complete

Steps:

- Define when the character-chat lane is considered complete. Completion keeps the existing first-run mark-complete behavior and returns character-intent users to the sanitized interrupted route.
- Keep user controls for skipping onboarding. Existing Done/finish flow remains available; non-character users retain the existing ingestion-first success screen.
- Verify with browser screenshots at desktop and mobile widths if layout changes. Focused route/success-screen tests and full UI typecheck passed; no browser screenshot pass was run in this slice.

## Verification

- RED: `bunx vitest run src/components/Common/__tests__/WorkspaceConnectionGate.test.tsx --testTimeout=20000` failed on `/characters` navigating to `/`.
- RED: `bunx vitest run src/routes/__tests__/core-route-identity.test.tsx --testTimeout=20000` failed on missing character-chat shell and missing return navigation.
- RED: `bunx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.tsx --testTimeout=20000` failed on missing character-chat success lane.
- GREEN: `bunx vitest run src/components/Common/__tests__/WorkspaceConnectionGate.test.tsx src/routes/__tests__/core-route-identity.test.tsx src/components/Option/Onboarding/__tests__/OnboardingConnectForm.success-screen.guard.test.tsx --testTimeout=20000` passed, 18 tests.
- GREEN: `../../tldw-frontend/node_modules/.bin/tsc --noEmit -p tsconfig.json --pretty false` passed.
- GREEN: `git diff --check` passed.
- Bandit skipped: touched scope is frontend TypeScript/tests plus this plan document.

## Risks

- Too many onboarding branches can obscure the main product model.
- Route-origin heuristics may be wrong if users deep-link or refresh.

## Handoff Notes

Coordinate with the intent-preservation and model-readiness packages. The onboarding lane should not duplicate their state models.
