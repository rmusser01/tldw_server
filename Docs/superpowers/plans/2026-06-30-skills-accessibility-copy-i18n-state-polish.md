## TASK-530.14: Skills Accessibility, Copy, I18n, And State Polish

**Goal**: Harden the `/skills` manager and test-run/import/create workflows so the current Skills UX is keyboard-operable, screen-reader clearer, locally translatable, responsive at extension widths, and covered by focused tests.

**Scope guardrails**:
- Keep the PR limited to Skills WebUI/extension-facing frontend, locale copy, E2E tests, Backlog task metadata, and this plan.
- Do not change backend execution behavior, permissions, runtime metadata semantics, API contracts, or unrelated design-system components unless a focused test exposes a direct Skills regression.
- Prefer existing Ant Design controls, shared state primitives, and current Skills page structure over a visual redesign.

## Stage 1: Current-State Tests And Accessible Names
**Goal**: Capture missing accessible names, focus return, and live status behavior with failing tests before production edits.
**Success Criteria**: Vitest assertions fail for the specific missing behavior and avoid duplicating existing row-action coverage.
**Tests**:
- `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
- `apps/packages/ui/src/components/Option/Skills/__tests__/SkillPreview.test.tsx`
**Status**: Complete

## Stage 2: Announced States And Focus Recovery
**Goal**: Add or refine `role="status"`, `aria-live`, `role="alert"`, button labels, and modal/drawer focus return for the affected flows.
**Success Criteria**: Loading, import review, success, selection count, and test-run execution states have useful announced text; closing test-run/create surfaces returns focus to the triggering control when possible.
**Tests**:
- Focused Vitest for manager status/focus behavior.
- Focused Vitest for test-run pending/result/error announcement.
**Status**: Complete

## Stage 3: Locale Copy Consolidation
**Goal**: Move touched persistent Skills copy from inline-only defaults into the English locale source and align stale E2E/UI copy.
**Success Criteria**: New/touched strings use stable `option:skills.*` keys; existing copy remains concise and implementation-aware.
**Tests**:
- Existing Vitest translation mocks continue to pass.
- Locale JSON remains valid.
**Status**: Complete

## Stage 4: Responsive And Keyboard E2E Coverage
**Goal**: Update the mocked Skills Playwright journey for current test-run semantics and add keyboard/responsive assertions that cover the core advanced workflow.
**Success Criteria**: The mocked `/skills` journey exercises seeding, opening test run, entering args, running or rendering, closing, searching, opening create, and cancelling without pointer-only dependencies or mobile overlap.
**Tests**:
- `apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts`
**Status**: Complete

## Stage 5: Verification And Finalization
**Goal**: Run focused frontend tests and record task results without overstating unrun checks.
**Success Criteria**: Focused Vitest and feasible Playwright checks pass or blockers are documented; `git diff --check` passes; Bandit is documented as skipped if no Python code is touched.
**Tests**:
- Focused Vitest for Skills components.
- Focused Playwright Skills spec when local frontend test setup allows it.
- `git diff --check`.
**Status**: Complete

## Verification Results
- `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx src/components/Option/Skills/__tests__/SkillPreview.test.tsx src/components/Option/Skills/__tests__/skills-locale-keys.test.ts --reporter=dot`: passed, 54 tests.
- `TLDW_WEB_URL=http://localhost:18087 TLDW_WEB_CMD='bun run dev -- -p 18087' npx playwright test e2e/workflows/tier-5-specialized/skills.spec.ts --project=tier-5 --reporter=line`: passed, 5 tests.
- `node -e` JSON parse check for `apps/packages/ui/src/public/_locales/en/option.json` and `apps/packages/ui/src/assets/locale/en/option.json`: passed.
- `git diff --check origin/dev...HEAD`: passed.
- Bandit: skipped, no Python files touched.
