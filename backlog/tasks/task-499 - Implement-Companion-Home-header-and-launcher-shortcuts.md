---
id: TASK-499
title: Implement Companion Home header and launcher shortcuts
status: Done
modified_files:
- apps/packages/ui/src/components/Layouts/ChatHeader.tsx
- apps/packages/ui/src/components/Layouts/Header.tsx
- apps/packages/ui/src/components/Layouts/__tests__/ChatHeader.test.tsx
- apps/packages/ui/src/components/Layouts/header-shortcut-items.ts
- apps/packages/ui/src/services/settings/ui-settings.ts
- apps/packages/ui/src/services/__tests__/ui-settings.header-shortcuts.test.ts
- apps/packages/ui/src/components/Layouts/__tests__/header-shortcut-items.hosted.test.ts
- apps/packages/ui/src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts
- apps/packages/ui/src/components/Layouts/HeaderShortcuts.tsx
- apps/packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Companion Home shortcut design.

Spec: Docs/superpowers/specs/2026-06-01-companion-home-header-shortcut-design.md
Plan: Docs/superpowers/plans/2026-06-01-companion-home-header-shortcut-implementation-plan.md

Scope:
- Add a Companion Home header icon button immediately left of the signpost shortcut button, navigating to `/`.
- Add a Companion Home shortcut/listing to the page directory and legacy sheet modal.
- Preserve existing persisted header shortcut selections by forcing Companion Home into coerced selection IDs.
- Keep `/` active state exact so `/chat` does not mark Companion Home current.

Verification:
- Focused Vitest coverage for ChatHeader, HeaderShortcuts, header shortcut item hosting, persona defaults, and UI settings shortcut coercion.
- TypeScript check for the UI package.
- Bandit noted as not applicable to frontend-only touched scope unless Python files become touched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 (Header Button Contract) complete:
- Added ChatHeader onOpenCompanionHome callback contract.
- Rendered a Companion Home House icon button immediately before the signpost shortcut toggle with the same header icon focus-ring treatment.
- Wired Header to navigate to `/` through useNavigate.
- Added focused ChatHeader coverage for accessible name, DOM/tab order, callback invocation, and focus-ring classes.

Verification:
- `bunx vitest run src/components/Layouts/__tests__/ChatHeader.test.tsx` from `apps/packages/ui`: 11 tests passed.
- `git diff --check`: passed with no output.
- Bandit: not applicable for this Task 1 frontend-only TypeScript/TSX touched scope.

Task 2 (Shortcut Metadata And Persisted Selection) complete:
- Added Companion Home shortcut metadata with id `companion-home`, target `/`, House icon, Start group placement, and hosted-mode visibility.
- Added `companion-home` to header shortcut ids, required shortcut ids, and persona shortcut expectations so persisted filtered selections still expose it.
- Added focused coverage for default/coerced shortcut selection, hosted shortcut visibility, and persona defaults.

Verification:
- `bunx vitest run src/services/__tests__/ui-settings.header-shortcuts.test.ts src/components/Layouts/__tests__/header-shortcut-items.hosted.test.ts src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts` from `apps/packages/ui`: 18 tests passed.
- Bandit: not applicable for this Task 2 frontend-only TypeScript/TS touched scope.

Task 3 (Launcher Rendering And Active Route Behavior) complete:
- Derived HeaderShortcuts test selection from shared `HEADER_SHORTCUT_IDS` so launcher tests include `companion-home`.
- Added current launcher coverage for Companion Home listing, home target, `/chat` non-current state, Chat current state, and `/` current state.
- Added legacy sheet coverage for Companion Home listing, `/chat` non-current state after moving selection away from Companion Home, and `/` current state.
- Replaced the duplicated `/` to `/chat` active alias with a shared exact-route helper used by both launcher rendering branches.

Verification:
- Red check: `bunx vitest run src/components/Layouts/__tests__/HeaderShortcuts.test.tsx` from `apps/packages/ui` failed before production changes with 2 expected active-state failures.
- Green check: `bunx vitest run src/components/Layouts/__tests__/HeaderShortcuts.test.tsx` from `apps/packages/ui`: 30 tests passed.
- Bandit: not applicable for this Task 3 frontend-only TypeScript/TSX touched scope.
Task 3 quality review follow-up complete:
- Added exact `aria-current` regression coverage for current launcher and legacy sheet links, including `/chat/thread/1` not marking Chat current.
- Added direct `aria-current` assertions for `/chat` Chat current state and `/` Companion Home current state.
- Added `end` to both launcher `NavLink` instances so React Router current semantics match the exact route helper and visual active styling.

Verification:
- Red check: `bunx vitest run src/components/Layouts/__tests__/HeaderShortcuts.test.tsx` from `apps/packages/ui` failed before production changes with 2 expected nested-route `aria-current` failures.
- Green check: `bunx vitest run src/components/Layouts/__tests__/HeaderShortcuts.test.tsx` from `apps/packages/ui`: 32 tests passed.
- Bandit: not applicable for this frontend-only TypeScript/TSX and Backlog markdown touched scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Companion Home in the shared header and launcher surfaces.

What changed:
- Added a Companion Home House icon button in `ChatHeader`, immediately before the signpost/page-directory button, wired through `Header` to navigate to `/`.
- Added `companion-home` as the first shared header shortcut ID and as a required ID so old persisted filtered selections are coerced forward.
- Added Companion Home to shortcut metadata with `/` target, Start group placement, House icon, hosted-mode visibility, and persona/default coverage.
- Updated current launcher and legacy sheet active-state logic to use exact route matching, including `NavLink end`, so `/chat` and nested chat routes do not mark Companion Home or Chat incorrectly.
- Made legacy sheet keyboard-selection styling distinct from route-current styling so the initially selected Companion Home row does not visually read as current on `/chat`.

Verification:
- `bunx vitest run src/components/Layouts/__tests__/ChatHeader.test.tsx src/components/Layouts/__tests__/HeaderShortcuts.test.tsx src/components/Layouts/__tests__/header-shortcut-items.hosted.test.ts src/components/Layouts/__tests__/persona-shortcut-defaults.test.ts src/services/__tests__/ui-settings.header-shortcuts.test.ts` from `apps/packages/ui`: 5 files passed, 61 tests passed.
- `git diff --check 4e682bea9f..HEAD`: passed.
- `bunx tsc --noEmit --project tsconfig.json` from `apps/packages/ui`: fails on existing package-wide unrelated errors; the touched-file `ChatHeader.test.tsx` type error found during verification was fixed, and the remaining diagnostics are outside the touched Companion Home files.
- Bandit: not applicable; touched scope is frontend TypeScript/TSX tests and Backlog markdown only.
- Browser smoke: Next dev server on `http://127.0.0.1:8080` with seeded local auth showed one Companion Home button immediately before the signpost button, page directory listing included Companion Home with `/` href, and clicking Companion Home navigated from `/chat` to `/`.

Known skips/blockers:
- Full UI package TypeScript remains blocked by pre-existing unrelated errors outside this change.
- Browser plugin was available but its in-app viewport rendered the mobile route without the target desktop header, so desktop smoke used a headless Playwright fallback.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
