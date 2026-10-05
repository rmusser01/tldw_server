# History-first navigation — TASK-13455.2

Baseline: local `55afb351e3`, latest `origin/dev c2260945ee` verified by fresh fetch. This approved Stage 2 child follows owned on-device discovery; it does not close the whole stage or program. The existing settings registry remains the only preference store, and source lists retain their ownership/lazy boundaries.

Design: Find conversations stays visible above recent history and platform shortcuts. Recent history defaults expanded after preference readiness; an explicit collapse/expand persists through reopening, route changes and remount. Active query temporarily reveals results without replacing the saved expansion preference; explicit collapse clears that query and closes results. Platform shortcuts retain their own explicit preference without automatic tools-first resets. A collapsed-rail Find action opens and focuses the existing input. Search commands labels the command palette and header action, preserving accelerators and actual focus handling. English source/extension catalogs agree. Full combined source search, offline mirrors/import, character paging and composer disclosure remain separate approved units.

## Stage 1: Reproduce navigation and preference failures
**Goal**: meaningful behavior tests against existing sidebar and actual preference hooks.
**Success Criteria**: first-use hierarchy, saved-read readiness, explicit persistence/reopen, active search and focus regressions fail before implementation; old tools-first tests are replaced only where the approved policy supersedes them, retaining selection/order/scroller coverage.
**Tests**: real useSetting/registry/Plasmo hook with memory storage IO; real English i18next and AntD; domain source-list IO doubles.
**Status**: Complete

## Stage 2: Implement hierarchy and stable preferences
**Goal**: discoverable search/history with current source list laziness and accepted callbacks.
**Success Criteria**: no automatic recent/shortcut preference writes on mount, reopening or route navigation; pending saved preference does not start history; typing/clear preserves preference; collapsed rail focus works.
**Tests**: mounted sidebar plus existing lazy/coordinator and Temporary invariants.
**Status**: Complete

## Stage 3: Clarify command search and verify compatibility
**Goal**: recognizable Search commands versus Find conversations without changing shortcuts.
**Success Criteria**: actual-English dialog/header labels and package parity; header action and existing command accelerators/focus checks pass.
**Tests**: affected header/palette/navigation, catalog parity, typecheck and no added touched lint.
**Status**: Complete

## Stage 4: Review, record and commit
**Goal**: checked incremental delivery with both authorized reviewers and honest coverage.
**Success Criteria**: no remaining Critical/Important findings; relevant regressions, diff and task normalization pass; no whole-program or native geometry/VoiceOver claim. Prior owned-discovery plan removed only after its archive at `55afb351e3`.
**Tests**: bounded checks; Bandit N/A for TypeScript/docs-only production changes.
**Status**: Complete

## Verification and review — 2026-10-05

- Red-first navigation/header run: 16 failures, 13 existing passes. After implementation, all 29 cases passed. Delayed history/source preference reads reproduced 2 additional failures; the readiness gates corrected both. Review then reproduced 3 delayed Find focus failures after another control, close/reopen and route replacement; the qualified transient focus request corrected all three while retaining successful delayed focus.
- Final bounded run: 9 suites / 139 tests passed, including 21 real-settings/registry/Plasmo/AntD/English navigation cases, existing lazy/coordinator and Temporary assertions, header/command accelerators, owned on-device loader races and persistence handoff. The obsolete tools-first reset tests were replaced with the new policy; accepted callbacks, shortcut order, scroller, selection and search coverage is retained. Legacy fixtures now provide a reactive setter for the persisted recent preference; a command test queries its updated placeholder.
- Frontend typecheck passed. No added touched-scope ESLint findings relative to `55afb351e3`; pre-existing findings remain. All 9 changed navigation/search English catalog values agree with the packaged-extension catalog. Diff whitespace and exact task normalization checked.
- Both independent reviewers found no remaining substantiated Critical/Important issue after the focus correction.
- Coverage limits: navigation uses real preference hooks and UI with memory storage IO, source-list and debounce doubles. Header tests use fallback translations; source/package parity is checked separately. This is not live backend/account-transition, browser geometry, native extension-shell or VoiceOver acceptance. Those remain program gates.
- Bandit N/A: TypeScript/documentation-only production changes. Previous owned-discovery plan removed after archival at `55afb351e3`. Whole Stage 2 and TASK-13453 remain open, including complete combined search/offline mirrors/import, character session paging, focused composer and remaining D7/roleplay/accessibility/extension/journey requirements.
