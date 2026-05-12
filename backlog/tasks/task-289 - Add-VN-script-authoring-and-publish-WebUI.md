---
id: TASK-289
title: Add VN script authoring and publish WebUI
status: Done
assignee: []
created_date: '2026-05-12 04:25'
labels:
  - vn-play
  - webui
  - vn-scripts
milestone: VN CYOA mode
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1597'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/superpowers/specs/2026-05-10-vn-platform-api-design.md
  - Docs/API/VN.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1597: add a bundled WebUI authoring surface for backend-owned VN Scripts API. Users should be able to create script shells, edit JSON drafts, validate diagnostics, publish immutable versions, inspect published summaries, and link published versions into scripted_story setup without duplicating backend validation policy manifest or generation-profile rules client-side.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend VN Scripts API helpers/types cover script CRUD draft validation diagnostics publish versions manifest snapshot and policy evaluation endpoints.
- [x] #2 A WebUI authoring route or panel lets users list/create/select scripts edit drafts with revision conflict handling validate diagnostics and publish with idempotency/readiness handling.
- [x] #3 Published versions show safe summary metadata and are linked into scripted_story setup without client-owned validation rules.
- [x] #4 Focused frontend tests cover helper contracts draft edit/save diagnostics rendering publish conflict/readiness states and setup linkage.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused frontend tests pass or known unrelated baseline failures are documented.
- [x] #8 Frontend lint/diff hygiene run and results recorded.
- [x] #9 Bandit applicability documented for frontend-only slice.
<!-- DOD:END -->

## Implementation Notes

- Added typed VN Scripts API client helpers and response/request contracts for script CRUD, draft save/validation/diagnostics, publish, versions, manifest snapshots, and policy evaluation.
- Added `/vn-scripts` WebUI workbench for script listing, shell creation, JSON draft editing, validation, diagnostics, publish, versions, and manifest/policy summaries.
- Added scripted-story setup integration to VN Play: published script-version selector, `/vn-scripts` empty-state guidance, exact script/session create payload fields, required top-level acknowledgements, and distinct Scripted Story mode labels/filtering.
- Review fixes applied: publish acknowledgements are not inferred, empty optional profile IDs are omitted from create payloads, validation blocks invalid visible JSON instead of validating stale draft content, version action buttons have version-specific accessible names, and scripted manual fallback requires script/version IDs.
- PR review follow-up applied: draft-scoped publish idempotency keys are stable across retries, publish success is separated from version-refresh failure, stale version refreshes cannot overwrite a newly selected script, selected script changes clear stale draft/version state immediately, diagnostics/manifest/policy displays are summarized with sensitive raw/debug/internal fields redacted, scripted-story branch navigation uses the same loading behavior as story mode, scripted-story acknowledgement payloads fall back to all warning codes when the summary requires acknowledgement, and duplicate no-script setup guidance is suppressed.

## Verification

- PASS: `cd apps/tldw-frontend && bun run test:run __tests__/vn-scripts/vnScriptsApi.test.ts __tests__/vn-scripts/VNScriptsWorkbench.test.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayApi.test.ts` -> 4 files, 61 tests passed.
- PASS: `cd apps/tldw-frontend && bunx vitest run __tests__/vn-scripts/VNScriptsWorkbench.test.tsx __tests__/vn-scripts/vnScriptsApi.test.ts __tests__/vn-play/VNPlayWorkspace.test.tsx` -> 3 files, 60 tests passed after PR review fixes.
- PASS: `cd apps/tldw-frontend && ./node_modules/.bin/eslint components/vn-scripts/VNScriptsWorkbench.tsx pages/vn-scripts.tsx lib/api/vnScripts.ts types/vn-scripts.ts types/vn-play.ts components/vn-play/NewSessionDialog.tsx components/vn-play/SessionList.tsx components/vn-play/VNPlayWorkspace.tsx __tests__/vn-scripts/vnScriptsApi.test.ts __tests__/vn-scripts/VNScriptsWorkbench.test.tsx __tests__/vn-play/VNPlayWorkspace.test.tsx __tests__/vn-play/vnPlayApi.test.ts` -> no output, exit 0.
- PASS: `cd apps/tldw-frontend && ./node_modules/.bin/eslint components/vn-scripts/VNScriptsWorkbench.tsx components/vn-play/NewSessionDialog.tsx components/vn-play/VNPlayWorkspace.tsx lib/api/vnScripts.ts __tests__/vn-scripts/VNScriptsWorkbench.test.tsx __tests__/vn-scripts/vnScriptsApi.test.ts __tests__/vn-play/VNPlayWorkspace.test.tsx` -> no output, exit 0 after PR review fixes.
- PASS: `git diff --check`.
- BASELINE BLOCKER: `cd apps/tldw-frontend && ./node_modules/.bin/tsc --noEmit --pretty false` and `./node_modules/.bin/tsc --noEmit` fail in pre-existing untouched files under `../packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/EmbeddingsModelSelectionConfig.tsx` and `../packages/ui/src/services/persona-visuals.ts`; no touched-file TypeScript diagnostics were reported before the baseline failure.
- Bandit: not applicable. This slice changes TypeScript/React/Markdown task/plan files only; no Python files were touched.

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the VN script authoring frontend and scripted-story setup bridge for issue #1597. The frontend now has typed VN Scripts API helpers, a bundled `/vn-scripts` authoring workbench, and VN Play setup support for published script versions without duplicating backend-owned validation/policy rules. Scripted-story creation now derives pack/character/rating from backend setup options, sends script identifiers and top-level acknowledgement codes, and keeps Scripted Story distinct in session controls.

PR review follow-up hardened retry/idempotency behavior, async selection races, publish refresh failure handling, safe summary rendering, scripted-story branch loading, acknowledgement fallback semantics, and empty-state duplication.
<!-- SECTION:FINAL_SUMMARY:END -->
