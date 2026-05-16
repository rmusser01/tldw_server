---
id: TASK-333
title: Add VN script graph and outline inspector
status: Done
assignee: []
created_date: '2026-05-14 04:15'
updated_date: '2026-05-14 05:10'
labels:
  - vn
  - frontend
  - scripts
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1680'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1680: add a WebUI consumer for the backend-owned VN script authoring graph API so authors can inspect saved draft, unsaved draft preview, and published version graph structure without duplicating VN graph derivation or validation rules in the frontend. Keep this as a focused frontend/API-client/documentation slice; visual node editing, frontend graph derivation, new DSL work, runtime VN Play behavior changes, and backend graph semantics changes are out of scope unless a blocking integration issue is discovered.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typed frontend helpers cover the VN script graph endpoints and use the existing backend graph contract.
- [x] #2 VN script authoring UI exposes a read-only graph or outline inspector gated by features.script_authoring_graph.
- [x] #3 Inspector supports saved draft graph, unsaved draft graph preview, and published version graph where available.
- [x] #4 Graph metadata and diagnostics are displayed without mixing them up with script validation diagnostics.
- [x] #5 Frontend does not derive graph semantics or validate script structure beyond rendering backend responses and client input states.
- [x] #6 Focused frontend tests cover capability gating, loading and error states, stale or unsaved draft preview behavior, graph/outline rendering, and diagnostics.
- [x] #7 Documentation explains the custom-frontend graph endpoint usage and cache/staleness fields.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added typed VN authoring graph response/request models and frontend API helpers for saved draft, unsaved preview, and version graph endpoints.
- Added a read-only Script graph panel to the VN script workbench gated by `features.script_authoring_graph`.
- The panel renders backend source/hash/revision/semantics metadata, outline rows, limits, graph diagnostics, and validation diagnostics as separate UI concepts.
- Added version-card Graph actions for published version graph inspection.
- Frontend only renders server-shaped graph data; it does not compute graph edges or validate script op semantics.
- PR #1681 review fixes added graph schema metadata rendering, source path selection, stale response guards, loading reset on script switch, duplicate graph-action disabling, and stricter validation diagnostics typing.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- `bun run test:run __tests__/vn-scripts/vnScriptsApi.test.ts __tests__/vn-scripts/VNScriptsWorkbench.test.tsx` from `apps/tldw-frontend`: 2 files passed, 53 tests passed.
- `git diff --check`: passed.
- `bun run lint -- components/vn-scripts/VNScriptsWorkbench.tsx lib/api/vnScripts.ts types/vn-scripts.ts __tests__/vn-scripts/vnScriptsApi.test.ts __tests__/vn-scripts/VNScriptsWorkbench.test.tsx` from `apps/tldw-frontend`: exited 0; repo-wide baseline warnings remain because the package lint script prepends `eslint .`.
- `./node_modules/.bin/tsc --noEmit` from `apps/tldw-frontend`: failed on pre-existing shared UI type errors in `../packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/EmbeddingsModelSelectionConfig.tsx` and `../packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`; no errors were reported in the touched VN script files before the baseline failures stopped the run.
- Bandit skipped: frontend/docs/backlog-only change, no Python touched.
- PR #1681 review-fix verification: `bun run test:run __tests__/vn-scripts/VNScriptsWorkbench.test.tsx` passed 43 tests; `bun run test:run __tests__/vn-scripts/vnScriptsApi.test.ts __tests__/vn-scripts/VNScriptsWorkbench.test.tsx` passed 57 tests; `git diff --check` passed; `bun run lint -- components/vn-scripts/VNScriptsWorkbench.tsx __tests__/vn-scripts/VNScriptsWorkbench.test.tsx __tests__/vn-scripts/vnScriptsApi.test.ts` exited 0 with the repo-wide warning baseline; `bun run tsc --noEmit` still fails on the known shared UI baseline files listed above. A follow-up review sweep also addressed duplicate graph-action disabling and stricter validation diagnostics typing.
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the VN script graph inspector WebUI slice for issue #1680. The frontend now has typed graph API helpers, a capability-gated read-only graph panel for saved drafts and unsaved draft previews, per-version graph inspection from published version cards, focused coverage for the new graph flows, and custom-frontend documentation for graph-inspector usage and staleness keys.
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
