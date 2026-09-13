---
id: TASK-13249
title: Expose Explainer in WebUI and extension navigation
status: Done
assignee: []
created_date: '2026-09-13 18:22'
updated_date: '2026-09-13 18:31'
labels:
  - frontend
  - explainer
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The merged Explainer workspace is reachable directly but missing from the WebUI Pages launcher and shared route metadata. Add it to Research and page search using existing visibility and personalization behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pages launcher search and Research category offer a link to /explainer in current and legacy views.
- [x] #2 Explainer is available in default and researcher shortcuts and shared command palette metadata, while saved custom selections remain respected.
- [x] #3 Focused navigation and settings checks pass with verification recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add focused launcher regression coverage; register Explainer in shortcut IDs, Research, English locale, and route metadata; verify settings compatibility, navigation suites, lint, and security applicability; self-review and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
User approved the bounded navigation follow-up. Backlog MCP resource and task search requests did not respond; using CLI fallback. Worktree: .worktrees/codex-explainer-navigation.

User explicitly requested WebUI and extension parity. Verified WXT resolves shared UI routes; the active shared route-registry.tsx is missing /explainer even though an older route file under tldw-frontend/extension contains it. Adding the active registration and a runtime lazy-route test. Initial 96 focused tests pass in both WebUI and extension Vitest configurations.

Implementation complete: shared Research launcher entry and persisted shortcut ID; default/Researcher presets; shared English label/description; command palette metadata; active extension route registration; WebUI and extension page inventories. Existing saved selections remain opt-in for Explainer via Show all features, and the previous Sources migration is preserved. Regression tests demonstrated missing launcher results and the extension Page not found state before the fix. WebUI and extension configurations each pass 106 focused tests across 11 suites. ESLint reports zero errors and six pre-existing no-explicit-any warnings on unchanged settings lines. Bandit is not applicable: touched code is TypeScript/TSX/JSON only. Next.js compiled /explainer and returned HTTP 200; live browser inspection stopped at the backend readiness guard because the local API was not ready. No live installed-extension E2E run or full production build performed.

Extension compile passed: bun run compile (tsc --noEmit -p tsconfig.compile.json). Self-review found no remaining issues in the changed scope. New route test formatted using the shared UI style; git diff --check passed. Follow-up runtime limitation: browser preview at http://127.0.0.1:8091/explainer requires a ready local backend; no backend settings changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Explainer is discoverable from Research and search in the shared Pages launcher for WebUI and extension, appears in default/Researcher shortcuts and shared page search, and resolves through the active extension options router. Both page inventories now include it. Saved custom selections are respected, and Sources migration compatibility is preserved. Verified 106 focused tests in each client configuration, extension TypeScript compile, zero lint errors, formatting, and diff checks. Live browser verification remains limited by local backend readiness; no production build or installed-extension E2E claimed.
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
