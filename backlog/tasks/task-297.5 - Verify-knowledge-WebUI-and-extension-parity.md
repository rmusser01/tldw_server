---
id: TASK-297.5
title: Verify /knowledge WebUI and extension parity
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-12 05:48'
updated_date: '2026-05-12 21:31'
labels:
  - extension
  - webui
  - knowledge
  - e2e
dependencies:
  - TASK-297.1
  - TASK-297.3
  - TASK-297.4
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-knowledge-qa-usability-source-scope-plan.md
parent_task_id: TASK-297
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR slice 5 for the /knowledge QA usability remediation. Determine whether the older apps/tldw-frontend/extension route tree is still shipped, then migrate, redirect, or document/deprecate it safely so WebUI and extension options expose the same /knowledge QA model. Also clarify that the extension/chat sidepanel KnowledgePanel is a search-and-insert/context surface, not the full /knowledge QA workspace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The active extension options route graph is verified with build/import evidence, including whether apps/tldw-frontend/extension/routes is still shipped.
- [x] #2 If the legacy extension /knowledge wrapper is active, it is migrated to the shared KnowledgeQA route; if inactive, it is safely deprecated, redirected, or removed with evidence.
- [x] #3 Extension options tests cover #/knowledge, #/knowledge/thread/:threadId, #/settings/knowledge, and #/knowledge/shared/:shareToken when product-supported.
- [x] #4 WebUI route mapping and extension page inventory no longer describe /knowledge as the older settings-only KnowledgeSettings surface.
- [x] #5 Sidepanel KnowledgePanel copy or docs clarify its role as search/context insertion for chat rather than the full /knowledge QA page.
- [x] #6 Targeted WebUI and extension E2E tests verify route parity after the changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 5 implementation plan:
1. Verify the active WebUI/extension route graph from code first: route registry, extension app entrypoints, legacy extension routes, and any KnowledgeSettings references.
2. Decide from evidence whether the legacy apps/tldw-frontend/extension tree is shipped, then either migrate active routes or document/deprecate inactive routes without disturbing unrelated pages.
3. Add or update focused tests for extension/options route inventory where route support is active and for any copy changes that clarify sidepanel KnowledgePanel scope.
4. Run targeted Vitest/type or route tests plus diff checks; record any repo-wide guard/typecheck limitations separately.
5. Keep this slice parity-focused: no repo-wide frontend cleanup and no backend changes unless route evidence requires it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified active extension options entrypoint imports @tldw/ui/entries/options/main, so the shipped extension route graph is the shared UI route registry rather than the legacy apps/tldw-frontend/extension route tree.

Aligned the legacy extension route mirror as a safety net: /knowledge now renders KnowledgeQA instead of KnowledgeSettings, and /knowledge/shared/:shareToken is present alongside /knowledge/thread/:threadId.

Updated WebUI/extension page inventories and mapping labels so /knowledge is documented as Knowledge QA, not a settings-only Knowledge page.

Added sidepanel KnowledgePanel scope copy clarifying that the panel searches/inserts context for the current chat and /knowledge is the full QA workspace.

Verification passed: apps/tldw-frontend static extension parity/route registry Vitest suite, apps/packages/ui KnowledgePanel unit suite, shared route loader tests, and git diff --check.

Known limitations: package-wide apps/packages/ui TypeScript check still fails on existing repo-wide drift outside this slice; design-state guard still fails on existing AgentRegistry/AgentTasks/baseline findings; optional broader e2e-harness and entry-shell-performance guards fail on existing non-knowledge issues. Filtered design-state output shows no new findings on this slice's changed surfaces.

Targeted browser-level extension E2E attempt: npx was unavailable in the shell; bunx playwright test tests/e2e/quick-chat-guides-tutorials.spec.ts -g "knowledge tutorial card" --reporter=line started WXT production build, emitted only existing duplicate-import build warnings, then hung with no output for several minutes. I stopped the remaining Playwright/WXT processes; command exited as build:chrome:prod code 1 after termination. No /knowledge runtime assertion failed because the tests did not reach the browser phase.

Acceptance criterion #6 remains partially blocked by the extension E2E harness/build hang. Static route parity and route-loader tests pass, but browser-level extension route verification could not be completed in this environment.

Additional blocker investigation on 2026-05-12: direct `bun run build:chrome:prod` reproduces the same WXT build/pre-render hang outside Playwright global setup. `bun run build:chrome:dev` and `bun run dev -- --host 127.0.0.1 --port 17311` also hang in WXT pre-render before writing `manifest.json`; dev server did not respond on the configured localhost port while pre-render was stuck.

Tried `wxt build --filter-entrypoint options --debug` and `wxt build --filter-entrypoint background --debug`; both reproduced the same no-output WXT hang after entrypoint transform/import logging. This suggests the blocker is a broader extension WXT build/pre-render environment or bundling issue, not a /knowledge route assertion failure and not specific to the options route alone.

Worktree remained clean after these investigations; no code changes were made for the build blocker.

Resolved the browser-level extension E2E blocker after TASK-306 unblocked WXT builds. The knowledge tutorial route tests now use the established offline bypass path for dummy-server route verification, navigate by domcontentloaded instead of networkidle, then force the connection store after the route document mounts.

Fixed a real route-parity usability issue found during E2E: the collapsed ChatSidebar could push Notes/Quick Chat/Settings footer actions below the viewport at 1280x720 because the shortcut icon list was not scroll-constrained. The shortcut list now owns the scroll area and the footer actions remain reachable.

Verification on 2026-05-12: bunx playwright test tests/e2e/quick-chat-guides-tutorials.spec.ts -g "knowledge tutorial card" --reporter=line passed with 2 tests; ChatSidebar focused Vitest suite passed; git diff --check passed. Existing broader package typecheck/design-state limitations remain as previously documented.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed /knowledge WebUI/extension route parity verification. The active extension route graph already pointed at shared UI routes, the legacy mirror was aligned to KnowledgeQA, route inventory/mapping labels were corrected, and sidepanel copy now distinguishes chat context insertion from the full /knowledge QA workspace. Browser-level extension coverage now verifies both #/knowledge/thread/:threadId and #/knowledge/shared/:shareToken tutorial guide behavior.

Additional fix: collapsed sidebar footer actions now remain reachable on shorter extension viewports by making only the shortcut icon list scroll. This was required for users and tests to open the Quick Chat Helper from /knowledge at 1280x720.

Verification: route parity/static Vitest suites from the earlier slice, KnowledgePanel route tests, ChatSidebar focused Vitest tests, extension build via Playwright global setup, targeted extension Playwright knowledge tutorial route tests, and git diff --check.
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
