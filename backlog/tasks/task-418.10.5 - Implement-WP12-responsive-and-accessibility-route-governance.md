---
id: TASK-418.10.5
title: Implement WP12 responsive and accessibility route governance
status: In Progress
labels:
- wp12
- webui
- route-governance
- responsive
- accessibility
- e2e
priority: High
ordinal: 5
parent_task_id: TASK-418.10
references:
- TASK-418.10
- https://github.com/rmusser01/tldw_server/pull/1970
documentation:
- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP12 Task 5 from the WebUI route governance QA plan: add responsive overflow governance, sidepanel-width route checks, metadata-aligned high-risk Axe coverage, and focused scripts for the route governance gate without changing product behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Responsive governance spec checks the planned representative routes at 390px for page-level horizontal overflow.
- [x] #2 Sidepanel-width governance covers the initial sidepanel-supported route set and is metadata-backed where possible.
- [x] #3 Stage 4 Axe high-risk route coverage is metadata-aligned or manual entries have explicit rationales.
- [x] #4 Package scripts expose route-governance and governance-gate commands using repository conventions.
- [x] #5 Focused responsive, Axe, lint/diff, and security verification results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation notes:
- Added apps/tldw-frontend/e2e/smoke/route-responsive-governance.spec.ts.
- Added responsive metadata checks against page-inventory.ts and route-metadata.ts before running overflow assertions.
- Added page-level horizontal overflow checks at 390x844 for the planned representative route set.
- Added sidepanel-width horizontal overflow checks at 360x720 for /chat, /flashcards, /companion, and /persona with extension_sidepanel metadata validation.
- Added Stage 4 high-risk route rationales and a metadata alignment guard.
- Removed an unused Stage 4 Axe helper while touching that file so focused lint stays clean.
- Added package scripts e2e:smoke:route-governance and e2e:smoke:governance-gate.

PR review fixes:
- Added responsive route post-navigation guards that fail on route error boundaries or uncaught page errors before measuring overflow.
- Raised responsive overflow tolerance from 1px to 4px to match the existing Stage 4 responsive landmark tolerance for sub-pixel/scrollbar variance.
- Moved Stage 4 Axe redirect target validation before unavailable-route skip handling so unexpected redirects fail instead of being skipped.
- Restored waitForVisualSettle before Axe analysis so scans happen after fonts and paint cycles settle.

Verification:
- RED: bunx playwright test e2e/smoke/route-responsive-governance.spec.ts --reporter=line --workers=1 failed with "No tests found" before the new spec existed.
- PASS: bun run e2e:smoke:route-governance -> 18 passed after initial implementation.
- PASS: bun run e2e:smoke:route-governance -> 18 passed after PR review fixes.
- PASS: bun run e2e:smoke:stage4 -> 29 passed, 1 skipped after initial implementation.
- PASS: bun run e2e:smoke:stage4 -> 29 passed, 1 skipped after PR review fixes.
- PASS: bunx eslint e2e/smoke/route-responsive-governance.spec.ts e2e/smoke/stage4-axe-high-risk-routes.spec.ts e2e/smoke/stage4-axe-high-risk-routes.helpers.ts -> 0 errors/warnings.
- PASS: git diff --check.
- PASS: Bandit via /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python on apps/tldw-frontend/e2e/smoke -> 0 findings, 0 LOC scanned because touched scope is TS.
- LIMITATION: bunx tsc --noEmit --pretty false still fails on repo-wide baseline errors outside this slice, including MediaReadAlongPopover, Watchlists quick setup/run detail, WorkspacePlayground StudioPane, keyboard shortcut config, persona live control, and admin-llamacpp E2E fixtures.
- LIMITATION: an initial broader route-governance script attempt that bundled existing route-heading and route-capability governance suites failed on pre-existing unrelated failures (/media H1; capability raw/error-state checks for Scheduled Tasks, Integrations, Model Settings, Evaluations, MCP Hub, Skills, TTS, Speech, Data Tables). The new package script was narrowed to the stable WP12 responsive governance spec.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented WP12 Task 5 responsive and accessibility route governance and addressed PR review feedback. Added route-responsive-governance.spec.ts for 390px page overflow checks and sidepanel-width checks, added metadata/rationale validation to Stage 4 high-risk Axe routes, wired e2e:smoke:route-governance and e2e:smoke:governance-gate package scripts, and updated the WP12 plan status. PR review follow-up now fails responsive checks on route error boundaries/page errors, uses the existing 4px overflow tolerance, validates redirect targets before skip handling, and waits for visual settle before Axe analysis. Verification passed for route-governance, full Stage 4, focused lint, git diff --check, and Bandit. Full TypeScript remains blocked by unrelated repo-wide baseline errors documented in implementation notes.
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
