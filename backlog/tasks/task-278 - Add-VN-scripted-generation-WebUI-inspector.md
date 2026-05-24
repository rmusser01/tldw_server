---
id: TASK-278
title: Add VN scripted generation WebUI inspector
status: Done
assignee: []
created_date: '2026-05-12 01:04'
updated_date: '2026-05-23 17:49'
labels:
  - vn-play
  - webui
  - scripted-generation
milestone: VN CYOA mode
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1578'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/pull/1571'
  - 'https://github.com/rmusser01/tldw_server/pull/1584'
documentation:
  - Docs/superpowers/specs/2026-05-10-vn-scripted-model-generation-design.md
  - Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md
  - Docs/API/VN.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the WebUI consumer for the backend-owned scripted VN generation API from GitHub issue #1578. The UI should let users inspect session generation history and revisions, run confirm/cancel/regenerate/activate actions through backend commands, and reveal debug details only through the dedicated guarded debug endpoint. Keep prompt construction provider routing moderation parser rules and generation state derivation owned by the API server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A session-scoped WebUI generation inspector lists generation points and revisions using the backend generation history API.
- [x] #2 Revision details show active status public generated output schema profile/provider metadata stable errors and timestamps without exposing raw debug fields by default.
- [x] #3 Confirm cancel regenerate and activate controls call backend commands with idempotency keys and handle stale scene in-progress and activation-blocked responses predictably.
- [x] #4 Debug reveal uses the dedicated debug endpoint and requires a second explicit confirmation before showing moderation-blocked raw output.
- [x] #5 The WebUI does not duplicate VN generation prompt provider moderation parser or state derivation rules client-side.
- [x] #6 Focused frontend tests cover API helper behavior list/detail rendering action controls debug reveal gating moderation-blocked confirmation and major error states.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes
<!-- SECTION:NOTES:BEGIN -->
- Started from GitHub issue #1578 after PR #1571 merged and issue #1535 was closed.
- Implemented the VN scripted generation WebUI inspector in the feature worktree. Verification: focused VN frontend tests pass with 31 tests; frontend lint exits 0 with existing repo-wide warnings only; git diff --check passes; ASCII scan over touched files found no non-ASCII characters. TypeScript check was run and now has no touched-file errors; it still fails on existing baseline errors in packages/ui EmbeddingsModelSelectionConfig.tsx and persona-visuals.ts. Bandit skipped because this slice only touches TypeScript/React frontend files and Backlog metadata.
- Documentation surface reviewed: no separate docs update required for this WebUI-only consumer slice because it implements the API documented in Docs/API/VN.md without changing the backend contract.
- Opened PR #1584 against dev for this WebUI inspector slice: https://github.com/rmusser01/tldw_server/pull/1584
- Addressed PR #1584 review comments: added the dedicated session generation inspector route link and route wrapper, consolidated session/generation loading, added generation pagination, rendered timestamps and mapped error-state guidance, gated debug controls for non-admin JWT users, cleared debug state on session changes, removed the duplicate post-action refresh, and added in-flight guards for generation actions. Verification after fixes: focused VN frontend tests pass with 37 tests; lint exits 0 with existing repo-wide warnings only; TypeScript still fails only on existing packages/ui baseline errors; git diff --check passes; touched files are ASCII-only.
- Follow-up PR review sweep found a remaining conditional refresh edge after the duplicate-refresh fix: generation actions returning a `session` payload updated scene state but skipped collection refresh. Added a failing regression test, then refreshed session collections exactly once in the `response.session` branch of `handleTurn`. Verification after the fix: focused VN frontend tests pass with 38 tests; lint exits 0 with existing repo-wide warnings only; TypeScript still fails only on existing packages/ui baseline errors; git diff --check passes.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a session-scoped VN scripted generation inspector for the WebUI. It consumes backend-owned generation history/debug/action APIs, renders only public generation output by default, supports confirm/cancel/regenerate/activate commands with idempotency keys, and gates moderation-blocked raw debug reveal behind an explicit confirmation path. Focused API and workspace tests cover endpoint wiring, rendering, backend action controls, and guarded debug reveal behavior.
<!-- SECTION:FINAL_SUMMARY:END -->

## Closeout Notes
<!-- SECTION:CLOSEOUT:BEGIN -->
Closed after verifying PR #1584 merged into `dev` on 2026-05-12 at merge commit `482e73c8ed082d01d7797b80269fdae9487bbda3`. The task already recorded completed acceptance criteria, completed Definition of Done items, implementation notes, verification evidence, and final summary; this closeout only corrects the stale Backlog status.
<!-- SECTION:CLOSEOUT:END -->
