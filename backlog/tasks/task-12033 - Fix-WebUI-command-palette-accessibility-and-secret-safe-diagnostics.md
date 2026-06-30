---
id: TASK-12033
title: Fix WebUI command palette accessibility and secret-safe diagnostics
status: Done
assignee: []
created_date: '2026-06-25 22:17'
updated_date: '2026-06-25 22:31'
labels:
  - webui
  - accessibility
  - search
  - health
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 4 of the WebUI audit remediation roadmap: ensure the global search/command palette trigger has stable accessible behavior by click and keyboard, keep focus management reliable, maintain secret-safe diagnostics, and verify mobile high-risk route governance where feasible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Header search button has a stable accessible name and opens the command palette by click.
- [x] #2 Cmd+K and Ctrl+K open the command palette where enabled, focus moves into the palette, Escape closes it, and focus returns to the trigger.
- [x] #3 Secret-bearing health diagnostics and visible debug/copy payloads remain redacted in app-owned UI paths.
- [x] #4 Mobile/high-risk route governance covers the audited setup, chat, media, settings health, admin/server, and scheduled-task states or documents existing coverage.
- [x] #5 Focused unit or Playwright coverage records the Stage 4 behavior changed or verified.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created after Stage 3 commit 5c03cbcf5d. Planned files: CommandPalette, CommandPaletteHost, WebLayout/header search trigger, Stage 4 smoke specs, route responsive governance, and any remaining diagnostics redaction helpers.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented Stage 4 accessibility/governance remediation. Added stable command palette trigger labeling, palette search input labeling, focus return on Escape for event and keyboard open flows, and route metadata/inventory coverage for health/admin/model settings routes. Expanded responsive and Axe high-risk route governance lists and confirmed health diagnostics redaction coverage remains active.

Modified files: Docs/superpowers/plans/2026-06-25-webui-stage4-accessibility-secret-safety-plan.md; apps/packages/ui/src/components/Common/CommandPalette.tsx; apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx; apps/packages/ui/src/components/Layouts/ChatHeader.tsx; apps/packages/ui/src/components/Layouts/__tests__/ChatHeader.test.tsx; apps/packages/ui/src/components/Layouts/__tests__/Header.character-mode.test.tsx; apps/packages/ui/src/routes/route-metadata.ts; apps/tldw-frontend/e2e/smoke/page-inventory.ts; apps/tldw-frontend/e2e/smoke/route-responsive-governance.spec.ts; apps/tldw-frontend/e2e/smoke/stage4-accessibility-controls.spec.ts; apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts.

Verification:
- PASS: focused Vitest suite for ChatHeader, Header, CommandPalette shortcuts/route targets, route metadata, Stage 4 Axe helper, health diagnostics redaction, and connection status (8 files, 60 tests).
- PASS: lightweight Playwright route governance metadata assertions with WebUI autostart disabled (2 tests).
- PASS: ESLint on touched TypeScript/TSX/spec files (0 errors; existing warnings remain in header/test mock patterns).
- PASS: git diff --check.
- N/A: Bandit, because touched implementation files are TS/TSX and docs/test metadata only.
- BLOCKED: full focused browser interaction smoke. Default Turbopack dev server failed on the worktree node_modules symlink; webpack dev server started but hit EMFILE watcher warnings, then Chromium launch failed in this sandbox with MachPort permission denied.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed WebUI command palette accessibility and Stage 4 route governance. The header search trigger now has a stable accessible name, the palette search input is labeled, Escape restores focus to the opener, command palette route labels are metadata-aligned, health/admin/model routes are first-class metadata-backed inventory entries, and high-risk responsive/Axe route lists cover the requested routes. Focused unit and governance checks pass; full browser interaction smoke is blocked by sandbox/browser launch restrictions.
<!-- SECTION:FINAL_SUMMARY:END -->

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
