---
id: TASK-418.3.1
title: Implement WebUI chat composer and global chrome remediation
status: Done
labels:
- ux
- webui
- extension
- chat
- navigation
- implementation
priority: high
parent_task_id: TASK-418.3
references:
- TASK-418.3
- Docs/superpowers/plans/2026-05-17-webui-chat-global-chrome-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the WP6 chat composer/global chrome remediation slice. Scope: keep /chat composer-first, preserve quick-chat as a helper surface, verify canonical command/header navigation targets, and ensure chat session controls do not foreground themselves on non-chat routes. No backend API changes, route renames, or broad visual redesign.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /chat foregrounds composer readiness over starter modes while keeping starter modes available through one obvious launcher.
- [x] #2 Command palette and header shortcuts continue to use canonical top-level route targets for Chat and MCP Hub.
- [x] #3 Header/global chrome exposes chat session controls only in chat-owned contexts while preserving global settings/theme/notifications/shortcuts.
- [x] #4 /quick-chat-popout remains a clearly labeled helper surface and is not treated as the main Chat target.
- [x] #5 Focused Vitest coverage passes for changed command palette, header, chat empty-state, and quick-chat behavior.
- [x] #6 Focused Playwright checks pass for /chat sticky/mobile behavior and responsive landmarks, or environment blockers are documented with evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-webui-chat-global-chrome-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented WP6 chat composer/global chrome remediation and PR #1898 review fixes.

Initial implementation notes:
- Changed Command Palette MCP Hub navigation to canonical /mcp-hub and canonicalized /settings/mcp-hub during target dedupe.
- Added header action policy so chat title/session/share actions render only for /chat while global chrome actions remain available elsewhere.
- Removed ChatHeader fallback session actions when chat-specific callbacks are omitted.
- Collapsed /chat starter modes behind an Explore chat modes launcher while leaving primary Start chatting and Quick Ingest visible.
- Added /quick-chat-popout route identity coverage as a helper surface.

Review fix notes:
- Removed duplicate pathname normalization in Header; Header now passes raw location.pathname into getHeaderActionPolicy.
- Updated header-action-policy normalization to strip repeated trailing slashes while preserving root.
- Added /chat// regression coverage to header-action-policy tests.
- Removed stale onClearChat from ChatHeader test props.

Verification:
- Red focused regression: header-action-policy test failed for /chat// before helper fix.
- Focused review-fix Vitest: 3 files / 27 tests passed.
- WP6 Vitest set: 11 files / 76 tests passed.
- git diff --check passed.
- bunx tsc --noEmit --pretty false still fails on repo-wide baseline TypeScript debt; captured 252 log lines and no errors mention touched WP6 files.
- CI before review-fix push: direct frontend/build/UX/E2E required checks passed; old full-suite macOS/windows jobs failed while the workflow was still in progress and logs were unavailable until completion. Recheck after pushing the review-fix commit.
- Bandit not applicable: this slice only touched TypeScript/React UI and Backlog markdown.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the WP6 chat composer/global chrome remediation slice:
- /chat now keeps starter modes available behind one explicit launcher instead of foregrounding every mode on first render.
- Command Palette MCP Hub navigation uses canonical /mcp-hub, with /settings/mcp-hub canonicalized during target dedupe.
- Header chat session/share/title controls are gated to /chat-owned context through a small header action policy.
- /quick-chat-popout has route-level test coverage that keeps it framed as a helper surface.
- PR #1898 review feedback was addressed: repeated trailing slash normalization, single policy-owned normalization source, and stale ChatHeader test props.
- Focused Vitest and Playwright verification passed; full package TypeScript remains blocked by existing baseline debt outside the touched files.
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
