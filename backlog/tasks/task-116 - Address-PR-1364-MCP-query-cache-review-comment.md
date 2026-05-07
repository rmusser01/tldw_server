---
id: TASK-116
title: Address PR 1364 MCP query cache review comment
status: Done
assignee: []
created_date: '2026-05-07 19:29'
labels:
  - mcp
  - chat
  - webui
  - extension
  - review-fix
dependencies:
  - TASK-113
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1364#discussion_r3202498493'
  - apps/packages/ui/src/hooks/useMcpTools.tsx
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the PR #1364 review finding that MCP React Query keys are not scoped by the active connection identity. Ensure health, tools, catalogs, and modules queries cannot reuse cached MCP results across server/auth/org/principal switches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP React Query keys include a stable active connection scope for health tools catalogs and modules queries
- [x] #2 Focused tests assert cache keys change when the connection scope changes
- [x] #3 PR #1364 review surface is rechecked after pushing the fix
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added the active MCP tool preference scope to the React Query keys for MCP health, tools, tool catalogs, and tool modules. Added a focused hook regression test that switches the mocked connection URL from port 8000 to port 9000, verifies a second tool fetch occurs, and checks that both scopes appear in the query cache keys.

Verification before push: `bunx vitest run src/hooks/__tests__/useMcpTools.gating.test.tsx` passed with 1 file / 6 tests. The broader focused MCP tool filter suite passed with 7 files / 22 tests. Bandit is not applicable because this review fix only changes TypeScript and Backlog task files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL:BEGIN -->
Resolved the PR #1364 MCP query cache bleed review finding by scoping all MCP React Query caches to the active connection identity. This prevents health, tools, catalogs, and module query results from being reused across server/auth/org/principal changes.
<!-- SECTION:FINAL:END -->
