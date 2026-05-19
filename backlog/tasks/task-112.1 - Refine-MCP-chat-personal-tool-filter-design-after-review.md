---
id: TASK-112.1
title: Refine MCP chat personal tool filter design after review
status: Done
assignee:
  - Codex
created_date: '2026-05-07 14:15'
updated_date: '2026-05-07 14:17'
labels:
  - mcp
  - chat
  - webui
  - extension
  - design
  - review-fix
dependencies: []
references:
  - Docs/superpowers/specs/2026-05-06-mcp-chat-personal-tool-filter-design.md
parent_task_id: TASK-112
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address review findings on the MCP chat personal tool filter design before implementation planning. The spec needs to distinguish raw/discovered tools from executable/chat-filtered tools, scope persisted disabled-tool preferences by connection/user, define normalized tool identity and collision behavior, choose an exact no-tools wire contract, and strengthen tests/rollout notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec distinguishes discoveredTools/raw tools from availableTools/chatTools and supports showing canExecute=false tools in the selector.
- [x] #2 Spec scopes disabled-tool persistence by connection fingerprint and user/principal identity when available.
- [x] #3 Spec defines shared normalized chat tool identity plus collision handling for toggles and request construction.
- [x] #4 Spec chooses one no-tools wire behavior and requires raw preview to match actual request behavior.
- [x] #5 Spec adds persistence reload/reopen coverage and external-server grouping fallback expectations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update the design spec with the four review corrections: discovered/raw tool exposure, scoped persistence, shared normalized identity/collision behavior, and exact no-tools wire behavior.
2. Add improvements for reload/reopen persistence coverage and external-server grouping fallback.
3. Mirror the Backlog task into the clean design worktree with the same final task record.
4. Run docs-only verification with `git diff --check` and staged `git diff --cached --check`.
5. Complete the Backlog task and commit the spec refinement on `codex/mcp-chat-tool-filter-design`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the MCP chat personal tool filter spec to address all review findings: hook contract now exposes discoveredTools separately from availableTools/chatTools; disabled preferences are versioned and scoped by connection/user; one shared chat tool normalization helper owns identity and collision behavior; no-tools requests omit tools/tool_choice on the wire while treating effective choice as none internally; tests now include scoped persistence, reload/reopen behavior, collisions, and grouping fallback. Verification: docs-only `git diff --check` passed before staging. Bandit skipped because only markdown/Backlog task documentation changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Refined the approved MCP chat personal tool filter design after review. The spec now requires raw discovered tool exposure for UI state, scoped persisted disabled-tool preferences, shared normalized chat tool identity with collision exclusion, exact no-tools wire behavior matching `ChatTldw`, and stronger persistence/grouping/collision test coverage. Verification is documentation-only: `git diff --check` passed; Bandit is skipped because no executable code changed.
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
