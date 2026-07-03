---
id: TASK-12108
title: Consolidate triplicated characterChatMode and add stream-inactivity watchdog to the live path
status: In Progress
labels:
- tech-debt
- medium
- chat
- character
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: Medium (correctness trap + a stalled stream hangs forever).** Round-2 audit finding R5.

There are **three** near-identical copies of `characterChatMode`:
- inline in `hooks/chat/useChatActions.ts` (Option/Playground, dispatched ~`:2728`) — LIVE
- inline in `hooks/useMessage.tsx` (Sidepanel, ~`:1170`) — LIVE
- extracted `hooks/chat/useCharacterChatMode.ts` (`createCharacterChatMode`) — **referenced only by tests, not wired to any production path** (verified: no non-test import).

The extracted (unused) copy has a **60s stream-inactivity watchdog** (`:902-911,985-991`) and failure-recovery classification (`buildCharacterChatAssistantErrorContent`, provider-unconfigured / model-unavailable recovery) that **neither live copy has** — the live paths use plain `buildAssistantErrorContent` and have no inactivity timer. So a live character stream that stalls mid-response hangs indefinitely, while the tests pass against the safer unused code (false confidence).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The two live `characterChatMode` copies and the extracted one are consolidated to a single implementation used by both Playground and Sidepanel (or the extracted copy is deleted and its tests re-pointed at the live path).
- [ ] #2 The live character streaming path has a stream-inactivity watchdog (times out a stalled stream instead of hanging) and the failure-recovery classification.
- [ ] #3 The contract/guard tests exercise the actual shipped code path, not an unused copy.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
