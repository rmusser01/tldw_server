---
id: TASK-13455
title: 'Chat UX stage 2: focused composer and complete conversation discovery'
status: To Do
labels:
- chat
- ux
- webui
- extension
- nng
dependencies:
- TASK-13454
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve findings 3 and 10 and the navigation part of finding 1. Make the everyday message workflow clear while retaining Pro controls and keyboard efficiency, and expose local/server history and complete character-session access through existing loaders.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Casual mode exposes message, model, Attach, Send, and Customize chat; active customization remains recognizable and all existing Pro tools and accelerators remain reachable.
- [ ] #2 Find conversations and recent history appear ahead of platform tools; opening the sidebar retains the user expansion preference and presents conversation identity and storage origin.
- [ ] #3 One conversation query reaches paged Server and On this browser results without title-based identity merging or account/workspace leakage; appropriate empty, loading, stale, and error states remain usable.
- [ ] #4 Search commands is distinct from Find conversations, with existing shortcuts documented and focus restored after either action.
- [ ] #5 More than five interleaved character sessions, duplicate titles, and large history are tested; All sessions for this character remains complete or explicitly reports partial coverage with paging rather than falsely empty results.
- [ ] #6 Mirror deduplication only hides a proven duplicate visible in current successful server results; filtering, paging and outage keep cached copies reachable, while local storage failure reports unavailable rather than empty.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Program: TASK-13453. Design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Base: origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488, verified 2026-10-04. User approved equal audience priority and all ten findings plus enhancements in five stages. Written-design approval and the requested critical review are recorded on 2026-10-04; application implementation has not begun. ADR required: no for the client remediation; ADR-049 and ADR-050 govern owner-bound history and native fork boundaries, ADR-059 governs canonical task editing. Reassess if implementation changes public APIs or persistence authority. Reuse existing shared UI, loaders, presets, and test harnesses; preserve temporary-chat privacy and account/workspace fences. Findings 3, 10 and history portion of 1. Reuse ChatSidebar, LocalChatList, useLoadLocalConversation, useServerChatHistory, and the existing command palette. #3178 supplies server content-search support when merged; do not duplicate it. No new search service or index.
Pre-implementation review gate: deduplicate cached mirrors only against the same proven scoped identity actually visible in current successful server results. Outage, stale pages and filters must not hide the only reachable cached copy. Both groups expose loading/no conversations/no matches/unavailable/partial states; local DB failures cannot become successful empty pages. Duplicate titles need origin plus distinguishing character/mode/date/excerpt. Nested disclosure closes one layer and returns focus to a visible control. Reuse existing paged loaders and the existing character_id endpoint filter.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
