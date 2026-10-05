---
id: TASK-13455
title: 'Chat UX stage 2: focused composer and complete conversation discovery'
status: In Progress
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
Stage 2 work started in bounded child TASK-13455.1 for lazy owned On this device discovery, qualified Temporary preview and normal selection. Latest dev remains c2260945ee after fresh fetch on 2026-10-05. This child does not close complete discovery, history-first preferences, focused composer, older character-session search, offline mirrors/import, Temporary Q3 retention or remaining approved stages.
TASK-13455.1 completed: qualified lazy On this device discovery and readonly Temporary preview, 326 bounded tests, frontend typecheck, no added touched lint, native IndexedDB evidence and two independent reviews. Remaining Stage 2 acceptance criteria and broader D7/program requirements remain open; continue with history-first navigation and complete active-character session paging.
TASK-13455.2 completed history-first navigation and retained expansion preferences, scoped always-reachable Find conversations, precise Search commands labels and qualified delayed focus. 139 bounded tests, frontend typecheck, no added touched lint, English parity and both independent reviews pass. Whole Stage 2 remains open: combined discovery/offline mirrors, complete character-session access and focused composer are next required units.
TASK-13455.3 started for complete selected-character session discovery using existing character_id filtering and paged history; independently bounded other-character recents, owned request/selection, failure/offline states and actual-English parity remain in scope. Latest dev c2260945ee verified against baseline 3485d54c42.
TASK-13455.3 completed selected-character filtering before sampling, All sessions pagination, bounded other-character recents, account/global qualification, initial verification and failed-page retry, imperative offline/disabled fences and duplicate references. 245 bounded tests, frontend typecheck, no added lint, 50 English mirror entries and both independent reviews pass. Native/live H1, very large-history, geometry/VoiceOver and whole-program journey acceptance are not established by this child. Focused composer and combined source discovery/offline mirrors remain required; parent acceptance criteria stay open pending complete stage evidence.
TASK-13455.4 started for approved Casual Customize chat hierarchy, recognizable active settings, Pro parity and keyboard disclosure/focus using existing shared full-page editor paths. Baseline 242f7aa268 and latest dev c2260945ee verified; no compact-sidepanel or whole-program completion claim.
TASK-13455.4 complete: Casual Customize hierarchy/active summaries, Pro parity, preserved listeners/preferences, full-page narrow reachability and qualified keyboard focus.268 affected tests,41 settled selector tests, frontend typecheck, zero added lint and two independent reviews pass. Bandit N/A. Combined Server/On this device query and honest cached-mirror outage behavior remain next; whole Stage2/program acceptance criteria remain open. Fresh dev025627214c requires branch update and affected settings regression.
TASK-13455.5 started for one-query All discovery and qualified cached server previews with exact visible-success deduplication. Latestdev025627214c now integrated; settings/composer/assistant104post-rebase tests pass. Reuse current lists/query/DB/preview authority; keep outage copies reachable and DB failures honest.
Independent discovery review identified the pre-existing nested folder/tag continuation boundary; tracked as TASK-13455.6 and kept open before Stage 2 closure. TASK-13455.5 native Chromium production-DB harness passed 2002 scoped histories across 67 pages (median9ms,p95 12ms), exact successful-visible dedup and retired-copy recovery, no foreign exposure or preview writes. Harness timings are this-machine storage reads, not packaged/live UI or VoiceOver acceptance. Bounded discovery regression14suites403passed; remaining static/review checks in progress.
TASK-13455.5 completed after two independent final reviews with no substantiated Critical/Important introduced findings. 14 suites/403 passed, types, entrypoint compile, 73 locale keys and no added touched lint; native Chromium IndexedDB 2002-row storage evidence recorded. Parent Stage 2 remains In Progress: TASK-13455.6 nested Folder/Tag continuation and later native/live/accessibility gates remain open.
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
