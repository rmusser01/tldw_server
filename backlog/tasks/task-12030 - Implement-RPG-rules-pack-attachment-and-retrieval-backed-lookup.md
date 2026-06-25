---
id: TASK-12030
title: Implement RPG rules-pack attachment and retrieval-backed lookup
status: To Do
created_date: 2026-06-25 23:29
dependencies:
- TASK-12029
labels:
- rpg
- ttrpg
- rag
- backend
- implementation
priority: high
documentation:
- Docs/superpowers/specs/2026-06-25-rpg-rules-pack-attachment-retrieval-design.md
- Docs/superpowers/plans/2026-06-25-rpg-rules-pack-attachment-retrieval-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved RPG rules-pack attachment feature from TASK-12029. Campaigns and sessions should attach user-owned media items or media collections as rules references, then use scoped retrieval to augment RPG rules lookup and context building with optional grounded answer mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Campaigns and sessions can list and replace normalized media_item/media_collection rules-pack refs with whole-list writes, expected_version checks, idempotency replay, and server-owned timestamps.
- [ ] #2 New REST endpoints expose campaign/session rules-pack ref list and replace operations with RPG permissions plus media.read requirements.
- [ ] #3 Session creation copies campaign refs by default while explicit session refs can diverge from campaign refs.
- [ ] #4 Rules lookup blends user-provided scoped retrieval snippets with bundled citation-only references, reports diagnostics, and never falls back to broad RAG or web search.
- [ ] #5 Answer mode generates grounded answers only from retrieved snippets using the existing async chat service and returns citation IDs limited to lookup evidence.
- [ ] #6 Session context building includes lookup-mode evidence within existing bounds and never invokes answer generation.
- [ ] #7 MCP tools expose the same ref-management and lookup semantics as REST.
- [ ] #8 Focused RPG, API, MCP, privilege catalog, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
- [ ] #7 Implementation plan tasks completed or consciously split into follow-up tasks with links.
- [ ] #8 Focused pytest commands recorded with results.
- [ ] #9 Privilege route catalog check recorded when endpoint metadata changes.
- [ ] #10 Bandit JSON report path and result recorded for touched Python scope.
- [ ] #11 Final summary explains what changed and why the chosen integration boundaries were used.
<!-- DOD:END -->
