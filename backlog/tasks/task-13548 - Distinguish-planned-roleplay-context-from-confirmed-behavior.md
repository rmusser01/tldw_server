---
id: TASK-13548
title: Distinguish planned roleplay context from confirmed behavior
status: Done
labels:
- chat
- ux
- roleplay
dependencies:
- TASK-13456
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Local UI compatibility, an uninserted quick prompt and zero extra pins currently imply active behavior or no context. Repair shared state/presentation so selected settings and previews do not claim server activation or omit possible inherited character lore.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting or cancelling a quick prompt does not mark active system behavior before insertion.
- [x] #2 Character, persona and scene selection feedback describes plans and local drafts rather than unacknowledged activation.
- [x] #3 Zero added pinned context is distinguished from inherited lore; absent composition is not shown as a ready assembled request or a zero provider footprint.
- [x] #4 Causal regressions, both client types, authored static checks, independent reviews and bounded native evidence are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Accepted all-ten remediation design Stage3, latest origin/dev2c5f19d0328360d45bce99b8bae75ce252cc312f included in d55d98ed42. Duplicate planned-roleplay/truthful-context searches empty; current remote/open/local max13530/13526/13547. ADR required:no; repair selection/preview projection under existing request/owner authority and ADR049, without new activation state or transport. Compatibility statuses remain routing predicates in raw request preview. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_planned_roleplay_context_13548.md. Broader scoped settings ACK/partial retry and durable native/D7 prerequisites remain separate required program units.
Final root repair includes quick-slot roleplay/template/cockpit summaries, comparison/conflict notices, shared context chips and adjacent rail, scene draft copy, additional versus unknown inherited lore and absent composition/footprint. Causal red controls then464/464 checks in25 suites; both client types;28 TS/TSX effective lint133/133/zero added and23 smaller-file formatting/diff checks. Two independent final source reviews clear. Actual retained IAB/fresh preview shows final unknown/selection labels; all5 saved transcript hashes unchanged. Evidence and exact limits in master design TASK13548 section. Owned four-stage plan Complete and retired. ADR no, ADR049/059. Bandit N/A TS/TSX/docs. Scoped ACK/editor/native inheritance and D7/Resume/Syncv2 plus packaged/physical/AT/participant gates remain required and parent remains open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Pending quick insertion no longer claims system behavior; selected roleplay/context and scene drafts do not claim activation; zero additional pins and absent previews no longer assert absent inherited lore or zero/ready provider composition. Verified464 checks, both client types, static delta, two reviews and bounded same-conversation native evidence without saved transcript mutation.
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
