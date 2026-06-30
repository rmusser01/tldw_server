---
id: TASK-196
title: Design VN Play branch navigation API
status: Done
assignee: []
created_date: '2026-05-09 22:03'
updated_date: '2026-05-09 22:11'
labels:
  - vn-play
  - design
  - api
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1463'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/API-related/VN_PLAY_API.md
  - Docs/superpowers/specs/2026-05-09-vn-play-branch-navigation-api-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design spec for GitHub issue #1463: expose backend-owned VN Play Story/CYOA branch navigation data and define safe branch rewind/resume semantics for custom frontends. Scope is design-first before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design proposes a backend-owned branch navigation/read model for VN Play sessions.
- [x] #2 Design explicitly decides whether branch rewind/resume is in scope for this sprint and defines stale/in-flight turn behavior if included.
- [x] #3 Design preserves current Story choice idempotency retry checkpoint restore and Freeform behavior.
- [x] #4 Design includes API docs and backend test plan for custom frontend consumption.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the VN Play branch navigation API design spec for issue #1463. The design keeps the navigation read model backend-owned and derived from branches/events/scene state, adds a branch-navigation endpoint, extends event listing with optional branch-aware filtering, and includes guarded branch restore with stale-scene checks, active-turn checks, idempotent session action rows, lease recovery, and scene-version compare-and-swap. Self-review tightened the restore target model from branch_start to choice_point plus branch_latest because restoring immediately after choice_selected would leave many completed Story branches with no visible choices.

Reopened after design review. Follow-up fixes needed: shared session mutation lock between turns and restore actions, branch ownership semantics, action idempotency key scope, bounded branch-aware event filtering, stable warning payload shape, and choice_point parent/sibling semantics.

Addressed design review findings: required a shared session mutation gate via active_session_action_id so turns and restore actions cannot race; defined direct event_range versus subtree_event_range ownership; made restore idempotency keys session-global across restore action types with action_type in the request hash; bounded branch-aware event filtering fallback replay; added stable warning payload schema; and specified choice_point restore returns the parent choice-presented state with parent active branch rather than the selected branch.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wrote and revised the design spec for the next VN Play/CYOA sprint: backend-owned branch navigation plus guarded branch rewind/resume semantics for custom frontends. Review fixes now cover shared turn/restore mutation locking, direct-vs-subtree branch ownership, session-global restore idempotency scope, bounded replay fallback for branch event filtering, stable warning payload shape, and precise choice_point parent/sibling restore semantics. This is design-only; Bandit is not applicable until implementation code is touched.
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
