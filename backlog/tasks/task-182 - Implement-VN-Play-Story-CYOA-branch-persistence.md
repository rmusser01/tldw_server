---
id: TASK-182
title: Implement VN Play Story/CYOA branch persistence
status: Done
assignee: []
created_date: '2026-05-09 19:24'
updated_date: '2026-05-09 20:19'
labels:
  - vn-play
  - story-mode
  - implementation
dependencies:
  - TASK-181
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1434'
documentation:
  - Docs/superpowers/specs/2026-05-09-vn-play-story-branch-persistence-design.md
  - >-
    Docs/superpowers/plans/2026-05-09-vn-play-story-branch-persistence-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement issue #1434 using the reviewed design spec and implementation plan. Persist selected Story choices as durable branch metadata with server-side choice validation, atomic branch/event/turn/scene-state persistence before model work, failure-only retry from the failed turn request input_event_id, and API/docs/test coverage. Keep branch_path list-shaped for the existing VNPlayBranchResponse contract and keep Story custom_action non-branching.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Story choice turns validate choice_id against persisted visible_choices and reject invalid or disallowed mode/input combinations with stable errors.
- [x] #2 Accepted Story choices atomically create a branch row, append turn_started and choice_selected, update the turn request, and persist scene state with active_branch_node_id set and visible_choices cleared before adapter/model work.
- [x] #3 Story retry-last-turn only retries failed model/parse/abandoned attempts using the failed turn request input_event_id and does not append duplicate choice_selected or create a new branch.
- [x] #4 VN Play API tests and docs cover Story choice branch behavior, branch_path list shape, invalid_choice_id, choice_not_allowed, retry_last_turn_not_failed, and failure-only retry semantics.
- [x] #5 Focused VN Play tests, Bandit on touched backend scope, and diff hygiene are run and recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 complete: added VNPlayRepository.record_story_choice_selection(), refactored append_event through transaction-local _insert_event(), and added repository coverage for happy path plus replay/scene-version rollback guards. Verification rerun by controller: python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py -q passed with 7 passed, 5 warnings; Bandit wrote /tmp/bandit_vn_play_story_choice_task1_controller_after_fix.json with no reported findings; git diff --check HEAD^ HEAD was clean. Commits: 986092c3a and bc139107e.

Task 2 complete after review fix: wired Story choice validation into submit_turn, added mode/input errors, kept custom_action non-branching, passed selected choice metadata to the adapter context, and moved decisive visible-choice/window revalidation into VNPlayRepository.record_story_choice_selection. Controller verification: python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py tldw_Server_API/tests/VN_Play/test_vn_play_state.py -q passed with 34 passed, 5 warnings; Bandit wrote /tmp/bandit_vn_play_story_choice_task2_controller_after_atomic_fix.json with no reported findings; git diff --check HEAD^ HEAD was clean. Commits: 59c54877f and e6ce9cf47.

Task 3 complete: rewrote retry_last_turn to create a new retry turn request from the latest retryable input-bearing failed request, reuse the original input_event_id, avoid appending duplicate user_turn/choice_selected events, and keep Story branch rows unchanged during retry. Added retry regression coverage for failed Story choices and completed Story choices. Verification: python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q passed with 22 passed, 5 warnings; Bandit wrote /tmp/bandit_vn_play_story_retry_task3_after_wrap.json with zero findings; git diff --check was clean. Commit: f44612490.

Task 4 complete: added API-level coverage for Story choice branch state in turn responses, invalid_choice_id on unknown Story choices, retry_last_turn_not_failed after completed Story turns, and branch_path list shape from GET /branches. Updated Docs/API-related/VN_PLAY_API.md with Story choice validation, non-branching Story custom_action, choice_selected payload, branch_path list shape, stable Story errors, and failure-only retry semantics. Endpoint code did not need changes because generic VNPlayTurnError already maps to HTTP 400 and conflict/model failures remain separately mapped. Verification: python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q passed with 23 passed, 5 warnings; git diff --check was clean. Commit: cdda06011.

Task 5 closeout verification complete: python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_state.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q passed with 59 passed, 5 warnings; Bandit wrote /tmp/bandit_vn_play_story_branch.json with zero findings across VNPlay_DB.py, service.py, constants.py, and vn_play.py; git diff --check was clean. Known skips/blockers: none.
<!-- SECTION:NOTES:END -->

## Final Summary

Implemented server-authoritative VN Play Story/CYOA branch persistence for issue #1434. The backend now validates Story choices against persisted visible choices, records accepted choices atomically as branch metadata plus turn_started/choice_selected events before model work, persists active_branch_node_id/cleared visible choices in scene state, and keeps Story custom_action non-branching. Retry-last-turn is now failure-only and rebuilds model context from the failed request's stored input_event_id, so failed Story choice retries reuse the original branch instead of appending duplicate choice_selected events or creating branch duplicates.

Added focused repository, service, state, and API coverage for valid/invalid Story choices, branch path list shape, pre-model scene state, mode validation, retry semantics, and endpoint-visible errors. Updated Docs/API-related/VN_PLAY_API.md to document choice validation, choice_selected payloads, branch_path shape, Story error codes, and failure-only retry behavior.

Verification: python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_state.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py tldw_Server_API/tests/VN_Play/test_vn_play_api.py -q passed with 59 passed, 5 warnings. Bandit on VNPlay_DB.py, service.py, constants.py, and vn_play.py wrote /tmp/bandit_vn_play_story_branch.json with zero findings. git diff --check was clean. Known skips/blockers: none.

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
