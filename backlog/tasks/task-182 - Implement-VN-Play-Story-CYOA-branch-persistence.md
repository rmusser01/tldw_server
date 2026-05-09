---
id: TASK-182
title: Implement VN Play Story/CYOA branch persistence
status: In Progress
assignee: []
created_date: '2026-05-09 19:24'
updated_date: '2026-05-09 20:13'
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
- [ ] #4 VN Play API tests and docs cover Story choice branch behavior, branch_path list shape, invalid_choice_id, choice_not_allowed, retry_last_turn_not_failed, and failure-only retry semantics.
- [ ] #5 Focused VN Play tests, Bandit on touched backend scope, and diff hygiene are run and recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 complete: added VNPlayRepository.record_story_choice_selection(), refactored append_event through transaction-local _insert_event(), and added repository coverage for happy path plus replay/scene-version rollback guards. Verification rerun by controller: python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py -q passed with 7 passed, 5 warnings; Bandit wrote /tmp/bandit_vn_play_story_choice_task1_controller_after_fix.json with no reported findings; git diff --check HEAD^ HEAD was clean. Commits: 986092c3a and bc139107e.

Task 2 complete after review fix: wired Story choice validation into submit_turn, added mode/input errors, kept custom_action non-branching, passed selected choice metadata to the adapter context, and moved decisive visible-choice/window revalidation into VNPlayRepository.record_story_choice_selection. Controller verification: python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py tldw_Server_API/tests/VN_Play/test_vn_play_state.py -q passed with 34 passed, 5 warnings; Bandit wrote /tmp/bandit_vn_play_story_choice_task2_controller_after_atomic_fix.json with no reported findings; git diff --check HEAD^ HEAD was clean. Commits: 59c54877f and e6ce9cf47.

Task 3 complete: rewrote retry_last_turn to create a new retry turn request from the latest retryable input-bearing failed request, reuse the original input_event_id, avoid appending duplicate user_turn/choice_selected events, and keep Story branch rows unchanged during retry. Added retry regression coverage for failed Story choices and completed Story choices. Verification: python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q passed with 22 passed, 5 warnings; Bandit wrote /tmp/bandit_vn_play_story_retry_task3_after_wrap.json with zero findings; git diff --check was clean. Commit: f44612490.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
