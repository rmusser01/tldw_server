---
id: TASK-253
title: Support generated VN script choices
status: Done
assignee:
  - codex
created_date: '2026-05-11 00:40'
labels:
  - vn
  - scripted-generation
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1535'
documentation:
  - Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md
  - Docs/superpowers/specs/2026-05-10-vn-scripted-model-generation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 6 from Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md: allow generated choice_set output while keeping branch targets authored by scripts.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Inspect current scripted generation runtime, active revision/public state shaping, generated output parser, and literal choice selection path.
2. Add focused failing tests first for generated choice public state, selection to authored on_generated_choice, branch/event metadata, inactive revision rejection, and parser rejection of model target fields.
3. Implement generated choice exposure by overlaying active choice_set revision choices into public script state with source, generation_id, and revision_id without raw prompt/raw output fields.
4. Extend choice selection to accept only choices from the active revision at the current generation point, record generated-choice branch metadata, set last_generated_choice.* system variables, and jump only to authored on_generated_choice.
5. Tighten parser/schema handling so model-provided target/control fields on choice_set choices are forbidden.
6. Run focused VN Play tests, generated output parser tests, diff checks, and Bandit on touched production Python paths; update TASK-253 acceptance criteria and final notes with results.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generated choice selection jumps to on_generated_choice
- [x] #2 Generated choice metadata is stored in branch events
- [x] #3 Choice from inactive revision cannot be selected
- [x] #4 Model-provided target fields are rejected by parser
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
- Added generated `choice_set` projection from active generation revisions into scripted public state.
- Generated choices retain internal authored `on_generated_choice` targets only in private script position; public state omits targets and exposes source, generation_id, and revision_id.
- Selection validates generated choices against the active revision for the current generation point before branching, including revision-owned id/text/metadata and the private authored target from the persisted opcode snapshot.
- Selection records generated-choice metadata on `choice_selected` events and writes `last_generated_choice.id`, `.text`, and `.metadata` variables.
- Generated-choice branch events and later scripted continuation events inherit the active branch id so branch event filtering does not depend on replaying untagged events.
- Parser coverage now explicitly rejects model-provided `target` and `next_label` fields on generated choices.
<!-- SECTION:NOTES:END -->

## Verification
<!-- SECTION:VERIFICATION:BEGIN -->
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_scripted_generation_runtime.py tldw_Server_API/tests/VN_Play/test_vn_play_generated_outputs.py tldw_Server_API/tests/VN_Play/test_vn_play_branch_navigation.py -q --tb=short` -> 36 passed, 8 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/VN_Play -q` -> 189 passed, 8 warnings.
- `git diff --check` -> passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m compileall tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/app/core/VN_Play/branch_navigation.py tldw_Server_API/tests/VN_Play/test_vn_play_generated_outputs.py tldw_Server_API/tests/VN_Play/test_vn_play_scripted_generation_runtime.py` -> passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/VN_Play/service.py tldw_Server_API/app/core/VN_Play/branch_navigation.py -f json -o /tmp/bandit_vn_scripted_generation_task6.json` -> 0 findings.
<!-- SECTION:VERIFICATION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Supported generated VN `choice_set` runtime choices while keeping control flow authored by script `on_generated_choice`. Public state now exposes generated choice source and generation/revision IDs, selection records generated-choice metadata and system variables, active-revision membership and private target validation reject stale or tampered choices, generated branch continuations stay explicitly branch-tagged, and parser tests reject model-provided routing fields.
<!-- SECTION:FINAL_SUMMARY:END -->
