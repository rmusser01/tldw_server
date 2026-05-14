---
id: TASK-322
title: Surface Persona runtime explorer diagnostics
status: Done
assignee: []
created_date: '2026-05-14 01:13'
updated_date: '2026-05-14 01:25'
labels:
  - persona
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1644'
  - 'https://github.com/rmusser01/tldw_server/pull/1647'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose trace-safe Persona runtime explorer diagnostics from the optional Persona Live planning path so fallback, circuit-open, and safe-denial outcomes are visible to clients/operators without changing Buddy rendering, visual-pack behavior, VN/CYOA behavior, or normal disabled-mode runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Enabled runtime explorer planning emits bounded diagnostics for fallback or safe-denial outcomes.
- [x] #2 Diagnostics never include raw user messages, prompts, memory, candidate text, plan args, provider exceptions, or exemplar content.
- [x] #3 Disabled runtime explorer behavior remains unchanged and emits no runtime-explorer notice.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_ws_dialogue_tree_runtime.py::test_runtime_explorer_fallback_notice_is_bounded_and_trace_safe tldw_Server_API/tests/Persona/test_persona_ws_dialogue_tree_runtime.py::test_runtime_explorer_disabled_emits_no_runtime_diagnostic_notice -q --tb=short failed because no RUNTIME_EXPLORER_FALLBACK notice was emitted.

Green verification: focused runtime websocket/explorer tests passed: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_ws_dialogue_tree_runtime.py tldw_Server_API/tests/Persona/test_runtime_explorer.py -q (26 passed). git diff --check passed. Bandit on tldw_Server_API/app/api/v1/endpoints/persona.py exited 0 with no findings in /tmp/bandit_persona_runtime_diagnostics.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a persona-only runtime explorer diagnostics slice for #1644. When the optional runtime explorer is enabled, fallback/circuit-open/safe-denial outcomes now emit bounded Persona Live notice diagnostics without exposing raw prompts, messages, memory, candidate text, plan args, provider exception messages, or exemplar content. Disabled mode still emits no runtime-explorer notice. Updated the Persona README and focused websocket runtime tests.
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
