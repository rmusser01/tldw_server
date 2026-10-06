---
id: TASK-13501
title: Router-groups contract tests fail when run alone or split across xdist workers
status: Done
labels:
- tests
- router-groups
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four tests in tldw_Server_API/tests/Services/test_router_groups_contract.py passed only when earlier tests in the same file had already imported real endpoint modules. Each fakes a module through sys.modules, and a real module it then imports needs a name from the fake: endpoints.media is faked as a plain module so media.ingest_jobs cannot import (two tests); notes_graph_suggestions imports _normalize_note_id from the faked notes_graph; prompts, chat and character_chat_sessions import get_configured_providers from the faked llm_providers; prompt_studio_websocket imports _authenticate_ws from the faked agent_client_protocol. Reported by the peer session on clean origin/dev 27ce976387; on dev, running the file with -n 4 --dist load also fails two of them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test in the file passes when run alone in its own process
- [ ] #2 The whole file passes serially and with pytest-xdist --dist load
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixes, all in the test file: fake media.ingest_jobs next to the media fake (2 tests); fake notes_graph_suggestions next to notes_graph; in test_qodo_reviewed_router_policy_regressions, resolve content routers before installing the llm_providers/vlm fakes, which only the core specs need (faking each importer one by one did not converge: prompts, then character_chat_sessions, then chat); fake prompt_studio_websocket next to agent_client_protocol.
Verification: all 178 tests run one per process: 0 failures (dev: 4). Whole file: 178 passed. -n 4 --dist load: 178 passed (dev: 2 failed). ruff clean. Bandit: not applicable (test-only change).
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
