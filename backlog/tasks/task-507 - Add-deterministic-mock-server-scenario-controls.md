---
id: TASK-507
title: Add deterministic mock-server scenario controls
status: Done
labels:
- mock_openai_server
- onboarding-uat
modified_files:
- mock_openai_server/mock_openai/config.py
- mock_openai_server/mock_openai/server.py
- mock_openai_server/mock_openai/responses.py
- mock_openai_server/tests/test_server.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 for the onboarding UAT harness: add static config-driven fail-once scenario controls to mock_openai_server and cover chat/embeddings matching plus config aliases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented deterministic mock OpenAI scenario failures for chat completions and embeddings. Added ScenarioFailure config parsing with both `type` and `error_type` aliases, fail-once counters scoped by config instance, startup counter reset, and request matching via existing ResponsePattern. Added focused tests for chat fail-once, embeddings fail-once, and alias parsing. Also made two narrow baseline fixes in mock_openai_server: default chat responses now echo the requested model, and the async streaming test uses ASGITransport instead of real DNS. Verification: focused new tests passed after red/green; full command `RUN_MOCK_OPENAI=1 /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest mock_openai_server/tests/test_server.py -q` reported `22 passed, 3 warnings`. Bandit command wrote `/tmp/bandit_mock_openai_scenarios.json` and reported three LOW B311 findings on pre-existing random.random() mock embedding/error simulation lines; no new scenario-control findings were identified.
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
