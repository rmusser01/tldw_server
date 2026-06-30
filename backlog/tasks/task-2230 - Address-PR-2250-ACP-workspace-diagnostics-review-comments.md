---
id: TASK-2230
title: Address PR 2250 ACP workspace diagnostics review comments
status: Done
assignee: []
created_date: '2026-06-03 06:22'
updated_date: '2026-06-03 06:35'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2250'
  - TASK-2227
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace context builder logs non-critical context failures with bounded context instead of silently swallowing them.
- [x] #2 ACP endpoint test stub has explicit return typing.
- [x] #3 workspace-live-e2e cannot produce real workspace_env pass evidence without ACP_E2E_WORKSPACE_ID, and evidence records workspace_id_source.
- [x] #4 ACP session list filtering uses dynamic predicates so workspace_id queries can use indexes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented PR #2250 review fixes: logged bounded workspace-context registry enrichment failures, added get_session_metadata return typing, made ACP_E2E_WORKSPACE_ID required for workspace-live-e2e and added workspace_id_source evidence, and replaced optional OR list_sessions filters with fixed-column dynamic predicates so workspace_id queries can use indexes. Verification so far: focused RED tests failed before implementation; focused tests now pass; helper + ACP sessions DB suites pass with 98 passed; ACP endpoint suite passes with 27 passed. Bandit on helper + ACP endpoint exits 0. Bandit on helper + endpoint + ACP_Sessions_DB exits nonzero only on pre-existing findings elsewhere in ACP_Sessions_DB.py; new list_sessions SQL lines are nosec B608 with fixed-column/bound-param rationale and no longer reported.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all concrete PR #2250 review comments. Workspace context enrichment now logs bounded warning context instead of silently swallowing registry failures. The ACP endpoint test stub has an explicit return type. workspace-live-e2e now requires ACP_E2E_WORKSPACE_ID, refuses without it, records workspace_id_source=env, and docs state that runs without a real Research Workspace id are not valid evidence. ACP session listing now builds direct fixed-column predicates for provided filters, with a regression test verifying workspace_id filtering no longer emits IS NULL OR predicates. Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py -> 125 passed, 6 warnings. git diff --check -> clean. workspace-live-e2e live check -> exit 2 safe refusal missing TLDW_E2E_SERVER_URL, TLDW_E2E_API_KEY, ACP_AGENT_PROFILE, ACP_E2E_WORKSPACE_ID. Bandit helper+endpoint scope exits 0. Bandit including ACP_Sessions_DB exits nonzero on pre-existing findings elsewhere in that file; the new list_sessions dynamic SQL lines are nosec B608 with fixed-column/bound-param rationale and are no longer reported.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused tests cover changed behavior.
- [x] #8 Bandit runs on touched Python files.
- [x] #9 PR branch rebased and pushed.
<!-- DOD:END -->
