---
id: TASK-12142
title: Fix audit ACP reconnect broadcaster cleanup
status: Done
assignee: []
created_date: 2026-07-04 06:55
updated_date: 2026-07-04 18:59
labels:
- audit
- remediation
- mcp
- acp
- websocket
- reliability
dependencies: []
references:
- AUDIT-2026-06-27-MCP-002
- https://github.com/rmusser01/tldw_server/pull/2619
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/mcp-sandbox-agent-protocol.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/specialists/reliability-lifecycle.md
priority: low
modified_files:
- tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py
- tldw_Server_API/app/core/Agent_Client_Protocol/consumers/ws_broadcaster.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py
- tldw_Server_API/tests/Agent_Client_Protocol/test_ws_broadcaster.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address audit finding MCP-002: ACP reconnect WebSocket replay creates a temporary WSBroadcaster but the endpoint finalizer does not remove the replay connection or stop the broadcaster, leaving event-bus subscribers/tasks behind after disconnect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reconnect stream retains enough broadcaster lifecycle state for finalizer cleanup.
- [x] #2 Endpoint finalizer removes the replay connection and stops the temporary broadcaster after disconnect/error.
- [x] #3 Session event-bus subscriber count returns to baseline after last_sequence reconnect disconnect.
- [x] #4 WSBroadcaster supports unique consumer ids so concurrent reconnect replay broadcasters do not overwrite each other.
- [x] #5 Focused endpoint-level regression covers last_sequence reconnect cleanup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused regression coverage for endpoint reconnect-disconnect cleanup and broadcaster consumer-id isolation.
2. Make the reconnect broadcaster lifecycle explicit in acp_session_stream, including connection removal and stop in the finalizer.
3. Allow WSBroadcaster instances to use unique consumer ids while preserving the default consumer id for existing callers.
4. Run focused ACP websocket/broadcaster tests, Bandit on touched production scope, and diff whitespace validation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Validation notes:
- Original PR branch was stale after latest dev advanced; fetched origin/dev and rebased cleanly onto fd5c152b065c408e4e8ee5f08da41589f21cb7f5. Post-rebase merge-base matched origin/dev before validation.
- Red check failed as expected before the original implementation: reconnect endpoint left `ws_broadcaster` in the session event bus and WSBroadcaster did not accept a custom `consumer_id`.
- Reviewed Gemini feedback on PR #2619. The `nonlocal` suggestion is technically invalid for the current code because `reconnect_broadcaster` and `reconnect_conn_id` are assigned in the `acp_session_stream` function scope, not inside `_send_callback`. No endpoint code change was made for that suggestion.
- Strengthened the endpoint reconnect regression so it preloads the session event bus with a buffered completion event and asserts the reconnect stream receives the replayed payload before verifying cleanup. This prevents the test from passing trivially when the reconnect broadcaster path is not exercised.
- Targeted regression command passed after rebase and review follow-up: `.venv/bin/python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::test_acp_session_stream_reconnect_cleans_up_replay_broadcaster tldw_Server_API/tests/Agent_Client_Protocol/test_ws_broadcaster.py::test_ws_broadcaster_allows_unique_consumer_ids -q` (2 passed).
- Bandit passed with 0 findings: `.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py tldw_Server_API/app/core/Agent_Client_Protocol/consumers/ws_broadcaster.py -f json -o /tmp/bandit_acp_reconnect_cleanup_latest_dev.json`.
- `git diff --check` passed.
- Broader file command on latest dev still has one unrelated baseline failure: `test_acp_websocket.py::TestACPRunnerClientPermissions::test_determine_permission_tier_batch` expects `fs.write` to be `batch`, but current dev returns `individual`; the focused ACP reconnect and broadcaster regressions pass.
PR opened: https://github.com/rmusser01/tldw_server/pull/2619 (draft against dev). Draft status is intentional until the human-authored Change summary required by project policy is added.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented MCP-002 remediation for ACP reconnect replay cleanup and refreshed it against latest dev. `acp_session_stream` retains the temporary replay broadcaster and connection id, uses a per-connection WSBroadcaster consumer id, removes the replay connection, and stops the broadcaster in the endpoint finalizer. `WSBroadcaster` accepts an optional consumer id while preserving the existing default. The endpoint regression now proves the reconnect broadcaster path by replaying a buffered completion event and asserting the stream receives it before disconnect cleanup.

Verification on latest dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5:
- Merge-base matched origin/dev after rebase.
- Passed: `.venv/bin/python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::test_acp_session_stream_reconnect_cleans_up_replay_broadcaster tldw_Server_API/tests/Agent_Client_Protocol/test_ws_broadcaster.py::test_ws_broadcaster_allows_unique_consumer_ids -q` (2 passed).
- Passed: `.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py tldw_Server_API/app/core/Agent_Client_Protocol/consumers/ws_broadcaster.py -f json -o /tmp/bandit_acp_reconnect_cleanup_latest_dev.json` with 0 findings.
- Passed: `git diff --check`.
- Broader run note: `test_acp_websocket.py test_ws_broadcaster.py -q` has one unrelated current-dev failure in `TestACPRunnerClientPermissions::test_determine_permission_tier_batch`.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused ACP websocket and broadcaster tests pass.
- [x] #2 Bandit over touched production files reports no new issues.
- [x] #3 git diff --check passes.
- [x] #4 Backlog task contains verification evidence and final summary.
<!-- DOD:END -->
