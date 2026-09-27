# Qodo transport review remediation

Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42`
Tracking: TASK-13263.1. No commits, branch changes, or publications performed.

## Dispositions

- **1 — Fixed; confirmed valid.** Brokered runtime credentials previously overrode transport-owned headers. The shared runtime credential boundary now rejects reserved header names case-insensitively before connecting or dispatching in both Streamable HTTP and legacy SSE. Reserved names include MCP session/protocol, content-type, accept, host, framing, connection, and encoding headers. Safe reason code `invalid_runtime_headers` exposes no credential values. Existing authorized per-call authentication tests remain passing.
- **22 — Fixed; confirmed valid.** Streamable HTTP JSON, matching SSE event responses, and legacy SSE correlated request responses now use one shared base validator requiring a dictionary with `jsonrpc == "2.0"`. Missing markers, JSON-RPC 1.0, and numeric versions fail with `invalid_response`. Unmatched malformed legacy SSE notifications remain ignored.
- **13 — Fixed; confirmed valid.** Added exactly one integration category marker to `test_guarded_websocket_replays_large_fragmented_message_exactly_without_compression`, preserving asyncio. Category-filtered collection selects it.
- **14 — Fixed; confirmed valid.** Replaced private `_idempotency`, `_create_finalizer`, and `_finalizers` accesses in the identified test with a registered blocking write module, a public `process_request` tool call carrying an idempotency key, and public `server.shutdown()`. Assertions observe returned output and operation-before-module-teardown ordering; the test verifies shutdown remains pending while the operation is blocked. Added a unit category marker.
- **23 — Fixed; confirmed valid.** Moved `WebhookKeyError` into central `app/core/exceptions.py`, keeping the error code and message behavior. Webhook consumers import the central definition, and the crypto module and package retain compatible exports. Regression test checks identity and actual key-load failure behavior through the central class.

## Changed files (repository-relative)

- apps/mcp-unified/src/mcp_unified/federation/http_transport.py
- tldw_Server_API/app/core/MCP_unified/tests/test_http_external_transport.py
- tldw_Server_API/app/core/MCP_unified/tests/test_guarded_slides_websocket.py
- tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
- tldw_Server_API/app/core/exceptions.py
- tldw_Server_API/app/core/Admin_Webhooks/crypto.py
- tldw_Server_API/app/core/Admin_Webhooks/__init__.py
- tldw_Server_API/app/core/Admin_Webhooks/control_plane.py
- tldw_Server_API/app/core/Admin_Webhooks/delivery.py
- tldw_Server_API/app/core/Admin_Webhooks/key_rotation.py
- tldw_Server_API/tests/Admin_Webhooks/test_crypto.py

## Verification

All Python commands used `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate` first.

- Red: 26 reserved-header parameter cases failed on the unexpected connection attempt before implementing validation. Six Streamable HTTP invalid-version cases and three legacy SSE invalid-version cases failed because no exception was raised. The central-exception test failed on missing export. Logs: `/tmp/qodo-transport-red.log`, `/tmp/qodo-transport-version-red.log`, `/tmp/qodo-legacy-sse-red.log`, `/tmp/qodo-webhook-red.log`.
- `python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_http_external_transport.py tldw_Server_API/app/core/MCP_unified/tests/test_guarded_slides_websocket.py --tb=short`: **74 passed** (8 existing warnings), run with approved loopback socket access. `/tmp/qodo-transport-green.log`.
- `python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py --tb=short`: **137 passed** (3 existing warnings). `/tmp/qodo-extraction-green.log`.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest --confcutdir=tldw_Server_API/tests -q tldw_Server_API/tests/Admin_Webhooks/test_crypto.py tldw_Server_API/tests/Admin_Webhooks/test_control_plane.py tldw_Server_API/tests/Admin_Webhooks/test_executor.py tldw_Server_API/tests/Admin_Webhooks/test_key_rotation.py --tb=short`: **206 passed** (4 existing warnings plus pytest temporary-directory cleanup warnings). `/tmp/qodo-webhook-scoped.log`.
- `python -m pytest -q --collect-only -m integration tldw_Server_API/app/core/MCP_unified/tests/test_guarded_slides_websocket.py`: exactly the identified test selected, **1/9 collected**. `/tmp/qodo-websocket-marker.log`.
- `python -m ruff check` covering all 11 changed files: **all checks passed**.
- `git diff --check` covering all changed files: **passed**.
- Scoped Bandit on all seven changed production files: **no new findings**. Three B110 findings remain in unchanged lines: control_plane.py:1598 and delivery.py:1129,1825. Running Bandit against HEAD copies confirms the same findings at the same lines. Reports: `/tmp/bandit_qodo_transport.json`, `/tmp/bandit_qodo_transport_baseline.json`.

Initial combined pytest invocation hit the repository's non-top-level `pytest_plugins` collection limitation; running the backend test scope with `--confcutdir=tldw_Server_API/tests` resolved it. The broad 775-test Admin_Webhooks run was interrupted while its PostgreSQL migration fixture waited for an unavailable database; the 206 tests directly exercising touched consumers were then run successfully. No claim is made about the interrupted full webhook suite.
