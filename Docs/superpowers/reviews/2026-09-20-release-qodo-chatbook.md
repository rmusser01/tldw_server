# Chatbook Qodo 10/11/12 compatibility results

Worktree: `/private/tmp/tldw-chatbook-release-compat`
Branch: `codex/server-0.1.43-client-compat`; base `ab57681864`.
Backlog: TASK-32881, Done via MCP, all five ACs checked. No commit/push performed.
ADR: `backlog/decisions/174-server-sharing-release-contracts.md` (written before implementation).

## Dispositions

- **10 — fixed.** Clone requests serialize canonical `name`, retaining `new_name` as a Python input alias. Frozen request owns one validated excluded idempotency key, sent as `Idempotency-Key`. Both Sharing service families accept caller-retained keys. Receipts preserve operation identity/status/progress/result/readiness/warnings/error/poll metadata and older job responses. Local authenticated receipt route uses validated operation UUID; arbitrary remote poll URLs are not followed. Server-authorized recipient receipts intentionally survive revocation.
  - Actual mounted Sharing panel owns app-lifetime replay keys: configured server ID/base URL + stable authenticated-user authority (existing provider resolver, credential-refresh independent) + share ID + exact server whitespace-normalized name. Same-input retry, remount and A/B/A account/input switches reuse keys. Explicit Start another clone clears only selected active identity. Quota100 fails closed, never evicts uncertain requests. Invalid reset input is contained and reset stops Button event propagation. Process restart does not persist this memory; callers requiring restart recovery must persist request/receipt lifecycle, as documented.
- **11 — fixed.** Explicit source-page API preserves items, pagination, summary and partial_errors and forwards filters/offset/limit. Canonical source_id/origin_url plus legacy id/url parsing/accessors remain available. Existing list API follows all pages; stalled pagination raises rather than silently returning incomplete results. Both Sharing families expose pages and normalize canonical identity.
- **12 — fixed.** Notes scope → service → HTTP deletion forwards caller-selected dataset_id, expected_version, idempotency_key and reason. No fetch-latest-version fallback; stale409 and missing428 remain visible. Existing workspace/local policy checks remain. Documentation shows selected link version usage.

## Files

Production:
- tldw_chatbook/tldw_api/{sharing_schemas.py,client.py,__init__.py}
- tldw_chatbook/Sharing/{server_sharing_service.py,server_sharing_scope_service.py}
- tldw_chatbook/Sharing_Interop/{server_sharing_service.py,sharing_scope_service.py}
- tldw_chatbook/Notes/{server_notes_workspace_service.py,notes_scope_service.py}
- tldw_chatbook/UI/Sharing_Panel.py

Tests:
- Tests/tldw_api/test_sharing_release_contracts.py (new)
- Tests/Sharing/test_clone_panel_replay.py (new)
- Tests/tldw_api/test_sharing_client.py
- Tests/Sharing/test_server_sharing_service.py (standard private_profile_test isolation for two existing provider-factory cases)

Docs/task:
- backlog/decisions/174-server-sharing-release-contracts.md
- backlog/decisions/README.md
- backlog/docs/server-sharing-release-client.md
- backlog/docs/lessons-testing-evidence.md (real caller-lifecycle incident)
- backlog/tasks/task-32881 - Align-connected-sharing-and-Notes-clients-with-server-release-contracts.md

## Verification

From worktree above:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate && PYTHONPATH="$PWD" python -m pytest Tests/tldw_api/test_sharing_release_contracts.py Tests/tldw_api/test_sharing_client.py Tests/tldw_api/test_notes_workspace_client.py Tests/Sharing Tests/Notes/test_server_notes_workspace_service.py Tests/Notes/test_notes_scope_service.py -q --tb=short --basetemp=/private/tmp/qodo-chatbook-final2-pytest
```

**143 passed in 4.61s**, log `/tmp/qodo-chatbook-final2-tests.log`.

Real httpx transport checks canonical JSON/header, lost-response replay, succeeded/failed operation receipts and polling, source pagination/legacy aliases/partial errors, no-progress rejection, both service families, exact Notes query preconditions, and 409/428 preservation. Real mounted Textual panel goes through real scope/service and httpx transport for timeout/remount/replay, explicit next copy, canonical whitespace, account/server isolation, no quota eviction, invalid reset input. Red-before-green logs: `/tmp/qodo-chatbook-red.log`, `/tmp/qodo-chatbook-panel-red.log`, `/tmp/qodo-chatbook-scoped-red.log`.

Independent reviewer `/root/qodo_transport` re-ran mounted panel tests **2 passed** and found no remaining actionable issue. Report `/tmp/qodo-chatbook-independent-review.md`, log `/tmp/qodo-chatbook-reviewer-panel-final.log`.

Bandit1.9.4 from project venv, all ten touched production Python paths: **0 findings, 0 parse errors**. Artifact `/tmp/qodo-chatbook-bandit-final.json`. Invocation constructed tracked changed production Python paths via `git diff --name-only`, then `python -m bandit <paths> -f json -o /tmp/qodo-chatbook-bandit-final.json`.

Ruff explicit new tests + sharing schema + changed sharing client test: **all checks passed**. New tests/schema `ruff format --check`: **3 files already formatted**. Tracked changed-code ranges formatted with Ruff range formatting (preserves unrelated legacy formatting). `git diff --check`: clean.

Whole touched tracked Python scope has existing lint debt: **868 current diagnostics vs 870 at exact HEAD**, **zero new diagnostic occurrences** comparing `(filename, code, message)` counters with `--stdin-filename` for baseline. Artifact `/tmp/qodo-chatbook-static-comparison.json`; script `/tmp/check-chatbook-static.py`. In particular Sharing_Panel's four existing N999/UP035/UP037/BLE001 diagnostics remain. Do not describe whole-scope lint as zero warnings.

Additional legacy UI verification was attempted: `Tests/UI/test_tools_settings_window.py -k sharing` has **3 failures before panel construction** in app_factory/load_settings `raw_source_selection_changed`. Standard profile helper did not resolve that broad pre-existing harness; those trial test edits were reverted. Logs `/tmp/qodo-chatbook-ui-existing.log`, `/tmp/qodo-chatbook-ui-isolated.log`. Focused mounted panel remains green. No full repository suite run.
