---
id: TASK-13358
title: Triage the 8 remaining red tests in the ungated MCP_unified tree
status: To Do
assignee: []
created_date: '2026-09-23 15:04'
labels:
  - testing
  - mcp
  - ci
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Blocks gating `app/core/MCP_unified/tests` in CI (TASK-13291). The tree is **13 failed / 3325 passed / 13 skipped** after the two fixes shipped with the triage; 5 of the 13 are covered by TASK-13375 (licensing) and TASK-13374 (publish workflow). These are the other 8.

They share one root cause -- the tree has never run in a CI job, so drift accumulated unseen. Each still needs its own verdict: **product defect**, **stale test assertion**, or **environment**. Do not xfail them as a batch; that reproduces the invisibility this task exists to end.

| Test | Symptom | First read |
|---|---|---|
| `test_refresh_token.py::test_refresh_token_rotation_flow` | `TypeError: refresh_token() missing 1 required positional argument: 'request'` | Stale test. The endpoint gained `request: Request` for `_require_demo_auth_enabled(request)`; the test predates that guard and must now supply a Request and enable demo auth. |
| `test_gateway_tool_discovery.py::test_tool_discovery_module_keeps_package_boundary_clean` | `FileNotFoundError: mcp_unified/gateway/tool_discovery.py` | Likely stale. Asserts on a path in the standalone package tree that does not exist; confirm whether the module moved or was renamed. |
| `test_gateway_fastapi_package.py::test_gateway_status_includes_package_boundary_metadata` | `StopIteration` | An unguarded `next()` over an empty match -- the metadata it looks for is absent. Establish whether the metadata should be there (product) or the test's selector is stale. |
| `test_gateway_admin_auth.py::test_gateway_admin_auth_does_not_gate_status_or_jsonrpc` | status payload has extra keys vs expected `{..., 'version': '0.0-test'}` | Probably an exact-dict assertion that new status fields broke. Decide whether the assertion should be subset-based. |
| `test_flashcards_module.py::test_flashcards_export_rejects_cross_workspace_card_in_apkg_path` | `KeyError: 'rows'` | **Look at this one first.** It is a cross-workspace isolation assertion. A KeyError suggests it fails before reaching the isolation check, so the isolation property is currently unverified either way. Relates to the standing cross-user isolation work. |
| `test_external_federation_integration.py::test_external_federation_module_integration_exposes_and_executes_virtual_tools` | `ValueError: Unknown external server: docs` from `services/mcp_credential_broker_service.py:150` | Fixture registers an external server named `docs` that the broker does not know. Registration drift or a missing test fixture. |
| `test_codegraph_module.py::test_codegraph_index_and_files_roundtrip` | `assert 1 == 2` | A count is off by one; needs reading. |
| `test_server_batch_and_formatting.py::test_tools_call_dict_result_is_json_content` | got `{'ok': True, ...'dict'...}` want `{'ok': True, 'x': 7}` | The dict result is being wrapped rather than serialized as JSON content. Could be a real protocol-shape regression. |

Sequence: settle the flashcards isolation one first (security), then the two `gateway_*` boundary ones together (same package-split cause), then the rest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the 8 has a recorded verdict: product defect, stale assertion, or environment
- [ ] #2 Product defects are fixed or have their own task; stale assertions are corrected
- [ ] #3 The flashcards cross-workspace isolation property is actually verified, not merely no longer erroring
- [ ] #4 The tree reaches zero failures so TASK-13291 can gate it
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
