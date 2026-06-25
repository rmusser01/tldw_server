---
id: TASK-2424
title: Verify and remediate MCP Unified review findings
status: Done
assignee: []
created_date: 2026-06-23 18:27
updated_date: 2026-06-25 02:22
labels:
- review
- mcp
- security
- refactor
dependencies: []
modified_files:
- Docs/superpowers/plans/2026-06-24-mcp-protocol-tool-execution-refactor.md
- Docs/superpowers/specs/2026-06-23-mcp-protocol-tool-execution-refactor-design.md
- tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py
- tldw_Server_API/app/core/MCP_unified/auth/rbac.py
- tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py
- tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py
- tldw_Server_API/app/core/MCP_unified/config.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
- tldw_Server_API/app/core/MCP_unified/protocol.py
- tldw_Server_API/app/core/MCP_unified/protocol_types.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/app/core/MCP_unified/tool_execution/__init__.py
- tldw_Server_API/app/core/MCP_unified/tool_execution/coordinator.py
- tldw_Server_API/app/core/MCP_unified/tool_execution/dependencies.py
- tldw_Server_API/app/core/MCP_unified/tool_execution/hooks.py
- tldw_Server_API/app/core/MCP_unified/tool_execution/models.py
- tldw_Server_API/app/core/MCP_unified/tool_execution/reporting.py
- tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py
- tldw_Server_API/app/core/MCP_unified/tool_execution/security.py
- tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py
- tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
- tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py
- tldw_Server_API/app/core/MCP_unified/tests/test_idempotency_and_category.py
- tldw_Server_API/app/core/MCP_unified/tests/test_protocol_governance_preflight.py
- tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py
- tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py
- tldw_Server_API/app/core/MCP_unified/tests/test_security_hardening.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_coordinator.py
- tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py
- tldw_Server_API/app/core/MCP_unified/tests/test_validation_and_sanitization.py
- tldw_Server_API/app/core/MCP_unified/tests/test_ws_per_ip_caps.py
- tldw_Server_API/tests/MCP_unified/test_mcp_config_sanitization.py
- tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py
- tldw_Server_API/tests/MCP_unified/test_phase3_3_small_core_sanitizers.py
references:
- https://github.com/rmusser01/tldw_server/pull/2513
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify current MCP Unified module review findings, address validated issues with focused tests and security verification, and capture the protocol.py refactor brainstorming/spec workflow separately before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated review findings are recorded as accepted or rejected with evidence
- [x] #2 Accepted bug/security/reliability findings have regression tests written before production changes
- [x] #3 Focused MCP Unified tests, diff check, and Bandit touched-scope verification are recorded
- [x] #4 protocol.py refactor brainstorming produces an approved design/spec before implementation planning
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification pass completed before implementation. Validated findings: MCP legacy refresh/revocation state is process-local; fs.write_text bypasses structured fs.write preimage/receipt protections and virtual CLI write maps to it; metadata category web/utility falls back to read rate bucket; fallback RBAC USER role wildcard-executes arbitrary tools; stale WebSocket cleanup removes connections without decrementing per-IP counts; MCP configure_logging removes existing global Loguru sinks; invalid tool names return INTERNAL_ERROR despite being invalid params. Focused existing suite: selected MCP tests passed 9/9 in 229.16s, with live WebSocket test slow but completed.

Implemented remediation for validated MCP Unified findings: gated legacy refresh behind demo auth, routed fs.write_text through structured preimage-checked writer, moved virtual CLI write to fs.write create mode, preserved network/utility metadata categories for rate limiting, narrowed fallback RBAC user/moderator tool execution, decremented WS per-IP counts during stale cleanup, preserved non-MCP Loguru sinks, and mapped invalid tool names to INVALID_PARAMS. Verification: focused MCP regression slice passed (21 passed); HTTP refresh gate tests passed (2 passed); touched modules py_compile passed; direct logging preservation check passed; Bandit on touched implementation files passed with 0 findings.

Protocol.py refactor brainstorming completed. Approved direction: security-pipeline extraction for tools/call, keeping MCPProtocol as JSON-RPC facade. Design spec written at Docs/superpowers/specs/2026-06-23-mcp-protocol-tool-execution-refactor-design.md and self-reviewed for placeholders, contradictions, scope drift, and ambiguity.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Spec review fixes applied to Docs/superpowers/specs/2026-06-23-mcp-protocol-tool-execution-refactor-design.md: added a compatibility callback ledger requirement, made Stage 3 introduce a ToolExecutionReporter facade, clarified coarse-vs-deep authorization characterization tests, and resolved IdempotencyManager ownership expectations before/during runtime extraction.
Follow-up spec review fixes applied: added the reporter facade to ToolExecutionDependencies and required an import-boundary test so tool_execution modules cannot import MCPProtocol or MCP_unified.protocol.
Implementation plan written at Docs/superpowers/plans/2026-06-24-mcp-protocol-tool-execution-refactor.md. Plan covers characterization tests, shared type extraction, coordinator delegation, security/hooks/runtime/reporting extraction, callback ledger removal, focused verification, and Bandit. Idempotency ownership is resolved by keeping IdempotencyManager import-compatible from protocol.py while injecting the manager instance into runtime.
Implementation Task 1 completed via subagent workflow: added protocol compatibility export and tool_execution import-boundary tests to test_extraction_contracts.py. Verification intentionally red at this stage: selected tests produced 1 passed, 1 failed because tool_execution/ is created in a later task. Spec review passed; code-quality re-review found no Critical/Important issues aside from the planned missing package directory.
Implementation Task 2 completed via subagent workflow: added coordinator characterization tests in test_tool_execution_coordinator.py. Verification intentionally red at this stage: pytest collected 2 tests and both failed with ModuleNotFoundError because tool_execution.coordinator is created later. Spec review passed; code-quality review found no Critical/Important issues.
Implementation Task 3 completed via subagent workflow: added coarse and deep tools/call authorization-boundary reporting tests in test_tool_use_reporting_protocol.py. Focused verification passed (2 passed). Spec review passed; code-quality review suggestions were applied by switching to _protocol(recorder=...), asserting exactly one event, and checking runtime_surface/requested_tool_name; final quality re-review found no Critical/Important issues.
Task 4 protocol type extraction completed: moved InvalidParamsException, GovernanceDeniedError, ApprovalRequiredError, RequestContext, trusted compatibility-claims sentinels/helpers, and PreparedToolCall into tldw_Server_API/app/core/MCP_unified/protocol_types.py. protocol.py now imports/re-exports the moved names, keeps MCPRequest/MCPResponse/MCPError/ErrorCode/IdempotencyManager in place, and preserves earlier local protocol.py remediation edits. Verification: `source .venv/bin/activate; python -m py_compile tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/protocol_types.py; TEST_MODE=true ENABLE_TRACING=false OTEL_METRICS_EXPORTER=none python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_protocol_reexports_tool_execution_shared_symbols tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py -q` exited 0 with 8 passed, 3 warnings. Touched-scope Bandit: `python -m bandit -r tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/protocol_types.py -f json -o /tmp/bandit_mcp_protocol_types_task4.json` exited 0 with `errors: []` and `results: []`.
Task 4 complete: shared protocol helper/types moved into protocol_types.py with protocol.py re-exports preserved. Spec review approved. Code-quality review initially found a PreparedToolCall runtime annotation regression; fixed by restoring runtime BaseModule import and adding a get_type_hints regression assertion. Focused py_compile, pytest subset, Ruff F/I/UP, and reviewer Bandit check passed.
Implementation Task 5 completed via subagent workflow: created the tool_execution package skeleton with coordinator, dependency, reporting, model, and placeholder modules. Focused verification passed: import-boundary/coordinator tests 3 passed, py_compile on all new files passed, and Ruff F/I/UP on tool_execution passed. Spec review passed. Code-quality review found no Critical/Important issues; minor notes are to use or defer execution_origin_for_failure and tighten callback typing in later extraction steps.
Implementation Task 6 completed via subagent workflow: MCPProtocol now constructs ToolExecutionReporter/ToolExecutionDependencies/ToolExecutionCoordinator, delegates _handle_tools_call plus public prepare/execute methods through the coordinator, and preserves original bodies as _prepare_tool_call_inline/_execute_prepared_tool_call_inline. Direct external_access_evaluator wiring was enforced and incomplete test dependency bundles were updated. Code-quality review identified a monkeypatch restore recursion trap in __setattr__; fixed with failing-first regression test test_restoring_public_prepare_tool_call_does_not_recurse. Verification passed: focused Task 6 slice 14 passed, py_compile passed, Ruff F/I/UP passed; spec and quality re-reviews approved.
Implementation Task 7 completed via subagent workflow: ToolExecutionSecurity now owns core validation/resolution/hardening/integrity helpers, MCPProtocol constructs it and keeps compatibility wrappers, and prepare/execute inline paths call it directly. Spec review found class-level MCPProtocol._hash_arguments compatibility regression; fixed with failing-first test test_protocol_hash_arguments_supports_class_level_compatibility_call and shared ToolExecutionSecurity.hash_arguments_with_exceptions helper. Verification passed: focused Task 7 suite 13 passed, py_compile passed, Ruff F/I/UP passed, and worker Bandit touched-scope scan reported zero findings. Spec and code-quality re-reviews approved.
Implementation Task 8 completed via subagent workflow: ToolExecutionSecurity now owns tool RBAC/scope/API-key/allowed-tools/alias authorization helpers, protocol private wrappers remain for compatibility and monkeypatch seams, _prepare_tool_call_inline still uses wrappers for context/module/tool gates, and ToolExecutionDependencies gained optional api_key_scope_normalizer with fallback parsing preserved. Verification passed: required focused authorization/scope/sandbox/import-boundary slice 25 passed, py_compile passed, Ruff F/I/UP passed; worker Bandit touched-scope scan reported zero findings and optional scope/fallback tests passed. Spec and code-quality reviews approved.
Implementation Task 9 completed via subagent workflow: ToolExecutionHooks now owns pre/post hook context construction, hook decision coercion, payload shaping, hook-visible metadata copying/redaction, and hook execution; MCPProtocol constructs the helper and keeps compatibility wrappers. Code-quality review found two Important issues, both fixed with regression tests: composite hook_results are sanitized/allowlisted before being stored in request metadata, and replacing protocol._tool_call_hook_manager now syncs the extracted helper. Verification passed: new red tests now pass, focused Task 9 slice passed (13 passed locally; reviewer rerun 16 passed), py_compile passed for protocol.py/hooks.py, Ruff F/I/UP passed, and import-boundary review confirmed no tool_execution import/reference to MCPProtocol.
Implementation Task 10 completed via subagent workflow: ToolExecutionSecurity now owns policy/governance/path/external helpers plus prepare_tool_call, with governance service/store/lock state moved into the security helper. MCPProtocol now routes coordinator prepare through a hooks-aware wrapper and keeps compatibility delegators/wrappers for moved methods. Existing protocol monkeypatch seams are preserved through late-bound prepare callbacks and dependency sync. During verification, the required suite initially had one failure in the run-chain path-scope fixture; root cause was a stale test fixture still advertising legacy fs.write_text while the current virtual CLI write adapter intentionally uses structured fs.write with mode=create from the earlier security remediation. Updated that fixture to fs.write. Verification passed: exact failing test passed, required Task 10 suite passed (57 passed), py_compile passed for protocol.py/security.py/hooks.py and the touched path-scope test, Ruff F/I/UP passed for protocol.py/security.py/hooks.py, and Ruff F/UP passed for the touched path-scope test. Spec and code-quality reviews approved.
Task 11 complete: extracted prepared-call runtime execution into ToolExecutionRuntime, wired coordinator execution to runtime, kept protocol compatibility wrappers, and added runtime post-hook/config/idempotency seam regressions. Review follow-up fixed _idempotency dependency sync, dynamic get_config provider resolution, and protocol post-hook wrapper compatibility. Verification: 150 selected MCP Unified tests passed; py_compile passed for touched protocol/runtime/hooks/tests; ruff F,I,UP passed on touched files; production Bandit passed with 0 findings; git diff --check passed. Reviewer recheck approved.
Task 12 complete: ToolExecutionReporter now owns tool-use reporting internals, process-request failure recording, event construction helpers, and audit logging; protocol.py keeps compatibility wrappers and constructs/syncs the reporter. Review follow-ups fixed runtime audit boundary so runtime calls reporter.audit_tool_event, synced _tool_name_re into ToolExecutionSecurity after regex replacement, added regressions for event-builder restoration and regex sync, and cleaned Bandit B110 fallback pass sites in security.py with sanitized debug logging. Verification: 63 selected MCP Unified reporting/runtime tests passed; py_compile passed; ruff F,I,UP passed; production Bandit passed with 0 findings; git diff --check passed. Reviewer rechecks approved.
Task 13 cleanup/final review complete. Usage search found the remaining protocol.py wrapper methods are still used by in-repo callers, compatibility imports, or monkeypatch seams, so no additional wrappers were removed. Compatibility exports were verified, the plan ledger was updated to a final empty active state with resolved history, and all plan checklist stages were marked complete. Final review follow-ups fixed two validated issues: ToolExecutionSecurity now delegates governance rollout mode resolution through the shared server config resolver with fallback, and ToolExecutionHooks no longer exposes raw hook decision messages or arbitrary hook metadata in caller-facing denial/approval responses while preserving sanitized hook reporting metadata. Additional final-contract fixes removed a direct AuthNZ exception import from server.py and corrected the standalone MCP extraction-contract root path. Final verification passed: py_compile for touched MCP modules, focused pytest slice 217 passed, Ruff F/I/UP passed, Bandit touched implementation scope reported 0 findings, and git diff --check passed. Final quality reviewer approved with no remaining Critical/Important/Minor findings.
Integration cleanup before commit: expanded the verification scope to include the earlier validated MCP Unified fixes and the protocol.py refactor together. Fixed a brittle fs.glob size-unavailable test shim so it raises on the intended follow_symlinks=False metadata read, and applied Ruff import/type cleanup in the expanded MCP touched test scope. Verification passed after cleanup: py_compile for touched MCP implementation files, expanded pytest slice 384 passed, Ruff F/I/UP passed, and Bandit commit-scope scan wrote /tmp/bandit_mcp_unified_final_commit_scope.json with no command-level failures.
PR opened against dev: https://github.com/rmusser01/tldw_server/pull/2513
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated and remediated the MCP Unified review findings, completed the protocol.py tool-execution refactor into focused tool_execution helpers, preserved compatibility seams through wrappers/re-exports, and recorded the design/plan/review history. Final expanded verification passed: 384 focused MCP tests, py_compile, Ruff F/I/UP, Bandit touched-scope 0 findings, git diff --check, and final reviewer approval.
<!-- SECTION:FINAL_SUMMARY:END -->
