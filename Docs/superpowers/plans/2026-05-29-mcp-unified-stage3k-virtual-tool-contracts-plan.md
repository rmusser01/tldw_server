## Stage 1: Contract Boundary Tests
**Goal**: Prove host external federation uses the package-owned virtual tool contract.
**Success Criteria**: Failing tests show the host manager still defines a duplicate `VirtualExternalTool` and the package contract lacks caller-owned copy behavior.
**Tests**: Focused `test_runtime_package_boundary.py` contract tests.
**Status**: Complete

## Stage 2: Package Contract Reuse
**Goal**: Reuse `mcp_unified.federation.models.VirtualExternalTool` from the host manager and add copy isolation to the package contract.
**Success Criteria**: Host imports remain compatible while virtual tool metadata is package-owned and caller-owned when copied.
**Tests**: Focused package-boundary and external server manager tests.
**Status**: Complete

## Stage 3: Verification And PR
**Goal**: Record focused test, lint, security, and whitespace verification before opening the PR.
**Success Criteria**: Pytest, Ruff, Bandit, and `git diff --check` results are captured in Backlog and the PR.
**Tests**: Focused pytest suite plus Ruff and Bandit on touched files.
**Status**: Complete

**Verification**:
- `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py::test_runtime_auth_summary_treats_none_maps_as_empty -q` -> 1 passed, 3 warnings.
- `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py::test_list_virtual_tools_returns_caller_owned_copies -q` -> 1 passed, 3 warnings.
- `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_external_federation_integration.py::test_external_federation_module_integration_exposes_and_executes_virtual_tools -q` -> 1 skipped, 3 warnings.
- `.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py tldw_Server_API/app/core/MCP_unified/tests/test_federation_shell_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_external_federation_integration.py tldw_Server_API/app/core/MCP_unified/tests/test_external_credential_broker_runtime.py -q` -> 39 passed, 2 skipped, 3 warnings.
- `.venv/bin/python -m ruff check mcp_unified/federation/models.py tldw_Server_API/app/core/MCP_unified/external_servers/manager.py tldw_Server_API/app/core/MCP_unified/modules/implementations/external_federation_module.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_external_server_manager.py tldw_Server_API/app/core/MCP_unified/tests/test_external_federation_integration.py` -> All checks passed.
- `.venv/bin/python -m bandit -r mcp_unified/federation/models.py tldw_Server_API/app/core/MCP_unified/external_servers/manager.py tldw_Server_API/app/core/MCP_unified/modules/implementations/external_federation_module.py -f json -o /tmp/bandit_mcp_stage3k_virtual_tool_contracts.json` -> 0 findings.
- `git diff --check` -> clean.
