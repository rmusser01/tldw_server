# MCP Unified Stage 4H Gateway Config File Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for behavior changes and keep this slice package-owned.

**Goal:** Add package-owned helpers that load standalone gateway profile bootstrap config from explicit JSON or TOML files.

**Architecture:** Keep the loader in `mcp_unified.gateway.config` so it reuses the Stage 4G dataclass validation and bootstrap seam. Use stdlib `json` and `tomllib` only, infer format from file suffix, and fail fast with clear `ValueError`s for unsupported formats, parse failures, and non-object top-level payloads. This slice deliberately stops before CLI commands, process entrypoints, external MCP lifecycle, and host route integration.

**Tech Stack:** Python 3.11, dataclasses, stdlib JSON/TOML parsers, package-local gateway/profile/storage primitives, pytest, Ruff, Bandit.

---

### Task 1: Config File Loader RED Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- Modify: `mcp_unified/gateway/config.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add failing loader tests**

Add tests that assert:
- a JSON config file loads into `GatewayProfileBootstrapConfig` and bootstraps a default preset;
- a TOML config file loads store/default values into the config model;
- unsupported suffixes, malformed JSON/TOML, and non-object top-level payloads raise clear `ValueError`s;
- importing `mcp_unified.gateway` exposes the loader without importing `tldw_Server_API`.

- [x] **Step 2: Run RED tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4h-gateway-config-loader/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: new tests fail because the loader/export does not exist.

Result: `7 failed, 51 passed, 4 warnings`; all new failures were expected missing loader/export imports.

### Task 2: Package Config Loader

**Files:**
- Modify: `mcp_unified/gateway/config.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [x] **Step 1: Add loader helpers**

Implement:
- `GatewayConfigFormat = Literal["json", "toml"]`;
- `load_gateway_profile_bootstrap_config(path: str | Path, *, format: GatewayConfigFormat | None = None) -> GatewayProfileBootstrapConfig`;
- `_detect_config_format(path, format)` and parser helpers for clear errors.

- [x] **Step 2: Preserve package isolation**

Keep imports stdlib plus package-local only. Do not import FastAPI, `tldw_Server_API`, PyYAML, click/typer, or host config helpers.

- [x] **Step 3: Export public loader APIs**

Export the loader and format alias from `mcp_unified.gateway` and `mcp_unified.gateway.config`.

- [x] **Step 4: Run GREEN tests**

Run the focused gateway package tests and confirm all pass.

Result: `58 passed, 4 warnings`.

### Task 3: Compatibility, Security, And PR Handoff

**Files:**
- Modify: `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4h-gateway-config-loader-plan.md`
- Modify: `backlog/tasks/task-565 - Implement-MCP-Unified-Stage-4H-gateway-config-file-loader.md`

- [x] **Step 1: Run host compatibility tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4h-gateway-config-loader/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -q
```

- [x] **Step 2: Run lint, security, and whitespace checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4h-gateway-config-loader/.venv/bin/python -m ruff check mcp_unified/gateway tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mcp-unified-stage4h-gateway-config-loader/.venv/bin/python -m bandit -r mcp_unified/gateway -f json -o /tmp/bandit_mcp_stage4h_gateway_config_loader.json
git diff --check
```

- [x] **Step 3: Update plan and Backlog task**

Record RED/GREEN and final verification evidence. Check off acceptance criteria and Definition of Done.

- [x] **Step 4: Commit, push, and open PR**

Commit the plan, gateway/test changes, and Backlog task update together, push the branch, and open a PR against `dev`.

Result: PR #2161 opened at `https://github.com/rmusser01/tldw_server/pull/2161`.

### Task 4: Review Follow-Up

**Files:**
- Modify: `mcp_unified/gateway/config.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [x] **Step 1: Verify and reproduce schema/type error feedback**

Gemini review noted that invalid config schema/type cases escaped as raw `TypeError`s. Added regression coverage for unknown top-level keys and invalid `store` value types.

Review RED result: `2 failed, 58 passed, 4 warnings`.

- [x] **Step 2: Wrap schema/type errors as ValueError**

Wrap `GatewayProfileBootstrapConfig(**payload)` `TypeError`s with `ValueError("Invalid gateway config schema or types: ...")`.

Review GREEN result: `60 passed, 4 warnings`.

- [x] **Step 3: Address Qodo docstring and JSON-location feedback**

Qodo review noted that the public loader docstring omitted key behavior and that JSON parse errors dropped line/column context. Added a JSON diagnostic regression and expanded the public loader docstring.

Qodo RED result: `1 failed, 60 passed, 4 warnings`.
Qodo GREEN result: `61 passed, 4 warnings`.

## Verification

- Baseline: `51 passed, 4 warnings` for `test_gateway_fastapi_package.py` on merged Stage 4G.
- RED: `7 failed, 51 passed, 4 warnings`; expected missing loader/export failures only.
- GREEN: `58 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Review RED: `2 failed, 58 passed, 4 warnings` for raw schema/type `TypeError` leakage.
- Review GREEN: `60 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Qodo RED: `1 failed, 60 passed, 4 warnings` for missing JSON parse location details.
- Qodo GREEN: `61 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Compatibility: `47 passed, 4 warnings` for `test_extraction_contracts.py` and `test_http_mapping.py`.
- Ruff: `All checks passed!` for `mcp_unified/gateway` and the focused gateway package tests.
- Bandit: `0` results and no errors for `mcp_unified/gateway`.
- Whitespace: `git diff --check` passed.
