# MCP Unified Stage 3D Security Test Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove direct host testing-helper imports from MCP Unified security/config code while preserving test-mode guard behavior.

**Architecture:** Add a package-local environment helper owned by MCP Unified and make `config.py`, `security/ip_filter.py`, and `security/request_guards.py` depend on it instead of `tldw_Server_API.app.core.testing`. Keep the helper dependency-free so it can move with the standalone package later, while matching the existing test-mode/env-flag semantics used by current guard paths.

**Tech Stack:** Python, FastAPI/Starlette requests, pytest, Ruff, Bandit.

**Backlog:** TASK-482

---

### Task 1: Boundary And Behavior Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py`

- [x] **Step 1: Write failing import-boundary test**

Add a contract test that scans `config.py`, `security/ip_filter.py`, and `security/request_guards.py` and fails if any import source resolves to `tldw_Server_API.app.core.testing`.

- [x] **Step 2: Write failing guard behavior tests**

Add tests that force MCP test mode via environment variables and verify:
- IP allowlist normalizes missing synthetic client IP to loopback.
- Client-certificate guard accepts `testclient` only in test mode while still requiring the configured certificate header/value.

- [x] **Step 3: Verify RED**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_security_and_config_use_package_local_environment_helpers tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py::test_ip_allowlist_normalizes_missing_client_ip_in_test_mode tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py::test_client_certificate_guard_allows_testclient_only_in_test_mode -q
```

Expected: import-boundary test fails before implementation; behavior tests may already pass through the old host helper.

### Task 2: Package-Local Environment Helper

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/environment.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/config.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/security/ip_filter.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/security/request_guards.py`

- [x] **Step 1: Add helper**

Implement `env_flag_enabled(name: str) -> bool`, `is_truthy(value: object) -> bool`, `is_explicit_pytest_runtime() -> bool`, and `is_test_mode() -> bool` using only `os.environ`/`PYTEST_CURRENT_TEST` and safe truthy-string handling.

- [x] **Step 2: Replace imports**

Update config/security modules to import from `..environment` or `.environment` as appropriate.

- [x] **Step 3: Verify GREEN**

Run the Task 1 pytest command again. Expected: all selected tests pass.

### Task 3: Focused Validation And Closeout

**Files:**
- Modify: `backlog/tasks/task-482 - Implement-MCP-Unified-Stage-3D-security-test-mode-seam-cleanup.md`

- [x] **Step 1: Run focused regression suite**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py -q
```

- [x] **Step 2: Run lint/security checks**

Run:
```bash
source .venv/bin/activate
python -m ruff check tldw_Server_API/app/core/MCP_unified/environment.py tldw_Server_API/app/core/MCP_unified/config.py tldw_Server_API/app/core/MCP_unified/security/ip_filter.py tldw_Server_API/app/core/MCP_unified/security/request_guards.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py
python -m bandit -r tldw_Server_API/app/core/MCP_unified/environment.py tldw_Server_API/app/core/MCP_unified/config.py tldw_Server_API/app/core/MCP_unified/security/ip_filter.py tldw_Server_API/app/core/MCP_unified/security/request_guards.py -f json -o /tmp/bandit_mcp_stage3d_security_test_seams.json
```

- [x] **Step 3: Update Backlog and commit**

Record touched files, verification output, known skips, and final summary on TASK-482. Commit the focused slice and open the PR.
