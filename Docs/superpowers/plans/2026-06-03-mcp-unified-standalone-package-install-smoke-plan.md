# MCP Unified Standalone Package Install Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated standalone package descriptor and an offline clean-environment install smoke test for `mcp_unified`.

**Architecture:** Keep the standalone package metadata beside the package source at `mcp_unified/pyproject.toml`. Validate it from the existing package-boundary test suite, then build/install it without dependency resolution into an isolated target directory and import from a `python -S` process outside the repo to prove `import mcp_unified` does not rely on the root `tldw-server` package surface.

**Tech Stack:** Python packaging with setuptools, pytest, pip `--target`, existing `mcp_unified.package_metadata` release metadata.

---

### Task 1: Package Descriptor Contract

**Files:**
- Create: `mcp_unified/pyproject.toml`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
- Modify: `Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md`

- [x] **Step 1: Write failing metadata tests**

Add tests that require `mcp_unified/pyproject.toml` to exist, match `mcp_unified.package_metadata`, expose `mcp-unified-gateway`, and keep heavyweight root dependencies out of standalone core dependencies.

- [x] **Step 2: Run tests to verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q`

Expected: FAIL because `mcp_unified/pyproject.toml` is not present.

- [x] **Step 3: Add minimal standalone descriptor**

Create `mcp_unified/pyproject.toml` using setuptools, package name `mcp-unified`, package-dir `mcp_unified = "."`, the existing CLI entry point, core dependency floors, and extras for `fastapi`, `sqlite`, `federation`, `gateway`, and `dev`.

- [x] **Step 4: Run tests to verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q`

Expected: PASS.

### Task 2: Offline Clean Install Smoke

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
- Modify: `Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md`

- [x] **Step 1: Write failing install smoke test**

Add a pytest smoke test that builds the standalone package wheel with `--no-deps --no-build-isolation`, installs that wheel into an isolated target directory with `--no-deps --no-index`, and imports only `mcp_unified` plus `mcp_unified.package_metadata` from a `python -S` subprocess.

- [x] **Step 2: Run smoke test to verify RED/GREEN state**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_standalone_package_installs_without_root_dependencies -q`

Expected before descriptor: FAIL due missing package descriptor. Expected after descriptor: PASS.

- [x] **Step 3: Document the release smoke command**

Document the standalone package descriptor and focused install-smoke command in `Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md`.

- [x] **Step 4: Run focused verification**

Run:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -q`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified -f json -o /tmp/bandit_mcp_package_build_smoke.json`

Expected: tests pass; Bandit reports no findings in package runtime source.

### Task 3: Backlog And Commit

**Files:**
- Modify: `backlog/tasks/task-601 - Add-MCP-Unified-standalone-package-install-smoke.md`

- [x] **Step 1: Update Backlog task**

Record implementation notes, touched files, verification results, and final summary in `TASK-601`.

- [x] **Step 2: Self-review diff**

Run: `git diff --check` and inspect `git diff --stat`.

- [x] **Step 3: Commit**

Stage the plan, descriptor, tests, docs, and Backlog task. Commit with message: `test: add mcp unified standalone install smoke`.
