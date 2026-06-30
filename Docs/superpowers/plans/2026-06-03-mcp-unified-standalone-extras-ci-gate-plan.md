# MCP Unified Standalone Extras CI Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a focused release gate that proves the in-repo `mcp_unified` standalone distribution artifacts preserve the package-local metadata, extras, and dependency boundary.

**Architecture:** Keep the standalone proof in the existing MCP Unified package-boundary tests, because those tests already own the runtime/package separation contract. Extend the existing PyPI package workflow with a package-local gate so `mcp_unified/**` changes trigger CI without changing the root package publishing flow.

**Tech Stack:** Python 3.12 CI, `pytest`, `python -m build --no-isolation`, stdlib wheel/sdist metadata inspection, GitHub Actions.

---

### Task 1: Artifact Metadata Regression Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write the failing tests**

Add tests that build `mcp_unified` wheel and sdist into a temp directory, parse the wheel `METADATA` and `entry_points.txt`, and assert:
- `Name`, `Version`, and console script match `mcp_unified/package_metadata.py` and `mcp_unified/pyproject.toml`.
- `Provides-Extra` contains `core`, `fastapi`, `sqlite`, `federation`, `gateway`, and `dev`.
- `Requires-Dist` entries for each extra match the standalone metadata dependency names.
- Heavy root-only dependencies are absent from artifact metadata.
- The sdist includes `pyproject.toml`, package code, and no host `tldw_Server_API` tree.

- [x] **Step 2: Run the new focused tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_standalone_distribution_metadata_matches_extras \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_standalone_sdist_contains_only_package_boundary \
  -q
```

Expected: fail because the new artifact gate helpers/assertions are not implemented yet.

- [x] **Step 3: Implement the minimal artifact helpers**

Add small helpers that:
- Copy only `mcp_unified/` into a temp source directory.
- Run `python -m build --wheel --sdist --no-isolation --outdir <dist> <source>`.
- Parse wheel metadata with `zipfile` and `email.parser.Parser`.
- Parse sdist members with `tarfile`.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the same focused pytest command. Expected: pass.

### Task 2: CI Release Gate Wiring

**Files:**
- Modify: `.github/workflows/pypi-package.yml`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write the failing CI-contract test**

Add a test that reads `.github/workflows/pypi-package.yml` and asserts:
- `mcp_unified/**` changes trigger the workflow.
- The workflow runs the standalone artifact metadata tests.
- The root package build/upload artifact behavior remains present.

- [x] **Step 2: Run the CI-contract test and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_pypi_workflow_runs_mcp_unified_standalone_artifact_gate \
  -q
```

Expected: fail because the workflow does not yet include `mcp_unified/**` paths or the standalone test step.

- [x] **Step 3: Update the workflow**

Add `mcp_unified/**` to pull request and push path filters. Add a step after packaging tool installation that runs the focused standalone artifact gate test.

- [x] **Step 4: Run the CI-contract test and verify GREEN**

Run the same focused pytest command. Expected: pass.

### Task 3: Documentation And Backlog

**Files:**
- Modify: `Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md`
- Modify: `backlog/tasks/task-602 - Harden-MCP-Unified-standalone-extras-and-CI-artifact-gate.md`

- [x] **Step 1: Update docs**

Clarify that release CI now builds and inspects package-local artifacts as a pre-publish gate, while PyPI publishing is still not enabled.

- [x] **Step 2: Update Backlog task**

Record touched files, verification commands, known skips, and final summary.

### Task 4: Final Verification

**Files:**
- No additional file edits expected.

- [x] **Step 1: Run focused MCP Unified package-boundary tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  -q
```

- [x] **Step 2: Run Bandit on touched Python test scope**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -f json -o /tmp/bandit_mcp_standalone_extras_ci_gate.json
```

- [x] **Step 3: Inspect git diff**

```bash
git diff --stat
git diff -- .github/workflows/pypi-package.yml Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/pypi-package.yml Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md Docs/superpowers/plans/2026-06-03-mcp-unified-standalone-extras-ci-gate-plan.md tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py "backlog/tasks/task-602 - Harden-MCP-Unified-standalone-extras-and-CI-artifact-gate.md"
git commit -m "test: add mcp unified standalone artifact gate"
```
