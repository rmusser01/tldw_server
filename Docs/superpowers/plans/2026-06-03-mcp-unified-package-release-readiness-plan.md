# MCP Unified Package Release Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested release-readiness metadata surface for `mcp_unified` so the package boundary advertises its current internal/experimental status, verified license, dependency extras, and import smoke expectations without implying unsupported standalone publication.

**Architecture:** Keep the slice intentionally small: add a pure-stdlib metadata module under `mcp_unified`, expose it through a read-only CLI command, and document the current packaging gate. Do not split a separate PyPI distribution in this task; the metadata becomes the testable contract that later packaging work can consume.

**Tech Stack:** Python 3.10+, argparse CLI, pytest, Bandit.

---

## File Structure

- Create `mcp_unified/package_metadata.py` for import-light package status, license, dependency extras, and JSON-safe summary helpers.
- Modify `mcp_unified/gateway/cli.py` to add a `package-info` command that prints the metadata summary without requiring gateway config or remote runtime state.
- Modify `Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md` to describe the current internal/experimental release status, GPL-3.0-only license decision, and dependency-extra intent.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py` for metadata shape and minimal import-boundary smoke coverage.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py` for CLI visibility.
- Update `backlog/tasks/task-599 - Harden-MCP-Unified-package-release-readiness-metadata-and-install-smoke.md` with progress and verification results.

### Task 1: Metadata Contract Tests

**Files:**
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write failing metadata tests**

Add tests that import `mcp_unified.package_metadata` and assert:
- `PACKAGE_STATUS` is internal/experimental.
- `PUBLISHING_STATUS` is not published.
- `LICENSE_EXPRESSION` is `GPL-3.0-only`.
- extras include exactly the required public groups: `core`, `fastapi`, `sqlite`, `federation`, `gateway`, `dev`.
- no extra dependency string mentions heavy host stacks such as ChromaDB, faster-whisper, torch, yt-dlp, or Next.js.

- [x] **Step 2: Verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_package_metadata_declares_release_gate -v
```

Expected: fail because `mcp_unified.package_metadata` does not exist yet.

- [x] **Step 3: Implement minimal metadata module**

Create `mcp_unified/package_metadata.py` with immutable constants and a `package_metadata_summary()` helper that returns JSON-safe dictionaries/lists.

- [x] **Step 4: Verify GREEN**

Run the focused metadata test and confirm it passes.

### Task 2: Import Smoke Test

**Files:**
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [x] **Step 1: Write failing import smoke test**

Add a subprocess test that imports only `mcp_unified` and `mcp_unified.package_metadata`, then asserts these modules are not imported:
- `tldw_Server_API`
- `chromadb`
- `torch`
- `faster_whisper`
- `yt_dlp`
- `next`

- [x] **Step 2: Verify RED**

Run the new subprocess test before production changes if Task 1 has not yet added the module. If Task 1 is complete, this may already pass; in that case record that the red condition was satisfied by Task 1's missing-module failure.

- [x] **Step 3: Keep implementation import-light**

Ensure `package_metadata.py` imports only standard-library modules and does not import gateway, storage, federation, or host code.

- [x] **Step 4: Verify GREEN**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_core_import_smoke_stays_minimal -v
```

Expected: pass.

### Task 3: CLI Visibility

**Files:**
- Modify: `mcp_unified/gateway/cli.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

- [x] **Step 1: Write failing CLI test**

Add a test that runs `gateway_cli.main(["package-info"])`, reads stdout JSON, and asserts:
- `ok` is true.
- package name and license match metadata.
- status is internal/experimental.
- extras include `gateway`.
- no stderr is emitted.

- [x] **Step 2: Verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py::test_gateway_cli_package_info_reports_release_gate -v
```

Expected: fail because the parser does not know `package-info`.

- [x] **Step 3: Implement CLI command**

Import `package_metadata_summary` from `mcp_unified.package_metadata`, add a `package-info` subparser, and implement `_handle_package_info()` using existing `_emit_json()`.

- [x] **Step 4: Verify GREEN**

Run the focused CLI test and confirm it passes.

### Task 4: Documentation

**Files:**
- Modify: `Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

- [x] **Step 1: Add release-readiness section**

Document that `mcp_unified` is currently an in-repo/internal experimental package boundary, not a separately published standalone package. Note the current canonical license metadata is GPL-3.0-only and that downstream users should not treat the package as independently release-ready until minimal install/extras CI is added.

- [x] **Step 2: Add CLI discoverability**

Show `mcp-unified-gateway package-info` as the way to inspect the current metadata contract.

- [x] **Step 3: Add docs visibility test**

Add a lightweight test that reads `Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md` and asserts it mentions `package-info`, `GPL-3.0-only`, and the internal/experimental release gate.

- [x] **Step 4: Keep wording conservative**

Avoid claims that the package is already available on PyPI or ready for third-party embedding.

### Task 5: Validation And Task Finalization

**Files:**
- Modify: `backlog/tasks/task-599 - Harden-MCP-Unified-package-release-readiness-metadata-and-install-smoke.md`

- [x] **Step 1: Run focused tests**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  -v
```

- [x] **Step 2: Run Bandit on touched Python**

```bash
source .venv/bin/activate
python -m bandit -r mcp_unified/package_metadata.py mcp_unified/gateway/cli.py -f json -o /tmp/bandit_mcp_package_release_readiness.json
```

- [x] **Step 3: Run diff whitespace check**

```bash
git diff --check
```

- [x] **Step 4: Update Backlog**

Record changed files, verification results, and any known skips in TASK-599. Mark acceptance criteria and Definition of Done complete only after verification passes.

- [x] **Step 5: Commit**

```bash
git add Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md Docs/superpowers/plans/2026-06-03-mcp-unified-package-release-readiness-plan.md mcp_unified/package_metadata.py mcp_unified/gateway/cli.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py "backlog/tasks/task-599 - Harden-MCP-Unified-package-release-readiness-metadata-and-install-smoke.md"
git commit -m "chore: document mcp unified package release gate"
```
