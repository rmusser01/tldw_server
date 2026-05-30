# MCP Unified Stage 4I Gateway CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrow package-owned MCP Unified gateway CLI for validating standalone gateway profile bootstrap config files and listing bundled profile presets.

**Architecture:** The CLI lives in `mcp_unified.gateway.cli`, uses stdlib `argparse`, and delegates all config validation to the existing `load_gateway_profile_bootstrap_config()` helper. Output is deterministic JSON so front ends, setup wizards, and shell scripts can consume it without importing FastAPI or starting any transport.

**Tech Stack:** Python 3.11, stdlib `argparse`/`json`, existing `mcp_unified.gateway.config`, existing `mcp_unified.profiles.presets`, pytest.

---

### Task 1: CLI Contract Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

- [x] **Step 1: Write the failing config validation success test**

```python
def test_gateway_cli_validate_config_reports_success_json(tmp_path, capsys):
    config_path = tmp_path / "gateway.json"
    config_path.write_text(json.dumps({"store": {"kind": "memory"}, "default_preset_id": "project-researcher"}))

    exit_code = gateway_cli.main(["validate-config", str(config_path)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["default_preset_id"] == "project-researcher"
    assert payload["store"]["kind"] == "memory"
```

- [x] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py::test_gateway_cli_validate_config_reports_success_json -q`

Expected: FAIL because `mcp_unified.gateway.cli` does not exist.

- [x] **Step 3: Write the failing validation error test**

```python
def test_gateway_cli_validate_config_reports_error_json(tmp_path, capsys):
    config_path = tmp_path / "gateway.json"
    config_path.write_text("{", encoding="utf-8")

    exit_code = gateway_cli.main(["validate-config", str(config_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert "Invalid gateway config JSON" in payload["error"]
    assert "Traceback" not in captured.err
```

- [x] **Step 4: Run test to verify it fails for the missing CLI**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py::test_gateway_cli_validate_config_reports_error_json -q`

Expected: FAIL because `mcp_unified.gateway.cli` does not exist.

- [x] **Step 5: Write the failing preset listing and entry point tests**

```python
def test_gateway_cli_list_presets_reports_builtin_summary(capsys):
    exit_code = gateway_cli.main(["list-presets"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    preset_ids = {preset["id"] for preset in payload["presets"]}
    assert "project-researcher" in preset_ids
    assert all({"id", "name", "description", "version"} <= set(preset) for preset in payload["presets"])


def test_gateway_cli_project_script_is_registered():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'mcp-unified-gateway = "mcp_unified.gateway.cli:main"' in pyproject
```

- [x] **Step 6: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -q`

Expected: FAIL for the missing CLI module and missing project script.

### Task 2: Minimal CLI Implementation

**Files:**
- Create: `mcp_unified/gateway/cli.py`
- Modify: `mcp_unified/gateway/__init__.py`
- Modify: `pyproject.toml`

- [x] **Step 1: Implement `mcp_unified.gateway.cli`**

Implementation requirements:
- `main(argv: Sequence[str] | None = None) -> int`
- Subcommand `validate-config PATH [--format json|toml]`
- Subcommand `list-presets`
- Success output goes to stdout as sorted-key JSON plus newline.
- Validation errors go to stderr as sorted-key JSON plus newline and return `1`.
- Do not import FastAPI, start HTTP/WebSocket/stdio transports, or instantiate external MCP services.

- [x] **Step 2: Export the CLI module public entry**

Add `gateway_cli_main` or a lazy `main` export only if needed by tests or public package ergonomics. Avoid importing the CLI from `mcp_unified.gateway.__init__` if doing so would add avoidable import work.

- [x] **Step 3: Register the project script**

Add:

```toml
mcp-unified-gateway = "mcp_unified.gateway.cli:main"
```

- [x] **Step 4: Run the CLI tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -q`

Expected: PASS.

### Task 3: Verification and Closeout

**Files:**
- Modify: `backlog/tasks/task-566 - Implement-MCP-Unified-Stage-4I-gateway-CLI.md`

- [x] **Step 1: Run focused gateway package tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q`

Expected: PASS.

- [x] **Step 2: Run extraction and HTTP compatibility tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -q`

Expected: PASS.

- [x] **Step 3: Run lint and security checks**

Run: `source .venv/bin/activate && python -m ruff check mcp_unified/gateway tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

Expected: PASS.

Run: `source .venv/bin/activate && python -m bandit -r mcp_unified/gateway -f json -o /tmp/bandit_mcp_stage4i_gateway_cli.json`

Expected: PASS with zero findings in touched package scope.

- [x] **Step 4: Run whitespace and status checks**

Run: `git diff --check`

Expected: no output.

Run: `git status --short`

Expected: only Stage 4I files, plan file, and Backlog task are modified.

- [ ] **Step 5: Update Backlog task and commit**

Record verification in `TASK-566`, then commit with a message like:

```bash
git add mcp_unified/gateway/cli.py mcp_unified/gateway/__init__.py pyproject.toml tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py Docs/superpowers/plans/2026-05-30-mcp-unified-stage4i-gateway-cli-plan.md "backlog/tasks/task-566 - Implement-MCP-Unified-Stage-4I-gateway-CLI.md"
git commit -m "feat: add MCP Unified gateway CLI"
```
