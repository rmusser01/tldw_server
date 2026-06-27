# MCP Unified Standalone UX Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Unified MCP embedded/standalone experience honest, launchable, understandable, diagnosable, and efficient for first-time users and experienced MCP operators.

**Architecture:** Treat current Unified MCP as an embedded TLDW Server capability, not a shipped standalone gateway. This plan first fixes product truth, quickstart, auth, and documentation contracts, then adds additive diagnostics and safer discovery behavior without forcing the full standalone extraction into this remediation slice.

**Tech Stack:** FastAPI, Pydantic, pytest, Loguru, Backlog.md, Markdown docs, existing MCP Unified runtime, existing wizard CLI.

---

## Scope And Constraints

- Backlog task: `TASK-2393`.
- This plan addresses the UX review findings without implementing the full standalone library/gateway extraction from `Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md`.
- Do not remove user or agent changes already present in the dirty worktree.
- Keep all runtime changes additive unless the step explicitly calls out a safety-motivated behavior change.
- Any code change must be test-first where practical.
- Use `source .venv/bin/activate` before Python, pytest, Bandit, or related commands.

## Files And Responsibilities

- `Docs/MCP/Unified/README.md`: primary MCP docs entry point and current-state banner.
- `Docs/MCP/Unified/User_Guide.md`: golden first-run flow, auth matrix, troubleshooting.
- `Docs/MCP/Unified/Client_Snippets.md`: copy-paste HTTP/WebSocket snippets aligned with default auth and strict catalog guidance.
- `Docs/MCP/Unified/Modules.md`: module surface, defaults, and risk tier explanation.
- `Docs/MCP/Unified/Using_Modules_YAML.md`: module configuration and starter profile guidance.
- `Docs/MCP/Unified/System_Admin_Guide.md`: operator auth, diagnostics, production hardening guidance.
- `Docs/MCP/Unified/Operator_Cheatsheet.md`: new compact power-user reference.
- `Docs/Operations/Env_Vars.md`: MCP env var coverage.
- `Docs/Development/Wizard.md`: wizard MCP install and verification behavior.
- `tldw_Server_API/app/core/MCP_unified/README.md`: developer/runtime README current-state banner and supported launch path.
- `tldw_Server_API/app/core/MCP_unified/docker/README.md`: new explicit status for the MCP-specific Dockerfile.
- `tldw_Server_API/app/core/MCP_unified/docker/Dockerfile`: either fixed to a smoke-tested supported path or quarantined as experimental.
- `tldw_Server_API/app/core/MCP_unified/module_surface.py`: new helper for module risk tiers and status summaries.
- `tldw_Server_API/app/core/MCP_unified/server.py`: status payload enrichment.
- `tldw_Server_API/app/core/MCP_unified/protocol.py`: catalog unresolved warning/fail-closed behavior.
- `tldw_Server_API/app/core/MCP_unified/config.py`: config warning collection for invalid env/YAML-derived settings.
- `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`: response models, safe config error handling, endpoint descriptions.
- `tldw_Server_API/cli/wizard/cli.py`: MCP client installer credential and verification flow.
- `tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py`: Docker/documentation contract.
- `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py`: catalog strict/unresolved behavior.
- `tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py`: config warning behavior.
- `tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py`: request config and diagnostic mapping.
- `tldw_Server_API/tests/wizard/test_cli_mcp.py`: client installer readiness behavior.
- `tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py`: new docs contract tests for current-state and quickstart promises.

---

### Task 1: Make The Current Product State Explicit

**Findings addressed:** ambiguous standalone mental model, incomplete discovery path.

**Files:**
- Modify: `Docs/MCP/Unified/README.md`
- Modify: `Docs/MCP/Unified/User_Guide.md`
- Modify: `tldw_Server_API/app/core/MCP_unified/README.md`
- Modify: `Docs/Product/MCP-Unified-Extraction.md`
- Create: `tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py`

- [x] **Step 1: Write docs contract tests for current-state wording**

Add tests that fail until primary docs clearly distinguish embedded MCP from planned standalone extraction.

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_unified_mcp_docs_state_embedded_today_standalone_planned():
    docs = "\n\n".join(
        [
            read("Docs/MCP/Unified/README.md"),
            read("Docs/MCP/Unified/User_Guide.md"),
            read("tldw_Server_API/app/core/MCP_unified/README.md"),
        ]
    ).lower()

    assert "embedded in tldw server today" in docs
    assert "standalone gateway" in docs
    assert "planned" in docs or "not yet shipped" in docs
```

- [x] **Step 2: Run the failing docs test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py::test_unified_mcp_docs_state_embedded_today_standalone_planned -v
```

Expected: FAIL because the current docs do not contain the required state language.

- [x] **Step 3: Add current-state banners**

Add this concise banner near the top of each primary doc:

```markdown
> **Current state:** Unified MCP is embedded in TLDW Server today. The standalone package/gateway is planned but not shipped in this tree yet. Use the TLDW Server launch path below unless a future release explicitly says the standalone gateway is available.
```

In `Docs/Product/MCP-Unified-Extraction.md`, add a note under the status header that the PRD describes planned extraction work and is not an install guide.

- [x] **Step 4: Re-run the docs test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py::test_unified_mcp_docs_state_embedded_today_standalone_planned -v
```

Expected: PASS.

- [x] **Step 5: Commit Task 1**

```bash
git add Docs/MCP/Unified/README.md Docs/MCP/Unified/User_Guide.md tldw_Server_API/app/core/MCP_unified/README.md Docs/Product/MCP-Unified-Extraction.md tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py
git commit -m "docs: clarify unified mcp embedded status"
```

---

### Task 2: Quarantine Or Verify The MCP-Specific Docker Path

**Findings addressed:** likely broken Docker launch path, launch trust.

**Decision:** Do not treat the MCP-specific Dockerfile as a supported standalone gateway in this remediation. Mark it experimental until the future extraction work can make it real. Keep the main TLDW Server launch path as the supported path.

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/docker/README.md`
- Modify: `tldw_Server_API/app/core/MCP_unified/README.md`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py`

- [x] **Step 1: Replace brittle Dockerfile text assertions with status assertions**

Update or add tests that assert the Docker directory has an explicit experimental status and that primary docs do not present it as the quickstart path.

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def test_mcp_specific_docker_path_is_marked_experimental():
    readme = ROOT / "app/core/MCP_unified/docker/README.md"
    text = readme.read_text(encoding="utf-8").lower()

    assert "experimental" in text
    assert "not the supported standalone gateway" in text
    assert "embedded in tldw server today" in text


def test_primary_mcp_readme_does_not_present_experimental_docker_as_quickstart():
    text = (ROOT / "app/core/MCP_unified/README.md").read_text(encoding="utf-8").lower()
    quickstart = text.split("## quick start", 1)[1].split("##", 1)[0]

    assert "docker build" not in quickstart
    assert "docker run" not in quickstart
```

Adjust `ROOT` to match the existing test file's location if needed.

- [x] **Step 2: Run the failing Docker contract tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py -v
```

Expected: FAIL until the README/status is added and old assumptions are updated.

- [x] **Step 3: Add Docker status README**

Create `tldw_Server_API/app/core/MCP_unified/docker/README.md`:

```markdown
# Unified MCP Docker Status

This directory is experimental. It is not the supported standalone gateway.

Unified MCP is embedded in TLDW Server today. Use the repository-level TLDW Server Docker or local server launch path for supported MCP usage.

The future standalone gateway work should replace this with a smoke-tested image that starts, imports the correct app target, and passes `/api/v1/mcp/health`.
```

- [x] **Step 4: Remove Docker from primary quickstart language**

Ensure the core MCP README points to the TLDW Server launch path and references the Docker subdirectory only as experimental.

- [x] **Step 5: Re-run Docker contract tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py -v
```

Expected: PASS.

- [x] **Step 6: Commit Task 2**

```bash
git add tldw_Server_API/app/core/MCP_unified/docker/README.md tldw_Server_API/app/core/MCP_unified/README.md tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py
git commit -m "docs: mark mcp docker path experimental"
```

---

### Task 3: Rewrite The Quickstart Around One Successful Workflow

**Findings addressed:** incomplete first-run path, fragmented auth, first-time usability.

**Files:**
- Modify: `Docs/MCP/Unified/User_Guide.md`
- Modify: `Docs/MCP/Unified/Client_Snippets.md`
- Modify: `Docs/MCP/Unified/System_Admin_Guide.md`
- Modify: `Docs/Operations/Env_Vars.md`
- Modify: `tldw_Server_API/app/core/MCP_unified/README.md`
- Modify: `tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py`

- [x] **Step 1: Add failing docs tests for quickstart completeness and auth matrix**

```python
def test_unified_mcp_quickstart_reaches_authenticated_tool_list():
    guide = read("Docs/MCP/Unified/User_Guide.md").lower()

    assert "golden path" in guide
    assert "authorization: bearer" in guide or "x-api-key" in guide
    assert "tools/list" in guide
    assert "tools/call" in guide
    assert "expected response" in guide


def test_unified_mcp_auth_matrix_exists():
    guide = read("Docs/MCP/Unified/User_Guide.md").lower()

    assert "auth method" in guide
    assert "recommended" in guide
    assert "disabled by default" in guide
    assert "query" in guide
```

- [x] **Step 2: Run the failing docs tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py -v
```

Expected: FAIL for new quickstart/auth expectations.

- [x] **Step 3: Add a "Golden Path" quickstart**

In `Docs/MCP/Unified/User_Guide.md`, add a first section that includes:

- Start TLDW Server.
- Confirm `/api/v1/mcp/health`.
- Choose one default auth method.
- Send JSON-RPC `initialize`.
- Send `tools/list`.
- Call one harmless read-only tool when available.
- Show expected response shape.
- Link to troubleshooting for auth, empty tool list, and degraded module status.

- [x] **Step 4: Add a canonical auth matrix**

Use this structure:

```markdown
| Auth method | Best for | Default status | Where it is sent | Notes |
| --- | --- | --- | --- | --- |
| AuthNZ JWT | Multi-user TLDW deployments | Recommended | `Authorization: Bearer ...` | Uses the main app identity and permissions. |
| Single-user API key | Local single-user deployments | Supported | `X-API-KEY` | Keep out of URLs and logs. |
| MCP JWT | Dedicated MCP integrations | Supported when configured | `Authorization: Bearer ...` or WebSocket subprotocol | Use when an MCP-only token lifecycle is desired. |
| Demo token | Local demos only | Disabled unless explicitly enabled | Header/subprotocol | Never use for shared deployments. |
| Query token/API key | Legacy/manual debugging only | Disabled by default | URL query string | Avoid because URLs are commonly logged. |
```

- [x] **Step 5: Align snippets and env var docs**

Update snippets so primary examples use header/subprotocol auth, not query auth. Add missing MCP env vars to `Docs/Operations/Env_Vars.md` with defaults and risk notes.

- [x] **Step 6: Re-run docs tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py -v
```

Expected: PASS.

- [x] **Step 7: Commit Task 3**

```bash
git add Docs/MCP/Unified/User_Guide.md Docs/MCP/Unified/Client_Snippets.md Docs/MCP/Unified/System_Admin_Guide.md Docs/Operations/Env_Vars.md tldw_Server_API/app/core/MCP_unified/README.md tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py
git commit -m "docs: add unified mcp first-run path"
```

---

### Task 4: Surface Effective MCP Capability Risk

**Findings addressed:** overwhelming defaults, unclear permissions/tools/servers mental model.

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/module_surface.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py`
- Modify: `Docs/MCP/Unified/Modules.md`
- Modify: `Docs/MCP/Unified/Using_Modules_YAML.md`

- [x] **Step 1: Write a focused helper test for risk tier summaries**

Create or extend a core MCP test:

```python
from tldw_Server_API.app.core.MCP_unified.module_surface import describe_module_surface


def test_describe_module_surface_groups_enabled_modules_by_risk():
    modules = {
        "media": {"enabled": True, "status": "healthy"},
        "filesystem": {"enabled": True, "status": "healthy"},
        "run_command": {"enabled": True, "status": "healthy"},
        "external_federation": {"enabled": False, "status": "disabled"},
    }

    surface = describe_module_surface(modules)

    assert "read_only" in surface["tiers"]
    assert "local_files" in surface["tiers"]
    assert "local_process" in surface["tiers"]
    assert "media" in surface["tiers"]["read_only"]["modules"]
    assert "filesystem" in surface["tiers"]["local_files"]["modules"]
    assert "run_command" in surface["tiers"]["local_process"]["modules"]
    assert surface["enabled_count"] == 3
```

- [x] **Step 2: Run the failing helper test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py::test_describe_module_surface_groups_enabled_modules_by_risk -v
```

Expected: FAIL because `module_surface.py` does not exist.

- [x] **Step 3: Implement `module_surface.py`**

Add a small, dependency-light helper:

```python
"""User-facing summaries of the effective Unified MCP module surface."""

from __future__ import annotations

from typing import Any


MODULE_RISK_TIERS: dict[str, tuple[str, str]] = {
    "media": ("read_only", "Search and retrieve existing media records."),
    "knowledge": ("read_only", "Search and retrieve knowledge records."),
    "chats": ("read_only", "Read chat/session context."),
    "prompts": ("read_only", "Read prompt library entries."),
    "prompts_catalog": ("read_only", "Expose configured prompt catalogs."),
    "mcp_discovery": ("read_only", "Inspect MCP capabilities."),
    "governance": ("write", "Manage or inspect policy/governance state."),
    "notes": ("write", "Create or modify note data."),
    "template": ("write", "Create or modify generated content templates."),
    "quizzes": ("write", "Create or modify quiz data."),
    "flashcards": ("write", "Create or modify flashcard data."),
    "kanban": ("write", "Create or modify board/task data."),
    "slides": ("write", "Create or export slide artifacts."),
    "filesystem": ("local_files", "Read or write configured local file scopes."),
    "codegraph": ("local_files", "Index and inspect configured source workspaces."),
    "external_federation": ("external_network", "Connect to external MCP servers."),
    "run_command": ("local_process", "Run configured local command families."),
    "sandbox": ("local_process", "Run code or workloads in configured sandboxes."),
    "persona_visuals": ("write", "Manage persona visual assets."),
    "characters": ("write", "Manage character-related data."),
}

TIER_LABELS: dict[str, str] = {
    "read_only": "Read-only data access",
    "write": "Writes to TLDW data",
    "local_files": "Local filesystem or workspace access",
    "external_network": "External server or network access",
    "local_process": "Local process or sandbox execution",
    "unknown": "Unclassified module",
}


def _is_enabled(payload: Any) -> bool:
    if isinstance(payload, dict):
        return bool(payload.get("enabled", payload.get("status") not in {"disabled", "not_loaded"}))
    return True


def describe_module_surface(modules: dict[str, Any]) -> dict[str, Any]:
    tiers = {
        key: {"label": label, "modules": []}
        for key, label in TIER_LABELS.items()
    }
    enabled_count = 0

    for module_name, payload in sorted(modules.items()):
        if not _is_enabled(payload):
            continue
        enabled_count += 1
        tier, description = MODULE_RISK_TIERS.get(module_name, ("unknown", "No risk tier is registered yet."))
        tiers[tier]["modules"].append({"id": module_name, "description": description})

    return {
        "enabled_count": enabled_count,
        "tiers": {key: value for key, value in tiers.items() if value["modules"]},
    }
```

- [x] **Step 4: Add `surface` to status responses**

In `server.py`, call `describe_module_surface()` inside the existing status builder. In `mcp_unified_endpoint.py`, add an optional `surface` field to `ServerStatusResponse`.

- [x] **Step 5: Document risk tiers and defaults**

Update module docs with the same tiers. Include a table of common default modules and whether each can read data, write data, access files, contact external servers, or run local processes.

- [x] **Step 6: Re-run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py tldw_Server_API/app/core/MCP_unified/tests/test_security_hardening.py -v
```

Expected: PASS.

- [x] **Step 7: Commit Task 4**

```bash
git add tldw_Server_API/app/core/MCP_unified/module_surface.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py Docs/MCP/Unified/Modules.md Docs/MCP/Unified/Using_Modules_YAML.md
git commit -m "feat: summarize unified mcp capability surface"
```

---

### Task 5: Make Catalog Misses Non-Silent

**Findings addressed:** catalog filters fail open, admin trust, tool discovery mental model.

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/mcp_discovery_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_mcp_discovery_module.py`
- Modify: `tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py`
- Modify: `Docs/MCP/Unified/Client_Snippets.md`
- Modify: `Docs/MCP/Unified/Developer_Guide.md`
- Modify: `Docs/MCP/Unified/User_Guide.md`

- [x] **Step 1: Write tests for unresolved catalog behavior**

Add tests that prove unresolved catalogs are visible to callers and do not silently broaden discovery.

```python
async def test_tools_list_unresolved_catalog_returns_warning_not_silent_full_list(protocol_with_tools):
    result = await protocol_with_tools.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "catalog-miss",
            "method": "tools/list",
            "params": {"catalog": "typo-catalog"},
        },
        user_context={"user_id": "test", "permissions": ["tools.execute:*"]},
    )

    payload = result["result"]
    assert payload.get("_meta", {}).get("catalog", {}).get("status") == "unresolved"
    assert payload["tools"] == []
```

Use the existing test fixture names in `test_protocol_catalog_filter.py`; do not invent a new large harness if a local fixture already covers tool listing.

- [x] **Step 2: Run the failing catalog tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py -v
```

Expected: FAIL until the protocol behavior changes.

Observed: the focused unresolved-catalog test failed because the old behavior returned the full tool list.

- [x] **Step 3: Change protocol behavior**

In the catalog resolution path:

- If `catalog` is supplied and unresolved, return no tools and attach `_meta.catalog.status = "unresolved"`.
- If a caller explicitly depends on legacy fail-open behavior, support it only through a clearly named config or parameter such as `catalog_fail_open=true`, not through the default.
- Keep RBAC `canExecute` filtering unchanged.

- [x] **Step 4: Update endpoint docs and snippets**

Make `catalog_strict` language clear, and update snippets to show strict/fail-closed discovery.

- [x] **Step 5: Re-run catalog tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py -v
```

Expected: PASS.

Observed: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py tldw_Server_API/app/core/MCP_unified/tests/test_mcp_discovery_module.py tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py -v` passed.

- [x] **Step 6: Commit Task 5**

```bash
git add tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py Docs/MCP/Unified/Client_Snippets.md Docs/MCP/Unified/User_Guide.md
git commit -m "fix: make mcp catalog misses explicit"
```

---

### Task 6: Add User-Facing Diagnostics And Config Warnings

**Findings addressed:** weak status, log-dependent diagnosis, silent config parse failures.

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/config.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py`
- Modify: `tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py`
- Modify: `Docs/MCP/Unified/User_Guide.md`

- [x] **Step 1: Write tests for invalid safe config**

```python
def test_http_request_invalid_safe_config_returns_400(client):
    response = client.post(
        "/api/v1/mcp/request",
        params={"config": "not-base64-json"},
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers={"X-API-KEY": "test-key"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_safe_config"
```

Adapt auth/client fixtures to the existing `test_http_mapping.py` patterns.

- [x] **Step 2: Write tests for status problem modules**

Add a focused test that registers or fakes one module with `ERROR` status and asserts status includes a sanitized `problem_modules` entry with `id`, `status`, `reason`, and `next_action`.

- [x] **Step 3: Run failing diagnostics tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py -v
```

Expected: FAIL for new diagnostics expectations.

Observed: the focused diagnostics tests failed because invalid safe config still returned 200, config warning collection was missing, and status had no `problem_modules`.

- [x] **Step 4: Reject invalid request safe config**

In `mcp_unified_endpoint.py`, replace log-and-continue parsing behavior with:

```python
raise HTTPException(
    status_code=400,
    detail={
        "code": "invalid_safe_config",
        "message": "The config query parameter must be base64url-encoded JSON.",
        "next_action": "Remove the config parameter or send a valid encoded JSON object.",
    },
)
```

Do not echo raw config values in responses or logs.

- [x] **Step 5: Add config warning collection**

Expose sanitized warnings for invalid optional config inputs such as bad tool category map JSON. Prefer a small helper, for example:

```python
def get_config_warnings() -> list[dict[str, str]]:
    return list(_CONFIG_WARNINGS)
```

Keep secret values out of warnings.

- [x] **Step 6: Add status diagnostics**

Add optional response fields:

- `problem_modules`: list of `{id, status, reason, next_action}`
- `config_warnings`: list of `{code, message, next_action}`

Use registry `error_message` where already sanitized; otherwise map to a generic message.

- [x] **Step 7: Update troubleshooting docs**

Add symptom-driven rows:

- Auth rejected.
- Query auth ignored.
- Catalog unresolved.
- Module degraded/unhealthy.
- Invalid safe config.
- Empty tool list.

- [x] **Step 8: Re-run diagnostics tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py -v
```

Expected: PASS.

Observed: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py -v` passed.

- [x] **Step 9: Commit Task 6**

```bash
git add tldw_Server_API/app/core/MCP_unified/config.py tldw_Server_API/app/core/MCP_unified/server.py tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py Docs/MCP/Unified/User_Guide.md
git commit -m "feat: surface unified mcp diagnostics"
```

---

### Task 7: Make The Client Installer Verify Readiness

**Findings addressed:** placeholder API key, almost-working client config, weak first-run recovery.

**Files:**
- Modify: `tldw_Server_API/cli/wizard/cli.py`
- Modify: `tldw_Server_API/tests/wizard/test_cli_mcp.py`
- Modify: `Docs/Development/Wizard.md`

- [x] **Step 1: Add tests for credential options**

```python
def test_mcp_add_accepts_api_key_option(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "cursor" / "mcp.json"
    result = run_cli(
        [
            "mcp",
            "add",
            "cursor",
            "--config-path",
            str(config_path),
            "--api-key",
            "test-key",
        ],
        monkeypatch,
    )

    assert result == 0
    assert "test-key" in config_path.read_text(encoding="utf-8")
    assert "verified" not in capsys.readouterr().out.lower()
```

Adapt to existing `test_cli_mcp.py` helpers.

- [x] **Step 2: Add tests for placeholder readiness messaging**

Assert that when no credential is provided, output says `configured but not ready` and names the target config file.

- [x] **Step 3: Add tests for `--verify`**

Use monkeypatching to fake the HTTP health/list-tools call. Assert:

- success prints `verified usable`;
- auth failure prints missing/invalid credential guidance;
- network failure prints server URL and next action.

- [x] **Step 4: Run failing wizard tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/wizard/test_cli_mcp.py -v
```

Expected: FAIL for new CLI options/messages.

Observed: wizard tests failed on missing `--api-key`, `--api-key-env`, `--verify`, and missing configured-but-not-ready messaging.

- [x] **Step 5: Implement CLI arguments**

Add:

- `--api-key`
- `--api-key-env`
- `--verify`

Rules:

- `--api-key` writes `X-API-KEY` directly.
- `--api-key-env NAME` writes an environment-variable reference if the target client format supports it; otherwise print a clear unsupported message.
- no credential keeps the placeholder but prints `configured but not ready`.
- `--verify` checks the MCP server URL and auth before claiming readiness.

- [x] **Step 6: Update wizard docs**

Document:

```bash
python -m tldw_Server_API.cli.wizard mcp add --client cursor --api-key "$SINGLE_USER_API_KEY" --verify
python -m tldw_Server_API.cli.wizard mcp add --client cursor --api-key-env SINGLE_USER_API_KEY --dry-run
```

- [x] **Step 7: Re-run wizard tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/wizard/test_cli_mcp.py -v
```

Expected: PASS.

Observed: `python -m pytest tldw_Server_API/tests/wizard/test_cli_mcp.py -v` passed.

- [x] **Step 8: Commit Task 7**

```bash
git add tldw_Server_API/cli/wizard/cli.py tldw_Server_API/tests/wizard/test_cli_mcp.py Docs/Development/Wizard.md
git commit -m "feat: verify mcp client installer readiness"
```

---

### Task 8: Add A Power-User Operator Cheatsheet

**Findings addressed:** advanced-user efficiency, scattered HTTP/WS/batch/session/status workflows.

**Files:**
- Create: `Docs/MCP/Unified/Operator_Cheatsheet.md`
- Modify: `Docs/MCP/Unified/README.md`
- Modify: `Docs/MCP/Unified/Client_Snippets.md`
- Modify: `tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py`

- [x] **Step 1: Add docs contract test for the cheatsheet**

```python
def test_unified_mcp_operator_cheatsheet_covers_power_user_workflows():
    text = read("Docs/MCP/Unified/Operator_Cheatsheet.md").lower()

    for phrase in [
        "tools/list",
        "tools/call",
        "batch",
        "session",
        "catalog",
        "metrics",
        "status",
        "websocket",
    ]:
        assert phrase in text
```

- [x] **Step 2: Run the failing docs test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py::test_unified_mcp_operator_cheatsheet_covers_power_user_workflows -v
```

Expected: FAIL because the cheatsheet does not exist yet.

- [x] **Step 3: Write `Operator_Cheatsheet.md`**

Keep it compact. Include:

- Base URL and auth variables.
- `initialize`.
- `tools/list` with strict catalog.
- `tools/call`.
- batch request.
- session header reuse.
- WebSocket auth subprotocol.
- status, health, metrics.
- client wizard dry-run and verify.
- common failure codes and next action.

- [x] **Step 4: Link the cheatsheet from the docs index and snippets**

Add a single link from the top-level Unified docs index and from Client Snippets.

- [x] **Step 5: Re-run docs tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py -v
```

Expected: PASS.

- [x] **Step 6: Commit Task 8**

```bash
git add Docs/MCP/Unified/Operator_Cheatsheet.md Docs/MCP/Unified/README.md Docs/MCP/Unified/Client_Snippets.md tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py
git commit -m "docs: add unified mcp operator cheatsheet"
```

---

### Task 9: Final Verification And Backlog Closeout

**Findings addressed:** all findings, quality gates.

**Files:**
- Modify: `backlog/tasks/task-2367 - Plan-and-implement-MCP-Unified-standalone-UX-remediation.md`

- [x] **Step 1: Run focused test suite**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py \
  tldw_Server_API/tests/wizard/test_cli_mcp.py \
  -v
```

Expected: PASS.

- [x] **Step 2: Run a broader MCP smoke subset**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_security_hardening.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_websocket_smoke.py \
  -v
```

Expected: PASS, or document unrelated pre-existing failures with evidence.

- [x] **Step 3: Run Bandit on touched code**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified \
  tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py \
  tldw_Server_API/cli/wizard \
  -f json -o /tmp/bandit_mcp_unified_ux.json
```

Expected: PASS with no new high/medium findings in touched code. Fix new findings before finishing.

- [x] **Step 4: Review docs as a user journey**

Manually walk:

1. Discover Unified MCP.
2. Start the supported embedded server path.
3. Authenticate.
4. List tools.
5. Understand enabled surface/risk tiers.
6. Diagnose a common failure.
7. Use the power-user cheatsheet.

Record gaps in `TASK-2393` if any remain.

- [x] **Step 5: Update Backlog task**

Add touched files, verification results, known skips, and final summary to `TASK-2393`.

- [x] **Step 6: Commit final task update**

```bash
git add "backlog/tasks/task-2393 - Plan-and-implement-MCP-Unified-standalone-UX-remediation.md"
git commit -m "chore: close mcp unified ux remediation task"
```

---

## PR Slice Recommendation

Use these as separate reviewable PRs if the branch gets large:

1. **PR 1:** Product truth, Docker quarantine, quickstart/auth docs.
2. **PR 2:** Capability surface and catalog miss behavior.
3. **PR 3:** Diagnostics/config warnings.
4. **PR 4:** Wizard readiness and operator cheatsheet.

## Non-Goals

- Publishing `mcp_unified` as a standalone package.
- Building the future standalone gateway.
- Redesigning the WebUI MCP Hub.
- Reworking AuthNZ, RBAC, or governance internals beyond user-facing clarity and additive diagnostics.
- Changing high-risk module defaults without a separate security/product decision.

## Review Checklist

- [ ] Primary docs are truthful about embedded vs standalone status.
- [ ] There is one complete first successful workflow.
- [ ] Auth docs distinguish recommended, supported, dev-only, and disabled-by-default methods.
- [ ] Effective module/tool surface is understandable by risk tier.
- [ ] Catalog misses are not silent.
- [ ] Diagnostics include sanitized reason and next action.
- [ ] Wizard output does not imply readiness when credentials are placeholders.
- [ ] Power-user command reference exists.
- [ ] Focused tests pass.
- [ ] Bandit touched-scope scan is clean or documented with no new findings.
