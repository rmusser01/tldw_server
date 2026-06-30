# MCP Unified Residual UX Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the residual Unified MCP UX, trust, defaults, and recovery gaps identified after TASK-2393 without expanding the standalone gateway scope.
**Architecture:** Apply contract-backed changes across documentation, module defaults, status metadata, HTTP/JSON-RPC recovery metadata, WebSocket auth copy, and package-gateway readiness status. The supported product remains embedded TLDW MCP at `/api/v1/mcp`; `apps/mcp-unified` remains an internal/experimental package boundary.
**Tech Stack:** Python 3.10+, FastAPI, Pydantic, pytest, PyYAML, Backlog.md, Bandit.

---

## Context

Design spec:
`Docs/superpowers/specs/2026-06-28-mcp-unified-residual-ux-hardening-design.md`

Tracking task:
`TASK-12054`

Related completed remediation:
`TASK-2393`

This plan intentionally does not add `mcp-unified-gateway serve`, package publishing, or a supported standalone gateway promise. The work is residual hardening around the already-supported embedded MCP surface and the internal package gateway boundary.

## File Map

Runtime and API files:

- `tldw_Server_API/app/core/MCP_unified/server.py`
  - Loads configured modules from `MCP_MODULES_CONFIG`, `MCP_MODULES`, and optional env flags.
  - Currently auto-enables `filesystem` by default when absent from YAML.
  - Builds `/api/v1/mcp/status` using module health only.
- `tldw_Server_API/app/core/MCP_unified/module_surface.py`
  - Groups enabled modules by user-facing risk tier.
  - Does not yet expose disabled-but-available high-risk modules.
- `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
  - HTTP helper endpoints, WebSocket endpoint docs, and HTTP error mapping.
  - Currently mixes string `detail` bodies and object `detail` bodies.
- `tldw_Server_API/app/core/MCP_unified/protocol.py`
  - JSON-RPC error shape and `error.data` hint enrichment.
- `apps/mcp-unified/src/mcp_unified/gateway/fastapi.py`
  - Package-local FastAPI gateway router and `/status`.
- `apps/mcp-unified/src/mcp_unified/package_metadata.py`
  - Static package status and publishing metadata.

Configuration and docs:

- `tldw_Server_API/Config_Files/mcp_modules.yaml`
  - Default module config. `filesystem` and `run_command` are currently enabled.
- `Docs/MCP/Unified/README.md`
- `Docs/MCP/Unified/User_Guide.md`
- `Docs/MCP/Unified/System_Admin_Guide.md`
- `Docs/MCP/Unified/Client_Snippets.md`
- `Docs/MCP/Unified/Smoke_Client.md`
- `Docs/MCP/Unified/Modules.md`
- `Docs/MCP/Unified/Using_Modules_YAML.md`
- `tldw_Server_API/app/core/MCP_unified/docker/Dockerfile`
- `tldw_Server_API/app/core/MCP_unified/docker/README.md`
- `apps/mcp-unified/README.md`
- `apps/mcp-unified/USER_GUIDE.md`
- `apps/mcp-unified/src/mcp_unified/README.md`

Tests to extend:

- `tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py`

## Guardrails

- Before making implementation edits, confirm there is an implementation Backlog task distinct from this planning task. This planning task is `TASK-12054`; execution should create or reuse a separate implementation task unless the user explicitly scopes the next step to planning only.
- Preserve JSON-RPC outer error shape: `{"jsonrpc":"2.0","error":{...},"id":...}`.
- Add JSON-RPC recovery fields only under `error.data`; do not rename `message`, `code`, or `id`.
- Preserve HTTP helper endpoint body compatibility:
  - If an endpoint currently returns string `detail`, keep string `detail` and add additive recovery headers.
  - If an endpoint already returns object `detail`, extend that object with `reason_code` and `next_action`.
- Keep query-string WebSocket auth documented as legacy and disabled by default. Do not present `?token=` or `?api_key=` as the normal path.
- Disable high-risk local file/process defaults, but keep explicit operator opt-ins honored:
  - `enabled: true` in the selected modules YAML still loads that module.
  - Explicit env flags such as `MCP_ENABLE_FILESYSTEM_MODULE=true` still load fallback modules when no YAML entry is present.
- Do not introduce new dependencies.
- Use `source .venv/bin/activate` before Python, pytest, or Bandit commands.
- Commit after each coherent working stage. Do not commit unrelated dirty worktree files.

---

## Stage 0: Backlog Execution Setup

**Goal:** Satisfy repository task-tracking requirements before any implementation file edits.

**Success Criteria:**

- The implementation worker has read the Backlog workflow instructions through MCP or CLI fallback.
- A Backlog task exists for implementation work, separate from planning task `TASK-12054`.
- The implementation task links this plan and the design spec.
- The task is marked `In Progress` before code/docs/config edits begin.

**Steps:**

- [ ] Read the Backlog workflow overview using the installed MCP server, for example:

```text
backlog://workflow/overview
```

If MCP resources are unavailable, use the official fallback:

```bash
backlog task list --plain
```

- [ ] Search for an existing implementation task to avoid duplicates:

```bash
backlog search "MCP Unified residual UX hardening" --plain
```

- [ ] If no implementation task exists, create one with labels `mcp`, `ux`, `security`, `docs`, and references to:
  - `TASK-12054`
  - `TASK-2372`
  - `Docs/superpowers/plans/2026-06-28-mcp-unified-residual-ux-hardening-implementation-plan.md`
  - `Docs/superpowers/specs/2026-06-28-mcp-unified-residual-ux-hardening-design.md`
- [ ] Mark the implementation task `In Progress`.
- [ ] Add an implementation note that Stage 0 is complete and record the task id in the first implementation commit message.

**Verification:**

```bash
backlog task <IMPLEMENTATION_TASK_ID> --plain
```

**Commit:**

Do not commit Stage 0 alone unless it creates/edits a Backlog task file. If it does, include that task file in the first implementation commit.

---

## Stage 1: Documentation Contract Tests And Copy Cleanup

**Goal:** Make the docs consistently explain the supported embedded surface, internal package boundary, legacy query auth, and exact first-run paths.

**Success Criteria:**

- Docs consistently direct users to `/api/v1/mcp/status` and `/api/v1/mcp/request` for TLDW Server.
- Package-local `/mcp/status` and `/mcp/request` are labeled as package/host-mounted examples only.
- Primary docs include a "Which path should I use?" table that separates embedded TLDW paths, package-local mounted paths, and future standalone gateway status.
- No quickstart instructs `python -m pip install -e "mcp_unified[gateway]"`.
- No normal WebSocket example uses `?token=jwt-token`.
- Package README/User Guide do not promise `serve`, publishing, or supported standalone operation.
- Admin guide no longer documents stale `MCP_*` variables as active config.

**Tests First:**

- [ ] Extend `tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py`.

Add tests equivalent to:

```python
def test_unified_mcp_docs_use_supported_embedded_paths_for_smoke_and_snippets() -> None:
    docs = "\n\n".join([
        _read("Docs/MCP/Unified/Client_Snippets.md"),
        _read("Docs/MCP/Unified/Smoke_Client.md"),
        _read("Docs/MCP/Unified/User_Guide.md"),
    ])
    _require("/api/v1/mcp/status" in docs, "Docs should show embedded status path.")
    _require("/api/v1/mcp/request" in docs, "Docs should show embedded request path.")
    _require('mcp_unified[gateway]' not in docs, "Docs should not install a non-root package path.")
```

```python
def test_unified_mcp_docs_label_every_package_local_request_example() -> None:
    smoke = _read("Docs/MCP/Unified/Smoke_Client.md")
    for line_no, line in enumerate(smoke.splitlines(), start=1):
        if "http://127.0.0.1:8000/mcp/request" not in line:
            continue
        window = "\n".join(smoke.splitlines()[max(0, line_no - 5): line_no + 4]).lower()
        _require(
            "package-local" in window or "host-mounted" in window,
            f"Unlabeled package-local /mcp/request example near Smoke_Client.md:{line_no}",
        )
```

```python
def test_unified_mcp_docs_have_path_decision_table() -> None:
    docs = "\n\n".join([
        _read("Docs/MCP/Unified/README.md"),
        _read("Docs/MCP/Unified/User_Guide.md"),
    ]).lower()
    for snippet in (
        "which path should i use",
        "/api/v1/mcp/status",
        "/api/v1/mcp/request",
        "/mcp/status",
        "package-local",
        "standalone gateway",
        "planned but not shipped",
    ):
        _require(snippet in docs, f"Path decision docs should mention {snippet}.")
```

```python
def test_unified_mcp_docs_do_not_normalize_query_token_auth() -> None:
    docs = "\n\n".join([
        _read("Docs/MCP/Unified/User_Guide.md"),
        _read("Docs/MCP/Unified/Client_Snippets.md"),
        _read("Docs/MCP/Unified/Smoke_Client.md"),
    ]).lower()
    _require("?token=jwt-token" not in docs, "Docs should not show query token auth as a normal example.")
    _require("disabled by default" in docs and "query auth" in docs, "Docs should frame query auth as legacy/disabled.")
```

```python
def test_unified_mcp_package_docs_do_not_promise_serve_or_publishing() -> None:
    docs = "\n\n".join([
        _read("apps/mcp-unified/README.md"),
        _read("apps/mcp-unified/USER_GUIDE.md"),
        _read("apps/mcp-unified/src/mcp_unified/README.md"),
    ]).lower()
    forbidden = [
        "mcp-unified-gateway serve",
        "pip install mcp-unified",
        "published to pypi",
        "production standalone gateway",
    ]
    found = [phrase for phrase in forbidden if phrase in docs]
    _require(not found, f"Package docs imply unsupported standalone/published flows: {found}")
    _require("not published" in docs, "Package docs should state the package is not published.")
    _require("internal" in docs and "experimental" in docs, "Package docs should state internal/experimental status.")
```

```python
def test_unified_mcp_docs_reference_existing_local_targets() -> None:
    import re

    docs_to_check = [
        "Docs/MCP/Unified/README.md",
        "Docs/MCP/Unified/User_Guide.md",
        "Docs/MCP/Unified/Smoke_Client.md",
        "Docs/MCP/Unified/Client_Snippets.md",
        "apps/mcp-unified/README.md",
        "apps/mcp-unified/USER_GUIDE.md",
    ]
    missing: list[str] = []
    for doc_path in docs_to_check:
        text = _read(doc_path)
        for target in re.findall(r"\]\(([^)#][^)]+)\)", text):
            if "://" in target or target.startswith("#") or target.startswith("mailto:"):
                continue
            normalized = target.split("#", 1)[0]
            if not normalized:
                continue
            candidate = Path(doc_path).parent / normalized
            if not candidate.exists():
                missing.append(f"{doc_path} -> {target}")
    _require(not missing, "Docs reference missing local targets: " + ", ".join(missing))
```

```python
def test_unified_mcp_admin_env_docs_do_not_include_known_stale_mcp_vars() -> None:
    guide = _read("Docs/MCP/Unified/System_Admin_Guide.md")
    stale = {
        "MCP_HOST",
        "MCP_PORT",
        "MCP_AUTH_MODE",
        "MCP_MODULES_ENABLED",
        "MCP_DATABASE_MAX_OVERFLOW",
        "MCP_TRUSTED_PROXIES",
        "MCP_MAX_REQUEST_SIZE",
        "MCP_REQUEST_TIMEOUT",
    }
    found = {name for name in stale if name in guide}
    _require(not found, f"System admin guide documents stale MCP env vars: {sorted(found)}")
```

Optional stronger contract after the known stale vars are removed:

```python
def test_unified_mcp_admin_env_docs_match_config_aliases_for_core_mcp_vars() -> None:
    import re
    from tldw_Server_API.app.core.MCP_unified.config import MCPConfig

    aliases = {
        str(field.validation_alias)
        for field in MCPConfig.model_fields.values()
        if str(field.validation_alias).startswith("MCP_")
    }
    documented = set(re.findall(r"\bMCP_[A-Z0-9_]+\b", _read("Docs/MCP/Unified/System_Admin_Guide.md")))
    allowed_module_or_runtime_vars = {
        "MCP_MODULES",
        "MCP_MODULES_CONFIG",
        "MCP_EXTERNAL_SERVERS_CONFIG",
        "MCP_ENABLE_FILESYSTEM_MODULE",
        "MCP_ENABLE_GIT_MODULE",
        "MCP_ENABLE_SANDBOX_MODULE",
        "MCP_ENABLE_BROWSER_CDP_MODULE",
        "MCP_BROWSER_CDP_URL",
        "MCP_ENABLE_WEB_FETCH_MODULE",
        "MCP_ENABLE_WEB_SEARCH_MODULE",
        "MCP_ENABLE_WEB_RESEARCH_MODULE",
        "MCP_RUN_COMMAND_SPILL_DIR",
    }
    stale = documented - aliases - allowed_module_or_runtime_vars
    _require(not stale, f"Unexpected documented MCP vars: {sorted(stale)}")
```

Run red:

```bash
source .venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py
```

Expected initial failures:

- `Smoke_Client.md` still references `mcp_unified[gateway]`.
- `Smoke_Client.md` still uses `/mcp/request` without sufficient package-local framing.
- `System_Admin_Guide.md` still lists stale `MCP_*` variables.

**Implementation:**

- [ ] Update `Docs/MCP/Unified/Smoke_Client.md`.
  - Use `python -m pip install -e "apps/mcp-unified[gateway]"` only if referring to editable package tests.
  - Prefer embedded TLDW examples:
    - `GET http://127.0.0.1:8000/api/v1/mcp/status`
    - `POST http://127.0.0.1:8000/api/v1/mcp/request`
  - Label `/mcp/request` as package-local/host-mounted only.
- [ ] Update `Docs/MCP/Unified/Client_Snippets.md`.
  - Add a status preflight against `/api/v1/mcp/status`.
  - Keep auth examples header-based.
- [ ] Update `Docs/MCP/Unified/User_Guide.md`.
  - Ensure query auth is explicitly "legacy, disabled by default".
  - Ensure the golden path shows status, initialize, tools/list, and tools/call with expected response markers.
- [ ] Add or tighten a "Which path should I use?" section in `Docs/MCP/Unified/README.md` or `Docs/MCP/Unified/User_Guide.md`.
  - Embedded TLDW Server: `/api/v1/mcp/status`, `/api/v1/mcp/request`, `/api/v1/mcp/ws`.
  - Package-local mounted gateway: `/mcp/status`, `/mcp/request`, only when an embedding app mounts `apps/mcp-unified`.
  - Standalone gateway process: planned but not shipped; no `serve` command.
- [ ] Update `Docs/MCP/Unified/System_Admin_Guide.md`.
  - Remove stale config block entries that are not `MCPConfig` aliases or explicitly supported module/runtime vars.
  - Replace with current AuthNZ plus MCP env examples: `AUTH_MODE`, `SINGLE_USER_API_KEY`, `MCP_JWT_SECRET`, `MCP_API_KEY_SALT`, `MCP_DATABASE_URL`, `MCP_RATE_LIMIT_*`, `MCP_WS_*`, `MCP_HTTP_MAX_BODY_BYTES`, CORS/security-header vars, and module opt-in vars.
- [ ] Update package docs if needed:
  - `apps/mcp-unified/README.md`
  - `apps/mcp-unified/USER_GUIDE.md`
  - `apps/mcp-unified/src/mcp_unified/README.md`
  - Keep package docs clear that CLI commands are management utilities, not a server launcher.

**Verification:**

```bash
source .venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py
```

**Commit:**

```bash
git add Docs/MCP/Unified apps/mcp-unified tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py
git commit -m "docs: clarify mcp embedded and package paths"
```

---

## Stage 2: Safer Module Defaults And Status Disclosure

**Goal:** Disable high-risk local file/process modules by default and make `/api/v1/mcp/status` explain which high-risk modules are available but require explicit opt-in.

**Success Criteria:**

- `filesystem` and `run_command` are disabled in the default `mcp_modules.yaml`.
- Missing-YAML fallback no longer auto-enables `filesystem` by default.
- Explicit YAML `enabled: true` and explicit `MCP_ENABLE_FILESYSTEM_MODULE=true` still opt in.
- `/api/v1/mcp/status` includes enabled risk tiers and a disabled-available list for high-risk modules.
- Users see a next action for enabling disabled high-risk modules.
- A migration note and explicit opt-in YAML example explain how existing operators can restore previous local file/process module behavior.

**Tests First:**

- [ ] Extend `tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py`.

Add or adapt:

```python
def test_describe_module_surface_reports_disabled_available_high_risk_modules():
    from tldw_Server_API.app.core.MCP_unified.module_surface import describe_module_surface

    surface = describe_module_surface({
        "media": {"enabled": True, "status": "healthy"},
        "filesystem": {"enabled": False, "status": "disabled"},
        "run_command": {"enabled": False, "status": "disabled"},
        "external_federation": {"enabled": False, "status": "disabled"},
    })

    assert surface["enabled_count"] == 1
    assert [m["id"] for m in surface["tiers"]["read_only"]["modules"]] == ["media"]
    disabled_ids = [m["id"] for m in surface["disabled_available"]]
    assert disabled_ids == ["external_federation", "filesystem", "run_command"]
    assert all(m["requires_explicit_opt_in"] is True for m in surface["disabled_available"])
```

```python
def test_default_mcp_modules_yaml_disables_local_file_and_process_modules():
    import yaml
    from pathlib import Path

    data = yaml.safe_load(Path("tldw_Server_API/Config_Files/mcp_modules.yaml").read_text())
    modules = {entry["id"]: entry for entry in data["modules"]}

    assert modules["filesystem"]["enabled"] is False
    assert modules["run_command"]["enabled"] is False
    assert modules["codegraph"]["enabled"] is False
```

```python
async def test_filesystem_fallback_requires_explicit_opt_in(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    missing_config = tmp_path / "missing-mcp-modules.yaml"
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(missing_config))
    monkeypatch.setenv("MCP_MODULES", "")
    monkeypatch.delenv("MCP_ENABLE_FILESYSTEM_MODULE", raising=False)

    registered = []
    server = MCPServer()

    async def _register_module(module_id, cls, config):
        registered.append(module_id)

    monkeypatch.setattr(server.module_registry, "register_module", _register_module)

    await server._register_default_modules()

    assert "filesystem" not in registered
```

```python
async def test_filesystem_fallback_registers_with_explicit_env_opt_in(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    missing_config = tmp_path / "missing-mcp-modules.yaml"
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(missing_config))
    monkeypatch.setenv("MCP_MODULES", "")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "true")

    registered = []
    server = MCPServer()

    async def _register_module(module_id, cls, config):
        registered.append(module_id)

    monkeypatch.setattr(server.module_registry, "register_module", _register_module)

    await server._register_default_modules()

    assert "filesystem" in registered
```

```python
async def test_explicit_yaml_enabled_high_risk_modules_still_register(monkeypatch, tmp_path):
    import textwrap
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    config_path = tmp_path / "mcp_modules.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            modules:
              - id: filesystem
                class: tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module:FilesystemModule
                enabled: true
                name: Filesystem
                version: "1.0.0"
                department: system
                settings: {}
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(config_path))
    monkeypatch.setenv("MCP_MODULES", "")
    monkeypatch.delenv("MCP_ENABLE_FILESYSTEM_MODULE", raising=False)

    registered = []
    server = MCPServer()

    async def _register_module(module_id, cls, config):
        registered.append(module_id)

    monkeypatch.setattr(server.module_registry, "register_module", _register_module)

    await server._register_default_modules()

    assert "filesystem" in registered
```

```python
async def test_server_status_includes_disabled_available_from_config(monkeypatch):
    from tldw_Server_API.app.core.MCP_unified.modules.base import HealthStatus, ModuleHealth
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    server = MCPServer()
    server.initialized = True
    server._configured_modules_for_status = {
        "media": {"enabled": True},
        "filesystem": {"enabled": False},
        "run_command": {"enabled": False},
    }

    async def _check_all_health():
        return {"media": ModuleHealth(status=HealthStatus.HEALTHY)}

    monkeypatch.setattr(server.module_registry, "check_all_health", _check_all_health)

    status = await server.get_status()

    assert status["surface"]["enabled_count"] == 1
    assert [m["id"] for m in status["surface"]["disabled_available"]] == ["filesystem", "run_command"]
```

Run red:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  -k "module_surface or default_mcp_modules_yaml or filesystem_fallback or explicit_yaml_enabled_high_risk or server_status_includes"
```

Expected initial failures:

- `filesystem` and `run_command` are enabled in YAML.
- `server.py` auto-enables `filesystem` when absent from YAML.
- `describe_module_surface()` omits disabled modules.
- `server.get_status()` has no configured disabled module memory.

**Implementation:**

- [ ] Update `tldw_Server_API/app/core/MCP_unified/module_surface.py`.
  - Add a constant such as:
    - `EXPLICIT_OPT_IN_TIERS = {"local_files", "local_process", "external_network"}`
  - Keep existing `enabled_count` and `tiers` behavior for compatibility.
  - Add:
    - `disabled_available_count`
    - `disabled_available`
  - Each disabled entry should include:
    - `id`
    - `tier`
    - `label`
    - `description`
    - `requires_explicit_opt_in: True`
    - `next_action`
  - Keep deterministic sorted ordering by module id.
- [ ] Update `tldw_Server_API/app/core/MCP_unified/server.py`.
  - Add an instance field, for example `self._configured_modules_for_status: dict[str, dict[str, Any]] = {}`.
  - Populate it during `_register_default_modules()` from all YAML/env/fallback candidates before skipping disabled modules.
  - Store only sanitized status fields: `enabled`, `status`, `name`, `department`.
  - Change filesystem fallback from default-on to explicit opt-in:
    - Use `self._env_flag_enabled("MCP_ENABLE_FILESYSTEM_MODULE")` instead of `os.getenv(..., "true")`.
  - In `get_status()`, merge health results into the configured module map before calling `describe_module_surface()`.
- [ ] Update `tldw_Server_API/Config_Files/mcp_modules.yaml`.
  - Set `filesystem.enabled: false`.
  - Set `run_command.enabled: false`.
  - Add short comments explaining explicit opt-in and status verification.
- [ ] Add an operator opt-in example for previous local module behavior.
  - Prefer `tldw_Server_API/Config_Files/mcp_modules.local_opt_in.example.yaml` or a clearly titled section in `Docs/MCP/Unified/Using_Modules_YAML.md`.
  - Include `filesystem.enabled: true` and `run_command.enabled: true` examples with risk notes.
- [ ] Update module docs:
  - `Docs/MCP/Unified/Modules.md`
  - `Docs/MCP/Unified/Using_Modules_YAML.md`
  - `Docs/MCP/Unified/User_Guide.md`
  - Mention `surface.disabled_available`, `requires_explicit_opt_in`, and the restart requirement.
  - Add a migration note: after this change, default installs no longer expose local file/process modules unless explicitly enabled.

**Verification:**

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  -k "module_surface or default_mcp_modules_yaml or filesystem_fallback or explicit_yaml_enabled_high_risk or server_status_includes"
```

**Commit:**

```bash
git add \
  tldw_Server_API/app/core/MCP_unified/module_surface.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/Config_Files/mcp_modules.yaml \
  tldw_Server_API/Config_Files/mcp_modules.local_opt_in.example.yaml \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  Docs/MCP/Unified/Modules.md \
  Docs/MCP/Unified/Using_Modules_YAML.md \
  Docs/MCP/Unified/User_Guide.md
git commit -m "feat: make risky mcp modules explicit opt in"
```

---

## Stage 3: HTTP And JSON-RPC Recovery Metadata

**Goal:** Make common failures actionable without breaking existing HTTP helper or JSON-RPC clients.

**Success Criteria:**

- HTTP string `detail` responses remain strings.
- HTTP string-detail recovery metadata is exposed via additive headers.
- HTTP object `detail` responses include `reason_code` and `next_action`.
- JSON-RPC known invalid-params and authorization failures include recovery metadata under `error.data`.
- `/api/v1/mcp/modules` permission copy mentions modules, not tools.
- WebSocket endpoint copy does not normalize query auth.

**Tests First:**

- [ ] Extend `tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py`.

Add tests equivalent to:

```python
def test_tools_execute_unauth_preserves_string_detail_and_adds_recovery_headers(client: TestClient):
    response = client.post(
        "/api/v1/mcp/tools/execute",
        json={"tool_name": "media.search", "arguments": {}},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required"
    assert response.headers["x-mcp-reason-code"] == "authentication_required"
    assert response.headers["x-mcp-next-action"]
```

```python
def test_modules_permission_detail_mentions_modules_and_recovery(client: TestClient):
    response = client.get("/api/v1/mcp/modules")

    assert response.status_code == 403
    detail = response.json()["detail"]
    assert detail["reason_code"] == "permission_denied"
    assert detail["next_action"]
    assert "modules" in detail["hint"].lower()
    assert "tools" not in detail["hint"].lower()
```

```python
def test_invalid_safe_config_keeps_structured_recovery_detail(client: TestClient):
    response = client.post(
        "/api/v1/mcp/request",
        json={"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1},
        params={"config": "not-base64-json"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_safe_config"
    assert detail["next_action"]
```

Add a focused JSON-RPC test in the same file or in a protocol test file:

```python
async def test_jsonrpc_invalid_params_error_data_includes_recovery_metadata():
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest
    from tldw_Server_API.app.core.MCP_unified.protocol_types import RequestContext

    protocol = MCPProtocol()
    response = await protocol.process_request(
        MCPRequest(method="tools/call", params={}, id=1),
        RequestContext(request_id="invalid-params-test", client_id="pytest", user_id="1"),
    )

    assert response.error is not None
    assert response.error.code == -32602
    assert response.error.data["reason_code"] == "invalid_params"
    assert response.error.data["next_action"]
```

Add a WebSocket copy/signature test. Do not assert on OpenAPI for this route; FastAPI WebSocket routes are not reliably emitted as HTTP OpenAPI paths.

```python
def test_websocket_query_auth_is_marked_legacy_in_endpoint_copy():
    import inspect
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as endpoint

    signature = inspect.signature(endpoint.websocket_endpoint)
    token_description = signature.parameters["token"].default.description.lower()
    api_key_description = signature.parameters["api_key"].default.description.lower()
    docstring = inspect.getdoc(endpoint.websocket_endpoint).lower()

    assert "legacy" in token_description
    assert "disabled by default" in token_description
    assert "legacy" in api_key_description
    assert "disabled by default" in api_key_description
    assert "?token=jwt-token" not in docstring
```

Run red:

```bash
source .venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py
```

Expected initial failures:

- String-detail HTTP errors have no recovery headers.
- `/modules` permission hint says "listing tools".
- JSON-RPC invalid params only adds a hint, not `reason_code` and `next_action`.
- WebSocket token query parameter is described as normal auth.

**Implementation:**

- [ ] Update `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`.
  - Add helpers:
    - `_mcp_recovery_headers(reason_code: str, next_action: str) -> dict[str, str]`
    - `_mcp_permission_detail(response: MCPResponse, *, hint: str, reason_code: str = "permission_denied", next_action: str) -> dict[str, str] | None`
    - Optional `_raise_string_detail_error(...)` wrapper to avoid repeated header boilerplate.
  - For existing string-body errors, keep the same string `detail` and add headers:
    - `401 Authentication required` -> `x-mcp-reason-code: authentication_required`
    - `400 invalid params` -> `x-mcp-reason-code: invalid_params`
    - `502 no response` -> `x-mcp-reason-code: upstream_no_response`
    - `500 generic MCP tool execution failed` -> `x-mcp-reason-code: mcp_tool_execution_failed`
  - For existing object-body permission errors, add `reason_code` and `next_action`.
  - Fix `/modules` hint from "listing tools" to "listing modules".
  - Update WebSocket `token` and `api_key` `Query(...)` descriptions to say legacy query auth is disabled by default and headers/subprotocol auth is preferred.
  - Update the WebSocket docstring example to omit `?token=jwt-token`.
- [ ] Update `tldw_Server_API/app/core/MCP_unified/protocol.py`.
  - Preserve existing `_attach_error_hint()` behavior.
  - When `data is None` and the code/message match a known recoverable case, return:
    - `reason_code`
    - `next_action`
    - existing `hint`, when applicable
  - Do not overwrite existing governance or approval `error.data`.
- [ ] Update troubleshooting docs if needed:
  - `Docs/MCP/Unified/User_Guide.md`
  - `Docs/MCP/Unified/Developer_Guide.md`
  - Document that HTTP helper endpoints may expose recovery metadata in `detail` or `x-mcp-*` headers depending on backward-compatible body shape.

**Verification:**

```bash
source .venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py
```

**Commit:**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py \
  Docs/MCP/Unified/User_Guide.md \
  Docs/MCP/Unified/Developer_Guide.md
git commit -m "feat: add mcp recovery metadata without breaking responses"
```

---

## Stage 4: Package Gateway Readiness Status

**Goal:** Make package-local `/status` useful for power users and embedders without implying the package is published or a supported standalone server.

**Success Criteria:**

- `GET /mcp/status` still returns `status`, `name`, and `version`.
- Response adds static package boundary metadata:
  - `package_status: internal-experimental`
  - `publishing_status: not-published`
  - `source_distribution: tldw-server`
- Response exposes best-effort readiness:
  - profile store kind/persistence
  - external registry store kind/persistence when mounted
  - default profile status when available
  - admin auth enabled/configured state
  - external server counts when a registry manager is mounted
  - warning/next action list
- No secrets, env var values, API keys, commands, or credential material are returned.

**Tests First:**

- [ ] Extend `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`.

Add tests near existing gateway route/status tests:

```python
def test_gateway_status_includes_package_boundary_metadata() -> None:
    app = create_gateway_app(_FakeGatewayRuntime())
    with TestClient(app) as client:
        response = client.get("/mcp/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["name"] == "unit-gateway"
    assert payload["version"] == "0.0-test"
    assert payload["package"]["package_status"] == "internal-experimental"
    assert payload["package"]["publishing_status"] == "not-published"
    assert payload["package"]["source_distribution"] == "tldw-server"
    assert "next_actions" in payload
```

```python
def test_gateway_status_reports_profile_store_admin_auth_and_default_profile() -> None:
    manager = _ProfileManagementManagerDouble("default")
    # If the status implementation reads manager.store_metadata directly, add the same
    # lightweight shape used by the real manager instead of changing unrelated doubles.
    manager.store_metadata = type(
        "_StoreMetadata",
        (),
        {"to_payload": lambda self: {"kind": "memory", "persistent": False}},
    )()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        profile_manager=manager,
        enable_profile_management=True,
        admin_auth=GatewayAdminAuthConfig(enabled=True, api_key="unit-test-key"),
    )
    with TestClient(app) as client:
        response = client.get("/mcp/status")

    payload = response.json()
    assert payload["profile_store"]["kind"] in {"memory", "sqlite"}
    assert payload["default_profile"]["configured"] is True
    assert payload["admin_auth"]["enabled"] is True
    assert payload["admin_auth"]["configured"] is True
    assert "unit-test-key" not in json.dumps(payload)
```

```python
def test_gateway_status_counts_external_servers_best_effort() -> None:
    manager = _ExternalRegistryManagerDouble("enabled")
    manager.store_metadata = type(
        "_StoreMetadata",
        (),
        {"to_payload": lambda self: {"kind": "memory", "persistent": False}},
    )()
    app = create_gateway_app(
        _FakeGatewayRuntime(),
        external_registry_manager=manager,
        enable_external_registry_management=True,
    )
    with TestClient(app) as client:
        payload = client.get("/mcp/status").json()

    assert payload["external_servers"]["total"] >= 1
    assert payload["external_servers"]["enabled"] >= 1
    assert payload["external_registry_store"]["kind"] in {"memory", "sqlite"}
```

Run red:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  -k "gateway_status"
```

Expected initial failures:

- `/mcp/status` only returns `status`, `name`, and `version`.

**Implementation:**

- [ ] Update `apps/mcp-unified/src/mcp_unified/gateway/fastapi.py`.
  - Import `package_metadata_summary`.
  - Add an async helper such as `_gateway_readiness_status(...)`.
  - Keep existing `status`, `name`, and `version` keys.
  - Include a compact `package` object with only non-sensitive fields:
    - `package_name`
    - `package_import_name`
    - `package_status`
    - `publishing_status`
    - `source_distribution`
    - `dependency_version_policy`
  - Use `profile_manager.store_metadata.to_payload()` when available.
  - Use `profile_manager.get_default_profile()` best-effort, catching known management errors and generic exceptions into non-secret warning objects.
  - Use `external_registry_manager.store_metadata.to_payload()` and `list_servers(enabled=None)`/`list_servers(enabled=True)` best-effort for counts.
  - Use `admin_auth.enabled` and non-secret configured state; do not return the API key.
  - Add warnings and next actions for:
    - package not published
    - memory/non-persistent store
    - default profile missing
    - admin auth enabled but key missing
    - external registry unavailable
- [ ] Update package docs:
  - `apps/mcp-unified/README.md`
  - `apps/mcp-unified/USER_GUIDE.md`
  - Mention `/status` as package-local readiness/status, not TLDW Server embedded status.

**Verification:**

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  -k "gateway_status"
```

**Commit:**

```bash
git add \
  apps/mcp-unified/src/mcp_unified/gateway/fastapi.py \
  apps/mcp-unified/README.md \
  apps/mcp-unified/USER_GUIDE.md \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
git commit -m "feat: expand package gateway readiness status"
```

---

## Stage 5: Docker Production-Signal Cleanup

**Goal:** Remove misleading production-readiness language from the MCP-specific Docker path while preserving the current experimental contract.

**Success Criteria:**

- MCP-specific Dockerfile and docs no longer describe the image as production-optimized or production-ready.
- Docker docs continue to say this path is experimental and not the supported standalone gateway.
- Existing entrypoint contract remains unchanged.

**Tests First:**

- [ ] Extend `tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py`.

Add:

```python
DOCKERFILE = Path("tldw_Server_API/app/core/MCP_unified/docker/Dockerfile")


def test_mcp_specific_dockerfile_does_not_claim_production_readiness() -> None:
    _ensure(DOCKERFILE.exists(), "MCP-specific Dockerfile is missing")
    text = DOCKERFILE.read_text(encoding="utf-8").lower()
    forbidden = [
        "optimized for production deployment",
        "production-ready",
        "production grade",
    ]
    found = [phrase for phrase in forbidden if phrase in text]
    _ensure(not found, f"MCP-specific Dockerfile has misleading production language: {found}")
    _ensure("experimental" in text, "MCP-specific Dockerfile should label this path experimental")
```

Run red:

```bash
source .venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py
```

Expected initial failure:

- Dockerfile contains "Optimized for production deployment with security best practices".

**Implementation:**

- [ ] Update `tldw_Server_API/app/core/MCP_unified/docker/Dockerfile`.
  - Replace production-optimized comments with experimental/package-specific comments.
  - Keep security hygiene comments if they are factual, but avoid production-readiness claims.
- [ ] Update `tldw_Server_API/app/core/MCP_unified/docker/README.md` only if needed to keep language aligned.
- [ ] Update `tldw_Server_API/app/core/MCP_unified/README.md` only if needed to keep Docker section aligned.

**Verification:**

```bash
source .venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py
```

**Commit:**

```bash
git add \
  tldw_Server_API/app/core/MCP_unified/docker/Dockerfile \
  tldw_Server_API/app/core/MCP_unified/docker/README.md \
  tldw_Server_API/app/core/MCP_unified/README.md \
  tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py
git commit -m "docs: mark mcp docker path experimental"
```

---

## Stage 6: Final Verification And Backlog Finalization

**Goal:** Prove the residual hardening work is coherent, then update Backlog with results.

**Success Criteria:**

- All targeted tests pass.
- Bandit is run on the touched code scope.
- Backlog task has final summary, verification notes, and completed DoD.
- Git history contains small coherent commits.

**Verification Commands:**

Run targeted tests:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Docs/test_mcp_unified_docs_contract.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py
```

Run package gateway status tests:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  -k "gateway_status"
```

Run Docker contract:

```bash
source .venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_docker_packaging_contract.py
```

Run Bandit on touched code paths:

```bash
source .venv/bin/activate
python -m bandit \
  tldw_Server_API/app/core/MCP_unified/module_surface.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py \
  apps/mcp-unified/src/mcp_unified/gateway/fastapi.py \
  -f json \
  -o /tmp/bandit_mcp_residual_ux.json
```

If Bandit reports existing baseline findings outside changed lines, record them in Backlog and do not silently ignore new findings in touched code.

**Backlog Finalization:**

- [ ] Update `TASK-12054` while planning work is complete.
- [ ] If executing this implementation in the same branch, create or update a separate implementation task before code edits beyond this plan.
- [ ] Record:
  - Test commands and pass/fail result.
  - Bandit output path.
  - Known skips or blockers.
  - Final summary.

**Final Implementation Commit:**

Only after all stages pass:

```bash
git status --short
git log --oneline --decorate -5
```

Confirm only intended files are changed, then commit any final Backlog/doc-status update:

```bash
git add backlog/tasks/task-12054\ -\ Plan-MCP-Unified-residual-UX-hardening-implementation.md
git commit -m "chore: finalize mcp residual ux plan tracking"
```

---

## Acceptance Checklist

- [ ] Supported product mental model is clear: embedded TLDW MCP at `/api/v1/mcp`.
- [ ] Internal package boundary is clear: `apps/mcp-unified` is experimental and not published.
- [ ] High-risk local file/process modules are disabled by default.
- [ ] Explicit operator opt-ins still work.
- [ ] `/api/v1/mcp/status` shows enabled tiers plus disabled-available high-risk modules.
- [ ] `/mcp/status` package-local response is useful but does not imply publishing/support status.
- [ ] HTTP helper errors are more actionable without body-shape breakage.
- [ ] JSON-RPC recovery metadata is additive under `error.data`.
- [ ] Query auth is framed as legacy/disabled by default.
- [ ] Docker path no longer sends production-readiness signals.
- [ ] Docs contract tests prevent regression on the above.
- [ ] Targeted tests and Bandit verification are recorded.
