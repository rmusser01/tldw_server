# MCP Default Profile Tooling Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first reviewable slice of MCP role profile tooling presets: rich preset metadata, a patchable recommendation catalog, profile-scoped tool discovery/ranking, gateway bridge tools, and package docs.

**Architecture:** Keep executable authority in `MCPProfile.policy_document` and grants. Put display/setup guidance in `profile.metadata["tooling"]` and a separate recommendation catalog helper. Add pure catalog/ranking helpers before wiring them into `ProfileAwareGatewayRuntime`, so policy behavior can be tested without a transport.

**Tech Stack:** Python 3.10+, Pydantic v2, FastAPI/JSON-RPC gateway contracts, pytest, pytest-asyncio, Bandit.

---

## Scope

This plan implements the metadata and discovery surface needed by the spec. It does not implement the native browser, git, safe test runner, LSP, issue tracker, or web-search tools themselves.

Follow-up plans should cover:

- Native CDP browser inspection tools.
- Safe test runner command registry and execution approval flow.
- Git inspect/conflict-read tools.
- Native file search/edit helpers.
- External MCP install/update templates beyond the initial CDP exact target.

## Source Spec

- `Docs/superpowers/specs/2026-06-03-mcp-default-profile-tooling-presets-design.md`

## File Structure

- Modify: `mcp_unified/profiles/presets.py`
  - Bump preset release metadata.
  - Populate `metadata["tooling"]` for built-in role presets.
  - Extend safety validation for the new reviewed risk classes.
- Create: `mcp_unified/profiles/tooling.py`
  - Define role tooling metadata helpers, recommendation catalog defaults, progressive-disclosure defaults, and patchable recommendation merge helpers.
- Create: `mcp_unified/gateway/tool_discovery.py`
  - Build profile-scoped direct/deferred tool catalogs from profile metadata, backend tools, and installed external binding state.
  - Implement deterministic filter-first/BM25 ranking.
  - Resolve bridge `tool_id` values to underlying callable backend tool names.
- Modify: `mcp_unified/gateway/profile_runtime.py`
  - Expose bridge tools when the active profile has deferred categories.
  - Intercept `tool_categories.list`, `profile.tools.list`, `tool_search`, `tool_describe`, and `tool_call`.
  - Keep backend policy checks authoritative for actual execution.
- Modify: `mcp_unified/gateway/cli.py`
  - Include compact tooling summary fields in `list-presets` output while preserving `show-preset` full output.
- Modify: `mcp_unified/USER_GUIDE.md`
  - Document mode presets, direct vs recommended tools, CDP exact target, and recommendation catalog patching semantics.
- Modify: `mcp_unified/README.md`
  - Add a short pointer to role presets and progressive disclosure.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
  - Extend existing preset tests for metadata, safety, version, recommendation authority, and exact CDP target.
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py`
  - Unit-test pure discovery, ranking, and bridge resolution.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
  - Add runtime/JSON-RPC integration tests for bridge tools.
- Modify if needed: `mcp_unified/__init__.py`, `mcp_unified/profiles/__init__.py`, `mcp_unified/gateway/__init__.py`
  - Export new helpers only when they are intended as package-local API.

---

### Task 1: Preset Tooling Metadata Tests

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`
- Modify: `mcp_unified/profiles/presets.py`
- Create: `mcp_unified/profiles/tooling.py`

- [ ] **Step 1: Write failing tests for required tooling metadata**

Add tests that assert every default mode preset has a `metadata["tooling"]` object with direct tools, capabilities, recommendations, external server categories, and progressive disclosure:

```python
def test_role_presets_include_tooling_metadata() -> None:
    role_ids = {
        "product-owner",
        "architect",
        "merge-conflict-resolver",
        "documentation-writer",
        "project-researcher",
        "code-reviewer",
        "devops-engineer",
        "backend-engineer",
        "frontend-engineer",
        "qa-engineer",
        "sdet",
    }
    presets_by_id = {preset.id: preset for preset in presets.list_builtin_presets()}

    for preset_id in role_ids:
        tooling = presets_by_id[preset_id].profile.metadata["tooling"]
        assert tooling["enabled_tools"]
        assert tooling["enabled_capabilities"]
        assert "recommended_tools" in tooling
        assert "recommended_servers" in tooling
        assert tooling["recommendation_catalog_patchable"] is True
        assert tooling["progressive_disclosure"]["max_direct_tools"] <= 24
```

Add targeted tests:

```python
def test_web_search_is_recommended_unavailable_not_enabled() -> None:
    product_owner = presets.get_builtin_preset("product-owner")
    assert product_owner is not None
    tooling = product_owner.profile.metadata["tooling"]
    assert "web.search" not in product_owner.profile.policy_document.allowed_tools
    assert any(
        item["category"] == "web_search"
        and item["required"] is False
        for item in tooling["recommended_servers"]
    )


def test_cdp_browser_exact_target_is_documented() -> None:
    frontend = presets.get_builtin_preset("frontend-engineer")
    assert frontend is not None
    browser_servers = [
        server
        for server in frontend.profile.metadata["tooling"]["recommended_servers"]
        if server["category"] == "browser"
    ]
    assert browser_servers
    assert any(
        option["id"] == "chrome-devtools-mcp"
        and option["install_target"] == "ChromeDevTools/chrome-devtools-mcp"
        and option["maturity"] == "exact_target"
        for server in browser_servers
        for option in server["binding_options"]
    )


def test_recommendation_catalog_patch_does_not_grant_authority() -> None:
    from mcp_unified.profiles.tooling import merge_tooling_recommendations

    product_owner = presets.get_builtin_preset("product-owner")
    assert product_owner is not None
    patched_tooling = merge_tooling_recommendations(
        product_owner.profile.metadata["tooling"],
        {
            "recommended_tools": [
                {
                    "id": "shell.run",
                    "category": "shell",
                    "activation": "requires_operator_enablement",
                }
            ]
        },
    )

    assert any(item["id"] == "shell.run" for item in patched_tooling["recommended_tools"])
    assert "shell.run" not in product_owner.profile.policy_document.allowed_tools
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q
```

Expected: FAIL because `metadata["tooling"]` and the exact CDP binding target do not exist yet.

- [ ] **Step 3: Create `mcp_unified/profiles/tooling.py`**

Implement package-local helpers with dict outputs, not executable policy side effects:

```python
"""Default profile tooling metadata and recommendation catalog helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_MAX_DIRECT_TOOLS = 24

CHROME_DEVTOOLS_MCP_OPTION: dict[str, Any] = {
    "id": "chrome-devtools-mcp",
    "category": "browser",
    "kind": "external_mcp",
    "install_target": "ChromeDevTools/chrome-devtools-mcp",
    "credential_slots": [],
    "required_scopes": [],
    "risk_classes": ["external_network"],
    "maturity": "exact_target",
    "setup_url": "https://github.com/ChromeDevTools/chrome-devtools-mcp",
}


def tooling_metadata(
    *,
    enabled_tools: list[str],
    enabled_capabilities: list[str],
    direct_categories: list[str],
    deferred_categories: list[str],
    recommended_tools: list[dict[str, Any]] | None = None,
    recommended_servers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return caller-owned tooling metadata for a role preset."""
    return {
        "enabled_tools": list(enabled_tools),
        "enabled_capabilities": list(enabled_capabilities),
        "recommended_tools": deepcopy(recommended_tools or []),
        "recommended_servers": deepcopy(recommended_servers or []),
        "recommendation_catalog_patchable": True,
        "progressive_disclosure": {
            "direct_categories": list(direct_categories),
            "deferred_categories": list(deferred_categories),
            "max_direct_tools": DEFAULT_MAX_DIRECT_TOOLS,
        },
        "tool_search": {
            "ranking": ["profile_grants", "installation_status", "category", "bm25"],
            "semantic_search": False,
        },
    }
```

Add small builders for common recommendation server entries:

```python
def browser_server_recommendation() -> dict[str, Any]:
    """Return the browser/CDP recommendation category."""
    return {
        "category": "browser",
        "required": False,
        "binding_options": [deepcopy(CHROME_DEVTOOLS_MCP_OPTION)],
    }


def web_search_server_recommendation() -> dict[str, Any]:
    """Return vendor-neutral web-search recommendation metadata."""
    return {
        "category": "web_search",
        "required": False,
        "binding_options": [
            {
                "id": "configured-web-search",
                "category": "web_search",
                "kind": "external_mcp",
                "install_target": None,
                "credential_slots": [],
                "required_scopes": ["search:read"],
                "risk_classes": ["external_network"],
                "maturity": "category_placeholder",
                "activation": "requires_configured_provider",
            }
        ],
    }


def issue_tracker_server_recommendation() -> dict[str, Any]:
    """Return vendor-neutral issue-tracker recommendation metadata."""
    return {
        "category": "issue_tracker",
        "required": False,
        "binding_options": [
            {
                "id": "jira",
                "category": "issue_tracker",
                "kind": "external_mcp",
                "install_target": None,
                "credential_slots": ["jira_api_token"],
                "required_scopes": ["issues:read", "issues:write"],
                "risk_classes": ["external_network", "mutating"],
                "maturity": "category_placeholder",
            },
            {
                "id": "linear",
                "category": "issue_tracker",
                "kind": "external_mcp",
                "install_target": None,
                "credential_slots": ["linear_api_key"],
                "required_scopes": ["issues:read", "issues:write"],
                "risk_classes": ["external_network", "mutating"],
                "maturity": "category_placeholder",
            },
        ],
    }


def merge_tooling_recommendations(
    tooling: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Return patched recommendation metadata without changing policy."""
    merged = deepcopy(tooling)
    for key in ("recommended_tools", "recommended_servers"):
        additions = patch.get(key)
        if isinstance(additions, list):
            merged.setdefault(key, [])
            merged[key].extend(deepcopy(additions))
    return merged
```

Keep helpers simple and data-oriented. Do not import from `tldw_Server_API`.

- [ ] **Step 4: Populate preset metadata in `mcp_unified/profiles/presets.py`**

Update imports:

```python
from .tooling import (
    browser_server_recommendation,
    tooling_metadata,
    web_search_server_recommendation,
)
```

Update `_profile()` to accept `tooling_metadata_document: dict[str, Any] | None = None` and merge it under `metadata`:

```python
metadata = {
    "agent_metadata": {
        "ui_label": name,
        **(agent_metadata or {}),
    }
}
if tooling_metadata_document is not None:
    metadata["tooling"] = tooling_metadata_document
```

Update `_preset()` to pass the new metadata. Populate at least the eleven requested role presets from the spec. Example for Product Owner:

```python
tooling_metadata_document=tooling_metadata(
    enabled_tools=[
        "tool_categories.list",
        "tool_search",
        "tool_describe",
        "profile.tools.list",
        "fs.list",
        "fs.read_text",
        "fs.write_text",
        "kanban.cards.create",
        "memory.recall",
    ],
    enabled_capabilities=[
        "filesystem.read",
        "filesystem.write_scoped",
        "issues.plan",
        "stories.write",
        "memory.read",
    ],
    direct_categories=["files", "tool_discovery", "issues", "memory"],
    deferred_categories=["issue_tracker", "web_search", "docs_search", "browser"],
    recommended_servers=[
        web_search_server_recommendation(),
        browser_server_recommendation(),
        issue_tracker_server_recommendation(),
    ],
),
```

Do not grant `web.search`, browser interaction, shell, SSH, deployment mutation, arbitrary process execution, or memory writes by default.

- [ ] **Step 5: Run preset tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mcp_unified/profiles/tooling.py mcp_unified/profiles/presets.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
git commit -m "feat: add mcp profile tooling metadata"
```

---

### Task 2: Risk-Class Safety Validation

**Files:**
- Modify: `mcp_unified/profiles/presets.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py`

- [ ] **Step 1: Write failing safety tests for reviewed risk classes**

Add tests for the new reviewed risk classes:

```python
def test_safety_validation_accepts_reviewed_high_risk_classes_with_approval_and_provenance() -> None:
    profile = MCPProfile(
        id="safe-browser",
        name="Safe Browser",
        policy_document={
            "risk_classes": ["browser_mutation", "git_mutation", "deployment_mutation", "memory_mutation", "test_execution"],
        },
        approval_policy={"required_for": ["browser_mutation", "git_mutation", "deployment_mutation", "memory_mutation", "test_execution"]},
        provenance={
            "high_risk": {
                "browser_mutation": "reviewed",
                "git_mutation": "reviewed",
                "deployment_mutation": "reviewed",
                "memory_mutation": "reviewed",
                "test_execution": "reviewed",
            }
        },
    )
    preset = presets.ProfilePreset(id="safe-browser", version="test", profile=profile)
    assert presets.validate_preset_safety(preset) == []
```

Add a negative test:

```python
def test_safety_validation_rejects_reviewed_high_risk_class_without_approval() -> None:
    profile = MCPProfile(
        id="unsafe-browser",
        name="Unsafe Browser",
        policy_document={"risk_classes": ["browser_mutation"]},
        provenance={"high_risk": {"browser_mutation": "reviewed"}},
    )
    preset = presets.ProfilePreset(id="unsafe-browser", version="test", profile=profile)
    assert "browser_mutation_requires_approval" in presets.validate_preset_safety(preset)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q
```

Expected: FAIL because the validator treats these as unknown high-risk classes or does not require approval/provenance.

- [ ] **Step 3: Extend safety validator**

In `mcp_unified/profiles/presets.py`, add reviewed classes:

```python
_APPROVAL_REQUIRED_RISK_CLASSES = {
    "browser_mutation",
    "deployment_mutation",
    "git_mutation",
    "memory_mutation",
    "test_execution",
}

_HIGH_RISK_RISK_CLASSES = {
    "credential_use",
    "destructive_filesystem",
    "external_network",
    "process_execution",
    *_APPROVAL_REQUIRED_RISK_CLASSES,
}
```

Then add a loop in `validate_preset_safety()`:

```python
for risk_class in sorted(risk_classes & _APPROVAL_REQUIRED_RISK_CLASSES):
    if not _approval_required_for(profile, risk_class):
        violations.append(f"{risk_class}_requires_approval")
    if not _has_high_risk_provenance(profile, risk_class):
        violations.append("high_risk_capability_requires_provenance")
```

Keep unknown risk classes failing review.

- [ ] **Step 4: Run safety tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_unified/profiles/presets.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py
git commit -m "test: cover mcp profile risk class validation"
```

---

### Task 3: Profile-Scoped Tool Discovery And Ranking

**Files:**
- Create: `mcp_unified/gateway/tool_discovery.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py`

- [ ] **Step 1: Write failing tests for filtering and ranking**

Create tests with backend tools and a profile:

```python
from mcp_unified.gateway.tool_discovery import search_profile_tools
from mcp_unified.profiles.models import MCPProfile, ProfilePolicy


def test_tool_search_filters_by_profile_before_bm25() -> None:
    profile = MCPProfile(
        id="reviewer",
        name="Reviewer",
        policy_document=ProfilePolicy(capabilities=["code_search"]),
    )
    tools = [
        {"name": "code.search", "description": "Search code", "metadata": {"capability": "code_search", "category": "code"}},
        {"name": "shell.run", "description": "Run shell commands", "metadata": {"capability": "process.execute", "category": "shell"}},
    ]

    results = search_profile_tools(profile, tools, query="run search")

    assert [item["tool_id"] for item in results] == ["code.search"]
```

Add ranking test:

```python
def test_tool_search_orders_installed_before_unavailable_then_bm25() -> None:
    profile = MCPProfile(
        id="frontend",
        name="Frontend",
        policy_document=ProfilePolicy(capabilities=["browser.inspect"]),
        metadata={
            "tooling": {
                "recommended_tools": [
                    {
                        "id": "browser.trace",
                        "category": "browser",
                        "description": "Browser trace capture",
                        "activation": "requires_browser_runtime",
                    }
                ],
                "progressive_disclosure": {"direct_categories": [], "deferred_categories": ["browser"], "max_direct_tools": 24},
            }
        },
    )
    installed = [
        {"name": "browser.snapshot", "description": "Browser DOM snapshot", "metadata": {"capability": "browser.inspect", "category": "browser"}}
    ]

    results = search_profile_tools(profile, installed, query="browser", category="browser")

    assert [item["tool_id"] for item in results] == ["browser.snapshot", "browser.trace"]
    assert results[0]["installation_status"] == "installed"
    assert results[1]["installation_status"] == "recommended_unavailable"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py -q
```

Expected: FAIL because `mcp_unified.gateway.tool_discovery` does not exist.

- [ ] **Step 3: Implement catalog and ranking helpers**

Implement these functions:

```python
def list_profile_tools(profile: MCPProfile, backend_tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Return profile-scoped direct, deferred, and recommended tool catalog data."""


def search_profile_tools(
    profile: MCPProfile,
    backend_tools: list[dict[str, Any]],
    *,
    query: str = "",
    category: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return ranked tool-search results scoped to profile policy."""


def describe_profile_tool(profile: MCPProfile, backend_tools: list[dict[str, Any]], tool_id: str) -> dict[str, Any] | None:
    """Return one visible tool descriptor by profile-scoped tool id."""


def resolve_profile_tool_call(profile: MCPProfile, backend_tools: list[dict[str, Any]], tool_id: str) -> dict[str, Any]:
    """Resolve a bridge tool id to an installed backend tool or an unavailable reason."""
```

Use `build_effective_policy_result()` for profile grants, not ad hoc allow logic.

Implement small local BM25 helpers with standard library only:

```python
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./:-]+", text.lower())
```

No semantic-search dependency is allowed in this first implementation.

- [ ] **Step 4: Run discovery tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp_unified/gateway/tool_discovery.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py
git commit -m "feat: add profile scoped tool discovery"
```

---

### Task 4: Gateway Bridge Tool Integration

**Files:**
- Modify: `mcp_unified/gateway/profile_runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Write failing runtime tests for bridge visibility**

Add tests using existing fake runtimes in `test_gateway_fastapi_package.py` or a new small runtime double:

```python
@pytest.mark.asyncio
async def test_profile_runtime_exposes_discovery_bridge_tools_for_deferred_categories() -> None:
    from mcp_unified.profiles.store import InMemoryProfileStore

    profile = MCPProfile(
        id="frontend",
        name="Frontend",
        policy_document=ProfilePolicy(capabilities=["browser.inspect"]),
        metadata={
            "tooling": {
                "progressive_disclosure": {
                    "direct_categories": ["tool_discovery"],
                    "deferred_categories": ["browser"],
                    "max_direct_tools": 24,
                },
                "recommended_tools": [],
                "recommended_servers": [],
            }
        },
    )
    runtime = ProfileAwareGatewayRuntime(
        _MultiToolGatewayRuntime(),
        profile_store=InMemoryProfileStore([profile]),
        default_profile_id="frontend",
    )

    tools = await runtime.list_tools(GatewayRequestContext(request_id="req"))

    assert any(tool["name"] == "tool_search" for tool in tools)
    assert any(tool["name"] == "tool_describe" for tool in tools)
    assert any(tool["name"] == "tool_call" for tool in tools)
```

Add tests for `tool_search`, `tool_describe`, and `tool_call`:

```python
@pytest.mark.asyncio
async def test_tool_call_rejects_recommended_unavailable_tool() -> None:
    result = await runtime.call_tool(
        "tool_call",
        {"tool_id": "browser.trace", "arguments": {}},
        GatewayRequestContext(request_id="req"),
    )
    assert result["error"]["reason_code"] == "tool_not_enabled"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: FAIL because bridge tools are not exposed or intercepted yet.

- [ ] **Step 3: Add bridge tool descriptors**

In `mcp_unified/gateway/profile_runtime.py`, add constants for direct bridge descriptors:

```python
_DISCOVERY_BRIDGE_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "tool_categories.list",
        "description": "List profile-visible tool categories.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "metadata": {"category": "tool_discovery", "capability": "tool_discovery.read"},
    },
    ...
)
```

Only append `tool_call` when `metadata["tooling"]["progressive_disclosure"]["deferred_categories"]` is non-empty.

- [ ] **Step 4: Intercept bridge calls**

In `ProfileAwareGatewayRuntime.call_tool()`:

1. Resolve profile first.
2. If `name` is one of the discovery bridge tools, call helper methods.
3. For `tool_call`, validate schema manually:

```python
if set(arguments) - {"tool_id", "arguments"}:
    raise GatewayPolicyDenied(... reason_code="invalid_tool_call_arguments")
if not isinstance(arguments.get("tool_id"), str) or not isinstance(arguments.get("arguments"), dict):
    raise GatewayPolicyDenied(... reason_code="invalid_tool_call_arguments")
```

4. Resolve the underlying installed tool with `resolve_profile_tool_call()`.
5. If unavailable, return a structured non-executing payload:

```python
return {
    "error": {
        "reason_code": "tool_not_enabled",
        "tool_id": requested_tool_id,
    }
}
```

6. If installed, run the existing policy path against the real backend tool name and call the backend.

- [ ] **Step 5: Run bridge integration tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mcp_unified/gateway/profile_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
git commit -m "feat: expose profile scoped tool discovery bridge"
```

---

### Task 5: CLI And Documentation

**Files:**
- Modify: `mcp_unified/gateway/cli.py`
- Modify: `mcp_unified/README.md`
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

- [ ] **Step 1: Write failing CLI test for preset tooling summaries**

In `test_gateway_cli_package.py`, add or extend a CLI test:

```python
def test_list_presets_includes_tooling_summary() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mcp_unified.gateway.cli", "list-presets"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    product_owner = next(item for item in payload["presets"] if item["id"] == "product-owner")
    assert product_owner["tooling"]["direct_categories"]
    assert product_owner["tooling"]["deferred_categories"]
    assert product_owner["tooling"]["recommendation_catalog_patchable"] is True
```

- [ ] **Step 2: Run CLI test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -q
```

Expected: FAIL because `list-presets` only emits id/name/description/version.

- [ ] **Step 3: Add compact CLI summary**

Update `_handle_list_presets()` to include a compact tooling summary:

```python
def _preset_tooling_summary(preset: ProfilePreset) -> dict[str, Any]:
    tooling = preset.profile.metadata.get("tooling")
    if not isinstance(tooling, dict):
        return {}
    progressive = tooling.get("progressive_disclosure")
    if not isinstance(progressive, dict):
        progressive = {}
    return {
        "direct_categories": list(progressive.get("direct_categories") or []),
        "deferred_categories": list(progressive.get("deferred_categories") or []),
        "recommended_server_categories": [
            server.get("category")
            for server in tooling.get("recommended_servers", [])
            if isinstance(server, dict) and isinstance(server.get("category"), str)
        ],
        "recommendation_catalog_patchable": tooling.get("recommendation_catalog_patchable") is True,
    }
```

Keep `show-preset` unchanged so users can inspect the full metadata.

- [ ] **Step 4: Update package docs**

In `mcp_unified/USER_GUIDE.md`, add a section after "Work With Profiles":

- Explain direct tools vs recommended unavailable tools.
- Explain progressive disclosure bridge tools.
- Document that recommendation catalog patches do not grant execution authority.
- Document CDP first and the `ChromeDevTools/chrome-devtools-mcp` exact target.

In `mcp_unified/README.md`, add a short bullet under "What Is Included" for role presets and progressive disclosure.

- [ ] **Step 5: Run CLI/docs-adjacent tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mcp_unified/gateway/cli.py mcp_unified/README.md mcp_unified/USER_GUIDE.md tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
git commit -m "docs: document mcp profile tooling discovery"
```

---

### Task 6: Final Verification And PR Update

**Files:**
- Modify: existing branch/PR only

- [ ] **Step 1: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run package artifact gate**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  -c mcp_unified/pytest-artifact-gate.ini \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_distribution_metadata_matches_extras \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_sdist_contains_only_package_boundary \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_artifacts_include_typed_marker \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_artifacts_include_package_docs \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched package scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r mcp_unified/profiles mcp_unified/gateway -f json -o /tmp/bandit_mcp_profile_tooling.json
```

Expected: exit 0 or only known non-touched baseline issues. Fix any new findings in touched code before continuing.

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit 0.

- [ ] **Step 5: Update the draft PR**

Run:

```bash
git status --short
git push
gh pr view 2251 --repo rmusser01/tldw_server --json url,isDraft,state,headRefName,baseRefName
```

Expected: working tree clean, branch pushed, PR #2251 remains open and draft until active MCP/ACP reconciliation is complete.

- [ ] **Step 6: Final commit if verification notes changed**

Only commit additional documentation/task-record changes if needed:

```bash
git add <changed-files>
git commit -m "docs: record mcp profile tooling verification"
```

---

## Implementation Notes

- Use @superpowers:test-driven-development for each task that changes code.
- Use @superpowers:verification-before-completion before claiming each task is complete.
- Keep recommendation metadata non-authoritative. Runtime execution must still go through profile policy, external-server grants, credential grants, approval policy, and audit.
- Do not add semantic search dependencies in this slice.
- Do not enable shell, SSH, browser mutation, deploy mutation, CI mutation, arbitrary process execution, or memory writes by default.
- Keep package boundary clean: files under `mcp_unified/` must not import `tldw_Server_API`.
