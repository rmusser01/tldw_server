# Standalone MCP Docs Stage 3 Server Mounting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mount the docs corpus as a standalone-first MCP docs module with local SQLite state enabled by default, while keeping the built-in `tldw_server` MCP integration as a thin host adapter.

**Architecture:** Add a runtime-neutral standalone docs mount/factory in `mcp_unified.docs` that constructs a `DocsMCPToolProvider` from local defaults or explicit config profiles. Add a small `tldw_server` docs host adapter module that translates host `ModuleConfig` and request context into standalone docs settings and scopes, then update `DocsModule` to delegate through that adapter. This stage does not extract the whole MCP gateway, add crawler/sync behavior, add embeddings, or bridge Media/RAG stores.

**Tech Stack:** Python 3.10+, dataclasses, `pathlib`, existing `mcp_unified.docs` SQLite + FTS5 provider, existing `tldw_Server_API` MCP `BaseModule` shim, pytest, Black, Bandit.

---

## Source References

- Design spec: `Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-catalog-design.md`
- Standalone gateway design: `Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md`
- Stage 1 plan: `Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-corpus-stage1-plan.md`
- Stage 2 plan: `Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-url-acquisition-implementation-plan.md`
- Backlog planning task: `TASK-12079`
- Implementation work must have its own Backlog.md task before code edits begin, per repo policy.

## Baseline Findings

- `mcp_unified.docs` already contains a runtime-neutral `DocsMCPToolProvider`, `DocsSettings`, SQLite store, importers, retrieval, aliases, and optional URL acquisition service.
- `mcp_unified.docs` import-boundary tests already prove the package does not import `tldw_Server_API` or eager optional web dependencies.
- `tldw_Server_API.app.core.MCP_unified.modules.implementations.docs_module.DocsModule` is already a thin `BaseModule` shim, but its host settings/scope translation still lives inline instead of behind an explicit host adapter boundary.
- `tldw_Server_API/Config_Files/mcp_modules.yaml` already enables the docs module in the built-in MCP server with `enable_web_acquisition: false`.
- The existing host MCP module loader is constrained to `tldw_Server_API.app.core.MCP_unified.modules.implementations`, so Stage 3 should not require the host loader to import `mcp_unified.docs` classes directly.
- There is no standalone docs mount/factory under `mcp_unified.docs` that a future standalone MCP gateway can use to enable docs by default with local SQLite state.

## Non-Negotiable Constraints

- `mcp_unified.docs` must not import `tldw_Server_API`.
- The built-in `tldw_server` MCP server must mount docs through a host-owned shim/adapter.
- Standalone docs state defaults to local SQLite and FTS5; no Media DB, ChromaDB, RAG service, AuthNZ, Jobs, Scheduler, or tldw scraping runtime dependency.
- Web acquisition remains optional and disabled in locked-down defaults.
- Profile defaults must be explicit and downgradeable: `locked_down` hides URL ingestion, while `local_first` and `online_capable` can enable URL ingestion under the existing policy gates.
- `docs.ingest_url` remains hidden when web acquisition is disabled.
- Do not add crawler/sitemap sync, `docs.sync_source`, embedding/rerank adapters, browser extraction, or Media/RAG bridge behavior in this stage.
- Do not add new required dependencies to `pyproject.toml`.

## File Structure

Create:

- `mcp_unified/docs/standalone.py` - runtime-neutral standalone docs mount, profile defaults, and provider factory.
- `tldw_Server_API/app/core/MCP_unified/adapters/__init__.py` - host adapter package marker.
- `tldw_Server_API/app/core/MCP_unified/adapters/docs/__init__.py` - exports host docs adapter helpers.
- `tldw_Server_API/app/core/MCP_unified/adapters/docs/config.py` - converts host module settings and request context into `DocsSettings` and `AccessScope`.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_standalone_mount.py`
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_host_adapter.py`

Modify:

- `mcp_unified/docs/__init__.py` - export standalone mount types and factory helpers.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py` - delegate settings and scope creation to the host docs adapter.
- `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py` - add a focused built-in server registration/tool discovery test for docs without Media/RAG.
- `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py` - include the new standalone module in boundary checks automatically and add a targeted import test if useful.

Do not modify:

- `mcp_unified.docs.acquisition` behavior except where tests reveal a Stage 3 regression.
- Existing `tldw_Server_API` scraping services.
- Existing Media/RAG code.
- `pyproject.toml` unless packaging verification proves the new files are excluded from the existing `mcp_unified.*` package discovery.

## Test Command Conventions

Use the project virtual environment explicitly from this worktree:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/docs -q
```

If `.venv` is missing or cannot import project dependencies, stop and report that environment problem instead of switching to global Python.

---

### Task 1: Standalone Docs Mount Defaults And Profiles

**Files:**

- Create: `mcp_unified/docs/standalone.py`
- Modify: `mcp_unified/docs/__init__.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_standalone_mount.py`

- [ ] **Step 1: Add failing standalone mount tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_standalone_mount.py`:

```python
from __future__ import annotations

from pathlib import Path

from mcp_unified.docs import AccessScope
from mcp_unified.docs.standalone import (
    StandaloneDocsProfile,
    create_standalone_docs_mount,
    standalone_docs_settings_for_profile,
)


def test_standalone_mount_defaults_to_docs_with_local_sqlite(tmp_path: Path) -> None:
    mount = create_standalone_docs_mount(
        {"db_path": str(tmp_path / "docs.db"), "trusted_roots": [str(tmp_path)]}
    )

    names = {tool["name"] for tool in mount.tool_definitions()}
    status = mount.execute_tool("docs.status", {}, scope=AccessScope())

    assert mount.module_id == "docs"  # nosec B101
    assert mount.name == "Docs Corpus"  # nosec B101
    assert mount.settings.db_path == tmp_path / "docs.db"  # nosec B101
    assert "docs.search" in names  # nosec B101
    assert "docs.import_path" in names  # nosec B101
    assert "docs.ingest_url" not in names  # nosec B101
    assert status["web_acquisition_enabled"] is False  # nosec B101


def test_standalone_mount_can_import_search_and_context(tmp_path: Path) -> None:
    guide = tmp_path / "guide.md"
    guide.write_text("# Guide\n\nSQLite FTS5 context for local agents.\n", encoding="utf-8")
    mount = create_standalone_docs_mount(
        {"db_path": str(tmp_path / "docs.db"), "trusted_roots": [str(tmp_path)]}
    )
    scope = AccessScope(owner_scope="owner-a", profile_scope="profile-a")

    imported = mount.execute_tool("docs.import_path", {"path": str(guide)}, scope=scope)
    search = mount.execute_tool("docs.search", {"query": "FTS5"}, scope=scope)
    context = mount.execute_tool("docs.context", {"query": "local agents"}, scope=scope)

    assert imported["status"] in {"created", "updated"}  # nosec B101
    assert search["results"]  # nosec B101
    assert context["chunks"]  # nosec B101


def test_standalone_profile_defaults_are_downgradeable(tmp_path: Path) -> None:
    locked = standalone_docs_settings_for_profile(
        StandaloneDocsProfile.LOCKED_DOWN,
        overrides={"db_path": str(tmp_path / "locked.db")},
    )
    local = standalone_docs_settings_for_profile(
        StandaloneDocsProfile.LOCAL_FIRST,
        overrides={"db_path": str(tmp_path / "local.db")},
    )
    online = standalone_docs_settings_for_profile(
        StandaloneDocsProfile.ONLINE_CAPABLE,
        overrides={
            "db_path": str(tmp_path / "online.db"),
            "allowed_url_prefixes": ["https://example.com/docs/"],
        },
    )

    assert locked.web_source_profile == "locked_down"  # nosec B101
    assert locked.enable_web_acquisition is False  # nosec B101
    assert locked.allow_arbitrary_public_domains is False  # nosec B101
    assert local.web_source_profile == "local_first"  # nosec B101
    assert local.enable_web_acquisition is True  # nosec B101
    assert local.allow_arbitrary_public_domains is False  # nosec B101
    assert online.web_source_profile == "online_capable"  # nosec B101
    assert online.enable_web_acquisition is True  # nosec B101
    assert online.allowed_url_prefixes == ("https://example.com/docs/",)  # nosec B101


def test_standalone_mount_online_profile_advertises_url_ingest_when_enabled(tmp_path: Path) -> None:
    mount = create_standalone_docs_mount(
        profile=StandaloneDocsProfile.ONLINE_CAPABLE,
        settings={
            "db_path": str(tmp_path / "docs.db"),
            "allowed_url_prefixes": ["https://example.com/docs/"],
        },
    )

    names = {tool["name"] for tool in mount.tool_definitions()}

    assert "docs.ingest_url" in names  # nosec B101
```

- [ ] **Step 2: Run tests to verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_standalone_mount.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mcp_unified.docs.standalone'`.

- [ ] **Step 3: Implement standalone mount**

Create `mcp_unified/docs/standalone.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .mcp_module import DocsMCPToolProvider
from .models import AccessScope
from .settings import DocsSettings


class StandaloneDocsProfile(str, Enum):
    LOCKED_DOWN = "locked_down"
    LOCAL_FIRST = "local_first"
    ONLINE_CAPABLE = "online_capable"


@dataclass(frozen=True)
class StandaloneDocsMount:
    module_id: str
    name: str
    settings: DocsSettings
    provider: DocsMCPToolProvider

    def tool_definitions(self) -> list[dict[str, Any]]:
        return self.provider.tool_definitions()

    def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        *,
        scope: AccessScope | None = None,
    ) -> Any:
        return self.provider.execute(tool_name, arguments or {}, scope=scope or self.settings.default_scope)


def standalone_docs_settings_for_profile(
    profile: StandaloneDocsProfile | str = StandaloneDocsProfile.LOCKED_DOWN,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> DocsSettings:
    profile_value = StandaloneDocsProfile(profile)
    values: dict[str, Any] = {
        "db_path": "Databases/mcp_docs.db",
        "enable_web_acquisition": False,
        "web_source_profile": profile_value.value,
        "allow_arbitrary_public_domains": False,
    }
    if profile_value in {StandaloneDocsProfile.LOCAL_FIRST, StandaloneDocsProfile.ONLINE_CAPABLE}:
        values["enable_web_acquisition"] = True
    if overrides:
        values.update(dict(overrides))
    return DocsSettings.from_mapping(values)


def create_standalone_docs_mount(
    settings: DocsSettings | Mapping[str, Any] | None = None,
    *,
    profile: StandaloneDocsProfile | str = StandaloneDocsProfile.LOCKED_DOWN,
    module_id: str = "docs",
    name: str = "Docs Corpus",
) -> StandaloneDocsMount:
    if isinstance(settings, DocsSettings):
        resolved_settings = settings
    else:
        resolved_settings = standalone_docs_settings_for_profile(profile, overrides=settings)
    provider = DocsMCPToolProvider(settings=resolved_settings)
    return StandaloneDocsMount(module_id=module_id, name=name, settings=resolved_settings, provider=provider)
```

Update `mcp_unified/docs/__init__.py` exports:

```python
from .standalone import StandaloneDocsMount, StandaloneDocsProfile, create_standalone_docs_mount
from .standalone import standalone_docs_settings_for_profile
```

- [ ] **Step 4: Run tests to verify green state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_standalone_mount.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add mcp_unified/docs/standalone.py mcp_unified/docs/__init__.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_standalone_mount.py
git commit -m "feat: add standalone docs mount"
```

### Task 2: Explicit tldw_server Docs Host Adapter

**Files:**

- Create: `tldw_Server_API/app/core/MCP_unified/adapters/docs/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/docs/config.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_host_adapter.py`
- Test: `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py`

- [ ] **Step 1: Add failing host adapter tests**

Create `tldw_Server_API/tests/MCP_unified/docs/test_docs_host_adapter.py`:

```python
from __future__ import annotations

from pathlib import Path

from tldw_Server_API.app.core.MCP_unified.adapters.docs.config import (
    docs_scope_from_context,
    docs_settings_from_module_config,
)
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


def test_docs_settings_from_module_config_keeps_locked_down_defaults(tmp_path: Path) -> None:
    config = ModuleConfig(
        name="docs",
        settings={"db_path": str(tmp_path / "docs.db"), "trusted_roots": [str(tmp_path)]},
    )

    settings = docs_settings_from_module_config(config)

    assert settings.db_path == tmp_path / "docs.db"  # nosec B101
    assert settings.trusted_roots == (tmp_path.resolve(),)  # nosec B101
    assert settings.enable_web_acquisition is False  # nosec B101
    assert settings.web_source_profile == "locked_down"  # nosec B101


def test_docs_scope_from_request_context_maps_user_and_profile() -> None:
    context = RequestContext(
        request_id="docs-scope",
        user_id="user-1",
        client_id="unit",
        metadata={"profile_scope": "profile-1"},
    )

    scope = docs_scope_from_context(context)

    assert scope.owner_scope == "user-1"  # nosec B101
    assert scope.profile_scope == "profile-1"  # nosec B101


def test_docs_scope_from_missing_context_uses_public_scope() -> None:
    scope = docs_scope_from_context(None)

    assert scope.owner_scope is None  # nosec B101
    assert scope.profile_scope is None  # nosec B101
```

- [ ] **Step 2: Add shim delegation regression test**

Extend `tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py` with:

```python
def test_docs_module_uses_host_adapter_boundary() -> None:
    import inspect
    from tldw_Server_API.app.core.MCP_unified.modules.implementations import docs_module

    source = inspect.getsource(docs_module.DocsModule)

    assert "docs_settings_from_module_config" in source  # nosec B101
    assert "docs_scope_from_context" in source  # nosec B101
```

This is intentionally a boundary test, not a style test: it guards against moving host context parsing back into the runtime-neutral docs package or the shim body.

- [ ] **Step 3: Run tests to verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_host_adapter.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py::test_docs_module_uses_host_adapter_boundary \
  -q
```

Expected: FAIL because `tldw_Server_API.app.core.MCP_unified.adapters.docs` does not exist.

- [ ] **Step 4: Implement host adapter helpers**

Create `tldw_Server_API/app/core/MCP_unified/adapters/__init__.py`:

```python
"""Host adapter implementations for MCP Unified compatibility shims."""
```

Create `tldw_Server_API/app/core/MCP_unified/adapters/docs/config.py`:

```python
from __future__ import annotations

from typing import Any

from mcp_unified.docs import AccessScope, DocsSettings
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig


def docs_settings_from_module_config(config: ModuleConfig) -> DocsSettings:
    return DocsSettings.from_mapping(config.settings or {})


def docs_scope_from_context(context: Any | None) -> AccessScope:
    metadata = getattr(context, "metadata", None) if context is not None else None
    profile_scope = metadata.get("profile_scope") if isinstance(metadata, dict) else None
    user_id = getattr(context, "user_id", None) if context is not None else None
    return AccessScope(
        owner_scope=str(user_id) if user_id is not None else None,
        profile_scope=str(profile_scope) if profile_scope else None,
    )
```

Create `tldw_Server_API/app/core/MCP_unified/adapters/docs/__init__.py`:

```python
from .config import docs_scope_from_context, docs_settings_from_module_config

__all__ = ["docs_scope_from_context", "docs_settings_from_module_config"]
```

- [ ] **Step 5: Update DocsModule to delegate through adapter**

In `tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py`:

```python
from tldw_Server_API.app.core.MCP_unified.adapters.docs import (
    docs_scope_from_context,
    docs_settings_from_module_config,
)
```

Change `_ensure_provider`:

```python
settings = docs_settings_from_module_config(self.config)
```

Change execution scope:

```python
return self._ensure_provider().execute(tool_name, args, scope=docs_scope_from_context(context))
```

Remove the private `_scope_from_context` method if it is no longer used.

- [ ] **Step 6: Run tests to verify green state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_host_adapter.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add tldw_Server_API/app/core/MCP_unified/adapters \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_host_adapter.py \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_module_shim.py
git commit -m "refactor: add docs host adapter boundary"
```

### Task 3: Built-In MCP Server Registration Guard

**Files:**

- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py`

- [ ] **Step 1: Add failing or guarding server registration test**

Add this test to `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py`:

```python
@pytest.mark.asyncio
async def test_server_registers_docs_module_without_media_or_rag_dependencies(monkeypatch, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.registry import reset_module_registry
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    await reset_module_registry()
    config_path = tmp_path / "mcp_modules.yaml"
    config_path.write_text(
        f"""
modules:
  - id: docs
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.docs_module:DocsModule
    enabled: true
    name: Docs Corpus
    version: "0.1.0"
    department: knowledge
    settings:
      db_path: {str(tmp_path / "docs.db")}
      trusted_roots:
        - {str(tmp_path)}
      enable_web_acquisition: false
      web_source_profile: locked_down
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(config_path))
    monkeypatch.setenv("MCP_MODULES", "")
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")
    server = MCPServer()

    try:
        await server._register_default_modules()
        docs_module = await server.module_registry.find_module_for_tool("docs.status")
        ingest_module = await server.module_registry.find_module_for_tool("docs.ingest_url")

        assert docs_module is not None  # nosec B101
        assert ingest_module is None  # nosec B101
        status = await docs_module.execute_tool("docs.status", {}, context=None)
        assert status["web_acquisition_enabled"] is False  # nosec B101
    finally:
        await server.module_registry.shutdown_all()
        await reset_module_registry()
```

- [ ] **Step 2: Run test to verify current behavior**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py::test_server_registers_docs_module_without_media_or_rag_dependencies \
  -q
```

Expected: PASS if Stage 1/2 registration already covers this path. If it fails because of leaked global registry state, fix the test setup with `reset_module_registry()` before changing production code. If it fails because docs initialization imports Media/RAG or advertises `docs.ingest_url` while disabled, fix that production regression.

- [ ] **Step 3: Run adjacent dynamic catalog tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit Task 3**

```bash
git add tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py
git commit -m "test: guard docs module server registration"
```

### Task 4: Boundary And Packaging Regression Checks

**Files:**

- Modify: `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py`
- Optionally modify: `pyproject.toml` only if the existing package discovery or package data misses the new standalone module.

- [ ] **Step 1: Add targeted standalone export/import boundary tests**

Extend `tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py`:

```python
def test_standalone_mount_imports_without_host_dependencies() -> None:
    module = importlib.import_module("mcp_unified.docs.standalone")

    assert hasattr(module, "create_standalone_docs_mount")  # nosec B101
    assert hasattr(module, "StandaloneDocsProfile")  # nosec B101


def test_docs_public_exports_include_standalone_mount() -> None:
    module = importlib.import_module("mcp_unified.docs")

    assert hasattr(module, "create_standalone_docs_mount")  # nosec B101
    assert hasattr(module, "StandaloneDocsProfile")  # nosec B101
```

The existing AST scan already covers `mcp_unified/docs/**/*.py`, so it will fail if `standalone.py` imports `tldw_Server_API`.

- [ ] **Step 2: Run boundary tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py -q
```

Expected: PASS after Task 1 exports are in place.

- [ ] **Step 3: Verify package discovery still includes new files**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python - <<'PY'
from setuptools import find_packages

packages = set(find_packages(where=".", include=["tldw_Server_API", "tldw_Server_API.*", "mcp_unified", "mcp_unified.*"]))
required = {
    "mcp_unified",
    "mcp_unified.docs",
    "tldw_Server_API.app.core.MCP_unified.adapters",
    "tldw_Server_API.app.core.MCP_unified.adapters.docs",
}
missing = sorted(required - packages)
print("missing=", missing)
raise SystemExit(1 if missing else 0)
PY
```

Expected: `missing= []`.

- [ ] **Step 4: Commit Task 4**

```bash
git add tldw_Server_API/tests/MCP_unified/docs/test_docs_import_boundaries.py pyproject.toml
git commit -m "test: enforce docs standalone mount boundaries"
```

If `pyproject.toml` was not modified, omit it from `git add`.

### Task 5: Focused Verification And Backlog Finalization

**Files:**

- Update: Backlog.md task created for Stage 3 implementation.
- Update: implementation plan task `TASK-12079` only if this planning task is still open.

- [ ] **Step 1: Run docs test suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified/docs -q --tb=short
```

Expected: PASS.

- [ ] **Step 2: Run focused MCP regression tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/MCP_unified \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  -k "docs or write_tools or validator or dynamic_module_catalog" \
  -q --tb=short
```

Expected: PASS.

- [ ] **Step 3: Run standalone import smoke**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python - <<'PY'
import sys

from mcp_unified.docs import StandaloneDocsProfile, create_standalone_docs_mount

mount = create_standalone_docs_mount(profile=StandaloneDocsProfile.LOCKED_DOWN)
names = {tool["name"] for tool in mount.tool_definitions()}
loaded_optional = sorted({"trafilatura", "bs4", "playwright", "requests", "httpx", "aiohttp"} & set(sys.modules))
print("docs.search" in names, "docs.ingest_url" in names, "loaded_optional=", loaded_optional)
raise SystemExit(1 if "docs.search" not in names or "docs.ingest_url" in names or loaded_optional else 0)
PY
```

Expected: `True False loaded_optional= []`.

- [ ] **Step 4: Run Black check on touched Python files**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m black --check \
  mcp_unified/docs/standalone.py \
  mcp_unified/docs/__init__.py \
  tldw_Server_API/app/core/MCP_unified/adapters/docs \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py \
  tldw_Server_API/tests/MCP_unified/docs \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py
```

Expected: PASS with files unchanged or already formatted.

- [ ] **Step 5: Run Bandit on touched Python scope**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit \
  -r mcp_unified/docs/standalone.py \
     tldw_Server_API/app/core/MCP_unified/adapters/docs \
     tldw_Server_API/app/core/MCP_unified/modules/implementations/docs_module.py \
  -f json -o /tmp/bandit_mcp_docs_stage3_server_mounting.json
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/bandit_mcp_docs_stage3_server_mounting.json").read_text())
print("errors:", payload.get("errors"))
print("results:", payload.get("results"))
raise SystemExit(1 if payload.get("errors") or payload.get("results") else 0)
PY
```

Expected: no errors and no findings.

- [ ] **Step 6: Review diff**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files remain unstaged or the tree is clean after final commit.

- [ ] **Step 7: Final commit if any final task metadata changed**

```bash
git add backlog/tasks Docs/superpowers/plans Docs/superpowers/specs
git commit -m "chore: close docs server mounting task"
```

## Review Notes

- The narrowest useful Stage 3 implementation is a standalone docs mount/factory, not a full standalone MCP gateway extraction. The full gateway remains governed by `Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md`.
- If implementation discovers a reusable runtime module registry already exists under `mcp_unified`, prefer adapting this plan to mount the docs provider there instead of creating a parallel registry. The current baseline investigation found only the docs package under `mcp_unified`.
- The profile defaults in Task 1 intentionally make `local_first` and `online_capable` web-capable but still policy-bound; they do not allow arbitrary public internet fetches unless config explicitly allows that.
- The tldw host adapter in Task 2 is intentionally small. Do not add Media/RAG bridging there in this stage.
