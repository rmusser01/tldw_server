# MCP Unified Profile Registry Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add package-local profile store and resolver primitives for standalone MCP profile resolution without changing current `tldw_server` MCP route or execution behavior.

**Architecture:** Keep this slice inside `mcp_unified.profiles` plus the package storage protocol. The store is an in-memory implementation for tests and future standalone bootstrap, while `StoreBackedProfileResolver` handles explicit profile id lookup, optional standalone default profile lookup, disabled-profile fail-closed behavior, and copy isolation. No SQLite, FastAPI route, MCP protocol dispatch, host adapter, approval, credential, or execution-policy wiring changes are included.

**Tech Stack:** Python 3.11, Pydantic v2 models, pytest, Ruff, Mypy, Bandit.

---

## Scope

In scope:
- Package-local profile store primitives.
- Store-backed profile resolver primitives.
- Package exports and storage protocol alignment.
- Focused tests under existing MCP Unified tests.
- Backlog task update for TASK-521.

Out of scope:
- SQLite persistence and migrations.
- User/profile assignment APIs.
- FastAPI route changes.
- Runtime `MCPProtocol` or `MCPServer` enforcement changes.
- Built-in preset mutation or automatic preset-as-runtime-profile behavior.
- External MCP registry lifecycle or gateway transport work.

## Files

- Create: `mcp_unified/profiles/store.py`
  - In-memory `ProfileStore` implementation that stores `MCPProfile` documents by id and returns deep copies.
- Modify: `mcp_unified/profiles/resolver.py`
  - Add `StoreBackedProfileResolver` alongside the existing `ProfileResolver` protocol.
- Modify: `mcp_unified/profiles/__init__.py`
  - Export the new store and resolver primitives.
- Modify: `mcp_unified/interfaces/storage.py`
  - Update `ProfileStore` protocol to describe package profile-store behavior.
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py`
  - Add package-boundary, store, resolver, fail-closed, and copy-isolation tests.
- Modify: `backlog/tasks/task-521 - Implement-MCP-Unified-Stage-2-profile-registry-resolver-primitives.md`
  - Record implementation notes, verification, final summary, and DoD.

## Task 1: RED Tests For Store And Resolver Contract

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py`

- [x] **Step 1: Write failing tests**

Add tests covering:

```python
def test_profile_registry_resolver_module_has_no_tldw_server_imports() -> None:
    package_root = Path(profile_store.__file__).resolve().parent
    ...


@pytest.mark.asyncio
async def test_in_memory_profile_store_returns_copy_isolated_profiles() -> None:
    store = InMemoryProfileStore()
    profile = MCPProfile(id="architect-workspace", name="Architect Workspace")
    await store.upsert_profile(profile)

    first = await store.get_profile("architect-workspace")
    assert first is not None
    first.name = "Mutated"

    second = await store.get_profile("architect-workspace")
    assert second is not None
    assert second.name == "Architect Workspace"


@pytest.mark.asyncio
async def test_store_backed_resolver_resolves_explicit_enabled_profile() -> None:
    store = InMemoryProfileStore()
    await store.upsert_profile(MCPProfile(id="code-reviewer", name="Code Reviewer"))
    resolver = StoreBackedProfileResolver(store)

    profile = await resolver.resolve_profile("code-reviewer")

    assert profile is not None
    assert profile.id == "code-reviewer"


@pytest.mark.asyncio
async def test_store_backed_resolver_uses_default_only_without_explicit_id() -> None:
    store = InMemoryProfileStore()
    await store.upsert_profile(MCPProfile(id="default", name="Default"))
    resolver = StoreBackedProfileResolver(store, default_profile_id="default")

    assert (await resolver.resolve_profile(None)).id == "default"
    assert await resolver.resolve_profile("missing") is None


@pytest.mark.asyncio
async def test_store_backed_resolver_returns_none_for_disabled_profiles() -> None:
    store = InMemoryProfileStore()
    await store.upsert_profile(MCPProfile(id="disabled", name="Disabled", enabled=False))
    resolver = StoreBackedProfileResolver(store, default_profile_id="disabled")

    assert await resolver.resolve_profile("disabled") is None
    assert await resolver.resolve_profile(None) is None
```

- [x] **Step 2: Run tests to verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py -v
```

Expected: FAIL because `mcp_unified.profiles.store` and `StoreBackedProfileResolver` do not exist yet.

## Task 2: Implement Package-Local Store And Resolver

**Files:**
- Create: `mcp_unified/profiles/store.py`
- Modify: `mcp_unified/profiles/resolver.py`
- Modify: `mcp_unified/interfaces/storage.py`
- Modify: `mcp_unified/profiles/__init__.py`

- [x] **Step 1: Add store implementation**

Implement a minimal in-memory store:

```python
class InMemoryProfileStore:
    """In-memory profile store for tests and standalone bootstrap."""

    def __init__(self, profiles: Iterable[MCPProfile | Mapping[str, Any]] | None = None) -> None:
        self._profiles: dict[str, MCPProfile] = {}
        ...

    async def get_profile(self, profile_id: str) -> MCPProfile | None:
        ...

    async def list_profiles(self) -> list[MCPProfile]:
        ...

    async def upsert_profile(self, profile: MCPProfile | Mapping[str, Any]) -> MCPProfile:
        ...

    async def delete_profile(self, profile_id: str) -> bool:
        ...
```

Use `MCPProfile.model_validate()` for mappings and `model_copy(deep=True)` on writes and reads.

- [x] **Step 2: Add store-backed resolver**

Implement:

```python
class StoreBackedProfileResolver:
    """Resolve profiles from a profile store with optional standalone default."""

    def __init__(self, profile_store: ProfileStore, *, default_profile_id: str | None = None) -> None:
        ...

    async def resolve_profile(
        self,
        profile_id: str | None,
        *,
        user_id: str | None = None,
    ) -> MCPProfile | None:
        resolved_id = profile_id or self.default_profile_id
        if resolved_id is None:
            return None
        profile = await self.profile_store.get_profile(resolved_id)
        if profile is None or not profile.enabled:
            return None
        return profile.model_copy(deep=True)
```

The `user_id` parameter remains accepted for protocol compatibility but is not used in this package-local primitive.

- [x] **Step 3: Update package exports and protocol**

Export `InMemoryProfileStore` and `StoreBackedProfileResolver` from `mcp_unified.profiles`.

Update `mcp_unified.interfaces.storage.ProfileStore` with package-local methods returning `MCPProfile` objects:

```python
async def get_profile(self, profile_id: str) -> MCPProfile | None: ...
async def list_profiles(self) -> list[MCPProfile]: ...
async def upsert_profile(self, profile: MCPProfile) -> MCPProfile: ...
async def delete_profile(self, profile_id: str) -> bool: ...
```

- [x] **Step 4: Run focused tests to verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py -v
```

Expected: PASS.

## Task 3: Regression, Quality Gates, And Task Closeout

**Files:**
- Modify: `backlog/tasks/task-521 - Implement-MCP-Unified-Stage-2-profile-registry-resolver-primitives.md`

- [x] **Step 1: Run focused regression tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -v
```

Expected: PASS.

Result: PASS, `20 passed, 3 warnings in 0.10s`.

- [x] **Step 2: Run static checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check mcp_unified tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m mypy mcp_unified --config-file pyproject.toml
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified -f json -o /tmp/bandit_mcp_unified_profile_registry.json
jq '.metrics._totals, (.results | length)' /tmp/bandit_mcp_unified_profile_registry.json
git diff --check
```

Expected: Ruff passes, Mypy passes, Bandit reports 0 findings, diff whitespace clean.

Result:
- Ruff PASS for `mcp_unified` and the new profile registry test.
- Mypy PASS for `mcp_unified/profiles`, `mcp_unified/interfaces/storage.py`, and the new profile registry test.
- Runtime Bandit PASS with 0 findings for `mcp_unified/profiles` and `mcp_unified/interfaces/storage.py`.
- Full touched-scope Bandit produced only pytest `assert` B101 findings in the new test file.
- `git diff --check` PASS.

- [x] **Step 3: Update Backlog task**

Record:
- plan path
- touched files
- RED/GREEN verification
- final quality gates
- known skips or blockers

Result: TASK-521 updated with implementation notes, final summary, completed acceptance criteria, and Definition of Done.

- [x] **Step 4: Commit**

Run:

```bash
git add \
  mcp_unified/interfaces/storage.py \
  mcp_unified/profiles/__init__.py \
  mcp_unified/profiles/resolver.py \
  mcp_unified/profiles/store.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py \
  Docs/superpowers/plans/2026-05-27-mcp-unified-profile-registry-resolver-implementation-plan.md \
  "backlog/tasks/task-521 - Implement-MCP-Unified-Stage-2-profile-registry-resolver-primitives.md"
git commit -m "feat: add mcp profile registry resolver primitives"
```

Expected: Commit succeeds.

Result: Commit `b9f1100a3` created with message `feat: add mcp profile registry resolver primitives`.
