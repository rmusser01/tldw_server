# MCP Filesystem Policy Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden MCP filesystem path/action enforcement and move the virtual CLI toward structured filesystem primitives without breaking legacy compatibility.

**Architecture:** Keep the existing filesystem module and path-enforcement service as the policy boundary. First repair the adapter seam so module-derived path candidates reach the real service path, then add service/protocol regression tests for action-aware `path_grants`, then update the virtual CLI registry/adapters to prefer `fs.read` and expose explicit structured create semantics.

**Tech Stack:** Python, pytest, FastAPI-side MCP Unified module, `mcp_unified` shared interfaces, Loguru, Bandit.

---

## File Structure

- Modify `tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py`
  - Accept and forward optional `path_scope_candidates` in `TldwPathScopeEnforcer.evaluate_tool_call`.
- Modify `tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py`
  - Add service/protocol tests for action-aware `path_grants`, deny-overrides, and bundle fail-closed behavior.
- Modify `tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py`
  - Update `cat` backing tools to include `fs.read` and legacy fallback `fs.read_text`.
  - Add `write-create` backed by `fs.write`.
  - Preserve only executable backing tools on descriptors returned by
    `visible_commands`, so adapters can choose the preferred visible tool.
- Modify `tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py`
  - Route `cat` to `fs.read` when visible, otherwise legacy `fs.read_text`.
  - Add `write-create <path> <content>` -> `fs.write` with `mode: "create"`.
- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py`
  - Treat `fs.write` as write-capable for `run`/`bash`/`shell` classification.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py`
  - Update registry visibility and backend mapping expectations.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py`
  - Update `cat` assertions and add `write-create` coverage.
- Modify `backlog/tasks/task-2331 - Harden-MCP-filesystem-path-policy-and-command-runtime-defaults.md`
  - Keep plan, verification, and final summary current.

## Task 1: Forward Module-Derived Path Candidates Through The Default Adapter

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py`

- [ ] **Step 1: Write the failing adapter-forwarding test**

Add a test that instantiates the real `TldwPathScopeEnforcer`, monkeypatches
`get_mcp_hub_path_enforcement_service`, and verifies `PathScopeCandidate` values
arrive at the fake service.

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_tldw_path_scope_enforcer_forwards_module_candidates(monkeypatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_policy import TldwPathScopeEnforcer
    from tldw_Server_API.app.services import mcp_hub_path_enforcement_service as path_service_mod

    class _Service:
        def __init__(self) -> None:
            self.calls = []

        async def evaluate_tool_call(self, **kwargs):
            self.calls.append(kwargs)
            return {"enabled": True, "within_scope": True, "reason": None, "force_approval": False}

    service = _Service()

    async def _fake_service():
        return service

    monkeypatch.setattr(path_service_mod, "get_mcp_hub_path_enforcement_service", _fake_service)
    candidates = [PathScopeCandidate(path="docs/a.txt", action="edit", source="test")]

    result = await TldwPathScopeEnforcer().evaluate_tool_call(
        effective_policy={"enabled": True},
        context=_context(),
        tool_name="fs.patch",
        tool_args={"diff": "..."},
        tool_def={"metadata": {"uses_filesystem": True}},
        path_scope_candidates=candidates,
    )

    assert result["within_scope"] is True
    assert service.calls[0]["path_scope_candidates"] == candidates
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py::test_tldw_path_scope_enforcer_forwards_module_candidates -q
```

Expected: `TypeError` because `TldwPathScopeEnforcer.evaluate_tool_call` does
not accept `path_scope_candidates`.

- [ ] **Step 3: Implement the adapter seam**

Update the method signature and forwarding call:

```python
    async def evaluate_tool_call(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        context: Any,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
        path_scope_candidates: list[Any] | None = None,
    ) -> dict[str, Any]:
        ...
        return await service.evaluate_tool_call(
            effective_policy=effective_policy,
            context=context,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_def=tool_def,
            path_scope_candidates=path_scope_candidates,
        )
```

- [ ] **Step 4: Run the focused adapter test**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit the seam**

```bash
git add tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py
git commit -m "fix: forward MCP path scope candidates"
```

## Task 2: Add Action-Aware Path Grant Regression Tests

**Files:**
- Modify: `tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py`

- [ ] **Step 1: Add minimal service test helpers**

Reuse the existing file and add small helpers near the existing fake classes:

```python
def _filesystem_tool_def(name: str, action: str) -> dict:
    return {
        "name": name,
        "description": name,
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "metadata": {
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
            "path_scope_action": action,
        },
    }


def _effective_path_policy(path_grants: list[dict]) -> dict:
    return {
        "enabled": True,
        "allowed_tools": ["fs.read", "fs.edit", "fs.write", "fs.patch"],
        "policy_document": {
            "path_scope_mode": "workspace_root",
            "path_grants": path_grants,
        },
    }
```

- [ ] **Step 2: Write failing read/edit/write action tests**

Add tests that instantiate `McpHubPathEnforcementService` with
`McpHubPathScopeService` and a `_FakeWorkspaceRootResolver` returning a temp
workspace root.

Coverage:

```python
@pytest.mark.asyncio
async def test_path_grants_keep_read_edit_and_write_distinct(tmp_path) -> None:
    # read grant allows fs.read
    # same grant denies fs.edit and fs.write with path_action_not_granted
    # edit grant allows fs.edit but denies fs.write
    # write grant allows fs.write but denies fs.read/edit
```

Assert on:

```python
assert result["within_scope"] is False
assert result["reason"] == "path_action_not_granted"
assert result["scope_payload"]["path_decisions"][0]["requested_action"] == "write"
assert result["scope_payload"]["path_decisions"][0]["redacted"] is True
```

- [ ] **Step 3: Write failing deny-override test**

Add a test where grants allow `read/edit/write` on `docs` and deny `edit/write`
on `docs/private`. Assert that `docs/private/a.txt` is denied for `fs.edit` and
the path decision reports `path_action_denied` plus matched prefix
`docs/private`.

- [ ] **Step 4: Write failing patch bundle candidate test**

Use direct service evaluation with `path_scope_candidates`:

```python
path_scope_candidates=[
    PathScopeCandidate(path="docs/allowed.txt", action="edit", source="filesystem_diff"),
    PathScopeCandidate(path="docs/new.txt", action="write", source="filesystem_diff", creates_file=True),
]
```

With only an `edit` grant on `docs`, assert the bundle is denied because the
create candidate requires `write`.

- [ ] **Step 5: Run the new tests and verify current behavior**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py \
  -k "path_grants_keep_read_edit_and_write_distinct or deny_override or patch_bundle" -q
```

Expected: these may already pass because the service implements the behavior. If
they pass before implementation, keep them as regression coverage and document
that no service code change was needed.

- [ ] **Step 6: Implement only if a test exposes a real service bug**

If a test fails, make the smallest change in:

```text
tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py
```

Do not change the policy model. Expected fixes should be limited to preserving
candidate action alignment, denial reason payloads, or redaction fields.

- [ ] **Step 7: Commit the regression tests**

```bash
git add tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py \
  tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py
git commit -m "test: cover MCP filesystem path grant actions"
```

Only include the service file if it actually changed.

## Task 3: Prefer Structured Filesystem Primitives In The Virtual CLI

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py`

- [ ] **Step 1: Update registry tests first**

Change expected mappings:

```python
assert registry.get_command("cat").backend_tools == ("fs.read", "fs.read_text")
assert registry.get_command("write").backend_tools == ("fs.write_text",)
assert registry.get_command("write-create").backend_tools == ("fs.write",)
```

Add visibility cases:

```python
visible = registry.visible_commands(allowed_tools={"fs.read"})
assert "cat" in visible

visible = registry.visible_commands(allowed_tools={"fs.read_text"})
assert "cat" in visible

visible = registry.visible_commands(allowed_tools={"fs.write"})
assert "write-create" in visible
assert "write" not in visible
```

- [ ] **Step 2: Run registry tests and verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py -q
```

Expected: FAIL until registry mappings are updated.

- [ ] **Step 3: Update the command registry**

In `_DEFAULT_COMMANDS`, update/add descriptors:

```python
CommandDescriptor(
    name="cat",
    summary="Read a UTF-8 text file from the current workspace scope.",
    backend_tools=("fs.read", "fs.read_text"),
),
CommandDescriptor(
    name="write-create",
    summary="Create a UTF-8 text file in the current workspace scope.",
    backend_tools=("fs.write",),
),
```

Leave `write` backed only by `fs.write_text`.

Then update `CommandRegistry.visible_commands` so non-pure descriptors returned
to the adapter retain only currently allowed backing tools:

```python
from dataclasses import dataclass, field, replace

...

def visible_commands(self, allowed_tools: set[str]) -> dict[str, CommandDescriptor]:
    visible: dict[str, CommandDescriptor] = {}
    for name, descriptor in self._commands.items():
        if descriptor.pure_transform:
            visible[name] = descriptor
            continue
        visible_backend_tools = tuple(tool for tool in descriptor.backend_tools if tool in allowed_tools)
        if visible_backend_tools:
            visible[name] = replace(descriptor, backend_tools=visible_backend_tools)
    return visible
```

This keeps `cat` visible for either `fs.read` or `fs.read_text`, while allowing
the adapter to prefer `fs.read` when both are visible.

- [ ] **Step 4: Update run-module protocol stub and cat tests**

In `test_run_command_module.py`, make `_ProtocolStub._handle_tools_list`
include `fs.read` and `fs.write`.

Add `fs.read` execution payload:

```python
if tool_name == "fs.read":
    return {
        "content": [{"type": "json", "json": {"path": "notes.txt", "content": self.read_text_content}}],
        "tool": tool_name,
    }
```

Update cat assertions from `fs.read_text` to `fs.read` where the stub exposes
`fs.read`. Add a dedicated legacy fallback test with a stub that exposes
`fs.read_text` but not `fs.read`.

- [ ] **Step 5: Add write-create run tests**

Add tests for:

```python
rendered = await module.execute_tool("run", {"command": "write-create notes.txt hello"}, context=context)
assert protocol.prepare_calls[0].params["name"] == "fs.write"
assert protocol.prepare_calls[0].params["arguments"] == {
    "path": "notes.txt",
    "content": "hello",
    "mode": "create",
}
```

Add a visibility/help test proving `write-create` appears when `fs.write` is
visible and legacy `write` appears only when `fs.write_text` is visible.

- [ ] **Step 6: Run command-runtime tests and verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py \
  -q
```

Expected: FAIL until adapters and write classification are updated.

- [ ] **Step 7: Implement adapter routing**

In `PhaseOneCommandAdapters._governed_plan`:

```python
if command == "cat":
    if len(argv) != 2:
        return _UsageError("usage: cat <path>")
    tool_name = "fs.read" if "fs.read" in self.context.visible_commands["cat"].backend_tools else "fs.read_text"
    # Better: choose based on currently visible allowed tools, not descriptor tuple alone.
```

Use the actual visible backing set. Since Task 3 Step 3 filters descriptor
backing tools to visible tools, add a small helper:

```python
def _visible_backend_tool(self, command: str, preferred: tuple[str, ...]) -> str | None:
    descriptor = self.context.visible_commands.get(command)
    if descriptor is None:
        return None
    available = set(descriptor.backend_tools)
    for tool_name in preferred:
        if tool_name in available:
            return tool_name
    return None
```

Do not infer from policy separately inside adapters; the filtered descriptor is
the adapter's source of truth.

Add:

```python
if command == "write-create":
    if len(argv) < 3:
        return _UsageError("usage: write-create <path> <content>")
    return _GovernedCallPlan(
        tool_name="fs.write",
        arguments={"path": argv[1], "content": " ".join(argv[2:]), "mode": "create"},
        renderer=self._render_write,
    )
```

- [ ] **Step 8: Update write-call classification**

In `run_command_module.py`, update:

```python
_RUN_WRITE_BACKEND_TOOLS = {"fs.write", "fs.write_text", "sandbox.run"}
```

Add or update tests proving `run.is_write_tool_call("run", {"command": "write-create notes.txt hi"})` returns `True`.

- [ ] **Step 9: Run focused command tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py \
  -q
```

Expected: PASS.

- [ ] **Step 10: Commit runtime changes**

```bash
git add tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py \
  tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py
git commit -m "feat: prefer structured MCP filesystem commands"
```

## Task 4: Validate, Document Skips, And Close Backlog

**Files:**
- Modify: `backlog/tasks/task-2331 - Harden-MCP-filesystem-path-policy-and-command-runtime-defaults.md`

- [ ] **Step 1: Run all touched focused tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_path_scope_candidates.py \
  tldw_Server_API/tests/MCP_unified/test_mcp_protocol_path_scope.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_command_runtime_registry.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_run_command_module.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run Bandit on touched implementation files**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/adapters/tldw_policy.py \
  tldw_Server_API/app/core/MCP_unified/command_runtime/registry.py \
  tldw_Server_API/app/core/MCP_unified/command_runtime/adapters.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/run_command_module.py \
  tldw_Server_API/app/services/mcp_hub_path_enforcement_service.py \
  -f json -o /tmp/bandit_mcp_fs_policy_hardening.json
```

Expected: no new findings in touched code. If `mcp_hub_path_enforcement_service.py`
was not modified, it may be removed from the Bandit command.

- [ ] **Step 3: Run diff hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intentional files changed before final commit.

- [ ] **Step 4: Update Backlog task**

Record:

- tests run and pass/fail status
- Bandit output path and summary
- any skipped implementation items, especially if `write-replace` is deferred
- final summary of behavior changes

- [ ] **Step 5: Commit final task updates**

```bash
git add 'backlog/tasks/task-2331 - Harden-MCP-filesystem-path-policy-and-command-runtime-defaults.md'
git commit -m "chore: close MCP filesystem policy hardening task"
```

- [ ] **Step 6: Prepare PR summary**

Include:

- Change summary with both what changed and why.
- Test commands and outcomes.
- Explicit note that legacy `write` remains backed by `fs.write_text`.
- Explicit note whether `write-replace` was implemented or deferred because of
  preimage authorization syntax.
