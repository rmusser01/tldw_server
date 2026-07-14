# Skills Render Catalog Matches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add advisory `catalog_matches` to MCP `skills.render` while preserving dry rendering, current response fields, failure boundaries, and authoritative call-time policy.

**Architecture:** Keep the change in `SkillsModule`. Replace the broad executor call with direct argument substitution, then compare normalized declarations with one `MCPProtocol._handle_tools_list()` result after the Skills database closes. Return a nullable best-effort list without a new service, schema, cache, or policy layer.

**Tech Stack:** Python 3.10+, asyncio, MCP Unified, pytest, Ruff, Bandit.

**Design:** `Docs/Design/2026-07-13-skills-render-tool-resolution-design.md`

**Backlog:** `TASK-2294.2`

## Global Constraints

- Work only in `.worktrees/skills-render-tool-binding` on `codex/skills-render-tool-binding`.
- Use TDD: run each focused test red before writing production code, then run it green.
- Preserve all existing successful render fields and the configured 100,000-character hard ceiling.
- `catalog_matches` is always present after successful rendering: list for computed results, `null` for whole lookup or malformed-envelope failure.
- A list is best effort because `_handle_tools_list()` suppresses individual module failures.
- Match exact, case-sensitive base names only when `canExecute is True`.
- Run at most one catalog lookup, only for non-empty valid declarations, and only after database close.
- Propagate cancellation even if the catalog handler suppresses a per-module `CancelledError`.
- Do not modify REST, frontend, browser-extension, gateway, profile, protocol, parser, executor, database, or response-model production files.
- Do not add limits, caches, persistence, route models, tokens, flags, dependencies, or generic abstractions.
- No model, tool, workflow, job, script, or subagent execution is permitted.
- Run Bandit on the touched production Python file.
- Before each stage, mark only that stage `In Progress`; after its green test and commit, mark it `Complete` before starting the next stage.

## File Map

- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py`: direct render, normalization, matching, failure handling, and cancellation propagation.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py`: contract, matching, lifecycle, privacy, and cancellation coverage.
- Modify `Docs/MCP/Unified/Modules.md`: operator-facing field semantics.
- Update TASK-2294.2 only through the Backlog CLI.

---

## Stage 1: Direct Dry Rendering

**Goal**: Remove `SkillExecutor.execute()` from the render path without changing valid output.

**Success Criteria**: Existing field values and output limits remain equivalent; parsed non-string declarations do not crash; execution branches cannot be reached.

**Tests**: `test_skills_module.py` render tests.

**Status**: Not Started

### Task 1.1: Lock equivalence and replace the executor call

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py:12-28,486-670`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py:276-314`

**Interfaces:**
- Consumes: `SkillExecutor.substitute_arguments(content: str, arguments: str) -> str`.
- Produces: the existing eight-field render dictionary with normalized `declared_tools`.

- [ ] **Step 1: Write failing direct-render tests**

Import `Mock`, remove the now-unused `SkillExecutionResult` import, and change the existing render test to install:

```python
execute = AsyncMock(side_effect=AssertionError("render must not call execute"))
substitute = Mock(wraps=module._executor.substitute_arguments)
monkeypatch.setattr(module._executor, "execute", execute)
monkeypatch.setattr(module._executor, "substitute_arguments", substitute)
```

Keep its exact existing response assertion, then assert:

```python
substitute.assert_called_once_with(
    "Review $ARGUMENTS",
    "--formal /* literal */\nnext",
)
execute.assert_not_awaited()
```

Add:

```python
@pytest.mark.asyncio
async def test_render_ignores_non_string_and_blank_parsed_declarations(
    user_catalogs: dict[int, UserCatalog],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_review_skill(user_catalogs[1].service)
    original_get_skill = SkillsService.get_skill

    async def legacy_get_skill(
        self: SkillsService,
        name: str,
        *,
        enforce_integrity: bool = True,
    ) -> dict[str, Any]:
        skill = await original_get_skill(self, name, enforce_integrity=enforce_integrity)
        skill["allowed_tools"] = [" rag.search ", 7, None, " "]
        return skill

    monkeypatch.setattr(SkillsService, "get_skill", legacy_get_skill)
    module = await _module()
    result = await module.execute_tool(
        "skills.render",
        {"skill_name": "review-paper"},
        context=user_catalogs[1].context,
    )
    assert result["declared_tools"] == ["rag.search"]
```

- [ ] **Step 2: Verify red**

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py -k 'render_is_forced_dry_and_preserves_arguments or render_ignores_non_string'
```

Expected: the first test reaches `execute()`; the second raises on non-string `.strip()`.

- [ ] **Step 3: Implement direct rendering**

Replace the executor block in `_render_skill()` with:

```python
        raw_declared_tools = skill_data.get("allowed_tools")
        declared_tools = (
            [
                item.strip()
                for item in raw_declared_tools
                if isinstance(item, str) and item.strip()
            ]
            if isinstance(raw_declared_tools, list)
            else []
        )
        rendered_prompt = self._executor.substitute_arguments(
            skill_data.get("content", ""),
            args.get("arguments", ""),
        )
        if len(rendered_prompt) > self._max_rendered_skill_chars:
            raise SkillsMCPRenderedTooLargeError(
                f"rendered_skill_too_large: limit={self._max_rendered_skill_chars}"
            )
        execution_context = skill_data.get("context", "inline")
        return {
            "skill_name": skill_data.get("name", "unknown"),
            "rendered_prompt": rendered_prompt,
            "declared_tools": declared_tools,
            "model_override": skill_data.get("model"),
            "execution_mode": "fork" if execution_context == "fork" else "inline",
            "supporting_files_omitted": bool(skill_data.get("supporting_files")),
            "dry_run": True,
            "version": skill_data.get("version"),
        }
```

Update `test_render_rechecks_visibility_after_verified_load` to patch and assert against `substitute_arguments` rather than obsolete `execute()` behavior.

- [ ] **Step 4: Verify green and commit**

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py
git commit -m "refactor(skills): keep MCP render outside execution"
```

Expected: the complete module suite passes with the old response shape.

---

## Stage 2: Exact Catalog Matching

**Goal**: Add `catalog_matches` from one embedded catalog result.

**Success Criteria**: Valid exact and command-restricted declarations match executable names in first-declaration order; duplicates, malformed wrappers, false/malformed descriptors, and unrelated names are omitted; no declarations skip protocol construction.

**Tests**: `test_skills_module.py` matching tests.

**Status**: Not Started

### Task 2.1: Add the field and happy-path resolver

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py`

**Interfaces:**
- Produces: `_declaration_base_name(str) -> str | None`.
- Produces: `_catalog_matches_from_listing(list[str], Any) -> list[str] | None`.
- Produces: `SkillsModule._resolve_catalog_matches(list[str], Any) -> list[str] | None`.

- [ ] **Step 1: Add deterministic failing contract tests**

Add:

```python
@pytest.fixture(autouse=True)
def catalog_protocol_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Mock, AsyncMock]:
    list_tools = AsyncMock(
        return_value={"tools": [{"name": "rag.search", "canExecute": True}]}
    )
    protocol = SimpleNamespace(_handle_tools_list=list_tools)
    factory = Mock(return_value=protocol)
    monkeypatch.setattr(skills_module, "MCPProtocol", factory, raising=False)
    return factory, list_tools
```

Add `"catalog_matches": ["rag.search"]` to the exact response test. Add a mixed case using declarations:

```python
[
    " rag.search ",
    "Bash(git *)",
    "rag.search",
    "Bash(",
    "RAG.SEARCH",
    "missing.tool",
]
```

and listing:

```python
{
    "tools": [
        {"name": "Bash", "canExecute": True},
        {"name": "rag.search", "canExecute": True},
        {"name": "missing.tool", "canExecute": False},
        {"name": "undeclared.tool", "canExecute": True},
        {"name": 7, "canExecute": True},
        "malformed",
    ]
}
```

Assert:

```python
assert result["catalog_matches"] == ["rag.search", "Bash"]
assert "undeclared.tool" not in result["catalog_matches"]
factory.assert_called_once_with()
list_tools.assert_awaited_once_with({}, user_catalogs[1].context)
```

Add a Skill without declarations and assert `catalog_matches == []` and both protocol mocks are untouched.

- [ ] **Step 2: Verify red**

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py -k 'render_is_forced_dry or render_matches_exact_catalog or no_declarations'
```

Expected: `catalog_matches` and resolver behavior are absent.

- [ ] **Step 3: Implement pure matching**

Add below `_clamped_integer`:

```python
def _declaration_base_name(declaration: str) -> str | None:
    """Return the exact catalog name represented by one declaration."""
    if "(" not in declaration:
        return declaration
    base_name, restriction = declaration.split("(", 1)
    if (
        not declaration.endswith(")")
        or not base_name.strip()
        or not restriction[:-1].strip()
    ):
        return None
    return base_name.strip()


def _catalog_matches_from_listing(
    declarations: list[str],
    listing: Any,
) -> list[str] | None:
    """Return unique declared names executable in a well-formed listing."""
    if not isinstance(listing, dict):
        return None
    tools = listing.get("tools")
    if not isinstance(tools, list):
        return None
    executable_names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if (
            isinstance(name, str)
            and name.strip()
            and tool.get("canExecute") is True
        ):
            executable_names.add(name)
    matches: list[str] = []
    seen: set[str] = set()
    for declaration in declarations:
        name = _declaration_base_name(declaration)
        if name is None or name not in executable_names or name in seen:
            continue
        seen.add(name)
        matches.append(name)
    return matches
```

- [ ] **Step 4: Append matches after database cleanup**

Import `MCPProtocol` from `tldw_Server_API.app.core.MCP_unified.protocol`. Change the render dispatch to:

```python
        if tool_name == "skills.render":
            rendered = await self._run_with_service(
                context,
                tool_name,
                lambda service: self._render_skill(service, args),
            )
            rendered["catalog_matches"] = await self._resolve_catalog_matches(
                rendered["declared_tools"],
                context,
            )
            return rendered
```

Add:

```python
    @staticmethod
    async def _resolve_catalog_matches(
        declarations: list[str],
        context: Any,
    ) -> list[str] | None:
        if not declarations:
            return []
        listing = await MCPProtocol()._handle_tools_list({}, context)
        return _catalog_matches_from_listing(declarations, listing)
```

- [ ] **Step 5: Verify green and commit**

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py
git commit -m "feat(skills): add render catalog matches"
```

Expected: all module tests pass. Do not add a factory abstraction, cache, status object, or second catalog API.

---

## Stage 3: Failure, Cleanup, And Cancellation

**Goal**: Keep catalog matching advisory without weakening cleanup, privacy, or cancellation.

**Success Criteria**: Whole lookup and malformed-envelope failure return `null`; partial lists remain computed lists; lookup starts after database close; cancellation propagates; logs expose only exception class.

**Tests**: Failure and lifecycle tests in `test_skills_module.py`.

**Status**: Not Started

### Task 3.1: Bound the post-render lookup

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py`
- Modify: `Docs/MCP/Unified/Modules.md:102-126`

**Interfaces:**
- Refines: `SkillsModule._resolve_catalog_matches(...)`.
- Preserves: the successful render when catalog discovery is unavailable.

- [ ] **Step 1: Add failing resilience tests**

For `RuntimeError("SENTINEL_PRIVATE_DETAIL")`, replace `skills_module.logger`
with `SimpleNamespace(warning=warning)` and assert:

```python
assert result["catalog_matches"] is None
warning.assert_called_once_with(
    "Skills catalog matching unavailable: {}",
    "RuntimeError",
)
assert "SENTINEL_PRIVATE_DETAIL" not in str(warning.call_args)
```

Parameterize malformed envelopes `None`, `{}`, and `{"tools": None}` to expect `None`. Return a partial listing containing only `rag.search` and expect `["rag.search"]`, not `None`. Using `TrackingDB` and `ScenarioService`, make the handler assert the database is closed before it returns.

Add:

```python
@pytest.mark.asyncio
async def test_catalog_matching_propagates_suppressed_cancellation(
    user_catalogs: dict[int, UserCatalog],
    catalog_protocol_stub: tuple[Mock, AsyncMock],
) -> None:
    await _seed_review_skill(user_catalogs[1].service)
    _factory, list_tools = catalog_protocol_stub
    started = asyncio.Event()

    async def suppress_cancellation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            return {"tools": []}

    list_tools.side_effect = suppress_cancellation
    module = await _module()
    task = asyncio.create_task(
        module.execute_tool(
            "skills.render",
            {"skill_name": "review-paper"},
            context=user_catalogs[1].context,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
```

- [ ] **Step 2: Verify red**

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py -k 'catalog_matching or catalog_lookup or catalog_runs_after_database_close'
```

Expected: lookup exceptions escape and suppressed cancellation returns normally.

- [ ] **Step 3: Implement the bounded wrapper**

Replace the resolver body with:

```python
        if not declarations:
            return []
        try:
            listing = await MCPProtocol()._handle_tools_list({}, context)
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - advisory lookup fails closed
            logger.warning(
                "Skills catalog matching unavailable: {}",
                exc.__class__.__name__,
            )
            return None
        return _catalog_matches_from_listing(declarations, listing)
```

Do not catch `BaseException`, log `str(exc)`, retry, or retain protocol state.

- [ ] **Step 4: Document the field**

Add after the existing `declared_tools` bullet:

```markdown
- `catalog_matches` is the unique subset of declared base names found with
  `canExecute: true` in one best-effort embedded catalog read. `[]` means the
  read completed with no match (or no declarations); `null` means matching was
  unavailable. It is advisory and does not replace effective-profile,
  approval, argument, path, credential, quota, or backend checks at tool-call
  time.
```

- [ ] **Step 5: Verify green and commit**

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py Docs/MCP/Unified/Modules.md
git commit -m "fix(skills): preserve render failure semantics"
```

Expected: all module tests pass, including cleanup and cancellation.

---

## Stage 4: Regression Gates And Finalization

**Goal**: Prove integration, scope, and security, then record exact evidence.

**Success Criteria**: Focused and adjacent tests, Ruff, compile, Bandit, and diff hygiene pass; branch scope remains TASK-2294.2-only; Backlog records observed results.

**Tests**: Skills module, dynamic registration, gateway policy regression, and package boundary.

**Status**: Not Started

### Task 4.1: Verify and close the implementation record

**Files:**
- Verify all touched files.
- Update TASK-2294.2 through Backlog CLI.
- Remove this plan only after all prior stages and final verification complete, per `AGENTS.md`.

**Interfaces:**
- Consumes: Stages 1-3.
- Produces: one reviewable implementation branch with reproducible evidence.

- [ ] **Step 1: Run focused and adjacent tests**

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_simulation.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_allowed_tools.py tldw_Server_API/app/core/MCP_unified/tests/test_mcp_discovery_module.py
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
```

Expected: all selected tests pass without weakened assertions or skips.

- [ ] **Step 2: Run static and security gates**

```bash
source ../../.venv/bin/activate
python -m ruff check tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py
python -m compileall -q tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py
python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py -f json -o /tmp/bandit_TASK-2294.2.json
git diff --check origin/dev...HEAD
```

Expected: Ruff and compile exit 0; Bandit reports no new findings; diff check is empty.

- [ ] **Step 3: Audit scope**

```bash
git diff --name-status origin/dev...HEAD
git status --short --branch
```

Expected production scope: `skills_module.py` and `Docs/MCP/Unified/Modules.md`. Expected test scope: `test_skills_module.py`. Design, the in-progress plan, and Backlog records are TASK-2294.2 artifacts. No REST, frontend, gateway, profile, protocol, parser, executor, database, or response-model production file appears.

- [ ] **Step 4: Finalize the task record**

Through `backlog task edit 2294.2`, check all seven acceptance criteria and applicable Definition of Done items. Record exact test counts and commands, Bandit result, commit hashes, any verified baseline limitation, and the final PR URL. Never edit task Markdown directly.

After every stage shows `Complete` and the evidence is recorded, delete only `Docs/Plans/IMPLEMENTATION_PLAN_skills_render_catalog_matches_TASK_2294_2.md` with `apply_patch`, clear its Backlog plan link through the CLI, and commit the final task/documentation update.

- [ ] **Step 5: Invoke completion workflows**

Use `superpowers:requesting-code-review`, address verified findings, rerun affected gates, then use `superpowers:verification-before-completion` and `superpowers:finishing-a-development-branch`. A PR may be opened for review, but it must not be declared merge-ready or merged until the requester supplies the repository-required human-written Change summary.
