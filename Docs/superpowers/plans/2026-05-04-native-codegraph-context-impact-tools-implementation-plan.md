# Native CodeGraph Context Impact Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Stage 4 native CodeGraph read-only `codegraph.impact` and `codegraph.context` tools.

**Architecture:** Keep traversal and relationship reads in `CodeGraphRepository`, add a focused `CodeGraphContextBuilder` for bounded source snippets and task-oriented payload assembly, and keep `CodeGraphModule` as the MCP validation/offload adapter. The implementation must remain read-only, workspace-bounded, and conservative: no Jobs mode, no new language extractors, and no unbounded source output.

**Tech Stack:** Python 3.11, Unified MCP `BaseModule`, SQLite, existing CodeGraph repository/model values, pytest/pytest-asyncio, Loguru, Ruff, Bandit.

---

## Scope

Implement only:

- `codegraph.impact`
- `codegraph.context`
- Repository read helpers needed by those tools
- A small context builder for bounded source extraction
- Focused tests and task documentation updates

Do not implement:

- File watching, background Jobs indexing, or Scheduler integration
- New C/C++/C#/Java/Kotlin extractors
- Whole-file source dumps
- Cross-file semantic resolution beyond graph rows already indexed

## File Structure

- Modify `tldw_Server_API/app/core/DB_Management/codegraph/repository.py`
  - Add read-only graph traversal helpers.
  - Add node batch lookup helpers if needed.
- Create `tldw_Server_API/app/core/CodeGraph/context.py`
  - Add `CodeGraphContextBuilder`.
  - Add workspace-safe source snippet extraction.
  - Add result shaping and truncation metadata.
- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`
  - Add tool definitions, validation, and `asyncio.to_thread` execution paths.
- Modify or add tests:
  - `tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py`
  - `tldw_Server_API/tests/CodeGraph/test_codegraph_context.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`
- Modify `backlog/tasks/task-46 - Implement-native-CodeGraph-context-and-impact-tools.md`
  - Record plan path, implementation notes, verification, and final summary.

## Behavior Contract

### `codegraph.impact`

Arguments:

- `symbol` or `node_id`: exactly one required
- `depth`: default `2`, positive integer, bounded to `4`
- `direction`: default `both`, one of `incoming`, `outgoing`, `both`
- `limit`: default `10`, bounded by `settings.max_search_results`

Response shape:

```python
{
    "workspace_key": "...",
    "index_present": True,
    "root": {...} | None,
    "nodes": [...],
    "relationships": [...],
    "depth": 2,
    "direction": "both",
    "truncated": False,
}
```

Missing index returns `index_present=False` with empty nodes/relationships. A selector that does not resolve to an indexed node returns `index_present=True`, `root=None`, and empty traversal results.

### `codegraph.context`

Arguments:

- `task`: required non-empty string
- `max_nodes`: default `8`, positive integer, bounded by `settings.max_search_results`
- `include_code`: default `True`
- `max_files`: default `5`, positive integer, bounded by `settings.max_search_results`

Response shape:

```python
{
    "workspace_key": "...",
    "index_present": True,
    "task": "...",
    "query": "...",
    "nodes": [...],
    "files": [
        {
            "path": "pkg/file.py",
            "language": "python",
            "snippets": [
                {
                    "start_line": 10,
                    "end_line": 18,
                    "text": "...",
                    "truncated": False,
                }
            ],
        }
    ],
    "relationships": [...],
    "truncation": {
        "max_context_chars": 35000,
        "used_chars": 1234,
        "truncated": False,
    },
}
```

The initial query can be the first useful task phrase after trimming. Keep it simple in this slice: search the task string through existing `search_nodes()`, use top nodes, include same-file callers/callees for those nodes, and group snippets by workspace-relative file path.

## Task 1: Repository Traversal Helpers

**Files:**

- Modify `tldw_Server_API/app/core/DB_Management/codegraph/repository.py`
- Test `tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py`

 **Step 1: Write failing repository traversal tests**

Add tests that seed a graph `entry -> helper -> leaf`, plus an incoming caller to `helper`, then assert:

```python
impact = repo.traverse_impact("node_helper", depth=1, direction="both", limit=10)
assert [node.id for node in impact.nodes] == ["node_entry", "node_helper", "node_leaf"]
assert {edge["edge"]["id"] for edge in impact.relationships} == {"edge_entry_helper", "edge_helper_leaf"}
assert impact.truncated is False
```

Also add a limit test:

```python
impact = repo.traverse_impact("node_helper", depth=2, direction="both", limit=1)
assert impact.truncated is True
assert len(impact.relationships) == 1
```

 **Step 2: Run RED test**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py::test_repository_traverses_bounded_impact_graph \
  tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py::test_repository_impact_traversal_reports_truncation \
  -q
```

Expected: fail because `traverse_impact` does not exist.

 **Step 3: Implement minimal repository helper**

Add a small frozen dataclass or dict return value near repository helpers:

```python
@dataclass(frozen=True)
class ImpactTraversal:
    nodes: tuple[CodeGraphNode, ...]
    relationships: tuple[dict[str, Any], ...]
    truncated: bool
```

Implement `CodeGraphRepository.traverse_impact(node_id, depth, direction, limit)` using breadth-first traversal over `edges`.

Rules:

- Direction `incoming`: follow `target == node_id`
- Direction `outgoing`: follow `source == node_id`
- Direction `both`: follow both
- Include the root node in `nodes`
- De-duplicate nodes and relationship ids deterministically
- Stop once `limit + 1` relationships are collected so truncation can be reported
- Reuse `_relationship_from_joined_row()` style payloads for consistency

 **Step 4: Run GREEN test**

Run the two repository tests again. Expected: pass.

 **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/codegraph/repository.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py
git commit -m "feat: add codegraph impact traversal"
```

## Task 2: Context Builder And Source Snippets

**Files:**

- Create `tldw_Server_API/app/core/CodeGraph/context.py`
- Test `tldw_Server_API/tests/CodeGraph/test_codegraph_context.py`

 **Step 1: Write failing context builder tests**

Add tests for:

- source snippets are read only under `workspace_root`
- snippets include a small line window around node start/end lines
- snippets respect `max_context_chars`
- duplicate file snippets are grouped
- missing source files produce metadata instead of raising

Example core test:

```python
result = builder.build(
    task="update helper",
    nodes=(helper_node,),
    relationships=(),
    max_files=3,
    include_code=True,
)
assert result["files"][0]["path"] == "pkg/sample.py"
assert result["files"][0]["snippets"][0]["start_line"] == 2
assert "def helper" in result["files"][0]["snippets"][0]["text"]
assert result["truncation"]["truncated"] is False
```

 **Step 2: Run RED test**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/CodeGraph/test_codegraph_context.py -q
```

Expected: fail because `context.py` does not exist.

 **Step 3: Implement context builder**

Create `CodeGraphContextBuilder` with:

```python
class CodeGraphContextBuilder:
    def __init__(
        self,
        *,
        workspace_root: Path,
        max_context_chars: int,
        max_file_size_bytes: int,
    ) -> None: ...
    def build(
        self,
        *,
        task: str,
        nodes: tuple[CodeGraphNode, ...],
        relationships: tuple[dict[str, Any], ...],
        max_files: int,
        include_code: bool,
    ) -> dict[str, Any]: ...
```

Snippet rules:

- Reject absolute node file paths and paths containing `..`.
- Resolve each file under `workspace_root`; skip if the resolved path escapes.
- Default snippet window: 3 lines before `start_line`, 3 lines after `end_line`.
- Never read files larger than `max_file_size_bytes`.
- Stop appending snippet text when `max_context_chars` would be exceeded.
- Return `truncation.used_chars`, `truncation.max_context_chars`, and `truncation.truncated`.

 **Step 4: Run GREEN test**

Run the context builder tests. Expected: pass.

 **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/CodeGraph/context.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_context.py
git commit -m "feat: add codegraph context builder"
```

## Task 3: MCP `codegraph.impact`

**Files:**

- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`
- Test `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`

 **Step 1: Write failing MCP impact tests**

Add tests asserting:

- `get_tools()` includes `codegraph.impact` with `readOnlyHint=True`
- unknown args are rejected
- invalid direction/depth/limit are rejected
- missing index returns `index_present=False`
- indexed fixture returns root, nodes, relationships, and truncation status
- `execute_tool("codegraph.impact", ...)` uses `asyncio.to_thread`

 **Step 2: Run RED test**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py::test_codegraph_impact_returns_bounded_relationship_neighborhood \
  -q
```

Expected: fail because the tool is not registered.

 **Step 3: Implement MCP impact wiring**

Update `CodeGraphModule`:

- add tool definition
- add validation branch
- add execution branch using `asyncio.to_thread`
- add `_impact(...)` sync helper
- resolve root node by `node_id` or `symbol`
- call `repository.traverse_impact(...)`
- serialize root via `_node_to_dict`

 **Step 4: Run GREEN MCP impact tests**

Run the impact-specific tests. Expected: pass.

 **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
git commit -m "feat: expose codegraph impact tool"
```

## Task 4: MCP `codegraph.context`

**Files:**

- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`
- Modify `tldw_Server_API/app/core/CodeGraph/context.py` if integration needs small builder changes
- Test `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`

 **Step 1: Write failing MCP context tests**

Add tests asserting:

- `get_tools()` includes `codegraph.context` with `readOnlyHint=True`
- invalid task/max_nodes/max_files/include_code inputs are rejected
- missing index returns `index_present=False`
- indexed fixture returns matching nodes and bounded snippets
- `include_code=False` returns file/node metadata without snippet text
- `execute_tool("codegraph.context", ...)` uses `asyncio.to_thread`

 **Step 2: Run RED test**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py::test_codegraph_context_returns_bounded_source_context \
  -q
```

Expected: fail because the tool is not registered.

 **Step 3: Implement MCP context wiring**

Update `CodeGraphModule`:

- add tool definition with `task`, `max_nodes`, `include_code`, `max_files`
- add validation branch
- add execution branch using `asyncio.to_thread`
- add `_build_context(...)` sync helper
- search nodes with `repository.search_nodes(task, limit=max_nodes)`
- include bounded callers/callees for selected nodes
- call `CodeGraphContextBuilder.build(...)`

 **Step 4: Run GREEN MCP context tests**

Run the context-specific tests. Expected: pass.

 **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  tldw_Server_API/app/core/CodeGraph/context.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
git commit -m "feat: expose codegraph context tool"
```

## Task 5: Verification And PR Prep

**Files:**

- Modify `backlog/tasks/task-46 - Implement-native-CodeGraph-context-and-impact-tools.md`

 **Step 1: Run focused regression suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  -q
```

Expected: all pass.

 **Step 2: Run Ruff**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/app/core/DB_Management/codegraph \
  tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
```

Expected: `All checks passed!`

 **Step 3: Run Bandit**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/app/core/DB_Management/codegraph \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  -f json -o /tmp/bandit_codegraph_context_impact.json
```

Expected: JSON reports `errors: []` and no results for touched scope.

 **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

 **Step 5: Update task and commit final verification**

Update TASK-46 acceptance criteria, notes, DoD, and final summary.

```bash
git add 'backlog/tasks/task-46 - Implement-native-CodeGraph-context-and-impact-tools.md' \
  Docs/superpowers/plans/2026-05-04-native-codegraph-context-impact-tools-implementation-plan.md
git commit -m "docs: record codegraph context impact verification"
```

## Open Review Points

- Keep `depth` bounded to `4` unless reviewers ask for a config field.
- Keep context query simple in this slice. If task-to-symbol matching is weak, improve later with a separate ranking task.
- Source snippets should be helpful but small. Prefer explicit truncation metadata over trying to be clever.
