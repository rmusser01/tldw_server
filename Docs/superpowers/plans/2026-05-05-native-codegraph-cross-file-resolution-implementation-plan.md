# Native CodeGraph Cross-File Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve conservative same-workspace import and call references across indexed files so CodeGraph caller/callee, impact, and context tools can return useful cross-file relationships.

**Architecture:** Keep extractors conservative and language-local. Add a small resolution pass after file graph persistence that reads import nodes and unresolved references from the SQLite repository, resolves only workspace-bounded targets, and writes deterministic edges while preserving enough reference state for later stale cleanup and re-resolution.

**Tech Stack:** Python 3.11, stdlib AST, Tree-sitter JS/TS extractors, SQLite repository helpers, Unified MCP CodeGraphModule, pytest/pytest-asyncio, Ruff, Bandit.

---

## Scope

Implement only:

- Python `from module import symbol` and aliased import call resolution.
- JavaScript/TypeScript relative and `tsconfig`/`jsconfig` path-alias import call resolution for named exports.
- Deterministic cross-file `calls` and `imports` edges for resolved references.
- Stale edge cleanup when source or target files are replaced or removed.
- Focused repository, indexer, and MCP tests proving cross-file relationships show up in existing read tools.

Do not implement:

- Full language-server semantics, package managers, `node_modules`, `sys.path`, class instance inference, overload resolution, wildcard imports, or dynamic imports.
- New MCP tools or a public schema rename.
- File watching, Scheduler sync, or automatic worker startup.
- Cross-language resolution between unrelated language ecosystems.

## File Structure

- Create `tldw_Server_API/app/core/CodeGraph/resolver.py`
  - `CodeGraphReferenceResolver` reads repository imports and unresolved refs.
  - Resolves same-workspace Python and JS/TS imports to existing target nodes.
  - Emits deterministic `imports` and `calls` edges with provenance.
- Modify `tldw_Server_API/app/core/DB_Management/codegraph/schema.sql`
  - Add nullable resolution columns to `unresolved_refs` for target node, edge id, resolution kind, and timestamp.
- Modify `tldw_Server_API/app/core/DB_Management/codegraph/repository.py`
  - Add compatibility migration for the new nullable columns.
  - Add query/update helpers used by the resolver.
  - Count only currently unresolved references in `counts()["unresolved_refs"]`.
- Modify `tldw_Server_API/app/core/CodeGraph/indexer.py`
  - Run the resolver after file graph replacement and once more after full candidate processing.
- Modify `tldw_Server_API/app/core/CodeGraph/extractors/python_extractor.py`
  - Emit import reference metadata sufficient for resolver input.
- Modify JS/TS extractor behavior only if needed to preserve named import metadata.
- Add or modify tests:
  - `tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py`
  - `tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`
  - New `tldw_Server_API/tests/CodeGraph/test_codegraph_resolver.py` if repository tests become too large.
- Modify `backlog/tasks/task-74 - Implement-native-CodeGraph-cross-file-symbol-resolution.md`
  - Record plan, implementation notes, verification, and final summary.

## Task 1: Repository Reference State

**Files:**

- Modify `tldw_Server_API/app/core/DB_Management/codegraph/schema.sql`
- Modify `tldw_Server_API/app/core/DB_Management/codegraph/repository.py`
- Test `tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py`

- [x] **Step 1: Write failing repository tests**

Add tests proving:

- Existing unresolved refs can be marked resolved without disappearing from the reference table.
- `counts()["unresolved_refs"]` counts only refs with no live resolved target.
- Deleting or replacing a target file clears stale resolved state and removes dangling edges.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py -q
```

Expected: fail because repository has no resolved-reference state helpers.

- [x] **Step 2: Implement schema compatibility and helpers**

Add nullable columns if absent:

- `resolved_target TEXT`
- `resolved_edge TEXT`
- `resolution_kind TEXT`
- `resolved_at TEXT`

Add repository helpers to list import nodes, list unresolved/current refs, insert deterministic resolved edges, mark refs resolved, and clear stale resolved refs whose target or edge no longer exists.

- [x] **Step 3: Verify repository tests pass**

Run the repository test command again.

Expected: pass.

## Task 2: Resolver Core

**Files:**

- Create `tldw_Server_API/app/core/CodeGraph/resolver.py`
- Modify `tldw_Server_API/app/core/CodeGraph/extractors/python_extractor.py`
- Test `tldw_Server_API/tests/CodeGraph/test_codegraph_resolver.py`

- [x] **Step 1: Write failing resolver tests**

Add tests proving:

- Python `from pkg.util import helper` resolves a call from `app.main` to `pkg/util.py::helper`.
- Python aliased imports resolve `from pkg.util import helper as h` plus `h()`.
- JS/TS named imports resolve through a precomputed `resolved_path` on import metadata.
- Resolver ignores external/unresolved imports and paths outside the workspace.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_resolver.py -q
```

Expected: fail because the resolver module does not exist.

- [x] **Step 2: Implement minimal resolver**

Build per-file import maps from import nodes:

- Python: imported module/symbol and alias from existing import metadata.
- JS/TS: `resolved_path`, imported name, local alias/name, and namespace imports from existing metadata.

Resolve call refs by local import alias first, then dotted namespace alias when present. Insert `calls` edges from call source node to target node. Insert `imports` edges from import nodes to target module/symbol nodes for impact traversal, but keep callers/callees restricted to `calls`.

- [x] **Step 3: Verify resolver tests pass**

Run the resolver test command again.

Expected: pass.

## Task 3: Indexer Integration

**Files:**

- Modify `tldw_Server_API/app/core/CodeGraph/indexer.py`
- Test `tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py`

- [x] **Step 1: Write failing indexer integration tests**

Add tests proving:

- A Python workspace with two files produces a cross-file call edge after `index_workspace`.
- Re-indexing after a target symbol rename removes the stale cross-file call edge.
- A TS alias import resolves after indexing when TS/TSX parsers are available; otherwise the parser-dependent test skips via existing guard helpers.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py -q
```

Expected: fail because indexer does not invoke cross-file resolution.

- [x] **Step 2: Wire resolver into indexing**

Instantiate `CodeGraphReferenceResolver` once per run and invoke it after file graph replacement. Run a final pass after all candidates are processed so references can resolve even when source files are indexed before target files.

- [x] **Step 3: Verify indexer tests pass**

Run the indexer test command again.

Expected: pass.

## Task 4: MCP Read Tool Coverage

**Files:**

- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`

- [x] **Step 1: Write failing MCP tests**

Add tests proving:

- `codegraph.callers` for a target function includes a caller from a different file.
- `codegraph.callees` for a caller function includes a target from a different file.
- `codegraph.impact` and `codegraph.context` include cross-file relationships while preserving configured limits and truncation metadata.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q
```

Expected: fail before resolver integration is complete.

- [x] **Step 2: Preserve read-tool caps and filter callers/callees**

If `imports` edges are added, ensure `list_callers` and `list_callees` return only `calls` relationships. Keep `impact` and `context` able to include both imports and calls through existing traversal.

- [x] **Step 3: Verify MCP tests pass**

Run the MCP test command again.

Expected: pass.

## Task 5: Final Verification And Closeout

**Files:**

- Modify `backlog/tasks/task-74 - Implement-native-CodeGraph-cross-file-symbol-resolution.md`
- Optionally update GitHub issue `#1259` with current merged stages and the new PR link after PR creation.

- [x] **Step 1: Run focused CodeGraph/MCP tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py \
  -q
```

- [x] **Step 2: Run Ruff**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/app/core/DB_Management/codegraph \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
```

- [x] **Step 3: Run Bandit**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/app/core/DB_Management/codegraph \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  -f json -o /tmp/bandit_codegraph_cross_file_resolution.json
```

- [x] **Step 4: Run whitespace check**

```bash
git diff --check
```

- [x] **Step 5: Update TASK-74 and commit**

Record verification, mark acceptance criteria and DoD complete, then commit the plan, task, tests, and implementation.
