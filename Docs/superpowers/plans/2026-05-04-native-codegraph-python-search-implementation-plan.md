# Native CodeGraph Python Extractor And Search Implementation Plan

Goal: add the Stage 2 Python-only CodeGraph slice on top of the merged foundation. The slice should turn indexed Python files into deterministic symbol nodes, same-file call edges or unresolved references, repository search/query results, and Unified MCP read tools.

Scope boundaries:

- Include Python extraction via the standard library `ast` module.
- Include repository graph persistence and query helpers needed by Stage 2 tools.
- Include MCP tools: `codegraph.search`, `codegraph.node`, `codegraph.callers`, and `codegraph.callees`.
- Preserve Stage 1 tools and bounded foreground behavior.
- Exclude JavaScript/TypeScript extraction, TypeScript path aliases, Jobs mode, `impact`, `context`, and planned-language extractors.

## Stage 1: Extractor And Graph Models

**Goal**: Define the data passed from extractors to storage and prove Python AST extraction behavior with focused tests.

**Success Criteria**:

- Module, class, function, method, and import nodes get deterministic IDs.
- Same-file direct calls resolve to `calls` edges where possible.
- Ambiguous or unresolved calls become unresolved refs rather than noisy edges.
- Optional Tree-sitter packages are not imported for Python extraction.

**Tests**:

- `tldw_Server_API/tests/CodeGraph/test_codegraph_python_extractor.py`

**Status**: Complete

## Stage 2: Repository Graph Persistence And Search

**Goal**: Persist graph rows safely and expose node/search/call relationship query helpers.

**Success Criteria**:

- Replacing a file deletes owned graph rows and dangling edges before inserting new graph rows.
- Search returns deterministic bounded symbol matches from indexed graph rows.
- Node lookup can resolve by `node_id` and by exact/qualified symbol name.
- Caller/callee queries do not return dangling relationships.

**Tests**:

- `tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py`

**Status**: Complete

## Stage 3: Bounded Indexer Integration

**Goal**: Use the Python extractor during indexing while keeping existing foreground limits and Stage 1 inventory behavior intact.

**Success Criteria**:

- Python files record node counts and graph rows during index/sync.
- JS/TS foundation inventory remains file-only until its extractor slice.
- Planned-language files are still skipped.
- Existing file, byte, and wall-clock bounds still prevent partial over-limit indexing.

**Tests**:

- `tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py`

**Status**: Complete

## Stage 4: MCP Stage 2 Read Tools

**Goal**: Expose Python graph search and relationship reads through `CodeGraphModule`.

**Success Criteria**:

- `get_tools()` includes Stage 1 tools plus `search`, `node`, `callers`, and `callees`.
- All Stage 2 tools are read-only and offload blocking repository work via `asyncio.to_thread`.
- Argument validation rejects unknown parameters and unsafe limits.
- Protocol tests cover at least one Stage 2 validation path.

**Tests**:

- `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`

**Status**: Complete

## Stage 5: Verification And Task Closeout

**Goal**: Verify the slice and record results in Backlog.

**Success Criteria**:

- Focused CodeGraph and adjacent MCP tests pass.
- Bandit reports no new actionable findings in touched CodeGraph/MCP code.
- `git diff --check` passes.
- TASK-27 includes verification notes and a final summary.

**Tests**:

- `python -m pytest tldw_Server_API/tests/CodeGraph tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py -q`
- `python -m bandit -r tldw_Server_API/app/core/CodeGraph tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py -f json -o /tmp/bandit_codegraph_python_search.json`
- `git diff --check`

**Status**: Complete

## Local Review Notes

- Python extraction should use `ast` for this slice. It is stable, available by default, and avoids coupling the Python extractor to optional Tree-sitter package availability.
- Keep search truthful: same-file direct calls can become resolved `calls` edges; anything uncertain should remain unresolved.
- Avoid expanding the module into source excerpt tooling. `include_code` may remain false or unsupported in this slice unless tests justify a bounded implementation.
