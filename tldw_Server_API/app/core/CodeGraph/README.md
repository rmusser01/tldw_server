# CodeGraph

CodeGraph indexes trusted workspace source trees into a local code graph for MCP tools. It discovers supported languages, extracts symbols and references, stores indexed files and edges in a per-workspace SQLite repository, resolves callers and callees, builds bounded source context snippets, and runs foreground or Jobs-backed index and sync operations.

## Start Here

- `workspace.py` resolves trusted workspace roots and per-workspace index database paths.
- `config.py` defines index limits, file size bounds, search limits, context limits, and excluded paths.
- `indexer.py` performs foreground index and sync operations.
- `language_registry.py` declares supported languages, extensions, and optional parser availability.
- `context.py` builds bounded source context from indexed workspaces.
- `jobs.py` and `jobs_worker.py` integrate CodeGraph indexing with the Jobs module.
- Related MCP surface: `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`.
- Related persistence: `tldw_Server_API/app/core/DB_Management/codegraph/repository.py`.
- Related tests: `tldw_Server_API/tests/CodeGraph/`.

## Responsibilities

- Resolve a trusted workspace root from MCP Hub workspace context.
- Build a stable workspace key and local index database path.
- Index and sync source files within configured file, byte, and time bounds.
- Extract symbols and references for Python and optional tree-sitter-backed languages.
- Store indexed files, graph nodes, edges, and unresolved references.
- Resolve search, node lookup, callers, callees, impact, and source context requests.
- Offload long-running index and sync work to Jobs workers when requested.

## Module Map

- `config.py`: `CodeGraphSettings` and operational bounds.
- `workspace.py`: workspace root resolution and index path calculation.
- `models.py`: language, workspace, file, node, edge, and unresolved reference models.
- `language_registry.py`: language metadata and parser availability reporting.
- `indexer.py`: foreground index and sync orchestration.
- `repository.py`: core-facing repository helpers around the DB management repository.
- `resolver.py`: call graph and reference resolution.
- `context.py`: bounded context snippet assembly.
- `jobs.py`, `jobs_worker.py`: Jobs enqueue and worker execution.
- `extractors/`: language-specific extractors and tree-sitter loader helpers.
- `dependencies.py`: dependency construction for CodeGraph services.

## How It Connects

- CodeGraph is exposed through MCP Unified, not a direct FastAPI endpoint found in this pass.
- `codegraph_module.py` exposes MCP tools for status, index, sync, files, search, node, callers, callees, impact, and context.
- Workspace resolution depends on MCP Hub trusted workspace roots rather than arbitrary user-supplied filesystem paths.
- Persistent graph data is stored through `DB_Management/codegraph/repository.py`.
- `jobs.py` enqueues `codegraph_index` work in the `codegraph` Jobs domain; `jobs_worker.py` consumes it.
- Optional tree-sitter packages determine whether JavaScript, TypeScript, Java, Kotlin, C#, C, and C++ extraction is available.

## Extension Points

- Add language support by updating `language_registry.py` and adding or extending an extractor under `extractors/`.
- Change index limits in `config.py` and cover the bounds in config or indexer tests.
- Add graph query behavior in `resolver.py` and repository tests.
- Add MCP tool behavior in `MCP_unified/modules/implementations/codegraph_module.py` after checking service contracts here.
- Change Jobs payload behavior in `jobs.py` and `jobs_worker.py`.

## Testing

- Direct tests live under `tldw_Server_API/tests/CodeGraph/`.
- Coverage includes config, workspace resolution, context, repository, resolver, indexer, jobs, jobs worker, language registry, and language extractor tests.
- MCP tool behavior is covered alongside CodeGraph and MCP Unified tests; inspect `Docs/MCP/Unified/CodeGraph.md` for the documented tool contract.

## Gotchas

- CodeGraph requires a trusted workspace root; it intentionally avoids arbitrary root path arguments.
- Foreground indexing is bounded by file count, total bytes, file size, and time limits, so partial status can be expected.
- Optional parser dependencies change language extraction coverage.
- Jobs worker payloads validate `index_base_dir` and `index_db_path` against the local worker configuration before writing.
