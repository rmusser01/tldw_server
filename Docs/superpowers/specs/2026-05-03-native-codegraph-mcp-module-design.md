# Native CodeGraph MCP Module Design

Date: 2026-05-03
Status: Approved design
Owner: Codex brainstorming session
Backlog: TASK-14

## Objective

Add a native Python CodeGraph-style capability to `tldw_server` as a Unified
MCP module. The feature should give agents fast, local codebase intelligence:
symbol search, file inventory, node details, callers, callees, impact radius,
and task-focused context without requiring an external Node-based MCP server.

The first implementation slice should prioritize depth and practical value for
the current `tldw_server` repository: Python plus JavaScript/TypeScript. The
architecture must also be ready for C, C++, C#, Java, and Kotlin extractors
without forcing those languages into the initial delivery.

## User-Approved Decisions

1. Build a native Python implementation rather than federating the upstream
   CodeGraph MCP server.
2. Scope the first useful implementation to deep Python plus
   JavaScript/TypeScript support.
3. Keep the graph model and parser contracts ready for C, C++, C#, Java, and
   Kotlin.
4. Package CodeGraph dependencies behind an optional extra, `.[codegraph]`, so
   normal media/RAG/chat deployments do not inherit developer-tooling
   dependencies.
5. Expose the capability through Unified MCP tools, not as a separate service
   plane.

## Goals

- Provide local, workspace-scoped code intelligence through MCP Unified.
- Reuse tldw's existing module lifecycle, RBAC, rate limits, circuit breakers,
  schema validation, and audit behavior.
- Keep indexed data under tldw-managed storage rather than writing `.codegraph`
  directories into user repositories by default.
- Make first-slice results useful enough for real agent exploration in this
  repo.
- Keep parser support incremental so language quality can grow without
  rewriting the storage or tool surface.

## Non-Goals

- Full semantic parity with upstream CodeGraph in the first slice.
- Full static type checking for TypeScript, Java, C#, Kotlin, C, or C++.
- Precise macro/template/preprocessor modeling for C and C++ in v1.
- A file watcher in the first slice. Incremental `sync` should be explicit and
  hash-based first.
- Arbitrary host filesystem indexing.
- Source-code editing tools. This module indexes and reads; it does not modify
  user repositories.
- Replacing RAG, notes, filesystem, or run-command MCP tools.

## Current Repo Fit

Unified MCP already provides the right control plane:

- New tools are added as modules under
  `tldw_Server_API/app/core/MCP_unified/modules/implementations/`.
- `BaseModule` provides lifecycle, health, metrics, timeouts, circuit breaker,
  input sanitization, and write-tool classification helpers.
- `ModuleRegistry` maps tools to modules and routes execution.
- The protocol enforces RBAC and write-tool validation before execution.
- The filesystem module already demonstrates workspace-root resolution and
  path-bound access.
- External federation remains useful as a comparison and fallback, but native
  CodeGraph should not depend on the upstream server at runtime.

The new feature should therefore be implemented as a native core package plus a
thin MCP module:

- Core package: `tldw_Server_API/app/core/CodeGraph/`
- MCP module:
  `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`
- Tests:
  `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`
  and focused core tests under an appropriate CodeGraph test location.

## Approaches Considered

### Approach 1: Equal shallow support for all requested languages

Implement basic symbol extraction for C, Python, C++, C#, JavaScript, Java,
Kotlin, and TypeScript in the first slice.

Pros:

- Broad language checkbox coverage.
- Early validation of the adapter abstraction.

Cons:

- Most results would be shallow and less useful for real agent work.
- Call graph and import resolution quality would vary sharply by language.
- Review and testing surface would be too broad for a focused first PR.

### Approach 2: Deep Python plus JavaScript/TypeScript first

Build the full service shape, storage model, and MCP tool surface, then
implement high-quality extraction for Python and JavaScript/TypeScript first.
Register the other requested languages as planned adapter targets.

Pros:

- Immediately useful for tldw's backend and frontend.
- Keeps the first implementation reviewable.
- Exercises both Python and TypeScript-style module systems.
- Avoids freezing weak extractor behavior across eight languages.

Cons:

- The first release will not cover all requested languages with real parsing.
- Follow-up tasks are required for JVM, .NET, and native language depth.

### Approach 3: Python-only MVP

Implement only Python, then add JS/TS and other languages later.

Pros:

- Fastest narrow implementation.
- Good fit for backend exploration.

Cons:

- Misses the Next.js frontend.
- Does not prove the adapter shape across multiple language families.
- Delays the main cross-repo value.

## Recommendation

Use Approach 2: deep Python plus JavaScript/TypeScript first, with a
language-neutral graph model and parser adapter contract.

## Proposed Architecture

### Package Layout

```text
tldw_Server_API/app/core/CodeGraph/
  __init__.py
  config.py
  models.py
  language_registry.py
  workspace.py
  repository.py
  schema.sql
  indexer.py
  sync.py
  query.py
  context_builder.py
  formatters.py
  extractors/
    __init__.py
    base.py
    tree_sitter_loader.py
    python_extractor.py
    javascript_extractor.py
    typescript_extractor.py
    planned.py
```

The MCP module should depend on this package rather than embedding indexing or
query logic directly in `execute_tool`.

### Core Components

`CodeGraphWorkspaceResolver`

- Resolves the active trusted workspace root from MCP request context.
- Reuses the same trust model as the filesystem MCP module.
- Rejects missing, untrusted, or outside-root paths.
- Produces a stable workspace key for tldw-managed index storage.

`CodeGraphRepository`

- Owns SQLite connection setup and schema migration.
- Stores files, nodes, edges, unresolved references, index runs, and metadata.
- Provides FTS-backed symbol search.
- Uses tldw SQLite tuning helpers where appropriate.
- Generates deterministic node IDs from stable workspace-relative identity
  fields so incremental sync can update graph rows without breaking cross-file
  edges.

`CodeGraphIndexer`

- Discovers source files under the trusted workspace root.
- Applies include/exclude rules and max-file-size limits.
- Hashes files to avoid unnecessary re-extraction.
- Deletes stale nodes/edges when files are removed.
- Delegates parsing to `ExtractorRegistry`.

`ExtractorRegistry`

- Maps language IDs and file extensions to extractor instances.
- Reports language availability and missing optional dependencies.
- Supports planned-language entries without claiming extraction support.

`GraphQueryService`

- Implements search, node lookup, callers, callees, impact radius, and file
  inventory queries over the stored graph.
- Keeps query logic independent from MCP formatting.

`CodeGraphContextBuilder`

- Builds bounded task context from search entry points plus graph traversal.
- Groups relevant source excerpts by file.
- Applies output caps before returning to MCP.

## Storage Design

Indexes should live under tldw-managed storage, not inside the target
repository by default:

```text
Databases/codegraph/<workspace_key>/codegraph.db
```

`workspace_key` should include enough scope to avoid collisions across users or
shared workspaces. At minimum, derive it from the resolved workspace root and
owner/workspace metadata when available. Use a stable hash rather than raw path
text in directory names.

### Schema Shape

The native schema should intentionally mirror the useful parts of upstream
CodeGraph while staying tldw-owned:

- `schema_versions`
- `files`
  - path, language, size, content_hash, modified_at, indexed_at, node_count,
    errors
- `nodes`
  - id, identity_key, kind, name, qualified_name, file_path, language, start_line, end_line,
    start_column, end_column, signature, docstring, visibility, flags, metadata
- `edges`
  - id, source, target, kind, file_path, line, column, metadata, provenance
- `unresolved_refs`
  - from_node_id, reference_name, reference_kind, file_path, line, column,
    candidates, language
- `index_runs`
  - run_id, workspace_key, started_at, finished_at, mode, status, counters,
    error_summary
- `project_metadata`
  - key, value, updated_at
- `nodes_fts`
  - FTS5 index over name, qualified_name, signature, docstring, and selected
    metadata text

The repository should expose migration functions instead of relying on ad hoc
table creation inside query code.

### Stable Identity And Cleanup

Node identity must be deterministic across re-indexes. The first implementation
should derive `nodes.id` from a normalized identity key instead of database
autoincrement state.

Suggested identity key:

```text
<workspace_key>:<language>:<file_path>:<kind>:<qualified_name>:<start_line>
```

If an extractor can provide a stronger language-native identity, such as a
qualified symbol path plus container hierarchy, it may use that as long as the
identity remains deterministic for unchanged source. The repository should store
the pre-hash identity string in `identity_key` for debugging and migration.

Edges should also have deterministic IDs derived from:

```text
<source_node_id>:<edge_kind>:<target_or_unresolved_ref>:<file_path>:<line>:<column>
```

Incremental sync must clean graph state file-by-file:

1. Mark the file as `indexing`.
2. Delete old nodes, edges, and unresolved references owned by that file.
3. Insert the new file record, nodes, direct edges, and unresolved refs.
4. Re-run resolution for references from the changed file and references in
   other files that previously targeted nodes from the changed file.
5. Remove or downgrade cross-file edges whose target node no longer exists.
6. Mark the file as `indexed` or `error`.

Query code should never return dangling edges. If cleanup discovers an edge
whose endpoint is missing, it should either delete the edge or surface it as an
unresolved reference during the next resolution pass.

### Node Kinds

Initial supported node kinds:

- `file`
- `module`
- `class`
- `interface`
- `function`
- `method`
- `constructor`
- `property`
- `field`
- `variable`
- `constant`
- `import`
- `export`
- `type_alias`
- `enum`
- `component`

Not every language needs every kind in v1. The schema must allow them so later
extractors do not require migrations for ordinary language concepts.

### Edge Kinds

Initial edge kinds:

- `contains`
- `imports`
- `exports`
- `calls`
- `references`
- `extends`
- `implements`
- `instantiates`
- `decorates`
- `type_of`
- `returns`

For v1, cross-file `calls` should be best-effort. Unresolved call and import
references should be stored explicitly rather than silently dropped.

## Parser Adapter Contract

Each extractor should implement a small, explicit contract:

```python
class CodeGraphExtractor(Protocol):
    language_id: str
    extensions: tuple[str, ...]

    def available(self) -> bool: ...

    def extract(
        self,
        *,
        root: Path,
        file_path: Path,
        source: bytes,
        settings: CodeGraphSettings,
    ) -> ExtractionResult: ...
```

`ExtractionResult` should contain:

- normalized file record fields
- extracted nodes
- extracted edges whose source and target are both known
- unresolved references for later resolution
- parser or extraction warnings
- duration and counters for status reporting

The contract should require extractors to return workspace-relative paths and
1-based line numbers. The repository layer should reject absolute file paths in
node and edge records.

### Primary Parsing Engine

Use Tree-sitter through Python bindings for the first implementation:

- `tree-sitter`
- `tree-sitter-python`
- `tree-sitter-javascript`
- `tree-sitter-typescript`

These dependencies belong in `.[codegraph]`. The module should remain importable
without them and report degraded health when the optional extra is missing.

### Python Extractor V1

Support:

- modules and files
- classes
- methods
- functions
- async functions and methods
- decorators
- imports and from-imports
- direct calls by identifier or attribute expression
- class inheritance names
- docstrings where easily available
- function signatures from source ranges where possible

Resolution should be conservative:

- resolve same-file calls by local symbol name
- resolve imported module or symbol references when the import target maps to an
  indexed file
- record unresolved references when there is ambiguity

### JavaScript/TypeScript Extractor V1

Support:

- `.js`, `.jsx`, `.mjs`, `.cjs`
- `.ts`, `.tsx`
- functions, arrow-function variables, classes, methods, constructors
- exported declarations
- import declarations and re-exports
- React-style component functions/classes as `component` when name casing or
  JSX usage makes this clear
- direct calls by identifier or member expression
- TypeScript interfaces, type aliases, enums, and class heritage where the
  grammar exposes them reliably
- `tsconfig.json` and `jsconfig.json` path aliases for workspace packages and
  frontend aliases such as `@/*`, `~/*`, `@web/*`, and `@tldw/ui/*`

Resolution should be conservative:

- resolve same-file calls by name.
- resolve relative imports to indexed files for common extensions.
- resolve configured TypeScript/JavaScript path aliases when the config file is
  inside the trusted workspace root and the target path resolves under the same
  root.
- record unresolved package imports instead of trying to inspect
  `node_modules`.
- do not require the TypeScript compiler or project type checking in v1.

The path-alias resolver should parse `compilerOptions.baseUrl` and
`compilerOptions.paths` from the nearest applicable `tsconfig.json` or
`jsconfig.json`. It should ignore aliases whose target escapes the trusted
workspace root and should record unresolved aliases with a clear reason.

### Planned Extractors

The registry should include planned-language metadata for:

- C
- C++
- C#
- Java
- Kotlin

Before their real extractors land, status should report them as `planned` or
`dependency_missing`, not silently unsupported. This makes the roadmap visible
without claiming false capability.

Later slices should add real extractors in this order:

1. Java and Kotlin
2. C#
3. C and C++

This order prioritizes languages with clearer class/method/import models before
the macro-heavy native-language pass.

## MCP Tool Surface

All tools should be exposed by a `CodeGraphModule`.

### `codegraph.status`

Read-only. Returns:

- dependency availability
- supported and planned languages
- workspace key
- index presence
- file, node, and edge counts
- last index run
- stale file count if cheap to compute

### `codegraph.index`

Management/write tool. Builds or rebuilds the workspace index.

Arguments:

- `force`: boolean, default false
- `languages`: optional list of language IDs
- `mode`: `foreground` in v1; `job` is reserved for the later Jobs-backed slice
- `max_files`: optional bounded test/development limit

V1 should use bounded foreground indexing only. This keeps the first
implementation reviewable and avoids coupling the core graph/indexer work to
Jobs worker registration and status APIs. Foreground indexing must enforce
strict file, byte, and wall-clock limits. If the workspace exceeds those limits,
`codegraph.index` should return `index_too_large_for_foreground` with the
estimated file count and guidance to enable the later Jobs-backed slice.

The Jobs-backed implementation should be a follow-up slice. Once it exists,
`mode` can accept `job`, and full workspace indexing can default to Jobs because
indexing is user-visible, cancellable, and potentially long-running work.

### `codegraph.sync`

Management/write tool. Incrementally updates changed and removed files.

Arguments:

- `languages`: optional list
- `mode`: `foreground` in v1; `job` is reserved for the later Jobs-backed slice
- `max_files`: optional limit

If the change set is too large for bounded foreground execution, the tool should
return `sync_too_large_for_foreground` rather than blocking a tool call.

### `codegraph.files`

Read-only. Lists indexed files.

Arguments:

- `path`: optional workspace-relative directory prefix
- `pattern`: optional glob
- `format`: `tree`, `flat`, or `grouped`
- `include_metadata`: default true
- `limit`: default bounded

### `codegraph.search`

Read-only. Searches symbols through FTS and exact-name fallback.

Arguments:

- `query`: required
- `kind`: optional node kind
- `language`: optional language ID
- `limit`: default 10, bounded

Returns locations and signatures by default, not source code.

### `codegraph.node`

Read-only. Returns one symbol's details.

Arguments:

- `symbol`: required unless `node_id` is provided
- `node_id`: optional exact id
- `include_code`: default false

### `codegraph.callers`

Read-only. Returns incoming `calls` and selected `references`.

Arguments:

- `symbol` or `node_id`
- `limit`

### `codegraph.callees`

Read-only. Returns outgoing `calls` and selected `references`.

Arguments:

- `symbol` or `node_id`
- `limit`

### `codegraph.impact`

Read-only. Traverses incoming and outgoing edges to show likely blast radius.

Arguments:

- `symbol` or `node_id`
- `depth`: default 2, bounded
- `direction`: `incoming`, `outgoing`, or `both`
- `limit`

### `codegraph.context`

Read-only. Builds task-oriented context.

Arguments:

- `task`: required
- `max_nodes`: default bounded
- `include_code`: default true
- `max_files`: default bounded

This tool should be careful with source output. It should include enough code
to support understanding but stay below configurable character limits.

## Workspace Safety

The module must not accept arbitrary host paths by default. Tool calls should
resolve the workspace from MCP request context, mirroring the filesystem module.

Rules:

- Only index files under the resolved trusted workspace root.
- Resolve symlinks and reject files that escape the workspace root.
- Store workspace-relative paths in the graph.
- Keep source excerpts bounded by file size, line count, and total output size.
- Skip binary files and files over `max_file_size`.
- Apply default excludes for dependency/build/cache folders such as
  `.git`, `node_modules`, `.venv`, `venv`, `__pycache__`, `dist`, `build`,
  `.next`, `coverage`, `target`, and generated site output.
- Do not log source text, signatures containing secrets, or full query outputs.
- Treat `codegraph.index` and `codegraph.sync` as write-capable because they
  mutate tldw-managed index storage.

## Dependency And Configuration Model

Add an optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
codegraph = [
  # Exact compatible ranges must be verified during implementation.
  # Do not land broad lower bounds without a tested parser matrix.
  "tree-sitter>=0.25,<0.26",
  "tree-sitter-python>=0.25,<0.26",
  "tree-sitter-javascript>=0.25,<0.26",
  "tree-sitter-typescript>=0.23,<0.24",
]
```

Implementation must verify and document a compatible parser matrix before
landing `.[codegraph]`. The matrix should include Python versions supported by
this repo, wheel availability on Linux/macOS/Windows where practical, and a
minimal parse smoke test for Python, JavaScript, TypeScript, and TSX. If the
candidate ranges above are not compatible with current package availability,
the implementation should adjust them and record the tested set in the task
notes or developer documentation. Planned-language parser packages should be
added only when those extractors are implemented.

Module config should live in `mcp_modules.yaml`:

```yaml
- id: codegraph
  class: tldw_Server_API.app.core.MCP_unified.modules.implementations.codegraph_module:CodeGraphModule
  enabled: false
  name: CodeGraph
  version: "0.1.0"
  department: code
  max_concurrent: 4
  settings:
    index_base_dir: Databases/codegraph
    max_file_size_bytes: 1048576
    max_context_chars: 35000
    max_search_results: 100
```

The default should remain disabled until dependencies and operator intent are
present.

## Indexing And Sync Lifecycle

### Full Index

1. Resolve workspace root and workspace key.
2. Initialize or migrate the workspace graph database.
3. Discover files by extension and include/exclude rules.
4. Hash candidate files.
5. Delete records for removed files.
6. Parse changed or forced files through registered extractors.
7. Delete stale graph rows for changed/removed files using the stable identity
   cleanup rules.
8. Persist file records, nodes, edges, unresolved references, and errors in one
   file-scoped transaction where practical.
9. Run a resolution pass over unresolved references and stale cross-file edge
   targets.
10. Update FTS tables and index run metadata.

### Incremental Sync

`sync` should perform the same flow but only for changed, removed, or newly
matched files. File watching can be added later by scheduling syncs or adding a
debounced watcher service, but it is not required for the first slice.

### Error Handling

Parser errors should not fail the entire index run unless core storage or
workspace resolution fails. Per-file errors should be stored in `files.errors`
and summarized by `codegraph.status`.

## Resolution Strategy

V1 should favor truthful, best-effort resolution over noisy precision claims.

Resolution order:

1. Same-file exact qualified name.
2. Same-file simple name.
3. Relative import/export target.
4. Python package/module path under workspace.
5. JS/TS relative module path under workspace.
6. JS/TS path alias target under workspace using `tsconfig.json` or
   `jsconfig.json`.
7. Ambiguous candidate set recorded in `unresolved_refs`.

External package imports should be represented as unresolved or external
references, not indexed by walking dependency folders.

## Output Formatting

Tool responses should be compact JSON-compatible dictionaries, with markdown
only where the MCP client benefits from narrative context.

`codegraph.context` can return:

- `summary`
- `entry_points`
- `relationships`
- `files`
- `source_sections`
- `additional_relevant_files`
- `truncated`
- `limits`

Source excerpts should include file path and line numbers so agents can follow
up with `fs.read_text` when needed and permitted.

## Test Strategy

### Core Unit Tests

- schema creation and migration
- file hashing and exclude matching
- workspace key generation
- repository CRUD for files, nodes, edges, and unresolved refs
- FTS search ranking and filtering
- graph traversal depth and direction
- context output caps and truncation

### Extractor Tests

Use small fixture files for each supported first-slice language.

Python:

- functions, async functions, classes, methods
- imports and from-imports
- decorators
- direct calls and attribute calls
- same-file caller/callee resolution

JavaScript/TypeScript:

- functions, arrow functions, classes, methods
- imports and exports
- TS interfaces, types, enums
- JSX/TSX component detection where supported
- relative import resolution
- `tsconfig.json` and `jsconfig.json` path alias resolution, including aliases
  used by `apps/tldw-frontend`

Planned languages:

- registry reports planned or missing support without crashing
- files with planned-language extensions are skipped with clear status until
  their extractor lands

### MCP Module Tests

- tool definitions include correct metadata and read/write categories
- missing optional dependencies produce degraded health and clear status
- `codegraph.index` and `codegraph.sync` require custom argument validation
- read tools reject unknown arguments and respect output limits
- workspace root unavailable returns a permission-style error
- indexing refuses paths outside workspace scope

### Integration Tests

- create a temp workspace with Python and TS files
- include a frontend-like `tsconfig.json` fixture with relative imports and path
  aliases
- index foreground with a low file limit
- assert search, node, callers, callees, impact, files, and context work through
  `MCPProtocol` or module execution
- verify index storage is written under a temp tldw index base, not the source
  workspace
- re-index a modified file and assert stable node IDs do not leave stale or
  dangling edges

### Security Validation

When implementation code exists, run focused tests and Bandit on touched
CodeGraph and MCP module paths before completion.

## Rollout Plan

### Stage 1: Native Graph Foundation

Goal: Add optional package dependencies, schema, repository, workspace resolver,
language registry, bounded foreground indexing mode, and module health/status
without real extraction depth.

Success criteria:

- `codegraph.status` reports dependency and language availability.
- Databases are created under the configured tldw index base.
- Missing optional dependencies degrade cleanly.
- `codegraph.index` and `codegraph.sync` run only in bounded foreground mode
  and reject over-limit workspaces with clear errors.
- The tested parser dependency matrix is recorded before adding `.[codegraph]`.

### Stage 2: Python Extractor And Search

Goal: Implement Python indexing, FTS search, node lookup, files, and basic
same-file callers/callees.

Success criteria:

- Python fixture project indexes successfully.
- `codegraph.search`, `codegraph.node`, `codegraph.callers`, and
  `codegraph.callees` work for fixture symbols.

### Stage 3: JavaScript/TypeScript Extractors

Goal: Add JS/TS/TSX extraction, import/export capture, and relative import
resolution, including TypeScript/JavaScript path aliases.

Success criteria:

- Frontend-like fixture project indexes successfully.
- TS symbols, components, imports, exports, path-alias references, and call
  references are searchable.

### Stage 4: Context And Impact Tools

Goal: Add graph traversal, impact radius, and bounded context builder.

Success criteria:

- `codegraph.impact` returns bounded relationship neighborhoods.
- `codegraph.context` returns useful source sections with truncation metadata.

### Stage 5: Deferred Language Extractors

Goal: Add real extractor implementations for Java/Kotlin, then C#, then C/C++.

Success criteria:

- Each language lands with fixtures, extraction coverage, and documented
  resolution limits.

## Risks And Mitigations

### Risk: Optional dependency drift

Tree-sitter language packages can change APIs or wheel availability.

Mitigation:

- Keep CodeGraph optional.
- Centralize parser loading in one module.
- Add dependency-health reporting to `codegraph.status`.

### Risk: Noisy or misleading call graphs

Dynamic languages and partial type information make perfect resolution
impossible.

Mitigation:

- Store unresolved references explicitly.
- Label best-effort resolution in output.
- Prefer fewer, higher-confidence edges over noisy guesses.

### Risk: Long indexing blocks MCP requests

Large workspaces may take too long for a normal tool call.

Mitigation:

- Default full indexing to Jobs after the Jobs-backed follow-up slice exists.
- Keep v1 foreground mode bounded and primarily for tests/small repos.
- Defer Jobs integration to its own slice so worker registration, status,
  cancellation, and quotas are reviewed separately from extraction correctness.
- Record index run status and expose it through `codegraph.status`.

### Risk: Workspace data leakage

The module reads source code and can return snippets.

Mitigation:

- Use trusted workspace root resolution.
- Enforce path and size bounds.
- Do not log source text.
- Keep source excerpts opt-in for detail tools and capped for context tools.

### Risk: First-slice scope expands into a static-analysis platform

Adding eight languages can become a broad roadmap.

Mitigation:

- Treat Python plus JS/TS as the only first-slice implementation target.
- Keep planned languages visible but non-claiming.
- Require each later language extractor to land as its own reviewable slice.

## Open Questions For Implementation

- Should `codegraph.context` return pure JSON, markdown text, or both?
- Should the module expose resources for indexed files/nodes, or keep v1 tool
  only?
- Should workspace keys include only resolved root path plus user/workspace
  metadata, or also a config hash so exclude/language changes create distinct
  indexes?

## Follow-Up Implementation Decisions Already Resolved

- V1 indexing and sync use bounded foreground execution only. Jobs integration
  is a later slice.
- JS/TS import resolution includes trusted-workspace `tsconfig.json` and
  `jsconfig.json` path aliases.
- Node and edge identity must be deterministic across re-indexes.
- The implementation must land a tested Tree-sitter dependency matrix, not
  broad unverified lower bounds.

## Success Criteria

- The native CodeGraph module can be enabled only when `.[codegraph]`
  dependencies are installed.
- A trusted workspace can be indexed without writing into that workspace.
- Python and JS/TS symbols can be searched and inspected.
- Basic callers/callees and impact/context tools work with documented
  best-effort semantics.
- Unsupported planned languages are visible in status without false capability
  claims.
- The module follows Unified MCP permission, validation, and workspace-safety
  patterns.
