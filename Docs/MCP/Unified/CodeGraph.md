# MCP Unified - Native CodeGraph Guide

> Part of the MCP Unified documentation set. See `Docs/MCP/Unified/README.md` for the full guide index.

Native CodeGraph is an optional MCP module that indexes trusted workspace source files into a local SQLite graph. Agents can use it to list indexed files, search symbols, inspect callers and callees, build bounded impact neighborhoods, and request task-oriented context snippets without sending the whole repository into a prompt.

## Current Coverage

CodeGraph currently targets this initial language set:

| Language | Extensions | Parser path |
| --- | --- | --- |
| Python | `.py`, `.pyi` | stdlib `ast` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` | Tree-sitter |
| TypeScript | `.ts`, `.tsx` | Tree-sitter |
| Java | `.java` | Tree-sitter |
| Kotlin | `.kt`, `.kts` | Tree-sitter |
| C# | `.cs` | Tree-sitter |
| C | `.c`, `.h` | Tree-sitter |
| C++ | `.cc`, `.cpp`, `.cxx`, `.hpp`, `.hh`, `.hxx` | Tree-sitter |

Python symbol extraction works without Tree-sitter. Install the `codegraph` extra for full dependency health and the Tree-sitter-backed languages.

## Install Dependencies

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e ".[codegraph]"
```

The pinned optional extra currently installs:

- `tree-sitter>=0.25,<0.26`
- `tree-sitter-python>=0.25,<0.26`
- `tree-sitter-javascript>=0.25,<0.26`
- `tree-sitter-typescript>=0.23,<0.24`
- `tree-sitter-java>=0.23,<0.24`
- `tree-sitter-kotlin>=1.1,<1.2`
- `tree-sitter-c-sharp>=0.23,<0.24`
- `tree-sitter-c>=0.24,<0.25`
- `tree-sitter-cpp>=0.23,<0.24`

Use `codegraph.status` after startup to confirm which parser packages are present. The response includes `dependency_present`, `dependency_missing`, and per-language `dependency_missing` fields.

## Enable The Module

The default module registry includes CodeGraph but leaves it disabled:

```yaml
modules:
  - id: codegraph
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.codegraph_module:CodeGraphModule
    enabled: true
    name: CodeGraph
    version: "0.1.0"
    department: code
    max_concurrent: 4
    settings:
      index_base_dir: Databases/codegraph
      max_file_size_bytes: 1048576
      foreground_max_files: 500
      foreground_max_bytes: 50000000
      max_index_seconds: 20
      max_context_chars: 35000
      max_search_results: 100
```

Edit `tldw_Server_API/Config_Files/mcp_modules.yaml`, or point `MCP_MODULES_CONFIG` at an alternate YAML file. Restart the server after changing module configuration.

### Settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `index_base_dir` | `Databases/codegraph` | Directory where per-workspace `codegraph.db` files are stored. |
| `max_file_size_bytes` | `1048576` | Per-file read limit for indexing and context snippets. |
| `foreground_max_files` | `500` | Default file cap for foreground `index` and `sync`. |
| `foreground_max_bytes` | `50000000` | Total byte cap for foreground `index` and `sync`. |
| `max_index_seconds` | `20` | Wall-clock cap for foreground indexing work. |
| `max_context_chars` | `35000` | Maximum source text returned by `codegraph.context`. |
| `max_search_results` | `100` | Upper bound for search, relationship, and impact result limits. |
| `exclude_dirs` | built-in list | Directory names ignored during discovery, such as `.git`, `node_modules`, `.venv`, `dist`, `build`, `.next`, and `target`. |

The `index_base_dir` is resolved locally. Jobs workers must use the same effective base path as the server process.

## Workspace Scope

CodeGraph never accepts arbitrary filesystem roots from tool arguments. It resolves the active trusted workspace through the MCP Hub workspace root resolver and stores the index under:

```text
<index_base_dir>/<workspace_key>/codegraph.db
```

The `workspace_key` is derived from the resolved user, workspace id, trust source, and workspace root. If the MCP context cannot resolve a trusted workspace, CodeGraph returns `workspace_root_unavailable` or the resolver's reason.

## Tools

| Tool | Mode | Purpose |
| --- | --- | --- |
| `codegraph.status` | read | Inspect dependency health, supported languages, workspace metadata, index presence, counts, and last run. This is read-only and does not create the index DB. |
| `codegraph.index` | write | Full bounded index of the active workspace. Supports `mode`, `force`, `languages`, and `max_files`. |
| `codegraph.sync` | write | Bounded sync of current workspace files. Supports `mode`, `languages`, and `max_files`. |
| `codegraph.files` | read | List indexed files, optionally filtered by path prefix or shell-style pattern. |
| `codegraph.search` | read | Search indexed symbols by query, kind, language, and limit. |
| `codegraph.node` | read | Fetch one indexed symbol by `node_id` or exact `symbol`. |
| `codegraph.callers` | read | List indexed relationships that target a symbol. |
| `codegraph.callees` | read | List indexed relationships emitted by a symbol. |
| `codegraph.impact` | read | Traverse a bounded incoming, outgoing, or bidirectional relationship neighborhood. |
| `codegraph.context` | read | Build task-oriented context with selected symbols, relationships, and bounded source snippets. |

`codegraph.index` and `codegraph.sync` are write-capable because they create or update local index storage. They are blocked when `MCP_DISABLE_WRITE_TOOLS=1` and require the same MCP tool-execution permissions as other write tools.

## Foreground Indexing

Foreground mode is the default for `codegraph.index` and `codegraph.sync`:

```bash
curl -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
        "tool_name": "codegraph.index",
        "arguments": {
          "mode": "foreground",
          "languages": ["python", "typescript"],
          "max_files": 200
        },
        "idempotency_key": "codegraph-index-demo"
      }' \
  http://127.0.0.1:8000/api/v1/mcp/tools/execute
```

Foreground runs are bounded by file count, total bytes, per-file size, and wall-clock time. If the workspace is too large for foreground execution, the result status is `index_too_large_for_foreground`. If the run exceeds the wall-clock cap after starting file work, the result status is `index_timed_out_for_foreground`.

## Jobs-Backed Indexing

Use `mode: "job"` or `mode: "background"` when indexing should be queued through the core Jobs system:

```bash
curl -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
        "tool_name": "codegraph.sync",
        "arguments": {
          "mode": "job",
          "languages": ["python", "javascript", "typescript"]
        },
        "idempotency_key": "codegraph-sync-demo"
      }' \
  http://127.0.0.1:8000/api/v1/mcp/tools/execute
```

A queued response includes `status: "queued"`, `job_id`, `job_uuid`, `job_status`, `queue`, `workspace_key`, and `index_db_path`.

Start a worker in a separate process:

```bash
source .venv/bin/activate
export CODEGRAPH_JOBS_QUEUE=default
export CODEGRAPH_JOBS_INDEX_BASE_DIR="$(pwd)/Databases/codegraph"
python -m tldw_Server_API.app.core.CodeGraph.jobs_worker
```

Relevant worker environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEGRAPH_JOBS_QUEUE` | `default` | Queue used by CodeGraph enqueue and worker code. |
| `CODEGRAPH_JOBS_INDEX_BASE_DIR` | `CODEGRAPH_INDEX_BASE_DIR` or settings default | Trusted local index base used by the worker to validate job payload paths. |
| `CODEGRAPH_JOBS_WORKER_ID` | `codegraph-jobs-<pid>` | Worker id recorded by Jobs leases. |
| `CODEGRAPH_JOBS_LEASE_SECONDS` | `JOBS_LEASE_SECONDS` or `60` | Jobs lease length. |
| `CODEGRAPH_JOBS_RENEW_JITTER_SECONDS` | `JOBS_LEASE_RENEW_JITTER_SECONDS` or `5` | Lease renewal jitter. |
| `CODEGRAPH_JOBS_RENEW_THRESHOLD_SECONDS` | `JOBS_LEASE_RENEW_THRESHOLD_SECONDS` or `10` | Renewal threshold. |

Monitor queued work with the Jobs admin endpoints:

```bash
curl -H "Authorization: Bearer <token>" \
  "http://127.0.0.1:8000/api/v1/jobs/list?domain=codegraph&queue=default&job_type=codegraph_index"
```

```bash
curl -H "Authorization: Bearer <token>" \
  "http://127.0.0.1:8000/api/v1/jobs/stats?domain=codegraph&queue=default&job_type=codegraph_index"
```

## Query Examples

Check status:

```bash
curl -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"codegraph.status","arguments":{}}' \
  http://127.0.0.1:8000/api/v1/mcp/tools/execute
```

Search for a symbol:

```bash
curl -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"codegraph.search","arguments":{"query":"CodeGraphIndexer","language":"python","limit":5}}' \
  http://127.0.0.1:8000/api/v1/mcp/tools/execute
```

Find a symbol's callees:

```bash
curl -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"codegraph.callees","arguments":{"symbol":"CodeGraphIndexer","limit":20}}' \
  http://127.0.0.1:8000/api/v1/mcp/tools/execute
```

Build source context for a task:

```bash
curl -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"tool_name":"codegraph.context","arguments":{"task":"update CodeGraph indexing bounds","max_nodes":8,"max_files":5,"include_code":true}}' \
  http://127.0.0.1:8000/api/v1/mcp/tools/execute
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `codegraph.*` tools are missing | Confirm the module is enabled in `mcp_modules.yaml` or the file selected by `MCP_MODULES_CONFIG`, then restart the server. |
| `dependency_missing` lists Tree-sitter packages | Reinstall the optional extra with `python -m pip install -e ".[codegraph]"` in the same environment that runs the server. |
| A language is listed but no symbols are extracted | Inspect that language's `dependency_missing` field in `codegraph.status`; files for languages with missing parser dependencies are skipped. |
| `workspace_root_unavailable` | Ensure the MCP session has a trusted workspace selected through MCP Hub workspace resolution. CodeGraph does not accept ad hoc root paths. |
| `index_too_large_for_foreground` | Lower `languages` or `max_files`, increase configured foreground bounds, or use Jobs-backed mode. |
| `index_timed_out_for_foreground` | Use Jobs-backed mode or raise `max_index_seconds` after confirming the workspace size is expected. |
| Queued jobs never run | Start `tldw_Server_API.app.core.CodeGraph.jobs_worker`, verify `CODEGRAPH_JOBS_QUEUE`, and confirm the worker `CODEGRAPH_JOBS_INDEX_BASE_DIR` matches the server's effective `index_base_dir`. |
| Read tools return `index_present: false` | Run `codegraph.index` or `codegraph.sync` for the active workspace. `codegraph.status` and read tools do not create storage on their own. |

