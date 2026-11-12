# CodeGraph

This document describes the code-structure graph: it indexes **trusted workspace source trees only** (MCP Hub workspace roots) into per-workspace SQLite graphs, extracting symbols/imports/calls/references (Python via stdlib `ast`; seven more languages via optional tree-sitter), and serves search/lookup/callers/callees/impact/context as MCP tools. It has **no `/api/v1` surface** — MCP Unified is its only interface.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Settings | `app/core/CodeGraph/config.py` | bounds: 1 MiB per file, 500 files / 50 MB foreground caps, 20 s index budget, 35 k context chars, 100 search results, exclude dirs |
| Workspace trust | `workspace.py` | `CodeGraphWorkspaceResolver` — roots must come from MCP Hub trusted context; the per-workspace index DB lives under `Databases/codegraph/ws_<hash>/` **outside** the workspace, keyed by a hash of user+workspace+trust+root; untrusted resolution raises `workspace_root_unavailable` |
| Indexer | `indexer.py` | `CodeGraphIndexer` — language registry (python + 7 tree-sitter), bounded foreground runs, file-scoped graph replacement with dangling-edge cleanup |
| Resolution | `resolver.py` | cross-file import/call resolution under the same deadline and item limits |
| Context | `context.py` | task-term ranking over indexed nodes with bounded workspace-relative snippets |
| Persistence | `app/core/DB_Management/codegraph/repository.py` + `schema.sql` | files/nodes/edges/unresolved_refs/index_runs tables (+optional FTS); `search_nodes`, callers/callees, bounded impact traversal |
| Jobs | `jobs.py`, `jobs_worker.py` | domain `codegraph`, default queue; the worker is a **separate manual process** (`python -m …jobs_worker`) with WorkerSDK leases; payloads re-validate `index_base_dir`/index paths against local config before writing (trust/anti-SSRF guard) |
| MCP module | `MCP_unified/modules/implementations/codegraph_module.py` | tools `codegraph.{status,index,sync,files,search,node,callers,callees,impact,context}`; capabilities `codegraph.read`/`codegraph.write`; foreground vs job/background modes |
| Models | `models.py` | stable hash node/edge ids, node/edge/unresolved-ref models, run summaries |

## Dataflows

1. **Tool call** → module → workspace resolution (trusted root; raw root-path arguments are never accepted) → the per-workspace repository.
2. **Foreground index/sync** (default): run recorded → candidate discovery → over the 500-file/50 MB caps ⇒ `index_too_large_for_foreground` (use job mode) → per file (skip >1 MiB/binary; extract or inventory-only) → file-scoped replacement → the 20 s wall-clock guard yields `index_timed_out_for_foreground` with partial inventory → on completion, the bounded cross-file resolution pass.
3. **Background index** (`mode: job|background`): a `codegraph`-domain job carries the serialized resolution+settings; the separately-run worker leases it and re-validates the payload paths locally before writing; the tool returns `queued` with the job id.
4. **Reads**: `files/search/node/callers/callees/impact/context` — all return `index_present:false` when the DB file is absent and **never create it as a side effect**; search and impact are bounded (truncation flagged); context ranks nodes against task terms within the char budget.

## Boundaries

- Trust: MCP Hub trusted roots only, revalidated in Jobs payloads — no arbitrary user-supplied roots.
- Storage: per-workspace SQLite under the index base dir (outside workspaces); graph file paths are workspace-relative; not part of per-user Media/Collections DBs.
- Dependencies: Python needs nothing; the other languages need the `codegraph` install extra — missing packages degrade language availability in `status`, never fatal.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Over foreground caps / over time budget | Typed too-large / timed-out statuses (partial inventory kept) |
| Oversized/binary/unreadable file | Skipped with counters |
| Untrusted workspace / Jobs path mismatch | Refusal; no writes |
| Absent index on reads | `index_present:false`, empty results, no auto-create |
| Missing tree-sitter | Languages reported unavailable; Python unaffected |

## Config keys (exact)

Module settings (MCP config, not config.txt): `index_base_dir`, `max_file_size_bytes`, `foreground_max_files/_max_bytes`, `max_index_seconds`, `max_context_chars`, `max_search_results`, `exclude_dirs`. Env (in the MCP doc, **not** `Env_Vars.md`): `CODEGRAPH_JOBS_QUEUE`, `CODEGRAPH_JOBS_INDEX_BASE_DIR` / `CODEGRAPH_INDEX_BASE_DIR`, `CODEGRAPH_JOBS_WORKER_ID`, and the lease/renew family (60/5/10 s).

## Verified gotchas

1. Read tools never create the index — `codegraph.index` (or a job) must run first.
2. `force=True` is currently **ignored** (Stage-1 inventory records it as unused); re-indexing is content-hash driven — don't promise force semantics.
3. The Jobs worker has no in-app lifespan wiring — it's a manual runbook process.
4. Large repos legitimately finish `truncated` — cross-file resolution shares the 20 s budget.
5. The same repo mounted under a different workspace id or trust source gets a separate index DB.
6. The module README's "no MCP tool coverage" note is mildly stale — MCP-level tool tests exist.

## Governing decisions and docs

No CodeGraph ADR; adjacent ADR-032/033 (MCP stdio contracts). Current: the module README and `Docs/MCP/Unified/CodeGraph.md` (install extra, tool contract, worker runbook, Jobs monitoring) — the module is unchanged since June. Tests: `tests/CodeGraph/` plus the MCP-level module tests.

## Related docs

- [mcp-unified.md](./mcp-unified.md) — the only surface CodeGraph exposes
- [jobs-background.md](./jobs-background.md) — the substrate for background indexing
- [agent-orchestration-acp.md](./agent-orchestration-acp.md) — the workspace trust neighborhood
