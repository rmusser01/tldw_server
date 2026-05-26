# Research Workspace Canonical Workspace Contract

Date: 2026-05-25

## Decision

Research Workspace is the research UI shell at `/research-workspace`. It does
not own a separate workspace identity model. Any durable workspace identity that
needs sharing, extension handoff, ACP execution, MCP tooling, sandbox policy, or
future agent use must use the canonical workspace ID exposed by the Workspaces
API.

The active canonical workspace source label is `research_workspace`. The old
`workspace_playground` label is legacy stored metadata only; current requests,
responses, route metadata, tests, and UI copy must not emit it. The old
`/workspace-playground`, `/workspace-studio`, and `/research-studio` WebUI pages
are not aliases and are not redirected.

## Ownership Boundaries

| Layer | Owns | Does not own |
| --- | --- | --- |
| Workspaces API | Canonical workspace IDs, source membership, selection context, ingestion/indexing status projection | ACP execution roots, MCP server config, sandbox runtime state |
| Research Workspace UI | Source collection, source selection, chat/studio context, browser-extension capture landing | A forked workspace model or route alias contract |
| Shared Workspaces | User-facing sharing, clone/view/edit permissions, shared workspace discovery | MCP filesystem trust or ACP runtime admission |
| MCP Hub Shared Workspaces | Tool/path trust registry, workspace sets, MCP server/path scoping | Research source ingestion or ACP task lifecycle |
| ACP | Execution workspace roots, projects, tasks, runs, reviews, completion artifacts | Canonical research source truth |
| Sandbox | Admission policy, session/run lifecycle, runtime isolation, diagnostic envelopes | Canonical workspace identity |

## API Contract

- Research Workspace UI and browser-extension handoffs use
  `/api/v1/workspaces/{workspace_id}` for canonical workspace identity.
- Source ingestion writes durable source rows under the canonical workspace ID
  and exposes per-source ingestion, extraction, chunking, embedding, and
  indexing state through the Workspaces API.
- ACP task handoff uses
  `/api/v1/agent-orchestration/workspaces/canonical-bridge` with:

```json
{
  "canonical_workspace_id": "workspace-alpha",
  "canonical_workspace_source": "research_workspace",
  "root_path": "/absolute/execution/root"
}
```

- ACP responses include `canonical_workspace` links using the same canonical ID
  and normalized `research_workspace` source label.
- Research Workspace capability readiness is exposed at
  `/api/v1/research-workspace/capabilities`.

## MCP, ACP, And Sandbox Handoff

1. Browser extension or WebUI capture creates or selects a canonical workspace.
2. Workspaces ingestion/indexing status records whether each source is ready for
   retrieval and citation.
3. Research Workspace passes selected canonical source IDs and workspace ID into
   chat, studio, and agent handoff payloads.
4. ACP bridge links the canonical workspace ID to one execution root when an
   agent task needs filesystem/runtime access.
5. MCP Hub Shared Workspaces remains the canonical path/tool trust registry. It
   may bind its workspace set entries to canonical workspace IDs, but it does
   not replace Research Workspace source membership.
6. Sandbox admission receives canonical workspace ID, ACP workspace ID, and MCP
   workspace scope metadata as policy inputs. Runtime state remains sandbox-owned.

## Deferred Work

- Add the first-class Workspaces source-status projection that joins ingestion,
  extraction, chunking, embedding, indexing, and last error state.
- Add MCP Hub UI affordances that show when a Research Workspace has a matching
  MCP workspace set/path trust entry.
- Add ACP run history and sandbox diagnostics filters keyed by canonical
  workspace ID.
- Add extension capture receipts that deep-link to `/research-workspace` with
  the canonical workspace ID and newly created source IDs.
