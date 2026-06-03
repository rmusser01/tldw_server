# Canonical Workspace Core and Project Workspace Model Design

## Status

Draft for TASK-2232.

## Purpose

tldw_server needs one Workspace model that can grow from a research-only notebook into an agentic project environment. The current codebase already has Research Workspace content APIs, MCP Hub Shared Workspaces, MCP Workspace Sets, ACP session workspace metadata, and Sandbox workspace lineage. Those pieces are useful, but they should not define separate product identities.

This design defines Workspace as the canonical product object. Research Workspace and Project Workspace become profiles or capability states of that object.

## Product Direction

A user should be able to begin with research: sources, media, notes, citations, grounded chat, summaries, and generated artifacts. Later, without migrating to a different product object, the user should be able to attach a project root, manage native files, use Git, run agent harnesses, execute in a sandbox, preview hosted work, and collaborate with a team.

This is the desired progression:

1. Create a research-only Workspace.
2. Add sources, notes, media, chats, citations, and work products.
3. Upgrade the same Workspace into a Project Workspace by attaching a primary root.
4. Use that root as a host-local folder or a sandbox-managed volume.
5. Track native files and Git state.
6. Explicitly choose what file content is indexed into research/RAG.
7. Pass a single runtime context envelope to MCP, ACP/harnesses, and Sandbox.
8. Use sandbox execution to build and test applications.
9. Expose private or team-scoped previews and deployment-like instances.
10. Add team membership, comments, governance, and audit around the same Workspace identity.

External product references that inform this direction:

- GitHub Copilot app: agent-native desktop experience around project work, local/cloud execution, review, and developer workflows.
- Claude Artifacts: generated artifacts that can be iterated, viewed, shared, and used as interactive outputs.
- Codex Sites: saved and deployed app previews with workspace-linked access modes.

## Core Decision

`workspace_id` is the canonical identity for all workspace-related features.

MCP Shared Workspaces, ACP workspace metadata, Sandbox workspace fields, and future agent harness fields must reference the canonical `workspace_id`. They may persist their own operational records, but they do not own a separate Workspace identity.

## Definitions

### Workspace

The canonical product object.

Owns:

- identity
- owner scope
- membership
- high-level lifecycle
- capability state
- activity and commentary
- governance posture
- content collections
- runtime bindings

### Research Workspace

A Workspace with research capabilities enabled and no primary project root.

Owns or references:

- sources
- media items
- notes
- citations
- grounded chat
- generated work products
- migration/import state
- source ingestion and indexing status

### Project Workspace

A Workspace with all Research Workspace capabilities plus a primary project root and agentic runtime bindings.

Owns or references:

- one primary root
- file tree metadata
- optional Git state
- sandbox volume/session/run bindings
- MCP trusted-root and tool-policy bindings
- ACP or adapter-harness runtime bindings
- preview and deploy instance records
- team collaboration and comments over project activity

Project Workspace is not a different top-level object. It is a Workspace whose `workspace_kind` or capability projection indicates `project`.

### Primary Root

The main filesystem root for a Project Workspace.

First-class backend types:

- `host_local`: an existing folder on the tldw_server host.
- `sandbox_volume`: a persistent sandbox-backed project volume.

`git_clone` is an action, not a root backend type. A clone action creates or populates either a `host_local` or `sandbox_volume` root.

### Secondary Roots

Out of scope for the first Project Workspace implementation. Future secondary roots should be represented as attachments or policy members, not peer primary roots. MCP Workspace Sets can still group trusted roots for tool policy, but the Project Workspace user experience starts with one primary root.

### Git Capability

Git is optional state on the primary root.

Git states:

- `absent`
- `initialized`
- `linked_remote`
- `clean`
- `dirty`
- `ahead`
- `behind`
- `diverged`
- `conflicted`
- `unknown`

A plain folder is valid. Git setup can happen later.

### File Inventory

File inventory tracks file metadata, not necessarily file content.

Tracked by default:

- path
- directory/file type
- size
- modified time
- hash when practical
- language or MIME type
- Git status when available
- ignore-policy status
- indexing eligibility

Content indexing is explicit.

### Workspace Runtime Context

The canonical envelope passed to ACP, MCP, Sandbox, and future harness adapters.

It must be stable enough that a new agent harness can consume it without inventing a new Workspace model.

## Conceptual Model

```mermaid
flowchart LR
  W["Workspace\ncanonical identity"] --> R["Research capabilities\nsources, notes, media, artifacts, RAG"]
  W --> P["Project capabilities\nprimary root, files, git"]
  P --> Root["Primary root\nhost_local or sandbox_volume"]
  P --> Files["File inventory\nmetadata first"]
  P --> Git["Git state\noptional"]
  W --> MCP["MCP bindings\ntrusted roots, tools, policies"]
  W --> ACP["ACP/harness bindings\nagent sessions and adapters"]
  W --> SB["Sandbox bindings\nsessions, runs, volumes"]
  W --> Prev["Preview/deploy instances\nprivate or team-scoped"]
  W --> Team["Membership, comments, governance, audit"]
```

## Canonical Workspace Fields

These fields should be common across Research and Project Workspaces:

- `workspace_id`
- `workspace_kind`: `research` or `project`
- `display_name`
- `description`
- `owner_scope_type`: `user`, `team`, `org`, `global`
- `owner_scope_id`
- `created_by`
- `created_at`
- `updated_at`
- `archived`
- `deleted`
- `version`
- `capability_summary`
- `allowed_actions`

`workspace_kind` should be computed from capability bindings where possible. A workspace with a primary root is a Project Workspace. A workspace without one remains research-only.

## Project Root Binding

The first Project Workspace slice should add a primary-root binding.

Conceptual fields:

- `root_id`
- `workspace_id`
- `is_primary`
- `backend`: `host_local` or `sandbox_volume`
- `absolute_root`: for host-local roots, redacted where needed
- `sandbox_volume_id`: for sandbox-managed roots
- `display_name`
- `created_by`
- `created_at`
- `updated_at`
- `git_state`
- `file_inventory_state`
- `indexing_state`
- `sandbox_mount_state`
- `mcp_trust_state`

Rules:

- Exactly one primary root per Project Workspace in the first implementation.
- A research-only Workspace may have zero roots.
- Attaching the first primary root upgrades the Workspace to project capability state.
- Removing or detaching the primary root should either downgrade the Workspace to research-only or leave it as `project` with `project_root_missing`, depending on audit/history needs. The safer first behavior is to keep `project` and mark the root as missing or detached until explicitly downgraded.
- Root paths must be validated through allowlists, sandbox volume ownership, and symlink controls before use.

## File Tracking and Indexing

Project Workspaces should track native files by default but should not index all contents automatically.

Default tracking:

- scan file tree metadata
- honor `.gitignore`
- honor workspace ignore policy
- exclude obvious secrets and generated directories
- detect likely source, docs, assets, and build artifacts

Default indexing:

- no automatic full-content indexing
- source/RAG context only includes files selected, attached, or allowed by explicit indexing policy

Indexing modes:

- `selected_files`
- `docs_and_source_policy`
- `all_allowed_by_policy`

Agents can use MCP/sandbox policy to inspect files when allowed, but grounded research answers should cite only indexed or explicitly attached file content.

## MCP Alignment

MCP Hub Shared Workspaces should become trusted-root bindings for canonical Workspaces.

Current MCP concepts should map as follows:

- Shared Workspace: trusted filesystem root for a canonical `workspace_id`.
- Workspace Set: policy grouping of one or more canonical Workspace ids.
- Path Scope: enforcement layer over trusted root and runtime CWD.
- Policy Assignment Workspace: policy membership keyed by canonical `workspace_id`.

MCP should not define a separate product Workspace. It should answer:

- which tools are available for this Workspace
- which roots are trusted for this Workspace
- which paths are in scope
- which policies govern tool use
- which approvals are required

## ACP and Harness Alignment

ACP sessions and generic harness adapters should receive the same Workspace runtime context.

ACP and harnesses should persist:

- `workspace_id`
- `workspace_group_id` when relevant
- `scope_snapshot_id`
- `sandbox_session_id`
- `sandbox_run_id`
- adapter identity
- policy snapshot metadata
- bounded diagnostics

They should not create independent Workspace identity records.

The runtime envelope must support ACP and non-ACP harnesses. Codex can be the first adapter, but the model must support future Claude Code, OSS harnesses, and partial-ACP adapters.

## Sandbox Alignment

Sandbox uses Workspace identity for tenancy, quotas, lifecycle, and access to project volumes.

Sandbox should answer:

- which persistent project volume is bound to the Workspace
- which sessions and runs are active
- which preview endpoints exist
- which users or teams can access those endpoints
- which filesystem paths are mounted
- which network policy applies

Sandbox should not own Workspace identity.

## Runtime Context Envelope

Every agentic harness should receive a bounded envelope shaped like this:

```json
{
  "workspace_id": "workspace-123",
  "workspace_kind": "project",
  "owner_scope": {
    "type": "user",
    "id": 1
  },
  "access": {
    "role": "owner",
    "allowed_actions": {
      "read_files": true,
      "write_files": true,
      "run_sandbox": true,
      "use_mcp_tools": true,
      "create_preview": true
    }
  },
  "project_root": {
    "root_id": "root-1",
    "backend": "sandbox_volume",
    "display_name": "Website build",
    "path_hint": "[redacted-or-relative]",
    "git_state": "dirty",
    "file_inventory_state": "current",
    "indexing_state": "partial"
  },
  "research": {
    "source_summary": {
      "total": 12,
      "queryable": 9,
      "processing": 1,
      "failed": 0
    }
  },
  "mcp": {
    "state": "available",
    "workspace_set_ids": [],
    "tool_profile_ids": [],
    "policy_assignment_ids": []
  },
  "sandbox": {
    "state": "available",
    "runtime": "docker",
    "volume_id": "volume-123",
    "active_session_id": "sandbox-session-1"
  },
  "agents": {
    "state": "available",
    "harnesses": ["codex"]
  },
  "policy": {
    "scope_snapshot_id": "scope-abc",
    "policy_snapshot_version": "v1",
    "policy_snapshot_fingerprint": "sha256..."
  }
}
```

Sensitive fields must be redacted or replaced with bounded identifiers. Local absolute paths should not appear in user-visible diagnostics unless the endpoint is explicitly privileged and intended for local admins.

## Preview and Deploy Trajectory

Project Workspaces should eventually support private and team-scoped hosted previews.

First-class future concepts:

- `preview_instance_id`
- `workspace_id`
- `sandbox_session_id`
- `sandbox_run_id`
- `origin_url`
- `access_mode`: `owner_private`, `workspace_members`, `team`, `public_link`
- `status`: `starting`, `running`, `stopped`, `failed`
- `created_by`
- `expires_at`
- `comments_enabled`

The first design should leave room for this without requiring a full deploy platform immediately.

## Team Governance Trajectory

Workspace should be ready to move from single-user local ownership to team-based workspaces.

Future membership fields:

- `workspace_id`
- `principal_type`: `user`, `team`, `org`, `service_account`
- `principal_id`
- `role`: `owner`, `admin`, `editor`, `agent_operator`, `viewer`, `commenter`
- `created_by`
- `created_at`

Future activity and comments:

- workspace activity events
- source comments
- file comments
- artifact comments
- preview comments
- agent run comments
- review decisions

Governance should apply to tools, file access, sandbox execution, preview access, and deployment actions.

## API Direction

The existing `/api/v1/workspaces` family should remain canonical.

Potential new or extended endpoints:

- `GET /api/v1/workspaces/{workspace_id}/context`
- `GET /api/v1/workspaces/{workspace_id}/capabilities`
- `GET /api/v1/workspaces/{workspace_id}/roots`
- `POST /api/v1/workspaces/{workspace_id}/roots/primary`
- `PATCH /api/v1/workspaces/{workspace_id}/roots/{root_id}`
- `GET /api/v1/workspaces/{workspace_id}/files`
- `POST /api/v1/workspaces/{workspace_id}/files/indexing-policy`
- `GET /api/v1/workspaces/{workspace_id}/runtime`
- `GET /api/v1/workspaces/{workspace_id}/activity`

MCP, ACP, Sandbox, and future harness APIs should accept or resolve `workspace_id` through the canonical Workspace context service.

## Backend Module Direction

Introduce a core Workspace module that provides identity and context resolution across existing stores.

Suggested module:

- `tldw_Server_API/app/core/Workspaces/models.py`
- `tldw_Server_API/app/core/Workspaces/context.py`
- `tldw_Server_API/app/core/Workspaces/bindings.py`
- `tldw_Server_API/app/core/Workspaces/capabilities.py`

Responsibilities:

- normalize Workspace identity
- resolve Workspace kind
- resolve primary root binding
- resolve file inventory state
- resolve MCP binding state
- resolve ACP/harness state
- resolve sandbox binding state
- build runtime context envelopes
- build capability projections
- centralize reason codes

The module should compose existing persistence layers instead of forcing an immediate physical database merge.

## Existing Module Mapping

| Existing surface | Future role |
| --- | --- |
| `/api/v1/workspaces` | Canonical Workspace API family |
| `workspace_schemas.py` | Canonical Workspace API schema home, expanded carefully |
| `core/Workspaces/status_projection.py` | Keep source readiness, split capability projection into shared module later |
| MCP Shared Workspaces | Trusted-root binding and admin registry |
| MCP Workspace Sets | Policy grouping of canonical Workspace ids |
| MCP Path Scopes | Path enforcement for primary root and trusted roots |
| ACP sessions | Runtime lineage and agent session diagnostics |
| ACP adapter registry | Harness capability provider, not workspace identity |
| Sandbox sessions/runs | Runtime execution and volume lineage |
| Research Workspace UI | Main shell, with project capability mode when primary root exists |
| Chat/Document workspaces | Specialized entry points or modes that should feed canonical Workspace state |
| Prototype Workspace | Separate experimental surface until deliberately migrated |

## Migration Strategy

Do not hard-cut existing MCP/ACP/Sandbox records.

Recommended sequence:

1. Add canonical context resolver and type contracts.
2. Extend capabilities API to include root/file/git/runtime states.
3. Teach MCP Shared Workspaces to optionally link to existing canonical Workspace ids.
4. Add primary root binding with `host_local` and `sandbox_volume`.
5. Add file inventory metadata scanning.
6. Add explicit file indexing policy.
7. Update ACP/harness session creation to resolve the runtime envelope from Workspace core.
8. Update Sandbox project volume/session creation to bind through Workspace core.
9. Add UI affordance to upgrade a research Workspace into a Project Workspace.
10. Add activity/commentary and preview access in later team-oriented slices.

## Implementation Slices

The first implementation should start with a narrow contract slice:

- add core Workspace model types and reason codes
- add read-only context resolver over existing Workspace, MCP, ACP, and Sandbox data
- extend capabilities response with Project Workspace fields in fail-closed states
- add schema for primary root binding
- add tests proving research-only workspaces remain valid
- add tests proving Project Workspace context can represent both `host_local` and `sandbox_volume`
- keep MCP Shared Workspace records as bindings
- keep ACP/Sandbox lineage as runtime records

The first Project Workspace root-creation slice should support both `host_local` and `sandbox_volume` primary roots. Root creation may be separate from the read-only contract slice, but it must not ship as host-local-only if it is presented as the first Project Workspace slice.

Subsequent slices should add:

- file inventory scanning
- explicit indexing policy
- harness runtime-envelope consumption
- preview instances
- team comments and governance workflows

## Security and Privacy Requirements

- Never index all project files by default.
- Honor `.gitignore` and workspace ignore policy before file content indexing.
- Exclude secrets and generated directories by default.
- Redact local absolute paths from non-admin diagnostics.
- Validate host-local roots against allowlists.
- Validate sandbox volumes against owner and workspace binding.
- Treat MCP tools as governed capabilities, not implicit file access.
- Treat sandbox preview URLs as access-controlled resources.
- Preserve audit trails for root attach/detach, indexing policy changes, agent runs, sandbox runs, and preview access changes.

## UX Requirements

The user should not need to understand MCP, ACP, or Sandbox to create a Workspace.

The UI should present:

- "Research Workspace" when no primary root exists.
- "Project Workspace" once a primary root is attached.
- A clear upgrade path: Attach folder, Create sandbox project volume, Clone repository.
- A file tree that is visible before content indexing.
- Explicit indexing controls.
- Git status only when Git is present.
- Agent and sandbox readiness as capabilities with remediation actions.
- Preview/deploy actions only when sandbox/runtime prerequisites are ready.

## Non-Goals

- No forced migration of existing Research Workspace data.
- No removal of ChatWorkspace or DocumentWorkspace routes in this design.
- No multiple peer roots in the first Project Workspace implementation.
- No automatic full-file RAG indexing.
- No public deployment platform in the first implementation slice.
- No immediate physical database consolidation across MCP, ACP, Sandbox, and Workspace stores.

## Open Implementation Questions

These should be answered during planning, not by changing the product model:

1. Whether `workspace_kind` is persisted or computed from bindings.
2. Whether primary root records live in ChaChaNotes DB, Sandbox DB, or a new Workspace DB table.
3. Whether sandbox volumes are created through Workspace API or Sandbox API with a Workspace-bound wrapper.
4. How much file metadata can be scanned synchronously before requiring Jobs.
5. Which role names are needed before team Workspaces ship.

## Design Principles

- One Workspace identity.
- Capabilities attach over time.
- Operational modules keep their specialized storage.
- Runtime modules consume a shared context envelope.
- Research stays useful without project roots.
- Project roots stay useful without Git.
- File metadata is safe by default.
- File content indexing is explicit.
- Team governance is designed in, but not overbuilt early.
