# Workspace Container Contract

Date: 2026-06-17
Status: Canonical Phase 2 contract
Tracking: [#1984](https://github.com/rmusser01/tldw_server/issues/1984), [#1988](https://github.com/rmusser01/tldw_server/issues/1988)

## Decision

`Workspace` is the durable single-user operating context that existing tldw
capabilities attach to. It is not primarily a new all-in-one Workspace UI and
must not become a duplicate Library, Notes, Chat, ACP, MCP, Sandbox, Jobs, or
Workflows surface.

The contract is aligned with the sibling `tldw_chatbook` workspace
operating-context model documented in the
[Workspace Operating Context And Handoff PRD](https://github.com/rmusser01/tldw_chatbook/blob/main/Docs/superpowers/specs/2026-05-20-workspace-operating-context-handoff-prd-design.md).
That model defines four layers that this server contract adopts:

1. Workspace Registry: durable workspace records, membership tags, authority
   state, sync state, and runtime bindings.
2. Active Workspace Context: the currently selected operating workspace used by
   staging, RAG grounding, tools, ACP, Sandbox, workflows, and watchlists.
3. Global Item Browser: user-owned resources remain globally visible with
   workspace tags and eligibility status.
4. Context Eligibility Gate: workspace-sensitive actions require the resource
   to belong to the active workspace, or to be explicitly linked/copied into it.

Workspace selection must not hide globally owned resources. Notes, media,
artifacts, chats, prompts, workflows, watchlists, and other user-owned records
remain browsable/searchable according to their owning domain permissions.
Workspace membership only determines container association, provenance display,
and active-context eligibility.

## Non-Goals

- Do not implement new services or UI in this contract slice.
- Do not introduce multi-user sharing, collaborator roles, external links, or
  stakeholder workflows.
- Do not move ACP execution, MCP trust, Sandbox runtime admission, Jobs
  execution, or domain-owned resource CRUD into the Workspace module.
- Do not use workspace selection as a hard global filter for Library, Notes,
  Media, Artifacts, Chat, Prompts, Workflows, or Watchlists.
- Do not persist raw secrets, private environment values, or unrestricted
  filesystem paths in Workspace membership or public runtime-binding metadata.

## Core Vocabulary

| Term | Contract |
| --- | --- |
| Workspace | Durable operating context identified by a canonical `workspace_id`. |
| Workspace Registry | The set of workspace records plus lifecycle, authority, membership, import, and runtime binding metadata. |
| Active Workspace Context | The selected workspace used for workspace-sensitive operations. |
| Global Item Browser | Existing owner surfaces that list/search/open/edit all resources visible to the user. |
| Context Eligibility Gate | Decision layer for whether a visible resource may be staged, grounded, manipulated, or executed in the active workspace. |
| Membership | Association between one workspace and one resource owned by another domain. |
| Runtime Binding | Secret-safe descriptor for repo/path/worktree/ACP/Sandbox/MCP runtime state associated with a workspace. |
| Transfer Policy | Explicit policy for how a resource or binding participates in copy/reference/import/handoff operations. |

## Workspace Identity And Lifecycle

The canonical server workspace record is exposed through
`/api/v1/workspaces/{workspace_id}`. `workspace_id` is the product identity used
by Research Workspace, Agent Tasks filters, ACP canonical bridge metadata,
workspace memberships, workspace roots, source status, and future workspace
activity/index views.

The current backend has these implemented identity fields:

| Field | Status | Contract |
| --- | --- | --- |
| `workspace_id` | Current | Stable string identifier. This is the product workspace ID. |
| `name` | Current | User-facing display name. |
| `workspace_profile` | Current | Persisted intent. Current values: `research`, `project`. |
| `workspace_kind` | Current read alias | Compatibility/display alias derived from `workspace_profile`; not a second source of truth. |
| `archived` | Current | Archived workspaces remain inspectable but cannot be modified or used for active-context operations. |
| `deleted` | Current | Soft-delete state. Deleted records are excluded from normal lists and preserved only as required for recovery/audit. |
| `version` | Current | Optimistic locking version. |
| `created_at`, `last_modified` | Current | Lifecycle timestamps. |

Phase 2 implementations must treat lifecycle as follows:

- Create and import produce a workspace record before resource memberships or
  runtime bindings are attached.
- Rename and metadata edits preserve `workspace_id`.
- Archive is reversible and blocks membership mutation, active-context
  eligibility, ACP runs, Sandbox operations, workflow launches, and watchlist
  runs.
- Delete is a soft delete for the workspace record and workspace-owned
  sub-resources. Domain-owned global resources are not deleted merely because a
  membership is removed or a workspace is deleted.
- Recreate after delete must not reuse deleted lineage silently. It must either
  restore the record through a recovery path or create a new workspace with new
  provenance.

## Authority, Status, And Import Provenance

Chatbook uses authority values to explain whether a workspace is local-only,
server-backed, syncing, conflicted, detached, remote-only, or missing runtime.
tldw_server Phase 2 remains single-user, but the same vocabulary is useful for
imports, future handoff, and client UX.

Canonical authority vocabulary:

| Authority | Meaning |
| --- | --- |
| `local-only` | Workspace exists only in the local/client context or imported metadata. |
| `server-backed` | Workspace has a canonical server record and can be resumed by server ID. |
| `syncing-to-server` | Local workspace package is being materialized into a server record. |
| `syncing-from-server` | Server workspace package is being materialized locally. |
| `conflict` | Local and server identity/version/provenance disagree and need user choice. |
| `detached` | A previously backed workspace cannot verify its server identity. |
| `remote-only` | Server workspace exists but is not materialized locally. |
| `runtime-missing` | Workspace metadata exists, but an ACP/Sandbox/path/runtime binding cannot be restored. |

Current backend status fields should map into that authority vocabulary without
claiming full sync. Existing `attention_state` values are `ready`,
`setup_pending`, `working`, `needs_attention`, `blocked`, and `archived`.
Existing resolution status values are `complete`, `partial`, and `failed`.

Import provenance must be explicit and secret-safe. Import-capable slices should
use bounded JSON metadata with fields such as:

- `import_source`: `repo`, `worktree`, `chatbook_manifest`, `workspace_record`,
  `acp_package`, or `manual`.
- `source_workspace_id`: source workspace ID when importing from another
  workspace record.
- `source_manifest_version`: manifest/schema version when importing a package.
- `source_repo_hint` and `source_root_hint`: redacted labels or path hints, not
  absolute private paths unless the endpoint is explicitly privileged.
- `imported_at`, `imported_by_surface`, and `handoff_history`.
- `redaction_report`: summary of omitted fields, omitted secrets, local-only
  bindings, and unsupported resources.

## Membership Contract

`workspace_resource_memberships` is the generic Workspace-to-resource
association layer. It does not transfer domain ownership, hide global records,
grant runtime trust, or replace resource-specific access checks.

The canonical membership shape is:

```json
{
  "workspace_id": "workspace-alpha",
  "resource_type": "media",
  "resource_id": "123",
  "role": "source",
  "label": "Optional display label",
  "transfer_policy": "link",
  "provenance": {},
  "metadata": {},
  "summary": {
    "title": "Display title",
    "subtitle": "Domain label",
    "href": "/media/123",
    "state": "available",
    "metadata": {}
  },
  "created_at": "2026-06-17T00:00:00Z",
  "updated_at": "2026-06-17T00:00:00Z",
  "version": 1,
  "deleted": false
}
```

Membership invariants:

- The owning domain validates visibility/access before a membership is created
  or resolved.
- Unsupported resource types fail closed.
- Membership rows are idempotent for the same
  `(workspace_id, resource_type, resource_id)` active association.
- Membership deletion is a soft unlink and must not delete the domain-owned
  resource.
- Summaries and metadata must avoid raw secrets, unredacted absolute paths,
  sandbox mount paths, prompt bodies, model outputs, file contents, API keys,
  tokens, or credentials.
- Backfills must be explicit and idempotent. Startup must not silently attach
  legacy rows to the generic membership table.

Current membership roles are `member`, `source`, `artifact`, `conversation`,
`runtime`, and `reference`.

## Resource Types

Phase 2 names these resource types and membership expectations:

| Resource | Current/Future wire value | Owning domain | Expected membership shape |
| --- | --- | --- | --- |
| Workspace notes | `workspace_note` | Workspaces/ChaChaNotes | Workspace-owned note row; integer resource ID; role `member` or `reference`. |
| Global notes | `note` (future) | Notes/ChaChaNotes | Domain-owned note visible globally; workspace link gates active-context use but does not hide note browsing. |
| Media and sources | `media`, `workspace_source` | Media DB and Workspaces source selection | `media` links global media records; `workspace_source` links workspace source rows that own ordering, selection, and readiness projection. |
| Artifacts | `workspace_artifact` | Workspaces artifact store | Workspace-owned generated or promoted artifact; includes lineage/review/export metadata. |
| Chats | `chat` | Chat/ChaChaNotes | Conversation resource. Global chats may be linked; workspace-scoped chats belong to exactly one workspace. Moving a chat across workspaces is a fork/copy with provenance, not silent reassignment. |
| Prompts | `prompt` | Prompt Library | Prompt record stays globally browsable; workspace membership gates active use in the selected context. |
| Workflows | `workflow` | Workflows/Scheduler | Workflow definition or launch context; launch requires active workspace and runtime readiness when applicable. |
| Watchlists | `watchlist` | Watchlists/Jobs or Scheduler | Watchlist definition or run context; run requires active workspace and runtime readiness when applicable. |
| ACP sessions | `acp_session` | ACP/Agent Orchestration | Links canonical workspace to ACP session metadata through a Workspace runtime binding. ACP remains execution owner. |
| ACP runs | `acp_run` (future) | ACP/Agent Orchestration | Reserved for run-level metadata after ACP run descriptors have a stable owning-domain contract. |
| Sandbox sessions | `sandbox_session` | Sandbox | Links workspace to sandbox session diagnostics through a Workspace runtime binding. Sandbox remains runtime/admission owner. |
| Project files | `project_file` (future) | Workspaces file inventory | Metadata-only path entry under a Workspace-owned root. Not source content and not a trust grant. |

The current backend supports `workspace_note`, `media`, `workspace_source`,
`workspace_artifact`, `chat`, `prompt`, `workflow`, `watchlist`,
`acp_session`, and `sandbox_session`. `membership_models.py` still reserves
future values for `note`, `acp_run`, `project_file`, `study_deck`, `quiz`, and
`study_pack`. Future child issues must add owning-domain adapters before those
reserved values become writable.

## Transfer Policies

The product vocabulary from Chatbook is `copy`, `reference`, `metadata-only`,
and `local-only`. The current server wire values are `link`, `copy`, `promote`,
and `import`. Phase 2 must map these deliberately:

| Product policy | Server value | Contract |
| --- | --- | --- |
| `reference` | `link` | Store a membership reference to an existing domain-owned resource. No ownership transfer. |
| `copy` | `copy` | Create an independent resource or package in the target workspace/domain with provenance back to the source. |
| Artifact promotion | `promote` | Convert accepted/generated execution output into a traceable workspace artifact. |
| Import/package materialization | `import` | Create workspace records, memberships, or sub-resources from an external repo/workspace/package source. |
| `metadata-only` | Future profile or transfer metadata | Store descriptors, provenance, status, and redaction report without copying source content. |
| `local-only` | Future profile or transfer metadata | Mark a membership or runtime binding as not portable to server/ACP/Sandbox handoff. |

No Phase 2 flow may silently "move" a resource between workspaces. A move-like
UX must be implemented as copy/link/fork plus explicit unlink, with provenance
and recovery messaging.

## Active Context Eligibility

Active-context eligibility is separate from browse/search visibility.

Visibility operations:

- `browse`
- `search`
- `open`
- `edit`

These remain allowed when normal domain permissions pass, even if no workspace
is active or the resource belongs to another workspace.

Active-context operations:

- `stage`
- `rag_ground`
- `prompt_use`
- `tool_use`
- `agent_manipulate`
- `acp_run`
- `sandbox_operation`
- `workflow_launch`
- `watchlist_run`

These require:

1. An active workspace ID.
2. The active workspace exists and is not archived.
3. The resource type is supported by a membership adapter.
4. The resource has an active membership in the active workspace.
5. The caller reports `permission_state="granted"`.
6. Runtime-bound operations report `runtime_state="ready"`.

Stable denial reason codes include `no_active_workspace`,
`workspace_not_found`, `workspace_archived`, `unsupported_resource_type`,
`resource_not_linked`, `cross_workspace_resource`, `missing_runtime`, and
`permission_denied`.

The eligibility gate is not a trust source for MCP, ACP, Sandbox path admission,
filesystem access, or execution environment access. Those systems must run
their own root/path/runtime/admission checks after Workspace eligibility passes.

## Runtime Binding Vocabulary

Runtime bindings describe executable or filesystem-adjacent state associated
with a workspace. They are metadata-first and secret-safe. They explain what
runtime context exists or is missing; they do not grant access by themselves.

Canonical descriptor fields:

| Field | Contract |
| --- | --- |
| `binding_id` | Stable descriptor ID. |
| `workspace_id` | Canonical product workspace ID. |
| `binding_kind` | Kind from the vocabulary below. |
| `owner_domain` | `workspaces`, `acp`, `sandbox`, `mcp`, `jobs`, or another owning domain. |
| `locator_ref` | Domain-owned opaque ID or safe reference. |
| `label` | User-facing label. |
| `status` | Readiness/status value. |
| `path_hint` | Redacted basename, repo label, or safe display hint. |
| `portability` | `reference`, `metadata-only`, `local-only`, or `copy` where supported. |
| `metadata` | Bounded JSON without secrets or private content. |
| `redaction_report` | Summary of omitted sensitive fields. |
| `created_at`, `updated_at` | Descriptor timestamps. |

Binding kinds:

- `repo`
- `git_worktree`
- `local_path`
- `workspace_project_root`
- `acp_execution_workspace`
- `acp_session`
- `acp_run`
- `sandbox_root`
- `sandbox_session`
- `mcp_workspace_set`

Readiness/status vocabulary:

- `ready`
- `missing`
- `inspect-only`
- `blocked`
- `provisioning`
- `unavailable`
- `detached`
- `conflict`
- `runtime-missing`

Current backend equivalents include Workspace root states
`not_configured`, `provisioning`, `attached`, `unavailable`, `missing`,
`detached`, `failed`, `cleanup_pending`, and `archived`; ACP workspace health
states such as `healthy`, `degraded`, and `missing`; and Sandbox mount states.
Child issue #1991 owns the durable runtime binding descriptor implementation and
must define exact wire schemas from these vocabulary values.

Secret handling:

- Runtime binding metadata must not embed API keys, tokens, passwords, private
  keys, raw environment values, server credentials, prompt contents, model
  outputs, or file contents.
- Public workspace responses must prefer `path_hint` over absolute paths.
- ACP `env_vars` are operational plaintext in the ACP execution DB today. They
  must not be copied into Workspace membership metadata or represented as a
  secret store.
- Recreating executable bindings from import/handoff requires explicit user
  approval in the owning runtime domain.

## Existing Surface Mapping

| Surface | Mapping |
| --- | --- |
| `/api/v1/workspaces` | Canonical Workspace CRUD, metadata, sources, notes, artifacts, roots, capabilities, context, and membership read/write routes. |
| `/api/v1/workspace-eligibility/check` | Shared active-context eligibility contract. |
| `/api/v1/workspaces/{id}/memberships` | Generic association table for supported resource types. |
| `/api/v1/workspaces/{id}/context` | Read model for Workspace Core shell state, source summary, capabilities, active operations, and membership summary. |
| `/api/v1/workspaces/{id}/runtime-bindings` | Secret-safe runtime binding descriptor list/create/read/archive routes for repo/path/ACP/Sandbox/MCP-adjacent state. |
| `/api/v1/workspaces/{id}/roots` | Workspace-owned project root read model with redacted path hints. |
| `/api/v1/workspaces/{id}/file-inventory/*` | Metadata-only project-root inventory. Not source content indexing and not a trust grant. |
| Sharing | Out of Phase 2 scope. Future sharing must consume the same `workspace_id` and membership model rather than inventing a parallel shared workspace identity. |
| ACP | `/api/v1/agent-orchestration/workspaces/canonical-bridge` links an ACP execution workspace to the canonical workspace ID with `canonical_workspace_source="research_workspace"`. ACP still owns projects, tasks, runs, reviews, execution roots, MCP injection, and env. |
| Sandbox | Workspace root provisioning may call Sandbox-owned services. Sandbox still owns admission, session/run lifecycle, mount readiness, isolation, and diagnostics. |
| Jobs | Workspace source ingestion and file inventory enqueue Jobs work. Jobs remains the queue/execution owner. |
| Frontend stores | `apps/packages/ui/src/store/workspace.ts` is browser cache/local state. Existing surfaces should read canonical workspace context and eligibility instead of inventing per-surface semantics. |
| Workspaces Manager UI | Management/status surface over the registry, roots, metadata, and recovery states. It must not replace domain owner UIs. |
| Research Workspace | First user-facing shell for broad research context, using canonical `workspace_id`. |
| Agent Tasks and ACP Playground | Execution/detail surfaces that may filter by canonical workspace ID and show setup gaps, but do not own product workspace identity. |

## Activity And Index Contract

`GET /api/v1/workspaces/{workspace_id}/index` is the concrete #1994
inspection/navigation contract. It is a server-owned read model over existing
Workspace registry data and owner-domain membership summaries. It is not a new
Workspace dashboard and must not implement edit/detail behavior that belongs to
Media, Notes, Chat, Prompts, Workflows, Watchlists, ACP, Sandbox, MCP, or Jobs.

The response shape is versioned with `schema_version: 1` and includes:

- `workspace`: workspace identity, profile, archive/delete state, and version.
- `membership_summary`: active membership totals by resource type and role.
- `resource_groups`: bounded resolved membership previews grouped by resource
  type. Each group includes an `owner_surface` `{label, href}` supplied by the
  server so clients can navigate back to the owning UI.
- `runtime_summary`: descriptor-only runtime binding totals, status counts, and
  redacted binding payloads.
- `warnings`: recovery/navigation hints for archived/deleted workspaces,
  unresolved resource previews, and missing/degraded runtime bindings.
- `recent_activity`: bounded newest-first activity rows.
- `partial_errors`: reserved list for future dependency-specific partial read
  failures.

Current activity categories and event types:

| Category | Event types | Source |
| --- | --- | --- |
| `membership` | `membership.linked`, `membership.restored`, `membership.unlinked` | Workspace membership service write paths. |
| `runtime_binding` | `runtime_binding.upserted`, `runtime_binding.archived` | Runtime binding endpoint helper write paths. |

Current warning reason codes:

| Reason code | Meaning |
| --- | --- |
| `workspace_archived` | Workspace is inspectable but write/active-context actions are blocked. |
| `workspace_deleted` | Deleted workspace state is exposed only for recovery-safe inspection. |
| `resource_unresolved` | A linked owner-domain resource could not be resolved for preview. |
| `resource_<state>` | A resolved resource preview reported a non-available state. |
| `runtime_binding_missing` | A runtime binding points at a missing runtime/path/session. |
| `runtime_binding_<status>` | A runtime binding reported another degraded status such as blocked, detached, conflict, unavailable, or unsupported. |

Security constraints:

- Activity metadata is bounded JSON and must omit secret-shaped keys, API keys,
  tokens, passwords, private keys, raw env values, prompt bodies, model outputs,
  file contents, and unredacted absolute paths.
- Runtime binding rows in the index use the same redacted descriptor contract as
  `/runtime-bindings`; the index must not become a secret store or runtime
  admission authority.
- Resource previews are summaries only. Clients must open the owner `href` for
  full content, editing, execution, or recovery actions.
- Unknown warning reason codes are forward-compatible and should be displayed as
  generic warnings by clients rather than rejected.

## Open Questions And Assigned Follow-Ups

All known open implementation questions are assigned to child issues:

| Question | Owner issue |
| --- | --- |
| Runtime consumer resume/import semantics beyond the descriptor read model | Follow-up after [#1991](https://github.com/rmusser01/tldw_server/issues/1991) evidence |
| Owning-domain adapters for global notes, ACP runs, project files, and study-related resource types | Follow-up after [#1995](https://github.com/rmusser01/tldw_server/issues/1995) evidence |
| Frontend route/store contract for active workspace context, global browsing labels, and eligibility UX | [#1993](https://github.com/rmusser01/tldw_server/issues/1993) |
| Workspace activity/index contract and inspection UI | [#1994](https://github.com/rmusser01/tldw_server/issues/1994) |
| End-to-end single-user evidence across create/import, attach, runtime context, activity, and recovery | [#1995](https://github.com/rmusser01/tldw_server/issues/1995) |

Any new ambiguity discovered during implementation should be added to one of
these issues or split into a new child issue before changing wire semantics.

## Implementation Guardrails

- Match existing Workspace Core literals before adding new wire values.
- Add fail-closed adapters before accepting new membership resource types.
- Preserve global visibility in owner domains.
- Keep runtime trust and path admission in ACP, MCP, and Sandbox.
- Keep file inventory metadata-only unless a separate indexing policy explicitly
  authorizes content extraction.
- Use bounded JSON and explicit redaction reports for provenance and runtime
  metadata.
- Prefer additive migrations and compatibility fields over replacing existing
  `workspace_profile`, `workspace_kind`, membership, or ACP bridge fields.
