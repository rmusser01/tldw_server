# Research Workspace Hard Replacement Roadmap Design

Date: 2026-05-23
Status: Draft for review
Backlog: TASK-463

## Summary

This design replaces the legacy Workspace Playground with a canonical,
server-backed Research Workspace at `/research-workspace`.

This is a hard replacement:

- `/research-workspace` is the only user-facing Research Workspace route.
- `/workspace-playground` is removed and returns the normal 404.
- There are no redirects, aliases, feature-flag fallback routes, or legacy
  route compatibility paths.
- `/research` remains a separate Deep Research product surface.

Research Workspace is the research-facing shell over the project's unified
Workspace model. Workspace is not only a notebook of sources. It is the durable
boundary for research content, ingestion and indexing, sharing, MCP tool access,
ACP agent orchestration, sandboxed execution, migration, and governance.

The roadmap order is:

1. Phase A: NotebookLM migrant first value.
2. Phase D: trust and transparency.
3. Phase B: experienced power-user workflow.
4. Phase C: browser extension capture loop.

## Context And Verified Current State

The current NotebookLM-style three-pane surface is implemented as
`/workspace-playground`. Its user-facing page title is "New Research", while
navigation and code also use labels such as Workspace Playground and Research
Studio. This creates route and terminology ambiguity for users.

The backend already exposes `/api/v1/workspaces` with workspace, source, note,
and artifact subresources. Current workspace source responses include identity,
media id, title, source type, URL, order, selection, timestamps, and version.
They do not expose first-class ingestion, extraction, chunking, embedding,
indexing, or queryability status.

The frontend store already models local source statuses such as processing,
ready, and error, but local state is not a sufficient source of truth for a
server-backed Research Workspace.

Live validation was performed against a healthy backend at
`http://127.0.0.1:8000/api/v1/health`. A correctly configured WebUI dev server
was started with:

```bash
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced \
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 \
NEXT_PUBLIC_API_VERSION=v1 \
bun run dev -- -H 127.0.0.1 -p 18080
```

`http://127.0.0.1:18080/workspace-playground` loaded the existing three-pane
workspace shell against the backend.

## Goals

- Make `/research-workspace` the canonical route and product name.
- Remove `/workspace-playground` without redirects or aliases.
- Make server state the source of truth for workspace data.
- Align Research Workspace with the larger unified Workspace model, including
  sharing, MCP, ACP, sandboxes, and governance.
- Automatically migrate legacy local workspace data into server-backed
  workspaces as a true move.
- Prevent local data deletion until server migration receipt, verification, and
  deletion eligibility are complete.
- Add first-class, Jobs-backed source ingestion and indexing status.
- Give first-time users a clear path from sources to grounded answers.
- Give power users scalable source management and evidence workflows.
- Make the browser extension save into the same canonical workspace identity.

## Non-Goals

- Merging `/research` into Research Workspace.
- Keeping `/workspace-playground` as an alias, redirect, or hidden fallback.
- Creating a parallel research-only workspace model.
- Replacing the existing sharing, MCP Hub, ACP, or sandbox domains with a
  monolithic workspace service.
- Shipping all MCP, ACP, and sandbox controls in Phase A.

## Glossary

Workspace:
The canonical durable boundary for research content, sharing, tools, agents,
approvals, sandbox execution, and governance.

Research Workspace:
The user-facing research shell at `/research-workspace`. It exposes source
collection, grounded chat, evidence, notes, outputs, and workspace capability
state.

Deep Research:
The separate `/research` run-oriented product surface for long-running research
runs. It is not merged with Research Workspace.

Shared Workspace:
The sharing and collaboration layer around a Workspace. It controls share
records, shared-with-me access, clone/fork, access levels, and cross-user DB
resolution.

MCP Hub Shared Workspace:
The MCP control-plane representation that exposes workspace boundaries to tools
and agents. It is part of the Workspace model, but it is not the same as the
research content API.

Workspace Set:
A governance/control-plane grouping of workspaces used by MCP and tool policy.

Path Scope:
A filesystem or resource boundary used to govern tool and sandbox access.

ACP:
Agent orchestration and approvals over a workspace-scoped runtime.

Sandbox:
The execution environment and policy boundary for tools or agents operating
against workspace-scoped resources.

## Product Boundary And Route Policy

`/research-workspace` becomes the only canonical Research Workspace route.
User-facing labels, route metadata, navigation, tests, docs, screenshots,
tutorials, and telemetry names should use Research Workspace where practical.

`/workspace-playground` is removed entirely. It should return the normal 404.
There should be no redirect, alias, compatibility route, hidden feature flag, or
production fallback path.

`/research` remains Deep Research. Research Workspace may later launch or
reference Deep Research runs, but the two products remain distinct.

## Unified Workspace Model

Research Workspace is the research-facing UI over the unified Workspace model.
A Workspace includes these first-class dimensions:

- Research content: sources, folders, tags, selected source sets, chats, notes,
  outputs, evidence lineage, and citations.
- Ingestion and indexing: source processing state, extraction, chunking,
  embeddings, queryability, failure recovery, and retries.
- Sharing and collaboration: private/shared/cloned state, share access levels,
  shared-with-me flows, clone/fork, and owner/accessor DB resolution.
- MCP and tool access: MCP Hub Shared Workspaces, workspace sets, path scopes,
  tool availability, governance policy, and capability visibility.
- ACP and agent orchestration: agent sessions, approvals, task progress,
  workspace-scoped runtime state, and tool-call history.
- Sandbox execution: sandbox policy, filesystem/path scope, execution risk,
  active sandbox sessions, and tool-produced artifacts.
- Migration and durability: automatic local-to-server true move, server-side
  migration receipt, recovery manifest, verification result, and deletion
  acknowledgement.

The implementation should not turn `/api/v1/workspaces` into a giant endpoint
that owns every subsystem. The design uses Workspace Core plus capability
providers:

- Workspace Core owns identity, metadata, content subresources, source
  placement, migration, and source status projection.
- Sharing owns access, shared-with-me, clone/fork, and cross-user DB
  resolution.
- MCP owns tool-control-plane exposure, workspace sets, path scopes, and
  governance.
- ACP owns agent sessions, approvals, task progress, and tool-call history.
- Sandbox owns execution policy, runtime isolation, and filesystem/path access.
- Provider/model services own model availability and external-provider state.

Research Workspace consumes a workspace capability projection that aggregates
these domains without coupling the UI to each subsystem's internal data shape.

## Server-Backed Source Of Truth

Server state is the source of truth for Research Workspace. Browser local
storage becomes cache, optimistic UI, draft state, and temporary recovery state.

Server-owned objects include:

- workspace identity and metadata;
- sources, folders, tags, selection, and order;
- source status projection;
- notes, chats, artifacts, outputs, and evidence lineage;
- migration receipt and recovery manifest;
- workspace capability projection;
- sharing and access metadata through the sharing layer;
- MCP, ACP, and sandbox readiness through capability providers.

Local-only state should be limited to:

- pane widths and layout preferences;
- active tab or focused item;
- local draft text;
- temporary optimistic state;
- migration handoff state before server eligibility;
- a non-content migration tombstone after local deletion.

## Source Status Contract

Workspace source status must be authoritative server state. The frontend should
not treat local `processing`, `ready`, or `error` values as truth.

Status should be derived from media, Jobs, extraction, chunking, embeddings,
indexes, permissions, and queryability. Static workspace source columns may
cache the latest status, but the status projection must remain authoritative.

Source lifecycle states:

- `queued`
- `ingesting`
- `extracting`
- `chunking`
- `indexing`
- `queryable`
- `partially_queryable`
- `failed`
- `retrying`
- `missing_media`
- `blocked_by_permissions`

Readiness dimensions:

- `metadata_ready`
- `text_extracted`
- `fts_ready`
- `vector_ready`
- `citation_ready`
- `summary_ready`
- `tool_accessible`

The status projection should include:

- current lifecycle state;
- readiness dimensions;
- associated job id or job ids;
- progress where available;
- failure code and user-facing message;
- retry eligibility;
- last refreshed timestamp;
- stale status indicator;
- source of truth used to compute the status.

Grounded chat can use only `queryable` sources by default. It may use
`partially_queryable` sources only when the user explicitly accepts the reduced
readiness.

## Jobs-Backed Ingestion And Indexing

Source ingestion and indexing are user-visible work. They need status, retry,
partial-success handling, quotas, and admin visibility. Per the project Jobs vs
Scheduler guidance, these flows should use Jobs where appropriate.

Adding a source should create or attach to Jobs for ingestion, extraction,
chunking, and indexing as needed. Research Workspace should display job-backed
status instead of generic local progress.

## Workspace Capability Projection

Research Workspace needs a summarized capability/status contract for the whole
workspace. This projection prevents the UI from directly importing internal
MCP, ACP, sandbox, sharing, and provider shapes.

Candidate endpoint:

```text
GET /api/v1/workspaces/{workspace_id}/capabilities
```

The projection should include:

- content readiness;
- source status summary;
- sharing state and effective access tier;
- MCP tool availability and governance state;
- ACP agent readiness, active sessions, approval requirements, and blocked
  state;
- sandbox policy, path scope, active sessions, and execution risk;
- provider/model readiness;
- external-provider privacy warnings;
- effective allowed actions with blocked or needs-approval reasons.

The design requires one effective workspace capability resolver so the UI does
not show contradictory affordances across sharing, MCP, ACP, sandbox, and model
availability.

### Phase A Minimum Capability Contract

Phase A must ship a small authoritative capability contract. It should not wait
for the full trust/transparency work in Phase D, and it should not expose
placeholder states that cannot be computed by the server.

Minimum Phase A fields:

- `workspace_id`
- `workspace_kind`: private, shared, cloned, or imported
- `effective_access_level`: owner, `view_chat`, `view_chat_add`, `full_edit`,
  or blocked
- `source_summary`: total, queryable, partially queryable, processing, failed,
  missing media, and blocked counts
- `migration_state`: none, pending, failed, or complete
- `sharing_state`: private, shared_by_me, shared_with_me, or cloned
- `mcp_state`: not configured, available, blocked, or unknown
- `acp_state`: not configured, available, needs approval, blocked, or unknown
- `sandbox_state`: not configured, available, blocked, or unknown
- `provider_state`: chat ready, RAG ready, degraded, unavailable, and external
  provider warning
- `allowed_actions`: action-specific booleans and reason codes for adding
  sources, editing sources, deleting sources, asking grounded questions,
  generating outputs, sharing, using tools, starting agents, and running
  sandboxed actions

Phase A UI should render this as a compact status panel. It should show
"available", "not configured", "blocked", "needs approval", "degraded", or
"unknown" without trying to expose every subsystem detail.

`unknown` is allowed only when the relevant capability provider is unreachable,
disabled by deployment configuration, or has not yet reported a status after a
bounded refresh attempt. It must not be used as the default value for an
unimplemented capability provider.

Phase D expands the same contract with drill-down data: detailed policy reasons,
tool lists, workspace sets, path scopes, active ACP sessions, approval history,
sandbox sessions, migration receipt details, and provider/model diagnostics.

## Access Enforcement

Every workspace mutation must be server-authorized against the effective
workspace context for that specific action:

- owner/private workspace;
- shared workspace access tier;
- cloned/forked workspace state;
- MCP/path-scope policy;
- ACP approval state;
- sandbox policy;
- provider/model availability.

This enforcement is action-specific. For example, renaming a workspace should
not depend on model/provider readiness, and organizing sources should not depend
on sandbox availability. Tool execution, agent starts, grounded chat, source
mutation, sharing, and sandboxed actions each resolve only the policies relevant
to that action. The frontend may hide or disable unavailable actions, but the
server must reject invalid mutations.

## Migration Policy

Existing local Workspace Playground data should be automatically migrated into
the server-backed unified Workspace model. This is a true move, not an optional
import path.

Migration entry point:

- User opens `/research-workspace`.
- Client detects legacy local workspace data.
- Client submits a migration manifest and chunked payloads to the server.
- Server validates, persists, writes a receipt and recovery manifest, and
  verifies read-back.
- Server marks migration `client_delete_eligible`.
- Client deletes local content payloads.
- Client writes a non-content tombstone.
- Client acknowledges deletion to the server.

Local deletion must not happen until the server says deletion is eligible.

The old `/workspace-playground` route is not a migration entry point because it
is removed.

## Legacy Store Inventory Gate

Phase A must begin with a legacy Workspace Playground storage inventory and
schema-mapping deliverable. This is a blocking gate before any implementation
that can delete local data.

Known current storage surfaces to inventory include:

- `tldw-workspace`
- `tldw-workspace:workspace:<workspace_id>:snapshot`
- `tldw-workspace:workspace:<workspace_id>:chat`
- IndexedDB database `tldw-workspace-storage`
- IndexedDB object store `workspace-chat-sessions`
- IndexedDB object store `workspace-artifact-payloads`

The inventory must also check historical and feature-flagged storage paths in
the current codebase before deletion logic is implemented.

The migration schema mapping must define, for every legacy field:

- destination server object and field;
- whether the field is content, metadata, UI-only, derived, obsolete, or
  unsupported;
- validation rule;
- checksum or comparison rule where applicable;
- behavior on missing, malformed, oversized, or unsupported values;
- whether the field is eligible for deletion after migration.

Local content keys that are not covered by this inventory and mapping must not
be deleted. After successful migration, the only local data retained should be
the non-content tombstone described below.

Implementation planning must explicitly classify UI-only legacy keys such as
pinned workspace preferences, add-source tab usage, recent output types,
feature rollout flags, and broadcast/sync flags as retained, renamed, or
deleted. Those keys are outside the migrated content payload unless the
inventory promotes one into server-backed workspace metadata.

## Migration Protocol

Large local workspaces may exceed request limits. The migration protocol should
support chunked upload:

1. Create migration session with manifest.
2. Upload object batches for sources, folders/tags, notes, chats, artifacts,
   outputs, and metadata.
3. Finalize migration.
4. Server writes migration receipt and recovery manifest.
5. Server verifies read-back.
6. Server marks deletion eligibility.
7. Client deletes local data and sends deletion acknowledgement.

Candidate endpoints:

```text
POST /api/v1/workspaces/migrations
PUT /api/v1/workspaces/migrations/{migration_id}/chunks/{chunk_id}
POST /api/v1/workspaces/migrations/{migration_id}/finalize
GET /api/v1/workspaces/migrations/{migration_id}
POST /api/v1/workspaces/migrations/{migration_id}/client-delete-ack
```

The migration endpoint must be idempotent. It accepts:

- `migration_id`
- `legacy_workspace_id`
- schema version;
- client build/version;
- idempotency key;
- manifest hash.

Migration states:

- `received`
- `persisting`
- `persisted`
- `receipt_written`
- `verifying`
- `verified`
- `client_delete_eligible`
- `client_delete_acknowledged`
- `failed_validation`
- `failed_persist`
- `failed_receipt`
- `failed_verification`
- `client_delete_blocked`

## Migration Receipt And Recovery Manifest

Because migration is a true move, the server-side receipt must be
recovery-grade. Counts alone are not enough.

The receipt and recovery manifest should include:

- legacy workspace id;
- new server workspace id;
- user id;
- migration schema version;
- client build/version;
- object mappings for sources, folders, tags, notes, chats, artifacts, outputs,
  and metadata;
- per-source media validation outcome;
- source media ids and stable source ids;
- object counts;
- checksums or hashes where available;
- warnings and failures;
- verification result;
- timestamp;
- deletion eligibility;
- deletion acknowledgement timestamp when received;
- bounded diagnostic payload metadata when safe and useful.

After local content deletion, the client should keep only a non-content
tombstone:

- legacy workspace id;
- server workspace id;
- migration id;
- deletion timestamp.

The tombstone prevents duplicate migration and helps support, but it must not
retain source content, chat content, notes, artifacts, or other migrated legacy
content payload data. This does not forbid separate local UI preferences or
drafts that are explicitly classified as non-content in the migration inventory.

## Media Reference Validation

Legacy local sources may reference media that no longer exists, belongs to
another user, is not indexed, or is otherwise not portable.

Migration must produce per-source outcomes:

- `migrated`
- `relinked`
- `missing_media`
- `needs_reingest`
- `not_queryable`
- `blocked_by_permissions`
- `failed`

These outcomes appear in the migration receipt and in the Research Workspace UI
after migration.

## Phase A: NotebookLM Migrant First Value

Goal: make `/research-workspace` immediately understandable and useful for a
first-time migrant.

Scope:

- Add `/research-workspace`.
- Remove `/workspace-playground`.
- Rename visible route labels to Research Workspace.
- Create/load server-backed workspaces by default.
- Complete the legacy store inventory and schema-mapping gate before migration
  deletion logic.
- Automatically migrate legacy local workspace data with receipt, verification,
  and true local deletion.
- Add or expose first-class workspace source status.
- Add job-backed source add/ingest/index status.
- Redesign first-run empty state.
- Make Add Sources start with upload, URL, paste, and existing media.
- Show source scope in the composer.
- Enable grounded chat only when queryable selected sources exist.
- Add compact workspace capability/status panel acknowledging content, sharing,
  MCP/tools, ACP/agents, sandbox, provider/model, and governance dimensions.
- Ship the Phase A minimum capability contract defined in this design.
- Add list/select recent workspaces API/client contract so Phase C extension
  work has a stable workspace picker foundation.

First screen should answer:

- What is this workspace?
- What data is inside it?
- What is processing or ready?
- What can I safely ask now?

Success criteria:

- A new user can create or open Research Workspace.
- The user can add sources and see processing/indexing/queryability state.
- The user can ask a grounded question after sources become queryable.
- The user understands local/server/external-provider boundaries.
- Legacy local workspace data is moved only after server receipt and
  verification.
- `/workspace-playground` is not registered and returns normal 404.

## Phase D: Trust And Transparency

Goal: users can understand and trust the workspace system.

Scope:

- Workspace health and capability panel.
- Per-source status drilldown.
- Job-backed retry and partial-success handling.
- Citation and evidence drawer.
- Provider and model visibility.
- External-provider privacy warnings.
- Migration receipt viewer.
- Access tier and governance explanations.
- MCP/tool readiness and blocked reasons.
- ACP agent readiness, approval requirements, active sessions, and tool-call
  history summary.
- Sandbox policy, path scope, active sessions, and execution risk summary.

Success criteria:

- Users can tell whether a source is queryable and why.
- Users can inspect why an action is disabled or requires approval.
- Users can inspect evidence behind grounded answers.
- Users can inspect migration receipt state.
- Health states are not contradictory.

## Phase B: Experienced Power-User Workflow

Goal: serious repeat research work scales.

Scope:

- Dense source table mode.
- Search and filters by status, type, tag, folder, selected/queryable,
  owner/shared, and tool-accessible state.
- Bulk actions.
- Saved source sets.
- Source inspection drawer.
- Compare sources output.
- Workspace activity/history.
- Export outputs with evidence lineage.
- Keyboard shortcuts and command palette actions.
- Governance-aware controls for shared/tool-enabled workspaces.

Success criteria:

- Power users can manage many sources without card-only scanning.
- Users can recover failed/partial source states in bulk.
- Users can inspect and compare evidence across sources.
- Users can resume prior work quickly.

## Phase C: Browser Extension Capture Loop

Goal: browser capture lands in the correct unified workspace.

Scope:

- Server-backed workspace picker.
- Current/recent workspace awareness.
- Access-tier-aware destination controls.
- Capture-to-source status.
- Save/open into `/research-workspace`.
- Tag, folder, and source-set placement.
- Clear privacy/provider warning.
- Capture failure recovery.

Success criteria:

- The extension can save captures into a server-backed Research Workspace.
- The user can see capture ingestion/indexing status in the WebUI.
- The extension respects shared workspace access levels.
- "Save and open" opens `/research-workspace`, not a legacy or unrelated
  workspace surface.

## Data Flow

1. User opens `/research-workspace`.
2. Client checks for legacy local workspace data.
3. Client starts an idempotent migration session if legacy data exists.
4. Client uploads manifest and chunks.
5. Server validates media and source references.
6. Server persists canonical workspace and subresources.
7. Server writes migration receipt and recovery manifest.
8. Server verifies read-back.
9. Server marks migration `client_delete_eligible`.
10. Client deletes legacy local content data.
11. Client writes a non-content tombstone.
12. Client sends deletion acknowledgement.
13. Client renders server workspace state.
14. User adds source.
15. Server creates or attaches Jobs for ingestion and indexing.
16. Source status progresses through ingestion, extraction, chunking, indexing,
    and queryability.
17. Chat enables grounded mode only for queryable selected sources.
18. Answers and outputs persist evidence lineage.
19. Workspace capability panel shows content, sharing, MCP, ACP, sandbox,
    provider, and governance state.

## Error Handling Requirements

Failures must produce durable status or recoverable UI actions, not only generic
toasts.

Primary failure modes:

- backend unavailable;
- migration validation failure;
- migration persistence failure;
- migration receipt failure;
- migration verification failure;
- client deletion blocked;
- missing or non-portable media references;
- source ingest failure;
- indexing failure;
- selected source not queryable;
- model/provider unavailable;
- access tier forbids action;
- MCP/tool policy blocks action;
- ACP approval required or denied;
- sandbox unavailable or policy-blocked.

## API Contract Candidates

These endpoint shapes are design candidates, not final implementation commands:

```text
GET  /api/v1/workspaces
PUT  /api/v1/workspaces/{workspace_id}
GET  /api/v1/workspaces/{workspace_id}
GET  /api/v1/workspaces/{workspace_id}/sources
GET  /api/v1/workspaces/{workspace_id}/sources/status
GET  /api/v1/workspaces/{workspace_id}/capabilities
POST /api/v1/workspaces/migrations
PUT  /api/v1/workspaces/migrations/{migration_id}/chunks/{chunk_id}
POST /api/v1/workspaces/migrations/{migration_id}/finalize
GET  /api/v1/workspaces/migrations/{migration_id}
POST /api/v1/workspaces/migrations/{migration_id}/client-delete-ack
```

`GET /api/v1/workspaces` should satisfy the Phase A recent/selectable workspace
picker contract unless implementation planning finds an existing pagination or
sorting constraint that requires a dedicated endpoint. The required picker data
is workspace id, name, last updated or last accessed timestamp, private/shared/
cloned state, and enough access metadata to disable invalid destinations.

Sharing remains under `/api/v1/sharing`.

MCP Hub, ACP, and sandbox services keep their own domain APIs. Research
Workspace consumes their summarized state through the workspace capability
projection.

## Testing And Verification

Backend tests:

- workspace source status schema and projection tests;
- Jobs-backed ingestion/indexing status tests;
- migration endpoint idempotency tests;
- chunked migration tests;
- migration receipt and recovery manifest tests;
- migration verification success and failure tests;
- media reference validation tests;
- local-deletion eligibility state tests;
- access enforcement tests for private, shared, cloned, MCP-enabled,
  ACP-enabled, and sandboxed workspaces;
- sharing access-level tests for source, chat, artifact, and tool mutations;
- workspace capabilities projection tests;
- route registry tests ensuring old workspace route is not registered.

Frontend tests:

- `/research-workspace` exists;
- `/workspace-playground` returns normal 404;
- first-run empty state;
- Add Sources default landing;
- source status rendering;
- readiness dimensions rendering;
- grounded chat enable/disable rules;
- migration progress, success, and failure states;
- deletion acknowledgement after eligibility;
- non-content tombstone behavior;
- capability panel rendering;
- access-tier disabled controls;
- mobile layout;
- keyboard/focus path.

E2E/CDP validation:

- Run backend and WebUI together with
  `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced` and
  `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`.
- Validate `/research-workspace` against a real backend.
- Create a source and observe status progression, using controlled mocks only
  where real ingest would be too expensive.
- Verify `/workspace-playground` returns normal 404.
- Verify no contradictory health states.
- Verify browser extension capture can select a workspace and open
  `/research-workspace`.

## Route Reference Inventory Gate

Phase A must include a route-reference inventory. User-facing references to
`/workspace-playground`, Workspace Playground, or Research Studio should be
removed or intentionally reclassified as historical internal documentation.

The build/test suite should fail if old user-facing route references remain in:

- route metadata;
- navigation;
- tutorials;
- screenshots;
- docs for current users;
- extension handoff routes;
- e2e route inventories;
- smoke tests;
- new telemetry labels.

Historical analytics event names may remain only if they are explicitly
classified as legacy telemetry identifiers and are not surfaced to users,
navigation, tutorials, or current documentation.

## Product Metrics

Recommended product and quality metrics:

- time to first queryable source;
- source ingestion success rate;
- source indexing success rate;
- migration success rate;
- migration verification failure rate;
- local deletion blocked rate;
- grounded-chat activation rate;
- citation/evidence drawer open rate;
- source failure recovery rate;
- extension capture success rate;
- average time from capture to queryable source;
- frequency of blocked actions by access/governance reason.

## Rollout Plan

This is a hard replacement, but implementation should still be staged
internally:

1. Add backend contracts and migration safety.
2. Add `/research-workspace`.
3. Remove `/workspace-playground` from route registration, navigation, docs,
   tests, and tutorials.
4. Wire live source status and first-run UX.
5. Add workspace capability projection and trust panel.
6. Expand power-user source management and evidence workflows.
7. Connect browser extension capture into canonical workspace identity.

## Open Questions For Implementation Planning

- Which existing Jobs primitives should own ingestion, extraction, chunking, and
  indexing progress for workspace sources?
- Should source status be computed on read, event-updated into a projection
  table, or both?
- What size limits should migration chunks enforce?
- How much bounded diagnostic metadata can the recovery manifest safely retain?
- Which old telemetry names should be renamed, aliased internally, or left as
  historical analytics labels?

## Acceptance Criteria

- The written roadmap preserves the hard route replacement decision.
- The design keeps `/research` separate from Research Workspace.
- The design treats MCP, ACP, and sandbox as core Workspace dimensions.
- The design aligns Research Workspace with sharing and shared workspace access.
- The design defines server-backed source of truth.
- The design includes migration idempotency, chunking, receipt, verification,
  and deletion acknowledgement.
- The design includes first-class source status and readiness dimensions.
- The design includes Jobs-backed ingestion/indexing status.
- The design includes a workspace capability projection.
- The design defines the Phase A minimum capability contract.
- The design defines the legacy store inventory and schema-mapping gate.
- The design includes testing, route inventory, and product metric gates.
