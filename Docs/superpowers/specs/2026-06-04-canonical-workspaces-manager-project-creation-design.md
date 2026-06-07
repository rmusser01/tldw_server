# Canonical Workspaces Manager And Project Workspace Creation Design

## Purpose

tldw_server needs a first-class Workspaces management surface that sits above
Research Workspace, MCP Hub, ACP, Sandbox, and future agent harnesses. The
canonical Workspace is the product identity. Research Workspace and Project
Workspace are profiles or capability states of that identity, not separate
object families.

This spec defines the roadmap for a server-backed `/workspaces` manager and the
backend contracts needed to create, edit, archive, unarchive, and upgrade
Workspaces into Project Workspaces with host-local or sandbox-managed primary
roots.

## Phase 2 Epic Alignment

This workstream is a bounded implementation slice under GitHub issue
[#1984](https://github.com/rmusser01/tldw_server/issues/1984), "Workspace
container foundation and single-user operating context." The epic defines
Workspace as the durable operating container that existing tldw capabilities
attach to: notes, media/sources, artifacts, chats, prompts, workflows,
watchlists, ACP sessions, sandbox sessions, runtime bindings, and metadata.

The `/workspaces` route in this spec is therefore not the whole Workspace
product and not a replacement for existing surfaces. It is the registry,
lifecycle, project-root, recovery, and cross-surface handoff surface for the
broader single-user Workspace container. Library, Notes, Artifacts, chats,
prompts, workflows, watchlists, ACP, MCP, Sandbox, and Research Workspace must
remain globally browsable or surface-owned while becoming Workspace-aware
through membership tags, active-context eligibility, and explicit copy/link
flows.

Issue alignment:

- Advances
  [#1988](https://github.com/rmusser01/tldw_server/issues/1988) through
  canonical Workspace profile/root/context contracts.
- Advances
  [#1989](https://github.com/rmusser01/tldw_server/issues/1989) through
  single-user Workspace registry, lifecycle, archive/unarchive, and manager
  metadata flows.
- Advances
  [#1991](https://github.com/rmusser01/tldw_server/issues/1991) through
  host-local and sandbox-managed root binding descriptors with redacted
  metadata.
- Partially advances
  [#1993](https://github.com/rmusser01/tldw_server/issues/1993) through
  manager/client contracts and selected Research Workspace, MCP Hub, ACP, and
  Sandbox handoffs.
- Partially advances
  [#1994](https://github.com/rmusser01/tldw_server/issues/1994) and
  [#1995](https://github.com/rmusser01/tldw_server/issues/1995) only for the
  manager/project-root loop covered by the UAT matrix.
- Does not close
  [#1990](https://github.com/rmusser01/tldw_server/issues/1990) cross-resource
  membership, [#1992](https://github.com/rmusser01/tldw_server/issues/1992)
  active-context eligibility/recovery, or the full end-to-end evidence required
  by [#1995](https://github.com/rmusser01/tldw_server/issues/1995).

## Current Verified Baseline

The current `origin/dev` branch already has substantial Workspace Core backend
support:

- `/api/v1/workspaces` exposes list, get, upsert, patch, delete, source,
  artifact, note, capability, context, roots, and file-inventory endpoints.
- Workspace Core persists `workspace_profile` as `research` or `project`.
- Workspace primary roots support `host_local` and `sandbox_volume` backend
  types.
- Workspace file inventory is Jobs-backed and metadata-only.
- Workspace Core can attach a `sandbox_volume` root by ID, but the default
  resolver reports `not_configured`.
- Sandbox has session and run APIs with `workspace_id` fields, but the current
  public surface is session-oriented and does not provide a durable
  workspace-volume creation command.
- Research Workspace has local saved-workspace switching, settings, archive,
  delete, source transfer, and server reconciliation for active workspace
  rows, but it is not a canonical Workspaces directory.
- MCP Hub has "Shared Workspaces" and Workspace Sets for trusted roots and
  tool policy, but those should not become the canonical Workspace identity.

## Product Decision

Create a new canonical `/workspaces` management surface for the single-user
Workspace container.

The `/workspaces` manager owns Workspace identity, profile intent, metadata,
archive/unarchive lifecycle, project-root lifecycle entry points, and
cross-surface navigation. Specialized subsystems keep their existing ownership:

| Surface | Owns | Does Not Own |
| --- | --- | --- |
| Workspace Core | Workspace identity, metadata, profile intent, primary root binding, context envelope | Sandbox runtime internals, MCP policy, ACP run lifecycle |
| Research Workspace | Research activity: sources, notes, chat, Studio outputs, source selection | Canonical Workspace directory or runtime policy |
| MCP Hub | Trusted roots, workspace sets, path scopes, tool policy, governance, audit | Canonical Workspace identity |
| Sandbox | Runtime admission, sessions, runs, durable volume mechanics, mounts, diagnostics | Product-level Workspace creation flow |
| ACP and harnesses | Agent sessions, run diagnostics, adapter state | Workspace identity or root ownership |

## Non-Goals

This workstream does not:

- Replace the full Research Workspace local store in one PR.
- Implement the full cross-resource membership service for notes,
  media/sources, artifacts, chats, prompts, workflows, watchlists, ACP sessions,
  and sandbox sessions.
- Implement the full active Workspace context eligibility gate across existing
  surfaces.
- Use Workspace selection as a hard browse/search filter that hides globally
  owned user records.
- Make MCP Hub the Workspace manager.
- Add multi-root Project Workspaces.
- Add team governance or sharing.
- Add deployment or hosted preview flows.
- Index all project files by default.
- Expose raw host-local paths beyond existing redaction and authorized user
  entry points.
- Treat Sandbox sessions as durable Workspace roots without a durable volume
  contract.

## Workstream Roadmap

### Task 1: Existing Workspace API Client Parity

Goal: make the WebUI capable of safely calling current Workspace backend APIs.

Scope:

- Add typed client methods and tests for existing Workspace APIs:
  - `PATCH /api/v1/workspaces/{workspace_id}`
  - `DELETE /api/v1/workspaces/{workspace_id}`
  - `GET /api/v1/workspaces/{workspace_id}/roots`
  - `PUT /api/v1/workspaces/{workspace_id}/roots/primary`
  - `POST /api/v1/workspaces/{workspace_id}/file-inventory/scan`
  - `GET /api/v1/workspaces/{workspace_id}/file-inventory/status`
  - `GET /api/v1/workspaces/{workspace_id}/file-inventory/items`
- Normalize response types for manager use, including `workspace_profile`,
  archive state, root state, context resolution, allowed actions, partial
  errors, and file inventory availability.
- Keep canonical Workspace client/types separate from ACP orchestration,
  prototype workspace, and MCP Hub shared-workspace client/types.
- Preserve existing Research Workspace API behavior.

Out of scope:

- New sandbox-volume provisioning methods.
- UI route implementation.

Acceptance:

- WebUI API client has typed methods for all existing Workspace manager
  dependencies.
- Tests verify HTTP method, route, body shape, response normalization, and
  error propagation.

### Task 2: Durable Sandbox Workspace-Volume Contract

Goal: add or formalize the durable Sandbox primitive needed for
sandbox-managed Project Workspace roots.

Sandbox should own runtime mechanics. Workspace should not fabricate a
sandbox root from a short-lived session.

Scope:

- Define a durable workspace-volume concept owned by Sandbox.
- Provide service-level operations for:
  - create or provision workspace-bound volume
  - validate volume binding for `workspace_id` and `user_id`
  - resolve mount/readiness state
  - report durable diagnostic state
  - support idempotent retry
- Define states at minimum:
  - `provisioning`
  - `ready`
  - `not_configured`
  - `unavailable`
  - `failed`
  - `cleanup_pending`
- Define safe diagnostics with bounded metadata.
- Avoid hard dependency on any one runtime, such as Docker or VZ, in the
  public Workspace-facing contract.
- Define how each Sandbox volume state projects into Workspace root state,
  mount state, capability state, inventory availability, and manager attention
  state before any UI implementation starts.

Out of scope:

- `/workspaces` UI.
- Workspace-owned provision-and-attach endpoint.
- ACP launch.

Acceptance:

- A Workspace-facing service can ask Sandbox to create, validate, and resolve
  a durable workspace volume.
- The existing Workspace Core sandbox resolver can be backed by this contract.
- Failure states are explicit and retryable where appropriate.

### Task 3: Workspace-Owned Sandbox Root Provision-And-Attach Command

Goal: expose one Workspace-owned product command for "make this a Project
Workspace with a sandbox-managed root."

Scope:

- Add a Workspace Core service that orchestrates:
  - load and version-check Workspace
  - ask Sandbox to provision a durable workspace volume
  - attach returned `sandbox_volume_id` as the primary root
  - set or preserve `workspace_profile: project`
  - return redacted roots, capabilities, and diagnostics
- Add an endpoint under `/api/v1/workspaces/{workspace_id}/...`.
- Require `Idempotency-Key` support owned by the Workspace command. The
  Workspace command fingerprints the product request, delegates to Sandbox with
  a linked idempotency key, returns the same operation/status for a matching
  retry, and returns `409` for a different request using the same key.
- Scope idempotency records by `user_id`, `workspace_id`, command name, and
  `Idempotency-Key`. Store only bounded metadata: request fingerprint, linked
  Sandbox idempotency key, operation id, latest status, result pointer, created
  timestamp, updated timestamp, and expiration timestamp. Do not store raw host
  paths, secrets, env vars, or full request bodies in the idempotency record.
  Keep successful and failed records for at least 24 hours, with a recommended
  V1 default of 7 days.
- Handle duplicate retry, stale version, existing primary root, Sandbox
  unavailable, resolver mismatch, and partial failure.
- Return a recoverable state if Sandbox provisioning fails after the Workspace
  row exists.
- Do not hold the HTTP request open until a runtime volume is fully mounted.
  Return an operation/status envelope with `202 Accepted` when provisioning is
  queued or active, and `200 OK` only when an equivalent root is already
  attached or the command completes synchronously.
- Return a pollable operation envelope when provisioning is queued or active.
  The preferred V1 status endpoint is:
  `GET /api/v1/workspaces/{workspace_id}/operations/{operation_id}`.
  Workspace context responses should also expose active Workspace operations so
  the UI can recover after refresh without remembering the original response.

Recommended endpoint shape:

```text
POST /api/v1/workspaces/{workspace_id}/roots/primary/sandbox-volume
```

Recommended request fields:

```json
{
  "display_name": "Project sandbox",
  "runtime": "docker",
  "base_image": "optional",
  "replace_existing": false,
  "expected_workspace_version": 3,
  "labels": {
    "purpose": "project-workspace"
  }
}
```

Out of scope:

- Host-local root attach, already covered by the existing primary-root
  endpoint.
- Full Sandbox UI or runtime session launch.

Acceptance:

- Workspace-owned command succeeds when Sandbox volume provisioning succeeds.
- Idempotent retry returns the same root or compatible result.
- Root conflict returns actionable conflict details.
- Sandbox failure produces a Workspace-safe recoverable state.
- Capabilities fail closed until the sandbox root is ready or mounted.
- Response semantics are explicit: active provisioning returns `202` with a
  status envelope; ready or already-attached roots return `200`.
- Operation status can be polled and recovered after refresh.

### Task 4: Canonical `/workspaces` Manager CRUD

Goal: create the product-level directory for Workspaces.

Scope:

- Add `/workspaces` route.
- Render a server-backed Workspace list with:
  - name
  - profile: Research or Project
  - root health summary
  - inventory summary when available
  - source count or content summary when available
  - updated/accessed timestamps when available
  - attention state
- Support:
  - search
  - filters for Research, Project, Archived, Needs attention
  - create Research Workspace
  - create or show an existing Project Workspace shell without root setup
  - edit metadata
  - archive and unarchive
  - open in Research Workspace
- Use server data as the source of truth.
- Do not expose hard delete or soft-delete restore in the V1 manager. Deleted
  Workspace rows remain outside this manager until a separate cleanup and
  restore contract exists.

Out of scope:

- Full local Research Workspace migration.
- Project root setup, host-local attach, and sandbox-managed provisioning UX.
  Those belong to Task 5.
- MCP policy editing.
- ACP or Sandbox run launch.

Acceptance:

- User can create, edit, archive, unarchive, and open a Workspace from the
  manager.
- Empty, loading, error, partial-success, and unavailable-backend states are
  visible and specific.
- Manager does not show local-only Research Workspace entries as if they were
  server-backed.

### Task 5: Project Upgrade And Root Panel

Goal: make Project Workspace root state legible and actionable.

Scope:

- Add "Upgrade to Project Workspace" for Research Workspaces.
- Support root type selection:
  - host-local root via existing attach endpoint
  - sandbox-managed root via new Workspace-owned command
- Show root status:
  - not configured
  - provisioning
  - attached
  - unavailable
  - failed
  - detached or missing
- Show file inventory status:
  - unavailable
  - idle
  - queued
  - running
  - completed
  - partial
  - failed
  - stale
- Capability-gate inventory actions. For sandbox roots, do not show scan as
  available until the Workspace API exposes `file_inventory.available: true`
  from a resolver-backed mounted local path.
- Add retry and remediation actions for root and inventory failures.

Out of scope:

- File-content indexing.
- Git state beyond limited future-link affordances.
- Multi-root support.

Acceptance:

- User can upgrade an existing Research Workspace to Project with host-local
  or sandbox-managed root.
- Root and inventory failures are understandable and recoverable.
- Raw host-local paths remain redacted in passive displays.

### Task 6: Local Research Workspace Reconciliation

Goal: avoid two disconnected Workspace worlds without forcing a risky
Research Workspace persistence rewrite.

Scope:

- Detect local Research Workspace saved entries from the existing local store.
- Show local-only entries separately from server-backed Workspaces.
- Provide dry-run states:
  - local-only
  - server row exists
  - name conflict
  - possible duplicate
  - unsupported local payload
  - ready to create metadata
- Let users create or link canonical backend Workspace metadata.
- Preserve local tombstones and undo behavior.
- After metadata promotion or link, write a minimal local reconciliation marker
  containing the server `workspace_id`, server name, profile, linked timestamp,
  and reconciliation status. Do not rewrite source, note, artifact, chat, or
  IndexedDB payloads in V1.

Out of scope for V1:

- Complete source, note, artifact, chat, and IndexedDB payload migration.
- Silent background migration.
- Deleting local entries after server metadata creation.

Acceptance:

- User can see which Research Workspace entries are local-only.
- User can promote/link metadata without losing local data.
- Conflicts are explicit.

### Task 7: Cross-Surface Links And Live UAT

Goal: prove the canonical manager works with the larger Workspace model.

Scope:

- Add deep links from Workspace manager rows/panels to:
  - Research Workspace
  - MCP Hub trusted-root or tool-scope views
  - ACP session diagnostics
  - Sandbox workspace diagnostics
- Use copy that avoids MCP naming confusion:
  - "MCP trusted root binding"
  - "MCP tool scope"
  - not "another Workspace"
- Add or update live backend and WebUI validation matrix.
- Use Playwright/CDP for WebUI validation.

Acceptance:

- A user can create a Workspace, upgrade it, inspect root/inventory state, and
  jump to the relevant specialized surface.
- Live validation covers backend, WebUI, and cross-surface links.
- No `/workspace-playground` aliases or redirects are introduced.

## Project Creation State Machine

Project Workspace creation must not be modeled as a single irreversible submit.
It crosses Workspace persistence and Sandbox runtime mechanics, so partial
failure is expected.

Recommended states:

| State | Meaning | User Action |
| --- | --- | --- |
| `workspace_created` | Metadata row exists | Continue setup or archive |
| `project_setup_pending` | User selected Project but no root exists | Choose root |
| `host_root_validating` | Host-local root attach in progress | Wait |
| `sandbox_root_provisioning` | Sandbox durable volume creation in progress | Wait |
| `root_attached` | Primary root is persisted | Start inventory scan |
| `root_unavailable` | Root exists but cannot currently be used | Retry, inspect diagnostics |
| `root_failed` | Provisioning or validation failed | Retry, replace root, archive |
| `inventory_pending` | Root exists but inventory not scanned | Start scan |
| `inventory_running` | Scan job active | Wait |
| `inventory_partial` | Scan completed with bounded failures | Inspect diagnostics, rescan |
| `project_ready` | Root and inventory are usable | Continue work |

The UI can collapse these states into fewer labels, but the API and tests
should preserve enough detail for recovery and diagnostics.

## Sandbox-To-Workspace State Projection

Task 2 and Task 3 planning should pin one projection table before coding. The
table below is the recommended V1 default.

| Sandbox Condition | Workspace Root State | Mount State | Capabilities | Inventory | Attention |
| --- | --- | --- | --- | --- | --- |
| `provisioning` | `provisioning` | `not_ready` | fail closed for file writes and execution | unavailable | `working` |
| `ready` with usable mount | `attached` | `ready` | allow actions backed by ready capabilities | available | `ready` |
| `ready` without usable mount | `attached` | `not_ready` | fail closed for file writes and execution | unavailable | `needs_attention` |
| `not_configured` | `unavailable` | `not_configured` | fail closed | unavailable | `blocked` |
| `unavailable` | `unavailable` | `unavailable` | fail closed | unavailable | `blocked` |
| `failed` | `failed` | `failed` | fail closed | unavailable | `blocked` |
| `cleanup_pending` | `cleanup_pending` | `unavailable` | fail closed | unavailable | `needs_attention` |

Implementation may refine enum names, but it must keep one shared projection
helper so Workspace API responses, WebUI normalization, and tests do not invent
separate interpretations.

## Manager Attention State Mapping

Implementation planning should define one shared mapping used by backend tests,
API-client normalization, and UI display.

Recommended V1 mapping:

| Attention State | Inputs | Meaning |
| --- | --- | --- |
| `ready` | Research Workspace, or Project Workspace with usable root and no active failures | User can continue work |
| `setup_pending` | Project profile with no primary root, or project setup before root choice | User needs to attach or provision a root |
| `working` | Root provisioning, root validation, inventory queued, or inventory running | Background work is active |
| `needs_attention` | Root unavailable, inventory partial or stale, local-only reconcile conflict | User action is useful but not destructive |
| `blocked` | Sandbox unavailable for sandbox-managed root, host path rejected, resolver failed, or hard conflict | User cannot complete the intended workflow until remediation |
| `archived` | Workspace archived | Hidden from the default active view |

The exact enum names may change during implementation, but one mapping should
exist before UI work starts. Component tests should assert the mapping rather
than each component deriving its own state.

## Archive And Delete Safety

Archive should be the default destructive action in V1.

V1 policy: hard delete is disabled for Project Workspaces. Allow archive and
unarchive only. The V1 manager should not expose soft-delete or restore of
deleted rows for any Workspace profile. A later cleanup contract can add
explicit choices such as:

- detach root but keep sandbox volume
- queue sandbox volume cleanup
- delete Workspace metadata after cleanup completes

If a future manager adds delete for Research Workspaces, it should stay scoped
to Workspace metadata and content records, not external filesystem content. The
default must never silently delete a sandbox-managed root or host-local
filesystem content. Deleted-row restore is out of scope until the API can list,
inspect, and restore deleted Workspaces without conflating them with archived
Workspaces.

## Implementation Guardrails

These constraints are part of the design, not later UI preferences.

Canonical naming and routes:

- `/workspaces` in the WebUI is the canonical Workspace manager.
- `/api/v1/workspaces` is the canonical Workspace Core API.
- ACP orchestration workspaces are execution projections. Client types and UI
  labels must use `ACPWorkspace` or "agent execution workspace" when referring
  to ACP rows.
- MCP Hub "Shared Workspaces" and Workspace Sets remain MCP policy objects.
  New manager copy must not call them canonical Workspaces.
- Prototype workspaces remain prototype workspaces and should not be imported
  into the manager unless a later bridge explicitly maps them to canonical
  Workspace rows.

Async operations:

- Workspace-owned async commands return an operation envelope containing at
  minimum `operation_id`, `workspace_id`, command name, status, started time,
  updated time, retryable flag, redacted diagnostics, and a poll href.
- `GET /api/v1/workspaces/{workspace_id}/operations/{operation_id}` is the
  preferred V1 operation poll endpoint.
- `GET /api/v1/workspaces/{workspace_id}/context` should include active or
  recently completed Workspace operations so refresh and deep-link recovery do
  not depend on local UI memory.
- Operation status must use the same root, mount, capability, inventory, and
  attention-state projection helper as normal context responses.

Client normalization:

- Task 1 must update WebUI Workspace API types so manager code can read
  `workspace_profile`, archive state, root state, context resolution,
  `project_root`, `allowed_actions`, service capability states, partial
  errors, and `file_inventory.available`.
- Manager components should consume normalized canonical Workspace models,
  never raw ACP, MCP, or prototype workspace response shapes.

Idempotency:

- Idempotency records are scoped by user, Workspace, command, and key.
- Fingerprints must be computed from redacted, stable request fields. Runtime
  secrets, env vars, raw host paths, and full payloads are excluded.
- Matching retry returns the same operation/status or already-attached result.
- Different request with the same key before expiration returns `409`.
- Expired records can be cleaned up. If cleanup removes the record but the root
  already exists, the command should still return the equivalent current root
  state rather than creating a second root.

## UX Copy Rules

Use these terms:

- "Workspace" for the canonical product object.
- "Research Workspace" for a Workspace focused on sources, notes, chat, and
  research outputs.
- "Project Workspace" for a Workspace with project intent and a primary root.
- "Host-local root" for a user-approved local folder.
- "Sandbox-managed root" for a durable sandbox volume.
- "MCP trusted root binding" or "MCP tool scope" for MCP Hub links.

Avoid these terms in new UI copy:

- "Workspace Playground"
- "Shared Workspace" as a synonym for canonical Workspace
- "workspace trust bar"
- raw sandbox or MCP internals without user-facing context

Example empty-state copy:

> No Workspaces yet. Create a Research Workspace for sources and notes, or a
> Project Workspace when you need files, a root, and sandbox-backed execution.

Example sandbox unavailable copy:

> Sandbox-managed roots are unavailable right now. The Workspace was saved, but
> project setup is paused until Sandbox readiness recovers.

Example local-only copy:

> This Research Workspace is stored locally in your browser. Link it to a
> server Workspace to use project roots, MCP tool scopes, and agent diagnostics.

## Validation Strategy

Each implementation task should include focused tests and a narrow verification
record.

Backend:

- Workspace root service tests for idempotency, conflict, and partial failure.
- Sandbox workspace-volume service tests for create, validate, resolve, and
  failure states.
- API tests for response shape, stale version, resolver mismatch, unavailable
  Sandbox, and delete/archive safety.
- Bandit on touched backend paths.

Frontend:

- API client unit tests for route, method, request, and normalization.
- Component tests for manager list, filters, create flow, edit/archive/unarchive,
  Project upgrade, inventory gating, and reconciliation dry run.
- Accessibility checks for keyboard focus, labels, alerts, and dialogs.

Live:

- Start real backend and WebUI.
- Use Playwright/CDP.
- Validate:
  - create Research Workspace
  - create Project Workspace with host-local root
  - create Project Workspace with sandbox-managed root when Sandbox is ready
  - sandbox unavailable recovery state
  - archive/unarchive
  - local-only reconciliation metadata flow
  - Research Workspace deep link
  - MCP trusted-root/tool-scope deep link
  - Sandbox diagnostics link

## Parallelization

The first tasks are dependency-bound:

1. Existing Workspace API client parity.
2. Durable Sandbox workspace-volume contract.
3. Workspace-owned sandbox root provision-and-attach command.

After those, work can split:

- `/workspaces` manager CRUD can proceed from Task 1.
- Project root panel can proceed after Tasks 1 and 3.
- Local reconciliation can proceed after Task 4 establishes the manager shell.
- Cross-surface links and live UAT can start once Task 4 has route-level test
  hooks.

## Open Implementation Questions

These should be resolved during task planning, not left to UI implementation:

1. What durable Sandbox storage object should represent a workspace volume?
2. Which Sandbox runtimes can create a workspace-bound volume in V1?
3. How long should failed or cleanup-pending sandbox volumes be retained?
4. Which local Research Workspace subresources can be safely migrated after
   metadata reconciliation?
5. Which route should be the stable deep link target for ACP workspace
   diagnostics?
6. What exact local reconciliation marker shape should be persisted in the
   Research Workspace store?
7. Which component or helper owns attention-state derivation so backend,
   client, and UI tests use the same mapping?
8. Which Workspace-owned service stores the provision-and-attach idempotency
   request fingerprint, operation id, retry response, conflict metadata, and
   7-day default expiration timestamp?
9. Which Sandbox runtimes can complete provisioning synchronously enough for
   the Workspace command to return `200` instead of `202`?
10. Which operation status response schema backs
    `GET /api/v1/workspaces/{workspace_id}/operations/{operation_id}` and the
    active operations summary in Workspace context?

## Planning-Time Decisions Required Before Coding

The implementation plan must resolve these items before assigning code slices:

1. Versioned local reconciliation marker schema, including the local store key,
   marker field name, server `workspace_id`, server profile, linked timestamp,
   reconciliation status, and conflict state.
2. Workspace provision-and-attach idempotency storage, request fingerprint
   rules, retry response shape, linked Sandbox idempotency key, 7-day default
   retention, cleanup behavior, and `409` conflict behavior.
3. Sandbox volume state projection into Workspace root, mount, capability,
   inventory, and attention states.
4. Task 3 response behavior: `202` for queued/active provisioning, `200` for
   already-attached or synchronously complete roots.
5. Workspace operation status envelope and polling schema.
6. Project Workspace hard-delete behavior: disabled in V1, archive/unarchive
   only.

## Success Criteria For The Workstream

The workstream is complete when:

- `/workspaces` is the canonical place to create and manage Workspaces.
- Research Workspace remains focused on research activity.
- MCP Hub remains focused on trusted roots and tool policy.
- A Research Workspace can be upgraded to Project Workspace.
- Host-local and sandbox-managed roots are both first-class in the creation
  flow.
- Sandbox-managed root creation goes through a Workspace-owned product command
  backed by a durable Sandbox volume primitive.
- Local Research Workspace entries can be reconciled metadata-first without
  data loss.
- Live backend and WebUI validation proves the primary flows are not hiddenly
  broken.
