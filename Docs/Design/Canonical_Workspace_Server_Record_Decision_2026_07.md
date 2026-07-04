# Canonical Workspace Server Record Decision - July 2026

Date: 2026-07-04
Status: Accepted
Tracking: [#1526](https://github.com/rmusser01/tldw_server/issues/1526)

## Decision

The canonical workspace product model is the server `Workspace` record exposed by
the `/api/v1/workspaces` API family. One server model supports multiple
specialized surfaces:

- `/workspaces` is the canonical lifecycle and management surface for creating,
  opening, editing, archiving, and configuring Workspaces.
- `/research-workspace` is the canonical research shell that consumes the server
  model for source-grounded research work.
- `/research` remains a separate research-provider and source-discovery surface.
  It is not an alias, redirect, or replacement for `/research-workspace`.

`WorkspacePlayground` is not the canonical shell. The legacy
`workspace_playground` label and `tldw.workspace-playground.bundle` payloads are
compatibility inputs only. The `/workspace-playground` route must remain removed
with no alias and no redirect.

`ChatWorkspace` and `DocumentWorkspace` may remain specialized entry points, but
they must not define separate workspace identity, membership, lifecycle, or
runtime-handoff semantics. Future route consolidation needs a superseding
decision record only if ownership changes.

## Server And Local Boundary

The minimum server-backed workspace record is already defined by the
[Workspace Container Contract](Workspace_Container_Contract_2026_06.md):

- `workspace_id` as the stable product identity.
- `name`, `workspace_profile`, lifecycle flags, version, and timestamps.
- membership rows for sources, artifacts, chats, prompts, workflows,
  watchlists, ACP sessions, and Sandbox sessions where adapters exist.
- source status, notes, artifacts, roots, capabilities, context, eligibility,
  runtime bindings, and activity/index read models through the Workspaces API.

Browser-local workspace state remains a cache and UI state layer. It may store
drafts, layout, local split-key payloads, import/export recovery state, and
offline-friendly snapshots. It is not authoritative for membership, lifecycle,
runtime trust, ACP admission, Sandbox admission, MCP policy, or cross-device
ownership.

Local-only or imported workspaces must materialize or reconcile a server
workspace record before cross-surface handoff. The server ID is the handoff key.

Workspace Core owns the canonical identity, profile, metadata, membership,
primary-root binding, and context envelope. The bound resource remains under its
owner-domain contract: Sandbox owns runtime and volume mechanics, MCP owns tool
trust and policy, and ACP owns agent projects, tasks, runs, sessions, reviews,
and harness execution roots. A runtime binding or active Workspace context does
not transfer authority between those domains.

## Handoff Map

| Surface | Contract |
| --- | --- |
| Workspace manager (`/workspaces`) | Creates and manages the canonical server record, profile intent, lifecycle, primary-root binding, and owner-domain handoffs. It does not duplicate owner-domain CRUD. |
| Research Workspace | Uses server `workspace_id` for source selection, chat grounding, studio artifacts, notes, import/export, and extension capture landing. |
| Research (`/research`) | Provides research-provider and source-discovery workflows. It remains distinct from the Research Workspace shell and does not own Workspace lifecycle. |
| ACP Agent Tasks and ACP Playground | Use `/api/v1/agent-orchestration/workspaces/canonical-bridge` and canonical workspace filters. ACP owns projects, tasks, runs, sessions, diagnostics, audit, reviewer state, promoted artifacts, and harness execution roots; it consumes rather than replaces the Workspace Core primary-root binding. |
| MCP Hub | Owns workspace sets, path/tool trust, policy assignment, and tool execution. It may bind to the canonical workspace ID but does not become the source-membership owner. |
| Sandbox | Owns volumes, admission, runtime/session lifecycle, run creation, diagnostics, and isolation. Workspace Core owns the association to a sandbox-managed primary root; Workspace context is an input, not a trust grant. |
| Browser extension capture | Targets canonical workspace/source IDs and deep-links to `/research-workspace`. |
| Chatbooks and migration | Use workspace bundles, migration receipts, and compatibility imports; they do not create a second workspace identity model. |
| ChatWorkspace and DocumentWorkspace | May link or promote resources into the server workspace model; they do not own a competing model. |

## Current State

Existing records and evidence already close the model-design portion of #1526:

- The [Canonical Workspace Model Decision](Workspace_Canonical_Model_Decision_2026_05.md)
  selected `ResearchWorkspace` as the first-slice canonical research shell.
- The [Research Workspace and Shared Workspace Model Contract](Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md)
  defines cross-domain ownership boundaries for Workspaces, Research Workspace,
  Shared Workspaces, MCP Hub, ACP, and Sandbox.
- The [Workspace Container Contract](Workspace_Container_Contract_2026_06.md)
  defines the Phase 2 server Workspace record, membership contract, runtime
  binding vocabulary, active-context eligibility rules, and surface mapping.
- The [Canonical Workspaces Manager and Project Creation design](../superpowers/specs/2026-06-04-canonical-workspaces-manager-project-creation-design.md)
  assigns canonical creation and management to `/workspaces` and defines the
  first Project Workspace root setup flow.
- The [frontend/server context contract](../superpowers/specs/2026-06-18-workspace-frontend-server-context-contract-design.md)
  records the frontend decision to unify on the server Workspace model.
- The [Workspaces manager UAT matrix](../Validation/workspaces-manager-uat-matrix.md)
  records live evidence for the canonical manager and Project Workspace handoffs.
- The [Research Workspace live UAT matrix](../Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md)
  records live evidence for the canonical research route, source lifecycle, ACP
  bridge, MCP handoff, Sandbox diagnostics, extension capture,
  migration/import/export, and current remaining Partial rows.

As of 2026-07-13, the latest full-app Research Workspace runner executed all 25
configured tests against the real application: 24 passed, one was skipped for an
environment capability, and there were no product failures. Certification
remains environment-blocked because that backend profile did not expose
`POST /api/v1/sandbox/runs`; this does not reopen the canonical model decision.

The remaining work is certification and product hardening, not a new workspace
architecture.

## Follow-Up Issues

- [#2605](https://github.com/rmusser01/tldw_server/issues/2605): certify the
  repeatable final Research Workspace UAT runner or accepted full fallback. The
  latest run has no product failures but remains environment-blocked on the
  Sandbox run API capability.
- [#2606](https://github.com/rmusser01/tldw_server/issues/2606): complete
  beginner/no-key UAT certification.
- [#2607](https://github.com/rmusser01/tldw_server/issues/2607): complete
  authenticated power-user UAT certification.
- [#2608](https://github.com/rmusser01/tldw_server/issues/2608): live recheck
  destructive and recovery actions.

All four follow-up issues remain open. Their issue records and the linked UAT
matrix are authoritative for current completion state.

## Guardrails

- Do not reintroduce `/workspace-playground`, aliases, redirects, or current UI
  copy that treats Workspace Playground as active.
- Do not use active workspace selection as a silent global filter for Library,
  Notes, Media, Artifacts, Chat, Prompts, Workflows, Watchlists, ACP, Sandbox, or
  MCP browse/search surfaces.
- Do not move ACP, MCP, Sandbox, Jobs, or owner-domain CRUD into the Workspaces
  module.
- Do not persist secrets, raw environment values, unrestricted filesystem paths,
  prompt bodies, model outputs, or file contents in workspace membership or
  public runtime-binding metadata.
- Add new follow-up issues only when implementation discovers a real contract
  gap that is not already covered by the documents or child issues above.

## #1526 Acceptance Mapping

| #1526 acceptance item | Status |
| --- | --- |
| Canonical workspace model and route strategy | The server Workspace model is canonical; `/workspaces` manages its lifecycle, `/research-workspace` is the canonical research shell, `/research` remains distinct, and Workspace Playground is compatibility-only. |
| Server/local bridge requirements | Server identity, membership, lifecycle, context, eligibility, runtime descriptors, and activity/index are authoritative; local state is cache/UI/recovery only. |
| Existing trackers linked without duplicate scope | This record links the existing May/June decisions, frontend contract, and UAT matrix instead of redefining them. |
| Follow-up issues are reviewable | #2605, #2606, #2607, and #2608 are narrow certification/recheck issues. |
