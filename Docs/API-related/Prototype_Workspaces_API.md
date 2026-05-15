# Prototype Workspaces API

## Overview

`Prototype Workspace` is the app-prototyping collaboration surface built on top of the existing ACP and Sandbox infrastructure.

The current slice supports:

- owner-created prototype workspaces with a canonical snapshot
- isolated owner and external collaborator branch sessions
- private-link exchange into a `Prototype Shared Actor`
- brokered preview grants and renewals
- promotion request submission and owner review
- a minimal WebUI route at `/prototype-workspaces`

The product model stays explicit:

- `Prototype Workspace`: the canonical artifact and promotion boundary
- `Prototype Session`: an isolated owner or collaborator branch
- `Prototype Snapshot`: a saved revision in the workspace lineage
- `Prototype Shared Actor`: an external collaborator identity created from a private link
- `Promotion Request`: a request to move a candidate snapshot into the canonical line

ACP and Sandbox remain execution backends, not the user-facing product model.

## Lifecycle

1. The owner creates a prototype workspace with a title, prompt, and runtime/share policy.
2. The service seeds the workspace with an initial canonical snapshot.
3. The owner can request an owner branch session for guided work on top of the canonical snapshot.
4. The owner can create a private share token for `prototype_workspace`.
5. A stakeholder opens `/share/:token`, verifies any password, and exchanges the token for a `Prototype Shared Actor` session token.
6. The collaborator uses that session token to create or reuse an isolated branch session.
7. The collaborator produces a candidate snapshot and submits a promotion request.
8. The owner reviews the request and either rejects it or promotes the candidate after validation.
9. Preview access is always brokered through a preview handle rather than exposing raw runtime URLs.

## Runtime Profiles

The design references several runtime policy profiles:

- `template_demo`: constrained demo-oriented runtime for seeded prototypes
- `repo_bootstrap`: profile for repository-backed scaffolds or richer bootstrap flows
- `locked_collab`: the default external collaborator runtime profile

The current code path actively uses these runtime-policy keys:

- `runtime_policy.owner_profile`
- `runtime_policy.external_collaborator_profile`
- preview-broker defaults such as `canonical_preview` and `owner_collab`

Important implementation note:

- external collaborator exchange defaults to `locked_collab` when no explicit external profile is set
- preview brokering resolves owner vs external runtime profiles from workspace/runtime context instead of trusting the client

## API Surface

### Create workspace

`POST /api/v1/prototype-workspaces`

Creates a new prototype workspace and its seed canonical snapshot.

Request highlights:

- `title`
- `creation_source`
- `description`
- `prompt`
- `preview_policy`
- `share_policy`
- `runtime_policy`
- `designated_promoter_ids`

Response highlights:

- workspace metadata
- canonical and last-known-good snapshot ids
- canonical preview and publish-validation state

### Get owner detail

`GET /api/v1/prototype-workspaces/{prototype_workspace_id}`

Owner-only detail response used by the minimal WebUI.

Response includes:

- workspace metadata
- `viewer_role`
- branch-session inventory
- snapshot inventory
- canonical and last-known-good markers on snapshots

### Create owner branch session

`POST /api/v1/prototype-workspaces/{prototype_workspace_id}/sessions`

Owner-only route that creates or reuses a branch session rooted at the canonical snapshot and enqueues bootstrap work.

Response highlights:

- `job_id`
- `job_type = "branch_session_bootstrap"`
- `prototype_session_id`
- `actor_type = "owner"`

### Create collaborator branch session

`POST /api/v1/prototype-sessions`

Consumes a collaborator `session_token` minted by the private-link exchange flow and creates or reuses the stakeholder branch session.

Response highlights:

- `prototype_workspace_id`
- `prototype_session_id`
- `shared_actor_id`
- `actor_type = "external_collaborator"`

### Submit promotion request

`POST /api/v1/prototype-promotions`

Creates a promotion request for a candidate snapshot.

Server-side validation binds the request to:

- the prototype workspace in the session token
- the branch session referenced by the request
- the candidate snapshot lineage
- the `Prototype Shared Actor` that owns the branch and snapshot

### Review promotion request

`POST /api/v1/prototype-promotions/{promotion_request_id}/review`

Owner-only approval or rejection flow.

Approval path:

- validates the requested baseline
- runs promotion validation
- updates canonical pointers only on success
- can return a refreshed preview handle for the promoted snapshot

Rejection path:

- records reviewer id and notes
- does not advance canonical state

### Renew preview grant

`POST /api/v1/prototype-previews/{preview_handle}/renew`

Renews a brokered preview grant for an active preview handle.

Response highlights:

- short-lived preview token
- preview URL
- expiry
- resolved runtime policy profile

### Private-link exchange

`POST /api/v1/sharing/public/{token}/prototype-session`

Turns a private share link into a `Prototype Shared Actor` and collaborator session token.

Request highlights:

- `display_name`
- `password` when the token is protected

Response highlights:

- `shared_actor_id`
- `actor_type = "external_collaborator"`
- `session_token`
- `runtime_policy_profile`

## Prototype Shared Actor Model

External stakeholders do not become first-class AuthNZ users.

Instead, the exchange flow creates or resumes a `Prototype Shared Actor` that is scoped to:

- one prototype workspace
- one share-link id
- a session binding / resume cookie chain
- an external runtime policy profile

This preserves auditability and keeps owner-only controls separate from collaborator capabilities.

## Risk Gate 1 Security Contract

The current productionization security model is documented in `Docs/Security/Prototype_Workspaces_Threat_Model.md`.

The short version:

- owner APIs must re-check AuthNZ owner or designated-promoter authority server-side
- public share exchange must keep invalid, expired, revoked, exhausted, missing, and mismatched link states non-enumerating where possible
- external shared actors are scoped to one prototype workspace and one share-link id
- collaborator session tokens must be rechecked against active shared-actor state, not trusted only because their signature is valid
- resume cookies are signed browser-binding hints and rotate their stored binding secret on successful resume
- preview grants are short-lived broker-issued grants behind opaque handles
- promotion submission must bind the token, active shared actor, branch session, candidate snapshot, and workspace before creating a request

Frontend/backend state names, HTTP status expectations, retryability, and Risk Gate 4 open questions live in `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`.

## Risk Gate 2 Persistence Contract

Prototype workspace persistence is owned by the AuthNZ repository boundary in `PrototypeWorkspacesRepo`.

Transaction rules:

- workspace creation and seed snapshot creation run in one AuthNZ transaction
- session snapshot save and session pointer update run in one AuthNZ transaction
- publish promotion still compensates around preview-broker side effects because preview grants include in-memory broker state plus persisted handles
- repository transaction adapters preserve the existing `?` placeholder style and convert to PostgreSQL `$N` placeholders when the underlying `DatabasePool` is PostgreSQL-backed

Cleanup and retention rules:

- archived workspaces are soft-archived first and can be deleted after the configured archive-retention cutoff; cascading foreign keys remove related prototype rows
- expired shared actors are soft-revoked by setting `revoked_at`; they are retained for audit until their workspace is deleted
- expired sessions are soft-revoked by setting `revoked_at`, `runtime_status = "revoked"`, and `preview_status = "revoked"`
- active preview handles attached to archived workspaces or revoked/expired sessions are deactivated and receive `revoked_at`
- inactive preview handles can be deleted after their inactive-preview cutoff
- old pending promotion requests can be marked `stale`; approved, rejected, promoted, or already stale requests are not rewritten by cleanup

SQLite/PostgreSQL behavior:

- migration 086 is the SQLite migration source for the prototype tables
- PostgreSQL compatibility is through the AuthNZ `DatabasePool` execution contract and repository table discovery via `information_schema`
- all repository SQL remains parameterized; no user input is interpolated into SQL strings
- cleanup accepts explicit cutoff timestamps so callers own retention policy and scheduling

Query/index review:

| Path | Query shape | Index coverage |
| --- | --- | --- |
| owner workspace detail | `prototype_workspaces.id`, workspace snapshots by `prototype_workspace_id` | primary key plus `idx_prototype_snapshots_workspace_created` |
| branch-session inventory | sessions by `prototype_workspace_id`, active first, ordered by update time | `idx_prototype_sessions_workspace_active_updated` |
| active session reuse | workspace, base snapshot, actor type, actor identity, share link, revoked/expiry/runtime filters | `idx_prototype_sessions_active_lookup` |
| active shared actor check | workspace-scoped actor activity and expiry checks | primary key plus `idx_prototype_shared_actors_active_lookup` |
| promotion listings/review | workspace and status with update-time cleanup | `idx_prototype_promotion_requests_workspace_status_updated` |
| preview handle lookup | handle id, active scope replacement, workspace/session cleanup scans | primary key plus preview handle workspace/session/scope indexes |

## Preview Broker Guarantees

Preview access is brokered through `preview_handle` records and signed preview grants.

Guardrails:

- raw runtime target URLs are not exposed to clients
- stale or revoked preview handles do not renew successfully
- grants are short-lived and scoped to a workspace/session/snapshot tuple
- runtime policy resolution happens server-side

## Promotion And Publish Guarantees

Promotion is intentionally explicit and validation-gated.

Guardrails:

- only the owner or a designated promoter can approve
- candidate snapshots must belong to the same workspace
- candidate snapshots must belong to the referenced branch session
- stale baselines do not silently overwrite canonical state
- failed validation does not update canonical or last-known-good pointers

## Minimal WebUI

Current frontend surfaces:

- option route: `/prototype-workspaces`
- public share route: `/share/:token`

Owner flow includes:

- workspace creation
- canonical preview state
- owner branch-session creation
- private-link generation
- branch inventory
- candidate snapshot inventory

Collaborator flow includes:

- private-link exchange
- collaborator branch-session creation
- candidate snapshot selection
- promotion request submission

## Verification Notes

Verified in the current implementation worktree with:

- `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q`
- `python -m pytest tldw_Server_API/tests/Sharing/test_sharing_endpoints.py -q`
- `bunx vitest run src/hooks/__tests__/usePrototypeWorkspaces.test.tsx src/hooks/__tests__/useSharing.auth.test.tsx src/routes/__tests__/option-prototype-workspaces.route.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx`
- `python -m bandit -f json -o /tmp/bandit_prototype_task6.json tldw_Server_API/app/api/v1/endpoints/prototype_workspaces.py tldw_Server_API/app/api/v1/schemas/prototype_workspace_schemas.py`
