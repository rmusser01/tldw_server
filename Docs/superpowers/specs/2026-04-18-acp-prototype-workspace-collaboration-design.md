# ACP-Backed Prototype Workspace Collaboration Design

## Summary

This design adds a new product surface for collaborative prototype building on top of `tldw_server`, using ACP and Sandbox as the execution backbone instead of treating prototype generation as a static artifact feature.

The target experience is:

1. an owner creates a real arbitrary code workspace from a prompt, a template, or an existing repo/workspace
2. tldw boots that workspace in ACP/Sandbox and exposes a live preview
3. the owner shares it internally or via a private external link
4. collaborators can fully operate in the environment, but only inside isolated branch or snapshot sessions
5. collaborators see their own preview branch immediately
6. owner or designated internal collaborators can promote a candidate branch back into the canonical shared prototype

The key product decision is that collaborators do **not** directly mutate the canonical shared filesystem. V1 should prefer isolated branch sessions plus explicit promotion over live multi-user mutation of one workspace.

## Goals

- Add a first-class `Prototype Workspace` product area for collaborative app prototyping.
- Reuse ACP and Sandbox for workspace execution, terminal access, file edits, runtime management, and preview boot.
- Support arbitrary code workspaces in V1 rather than only research artifacts or constrained React outputs.
- Let owners start from:
  - a prompt-from-scratch flow
  - a template
  - an existing repo or workspace
- Support internal authenticated collaboration and private external share-link collaboration, including link expiry, revoke, and optional password controls.
- Give collaborators full workspace controls inside their isolated session branch.
- Keep the canonical shared prototype stable until an authorized promotion occurs.
- Make external collaboration demo-safe by default:
  - no inherited real secrets
  - no owner environment parity by default
  - tighter quotas and stricter runtime policies
- Extend the existing sharing model so shared workspaces become a real cross-user collaboration surface instead of a read/chat/clone-only flow.
- Make promotions auditable, explicit, and reversible at the product-model level.

## Non-Goals

- Reusing `WorkspacePlayground` as the main long-term product surface for arbitrary code workspaces.
- Letting multiple collaborators directly edit the same canonical filesystem at once in V1.
- Exposing the owner’s real secrets, local machine, or real repo checkout to collaborators.
- Building a full git-style merge and rebase UI in V1.
- Providing stable long-lived production hosting from this feature.
- Supporting general shared real-environment parity for external collaborators.
- Treating ACP or Sandbox raw APIs as the user-facing product API.
- Solving all future multi-branch collaboration workflows in V1.

## Requirements Confirmed With User

- The feature should use `tldw_server` and `tldw_frontend`, not Claude-specific infrastructure.
- The long-term goal is broader than simple artifact sharing.
- The desired product is closer to a collaborative builder for app ideas than to a bespoke design tool.
- V1 should support arbitrary workspaces rather than only a constrained React-only flow.
- The feature should tie into ACP and Sandbox for backend and execution infrastructure.
- Sharing needs to expand beyond the current limited workspace-sharing patterns.
- The first spec should target the core loop:
  - owner creates a workspace
  - owner gets a preview
  - owner shares it privately
  - collaborators prompt and operate
  - the same shared prototype outcome advances via promotion
- External access in V1 should be mixed:
  - internal authenticated users
  - external private-link users
- Collaborators in V1 should have full collaborator capability in their own session, not prompt-only access.
- Simultaneous work in V1 should use isolated branch or snapshot sessions, not direct concurrent edits on one shared filesystem.
- Shared prototype sessions should be demo-safe by default with no real secrets exposed.
- Promotion authority in V1 should belong to:
  - the owner
  - explicitly designated internal collaborators
- When an external stakeholder makes changes, they should immediately see those changes in **their own** private preview branch, while the shared canonical version remains unchanged until promotion.

## Current State

- The frontend already has a mature shared UI layer under `apps/packages/ui/src/`.
- `WorkspacePlayground` already models sources, chat, generated artifacts, and workspace sharing UI.
- The backend already exposes workspace CRUD and artifact sub-resource APIs under `/api/v1/workspaces/*`.
- The backend already exposes sharing APIs under `/api/v1/sharing/*` for:
  - team/org shares
  - private share tokens
  - public preview
  - cloning
  - shared-with-me listing
- ACP already has session concepts, terminal-oriented UI patterns, and workspace metadata such as `workspace_id`.
- Sandbox already supports:
  - isolated sessions
  - isolated runs
  - artifacts
  - workspace association
  - streaming logs
  - runtime policy controls

The repo therefore already has several reusable primitives, but they do not currently form a single collaborative prototype-builder loop. The biggest gap is that current workspace and sharing flows are research-oriented and read/chat/clone oriented, not canonical-branch-preview-promotion oriented.

## Project Decomposition

This product breaks into four related sub-projects:

1. `Prototype Workspace` domain and owner flow
2. preview/runtime orchestration for arbitrary workspaces
3. cross-user sharing and external-link collaboration
4. branch snapshot promotion and audit timeline

This spec intentionally covers the first integrated slice of that full product: the owner-to-collaborator-to-promotion loop. It does **not** attempt to fully design future merge-heavy collaboration or secret-sharing parity.

## Design Constraints Discovered During Review

### Research Workspace Mismatch Constraint

`WorkspacePlayground` already contains useful primitives, but its product model is centered on sources, notes, summaries, flashcards, and similar research artifacts. Arbitrary app workspaces and preview runtimes are a different product shape.

V1 can reuse pieces of the existing UI and sharing logic, but the canonical product model should become a new `Prototype Workspace` domain rather than stretching the current research workspace semantics too far.

### Canonical Versus Branch Visibility Constraint

The user wants collaborators to see immediate results from their own prompts while also protecting the shared prototype from uncontrolled concurrent mutation.

That forces a clear split:

- `canonical shared prototype`
- `private collaborator branch session`

The canonical prototype must stay on the last promoted snapshot until an authorized promotion occurs.

### Full Collaboration Constraint

The user explicitly wants collaborators to have full collaborator capability, not prompt-only access. That means the design must assume:

- terminal usage
- file inspection
- file editing
- runtime interaction
- preview usage

This raises the risk of conflicts and safety issues. The branch-session model is the required control boundary that keeps this feasible in V1.

### Demo-Safe Collaboration Constraint

The user wants demo-safe shared workspaces by default. Collaborators, especially external ones, must not inherit the owner’s real credentials or production-like integrations.

This means V1 must default to strict isolation and treat real secret access as explicitly out of scope.

### Promotion Authority Constraint

Not every collaborator should be able to publish to the canonical prototype. Promotion must be limited to the owner and explicitly designated internal collaborators.

This requires a product-level promotion role that is distinct from ordinary collaborator access.

### ACP/Sandbox Separation Constraint

ACP and Sandbox should remain infrastructure primitives, not the canonical product model. They are execution backends. The user-facing model should be `Prototype Workspace`, `Prototype Session`, `Prototype Snapshot`, and `Promotion`.

Without that separation, the feature becomes an ACP wrapper rather than a durable collaboration product.

### Preview Isolation Constraint

Preview state must be tied to the collaborator’s session branch, not to the canonical prototype by default. Promotion should publish a known filesystem snapshot, then boot a fresh canonical preview from that snapshot.

This avoids “it only worked because of branch-local live process state” failures.

## Approaches Considered

### Approach 1: Extend `WorkspacePlayground` Into a Prototype Builder

Reuse the existing research workspace route and incrementally add ACP-backed code execution, preview, collaborator access, and promotion.

Pros:

- Reuses existing workspace and share UI quickly
- Lower short-term routing and layout cost
- Can borrow current artifact and header patterns

Cons:

- Overloads a research-oriented product model with arbitrary code workspace concerns
- Risks long-term UX confusion between research workspaces and prototype workspaces
- Makes runtime and preview concepts feel bolted onto a source-summary-note surface

### Approach 2: Build a Dedicated `Prototype Workspace` Surface on Top of ACP/Sandbox and Extended Sharing

Create a new product domain and UI while reusing ACP/Sandbox for runtime execution and extending the sharing system for prototype resources.

Pros:

- Clean product boundary
- Matches the confirmed long-term direction
- Lets the data model reflect canonical snapshots, branch sessions, preview broker state, and promotion
- Avoids forcing general app-building into the current research workspace abstraction

Cons:

- Higher initial product-surface cost than reusing `WorkspacePlayground`
- Requires new routing, API orchestration, and metadata models

### Approach 3: Ship a Thin ACP-First Wrapper

Treat the feature mostly as ACP session creation plus preview URLs plus minimal collaboration metadata.

Pros:

- Shortest path to an internal demo
- Minimal initial orchestration layer

Cons:

- Weak persistence model
- Weak promotion model
- Poor long-term audit and sharing semantics
- Likely to require a large rewrite once real stakeholders use it

## Recommendation

Use **Approach 2**.

Build a dedicated `Prototype Workspace` product surface that reuses ACP and Sandbox as runtime infrastructure and extends the current sharing system to new resource types. Reuse existing UI primitives where they help, but do not make `WorkspacePlayground` the canonical model for arbitrary collaborative app-building.

## Proposed Architecture

### Product Shape

V1 adds a new top-level product area for `Prototype Workspaces`.

The owner’s canonical surface includes:

- canonical preview
- branch and session inventory
- prompt and run timeline
- promotion controls
- sharing controls
- role and promoter management

The collaborator surface includes:

- the collaborator’s own branch preview
- prompt/chat panel
- terminal and file views
- diff or activity view
- a promotion-request action

### Core Domain Objects

- `Prototype Workspace`
  - canonical shared prototype record
  - owner metadata
  - share policy
  - preview policy
  - current promoted snapshot
  - designated internal promoters
- `Prototype Snapshot`
  - immutable saved filesystem state with provenance
  - parent snapshot linkage
  - author and session metadata
  - diff summary
  - preview health summary
- `Prototype Session`
  - a mutable collaborator working branch derived from a snapshot
  - linked ACP session/runtime state
  - linked preview state
  - actor identity and access context
- `Prototype Shared Actor`
  - a first-class audited external collaborator principal bound to a private share-link entry flow
  - not a full AuthNZ user
  - carries revocation state, quota policy, and audit identity for one external collaborator
- `Promotion Request`
  - a candidate branch/snapshot waiting for review, approval, rejection, or direct promotion

### Architectural Layers

- `Prototype Workspace domain`
  - canonical product model and permissions
- `Snapshot/Branch layer`
  - immutable snapshots plus isolated collaborator working branches
- `ACP/Sandbox runtime layer`
  - command execution, terminal, file edits, dev-server lifecycle, job execution
- `Preview broker`
  - preview URL issuance, runtime health, port binding abstraction, TTL and revocation
- `Sharing and access layer`
  - internal auth grants plus external private links
- `Audit and timeline layer`
  - branch creation, prompts, runs, preview state, promotion actions, revokes

### Canonical And Branch Preview Model

There are always two preview concepts:

- `canonical preview`
  - tied to the currently promoted snapshot
  - this is the shared prototype the owner is publishing
- `branch preview`
  - tied to one collaborator’s isolated session branch
  - this updates immediately as the collaborator works

Promotion does **not** adopt a running process wholesale. Promotion publishes a known snapshot into the canonical record and boots or refreshes the canonical preview from that promoted snapshot.

### External Collaborator Identity Model

External private-link access needs a real collaboration principal, not just a validated share token. V1 should introduce a `Prototype Shared Actor` model for external collaborators.

Flow:

1. external user opens a private prototype share link
2. link password and expiry policy are verified through the existing share-token path
3. system issues a short-lived collaboration session token and materializes or resumes a `Prototype Shared Actor`
4. all ACP/Sandbox sessions, snapshots, preview access, and audit events attach to that `Prototype Shared Actor`

This avoids overloading `user_id`-only assumptions in current sharing and ACP flows. It also gives V1 a concrete place to enforce:

- per-external-collaborator quotas
- link-level revocation
- actor display name or lightweight guest identity
- audit history without requiring full account creation

`Prototype Shared Actor` should be the only identity allowed to back unauthenticated external collaboration in V1. Public preview tokens alone are not sufficient for full branch-session creation.

## Roles And Permissions

### Owner

- create and delete prototype workspaces
- configure sharing
- designate internal promoters
- create or revoke collaboration access
- view all sessions, branches, and requests
- promote any eligible candidate branch

### Internal Collaborator

- authenticate into a shared prototype workspace
- create an isolated branch session from the current canonical snapshot
- use prompt, terminal, file, and preview tools in that branch
- submit a promotion request
- promote only if explicitly designated as a promoter

### External Collaborator

- enter via a private external share link
- create and operate inside an isolated branch session
- view only their branch preview and permitted metadata
- submit promotion requests
- cannot promote directly
- cannot become a promoter in V1
- cannot access secrets or owner-only environment controls
- is represented internally as a `Prototype Shared Actor`, not a full user record

### Suggested Permission Shape

- `prototype.view`
- `prototype.collaborate`
- `prototype.request_promotion`
- `prototype.promote`
- `prototype.manage_shares`
- `prototype.manage_runtime_policy`

The `prototype.promote` permission should be owner-only by default and extend only to designated internal collaborators.

## Session And Promotion Flow

### Owner Flow

1. Owner creates a `Prototype Workspace` from:
   - prompt
   - template
   - existing repo/workspace
2. System scaffolds or imports the workspace through ACP/Sandbox-backed jobs.
3. System boots the canonical preview.
4. Owner configures sharing and optional promoter assignments.

### Collaborator Flow

1. Collaborator opens the shared prototype through internal auth or a private external link.
2. System creates a new isolated `Prototype Session` from the current promoted snapshot.
3. Collaborator works inside that branch via prompts, file edits, terminal operations, and preview interactions.
4. Their brokered preview access reflects their branch state only.
5. Collaborator requests promotion when ready.

### Promotion Flow

1. Owner or designated internal promoter reviews the candidate branch.
2. System validates that:
   - the branch belongs to the prototype workspace
   - the branch snapshot is durable
   - the candidate is not stale against the canonical pointer
3. System runs a publish-validation job against the candidate snapshot in a fresh canonical runtime:
   - restore snapshot into a clean runtime
   - execute required bootstrap or install steps for the workspace profile
   - boot the preview through the canonical preview broker path
   - verify preview health and any required build checks
4. Only after publish validation succeeds:
   - canonical promoted snapshot pointer is updated atomically
   - audit log records the event
   - canonical preview is rebuilt or refreshed from the promoted snapshot

V1 should not support force-publishing a failed candidate into the canonical prototype. The safe default is that only a candidate that can boot cleanly from snapshot in a fresh runtime is promotable.

### Out-Of-Date Handling

If the canonical prototype advanced while a collaborator was still editing, promotion should fail with a clear stale-state result. V1 should prefer explicit restart or later rebase/replay flows over silent overwrite.

## Data Model

### `prototype_workspaces`

Recommended fields:

- `id`
- `owner_user_id`
- `title`
- `description`
- `creation_source`
  - `prompt`
  - `template`
  - `existing_workspace`
- `canonical_snapshot_id`
- `last_known_good_snapshot_id`
- `canonical_preview_status`
- `publish_validation_status`
- `preview_policy_json`
- `share_policy_json`
- `runtime_policy_json`
- `designated_promoter_ids_json`
- `created_at`
- `updated_at`
- `archived_at`

### `prototype_snapshots`

Recommended fields:

- `id`
- `prototype_workspace_id`
- `parent_snapshot_id`
- `created_from_session_id`
- `author_user_id`
- `author_shared_actor_id`
- `storage_ref`
- `diff_summary_json`
- `prompt_summary`
- `preview_health_json`
- `created_at`

### `prototype_shared_actors`

Recommended fields:

- `id`
- `prototype_workspace_id`
- `share_link_id`
- `display_name`
- `session_binding_id`
- `runtime_policy_profile`
- `quota_policy_json`
- `last_activity_at`
- `expires_at`
- `revoked_at`
- `created_at`
- `updated_at`

### `prototype_sessions`

Recommended fields:

- `id`
- `prototype_workspace_id`
- `base_snapshot_id`
- `actor_user_id`
- `actor_shared_actor_id`
- `actor_type`
  - `owner`
  - `internal_collaborator`
  - `external_collaborator`
- `share_link_id`
- `acp_session_id`
- `sandbox_session_id`
- `sandbox_run_id`
- `runtime_status`
- `preview_handle`
- `preview_status`
- `last_saved_snapshot_id`
- `last_activity_at`
- `expires_at`
- `revoked_at`

### `prototype_promotion_requests`

Recommended fields:

- `id`
- `prototype_workspace_id`
- `prototype_session_id`
- `candidate_snapshot_id`
- `requested_by_user_id`
- `requested_by_shared_actor_id`
- `status`
  - `pending`
  - `approved`
  - `rejected`
  - `promoted`
  - `stale`
- `reviewed_by_user_id`
- `review_notes`
- `created_at`
- `updated_at`

Identity rule:

- internal actors use `*_user_id`
- external actors use `*_shared_actor_id`
- V1 should enforce that exactly one actor identity column is populated for snapshot authorship, session ownership, and promotion requests

### Storage Recommendation

- Keep product metadata in the shared AuthNZ and ops-side persistence layer so cross-user access and auditing are first-class.
- Treat the shared ops/AuthNZ-backed prototype tables as the single source of truth for `Prototype Workspace`, sessions, shared actors, and promotion state.
- If a prototype starts from an existing research workspace, import or snapshot its contents into the prototype domain once, then decouple it. Do not dual-write canonical prototype state back into per-user ChaChaNotes workspace tables.
- Keep filesystem snapshot payloads in storage owned by ACP/Sandbox orchestration.
- Treat ACP/Sandbox identifiers as transient runtime attachments, not as the canonical product identity.

## Runtime And Safety Boundaries

### Demo-Safe Default

Every collaborator session should start in a demo-safe policy profile:

- no inherited owner secrets
- no implicit provider credentials
- no local-machine access
- no direct access to the owner’s real repo checkout

### Runtime Isolation

- each collaborator gets a fresh ACP/Sandbox-backed workspace
- session filesystems are disposable but snapshotable
- runtime state is ephemeral
- durable promotion uses snapshots, not running processes

### Runtime Profiles

Supporting arbitrary workspaces in V1 requires explicit runtime profiles instead of one blanket demo-safe mode.

Recommended profiles:

- `template_demo`
  - for prompt-from-scratch or trusted starter templates
  - bootstraps from pre-approved base images and known dependency sets
- `repo_bootstrap`
  - for existing repo or imported workspace boot
  - allows temporary, allowlisted egress only during controlled scaffold/import/build jobs
  - examples: package registries, trusted git hosts, language-specific dependency mirrors
- `locked_collab`
  - default for external collaborator branch sessions
  - no secret inheritance
  - no arbitrary internet egress
  - assumes dependencies are already materialized in the snapshot or base image

This means V1 can honestly support arbitrary workspaces while still remaining demo-safe:

- provisioning and import jobs may need a broader but allowlisted runtime profile
- external collaboration sessions remain intentionally narrower
- actions inside a locked collaborator session that require new networked dependency resolution should fail closed and instruct the owner to reprovision or rebuild from an internal flow

### Network And Quotas

External collaborators should get stricter default runtime policy than internal authenticated collaborators:

- shorter session TTLs
- lower CPU and memory ceilings
- narrower egress policy
- more aggressive preview idle shutdown

### Secret Handling

V1 should explicitly reject real secret inheritance and real-integration parity. A collaborator session that needs unavailable secrets should fail closed with a clear policy error.

### Preview Access Model

Preview access should be brokered through tldw, not exposed as raw runtime endpoints.

Requirements:

- canonical and branch previews are addressed by `preview_handle`, not by directly exposing underlying dev-server URLs
- the preview broker issues short-lived signed preview URLs or proxy sessions
- preview access is bound to the authenticated internal user or `Prototype Shared Actor`
- share revocation invalidates future preview access immediately and causes existing preview grants to age out quickly

This is especially important for external links. A leaked raw runtime URL would bypass the collaboration permission model and make revocation weak.

## API And Product Surface

### Frontend Routes

Suggested new route family:

- `/prototype-workspaces`
- `/prototype-workspaces/:id`
- `/prototype-workspaces/:id/sessions/:sessionId`
- internal collaborator entry routes as needed
- external private-link entry routes as needed

### Frontend Surfaces

Owner view:

- canonical preview panel
- active branch sessions
- candidate branch list
- run and prompt timeline
- sharing controls
- promotion controls

Collaborator view:

- branch preview
- prompt/chat panel
- terminal/files/diffs
- branch activity
- submit-for-promotion control

### Backend Endpoint Families

Recommended new endpoint families:

- `/api/v1/prototype-workspaces/*`
  - create
  - list
  - get
  - update
  - archive
  - manage roles and promoters
- `/api/v1/prototype-sessions/*`
  - create branch session
  - resume
  - get status
  - save snapshot
  - close or revoke
- `/api/v1/prototype-previews/*`
  - preview status
  - preview health
  - preview URL issuance or renewal
- `/api/v1/prototype-promotions/*`
  - request promotion
  - review request
  - promote candidate snapshot
  - reject or mark stale

### Sharing Extension

The existing sharing system should be extended so it can target prototype resources. The current sharing logic already provides useful patterns for:

- internal grants
- private links
- expiry
- revocation
- optional password protection
- audit logging

But `prototype_workspace` should become a first-class share resource rather than trying to masquerade as a research workspace or chat share.

The sharing extension also needs one new capability beyond current preview/import flows:

- exchange a valid private share link for a prototype-collaboration session bound to a `Prototype Shared Actor`

## Jobs And Execution Strategy

User-visible long-running work should use **Jobs** rather than ad hoc background tasks. Recommended job-backed operations:

- scaffold prototype workspace
- import existing repo or workspace
- create collaborator branch session
- boot or restart preview
- save durable snapshot
- promote candidate snapshot

ACP and Sandbox remain the execution engines beneath these jobs. The `Prototype Workspace` domain should orchestrate them rather than exposing raw infrastructure concerns to end users.

### Idempotency And Cleanup Rules

These jobs need explicit idempotency and cleanup semantics in V1, because retries can otherwise leak ACP/Sandbox resources or publish inconsistent state.

Recommended rules:

- `scaffold/import`
  - idempotency key scoped to `prototype_workspace_id + creation_source fingerprint`
  - retries should reuse or reconcile the same bootstrap job
- `create branch session`
  - idempotency key scoped to `prototype_workspace_id + actor principal + base_snapshot_id + request nonce`
  - duplicate retries should return the existing live branch session when possible
- `boot preview`
  - idempotency key scoped to `session_id + snapshot_id + runtime profile version`
  - only one active preview broker target per scope
- `save snapshot`
  - idempotency key scoped to `session_id + save request id`
  - repeated save calls should return the same durable snapshot if content and lineage match
- `promote`
  - idempotency key scoped to `prototype_workspace_id + candidate_snapshot_id + canonical_snapshot_id at review time`
  - retries must not republish a newer canonical pointer accidentally

Cleanup expectations:

- retries must reconcile or terminate orphan ACP sessions, sandbox sessions, and preview processes before creating new ones
- promotion must be modeled as a state machine, not a single best-effort step:
  - `queued`
  - `validating`
  - `ready_to_publish`
  - `publishing`
  - `published`
  - `failed`

## Failure Handling

### Branch Preview Failure

If a collaborator branch preview fails to boot, that branch is marked unhealthy. The canonical prototype remains unaffected.

### Session Expiry

If a collaborator session expires:

- saved durable snapshots may remain reviewable or promotable based on policy
- unsaved ephemeral branch state is discarded

### Stale Promotion

If canonical advanced since the collaborator’s branch was created, promotion must fail with a structured stale result rather than silently overwriting newer canonical state.

### Share Revocation

If an internal grant or external link is revoked:

- new control-plane actions are denied immediately
- session renewal is blocked
- preview URLs should age out quickly

### Publish Validation Failure

If a candidate snapshot cannot boot cleanly in a fresh canonical runtime, promotion remains failed and the canonical pointer must stay on the last known good snapshot. The failure should be visible to reviewers as a publish-validation error, not as a silent preview regression after promotion.

### Runtime Policy Denial

If a collaborator action requires secrets, network, or capabilities outside the demo-safe policy, the system should reject it explicitly instead of letting it fail ambiguously at runtime.

## Observability And Audit

V1 should record structured audit and product events for:

- prototype workspace created
- branch session created
- share opened
- prompt submitted
- command or run executed
- preview booted or failed
- snapshot saved
- promotion requested
- promotion approved, rejected, or marked stale
- grant revoked or link expired

The owner should have a readable timeline view that explains who changed what and when.

## Testing Strategy

### Unit Tests

- permission rules for owner, internal collaborator, external collaborator, and promoter roles
- branch and snapshot invariants
- stale-promotion detection
- preview policy and demo-safe runtime policy resolution
- share-link and grant policy resolution for prototype resources

### Integration Tests

- prototype workspace lifecycle
- branch session creation from canonical snapshot
- preview boot and restart flows
- snapshot save and lineage correctness
- promotion pointer update and canonical preview refresh
- revocation and expiry behavior

### End-To-End Tests

Core loop:

1. owner creates prototype workspace
2. canonical preview boots
3. collaborator opens share and receives an isolated branch
4. collaborator edits and sees their own preview update
5. collaborator submits a candidate
6. owner or designated internal promoter promotes it
7. canonical preview updates to the promoted snapshot

### Security And Boundary Tests

- external collaborator cannot access owner-only controls
- external collaborator must resolve to a `Prototype Shared Actor`, not an internal user principal
- external collaborator receives demo-safe runtime policy
- no real secret inheritance
- revoked links cannot create new active sessions
- revoked preview grants cannot continue receiving fresh signed preview access
- stale candidates cannot silently overwrite canonical state
- failed publish validation cannot advance the canonical snapshot pointer

## Open Questions For Later Phases

- whether later versions should support optional narrowly scoped shared secrets
- whether internal collaborators should get richer compare-and-review tooling before promotion
- whether V2 should add merge or replay flows instead of simple stale rejection
- whether canonical preview should support version pinning and historical rollback UI
- whether some branch sessions should support persistent branch URLs beyond session lifetime
- whether V2 should offer an owner-only force-publish override for intentionally broken but reviewable states

## Recommendation On Implementation Order

Recommended implementation order after this design:

1. create the new `Prototype Workspace` domain and owner canonical-preview flow
2. add branch-session creation and isolated collaborator previews on ACP/Sandbox
3. extend sharing to prototype resources for internal auth and private external links
4. add promotion requests, promoter roles, and canonical promotion flow

This order preserves the user’s desired product loop while keeping runtime, safety, and publishing boundaries explicit from the beginning.
