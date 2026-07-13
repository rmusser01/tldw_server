# User-Customizable Service Prompts Design

**Date:** 2026-07-12

**Status:** Approved in brainstorming; amended after three independent review passes; pending human final approval

**Backlog:** TASK-12112

**Related:** TASK-2341 (broader shared prompt-registry follow-up)

## Summary

tldw_server should let each authenticated user customize a curated set of prompts that backend services apply automatically when creating or analyzing content. The capability will appear in a dedicated **Service Prompts** settings page shared by the WebUI and browser-extension options app.

The selected architecture is a typed `ServicePromptRegistry` backed by immutable server defaults and versioned per-user overrides. Services resolve prompts through one governed resolver rather than reading files or embedding strings directly. The design preserves existing explicit request overrides and deployment prompt files, validates prompt variables, supports deterministic previews, keeps revision history, pins queued work to an immutable full-bundle pin set, and integrates with the existing context-integrity system.

This feature is deliberately separate from:

- The reusable Prompt Library and Prompt workspace
- Prompt Studio projects, experiments, and evaluations
- Conversation-specific system prompts and `preferences.chat.system_prompt`
- MCP prompt-catalog exposure
- Security, authorization, routing, judge, and machine-protocol prompts

The broad content-facing rollout is one product initiative but must be delivered as multiple Backlog tasks and reviewable pull requests.

## Context

The current codebase has several overlapping prompt mechanisms:

- `tldw_Server_API/Config_Files/Prompts` contains editable YAML and Markdown defaults loaded through `prompt_loader`.
- `TLDW_PROMPT_FILE_<MODULE>__<KEY>` environment variables can replace individual prompt assets at deployment time.
- User-authored reusable prompts live in per-user Prompts databases and are managed through `/api/v1/prompts` and the Prompt workspace.
- Prompt Studio has a separate project, prompt, test, evaluation, and revision model.
- Some service prompts remain embedded directly in Python or TypeScript.
- Many request schemas already accept explicit `system_prompt` or `custom_prompt` values.
- Context-integrity enforcement now protects prompt files and database prompt versions at use time.

The WebUI and extension already share route, settings, service, and component code under `apps/packages/ui/src`. The extension options application and Next.js WebUI can therefore use one implementation. The narrow extension sidepanel should deep-link to the full options editor rather than host the editor itself.

## Goals

1. Give each user one active override for every eligible service-prompt definition.
2. Apply that override to the user's WebUI, extension, API, synchronous, scheduled, and background activity on the same server.
3. Keep shipped and deployment-managed defaults immutable from the user settings surface.
4. Preserve explicit per-request prompts as the highest-precedence user choice.
5. Validate required variables and locked output contracts before activation.
6. Provide safe preview, comparison, reset, history, restore, and upstream-default awareness.
7. Keep already queued work reproducible after later edits or server upgrades.
8. Migrate the broad set of eligible content-facing prompts, including currently hardcoded prompts.
9. Preserve multi-user isolation, context-integrity enforcement, and operational privacy.
10. Maintain no-override behavior at byte-equivalent LLM-provider message boundaries unless a behavior change is separately approved.

## Non-Goals

- A user-facing editor for authentication, authorization, safety policy, tool permissions, provider routing, evaluator/judge, or prompt-enforced machine-protocol instructions
- Administrator or organization-wide prompt overrides in this release
- Multiple named variants per service slot
- Binding service slots to reusable Prompt Library records
- Editing Prompt Studio artifacts
- Live LLM test calls or side-by-side model evaluations from the settings editor
- Automatic merges between a user override and a changed server default
- Persisting unsaved drafts automatically on a browser device
- Treating prompt text as secret or credential storage

## Product Decisions

- Exposure uses a curated allowlist, not a denylist.
- Ownership is per-user. Existing deployment-file overrides remain compatible but are not a new admin UI feature.
- Each definition has one active user override.
- Coordinated prompt parts are edited and versioned atomically.
- Required variables are strict; softer quality diagnostics are warnings.
- Preview uses deterministic safe sample values and never calls an LLM.
- Saved overrides are server-side and account-scoped. Saving creates a pending revision; model use requires a separate explicit operator approval under the existing context-integrity policy.
- Full revision history is retained subject to normal storage quotas and lifecycle policy.
- User overrides remain pinned when the server default changes.
- Queued work pins its prompt at enqueue time.
- The first product release targets the broad content-facing eligibility set.
- The settings route is `/settings/service-prompts`.

## Architecture

### ServicePromptRegistry

`ServicePromptRegistry` is the canonical allowlist of editable service-prompt definitions. A definition represents one workflow contract, not necessarily one string. For example, `web.article_summary` may contain coordinated `system` and `user` parts.

Each typed definition declares:

- Stable definition ID and schema version
- Localization label and description keys with English fallbacks
- Primary category, searchable tags, and affected workflow identifiers
- Named prompt parts and their assembly order
- Immutable packaged/default asset references for each part
- Compatibility mappings for existing prompt-loader module/key pairs and environment override variables
- Required and optional variables
- Safe deterministic sample values
- Per-template, per-variable, repetition, and assembled-size budgets
- Whether oversize runtime values are rejected or deterministically truncated
- Editable sections and server-managed locked sections
- Locked-section visibility: `visible` or `hidden`
- Output-contract classification and eligibility evidence
- Sensitivity, deprecation, and replacement metadata

Startup validation rejects duplicate IDs, missing assets, invalid part mappings, undeclared placeholders, invalid sample values, contradictory size budgets, or incomplete replacement metadata.

### Atomic definitions and locked assembly

System/user pairs and other coordinated fragments are one definition and one revision. Users cannot save only half of a coordinated contract.

Editable text is not always the complete prompt. The server may prepend or append locked fragments. Visible contract fragments are shown read-only. Hidden server-managed fragments are represented in preview and provenance without returning their bodies.

A structured-output workflow is eligible only when the output schema is enforced independently, such as by provider response-format support or deterministic validation and retry. If downstream correctness relies primarily on prompt wording, the prompt remains fully locked.

### PromptExecutionContext

Consumers resolve prompts using a typed context rather than a bare user ID. It contains:

- Canonical owner identity, when user-owned
- Operation/workflow identifier
- Optional explicit overrides keyed by named editable part
- For each explicit part, its kind: `literal` or `template`
- Optional immutable prompt pin set
- Request, trace, and job identifiers as safe metadata

Definitions declare which named parts map to legacy fields such as `system_prompt` and `custom_prompt`, whether each part accepts literal or template input, and how omitted parts are filled. A partial explicit override replaces only its named editable parts. Other editable parts continue through normal precedence, while locked parts can never be replaced.

`literal` means the supplied text is used as the final value for that editable part without placeholder parsing. `template` means the constrained renderer substitutes the definition's declared variables. The migration inventory must preserve the semantics of every existing explicit field rather than guessing from its name.

Userless maintenance activity deliberately uses the server default. Store errors or missing ownership never cause an accidental cross-user or default fallback.

### Resolution precedence

The resolver selects each named editable part in this order:

1. Explicit request override
2. Authorized pinned bundle snapshot
3. Active per-user override
4. Deployment default, including existing environment/file overrides
5. Packaged default

An active user override is atomic: when selected, it supplies its complete set of editable parts. Explicit overrides may then replace a declared subset part by part. Locked parts always come from the resolved server-managed definition.

All parts are resolved before the resolver computes the atomic server-default bundle digest. That bundle digest drives revision provenance and upstream-change detection. `bypass_stored_overrides` skips only stored user overrides; it does not suppress pre-existing explicit request fields. This preserves explicit overrides as the highest-precedence request choice.

The immutable `ResolvedServicePrompt` contains rendered or render-ready named parts plus safe provenance:

- Definition ID and schema version
- Per-part source kinds
- User revision and explicit-override provenance, when applicable
- Server-default bundle digest
- Canonical content digest
- Assembly order and locked-section markers

The resolver runs once per request, batch, or job. Lower-level loops receive the immutable result and do not perform repeated database lookups.

### Constrained renderer

The renderer accepts only simple declared placeholders such as `{context}`. It supports escaped literal braces but rejects:

- Attribute access
- Indexing
- Expressions
- Format specifiers
- Filters
- Function calls
- Arbitrary code or template control flow

It uses a linear parser, reports part/variable/line/column diagnostics without echoing surrounding prompt text, and enforces template, variable, repetition, and final assembly budgets. Save, preview, synchronous execution, and worker execution use the same validation and rendering implementation.

## Persistence

Persistent user-authored service-prompt state and revisions live behind a dedicated repository using each user's existing Prompts database boundary. Execution-only snapshots and pin sets live in a separate server-managed protected execution store because they may contain hidden deployment instructions or request-bound literal content. Neither record family appears in ordinary Prompt search, Prompt Studio, MCP prompt listings, or Prompt exports.

The conceptual data model contains the following record types. The implementation plan may consolidate physical tables if it preserves these contracts.

### Service prompt state

One mutable state record per definition contains:

- Definition ID
- Active revision ID or null
- Current pending revision ID or null
- Monotonically increasing generation
- Last acknowledged server-default digest
- Updated timestamp

The generation changes for pending save, approval, rejection/supersession, restore, reset, and default-change acknowledgement. It is the optimistic-concurrency token even when the active revision remains null. A user may have at most one current pending revision per definition; a newer pending save explicitly supersedes the previous pending revision without activating either one. After the resolver verifies a trusted server default, reset supersedes any pending revision and clears the active revision in the same repository transaction.

A separate per-user `catalog_generation` metadata value increments in the same transaction as every definition-state or pending-state change. The catalog ETag combines the current registry/server-default catalog digest, capability/operator mode, and this user catalog generation. This gives the list endpoint an O(1) invalidation token without relying on timestamps or scanning every revision.

### Immutable revisions

Each revision contains:

- Revision ID and sequence number
- Definition ID
- Atomic editable parts
- Origin action: save or restore
- Registry schema version
- Base server-default bundle digest
- Canonical content digest
- Creation timestamp
- Creator identity metadata

Revision content is immutable. Integrity/activation status is tracked separately so the content row never changes.

Reset clears the active pointer but preserves history. Reset, acknowledgement, approval, rejection, and supersession are immutable state-history events rather than content revisions. They are visible in combined history but are not restorable. Restore revalidates an older content revision against the current definition and creates a new pending revision; it never rewinds the pointer. An incompatible historical revision returns an actionable compatibility report.

### Content-addressed snapshots

Every queued job logically pins the complete resolved template bundle, including editable parts, locked fragments, assembly order, and safe provenance. This is necessary because partial explicit overrides and locked server fragments can otherwise change independently after enqueue.

The physical snapshot is a protected component manifest:

- User-authored and explicit-request parts are stored as owner-scoped execution components.
- Visible and hidden server-managed parts reference immutable protected server-asset snapshots by asset ID and digest.
- The manifest records assembly order and one canonical full-bundle digest.

Server-managed components may be deduplicated globally by trusted asset digest. User-authored and explicit-request components may be deduplicated only within the same owner scope; APIs never reveal whether another owner has matching content. Hidden server-managed bytes are never copied into a per-user Prompts database.

Template snapshots do not contain rendered source documents or runtime variable values. A queued explicit `literal` override is an exception because the literal itself is the exact prompt part to reproduce; it is treated as sensitive request-bound job input, protected and retained under Jobs policy, and excluded from user prompt history and backups.

The protected execution store is not exposed through Service Prompts detail/history APIs or user exports. It uses the server's Jobs encryption-at-rest facility when enabled and retains components only as long as active/retained jobs reference them.

Encryption is optional confidentiality, not the integrity boundary. Every execution component, component manifest, pin set, and later job-binding record must carry a signature, MAC, authenticated-encryption tag, or equivalent cryptographic authenticator whose key/trust anchor is held outside the protected execution store. A colocated digest alone is insufficient.

The canonical authenticated envelope covers owner, submission ID, definition/component identifiers, component digests, full-bundle/set digest, source flags, creation/expiry metadata, and key ID. Binding produces a second authenticated record covering the original envelope digest and exact job UUID. Workers verify the trust anchor, both authenticators, expiry, owner, job binding, and all content digests before using any item.

Default mode may use an online execution-artifact signing/MAC key managed by context integrity or the existing Jobs key facility. Hardened mode requires that online key's identity/public verification material to be anchored by the external approved trust state; if no trusted execution-artifact signer is available, prompt-bearing queued work fails before job creation. Key rotation retains verification material for at least the maximum retained-job lifetime.

Automatic authentication of a request-bound explicit override proves enqueue provenance and immutability; it does not grant reusable prompt approval or replace the operator approval required for persistent Service Prompt revisions.

### Prompt pin sets and mutation receipts

A prompt pin set atomically binds one or more logical full-bundle manifests, their owner, a one-time submission ID, and an overall set digest to queued work. Each item records the definition ID, protected manifest reference/digest, and source flags needed by operator modes. The whole set is committed in one protected-execution-store transaction before job creation.

Mutation receipts map a client mutation ID to the completed result so safe network retries do not create duplicate revisions or misleading concurrency errors.

### Concurrency and integrity activation

Mutations use `expected_generation`. A stale generation returns a conflict and preserves the client's draft.

Activation follows the existing context-integrity approval policy:

1. Validate the draft against the current registry contract.
2. Append an immutable pending revision.
3. Present an escaped canonical diff for explicit non-model operator review.
4. On approval, re-resolve the current definition schema, part contract, locked assembly, and server-default bundle digest; then revalidate the pending revision.
5. If the registry schema, locked assembly, or server-default bundle digest changed since save, refuse activation with `pending_revision_stale`. The owner must preview and resubmit against the new baseline, creating a new pending revision.
6. Re-check the revision digest and manifest version, sign/register the new approved manifest version, and record the approval event.
7. Compare the pending revision ID and state generation again, then advance the active state and generation only after approval succeeds.

An owner save does not itself count as integrity approval. In single-user mode the same person is normally both owner and local operator, but activation still requires a distinct explicit confirmation. In multi-user mode, an authorized operator/admin approves the owner's pending revision through the context-integrity review flow; this governance action does not transfer ownership or create an administrator-wide override.

Per-user ownership is therefore an execution and mutation boundary, not secrecy from an authorized integrity reviewer. The review surface may disclose the pending revision's escaped canonical diff only to principals holding the existing context-integrity approval privilege. Ordinary administrators and other users do not gain read access through the Service Prompts API.

If approval fails, the digest/baseline changes during review, or no authorized operator approves, the prior effective revision remains active and the new revision remains pending, stale, or rejected. This design follows, and does not supersede, `2026-06-25-context-integrity-skills-prompts-design.md`.

### Cross-database enqueue protocol

Prompt storage and Jobs storage do not share a transaction. Enqueue therefore uses a one-time submission ID and compare-and-set binding:

1. Before enqueue, the job producer declares every service-prompt definition that the job can use.
2. Resolve and snapshot all declared definitions, authenticate the canonical component manifests/pin-set envelope, then commit one complete pin set with owner, submission ID, item digests, set digest, and authenticator.
3. Enqueue the owner-scoped job with the pin-set ID, submission ID, and set digest.
4. Compare-and-set bind the pin set to the returned job UUID and persist an authenticated binding record.

If a worker acquires the job before step 4 finishes, the worker may atomically perform the same compare-and-set binding and authenticated binding record only when owner, submission ID, pin-set ID, set digest, and original envelope authentication all match. Execution begins only after the pin set is cryptographically bound to that exact job UUID. A second job cannot reuse the set.

A reconciler repairs binding failures and garbage-collects unreferenced pin sets only after a grace period and a check of active and retained jobs. Workers verify the set digest and every item before beginning any stage. If one item fails, no prompt in the set is used.

Jobs with data-dependent prompt selection must pin all possible registered candidates plus the versioned selection policy before enqueue. A queued workflow that cannot declare a finite prompt requirement set is not eligible for migration until refactored.

Pinned jobs are retained according to Jobs retention. Referenced revisions and snapshots cannot be removed. Revision history is cursor-paginated and subject to account storage quotas.

## API Design

Add an authenticated `/api/v1/service-prompts` namespace. All ownership comes from the authenticated principal; endpoints never accept a target user ID.

### Capability contract

The server exposes:

```json
{
  "service_prompts": {
    "enabled": true,
    "mode": "enabled",
    "availability": "experimental",
    "contract_version": 1,
    "can_approve_pending": false
  }
}
```

Clients use this to distinguish supported, unavailable, read-only, `bypass_stored_overrides`, incompatible, experimental, and general-availability servers.

### Endpoints

- `GET /catalog`
  - Metadata and current-user state summaries
  - Search/filter inputs as needed
  - No prompt bodies
  - Private composite ETag based on registry/server-default catalog digest, capability/operator mode, and per-user catalog generation

- `GET /{definition_id}`
  - Server-default, active, and effective editable parts when visible and trusted
  - Structured provenance, validation contract, generation, and upstream-change state
  - Hidden or quarantined bodies omitted
  - `Cache-Control: private, no-store`

- `POST /{definition_id}/preview`
  - Validates draft parts
  - Renders deterministic safe samples
  - Returns named rendered parts, variable diagnostics, locked-section markers, and assembly order
  - Never invokes an LLM

- `PUT /{definition_id}/override`
  - Saves all editable parts atomically as the current pending revision
  - Requires expected generation and client mutation ID
  - Does not activate the revision

- `POST /{definition_id}/reset`
  - First resolves and verifies the current server default
  - Only then supersedes any pending revision and clears the active override after concurrency checks
  - If the required server default is unavailable or quarantined, preserves active/pending state and returns a safe conflict/unavailable error

- `POST /{definition_id}/acknowledge-default`
  - Records acknowledgement only for the currently resolved trusted server-default digest without rewriting prompt content
  - Rejects unavailable, quarantined, or stale digests

- `GET /{definition_id}/history`
  - Cursor-paginated combined content-revision and state-event history without bodies
  - Reset and acknowledgement events are marked non-restorable

- `GET /{definition_id}/revisions/{revision_id}`
  - Owner-scoped revision detail with private/no-store caching

- `POST /{definition_id}/revisions/{revision_id}/restore`
  - Validates the historical revision against the current contract and creates a new pending revision

Approval and rejection reuse the context-integrity review API and signed-manifest flow rather than introducing a second trust mechanism. The Service Prompts detail response links to the applicable pending asset and reports whether the current principal can approve it. In single-user mode the UI may offer a distinct **Review and approve** step, but it must remain a separate explicit action after save and must use the same integrity API.

Preview and mutation endpoints share runtime validation, body limits, rate limits, idempotency behavior, and stable machine-readable error codes.

### API failure semantics

- `422`: invalid draft with part-specific diagnostics
- `409`: stale generation, incompatible restore, pending-revision conflict/staleness, unsafe reset, or stale acknowledgement digest
- `413`: template or request body too large
- `403`: authenticated principal lacks access
- `404`: capability/definition is unavailable to this principal
- `429`: mutation or preview rate limit
- `503`: expected prompt store or integrity service is temporarily unavailable

Quarantined or incompatible expected content fails closed with a recovery code. Responses, logs, and telemetry do not echo prompt bodies.

## Runtime and Job Behavior

Synchronous services resolve once at their public service boundary. Existing explicit request prompt fields remain supported and take precedence. A migration must preserve whether an existing field is literal final text or a template.

Explicit request overrides retain their existing request-authorization semantics. They are request data, not reusable saved prompt versions, and do not require the separate operator approval used for persistent Service Prompt overrides. For queued work, the authenticated enqueue request authorizes creation of a one-job execution component; the protected store binds it to the owner/submission/job and verifies its digest at use. It cannot be discovered, restored, rebound, or reused as a saved prompt.

Queued endpoints declare their finite prompt requirements and commit one complete pin set before enqueue. Job payloads and job-status responses contain only safe references and digests, never raw prompt bodies. Workers bind the set to their exact job UUID before using any item and distinguish:

- Temporary prompt-store unavailability: retryable
- Missing or digest-mismatched pin set/item: permanent integrity failure
- Stored-override execution held by operator mode: held/retryable without substituting a different prompt

Editing, resetting, acknowledging, or upgrading defaults does not alter already pinned work.

## WebUI and Browser Extension

### Information architecture

Add `/settings/service-prompts` as a dedicated advanced page under AI & Models. Keep `/settings/prompt` as the link to the reusable Prompt workspace.

The page introduction explains the difference among:

- Service Prompts: automatic backend workflow defaults
- Prompt Library: reusable prompts selected by the user
- Conversation system prompts: chat-specific behavior

The implementation belongs in the shared UI package. The WebUI and extension options app render the same route. The narrow extension sidepanel uses the existing platform navigation abstraction to open the full options route.

### Catalog experience

The catalog provides:

- Search across names, descriptions, tags, and affected workflows
- Primary categories with secondary tags
- Filters for Customized, Server default, Upstream changed, and Needs attention
- Status summaries without loading prompt bodies
- Clear workflow-impact descriptions

Categories include summarization, RAG, media/audio, documents/web, extraction/chunking, and reports/digests, but eligibility is governed by the registry matrix rather than category alone.

### Editor experience

The editor shows:

- Definition description and affected workflows
- Named editable parts in lightweight accessible text areas
- Required/optional variable chips with cursor insertion
- Visible locked sections as read-only text
- Hidden locked sections as labeled server-managed markers
- Server-default, customized, and effective provenance
- Part-by-part plain-text comparison of visible content with unchanged regions collapsed; hidden components expose only changed/unchanged markers and safe digests
- Deterministic preview and assembly order
- Save pending revision, reset, acknowledgement, history, restore, and approval-status actions

The UI does not add a heavyweight code-editor dependency. Prompt text and diffs are escaped plain text and are never rendered as rich Markdown or HTML.

### Interaction states

- Inline validation plus an accessible error summary
- Dirty-state navigation and native close protection
- In-memory draft preservation during connection errors
- Retry and Copy Draft actions; no automatic device persistence
- Conflict handling that preserves the local draft and offers reload/copy, never automatic merge
- Upstream-change banner with Compare, Keep override, and Use server default actions
- Save messaging that the draft is pending explicit approval and does not yet affect runtime
- Approval/reset messaging that active changes apply to new work while queued jobs keep their pin sets
- Explicit disconnected, unsupported-server, read-only, bypass, pending-approval, quarantined, and incompatible states

“Keep override” stores the acknowledged default digest. “Use server default” clears the override rather than copying the default, so later server-default updates continue to flow through.

### Responsive and accessible behavior

Desktop may use a master-detail layout. Narrow screens use list-first navigation followed by a full-width editor with a clear Back action. The route must not create page-level horizontal scrolling at 390 px or extension-option widths.

Keyboard operation, focus restoration, labeled controls, screen-reader status updates, touch targets, and validation focus are required. Localization uses client translation keys with server fallbacks; placeholder identifiers remain literal and untranslated.

## Eligibility and Security

### Eligible prompts

Eligible definitions are user-visible content-generation or content-analysis instructions whose modification does not control authorization or an unenforced machine protocol. The inventory must record:

- Source and runtime consumer
- Data owner and workflow owner
- Variables and assembly behavior
- Output dependency
- Locked fragments
- Visibility
- Eligibility decision and reason

### Excluded prompts

The following remain locked:

- Authentication and authorization instructions
- Safety and moderation policy
- Tool and MCP permission/policy prompts
- Provider/model routing and tool-selection control prompts
- Evaluator, judge, grader, and security-review prompts
- Instructions whose output is parsed as a machine protocol without independent enforcement
- Hidden provider- or deployment-sensitive instructions

### Authorization boundary

Prompt text never grants authority. Tools, MCP calls, network access, filesystem access, data retrieval, and provider actions remain governed by existing RBAC, scope, policy, and argument validation on every call.

### Privacy

Prompt content is not secret storage. The UI warns against credentials. Bodies and rendered values are excluded from:

- Ordinary logs
- Audit message text
- Metrics labels
- Job payload/status summaries
- Notifications
- Validation/error responses

Audit metadata includes actor, definition ID, action, revision/digest, result, and timestamp.

Private account backups include only user-authored service-prompt content revisions and state-history provenance. They exclude execution snapshots, pin sets, hidden server-managed bytes, mutation receipts, integrity-review artifacts, and transferable active/pending trust state.

On import:

1. Preserve every imported content revision as owner-visible historical content marked `unapproved_import`.
2. Preserve exported state events only as non-operative provenance; do not replay activation, approval, reset, acknowledgement, or pending pointers.
3. Initialize the local definition state with `active_revision_id = null`, `pending_revision_id = null`, no acknowledged default digest, and a fresh local generation.
4. Do not automatically choose between the formerly active and formerly pending exported revisions.
5. Let the owner select one imported historical revision, preview it against the current local registry/server default, and use restore/resubmit to create the sole current local pending revision.
6. Require normal local explicit operator approval before activation.

This avoids trusting another deployment's state, preserves all user-authored choices, and keeps the one-current-pending invariant deterministic.

Shareable Chatbooks, ordinary Prompt exports, and MCP prompt catalogs exclude service prompts by default unless a later explicitly approved portability design changes that boundary.

### Context integrity

Server defaults use existing exact-byte file verification. Persistent user revisions use canonical asset identities scoped by owner, definition, and immutable version. At use time, the resolver verifies the exact bytes or row version consumed.

Request-bound execution components are not reusable DB prompt versions. They preserve the existing ability to send explicit request prompts, while their externally anchored authenticated envelope, owner/submission/job binding, and canonical digest prevent post-enqueue mutation or cross-job reuse. Server-managed hidden components remain protected context assets and must be trusted before a pin set can reference them.

Out-of-band mutations are quarantined. Every owner save remains pending until an explicit authorized operator approval updates the signed trust manifest. This requirement applies in normal and single-user modes; in single-user mode the owner/operator performs a separate non-model confirmation. Hardened deployments may additionally require an external trust decision. The previous effective revision remains active throughout review.

Review and settings surfaces never render untrusted content as rich HTML or feed it into model-assisted review.

## Compatibility

### Existing prompt loader and deployment overrides

The legacy `prompt_loader` remains for locked and not-yet-migrated prompts. A migrated workflow cannot mix registry-managed and direct-loaded versions of the same definition.

Existing module/key and `TLDW_PROMPT_FILE_*` mappings become deployment-default providers for eligible parts. The registry computes the atomic bundle after resolving every part. Hidden deployment defaults reveal only safe provenance and digest state.

Deployment-default failure behavior is intentionally stricter for migrated definitions than the legacy loader:

- An unset or blank environment override is not configured, so resolution proceeds to the packaged default.
- A configured nonblank override that is missing, unreadable, invalidly encoded, or integrity-blocked makes that required part and the atomic server-default bundle unavailable. The resolver does not fall through to packaged content.
- The operator recovers by fixing/removing the configured override or approving the expected asset; the UI/API reports a safe unavailable state without the path or body.
- Optional parts may be omitted only when the registry definition explicitly marks them optional.

Locked and not-yet-migrated consumers retain legacy fallback behavior until their domain migration. Domain compatibility tests must assert the stricter behavior at the migration boundary.

### Chat and reusable prompts

`preferences.chat.system_prompt`, composer prompt selection, and conversation-specific system prompts remain under Chat settings. Prompt Library and Prompt Studio records retain their existing APIs and user experiences. No new precedence is introduced between those systems and Service Prompts.

### Operator modes

Expose an operator-controlled mode:

- `enabled`: normal read/write and runtime behavior
- `read_only`: existing overrides execute, but mutations are blocked
- `bypass_stored_overrides`: explicit request fields still apply, but stored user overrides are skipped for new work and the remaining parts visibly use server defaults; existing pin sets containing stored-override content are held rather than silently substituted

Mode changes are visible through capabilities, UI state, logs, and safe audit metadata. They never rewrite or delete history.

## Rollout Plan

This umbrella design is implemented through separate Backlog tasks and reviewable changes.

### Slice 1: Inventory and eligibility matrix

- Inventory centralized and hardcoded prompt sources.
- Trace runtime consumers and prompt assembly.
- Assign stable definition IDs.
- Classify eligible, locked, or deferred with reasons.
- Identify exact explicit-override compatibility behavior.

### Slice 2: Registry and resolver foundation

- Typed definitions and startup validation
- Default-provider compatibility
- Constrained renderer and budgets
- Execution context and immutable resolver result
- No-override provider-message goldens

### Slice 3: Persistence, integrity, and API

- Per-user repository and migrations
- Revisions, generations, acknowledgement, and receipts in the per-user store
- Protected execution components, manifests, pin sets, retention, and encryption boundaries
- Two-phase integrity activation
- Cross-database enqueue reconciliation
- Authenticated API and capability contract

### Slice 4: Shared settings experience

- Shared route, catalog, editor, preview, compare, history, restore, conflicts, and capability states
- WebUI and extension-options integration
- Responsive, accessibility, and localization coverage

### Slice 5: Broad domain migrations

Migrate in independent domain units:

1. Summarization and media/audio analysis
2. Document and web analysis
3. User-visible RAG generation
4. Reports, digests, watchlists, and output creation
5. Eligible extraction/chunking flows with independently enforced contracts

Each domain change includes registry entries, consumer migrations, eligibility-matrix updates, default-message goldens, integration tests, and documentation together.

### Launch and completion gates

- Development deployments may expose the route with `availability: experimental` after the capability contract and at least one complete domain are available; the nav must label that state as experimental/beta.
- The first general/public release uses `availability: general` only after every approved broad content-facing domain is complete, matching the selected broad-first-release scope.
- A domain is complete only when every eligible call site uses the resolver and remaining prompt sources are explicitly locked or deferred.
- Broad-release completion requires all approved content-facing domains.
- Existing no-override provider messages are byte-equivalent unless separately approved.
- Existing deployment file/environment overrides remain effective.
- Rollback modes are visible and non-destructive.

## Verification Strategy

### Backend unit tests

- Registry uniqueness, asset presence, definition schema, deprecation, and eligibility contracts
- Placeholder parsing, escaped braces, invalid syntax, deterministic rendering, budgets, and locked assembly
- Resolution precedence, literal/template explicit overrides, server-default bundles, userless activity, and hidden defaults
- Revision immutability, generation conflicts, idempotency receipts, acknowledgement, reset, and compatible/incompatible restore
- Approval-time registry/default revalidation, stale-pending refusal, two-phase integrity activation, and prior-effective-state preservation
- Catalog-generation increments and composite ETag invalidation
- Trusted-default preconditions for reset and acknowledgement
- Protected snapshot/component creation, owner-scoped deduplication, authenticated envelopes/bindings, key rotation, reconciliation, retention, and garbage collection
- Operator modes and stable failure codes

### Property-based tests

- Arbitrary brace and placeholder inputs never execute code or crash the parser
- Rendering is deterministic for identical definitions and variables
- Unknown or malformed expressions are always rejected
- Size and repetition budgets hold across generated templates and values

### API and isolation tests

- Principal-derived ownership and cross-user denial
- Catalog body omission and private composite ETags
- Detail/revision no-store behavior
- Preview/save validator parity
- Stale generation, idempotent retry, limits, rate limits, and safe errors
- Hidden, quarantined, pending, read-only, `bypass_stored_overrides`, and unsupported states
- Experimental versus general capability availability

### Runtime and job integration tests

- Two users receive their own overrides for the same definition
- Explicit request prompts retain precedence
- Reset returns to the current server default
- Default changes do not rewrite user revisions
- Queued work stays pinned after edits, reset, acknowledgement, or upgrade
- Missing/tampered pin sets or items fail permanently; temporary stores retry
- Queued explicit request overrides remain request-authorized, job-bound, non-discoverable, and non-reusable
- Hidden server-managed bytes never enter per-user Prompt databases, APIs, or backups
- Offline protected-store tampering cannot succeed by changing content and a colocated digest together
- Provider message arrays are byte-equivalent with no override

### Frontend tests

- Shared routing and capability detection
- Search, categories, filters, and status summaries
- Atomic multi-part editing, variable insertion, preview, and locked sections
- Save, reset, acknowledgement, history, restore, conflict, and pending approval
- Offline draft recovery without persistent storage
- Escaped rendering and Copy Draft
- No horizontal overflow at 390 px and extension widths
- Keyboard, focus, screen-reader, and touch-target behavior

### End-to-end tests

- WebUI user edits a service prompt and the affected workflow uses it
- Packaged extension options edits the same account override and WebUI observes it
- Background job pins the selected revision and remains reproducible
- Upstream server-default change produces compare/keep/reset behavior
- Unsupported and disconnected extension states provide recovery actions

### Security and operational validation

- Cross-user access and path-like definition IDs
- XSS and rich-rendering regression tests
- Prompt-body log, metric, error, notification, and job-status redaction
- Context-integrity tamper and quarantine behavior
- Storage quota and mutation abuse
- Fresh and migrated per-user Prompts databases
- Private backup/restore and older-version ignore behavior
- Imported revisions become unapproved history with no active/current-pending pointer; owner resubmission creates the one local pending revision
- Execution snapshots/pin sets and cryptographic key material are not exported
- Bounded-cardinality metrics for definition ID, source kind, validation outcome, and latency
- Touched-scope Bandit, frontend lint/type checks, and focused test suites

## Implementation Planning Decisions

The implementation plan should decide, without changing this product contract:

- Exact Python module and repository class names
- Physical table consolidation and migration numbering
- Protected execution-store backend, mandatory authentication, optional encryption, component-manifest, tenant-scoped deduplication, and verification-key retention details
- Default storage quota and snapshot-retention durations
- Exact capability endpoint integration point
- Domain-by-domain stable ID catalog from the completed eligibility inventory
- Which existing diff component is reused by the settings editor
- The concrete Jobs held/retry representation for `bypass_stored_overrides`

## Acceptance Criteria

The design is successfully implemented when:

1. Authenticated users can discover and customize every approved service-prompt definition from the shared WebUI/extension settings route.
2. Overrides are validated, previewed, versioned, restorable, server-side, owner-isolated, and context-integrity checked.
3. Runtime resolution obeys the approved precedence and never silently falls back after an expected-source failure.
4. Queued work references immutable prompt content and remains reproducible.
5. Structured-output, security, routing, judge, and permission boundaries remain protected.
6. Existing deployment defaults, explicit request overrides, chat system prompts, Prompt Library, Prompt Studio, and MCP prompt behavior remain compatible.
7. Broad content-facing domains are migrated through documented eligibility decisions and byte-equivalent default-provider message tests.
8. The settings experience is responsive, accessible, explicit about capability/failure states, and does not persist drafts locally without user action.
