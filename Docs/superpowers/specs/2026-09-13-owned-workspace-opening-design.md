# Authorized Server-Owned Workspace Opening

**Status:** Read-only loader and scoped Zustand activation/draft persistence implemented and reviewed; route/account-lifecycle integration and live acceptance pending.
**Tracking:** TASK-12020.50.
**Parent:** [Shared workspace clone Jobs design](2026-08-25-shared-workspace-clone-jobs-design.md).

## Goal

Opening `/research-workspace?workspace=<workspace-id>` loads that authorized,
recipient-owned workspace, including a completed clone. Opening must not create
an empty replacement, upload stale browser state, discard drafts, or expose a
previous account's content. The existing shared-recipient view remains separate.

## Constraints

- Use existing workspace APIs; no new bootstrap endpoint in this slice.
- No route aliases, redirects, or additional trust banners.
- Opening and revalidation are read-only; only explicit user edits cause writes.
- Preserve legacy local workspaces without inferring their server/account owner.
- Use CDP and a real isolated backend/WebUI for acceptance.
- Preserve existing TASK-12020.49 and TASK-12020.50 changes.
- Do not claim full clone certification until the outstanding acceptance matrix passes.

## 1. Opening And Readiness

The route gate resolves the target before mounting the editable ResearchWorkspace.
It waits for persisted-store hydration, verifies the current principal, and loads
the target through existing APIs. An explicitly empty, malformed, duplicate, or
conflicting target selector fails closed. A route with no target selector retains
the existing local-workspace experience. `shared` mode cannot fall back to owned
or local mode on authorization failure.

`source_workspace_id` remains deep-research return context, not an alias for
`workspace`. Owned-workspace Open link producers must supply `workspace`; preserve
return context separately when needed. A conflicting `shared` and `workspace`
combination is an invalid route rather than an implicit priority decision.

Successful activation installs server identity, version, settings, and a clean
baseline together. Source-status and capability reads can start from that readiness.
Opening, refreshing, and React StrictMode remounts must not call `upsertWorkspace`,
`addWorkspaceSource`, or `updateWorkspaceSourceSelection`. Do not force the study
policy, select all sources, attach a root, start a sandbox, or launch MCP/agents.

The existing local-to-server reconciliation path remains available only for its
explicit legacy-local workflow. An owned server workspace never falls through to
it. Subsequent user edits use the established mutation APIs, including version-aware
metadata PATCH and explicit conflicts; disabling all synchronization is not a fix.

## 2. Complete Load Contract

All of these reads must succeed and validate before activation:

| Required resource | Existing API | Fields retained |
| --- | --- | --- |
| Workspace | `getWorkspace(id)` | Exact ID, name, archived/deleted, profile, timestamps, version, study policy, banner/audio settings, assistant defaults and effective default |
| Sources | `getWorkspaceSources(id)` | Source IDs, media references, order, explicit selection, review fields and versions |
| Artifacts | `getWorkspaceArtifacts(id)` | Content, provenance, lineage, review metadata and version data through the existing mapper |
| Saved notes | `getWorkspaceNotes(id)` | Workspace associations and note identities/content; not the current unsaved note editor |

Reject target ID mismatch and archived/deleted targets. Validate collection shape
and per-resource association where supplied. A successful empty list is valid; a
failed, absent, or malformed response is not an empty list. Do not use the existing
partial `hydrateWorkspaceFromServer` helper without replacing its incomplete
contract and migrating its tests/callers.

Each attempt has a 30-second deadline, uses cancellation where the transport
supports it, and always uses generation checks to discard late results. A required
failure leaves the old draft intact but keeps the target view gated. Offer Retry
and Back to Workspaces with distinct unavailable, denied, connection, timeout, and
invalid-response states. Back is an explicit navigation, not an automatic redirect.

Context, capability, and indexing reads may follow activation. Failures remain
explicitly unavailable/degraded rather than reporting empty, disconnected, or
ready. An expired authorization result invalidates the active view, not merely a
status indicator.

The existing source and artifact mappers should be reused. Browser-only source
folders have no identified canonical endpoint in this slice: recover them only
from a same-scope, same-workspace draft, pruning references absent from the loaded
sources. Do not synthesize server folders. Preserve the effective assistant default
instead of allowing the generic snapshot application to clear it.

These separate GETs are not a transactional server snapshot. Atomicity here means
one client activation after complete validation. Do not claim workspace metadata
version covers source, artifact, or note changes; retain available per-resource
versions for subsequent edits and present mutation conflicts explicitly.

## 3. Atomic Activation And Draft Preservation

Do not use `loadWorkspace(config)` as the server activation operation: it prefers
cached target content and does not capture the current outgoing draft. Add one
store action that verifies expected scope/generation, captures the latest outgoing
draft at commit time, and installs the complete target bundle plus readiness in one
state transition. Waiting for persistence hydration prevents a late hydration from
overwriting the authorized state.

Separate server-authoritative collections from browser editor/composer/layout
state. Preserve unsaved note text and dirty edits with their base versions where
available. Reopening the same target reauthorizes and refreshes server collections
but may recover only that scope's draft. A draft conflicting with current server
content remains recoverable and requires explicit reconcile/discard; never
auto-save it. A collection without a usable version requires explicit revalidation,
not a fabricated metadata-version guarantee.

A failed target load must not show the previous workspace under the target URL.
Keep its draft recoverable while rendering the target error. If draft persistence
fails, retain it in memory and report that it will not survive reload; do not
silently promise durable recovery or proceed with destructive draft replacement.

## 4. Scope Throughout The Lifecycle

Scope consists of normalized server origin plus deployment subpath, verified
principal ID, and relevant organization context. Never place access/refresh tokens
in storage keys. Request generation is separate from durable scope: token refresh
alone does not change draft ownership.

On logout or account/server/organization changes, immediately gate rendering and
suspend new reads/writes, preserve the outgoing scoped draft, clear active content
and readiness, and verify the new scope before loading again. Check scope before
request dispatch and before committing responses. Use existing auth/config events
and request-context conventions rather than adding a second authentication manager.
Unmount cancels the load; StrictMode replay remains idempotent and write-free.

Already-dispatched mutations cannot be undone by hiding the UI. Keep their original
request identity pinned, reject stale completion handlers, and never retry them
using the new account's credentials. Audit the owned view's mutation entry points,
not just its bootstrap hook.

Server-loaded snapshots and drafts must not enter the global UUID-only legacy
snapshot/saved-workspace lists. Track active origin and store drafts in a distinct
scoped namespace. Adapt save, persist, and switch boundaries so the same UUID on
two servers/accounts cannot collide. Leave legacy local records untouched; opening
must not upload them or infer ownership. Ambiguous pre-fix records require the
existing explicit reconciliation workflow, not a destructive automatic migration.

## 5. Canonical Notes

Use workspace-ID associations for the owned notebook's saved list and note
selection. Apply notebook search to that canonical set, never a global tag query
that can include unrelated same-tag notes. Preserve the legacy tag path only for
legacy local workspaces. A failed note fetch is not an empty notebook. Saved notes
and the unsaved editor draft remain separate state.

## Verification And Review Disposition

| Review finding | Design repair | Required evidence |
| --- | --- | --- |
| Opening performs writes or loses readiness | Sections 1-2 | Exact target, zero opening mutations including StrictMode, working status reads, explicit edits still save |
| Activation loses drafts | Section 3 | Delayed hydration, edits during load, same-target recovery, failed load, local/server switches, persistence failure |
| Notes use the wrong authority | Section 5 | Copied notes visible; unrelated same-tag note excluded; failed GET not empty |
| Scope guard ends after loading | Section 4 | Account/server/org changes during and after activation, same UUID across scopes, stale requests/mutations fenced |
| Incomplete hydration masquerades as success | Section 2 | Complete settings/content, each required failure, malformed/mismatched/deleted/archived responses, no partial activation |

Live acceptance must open and reload the exact cloned UUID through CDP, inspect
content and saved notes, wait for search readiness, then ask a real configured
model and follow its source evidence. A route screenshot or mock response does not
prove grounded chat. Keep PostgreSQL/environment limitations and the remaining
clone recovery/fault scenarios explicit in TASK-12020.50.

All five review findings are incorporated into this design. The complete read-only
loader now has focused regression coverage; the route does not yet consume it.
Atomic activation, scoped persistence/lifecycle, canonical notebook integration,
and live verification remain pending in the linked implementation plan.
