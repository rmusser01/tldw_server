# Workspace Frontend Server Context Contract Design

Date: 2026-06-18
Status: Approved For Implementation Planning
Owner: Codex brainstorming session
Backlog: TASK-2386

Related:

- GitHub issue #1993: Workspace Phase 2 frontend context contracts
- GitHub issue #1984: Workspace Phase 2 release tracker
- GitHub issue #1994: Workspace activity and resource index follow-up
- `Docs/superpowers/specs/2026-06-17-workspace-membership-adapters-design.md`

## Goal

Implement the #1993 frontend contract slice by making the server Workspace model the authoritative source for Workspace identity, membership, active-context state, eligibility decisions, and recovery copy.

The frontend may cache, hydrate, decorate, and normalize server responses for rendering. It must not create another independent Workspace semantic model.

## Product Decision

Unify on the server Workspace model.

`apps/packages/ui/src/services/tldw/domains/workspace-api.ts` already exposes the raw server Workspace DTOs, including workspaces, sources, context, capabilities, roots, allowed actions, and partial errors. The #1993 contract layer should sit above those DTOs and below feature UI. Its job is to normalize server-owned state into stable client display helpers and action guards.

The existing Research Workspace store remains useful for local UI state, browser persistence, drafts, layout state, import/export, source-pane state, and display decoration. It is not authoritative for Workspace membership, server resource ownership, project roots, ACP runtime association, or whether a Workspace-sensitive action is available.

## Scope

In scope:

- Shared frontend contract types for normalized server Workspace summaries, active Workspace context, membership labels, eligibility decisions, and recovery actions.
- Pure helper functions that consume existing server DTOs and produce display/action contracts.
- Thin hooks or services that fetch active Workspace context and capability data through the existing Tldw API client.
- Research Workspace pilot integration that displays server-authoritative active Workspace context and uses shared recovery copy for Workspace-sensitive operations.
- ACP Playground pilot integration that displays ACP session Workspace state, including active-session and active-Workspace mismatch copy.
- Tests proving active Workspace selection does not silently filter global browse/search or list rendering.
- Adoption guidance for later surfaces to reuse the same contracts.

Workspace tag/status display rules for later surfaces:

- Notes, Media, Sources, Artifacts, Chats, Prompts, Workflows, and Watchlists should render Workspace membership as a display badge derived from the shared membership contract, not from ad hoc local tag strings.
- Rows with no server `workspace_id` render a neutral `Global` badge when a membership badge is needed. The badge is informational and must not imply the row is unavailable in global browse/search.
- Rows with a known active server Workspace render `Workspace: {workspace label}`. The label comes from the server Workspace list/context response; the server Workspace ID remains the identity key.
- Rows whose server Workspace is archived render `Archived Workspace: {workspace label}` with a warning tone. Rows whose server Workspace is deleted render `Deleted Workspace: {workspace label}` with an error tone.
- Rows whose `workspace_id` cannot be resolved from the current server Workspace list render `Unknown Workspace` with a warning tone and preserve the unresolved ID for diagnostics.
- Surface-specific lifecycle state remains separate from membership. For example, Sources keep source readiness/status, Artifacts keep artifact status/review state, Workflows/Watchlists keep run state, and Chats/Prompts keep their own availability state alongside the Workspace badge.
- Active Workspace selection must never silently hide global rows or rows from other Workspaces. Any filtering must be user-selected and visibly represented as a filter, not inferred from active context.
- These display rules are defined in this contract slice so later migrations can adopt the same labels and tones incrementally. Full UI adoption across every listed surface remains staged work.

Out of scope:

- #1994 contained-resource index, activity feed, or Workspace dashboard UI.
- Broad migration of all Notes, Media, Sources, Prompts, Workflows, Watchlists, and Chats surfaces.
- Backend API changes unless implementation discovers a blocking contract gap.
- Changing ACP execution admission, sandbox trust, path authorization, MCP permissions, or agent runtime policy.
- Treating local Research Workspace state as a replacement for server membership.

## Architecture

Use three explicit layers.

| Layer | Responsibility |
| --- | --- |
| Server DTOs | Raw API response types and request methods in `workspace-api.ts`. These mirror the backend contract. |
| Frontend contract | Pure normalization, labels, recovery copy, and action decisions derived from server DTOs. |
| Feature UI | Research Workspace and ACP Playground consume the contract without reinterpreting Workspace semantics. |

The contract layer should live in a shared frontend location such as `apps/packages/ui/src/services/workspace-context/` or another existing shared-services convention if code review shows a better local pattern.

The contract layer should not duplicate `WorkspaceApiResponse` or `WorkspaceContextResponse` as a competing model. It should import the DTO types, keep server enum values visible where possible, and add only display-focused fields that are clearly derived.

## Contract Shapes

### Workspace Summary

Normalize `WorkspaceApiResponse` into a compact display summary:

- `id`
- `name`
- `profile`
- `archived`
- `deleted`
- `studyMaterialsPolicy`
- `label`
- `statusLabel`
- `version`
- `lastModified`

The summary keeps the server ID as the identity key. Generated labels are display-only fallbacks.

### Active Workspace Context

Represent active context as:

- `state`: `none`, `loading`, `ready`, `partial`, `missing`, or `error`
- `workspaceId`
- `workspace`
- `attentionState`
- `resolution`
- `projectRoot`
- `sourceSummary`
- `allowedActions`
- `partialErrors`
- `recovery`

`ready` and `partial` are derived from `WorkspaceContextResponse.resolution.status`. `missing` is used only when a local active Workspace ID cannot be resolved from the server.

### Membership Label

Membership labels should answer "where does this resource belong?" without implying filtering or authorization:

- `workspaceId`
- `workspaceLabel`
- `membershipLabel`
- `tone`
- `isAuthoritative`
- `reasonCode`

Membership labels are display metadata. They do not grant access.

### Eligibility Decision

Eligibility should wrap `WorkspaceAllowedAction` and capability data:

- `action`
- `allowed`
- `reasonCode`
- `severity`
- `primaryMessage`
- `nextStepLabel`
- `nextStepHref`

Live eligibility checks should be action-scoped. Lists should not issue eager eligibility checks for every row.

### ACP Session Context

ACP session context should compare the ACP session's server Workspace association against the currently active server Workspace context:

- `sessionWorkspaceId`
- `activeWorkspaceId`
- `state`: `aligned`, `mismatch`, `session_only`, `active_only`, `missing`, or `unknown`
- `message`
- `recovery`

Mismatch handling must be visible. It must not silently switch Research Workspace state or mutate the ACP session.

## Data Flow

1. Research Workspace reads its current local Workspace ID and asks the server contract hook to resolve the server Workspace context.
2. The hook fetches server context or capabilities through the existing Tldw API client only when a server Workspace ID exists.
3. Contract helpers normalize the server response into display and action contracts.
4. Research Workspace renders active context and recovery notices from the shared contract.
5. ACP Playground reads the active ACP session Workspace ID from its session store and compares it with the shared active Workspace context.
6. Global browse/search/list surfaces keep using their existing data sources and must not filter rows just because an active Workspace context exists.

## Error Handling

Use conservative, visible fallbacks:

- Missing server Workspace: show missing context and point to the Workspaces manager.
- Archived Workspace: show archived context and disable Workspace-sensitive mutations.
- Partial server resolution: show partial context and surface partial-error messages without blocking read-only rendering.
- Unsupported action: render stable disabled-action copy from the server reason code.
- ACP mismatch: show both IDs or labels when available and explain that the session is attached to a different server Workspace.
- API failure: show degraded context and avoid allowing Workspace-sensitive actions until the server state can be resolved.

Reason-code mapping should be centralized so later surfaces use the same copy.

## Pilot Details

### Research Workspace

Add a compact server Workspace context indicator near the existing header status controls. It should answer:

- Which server Workspace is active?
- Is server context ready, partial, missing, archived, or degraded?
- What next step is available when a Workspace-sensitive action cannot proceed?

This is not a new Workspace browser. Existing local browser behavior can remain, but server authority should be clear in the pilot copy and action guards.

### ACP Playground

Add a compact terminal/session Workspace notice in `ACPWorkspacePanel`:

- No session: continue linking to canonical Workspaces.
- Session without Workspace terminal: continue showing sandbox-required copy.
- Session Workspace aligned with active Workspace: show aligned server Workspace context.
- Session Workspace differs from active Workspace: show mismatch copy and link to canonical Workspaces.

Do not auto-switch the active Research Workspace from ACP session state.

## Testing

Add tests before implementation for:

- Pure normalization from `WorkspaceApiResponse`, `WorkspaceContextResponse`, `WorkspaceCapabilitiesResponse`, and `WorkspaceAllowedAction`.
- Recovery-copy mapping for common reason codes and unknown reason codes.
- ACP session/active Workspace comparison states.
- Research Workspace pilot rendering for ready, partial, missing, and failed server context.
- ACP Playground pilot rendering for aligned and mismatch session Workspace states.
- A global list/search rendering guard proving active Workspace context does not filter existing global browse/search data.

Focused verification should run the new frontend tests plus nearby existing Research Workspace and ACP Playground tests. Bandit is not applicable to this frontend/docs-only slice unless backend Python files are touched.

## Adoption Guidance

Future Workspace-aware surfaces should:

- Import shared contract helpers instead of reading raw DTOs directly for display/recovery copy.
- Treat membership badges as labels, not authorization.
- Use action-scoped eligibility checks for Workspace-sensitive mutations.
- Keep global browse/search/list surfaces global unless the user explicitly chooses a Workspace filter.
- Link back to #1994 for resource index and activity affordances instead of building them inline.

## Open Questions

- Whether the API should eventually expose a dedicated "active context" endpoint is out of scope for #1993. The first implementation should use existing context and capabilities endpoints.
- If any pilot action needs a reason code that the server does not currently return, this slice should add a frontend `unknown` fallback and document the API gap instead of inventing client-only semantics.
