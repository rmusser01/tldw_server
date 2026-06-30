# Sharing

Sharing owns the core services behind workspace, chatbook, and prototype workspace sharing. The package creates and validates private share tokens, resolves owner-side data access for shared workspaces, clones shared content into a recipient workspace, and records share audit events used by the sharing API.

## Start Here

- `share_token_service.py` is the token lifecycle entry point for create, validate, revoke, and private-link exchange behavior.
- `shared_workspace_resolver.py` resolves owner databases, shared workspace context, source content access, and clone permissions.
- `clone_service.py` copies shared workspace data into a recipient-owned workspace.
- Related API surface: `app/api/v1/endpoints/sharing.py`.
- Related schemas: `app/api/v1/schemas/sharing_schemas.py`.
- Related tests: `tests/Sharing/`.

## Responsibilities

- Generate random share tokens, store only hashed token values, and optionally protect links with bcrypt-checked passwords.
- Validate expiration, revocation, use limits, access level, and legacy prototype workspace resource identifiers before granting shared access.
- Resolve the owner-side ChaChaNotes, Media, and embedding namespaces needed when a recipient views shared workspace data.
- Clone workspace metadata, sources, notes, media rows, transcripts, and artifacts into a recipient workspace without copying vector embeddings directly.
- Record sharing audit events and migrate older share-audit records into the unified audit table.
- Hook workspace deletion so related sharing state is cleaned up when a workspace is removed.

## Module Map

- `share_token_service.py` - token creation, validation, revocation, password checking, and use-count updates.
- `shared_workspace_resolver.py` - owner/recipient access resolver for shared workspace reads, writes, and cloning.
- `clone_service.py` - deep-copy workflow for accepted shared workspaces.
- `share_audit_service.py` and `unified_share_audit.py` - audit recording and query helpers.
- `share_audit_unified_migration.py` - migration helper for legacy share audit data.
- `workspace_deletion_hook.py` - cleanup hook for workspace-scoped sharing records.

## How It Connects

- `app/api/v1/endpoints/sharing.py` wires these services to workspace shares, organization/team sharing, public share preview, prototype private-link exchange, and admin/audit routes.
- `app/core/AuthNZ/repos/shared_workspace_repo.py` supplies persistence for shared workspace access records.
- `app/core/Prototype_Workspaces/` uses the same token service for prototype workspace private-link sharing.
- The resolver crosses `ChaChaNotes_DB`, `Media_DB_v2`, and embedding namespaces so shared reads use owner data while mutations remain permission-checked.

## Architecture Notes

### Core Flow

- Authenticated workspace sharing starts in `sharing.py`, verifies workspace
  ownership, persists share rows through the AuthNZ shared-workspace repo, and
  uses `ShareTokenService` only when a private/public token is requested.
- Public token preview and import flows validate the token hash, optional
  bcrypt-protected password, expiry, revocation, use limits, resource type, and
  access level before resolving owner-side data.
- `SharedWorkspaceDBResolver` builds a `SharedWorkspaceContext` that reads from
  the owner user's ChaChaNotes/Media/embedding data. `CloneService` is the
  write boundary that copies allowed data into recipient-owned workspace state.

### State And Security

- Raw share tokens must never be persisted or logged; validation uses token
  hashes and creation is the only time the raw token is returned.
- Shared reads and clone writes intentionally use different identities. Reads
  inspect owner data through a validated share context, while clone/write
  decisions use the recipient principal and destination workspace ownership.
- Audit writes are split between legacy sharing audit helpers and unified audit
  migration helpers; new event fields should be safe for admin display.

### Extension Checklist

- New shareable resource: update `ResourceType`, `share_token_service.py`,
  endpoint schema/route handling, resolver behavior, and cross-user access
  tests.
- New clone subresource: extend `clone_service.py`, add rollback/orphan-state
  coverage, and ensure embeddings are regenerated or explicitly skipped.
- New public flow: add rate-limit, token validation, password, and audit tests
  in `tests/Sharing/` before exposing the route.

## Extension Points

- For a new shareable resource type, inspect `share_token_service.py`, `sharing_schemas.py`, and `sharing.py` first.
- For cloning additional workspace subresources, extend `clone_service.py` and add focused tests in `tests/Sharing/test_clone_service.py`.
- For audit fields or retention behavior, update `share_audit_service.py`, `unified_share_audit.py`, and the sharing endpoint tests together.
- For prototype workspace link changes, check the legacy resource-id normalization paths in `share_token_service.py`.

## Testing

- `tests/Sharing/test_share_token_service.py`
- `tests/Sharing/test_sharing_endpoints.py`
- `tests/Sharing/test_clone_service.py`
- `tests/Sharing/test_share_audit_service.py`
- `tests/Sharing/test_unified_share_audit.py`
- `tests/Sharing/test_share_audit_unified_migration.py`
- `tests/Sharing/test_cross_user_access.py`
- `tests/Sharing/test_workspace_deletion_hook.py`

## Gotchas

- Raw share tokens are only available at creation time; persistence and validation use token hashes.
- Prototype workspace resource identifiers have legacy aliases that are normalized in the token service.
- Shared workspace reads and writes intentionally use different data contexts: owner data for shared reads, recipient authorization for clone/write decisions.
- Cloning workspace content does not copy embeddings directly, so downstream indexing or re-embedding may still be needed.
