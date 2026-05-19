# Workspace Persistence Architecture

## Purpose

The Workspace Playground persistence path moved from a single monolithic `localStorage` blob to a split-key model with optional IndexedDB offload for heavy payloads. This document defines the storage model, gating/rollout controls, migration behavior, and diagnostics.

Primary implementation: `apps/packages/ui/src/store/workspace.ts`.

Current first-slice decision: `WorkspacePlayground` is the canonical shell for
the roadmap first slice, with `ChatWorkspace` and `DocumentWorkspace` kept as
specialized routes during this slice. See
`Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md`. Server sync should
use the existing `/api/v1/workspaces` family first, with this browser-local store
remaining the responsive cache and offline-friendly UI state.

## Storage Topology

`WORKSPACE_STORAGE_KEY` is `tldw-workspace`. In split mode this key is the index key, and per-workspace payloads are stored separately.

### Split-Key Layout

Index key:

- `tldw-workspace`

Per-workspace keys:

- `tldw-workspace:workspace:${encodeURIComponent(workspaceId)}:snapshot`
- `tldw-workspace:workspace:${encodeURIComponent(workspaceId)}:chat`

Index envelope schema:

```json
{
  "schema": "workspace_split_v1",
  "splitVersion": 1,
  "version": 1,
  "state": {
    "workspaceId": "<activeWorkspaceId>",
    "savedWorkspaces": [],
    "archivedWorkspaces": [],
    "workspaceIds": ["<workspace-a>", "<workspace-b>"],
    "workspaceSnapshots": {
      "<activeWorkspaceId>": { "...": "active snapshot fallback" }
    },
    "workspaceChatSessions": {
      "<activeWorkspaceId>": { "...": "active chat fallback or pointer" }
    }
  }
}
```

Important detail: the index only carries active-workspace snapshot/chat fallback data. Authoritative per-workspace snapshot/chat payloads are read from per-workspace keys using `workspaceIds`.

### Access Pattern

Read path (`getItem` for `tldw-workspace`):

1. If split mode is disabled, return monolithic `localStorage` value directly.
2. If split mode is enabled:
   - If current value is split envelope, reconstruct full persisted state by loading each `workspaceIds` snapshot/chat key.
   - If current value is legacy monolith, migrate in-memory and trigger best-effort background write to split keys.

Write path (`setItem` for `tldw-workspace`):

1. Parse envelope and normalize/migrate state.
2. Compute target workspace IDs.
3. Write only changed per-workspace snapshot/chat keys.
4. Clean up stale per-workspace keys and stale IndexedDB payload records.
5. Rewrite the index envelope.

## IndexedDB Offload

IndexedDB database:

- Name: `tldw-workspace-storage`
- Version: `1`
- Stores:
  - `workspace-chat-sessions`
  - `workspace-artifact-payloads`

### What Gets Offloaded

1. Chat sessions:
   - Offload when serialized chat session size is `>= 8 KB`.
   - LocalStorage chat key stores a pointer object instead of full messages.
2. Artifact payloads (`content` and/or `data`):
   - Offload when serialized payload size is `>= 12 KB`.
   - LocalStorage snapshot keeps artifact metadata and stores a payload pointer.

Pointer metadata stored in localStorage:

Chat pointer:

```json
{
  "offloadType": "workspace_chat_session_v1",
  "key": "workspace:<id>:chat",
  "historyId": "<nullable>",
  "serverChatId": "<nullable>",
  "updatedAt": 0
}
```

Artifact payload pointer (under `__tldwArtifactPayloadRef` on artifact objects):

```json
{
  "offloadType": "workspace_artifact_payload_v1",
  "key": "workspace:<id>:artifact:<artifactId>",
  "fields": ["content", "data"],
  "updatedAt": 0
}
```

### Offload Read/Write Flow

Write:

1. During split-key write, attempt offload via IndexedDB adapter.
2. On success:
   - Replace chat payload with pointer in `...:chat`.
   - Remove artifact `content`/`data` from snapshot and add `__tldwArtifactPayloadRef`.
3. On failure or unavailable IndexedDB:
   - Persist inline payloads in localStorage.

Read/rehydrate:

1. Reconstruct state from split keys.
2. Detect chat/artifact pointers.
3. If IndexedDB is available, hydrate full payloads from pointer keys.
4. If not available:
   - Chat pointer returns a minimal empty-message session retaining `historyId`/`serverChatId`.
   - Artifact pointer stays unresolved; pointer metadata is removed from hydrated artifact object.

Cleanup:

- Deleting workspace persistence removes per-workspace split keys and related IndexedDB chat/artifact records.
- Stale workspace cleanup runs during incremental writes.

## Feature Flag Rollout Controls

Both split-key persistence and IndexedDB offload are feature-gated with localStorage and env controls.

### Flags

Split-key storage enablement:

- localStorage: `tldw:feature-rollout:workspace_split_storage_v1:enabled`
- env (Vite): `VITE_WORKSPACE_SPLIT_STORAGE_V1_ENABLED`
- env (Next): `NEXT_PUBLIC_WORKSPACE_SPLIT_STORAGE_V1_ENABLED`

IndexedDB offload enablement:

- localStorage: `tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled`
- env (Vite): `VITE_WORKSPACE_INDEXEDDB_OFFLOAD_V1_ENABLED`
- env (Next): `NEXT_PUBLIC_WORKSPACE_INDEXEDDB_OFFLOAD_V1_ENABLED`

Accepted values: boolean, `1`/`0`, and string forms (`true/false`, `on/off`, `yes/no`, `enabled/disabled`).

Resolution order:

1. localStorage override
2. Vite env
3. Next env
4. default (`true`)

Gated behavior:

- Split-key logic only runs when split flag resolves to enabled.
- IndexedDB offload only runs when split-key is enabled and IndexedDB flag resolves to enabled.
- If split-key is disabled, behavior remains monolithic single-key persistence.

## Legacy Monolith Migration

Legacy monolith shape (`tldw-workspace` single payload) is still accepted and normalized.

Migration behavior:

1. Parse current payload as either split envelope or legacy envelope.
2. Normalize to canonical persisted shape:
   - Ensure `workspaceSnapshots` and `workspaceChatSessions` are map-shaped.
   - Convert legacy top-level fields (`workspaceName`, `sources`, `notes`, etc.) into snapshot for active workspace when needed.
   - Rehydrate/normalize dates and legacy array/object variants.
   - Normalize chat sessions to messages-canonical persisted format.
3. Ensure active workspace snapshot exists; generate fallback snapshot if needed.
4. On first read of a monolithic payload, return migrated state to app immediately and asynchronously attempt split-key write.

Compatibility notes:

- Read path supports split envelope, monolith envelope, and direct state shapes.
- Split index fallback fields (`state.workspaceSnapshots` and `state.workspaceChatSessions`) provide resilience when per-workspace keys are missing for active workspace.

## Data Transformation Rules During Persistence

1. Chat retention bound:
   - Persist only the most recent `250` messages per workspace chat session.
2. Server-backed artifact safety bounds:
   - For artifacts with `serverId`, truncate `content` above `24 KB` with a truncation suffix.
   - For artifacts with oversized `data` above `16 KB`, strip `data`.
3. IndexedDB offload may additionally replace artifact payload fields with pointer metadata when offload succeeds.

## Diagnostics (Payload Size and Write Count)

Development diagnostics are emitted in non-production builds.

Runtime snapshot:

- `window.__tldwWorkspacePersistenceMetrics`

Fields tracked:

- `key`
- `writeCount`
- `maxTotalBytes`
- `updatedAt`
- `totalBytes`
- `sections`:
  - `workspaceSnapshots`
  - `workspaceChatSessions`
  - `generatedArtifacts`
  - `notes`
  - `sources`
  - `selectedSourceIds`
  - `savedWorkspaces`
  - `archivedWorkspaces`
  - `other`

This is used to monitor payload growth, write churn, and section-level contribution during rollout.
