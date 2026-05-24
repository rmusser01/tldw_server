# Research Workspace Legacy Storage Inventory

Date: 2026-05-23
Backlog: TASK-468

## Purpose

This document is the deletion-safety inventory for moving legacy local Research
Workspace data into server-backed workspaces. It is intentionally
non-destructive: no local content deletion, migration endpoint, or automatic
true-move workflow is implemented by this slice.

Any future migration that deletes browser-local workspace payloads must first
cover every content-bearing surface below in a server migration manifest,
receive a server migration receipt, pass read-back verification, and receive
`client_delete_eligible` from the server.

Unknown workspace-prefixed storage blocks deletion by default.

## Inventory

| Surface | Classification | Content classes | Server destination | Deletion rule |
| --- | --- | --- | --- | --- |
| `localStorage:tldw-workspace` | content | workspace identity, workspace list, sources, folders, notes, artifacts, chat messages, metadata | Workspace core, sources, notes, artifacts, chats, migration receipt | Requires server receipt |
| `localStorage:tldw-workspace:workspace:*:snapshot` | content | workspace identity, sources, folders, selected sources, notes, artifacts, metadata | Workspace core, sources, folders/tags, notes, artifacts, migration receipt | Requires server receipt |
| `localStorage:tldw-workspace:workspace:*:chat` | content | chat messages, chat session metadata | Workspace chat history, migration receipt | Requires server receipt |
| `indexedDB:tldw-workspace-storage/workspace-chat-sessions` | content | chat messages, chat session metadata | Workspace chat history, migration receipt | Requires server receipt |
| `indexedDB:tldw-workspace-storage/workspace-artifact-payloads` | content | artifacts, artifact payloads | Workspace artifacts/outputs, migration receipt | Requires server receipt |
| `localStorage:tldw:research-workspace:pinned-workspaces:v1` | ui_only | none | none | Retain local |
| `localStorage:tldw:research-workspace:recent-output-types:v1` | ui_only | none | none | Retain local |
| `localStorage:tldw:research-workspace:add-source-tab-usage:v1` | ui_only | none | none | Retain local |
| `localStorage:tldw:research-workspace:onboarding-dismissed:v1` | ui_only | none | none | Retain local |
| `localStorage:tldw:research-workspace:telemetry` | metadata | none | none | Retain local |
| `localStorage:tldw:workspace:playground:telemetry` | metadata | none | none | Delete only after one-time import |
| `localStorage:workspace_migrated` | obsolete | none | none | Retain or ignore; never authoritative |
| `localStorage:tldw:feature-rollout:workspace_split_storage_v1:enabled` | ui_only | none | none | Retain local |
| `localStorage:tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled` | ui_only | none | none | Retain local |
| `localStorage:tldw:workspace:broadcast-sync` | ui_only | none | none | Retain local |
| `broadcastChannel:tldw-workspace-sync` | derived | none | none | Runtime-only, no deletion action |

## Deletion Eligibility Rules

Deletion is allowed only when all discovered content surfaces are listed in the
server migration manifest and covered by a verified server receipt.

Deletion is blocked when any of these are true:

- a content-bearing surface is discovered but not covered by the manifest;
- an unsupported surface is discovered;
- an unknown key under `tldw-workspace:*`,
  `tldw:research-workspace:*`, or `tldw:workspace:playground:*` is discovered;
- an unknown object store is discovered inside IndexedDB database
  `tldw-workspace-storage`.

UI-only and local diagnostic surfaces do not block content deletion, but they
are retained unless a future mapping explicitly says otherwise.

The legacy `workspace_migrated` flag must not be used to skip true migration.
It belongs to an older non-receipted helper and is not proof of server receipt,
read-back verification, or local deletion eligibility.

## Code Contract

The implementation lives in:

`apps/packages/ui/src/store/research-workspace-legacy-storage-inventory.ts`

It exports:

- `RESEARCH_WORKSPACE_LEGACY_STORAGE_INVENTORY`
- `classifyResearchWorkspaceLegacyStorageSurface`
- `evaluateResearchWorkspaceLegacyDeletionEligibility`

The evaluator is deliberately a gate, not a migrator. It does not read browser
storage, write browser storage, upload data, or delete data. Callers pass
already-discovered keys/stores and a set of manifest-covered surface IDs.

## Future Migration Requirements

Before implementing deletion:

1. Discover current localStorage keys and IndexedDB stores.
2. Classify them through the inventory module.
3. Build a server migration manifest covering every content-bearing surface.
4. Upload content through the migration protocol.
5. Verify server read-back against the receipt.
6. Wait for server `client_delete_eligible`.
7. Delete only manifest-covered content surfaces.
8. Keep a non-content tombstone with legacy workspace id, server workspace id,
   migration id, and deletion timestamp.

The old `/workspace-playground` route is not a migration entrypoint and must not
be restored as a redirect, alias, or hidden fallback.

