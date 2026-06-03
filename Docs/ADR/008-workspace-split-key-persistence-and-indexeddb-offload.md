# ADR-008: Workspace Split-Key Persistence And IndexedDB Offload

**Status:** Accepted
**Date:** 2026-06-03
**Backfilled from:** `Docs/Design/Workspace_Persistence_Architecture.md`
**Decision owner:** Human requester approval of TASK-509 inventory defaults
**Related task:** TASK-514
**Related spec/plan:** `Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md`

## Decision

Persist browser-local workspace state using split `localStorage` keys with optional IndexedDB offload for heavy chat sessions and artifact payloads.

## Context

The Research Workspace persistence path outgrew a single monolithic `localStorage` blob. Workspaces need responsive browser-local state, offline-friendly behavior, migration from legacy payloads, and bounded storage growth while server-backed workspace APIs continue to provide the long-term sync surface.

The split-key model stores an index under `tldw-workspace` and per-workspace snapshot/chat payloads under workspace-specific keys. Heavy chat and artifact payloads can be offloaded to IndexedDB and replaced with pointer metadata.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep one monolithic `localStorage` payload | Increases write churn, payload size risk, and recovery difficulty as workspace count and artifact size grow. |
| Move all browser workspace persistence to IndexedDB | Adds complexity for every read/write path and removes a simple compatibility path for smaller payloads. |
| Make server persistence the only source for workspace UI state | Loses responsive local cache behavior and offline-friendly workflows; server sync is not yet the only required client state surface. |

## Consequences

Workspace persistence has a split index plus per-workspace payload keys. IndexedDB offload is feature-gated and optional, and failures fall back to inline payloads when possible. Legacy monolith payloads remain readable and are migrated into the split model.

Persistence code must preserve payload bounds, cleanup stale per-workspace keys/offload records, and keep source-lineage and artifact metadata available even when heavy payload fields are offloaded.

## Follow-up

- Use this ADR as the covering record for `INV-018`.
- If server-backed workspace sync becomes the only accepted source of persisted workspace truth, create a superseding ADR.
