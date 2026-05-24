# Workspace Canonical Model Decision - May 2026

## Decision

`ResearchWorkspace` is the canonical shell for the roadmap first slice.
`ChatWorkspace` and `DocumentWorkspace` remain separate routes during this slice
and are treated as specialized entry points/modes, not deleted or fully merged.

## Current Entry Points

- `/research-workspace`: broad research workspace and best candidate for the
  canonical shell.
- `/chat-workspace`: chat-first and staged-context workspace.
- `/document-workspace`: document-focused reading and annotation workspace.

## Reasons

- ResearchWorkspace already contains sources, selected sources, chat, quick notes,
  generated artifacts, saved workspaces, source transfer, local persistence, and
  artifact payload offload.
- ChatWorkspace validates a chat-first route but should not own a separate product model.
- DocumentWorkspace validates deep document reading and annotation but should feed
  workspace sources/artifacts rather than define a parallel canonical workspace model.

## First-Slice Boundary

This slice does not consolidate routes. It defines the shared model and implements
one golden path inside ResearchWorkspace.

## Server/Local Boundary

The server already exposes `/api/v1/workspaces` for workspace metadata, sources,
artifacts, and notes. The browser-local Zustand store remains the responsive cache
and offline-friendly UI state. The first implementation must reconcile field names,
artifact status semantics, and persistence behavior between the two.

## Follow-Up Decisions

- Whether ChatWorkspace becomes a mode inside ResearchWorkspace.
- Whether DocumentWorkspace writes selected documents into workspace sources by default.
- Which collaboration semantics are required before enterprise pilots.
