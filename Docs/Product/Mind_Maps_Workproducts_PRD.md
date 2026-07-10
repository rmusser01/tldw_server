# Mind Maps Work Products PRD

Status: Proposed
Owner: Product / WebUI / API
Created: 2026-07-10
Tracking: TASK-12102

## Summary

tldw_server already has graph-related foundations: a notes graph API, frontend
graph libraries, Mermaid rendering, timeline graph components, and source-aware
RAG. This PRD defines a separate first-class Mind Maps work-product feature for
creating dynamic, presentable, downloadable concept maps from media, notes,
chats, and RAG results.

The key product distinction is that a notes graph shows relationships between
stored objects, while a mind map is a user-facing knowledge artifact. Mind maps
may be generated from notes graph data, but they also need semantic grouping,
editable labels, source citations, layout state, and export formats.

## Problem

Users can search and summarize their knowledge base, but they lack a visual
work product that helps organize concepts, relationships, and evidence into a
map they can inspect, edit, present, and download. Existing graph APIs expose
useful structure, but they are not a complete mind-map product.

## Goals

- Generate mind maps from media, notes, chats, RAG queries, or selected text.
- Persist maps as editable work products with nodes, edges, groups, layout, and
  source citations.
- Render maps interactively in WebUI and, where practical, extension routes.
- Export maps as JSON, Markdown outline, Mermaid, SVG/PNG, and PDF when
  available.
- Keep generated maps bounded, readable, and source-traceable.
- Reuse existing graph infrastructure where it fits without coupling mind maps
  to notes graph internals.

## Non-Goals

- Replacing the existing notes graph API.
- Infinite canvas whiteboarding.
- Real-time collaborative map editing.
- Arbitrary graph queries over all user data without limits.
- Knowledge graph ontology management.
- Fully automated correctness guarantees for generated semantic relationships.

## Existing Foundation

- Notes graph PRD: `Docs/Product/Graphing-Notes-PRD.md`
- Notes graph endpoint: `tldw_Server_API/app/api/v1/endpoints/notes_graph.py`
- Notes graph schemas: `tldw_Server_API/app/api/v1/schemas/notes_graph.py`
- Notes graph core: `tldw_Server_API/app/core/Notes_Graph/`
- Frontend graph dependencies: `@xyflow/react`, Cytoscape, dagre, Mermaid
- Existing Mermaid chat rendering harness and common rendering components
- RAG and media source metadata available through existing services

## User Stories

- As a student, I can generate a mind map from a lecture and download it for
  revision.
- As a researcher, I can map themes across selected notes and inspect the
  source evidence behind each relationship.
- As a journalist, I can generate a map of people, organizations, claims, and
  source links from collected material.
- As a browser extension user, I can create a quick map from the current page or
  selection and continue editing in the WebUI.
- As a self-hosted user, I can export a readable map without relying on a cloud
  mind-mapping service.

## Product Requirements

### PR-1: Mind Map Artifact Model

Mind maps must be persisted independently from the notes graph. A map should
store:

- Title and description
- Source bundle
- Nodes
- Edges
- Groups or clusters
- Layout state
- Theme/style metadata
- Citation metadata
- Version and timestamps

The artifact model may start in an existing user artifact/output store if it
supports versioned structured JSON. If not, add a small per-user MindMaps DB.

### PR-2: Generation Sources

MVP generation should support:

- Media item transcript/content
- Selected notes
- RAG query result
- Pasted text or browser extension selection

Later phases can add chat session summaries, collections, and multi-source
bundles.

### PR-3: Map Semantics

Generated maps should normalize relationships into a bounded schema:

- Concept
- Person / organization / place when detected
- Claim
- Evidence
- Term / definition
- Process step
- Cause / effect
- Parent / child
- Related
- Contrasts with
- Supports / refutes

MVP can use a smaller edge set: parent, related, supports, contrasts,
sequence.

### PR-4: Readability Controls

Every generated map must include controls for:

- Maximum nodes
- Maximum depth
- Grouping strategy
- Include citations
- Include low-confidence relationships
- Collapse/expand groups
- Show/hide edge labels
- Layout orientation

Defaults should prefer readable maps over exhaustive maps.

### PR-5: Editing

Users must be able to:

- Rename nodes
- Edit node summaries
- Add/remove nodes
- Add/remove edges
- Reposition nodes
- Collapse/expand groups
- Inspect source citations
- Save layout changes

Advanced styling is later-phase work.

### PR-6: Export

MVP export targets:

- JSON, preserving full structured data
- Markdown outline
- Mermaid mind map or flowchart where representable

Later export targets:

- SVG
- PNG
- PDF

Image/PDF export can be frontend-rendered first if backend rendering would
introduce significant dependency or sandbox cost.

### PR-7: Relationship To Notes Graph

Mind maps may use notes graph as an input source, but they should not expose
the notes graph response shape as the user-facing artifact contract.

Required bridge behavior:

- Import notes graph nodes and edges into a mind-map draft.
- Preserve source object references.
- Allow semantic regrouping and label rewriting.
- Keep graph truncation warnings visible.

## UX Requirements

- Map generation should start from a concrete source selection or query.
- The map viewer must support zoom, pan, fit-to-view, search, and selected-node
  details.
- The selected-node panel must show citations and source links.
- Empty states should explain that maps need selected sources or a query.
- Large-map warnings should be visible before generation and after truncation.
- Extension should provide a lightweight "Create mind map from selection/page"
  handoff, not a full canvas in MVP.

## API And Data Model Direction

Potential endpoints:

- `POST /api/v1/mind-maps`
- `GET /api/v1/mind-maps`
- `GET /api/v1/mind-maps/{id}`
- `PATCH /api/v1/mind-maps/{id}`
- `DELETE /api/v1/mind-maps/{id}`
- `POST /api/v1/mind-maps/generate`
- `GET /api/v1/mind-maps/{id}/export?format=json|markdown|mermaid`

Potential schema:

- `MindMap`
- `MindMapNode`
- `MindMapEdge`
- `MindMapGroup`
- `MindMapSourceCitation`
- `MindMapLayout`
- `MindMapGenerationRequest`

The API should use optimistic concurrency for edits if persisted maps are
versioned.

## Backend Requirements

- Resolve source bundles through existing media, notes, and RAG services.
- Bound source text and generated map size.
- Validate generated JSON with strict schemas.
- Persist generated maps with user isolation.
- Preserve citations and source references.
- Record metrics for generation latency, error type, node count, edge count,
  truncation reason, and export format.

## WebUI And Extension Requirements

- Use existing graph rendering libraries instead of hand-rolling layout logic.
- Store layout state separately from semantic graph content.
- Keep the renderer usable on small screens, with a list/detail fallback if the
  canvas is too constrained.
- Extension MVP should create and open maps, not replicate the full editor.
- Route availability should be gated by backend capabilities.

## Security And Privacy

- Do not log source text, generated map content, or citation snippets.
- Bound generated node/edge labels and citation text.
- Sanitize Markdown/Mermaid exports.
- Prevent arbitrary URL schemes in citation links.
- Enforce per-user access and existing auth modes.

## Success Metrics

- Users can generate a map from one media item and save it.
- Users can generate a map from selected notes or a RAG query and save it.
- Users can edit node labels and persist layout.
- Users can export JSON, Markdown, and Mermaid.
- Large inputs are bounded with clear truncation warnings.
- Extension can start a map from a page or selection.

## Rollout Plan

### Phase 1: Artifact Contract And Persistence

Define schemas, source bundle format, citation format, versioning, and storage.
Add unit tests for validation and serialization.

### Phase 2: Generation MVP

Generate bounded maps from media, notes, pasted text, and RAG. Add strict
validation, source citations, and truncation warnings.

### Phase 3: WebUI Renderer And Editor

Build a canvas with zoom, pan, search, node details, citation inspection, basic
editing, and layout save.

### Phase 4: Exports

Add JSON, Markdown, and Mermaid exports. Add SVG/PNG/PDF investigation after
renderer behavior is stable.

### Phase 5: Extension Handoff And Docs

Add extension capture/start flow and update user/developer docs.

## Backlog Task Set

- Parent: Implement Mind Maps work products.
- Phase 1: Define mind-map artifact schemas and persistence.
- Phase 2: Add source-backed mind-map generation.
- Phase 3: Build WebUI mind-map viewer/editor.
- Phase 4: Add mind-map exports.
- Phase 5: Add extension handoff and documentation.

## Open Questions

- Should mind maps be stored in a new MindMaps DB or a generalized artifact
  store?
- Which export formats are required for MVP beyond JSON, Markdown, and Mermaid?
- Should generated maps support non-note sources as first-class nodes, or only
  citations attached to concept nodes?
- Should concept extraction use only the configured LLM, or also deterministic
  NLP heuristics for offline use?
