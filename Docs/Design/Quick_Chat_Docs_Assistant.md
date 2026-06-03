# Quick Chat Docs Assistant Design

**Covered by:** `Docs/ADR/009-quick-chat-docs-assistant-modes.md`

## Feature Goals And Scope

### Goals

- Provide lightweight help in the same modal through three intentional modes:
  - `Chat` for normal model conversation.
  - `Docs Q&A` for retrieval-grounded answers with citations.
  - `Browse Guides` for deterministic tutorials and curated workflow cards.
- Keep discovery fast by making route-aware recommendations from current page context.
- Keep docs retrieval scoped to project documentation by default.

### Scope

- Frontend Quick Chat orchestration, retrieval profile shaping, response formatting, and tutorial/workflow surfacing.
- Configuration through existing Quick Chat storage settings.

### Out Of Scope

- Replacing the main chat UI.
- Building a new backend endpoint (uses existing `POST /api/v1/rag/search`).
- Dynamic workflow-card generation (workflow cards remain curated/static plus user overrides).

## Architecture Decisions

### Component Mapping

- `apps/packages/ui/src/components/Common/QuickChatHelper/QuickChatHelperModal.tsx`
  - Owns mode switching UI (`Chat`, `Docs Q&A`, `Browse Guides`) and route handoff.
  - Routes interactions to `useQuickChat` or `QuickChatGuidesPanel`.
- `apps/packages/ui/src/hooks/useQuickChat.tsx`
  - Central send pipeline.
  - Dispatches `docs_rag` requests via `tldwClient.ragSearch(...)`.
  - Reads scope controls from storage (`quickChatStrictDocsOnly`, `quickChatDocsIndexNamespace`, `quickChatDocsProjectMediaIds`).
- `apps/packages/ui/src/components/Common/QuickChatHelper/workflow-guides.ts`
  - Canonical curated workflow guide data and normalization/validation utilities.
  - Route normalization and recommendation scoring for suggested pages.
- `apps/packages/ui/src/tutorials/registry.ts`
  - Source of truth for tutorial registration and route matching.
  - Enables route-scoped tutorial lookup used by Quick Chat browse mode.

### Mode-Specific Decisions

1. `Chat`
   - Uses standard streaming chat path and selected model.
   - Purpose: lightweight side conversation, no retrieval contract.
2. `Docs Q&A`
   - Uses retrieval profile builder and formatted docs reply with references.
   - Purpose: answer “how/where/what” docs questions with traceable sources.
3. `Browse Guides`
   - Deterministic local layer:
     - page tutorials from tutorial registry
     - workflow cards from curated definitions (or validated user overrides)
   - Purpose: low-latency guidance even when retrieval quality varies.

## RAG Integration And Configuration Strategy

### Request Profile (`docs-rag-profile.ts`)

`buildQuickChatDocsRagProfile(...)` sets a stable base profile:

- `search_mode: "hybrid"`
- `enable_generation: true`
- `enable_citations: true`
- `enable_reranking: true`
- `reranking_strategy: "flashrank"`

Strict docs defaults:

- `sources: ["media_db"]`
- `corpus: "media_db"`
- `index_namespace`: configured namespace or fallback `project_docs`
- `include_media_ids`: added when `quickChatDocsProjectMediaIds` is configured

Profile tuning:

- Synopsis-like queries increase recall/context depth (`top_k`, parent-document settings, generation budget).
- Troubleshooting queries lower `min_score` for higher recall.

Route-aware augmentation:

- If user query references “this page/current page/here”, append:
  - `Current page context: <routeLabel> (<route>)`

### Response Construction (`rag-response.ts`)

`buildQuickChatRagReply(...)`:

- Normalizes generated answer fields and citations across response shapes.
- Falls back to snippet-style summaries when answer text is absent but docs exist.
- Adds `### References` section when citations are available.
- Adds `### Suggested Pages` using recommendation scoring from `workflow-guides.ts` with inputs:
  - query text
  - generated answer text
  - citation titles/sources
  - current route

## Tutorial System Integration

Per-page tutorials are defined in `apps/packages/ui/src/tutorials/definitions/*.ts` and merged into `TUTORIAL_REGISTRY` in `apps/packages/ui/src/tutorials/registry.ts`.

Quick Chat integration path:

1. `QuickChatHelperModal.tsx` passes current route to browse mode.
2. `QuickChatGuidesPanel.tsx` resolves tutorials via `getTutorialsForRoute(...)`.
3. Tutorial cards render with lock/completion state from tutorial store.
4. `Start`/`Replay` launches tutorial by ID.

Route matching uses normalization logic so canonical routes, hash routes, and legacy options URLs resolve consistently.

Representative route-to-basics mapping:

- `/chat` -> `playground-basics`
- `/workspace-playground` -> `workspace-playground-basics`
- `/media` -> `media-basics`
- `/knowledge` -> `knowledge-basics`
- `/characters` -> `characters-basics`

## Testing And Validation Guidance

Primary tests:

- `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/docs-rag-profile.test.ts`
  - covers strict profile defaults, media ID scoping, route-aware augmentation, and tuning branches.
- `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/workflow-guides.test.ts`
  - checks route normalization, workflow filtering, recommendation behavior, and JSON validation.
- `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/rag-response.test.ts`
  - verifies reference formatting, no-context fallback, and suggested-pages composition.

Validation expectations:

- Mode switching must preserve intended send path (`chat` vs `docs_rag` vs browse-only).
- Docs scope controls must deterministically alter request options.
- Browse mode must surface route-relevant tutorials from registry and curated workflow cards without backend dependency.

## Relationship To User Guide

The user-facing operational walkthrough remains in `Docs/User_Guides/WebUI_Extension/Quick_Chat_Docs_Assistant_Guide.md`. This design doc is the implementation rationale and verification reference for reviewers.
