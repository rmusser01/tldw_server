# Explainer Workspace Design

## Problem

Users need a way to turn either a learning goal or selected tldw sources into a durable, recursively expandable explanation. The reference page, Breakdowner, shows a useful interaction pattern: enter a goal, answer one clarifying question, receive concrete branches, and expand any branch again.

tldw should adopt the recursive breakdown pattern without copying the toy-like constraints. The feature must fit a source-grounded research product: selected sources, citations, background generation, resumable sessions, and clear labeling when outside knowledge is used.

## Goals

- Add a standalone `Explainer` workspace with explicit `Goal` and `Sources` tabs.
- Support both learning-goal breakdowns and selected-source explanations.
- Persist sessions, nodes, clarifying questions, selected answers, citations, source snapshots, and generation metadata server-side from day one.
- Use Jobs-backed generation for node expansion instead of blocking UI requests.
- Make grounding behavior configurable and enforceable.
- Let users choose the output intent per session and per expansion: `Explain`, `Plan`, or `Both`.
- Provide clear handoff actions into notes, quiz generation, flashcard generation, and follow-up prompts.

## Non-Goals

- Do not build a visual mind-map canvas in the first slice.
- Do not merge this into Research Workspace. Explainer should have its own in-page source picker/search.
- Do not rely on local-only persistence for explainer sessions.
- Do not allow `Source-only` mode to silently fill unsupported claims with outside knowledge.
- Do not build a new generic job system; use the existing Jobs backend for user-visible generation work.

## Reference Page Review

Breakdowner is valuable because it is sparse and recursive. Its useful behaviors are:

- A single focused composer.
- One clarifying question at a time.
- Quick-answer chips plus a custom answer field.
- A recursive tree where every item can be expanded again.
- Compact controls for collapse, expand, and delete.

The parts tldw should not copy directly:

- No durable provenance.
- No source selection.
- No citations or evidence state.
- No visible job/retry model.
- Tree-only reading, which becomes weak for longer explanations.
- Minimal accessibility semantics.

## UX Design

`Explainer` is a standalone workspace route, likely `/explainer`, in the WebUI Workspace nav group.

The top of the page has two explicit tabs:

- `Goal`: starts from a user-entered learning goal.
- `Sources`: starts from an in-page source picker/search.

Both tabs share a compact setup composer:

- Output intent: `Explain`, `Plan`, `Both`.
- Depth preset: `Quick`, `Standard`, `Deep`.
- Model/provider selector or inherited default, following existing tldw model settings patterns.

The `Sources` tab adds:

- Source search and picker.
- Selected source list with remove controls.
- Grounding mode: `Source-only`, `Source-led`, `Open explainer`.
- Retrieval settings summary with a link to detailed settings when needed.

The `Goal` tab does not expose `Source-only` unless the user attaches sources later. Goal sessions without selected sources use general model knowledge and should label that state clearly.

The main workspace has two reading levels:

- Tree rail: recursive node navigation, compact labels, generation state, evidence state, and expand/collapse controls.
- Detail panel: the selected node's readable explanation, plan steps, citations, outside-knowledge labels, retry controls, and handoff actions.

On desktop, the source/session/settings rail can sit on the right. On mobile, it becomes a drawer. The tree remains the primary navigation surface, but the detail panel is the primary reading surface.

## Grounding Modes

### Source-only

The model may only use selected source context. Every substantive claim must be supported by citations from selected sources. If retrieval cannot support the requested expansion, the backend returns an `insufficient evidence` node with suggested recovery actions:

- Select more sources.
- Switch to `Source-led`.
- Ask a narrower question.
- Search the library for additional material.

### Source-led

Selected sources are prioritized and cited when used. The model may add outside knowledge, but outside knowledge must be labeled in the returned node. The UI should separate source-backed claims from uncited general context whenever practical.

### Open explainer

The model may use general knowledge. Selected sources are still cited when used, but citation coverage is not required. The UI must avoid implying that uncited claims came from selected sources.

## Backend Data Model

Persist explainer data in a first-class backend store. The default implementation can use SQLite tables alongside the existing per-user database conventions, with user ownership enforced through the same AuthNZ patterns used by other user-visible resources.

### Session

```ts
ExplainerSession {
  id: string
  userId: string
  title: string
  mode: "goal" | "sources"
  status: "draft" | "active" | "archived" | "error"
  outputIntent: "explain" | "plan" | "both"
  grounding: "source_only" | "source_led" | "open"
  depthPreset: "quick" | "standard" | "deep"
  selectedSources: ExplainerSelectedSource[]
  rootNodeIds: string[]
  createdAt: string
  updatedAt: string
  archivedAt?: string
}
```

### Selected Source

```ts
ExplainerSelectedSource {
  sourceId: string
  sourceType: "media" | "note" | "document" | "web" | "unknown"
  title: string
  addedAt: string
  snapshotVersion?: string
  metadata?: Record<string, unknown>
}
```

### Node

```ts
ExplainerNode {
  id: string
  sessionId: string
  parentId?: string
  ordinal: number
  title: string
  body?: string
  kind: "question" | "answer" | "explanation" | "step" | "summary"
  intent: "explain" | "plan" | "both"
  status: "idle" | "queued" | "generating" | "error" | "complete"
  evidenceState: "supported" | "partially_supported" | "uncited" | "insufficient"
  citations: ExplainerCitation[]
  outsideKnowledgeUsed: boolean
  questionOptions?: ExplainerQuestionOption[]
  selectedOptionId?: string
  selectedCustomAnswer?: string
  childNodeIds: string[]
  generationMetadata?: ExplainerGenerationMetadata
  createdAt: string
  updatedAt: string
}
```

### Question Option

```ts
ExplainerQuestionOption {
  id: string
  label: string
  description?: string
  ordinal: number
}
```

### Citation

```ts
ExplainerCitation {
  id: string
  sourceId: string
  sourceType: string
  title: string
  excerpt: string
  locationLabel?: string
  startOffset?: number
  endOffset?: number
  url?: string
  snapshotHash?: string
}
```

Citation records should store enough excerpt and location data to explain old sessions even when source content changes later. They should not store excessive source content beyond what is needed for provenance and display.

### Generation Metadata

```ts
ExplainerGenerationMetadata {
  provider: string
  model: string
  promptTemplateVersion: string
  grounding: "source_only" | "source_led" | "open"
  retrievalSettings?: Record<string, unknown>
  jobId?: string
  generatedAt: string
  tokenUsage?: {
    promptTokens?: number
    completionTokens?: number
    totalTokens?: number
  }
}
```

## API Design

Use first-class Explainer endpoints under `/api/v1/explainer`.

```text
POST   /api/v1/explainer/sessions
GET    /api/v1/explainer/sessions
GET    /api/v1/explainer/sessions/{session_id}
PATCH  /api/v1/explainer/sessions/{session_id}
DELETE /api/v1/explainer/sessions/{session_id}

POST   /api/v1/explainer/sessions/{session_id}/nodes
PATCH  /api/v1/explainer/sessions/{session_id}/nodes/{node_id}
DELETE /api/v1/explainer/sessions/{session_id}/nodes/{node_id}

POST   /api/v1/explainer/sessions/{session_id}/nodes/{node_id}/expand
POST   /api/v1/explainer/sessions/{session_id}/nodes/{node_id}/answer-question
```

Deletes should behave as soft deletes or archival operations by default, consistent with the project's user-data recovery expectations. Permanent deletion can be a later explicit admin/user-data-management feature if needed.

`expand` creates a Jobs record and returns a job reference:

```ts
{
  jobId: string
  sessionId: string
  nodeId: string
  status: "queued"
}
```

The UI follows existing Jobs status endpoints for progress and refreshes the session or affected node after the job completes.

## Generation Flow

1. User creates a session from `Goal` or `Sources`.
2. Backend persists the session and initial root node.
3. Backend may create a persisted `question` node when a clarifying question is needed.
4. User selects a quick answer or enters a custom answer.
5. Backend persists the selected answer.
6. User expands a node.
7. Backend creates a Jobs-backed expansion task.
8. Worker loads session settings, node context, selected source context, and prompt template.
9. Worker performs retrieval when sources are selected.
10. Worker validates grounding semantics before writing child nodes.
11. Worker persists generated nodes, citations, evidence state, and generation metadata.
12. UI refreshes the tree and detail panel.

## Error Handling

- If a generation job fails, the node status becomes `error` and the UI offers retry.
- If `Source-only` retrieval is insufficient, the result is a complete `insufficient evidence` node, not an error.
- If selected sources are unavailable, show a state panel with recovery actions.
- If a session cannot be loaded, use the existing route error boundary and a route-specific recovery message.
- If Jobs polling is unavailable, keep the queued/generating node visible and provide a manual refresh action.

## Frontend Implementation Shape

Expected frontend files:

- `apps/tldw-frontend/pages/explainer.tsx`
- `apps/tldw-frontend/extension/routes/option-explainer.tsx`
- `apps/packages/ui/src/components/Option/Explainer/`
- Route registry entry in `apps/tldw-frontend/extension/routes/route-registry.tsx`
- i18n strings in the shared locale files
- Smoke inventory entry and page object for `/explainer`

Primary components:

- `ExplainerWorkspace`
- `ExplainerModeTabs`
- `ExplainerGoalComposer`
- `ExplainerSourcePicker`
- `ExplainerTree`
- `ExplainerNodeRow`
- `ExplainerDetailPanel`
- `ExplainerCitationList`
- `ExplainerSessionRail`
- `ExplainerJobStatusBanner`

## Backend Implementation Shape

Expected backend files:

- `tldw_Server_API/app/api/v1/endpoints/explainer.py`
- `tldw_Server_API/app/api/v1/schemas/explainer.py`
- `tldw_Server_API/app/core/Explainer/`
- Database management integration under the existing DB abstraction layer.
- Jobs worker handler for explainer node expansion.

The core module should keep generation orchestration separate from persistence:

- Repository: session/node CRUD and ownership checks.
- Service: create sessions, answer questions, enqueue expansion jobs.
- Worker handler: retrieval, prompt execution, grounding validation, citation extraction, child-node persistence.
- Prompt templates: versioned strings or structured prompt builders.

## Security And Privacy

- Enforce per-user access on sessions, nodes, selected sources, citations, and jobs.
- Do not expose source excerpts from another user through citations or job status.
- Do not log prompts, source excerpts, API keys, or generated content unless existing debug settings explicitly allow safe redacted logging.
- Treat selected source IDs as untrusted input and verify ownership before retrieval.
- Store citation excerpts only to the extent needed for provenance display.

## Accessibility

- Tree controls need semantic buttons with accessible names.
- Node status and evidence state should not rely on color alone.
- Keyboard users must be able to move through the tree, select a node, expand it, retry generation, and open citations.
- The mobile rail drawer must trap focus while open and restore focus on close.
- Motion should be minimal and respect reduced-motion preferences.

## Testing

### Backend

- Session CRUD enforces ownership.
- Node CRUD enforces session ownership.
- `expand` creates a Jobs record and marks the node `queued`.
- Worker writes child nodes and generation metadata on success.
- Worker writes error state on provider failure.
- `Source-only` insufficient retrieval writes an `insufficient` node without outside knowledge.
- Citation snapshots include source ID, title, excerpt, and location metadata.

### Frontend

- `/explainer` route renders in the WebUI shell.
- Goal tab can create a persisted session.
- Sources tab can search/select sources and create a persisted session.
- Tree renders persisted nodes and generation states.
- Detail panel shows citations and outside-knowledge labels.
- Grounding and output-intent controls persist in session settings.
- Jobs-backed expansion transitions through queued/generating/complete/error states.

### E2E

- Smoke test loads `/explainer`.
- Mocked API flow creates a Goal session and renders the first node.
- Mocked Sources flow selects a source and renders citation chips.
- Job polling completion updates an expanded node.

## Open Questions

- Should Explainer sessions live in the per-user notes/chats database or a new explainer-specific table group?
- Should session exports produce Chatbooks, notes, Markdown, or all three in the first release?
- Should imported Research Workspace sources be supported in the first release or a follow-up?
- Which existing prompt/template registry should own Explainer prompt versions?

## Backlog

Tracked by `TASK-546`.
