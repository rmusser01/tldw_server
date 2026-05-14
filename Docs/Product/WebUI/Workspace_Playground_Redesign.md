# PRD: Workspace Playground Redesign

Product Requirements Document

Feature: NotebookLM-Style Three-Pane Research Interface
Location: apps/packages/ui/src/components/Option/WorkspacePlayground/
Status: Planning

Current first-slice decision: `WorkspacePlayground` is the canonical shell for
the roadmap first slice, while `ChatWorkspace` and `DocumentWorkspace` remain
specialized routes. See
`Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md`. Server sync should
use the existing `/api/v1/workspaces` family first.

First implementation slice: the Studio pane now includes a work-product
template entry point with Executive Brief as the first actionable path. Research
Dossier, Competitive Market Memo, and Technical Project Spec are visible as
planned templates, but remain metadata-only in this slice. Generated work
products can carry template ID, review state, source lineage, review checklist,
and export intent separately from generation status.

---
## 1. Executive Summary

### Problem Statement

The current WorkspacePlayground is a simplified settings card layout that links users to other features. It doesn't provide an integrated research workflow where users can add sources, chat with context, and
generate outputs in a unified interface.

### Solution

Redesign WorkspacePlayground into a NotebookLM-style three-pane research interface:
- Sources Pane (left): Add and manage research sources
- Chat Pane (middle): RAG-powered conversation with selected sources
- Studio Pane (right): Generate outputs (summaries, quizzes, flashcards, etc.)

### Success Metrics

- Users can complete an end-to-end research workflow without leaving the workspace
- Source-to-chat-to-output pipeline is intuitive and efficient

---
## 2. Current State Analysis

### What Exists Today

- WorkspacePlayground: Simplified settings card UI (post-refactor)
- ModelPlayground: Full developer interface with chat + parameters + debug panels
- Backend APIs: All required endpoints exist and are verified
- Stores: Core option store with RAG slice, data-tables store, evaluations store
- Services: TldwApiClient, flashcards service, quizzes service

### What's Missing

- Workspace store for source/artifact management
- Workspace types definition
- Three-pane layout components
- Source management UI
- Studio output generation UI
- Workspace tagging system implementation

### Reference Implementation

ModelPlayground at apps/packages/ui/src/components/Option/ModelPlayground/ provides the blueprint for:
- Three-pane flex layout with collapsible sidebars
- Storage-based UI state persistence (@plasmohq/storage/hook)
- Zustand store integration patterns
- Ant Design component usage (Collapse, Tabs, Select)

---
## 3. Technical Architecture

### 3.1 Three-Pane Layout

```text
+------------------+------------------------+------------------+
|   Sources Pane   |      Chat Pane         |   Studio Pane    |
|   (collapsible)  |      (flex-1)          |   (collapsible)  |
|     ~280px       |                        |     ~320px       |
+------------------+------------------------+------------------+
```

CSS Structure (following ModelPlayground pattern):
```tsx
<div className="flex h-full flex-col bg-bg text-text">
  <WorkspaceHeader />
  <div className="flex min-h-0 flex-1">
    {leftPaneOpen && <SourcesPane className="w-72 shrink-0 border-r" />}
    <ChatPane className="flex min-w-0 flex-1 flex-col" />
    {rightPaneOpen && <StudioPane className="w-80 shrink-0 border-l" />}
  </div>
</div>
```

### 3.2 Component Structure

```text
WorkspacePlayground/
├── index.tsx                    # Main three-pane layout
├── WorkspaceHeader.tsx          # Title + pane toggle buttons
├── SourcesPane/
│   ├── index.tsx                # Left pane container
│   ├── SourcesHeader.tsx        # "Sources" + Add button
│   ├── AddSourceModal.tsx       # Tabbed: Upload/URL/Paste/Search
│   ├── SourceList.tsx           # Scrollable source list with search
│   ├── SourceItem.tsx           # Source card with checkbox
│   └── ExistingMediaPicker.tsx  # Pick from already-ingested media
├── ChatPane/
│   ├── index.tsx                # Middle pane container
│   ├── ChatContextIndicator.tsx # "Using X sources" badge
│   └── (reuses PlaygroundChat components)
└── StudioPane/
    ├── index.tsx                # Right pane container
    ├── StudioHeader.tsx         # "Studio" label
    ├── OutputGrid.tsx           # 3x3 output type buttons
    ├── GeneratedArtifacts.tsx   # Aggregated outputs list
    ├── ArtifactItem.tsx         # Artifact card with actions
    └── NotesArea.tsx            # Quick notes textarea
```

### 3.3 State Management

New Store: apps/packages/ui/src/store/workspace.ts

```ts
interface WorkspaceSource {
  id: string
  mediaId: number           // Server-side media ID
  title: string
  type: 'pdf' | 'video' | 'audio' | 'website' | 'document' | 'text'
  thumbnailUrl?: string
  addedAt: Date
}

interface GeneratedArtifact {
  id: string
  type: 'summary' | 'audio_overview' | 'mindmap' | 'report' |
        'flashcards' | 'quiz' | 'timeline' | 'slides' | 'data_table'
  title: string
  status: 'pending' | 'generating' | 'completed' | 'failed'
  serverId?: number         // ID from outputs/quizzes/data-tables/slides endpoint
  createdAt: Date
}

interface WorkspaceState {
  // Identity (local persistence)
  workspaceId: string
  workspaceName: string
  workspaceTag: string      // Format: "workspace:<slug>"

  // Sources
  sources: WorkspaceSource[]
  selectedSourceIds: string[]
  sourceSearchQuery: string

  // Studio
  generatedArtifacts: GeneratedArtifact[]
  notes: string
  isGeneratingOutput: boolean

  // UI State
  leftPaneCollapsed: boolean
  rightPaneCollapsed: boolean

  // Actions
  addSource: (source: Omit<WorkspaceSource, 'id' | 'addedAt'>) => void
  removeSource: (id: string) => void
  toggleSourceSelection: (id: string) => void
  selectAllSources: () => void
  deselectAllSources: () => void
  addArtifact: (artifact: Omit<GeneratedArtifact, 'id' | 'createdAt'>) => void
  updateArtifactStatus: (id: string, status: GeneratedArtifact['status']) => void
  setNotes: (notes: string) => void
  toggleLeftPane: () => void
  toggleRightPane: () => void
  reset: () => void
}
```

#### Reuse Existing Stores:
- useStoreMessageOption - Chat messages and streaming (via core-slice)
- useStoreChatModelSettings - Model parameters
- RAG slice from option store - ragMediaIds, ragSearchMode, ragTopK, etc.

### 3.4 API Integration

#### Source Operations
┌─────────────┬───────────────────────────────────────┬─────────────────────────────────────────────┐
│   Action    │               Endpoint                │                    Notes                    │
├─────────────┼───────────────────────────────────────┼─────────────────────────────────────────────┤
│ Upload file │ POST /api/v1/media/add                │ Via tldwClient.addMedia()                   │
├─────────────┼───────────────────────────────────────┼─────────────────────────────────────────────┤
│ Add URL     │ POST /api/v1/media/ingest-web-content │ Via tldwClient.webSearch() with ingest flag │
├─────────────┼───────────────────────────────────────┼─────────────────────────────────────────────┤
│ Web search  │ POST /api/v1/research/websearch       │ Via tldwClient.webSearch()                  │
├─────────────┼───────────────────────────────────────┼─────────────────────────────────────────────┤
│ List media  │ GET /api/v1/media/list                │ Filter by workspaceTag keyword              │
└─────────────┴───────────────────────────────────────┴─────────────────────────────────────────────┘
#### Chat & RAG
┌────────────┬───────────────────────────────┬──────────────────────────────────┐
│   Action   │           Endpoint            │              Notes               │
├────────────┼───────────────────────────────┼──────────────────────────────────┤
│ RAG search │ POST /api/v1/rag/search       │ Via tldwClient.ragSearch()       │
├────────────┼───────────────────────────────┼──────────────────────────────────┤
│ Chat       │ POST /api/v1/chat/completions │ Via existing chat infrastructure │
└────────────┴───────────────────────────────┴──────────────────────────────────┘
RAG Integration: Use existing ragMode by setting:
- ragMediaIds = selected source media IDs
- ragSearchMode = "hybrid_rerank" (recommended)
- ragEnableCitations = true

#### Output Generation
┌────────────────┬────────────────────────────────────┬─────────────────────────────────────────────────────────┐
│  Output Type   │              Endpoint              │                         Method                          │
├────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Summary        │ POST /api/v1/outputs               │ tldwClient.generateOutput() with template               │
├────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Report         │ POST /api/v1/outputs               │ tldwClient.generateOutput() with template               │
├────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Timeline       │ POST /api/v1/outputs               │ tldwClient.generateOutput() with template               │
├────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Audio Overview │ POST /api/v1/audio/speech          │ TTS on generated summary                                │
├────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Flashcards     │ POST /api/v1/flashcards            │ Via flashcards service                                  │
├────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Quiz           │ POST /api/v1/quizzes/generate      │ Via quizzes service                                     │
├────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Data Table     │ POST /api/v1/data-tables/generate  │ tldwClient.generateDataTable()                          │
├────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Mind Map       │ POST /api/v1/chat/completions      │ Prompt for Mermaid/JSON structure                       │
├────────────────┼────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Slides         │ POST /api/v1/slides/generate/media │ Full slides API with templates, export to PDF/reveal.js │
└────────────────┴────────────────────────────────────┴─────────────────────────────────────────────────────────┘
#### Notes
┌───────────┬────────────────────┬─────────────────────────┐
│  Action   │      Endpoint      │          Notes          │
├───────────┼────────────────────┼─────────────────────────┤
│ Save note │ POST /api/v1/notes │ tldwClient.createNote() │
└───────────┴────────────────────┴─────────────────────────┘
### 3.5 Workspace Tagging Strategy

Format: workspace:<slug-or-id> (e.g., workspace:my-research-project)

Implementation:
1. Media items: Tagged via keywords field (existing endpoint supports add/remove)
2. When adding source to workspace: Update media keywords to include workspace tag
3. When listing workspace sources: Filter media by workspace tag keyword
4. Outputs/quizzes/data-tables: Use workspace_tag field where API supports it

Note: Server-side workspace_tag field support may need to be added to some endpoints. For initial implementation, use keyword-based tagging which is already supported.

---
## 4. User Experience

### 4.1 Sources Pane (Left)

#### Add Sources Button - Opens modal with tabs:
1. Upload: Drag-and-drop or file picker for PDF, audio, video, documents
2. URL: Paste URL for website, video, or document
3. Paste Text: Paste raw text content
4. Web Search: Search and ingest results
5. Existing Media: Browse and select from already-ingested media

#### Source List:
- Search/filter bar at top
- Each source shows: type icon, title, checkbox for selection
- Click to view source details (expandable or modal)
- Hover actions: remove from workspace

#### Selection:
- Checkboxes for multi-select
- "Select All" / "Deselect All" buttons
- Selected sources are used as RAG context in chat

### 4.2 Chat Pane (Middle)

#### Context Indicator:
- Badge showing "Using X sources" when sources selected
- Clicking shows list of active sources

#### Chat Interface:
- Reuse existing PlaygroundChat components
- Messages render with source citations when RAG enabled
- Standard features: streaming, markdown, code highlighting, copy

#### RAG Behavior:
- When sources selected: Automatic RAG context injection
- Empty sources: Regular chat without RAG

### 4.3 Studio Pane (Right)

#### Work Product Templates:
- Executive Brief: actionable first-value template using selected sources and
  the report generation path.
- Research Dossier, Competitive Market Memo, Technical Project Spec: visible
  planned templates for roadmap alignment, not end-to-end flows yet.
- Generated work-product artifacts retain source lineage and review checklist
  metadata aligned to
  [Traceable Work Product Artifact Contract](../Traceable_Work_Product_Artifact_Contract.md)
  so review and export affordances can be layered on later.

#### Output Grid (3x3):
```text
+-------------+-------------+-------------+
| Audio       | Summary     | Mind Map    |
| Overview    |             |             |
+-------------+-------------+-------------+
| Report      | Flashcards  | Quiz        |
+-------------+-------------+-------------+
| Timeline    | Slides      | Data Table  |
+-------------+-------------+-------------+
```

#### Generation Flow:
1. Click output type button
2. Optional: Configure output parameters (modal or inline)
3. System uses selected sources + optional chat history
4. Progress indicator during generation
5. Artifact appears in list below

#### Artifacts List:
- Shows all generated outputs for this workspace
- Each artifact: icon, title, status, timestamp
- Actions: preview, download, regenerate, delete

#### Notes Area:
- Simple textarea at bottom
- Auto-save on blur or interval
- "Add Note" button to save as formal note

### 4.4 Responsive Behavior
┌─────────────────┬────────────────────────────────────────────────┐
│   Breakpoint    │                     Layout                     │
├─────────────────┼────────────────────────────────────────────────┤
│ lg+ (1024px+)   │ Full three-pane                                │
├─────────────────┼────────────────────────────────────────────────┤
│ md (768-1023px) │ Chat main, Sources/Studio as slide-out drawers │
├─────────────────┼────────────────────────────────────────────────┤
│ sm (<768px)     │ Bottom tab navigation between panes            │
└─────────────────┴────────────────────────────────────────────────┘
---
## 5. Implementation Stages

### Stage 1: Foundation (Est. files: 5)

Goal: Scaffold three-pane layout with working collapse/expand

Deliverables:
- Create apps/packages/ui/src/types/workspace.ts - Type definitions
- Create apps/packages/ui/src/store/workspace.ts - Zustand store
- Rewrite WorkspacePlayground/index.tsx - Three-pane container
- Create WorkspacePlayground/WorkspaceHeader.tsx - Header with toggles
- Add storage keys for pane collapse state

Verification:
- Three panes render correctly
- Pane toggles work and persist across refresh
- Store initializes with default state

### Stage 2: Sources Pane (Est. files: 6)

Goal: Full source management functionality

Deliverables:
- Create SourcesPane/index.tsx - Container
- Create SourcesPane/SourcesHeader.tsx - Title + Add button
- Create SourcesPane/AddSourceModal.tsx - Tabbed modal
- Create SourcesPane/SourceList.tsx - List with search
- Create SourcesPane/SourceItem.tsx - Source card
- Create SourcesPane/ExistingMediaPicker.tsx - Media browser

Verification:
- Can upload a PDF and see it in source list
- Can add a URL and see it in source list
- Can paste text and see it as source
- Can run web search and add results
- Can pick existing media
- Search/filter works on source list
- Multi-select checkboxes work

### Stage 3: Chat Pane (Est. files: 2)

Goal: RAG-powered chat using selected sources

Deliverables:
- Create ChatPane/index.tsx - Container with PlaygroundChat integration
- Create ChatPane/ChatContextIndicator.tsx - Source count badge
- Wire up RAG context injection using selected ragMediaIds

Verification:
- Chat renders and works without sources (regular mode)
- With sources selected, chat uses RAG context
- Source citations appear in responses
- Context indicator shows correct count
- Streaming works correctly

### Stage 4: Studio Pane (Est. files: 5)

Goal: Output generation and artifacts management

Deliverables:
- Create StudioPane/index.tsx - Container
- Create StudioPane/StudioHeader.tsx - Title
- Create StudioPane/OutputGrid.tsx - 3x3 output buttons
- Create StudioPane/GeneratedArtifacts.tsx - Artifacts list
- Create StudioPane/ArtifactItem.tsx - Artifact card
- Create StudioPane/NotesArea.tsx - Quick notes

Verification:
- All 9 output types trigger generation
- Progress indicator shows during generation
- Generated artifacts appear in list
- Can preview/download artifacts
- Notes save correctly

### Stage 5: Polish & Mobile (Est. files: 2-3)

Goal: Responsive design and UX polish

Deliverables:
- Implement responsive breakpoints
- Add drawer navigation for tablet
- Add tab navigation for mobile
- Empty states for each pane
- Loading states and error handling
- Accessibility improvements (aria labels, keyboard nav)

Verification:
- Test at mobile/tablet/desktop breakpoints
- Drawer/tab navigation works correctly
- Empty states display appropriately
- All interactive elements are accessible

---
## 6. Files to Modify/Create

### New Files (~22 total)

apps/packages/ui/src/
├── types/workspace.ts                           # Type definitions
├── store/workspace.ts                           # Zustand store
└── components/Option/WorkspacePlayground/
    ├── index.tsx                                # Main layout (rewrite)
    ├── WorkspaceHeader.tsx                      # Header component
    ├── SourcesPane/
    │   ├── index.tsx
    │   ├── SourcesHeader.tsx
    │   ├── AddSourceModal.tsx
    │   ├── SourceList.tsx
    │   ├── SourceItem.tsx
    │   └── ExistingMediaPicker.tsx
    ├── ChatPane/
    │   ├── index.tsx
    │   └── ChatContextIndicator.tsx
    └── StudioPane/
        ├── index.tsx
        ├── StudioHeader.tsx
        ├── OutputGrid.tsx
        ├── GeneratedArtifacts.tsx
        ├── ArtifactItem.tsx
        └── NotesArea.tsx

### Files to Reference (patterns)

- ModelPlayground/index.tsx - Three-pane layout, storage persistence
- ModelPlayground/ParametersSidebar.tsx - Collapsible sections
- Playground/PlaygroundChat.tsx - Message rendering
- Playground/PlaygroundForm.tsx - Chat input
- DataTables/SourceSelector.tsx - Source selection patterns
- store/option.tsx - Zustand slice patterns
- store/data-tables.tsx - Feature store patterns

### Storage Keys to Add

// In utils/storage-migrations.ts
export const WORKSPACE_LEFT_PANE_KEY = "workspaceLeftPaneOpen"
export const WORKSPACE_RIGHT_PANE_KEY = "workspaceRightPaneOpen"

---
## 7. Technical Decisions
┌───────────────────────┬─────────────────────────────────────┬─────────────────────────────────────────────────────────┐
│       Decision        │               Choice                │                        Rationale                        │
├───────────────────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Chat mode             │ Reuse ragMode                       │ Already handles ragMediaIds, no new mode needed         │
├───────────────────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Workspace persistence │ Local-only (zustand + localStorage) │ Simplifies MVP; server sync can come later              │
├───────────────────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Artifact storage      │ Server-side                         │ Outputs/quizzes/data-tables already persist server-side │
├───────────────────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ Workspace tagging     │ Keyword-based                       │ Uses existing media keywords API; no backend changes    │
├───────────────────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ UI framework          │ Ant Design + Tailwind               │ Consistent with rest of codebase                        │
├───────────────────────┼─────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ State management      │ Zustand                             │ Consistent with existing stores                         │
└───────────────────────┴─────────────────────────────────────┴─────────────────────────────────────────────────────────┘
---
## 8. Out of Scope (Future)

- Server-side workspace persistence (workspace metadata, sharing)
- Real-time collaboration
- Workspace templates
- Advanced workspace organization (folders, tags)
- Bulk export of workspace contents
- Workspace analytics/insights

---
## 9. Verification Plan

### End-to-End Test Scenario

1. Create workspace: Open WorkspacePlayground, verify three panes render
2. Add sources:
  - Upload a PDF -> verify appears in source list
  - Add a URL -> verify ingests and appears
  - Add existing media -> verify workspace tag applied
3. Use chat:
  - Select 2 sources via checkboxes
  - Verify "Using 2 sources" badge appears
  - Send a question -> verify RAG context in response
  - Verify citations appear
4. Generate outputs:
  - Click "Summary" -> verify progress, then artifact appears
  - Click "Quiz" -> verify quiz generated
  - Preview/download artifacts -> verify working
5. Notes: Add a note, refresh page, verify note persists
6. Pane state: Collapse left pane, refresh, verify still collapsed
7. Responsive: Test at 768px and 480px breakpoints

### Unit Test Coverage

- Workspace store: all actions and selectors
- AddSourceModal: each tab's submission flow
- OutputGrid: each output type's generation trigger
- ChatContextIndicator: count display logic

---
## 10. Dependencies & Risks

### Dependencies

- Backend endpoints all exist (verified)
- Flashcards/quizzes services exist
- TldwApiClient has required methods

### Risks
┌─────────────────────────────────────┬───────────────────────────────────────────────────────┐
│                Risk                 │                      Mitigation                       │
├─────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Output generation may be slow       │ Show progress indicators, allow background generation │
├─────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Large source lists may be slow      │ Implement virtualization if >100 sources              │
├─────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ RAG context may exceed token limits │ Implement context truncation/selection                │
├─────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Mobile UX may be awkward            │ Prioritize tablet-first responsive design             │
└─────────────────────────────────────┴───────────────────────────────────────────────────────┘
---
## 11. Resolved Design Decisions
┌──────────────────────┬──────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
│       Question       │           Decision           │                                           Notes                                           │
├──────────────────────┼──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Workspace naming     │ User-provided names          │ User enters name when creating workspace                                                  │
├──────────────────────┼──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Output types for MVP │ All 9 types                  │ Audio Overview, Summary, Mind Map, Report, Flashcards, Quiz, Timeline, Slides, Data Table │
├──────────────────────┼──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Image generation     │ Slides API for presentations │ Use /api/v1/slides endpoints; Mind Map uses Mermaid text output                           │
├──────────────────────┼──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Chat threads         │ Single chat per workspace    │ One continuous conversation like NotebookLM                                               │
├──────────────────────┼──────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
│ Slides generation    │ Full Slides API              │ Supports generation from media, templates, export to PDF/reveal.js                        │
└──────────────────────┴──────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────┘
---
## 12. Open Questions

1. Source limits: Is there a max number of sources per workspace?
2. Output templates: Which output templates should be pre-configured vs. customizable?
3. Audio Overview: Should this use default TTS voice or allow selection?

---
Document Version: 1.1
Last Updated: 2026-01-27
