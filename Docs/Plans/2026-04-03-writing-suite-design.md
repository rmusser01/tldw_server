# Writing Suite Design: Evolving WritingPlayground into a Creative Writing Platform

## Context

The tldw_server Writing Playground is currently a text-generation sandbox focused on LLM parameter control (sessions, templates, themes, tokenization, wordcloud). It uses a plain textarea editor and stores session content as opaque JSON blobs.

The goal is to evolve it into a **research + creative hybrid writing suite** inspired by the NovelWriter project, leveraging tldw's existing media ingestion, RAG pipeline, and LLM infrastructure. The writing module becomes the "output" side of the research workflow — users ingest media, build knowledge, then write with that knowledge embedded in their creative environment.

**Key decisions made:**
- **Editor**: TipTap/ProseMirror (rich text, extensible custom nodes) — NEW dependency, not currently in project
- **Storage**: Structured DB tables (not JSON blobs) for manuscript hierarchy
- **Architecture**: Evolve existing WritingPlayground (not a separate page)
- **Phasing**: 4-phase rollout
- **Research integration**: Full (inline citations + sidebar + AI context injection)

## Review Notes & Mitigations

Issues identified during design review:

1. **Characters/WorldBooks are NOT embeddable** — Both are monolithic page-level workspaces (Manager.tsx at 44KB and 101KB). Plan revised: Phase 2 builds **new compact manuscript-specific character/world components** backed by the new manuscript_characters/manuscript_world_info tables, NOT embedding existing page components. The existing Characters and WorldBooks pages remain separate.

2. **FTS5 sync strategy** — Must follow the existing sync_log + trigger pattern used by Prompts_DB, MediaDB, Slides_DB, and Collections_DB. Added explicit sync trigger requirement.

3. **Word count propagation** — Application-level cascading on scene PATCH: update scene.word_count → SUM chapter.word_count → SUM part.word_count → SUM project.word_count. Done in a single transaction in ManuscriptDB.py. Not via DB triggers (too fragile with soft deletes).

4. **sort_order REAL precision** — After 1000+ fractional insertions between the same two nodes, call a re-normalization endpoint that reassigns integer sort_orders (1.0, 2.0, 3.0...). The reorder endpoint does this automatically when gap < 0.001.

5. **Data flow clarified** — Manuscript structure lives in structured DB tables (backend). The Zustand store holds a lightweight mirror of the tree for UI navigation. React Query fetches/caches the tree from the API. No manuscript data in session payloads.

6. **Focus mode shortcut** — Use `Ctrl+Shift+F` (Cmd+Shift+F on Mac) instead of F11 which conflicts with browser fullscreen.

7. **Real-time feedback cost (Phase 4)** — Mood detection uses a small/fast model (configurable, default Haiku-class). Echo Chamber triggers only after 500+ characters typed (not on every keystroke). Both are opt-in with persistent toggle. Rate-limited to max 1 call per 10 seconds.

8. **TipTap bundle size** — Lazy-loaded (`React.lazy`) only when user selects TipTap mode. Plain textarea remains default for existing sessions. TipTap's tree-shakeable architecture means only imported extensions are bundled (~150KB gzipped for StarterKit).

---

## Phase 1: Foundation — TipTap + Manuscript Structure

### Backend

**New DB tables** (migration V40→V41 in ChaChaNotes_DB.py):

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `manuscript_projects` | Top-level container | title, author, genre, status, synopsis, target_word_count, word_count, settings_json |
| `manuscript_parts` | Optional grouping (Book 1, Act II) | project_id FK, title, sort_order (REAL for fractional), word_count |
| `manuscript_chapters` | Chapters | project_id FK, part_id FK (nullable), title, sort_order, synopsis, pov_character_id, status |
| `manuscript_scenes` | Leaf content nodes | chapter_id FK, project_id FK, content_json (TipTap), content_plain (extracted for search), word_count, status |
| `manuscript_scenes_fts` | FTS5 virtual table | title, content_plain, synopsis |

All tables follow existing patterns: UUID TEXT PKs, soft delete, version tracking, client_id, created_at/last_modified, sync triggers (4 per table: create/update/delete/undelete → sync_log, matching the pattern in Prompts_DB.py and existing ChaChaNotes tables).

**FTS5 sync**: `manuscript_scenes_fts` uses `content='manuscript_scenes'` with application-level rebuild on INSERT/UPDATE/DELETE of scenes (same pattern as `prompts_fts` and `chunk_fts_ops.py`).

**Word count propagation**: On scene content update, ManuscriptDB.py recalculates `word_count` from `content_plain`, then cascades: `UPDATE manuscript_chapters SET word_count = (SELECT SUM(word_count) FROM manuscript_scenes WHERE chapter_id = ? AND deleted = 0)`, similarly for parts and projects. All in one transaction.

**New endpoint file**: `writing_manuscripts.py` at `/api/v1/writing/manuscripts/...`

Core endpoints:
- `GET/POST /manuscripts/projects` — list/create projects
- `GET/PATCH/DELETE /manuscripts/projects/{id}` — project CRUD
- `GET /manuscripts/projects/{id}/structure` — full tree (parts→chapters→scenes with word counts)
- `POST /manuscripts/projects/{id}/reorder` — batch reorder with `ReorderItem(id, sort_order, new_parent_id)`
- CRUD for parts, chapters, scenes at nested paths
- `GET /manuscripts/projects/{id}/search` — FTS5 full-text search across scenes

**New schema file**: `writing_manuscript_schemas.py`
- ManuscriptProjectCreate/Update/Response
- ManuscriptSceneCreate/Update/Response (content as TipTap JSON dict)
- ManuscriptStructureResponse (PartSummary→ChapterSummary→SceneSummary tree)
- ReorderRequest with entity_type and items list

**New DB helper**: `ManuscriptDB.py` — receives `CharactersRAGDB` instance, delegates to its connection/transaction infrastructure. Keeps CRUD code out of the 26K-line ChaChaNotes_DB.py.

### Frontend

**index.tsx decomposition** (prerequisite refactoring, no behavior change):
- Extract `WritingEditorToolbar.tsx` (lines ~2107-2139)
- Extract `WritingPlainEditor.tsx` (lines ~2167-2192, kept as fallback)
- Extract `WritingEditorStatusBar.tsx` (lines ~2253-2260)
- Extract `SessionListPanel.tsx` (lines ~1932-1999)
- Extract `WritingSearchReplace.tsx` (lines ~2140-2166)
- Extract generation logic into `useWritingGeneration` hook

**TipTap integration** (NEW dependency — not currently in project, requires installation in @tldw/ui, tldw-frontend, and extension package.json files):
- New packages: `@tiptap/react`, `@tiptap/starter-kit`, `@tiptap/extension-placeholder`, `@tiptap/extension-character-count`, `@tiptap/pm`
- `WritingTipTapEditor.tsx` — EditorContent wrapper with StarterKit + custom extensions
- `SceneBreakExtension` — custom block node for `***` scene breaks
- Editor mode toggle (plain/tiptap) persisted to storage
- Backward compat bridge: `plainTextToTipTap()` / `tipTapToPlainText()` for generation pipeline

**Manuscript tree**:
- `ManuscriptTreePanel.tsx` in library panel with `@dnd-kit/react` drag-drop (same pattern as AudiobookStudio's ChapterList)
- Library panel gets `[Sessions] [Manuscript]` segmented control
- Tree shows project→parts→chapters→scenes with collapse/expand and word counts

**Zustand store extensions**:
```
activeProjectId, activeNodeId, editorMode, focusMode
```
Session content state stays in hooks. UI navigation state (`activeProjectId`, `activeNodeId`, `editorMode`, `focusMode`) moves to store. Manuscript tree data is **NOT** in Zustand — it's fetched and cached via React Query (`useQuery` on `/manuscripts/projects/{id}/structure`) to stay consistent with the server-authoritative data model. This avoids dual-source-of-truth issues between Zustand and the DB.

**Focus mode**: `Ctrl+Shift+F` (Cmd+Shift+F on Mac) hides both sidebars, shows only editor with comfortable margins. Escape exits. (Avoids F11 which conflicts with browser fullscreen.)

### Critical files
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` — migration SQL
- `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx` — decomposition source
- `apps/packages/ui/src/store/writing-playground.tsx` — store extensions
- `apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundShell.tsx` — focus mode

---

## Phase 2: Knowledge — Characters, World Info, Research

### Backend

**New DB tables** (same migration or V41→V42):

| Table | Purpose |
|-------|---------|
| `manuscript_characters` | name, role (protagonist/antagonist/supporting/minor/mentioned), cast_group, appearance, personality, backstory, motivation, arc_summary, custom_fields_json |
| `manuscript_character_relationships` | from_character_id, to_character_id, relationship_type, description, bidirectional |
| `manuscript_world_info` | kind (location/item/faction/concept/event/custom), name, description, parent_id (hierarchical), properties_json, tags_json |
| `manuscript_plot_lines` | title, description, status (active/resolved/abandoned/dormant), color |
| `manuscript_plot_events` | plot_line_id FK, scene_id FK, chapter_id FK, title, description, sort_order |
| `manuscript_plot_holes` | title, severity (low-critical), status (open/investigating/resolved/wontfix), detected_by (manual/ai) |
| `manuscript_scene_characters` | scene_id, character_id linking table |
| `manuscript_scene_world_info` | scene_id, world_info_id linking table |
| `manuscript_citations` | scene_id FK, source_type, source_id, excerpt, query_used, anchor_offset |

**New endpoints**:
- Character CRUD at `/manuscripts/projects/{id}/characters`
- Relationship graph CRUD at `/manuscripts/projects/{id}/characters/relationships`
- World info CRUD at `/manuscripts/projects/{id}/world-info`
- Plot lines/events/holes CRUD
- Research: `POST /manuscripts/scenes/{id}/research` — RAG query contextualized with scene content
- Citations: CRUD at `/manuscripts/scenes/{id}/citations`

**Research integration**: The research endpoint gathers scene content_plain, synopsis, and character/world context, constructs a contextualized query, and calls `unified_rag_pipeline()` from the existing RAG service.

### Frontend

**`CharacterWorldTab.tsx`** — new inspector tab with **purpose-built compact components** for manuscript characters and world info. NOT embedding existing Characters/WorldBooks page components (those are 44KB and 101KB monolithic workspaces that can't be easily composed). Instead, builds lightweight list+detail views backed by the new `manuscript_characters` and `manuscript_world_info` DB tables. Segmented: `[Characters] [World Info] [Plot Grid]`.

**`ResearchTab.tsx`** — new inspector tab with:
- RAG search input querying `/api/v1/rag/search`
- Compact result cards (title, snippet, source)
- Drag-to-cite: drop result into TipTap to create inline `citation` node

**`CitationExtension`** — TipTap inline node:
- Attrs: sourceId, text, mediaId
- Renders as styled inline span
- Click opens source preview in Research tab

**Inspector tab evolution**: `InspectorTabKey` gains `"characters"` and `"research"`. Dynamic tab array with `extraTabs` prop.

### Critical files
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` — `unified_rag_pipeline()`
- `apps/packages/ui/src/services/writing-playground.ts` — API client extensions
- Existing `Characters/` and `WorldBooks/` components as design reference (not direct reuse)

---

## Phase 3: Analysis — AI Tools + Agent Chat

### Backend

**New DB table**:

| Table | Purpose |
|-------|---------|
| `manuscript_ai_analyses` | scope_type, scope_id, analysis_type (pacing/tension/mood/plot_holes/reader_reaction/consistency), provider, model, result_json, score, stale flag |

**New endpoints**:
- `POST /manuscripts/scenes/{id}/analyze` — trigger AI analysis (pacing, tension, mood)
- `POST /manuscripts/chapters/{id}/analyze` — chapter-level analysis
- `POST /manuscripts/projects/{id}/analyze/plot-holes` — AI plot hole detection
- `POST /manuscripts/projects/{id}/analyze/consistency` — character/world consistency check
- `GET /manuscripts/projects/{id}/analyses` — list cached results

Analysis flow: gather scope content → construct structured LLM prompt → call via existing LLM_Calls provider → parse structured JSON response → store in manuscript_ai_analyses. Scene PATCH marks matching analyses as `stale=1`.

### Frontend

**`WritingAnalysisModalHost.tsx`** — lazy-loaded modals launched from toolbar/inspector:

1. **Story Pulse modal** — multi-metric line chart (pacing, tension, atmosphere per chapter) using recharts or similar. AI scores each chapter via structured output prompt.

2. **Plot Tracker modal** — Ant Table showing open/concluded plot lines and plot holes with severity badges. Supports both AI-generated and manual entries.

3. **Event Line modal** — vertical timeline of story events, each linked to a chapter/scene, color-coded by type (setup, conflict, action, emotional).

4. **Connection Web modal** — relationship graph using Cytoscape.js (already a project dependency) with cytoscape-dagre layout. Nodes = characters/factions/locations, edges = relationship types.

**`AIAgentTab.tsx`** — new inspector tab with context-aware chat:
- Mode selector: Quick (single-turn), Planning (structured output), Brainstorm (free-form)
- Context injection: current scene text + character sheets + world info + chapter synopsis
- Per-project conversation history stored in session payload
- Uses existing `TldwChatService` for LLM communication

### Critical files
- `tldw_Server_API/app/core/LLM_Calls/` — LLM provider abstraction
- Cytoscape.js (already in dependency tree)

---

## Phase 4: Live Feedback — Mood Detection + Echo Chamber

### Frontend-heavy (minimal backend)

**`FeedbackTab.tsx`** — new inspector tab (opt-in toggle):

1. **Mood detection** — debounced after typing (min 10s between calls), sends last N paragraphs to a small/fast model (configurable, default Haiku-class) for mood classification (tense, romantic, melancholic, action, calm, mysterious, humorous). Colored indicator in status bar. Opt-in with persistent toggle.

2. **Echo Chamber** — 5 simulated reader personalities react to passages after 500+ characters typed (not per-keystroke). Rate-limited to max 1 call per 30 seconds. Opt-in. Personalities:
   - Alex (Analyst) — notices plot holes and structure
   - Sam (Shipper) — obsessed with relationships
   - Max (Skeptic) — questions motivations
   - Riley (Hype) — excited about action/twists
   - Jordan (Lore Keeper) — tracks world-building

3. **`AIAnnotationExtension`** — TipTap mark for AI-generated/suggested text with visual styling.

4. **`useWritingFeedback` hook** — manages debounced analysis, result caching, opt-in state.

---

## Architecture Summary

```
Backend:
  ChaChaNotes_DB.py (migration V40→V41)
    └── 12 new manuscript_* tables + FTS5
  ManuscriptDB.py (new helper)
    └── CRUD methods delegating to CharactersRAGDB
  writing_manuscripts.py (new endpoint file)
    └── ~50 REST endpoints under /api/v1/writing/manuscripts/
  writing_manuscript_schemas.py (new schemas)
    └── ~25 Pydantic models

Frontend:
  WritingPlayground/ (evolved)
    ├── index.tsx (slimmed from 2359 to ~500 lines)
    ├── WritingTipTapEditor.tsx (TipTap + custom extensions)
    ├── ManuscriptTreePanel.tsx (@dnd-kit tree)
    ├── CharacterWorldTab.tsx (embeds existing components)
    ├── ResearchTab.tsx (RAG search + drag-to-cite)
    ├── AIAgentTab.tsx (context-aware chat)
    ├── FeedbackTab.tsx (mood + Echo Chamber)
    ├── WritingAnalysisModalHost.tsx (4 analysis modals)
    └── 6 extracted components from index.tsx
  store/writing-playground.tsx (extended with manuscript state)
```

## Verification Plan

**Phase 1**:
- DB migration test: create V41 tables on fresh and existing DBs
- API smoke test: create project → add part → add chapter → add scene → update scene content → verify word count propagation → reorder → search via FTS5
- Frontend: TipTap renders, mode toggle works, plain↔TipTap conversion preserves content, generation still works via plain text bridge
- Extension parity: same layout and behavior in browser extension
- Focus mode: keyboard shortcut toggles, Escape exits

**Phase 2**:
- Character CRUD through API and UI
- Research search returns RAG results
- Drag citation into editor creates citation node
- Character/world entries auto-inject into generation context

**Phase 3**:
- Analysis endpoints return structured results
- Story Pulse chart renders with mock data, then with real LLM analysis
- Cytoscape graph renders relationship data
- AI agent chat sends context-aware prompts

**Phase 4**:
- Mood detection fires on debounced typing
- Echo Chamber reactions display in Feedback tab
- All features opt-in with persistent toggle state
