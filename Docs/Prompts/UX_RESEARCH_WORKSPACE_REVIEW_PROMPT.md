# UX/HCI Expert Review: Research Workspace

## What You're Reviewing

The **Research Workspace** is a NotebookLM-inspired three-pane research interface in a media analysis and knowledge management application. It lets users ingest sources, chat with them using RAG (Retrieval-Augmented Generation), and generate study/analysis outputs.

### Current Layout (Desktop 1024px+)
```
┌──────────────────────────────────────────────────────┐
│ WorkspaceHeader: title (editable), pane toggles,     │
│ workspace browser, import/export, global search      │
├──────────┬───────────────────────┬───────────────────┤
│ Sources  │      Chat Pane        │   Studio Pane     │
│  Pane    │                       │                   │
│ (280px)  │    (flexible)         │    (320px)        │
│          │                       │                   │
└──────────┴───────────────────────┴───────────────────┘
```
- Tablet (768-1023px): Chat main + slide-out drawers for Sources/Studio
- Mobile (<768px): Bottom tab navigation switching between 3 panes

### What Each Pane Does

**Sources Pane (Left)**
- Add sources via: file upload, URL, paste text, web search, existing media library
- Source types: PDF, DOCX, EPUB, video, audio, website, plain text
- Each source shows: type icon, title, file size, duration/pages, status (processing/ready/error)
- Select/deselect sources (checkboxes) to scope RAG retrieval
- Search, reorder (drag-drop), remove (with undo)

**Chat Pane (Center)**
- RAG-powered Q&A against selected sources
- Shows selected sources as horizontal tags
- Streams LLM responses with citations back to sources
- Configurable: search mode (hybrid/vector/FTS), top-K, min relevance, reranking toggle
- Optional retrieval diagnostics (chunks retrieved, sources used, relevance scores)

**Studio Pane (Right)**
- Generate 10 artifact types from selected sources:
  - Study Aids: Quiz, Flashcards
  - Analysis: Summary, Report, Compare Sources, Timeline, Data Table
  - Creative: Mind Map, Slides, Audio Overview (TTS)
- Each artifact shows: status, progress, preview, export/download, delete/restore
- Quick Notes section for capturing knowledge
- Audio settings (provider, voice, speed, format)

### Available Keyboard Shortcuts
- Cmd+K: Global search
- Cmd+1/2/3: Focus pane
- Cmd+N: New note
- Cmd+Shift+N: New workspace
- Cmd+Z: Undo (10s window)

### Backend Capabilities Available (not all surfaced in UI)
- Hybrid RAG pipeline with FTS5 + vector + reranking + optional research loops
- Implicit feedback system for personalizing search results
- Chat analytics and queue status monitoring
- RAG ablation testing (baseline vs reranked vs agentic comparison)
- Batch RAG search with checkpoint/resume
- Conversation sharing via share links
- Note collections with keyword tagging
- Chat slash commands
- RAG feature toggles (query classification, reformulation, discussion search, search depth mode)
- 16+ LLM providers with model metadata

---

## Review Dimensions

Please evaluate the page across these dimensions, providing specific findings for each:

### 1. Information Architecture & Discoverability
- Is the three-pane mental model (Input → Discovery → Output) clear to first-time users?
- Can users understand what each pane does without a tutorial?
- Are features discoverable or buried? (e.g., RAG settings, keyboard shortcuts, workspace switching)
- Is the "Add Source" flow intuitive across its 5 tabs?
- How does a new user know what artifact types are available and what they produce?

### 2. Information Density & Missing Signals
- What information does the user *need* to see that isn't shown? Consider:
  - Source processing progress (% complete, ETA, error details)
  - RAG retrieval transparency (why was this chunk selected? what was the relevance score?)
  - Token/cost tracking per chat message and per artifact
  - Model currently in use and its capabilities
  - Workspace storage usage (localStorage quota)
  - Source chunk count and quality metrics
  - Citation provenance (which exact passage was cited?)
- What information is shown but shouldn't be (noise)?
- Are status indicators (processing, generating, error) clear and actionable?

### 3. User Flows & Task Completion
Evaluate these critical flows for friction, dead ends, and missing affordances:
- **First-time setup**: Empty state → add first source → first chat → first artifact
- **Multi-source research**: Add 5+ sources → select subset → compare → generate report
- **Iterative refinement**: Generate quiz → review → adjust parameters → regenerate
- **Cross-pane workflows**: Find insight in chat → capture to notes → generate summary
- **Error recovery**: Source fails to process → what can the user do?
- **Workspace management**: Create, switch, duplicate, archive, export/import workspaces

### 4. Responsive Design & Device Parity
- Do tablet drawer overlays feel natural or intrusive?
- Does the mobile bottom-tab pattern lose critical context when switching panes?
- Is drag-and-drop (sources to chat) viable on touch devices?
- Are there features that silently disappear on smaller screens?

### 5. Progressive Disclosure & Complexity Management
- Are advanced features (RAG tuning, audio settings, search depth) appropriately hidden from beginners?
- Is there a clear path from basic usage to power-user workflows?
- Do sensible defaults reduce initial cognitive load?
- Are 10 artifact types too many to present at once? How should they be organized?

### 6. Feedback & System Status
- Does the user always know what the system is doing? (loading, searching, generating, streaming)
- Are progress indicators meaningful (determinate vs indeterminate)?
- Is the undo system (Cmd+Z, 10s window) discoverable?
- How does the system communicate storage quota warnings and cross-tab conflicts?

### 7. Accessibility & Inclusivity
- Keyboard navigation completeness across all three panes
- Screen reader experience (ARIA labels, live regions for streaming)
- Color contrast and color-independence of status indicators
- Focus management on pane switches and modal opens
- Touch target sizes on mobile

### 8. Missing Functionality & Feature Gaps
Based on comparable tools (NotebookLM, Elicit, Semantic Scholar, Readwise Reader), what capabilities would a user expect that are missing? Consider:
- Source annotation/highlighting within the workspace
- Collaborative features (sharing workspaces, co-editing)
- Version history for workspaces (not just artifacts)
- Source quality indicators (reliability, recency, relevance)
- Export entire research session (sources + chat + artifacts + notes)
- Template workspaces for common research patterns
- Integration with reference managers (Zotero, Mendeley)
- Automated source suggestions ("you might also want to read...")

---

## Output Format

For each finding, provide:

| Field | Description |
|-------|-------------|
| **ID** | Sequential (e.g., UX-001) |
| **Dimension** | Which review dimension (1-8) |
| **Severity** | Critical / Major / Minor / Enhancement |
| **Finding** | Clear description of the issue or gap |
| **Impact** | Who is affected and how (new users, power users, mobile users, etc.) |
| **Recommendation** | Specific, actionable suggestion |
| **Comparable** | How other tools handle this (if applicable) |

### Severity Definitions
- **Critical**: Blocks core task completion or causes data loss
- **Major**: Significant friction in common workflows; users may abandon the feature
- **Minor**: Noticeable but workaroundable; affects polish and trust
- **Enhancement**: Not a problem today but would meaningfully improve the experience

### Summary Deliverables
1. **Executive Summary**: 3-5 sentence overview of the page's UX maturity
2. **Top 5 Priority Fixes**: Highest-impact improvements ranked by effort-to-impact ratio
3. **Information Gaps Table**: Backend data available but not surfaced, with priority rating
4. **Competitive Gap Analysis**: Features present in comparable tools but missing here
5. **Detailed Findings Table**: All findings in the format above
