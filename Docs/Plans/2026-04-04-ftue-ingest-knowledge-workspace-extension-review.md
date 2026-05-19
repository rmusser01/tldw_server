# FTUE Review: Ingest Modal, /knowledge, /document-workspace, and Extension

**Date:** 2026-04-04
**Scope:** Fresh walkthrough of in-app UX for first-time users of the ingest modal, /knowledge page, /document-workspace page, and extension-specific flows
**Personas:** Competent non-technical user (Persona A), Competent technical user (Persona B)
**Builds on:** Existing v1 audit (33 issues) and v2 audit (13 issues) which focused on setup/config/docs
**Tutorials count:** 15 distinct tutorials (not 18 as previously estimated)

---

## Review Notes (self-review corrections)

The following corrections were identified during plan review and are incorporated below:

1. **DOC-002 escalated to P0** — QuickIngest wizard never sends `keep_original_file=true` (backend defaults to `False`). Document Workspace is broken for ALL documents ingested via the standard path. Must be Sprint 1.
2. **KNW-001 demoted to P3** — Source categories ARE pre-selected by default (`media_db`, `notes`, `characters`, `chats` in `unified-rag.ts:260`). "No sources selected" only occurs for returning users who explicitly cleared sources.
3. **ING-003 clarification** — "Use defaults & process" button disappears entirely for batches of 2+ items (`queueItems.length <= 1`). Not just visually secondary — absent for multi-item queues.
4. **EXT-003 label correction** — Actual label is "Process with tldw_server (don't store)", not "Process page locally". Both options use the server; the distinction is storage, not processing location.
5. **Overlaps with existing audits acknowledged** — XC-003 overlaps v1 FE-004/Improvement 11, EXT-006 overlaps v1 Improvement 12, XC-004 overlaps v1 ERR-004/Improvement 10. These should be merged with existing work, not duplicated.
6. **Missing perspectives added** — Mobile users, multi-user/team scenarios, extension-first users, accessibility, upgrade users with existing content.

---

## Part 1: Narrative Walkthroughs

### 1.1 Ingest Modal (QuickIngest Wizard)

**Scenario:** User has completed onboarding and wants to add their first content.

#### Persona A (Non-technical) Walkthrough

The user opens the Quick Ingest modal. They see a file drop zone and a URL input area.

**What works well:**
- Drop zone has clear "Drag and drop files" messaging with supported format hints
- URL placeholder shows realistic examples (`https://example.com/article`, `https://youtube.com/watch?v=...`)
- File validation is immediate and clear (size limits, type checks)
- Auto-detection of media type with "(auto)" indicator is reassuring
- FFmpeg missing warning is proactive and helpful

**Where the user gets confused:**

1. **The "Configure" step is overwhelming.** After adding a file, the user hits "Configure" and sees: Analysis toggle, Chunking toggle, Overwrite toggle, Audio options (language, diarization, transcription model), Document options (OCR), Video options (captions), plus Storage configuration with "Review before saving" mode. A non-technical user doesn't know what "chunking" means, what "diarization" is, or why they'd want "review before saving." The preset selector (Quick/Standard/Deep) helps but doesn't explain the trade-offs in plain language.

2. **Storage configuration is confusing.** "Where ingest results are stored" with Server vs Local toggle, plus "Review before saving" with a "Review mode" badge and local draft cap. A non-technical user just wants their file processed — they don't understand the architecture of server-side vs browser-side storage.

3. **No "just do it" fast path.** There IS a "Use defaults & process" button, but it has two problems: (a) it's positioned as a secondary button alongside the primary "Configure" button, making it easy to miss, and (b) it **disappears entirely when 2+ items are queued** (`queueItems.length <= 1` gate in `AddContentStep.tsx:469`). A user who adds two PDFs has no fast path at all.

4. **Post-completion guidance is weak.** The Results step shows "Open" and "Chat" buttons per item, plus "Ingest More" and "Done". But there's no guidance like "Your document is now searchable in Knowledge QA" or "Open in Document Workspace to annotate and study." The user doesn't know what they unlocked.

5. **Inspector drawer is hidden knowledge.** The Inspector (click a queued item to see details) has a helpful "How to use the Inspector" intro, but users may never discover it — there's no visual cue that items are clickable.

#### Persona B (Technical) Walkthrough

**What works well:**
- Full control over processing options (presets, per-type options, OCR, diarization)
- Inspector drawer with detailed validation info, MIME types, and advanced hints
- Error classification in results with retry/remove actions and category badges
- Real-time SSE progress with per-stage indicators (Upload → Process → Analyze → Store)
- Background processing via "Minimize to Background"

**Where the technical user hits friction:**

1. **No batch configuration per-item.** The configure step sets defaults for ALL items. If you queue 5 URLs of different types and want different settings per item, the only way is the Inspector drawer (which only inspects, doesn't clearly allow per-item config overrides).

2. **Transcription model dropdown loads dynamically** but shows a spinner with no indication of what models are available or why they matter. A technical user wants to know: which STT engine? What's the speed/accuracy trade-off?

3. **No API-level feedback.** The processing step shows stages (Upload/Process/Analyze/Store) but doesn't expose the actual API job ID, server logs, or debugging info. When something fails with "Processing: Server processing error" the user can't dig deeper.

4. **Error classification is good but remediation is generic.** Timeout errors say "Processing took too long" with a suggestion but no link to server config for timeout tuning. Network errors don't suggest checking CORS or firewall rules.

---

### 1.2 /knowledge Page (Knowledge QA)

**Scenario:** User has ingested 1-2 documents and navigates to /knowledge for the first time.

#### Persona A (Non-technical) Walkthrough

The user arrives at the Knowledge QA page. They see "Ask Your Library" with a search bar and a "How It Works" section.

**What works well:**
- "How It Works" 3-step explanation is clear and collapses for returning users
- Suggested prompts adapt: users with no sources get onboarding prompts ("How do I add my first source?"), users with sources get research prompts
- Search bar rotates example queries as placeholder text — good discovery mechanism
- No-results recovery offers three actionable buttons (broaden scope, enable web search, show nearest)
- Error messages are categorized with specific guidance (timeout → "try Fast preset")

**Where the user gets confused:**

1. **~~Source selection first step~~ [CORRECTED: Sources ARE pre-selected by default].** Sources default to `media_db`, `notes`, `characters`, `chats` in `unified-rag.ts:260`. The "No sources selected" warning only appears if a returning user previously cleared their selection. However, the source *category names* (Documents & Media, Notes, Character Cards, Chat History, Kanban) are still confusing for non-technical users — the issue is naming, not selection state.

2. **The relationship between ingest and searchability is opaque.** User ingested a PDF via Quick Ingest. They come to /knowledge and search for something in it. If the document wasn't chunked or embedded (because they used wrong settings in ingest), they get no results with no explanation of why. There's no "Your document may not be indexed yet — check processing status" message.

3. **Preset selector (Fast/Balanced/Deep) lacks plain-language trade-offs.** The compact toolbar shows just the preset name. The full context bar adds descriptions, but even "Quick lookup with minimal retrieval and rerank depth" uses jargon. A non-technical user needs: "Fast: quick answers, might miss details" / "Deep: thorough but slower."

4. **Settings panel has a Basic/Expert mode toggle** that's good in concept, but Basic mode still shows "Search Mode: Hybrid (Recommended) / Vector Only / Full-Text Only" — a non-technical user doesn't know what vector or full-text search means.

5. **Web search fallback is powerful but confusing.** The toggle says "Falls back to web search when local source relevance is below threshold" — what threshold? When? The warning "Web search isn't available on this server" appears without guidance on how to enable it.

6. **Faithfulness/trust indicators appear without context.** "Strong/Partial/Weak" faithfulness labels appear on answers, but there's no tooltip or explanation of what faithfulness means or why a "Weak" answer might still be useful.

#### Persona B (Technical) Walkthrough

**What works well:**
- Expert mode with 150+ RAG options is comprehensive
- Query suggestions from history/sources/examples are helpful
- Source filtering and sorting (relevance, title, date, cited) with type facets
- Streaming stages (searching → ranking → generating → verifying) provide transparency
- Chatbook export for sharing threads

**Where the technical user hits friction:**

1. **No visibility into retrieval pipeline.** Which chunks were retrieved? What were their scores? Was reranking applied? The answer shows citations but not the underlying retrieval metrics. A technical user tuning RAG parameters needs this feedback loop.

2. **Expert settings are a wall of options.** 150+ settings with a key-level JSON editor is powerful but overwhelming even for technical users. No grouping by "most impactful" or "commonly tuned" parameters.

3. **Web fallback behavior is a black box.** When does it trigger? What provider is it using? What was the relevance threshold? The UI doesn't expose this.

4. **No A/B comparison.** Can't run the same query with different presets side-by-side to compare results. Feature flags exist for comparison mode (`ff_knowledgeQaComparison`) but it's behind a flag.

---

### 1.3 /document-workspace Page

**Scenario:** User wants to read and study a document they ingested.

#### Persona A (Non-technical) Walkthrough

The user navigates to /document-workspace. They see an empty three-pane layout.

**What works well:**
- Empty state shows 4 feature discovery cards (Highlight & Annotate, Chat with Documents, AI Insights, Quiz Yourself) — excellent for setting expectations
- "Open document" and "Upload" buttons are prominent
- Document picker has Library + Upload tabs
- Mobile responsive with bottom tab navigation
- Keyboard shortcuts modal is discoverable via "?" key

**Where the user gets confused:**

1. **No connection to their recent ingest.** If the user just ingested a PDF via Quick Ingest, there's no "Open your recently ingested document" prompt. They have to click "Open document" → search the library → find it. The recent documents section in the picker helps but only if they've opened documents before (not on first visit).

2. **Documents ingested via QuickIngest are ALWAYS broken in workspace.** This is the single most damaging FTUE issue. The QuickIngest wizard never sends `keep_original_file=true` (the backend `media_request_models.py:350` defaults to `False`). The document workspace needs the actual PDF/EPUB file, not just extracted text. So ANY document ingested through the standard QuickIngest path will not be openable in the workspace. The upload tab in the workspace picker does set `keep_original_file: true`, but that's a separate path. The error is only shown after attempting to open: "The original file is not available." **Note:** This is NOT a UI toggle the user forgot to check — the option doesn't exist in the QuickIngest UI at all.

3. **Left sidebar tabs (Insights, Figures, TOC, Info, References) are empty with no document** — they show "Open a document to generate insights" but the tab icons give no indication that they need a document first. A user might click through all 5 tabs wondering why everything is blank.

4. **"AI Insights" requires a configured LLM provider.** If the server has no LLM API keys configured, clicking "Generate Insights" will fail. There's no pre-check or guidance about this prerequisite.

5. **Health warnings about annotation/progress storage** appear after the user has already started using the workspace. These should be surfaced during setup or on first visit, not as a surprise mid-session.

6. **Feature discovery cards are dismissible and don't return.** Once dismissed (stored in localStorage), the user can never see them again. No way to re-trigger from settings or help.

#### Persona B (Technical) Walkthrough

**What works well:**
- IndexedDB-based annotations with server sync
- Reading progress auto-save (every 5s) with save-on-close
- Resizable panes with pixel-level control (200-400px left, 240-480px right)
- PDF search integration
- EPUB support alongside PDF

**Where the technical user hits friction:**

1. **No document workspace tutorial in the tutorial system.** There are 15 tutorials for other pages but none for document-workspace. A technical user exploring the tutorial system would notice this gap.

2. **Annotation sync status is opaque.** The SyncStatusIndicator shows sync state but doesn't expose conflict resolution details or sync history. For a technical user working across devices, this matters.

3. **Quiz and Citation tabs lack documentation** on what format/depth they produce. No way to customize quiz difficulty or citation style without trying it.

4. **30-second timeout for document download** is hardcoded. For large documents on slow connections, this may not be enough. No way to configure.

---

### 1.4 Extension-Specific Flows

**Scenario:** User installed the extension and completed setup. Now exploring what it can do.

#### Persona A (Non-technical) Walkthrough

**What works well:**
- Context menu options are descriptive (Summarize, Explain, Rephrase, Translate)
- Sidepanel auto-resumes previous chat or shows companion
- "Send to tldw_server" context menu item for quick page capture
- Toast notifications for ingest progress

**Where the user gets confused:**

1. **13+ context menu items are overwhelming.** Right-clicking shows: Open sidebar/webui, Summarize, Explain, Rephrase, Translate, Custom, Contextual action, Narrate selection, Save to Notes, Save to Companion, Send to tldw_server, Process locally, Transcribe, Transcribe & summarize. A non-technical user doesn't know which to use. No grouping or progressive disclosure.

2. **"Contextual action" vs "Custom" vs "Summarize/Explain/Rephrase"** — the distinction is unclear. What's a "contextual action"? How is "Custom" different from the specific actions?

3. **"Send to tldw_server" vs "Process with tldw_server (don't store)"** — both options use the server; the distinction is whether results are stored, not where processing happens. But the labels make it sound like a local-vs-remote distinction. A non-technical user doesn't understand what "don't store" means in this context.

4. **"Narrate selection"** is a TTS feature but labeled in a way that doesn't communicate that. "Read aloud" would be clearer.

5. **Sidepanel companion vs chat** — when the user opens the sidepanel, they might get either the companion screen or chat screen depending on history. No explanation of what each is or how to switch.

6. **No onboarding tour for extension features.** After setup, the extension just works — but the user doesn't know about context menus, keyboard shortcuts, or the sidepanel. The Quick Ingest modal opens but that's about it.

7. **Save to Notes vs Save to Companion** — what's the difference? Where do saved items go? How do you find them later?

#### Persona B (Technical) Walkthrough

**What works well:**
- Background service worker with model warm-up
- URL normalization for deduplication
- YouTube timestamp extraction from context menu
- Host permission request with clear purpose
- Ingest session management with cancellation

**Where the technical user hits friction:**

1. **No keyboard shortcut for ingest.** The extension relies heavily on context menus and clicks. Power users want Cmd+Shift+S or similar to quickly send a page.

2. **No extension-specific settings page for context menu customization.** Can't disable unwanted context menu items, reorder them, or set defaults. All 13+ items always show.

3. **Background model warm-up is invisible.** No indicator that models are being cached or that the extension is "ready." A technical user would want to see warm-up status.

4. **No debugging/diagnostics view** for extension. Can't see API calls, ingest job status, or error logs from within the extension UI.

5. **OpenAPI drift detection** exists in background but user-facing notification is unclear about what to do when endpoints are missing.

---

## Part 2: Consolidated Issue Table

### Category: Ingest Modal (ING)

| ID | P | Title | Persona | Fix Direction |
|----|---|-------|---------|---------------|
| ING-001 | P1 | Configure step overwhelms non-technical users with jargon (chunking, diarization, OCR) | A | Add plain-language descriptions or hide behind "Advanced" toggle. Default to "Use defaults & process" as primary CTA |
| ING-002 | P2 | Storage configuration (Server vs Local, Review mode) is architecturally exposed | A | Simplify to single toggle "Save to your library" with Review mode as advanced option |
| ING-003 | P1 | "Use defaults & process" button is visually secondary AND disappears for 2+ items | A | Make it primary (filled) button; extend to multi-item queues (not just single items) |
| ING-004 | P1 | Post-completion has no guidance on what user unlocked (searchable in Knowledge, openable in Workspace) | Both | Add contextual CTA cards: "Search in Knowledge QA", "Open in Document Workspace", "Chat about this document" |
| ING-005 | P2 | Inspector drawer is undiscoverable — no visual cue items are clickable | Both | Add subtle "click for details" hint on first item added, or expand inspector automatically for first item |
| ING-006 | P2 | No per-item config override in batch mode — configure step is all-or-nothing | B | Allow per-item preset or type-specific overrides in configure step |
| ING-007 | P2 | Transcription model dropdown has no context on model trade-offs | B | Add speed/accuracy/language hints per model option |
| ING-008 | P2 | Processing errors lack debugging info (no job ID, no server log link) | B | Show job ID in error details, link to server diagnostics endpoint |
| ING-009 | P3 | Error remediation suggestions are generic, don't link to relevant settings | Both | Add deep links to server config for timeout, CORS, provider settings |
| ING-010 | P2 | Large file warning thresholds (50MB in Review step, 500MB upload limit) but no guidance on expected processing times per file type | A | Add estimated time per file type (e.g., "PDFs: ~10s, Videos: ~2min per 10min of content") |
| ING-011 | P0 | QuickIngest wizard never sends `keep_original_file=true` — documents are unopenable in Document Workspace | Both | Either: (a) add `keep_original_file` toggle to configure step (default on for doc types), (b) change batch ingest code path to send `true` for PDF/EPUB, or (c) change server default for document types. `WizardConfigureStep.tsx` has no reference to this param; `quick-ingest-batch.ts` omits it |

### Category: Knowledge Page (KNW)

| ID | P | Title | Persona | Fix Direction |
|----|---|-------|---------|---------------|
| KNW-001 | P3 | ~~Source categories must be explicitly selected~~ [CORRECTED: pre-selected by default]. Issue is naming, not selection | A | Rename categories to plain language when displaying (merged into KNW-010) |
| KNW-002 | P1 | No feedback when ingested content isn't searchable (wrong ingest settings, not yet indexed) | Both | Show indexing status per document, surface "not yet indexed" warning in search results |
| KNW-003 | P2 | Preset descriptions use jargon ("retrieval depth", "rerank") | A | Rewrite as: Fast = "Quick answers, might miss details" / Balanced = "Good coverage, moderate speed" / Deep = "Thorough, includes verification" |
| KNW-004 | P2 | Basic settings still show "Vector Only / Full-Text Only" search modes | A | Hide non-Hybrid modes in Basic mode; only show in Expert |
| KNW-005 | P2 | Web search fallback toggled but "threshold" and provider not explained | Both | Add tooltip: "Uses [provider name] when local confidence < 0.6" (or whatever the threshold is) |
| KNW-006 | P2 | Faithfulness indicators (Strong/Partial/Weak) have no tooltip or explanation | A | Add info icon with "Faithfulness measures how well the answer is supported by the cited sources" |
| KNW-007 | P2 | No visibility into retrieval pipeline (chunks, scores, reranking applied) | B | Add collapsible "Debug" section below answer showing retrieval details |
| KNW-008 | P3 | Expert settings wall of 150+ options with no "commonly tuned" grouping | B | Add "Most impactful" section at top of Expert settings |
| KNW-009 | P3 | A/B comparison mode behind feature flag, not discoverable | B | Graduate comparison mode or add mention in Expert mode settings |
| KNW-010 | P1 | "Open source settings" link opens drawer but categories (Notes, Character Cards, Kanban) confuse non-technical users | A | Rename categories to plain language: "Your Documents", "Your Notes", "Character Profiles", "Chat Logs", "Boards" |

### Category: Document Workspace (DOC)

| ID | P | Title | Persona | Fix Direction |
|----|---|-------|---------|---------------|
| DOC-001 | P1 | No connection to recently ingested documents — user must manually search library | Both | Show "Recently ingested" section at top of document picker, or prompt "Open [filename] you just ingested?" |
| DOC-002 | P0 | QuickIngest never sends `keep_original_file=true` — ALL standard-ingested documents are broken in workspace. No UI toggle exists. | Both | Root cause is ING-011. Workspace should also detect and explain: "This document's original file was not preserved during ingest. Re-upload from the Upload tab to enable workspace features." |
| DOC-003 | P2 | Left sidebar tabs show generic "Open a document" message — 5 tabs all saying the same thing | A | Consolidate empty state to single message in center pane only; disable sidebar tabs when no document |
| DOC-004 | P2 | AI Insights requires LLM provider but no pre-check or guidance | A | Check provider availability on page load; show "Configure an AI provider in Settings to use AI features" |
| DOC-005 | P2 | Health warnings (annotation/progress storage) appear mid-session as surprise | Both | Run health check on first visit, show banner at top before user starts working |
| DOC-006 | P3 | Feature discovery cards are permanently dismissible — no way to re-access | A | Add "Show feature tips" toggle in workspace settings or help menu |
| DOC-007 | P2 | No document-workspace tutorial in the 18-tutorial system | Both | Create document-workspace-basics tutorial (5-6 steps: open doc, annotate, chat, insights, quiz) |
| DOC-008 | P3 | Annotation sync status doesn't expose conflict resolution details | B | Add sync history view accessible from status indicator |
| DOC-009 | P3 | Quiz and Citation tabs lack format/depth documentation | Both | Add brief "About" section in each tab explaining output format |
| DOC-010 | P3 | 30-second document download timeout is hardcoded | B | Make configurable in settings, or at minimum show progress bar during download |
| DOC-011 | P2 | "Show non-document media" toggle in picker defaults OFF — HTML documents and other ingested non-PDF/EPUB content are hidden without explanation | A | Make toggle more prominent or explain why some ingested items don't appear: "Only PDF and EPUB files can be viewed here. Other formats are available in the Media page." |

### Category: Extension (EXT)

| ID | P | Title | Persona | Fix Direction |
|----|---|-------|---------|---------------|
| EXT-001 | P1 | 13+ context menu items are overwhelming with no grouping | Both | Group into submenus: "AI Actions" (Summarize/Explain/Rephrase/Translate), "Save" (Notes/Companion), "Ingest" (Send/Process/Transcribe). Or allow user to configure which items show |
| EXT-002 | P1 | "Contextual action" vs "Custom" vs specific actions — unclear distinction | A | Rename "Contextual action" to "AI popup" or merge with Custom. Add tooltip on first use |
| EXT-003 | P1 | "Send to tldw_server" vs "Process with tldw_server (don't store)" — both use server, distinction is storage not processing location | A | Rename to "Save to library" vs "Analyze without saving" (or similar) |
| EXT-004 | P2 | "Narrate selection" should be "Read aloud" | A | Rename to "Read aloud" with speaker icon |
| EXT-005 | P2 | Sidepanel companion vs chat — no explanation of difference or how to switch | A | Add header toggle with tooltip: "Chat: AI conversation / Companion: Quick actions & status" |
| EXT-006 | P1 | No onboarding tour for extension features after setup | Both | Add 3-step extension tutorial: "Right-click to use AI actions, open sidepanel for chat, send pages to your library" |
| EXT-007 | P2 | "Save to Notes" vs "Save to Companion" — unclear where items go and how to find them | A | Add destination hint: "Save to Notes (opens in /notes)" and "Save to Companion (opens in sidebar)" |
| EXT-008 | P3 | No keyboard shortcut for quick page ingest | B | Add configurable shortcut (default: Cmd+Shift+S) for "Send to tldw_server" |
| EXT-009 | P3 | Can't customize which context menu items appear | B | Add settings page for context menu configuration (enable/disable/reorder) |
| EXT-010 | P3 | No extension diagnostics view (API calls, job status, errors) | B | Add "Debug" tab in extension options page showing recent API calls and errors |
| EXT-011 | P2 | Background model warm-up has no user-visible status indicator | B | Add subtle indicator in sidepanel header: green dot = ready, yellow = warming |

### Category: Cross-Cutting (XC)

| ID | P | Title | Persona | Fix Direction |
|----|---|-------|---------|---------------|
| XC-001 | P0 | No guided path from ingest → knowledge → workspace — each page is an island | Both | After ingest completion, show clear next-step CTAs. On knowledge page, link to "Open in Workspace" per result. On workspace, link to "Search in Knowledge" |
| XC-002 | P1 | "Knowledge" vs "Document Workspace" naming confusion — what's the difference? | A | Add brief explanation in sidebar tooltip or on each page: "Knowledge: search across all your content" / "Workspace: deep-read a single document" |
| XC-003 | P1 | Tutorial system has no sequenced first-run flow connecting ingest → knowledge → workspace | Both | Create "Getting Started" meta-tutorial that chains: "1. Add content (ingest) → 2. Search it (knowledge) → 3. Study it (workspace)" |
| XC-004 | P2 | LLM provider not configured → multiple features silently fail (AI Insights, Chat, Knowledge answer generation) | Both | Add global "AI features require a configured provider" banner that appears once on any page needing LLM, links to provider settings |
| XC-005 | P2 | Embedding provider not configured → knowledge search returns no results with no explanation | Both | Check embedding config on knowledge page load; show "Configure embeddings in Settings to enable search" |
| XC-006 | P2 | Extension and web UI share code but extension context menu items can't be discovered from web UI | A | In web UI sidebar, add "Browser Extension" link with feature overview |

---

## Part 3: Priority Summary

### P0 (Blocks first value — fix immediately)
- **ING-011 / DOC-002**: QuickIngest never sends `keep_original_file=true` — Document Workspace is broken for ALL ingested documents
- **XC-001**: No guided path between ingest → knowledge → workspace (but CTA to workspace is useless until ING-011 is fixed)

### P1 (30+ min friction — fix in 1-2 sprints)
- **ING-001**: Configure step overwhelms non-technical users
- **ING-003**: "Use defaults & process" is visually secondary AND absent for multi-item queues
- **ING-004**: Post-completion has no guidance on what was unlocked
- **KNW-002**: No feedback when content isn't searchable
- **KNW-010**: Source category names confuse non-technical users
- **DOC-001**: No connection to recently ingested documents
- **EXT-001**: 13+ context menu items with no grouping
- **EXT-002**: "Contextual action" vs "Custom" distinction unclear
- **EXT-003**: Architecture jargon in context menu labels
- **EXT-006**: No onboarding tour for extension features (overlaps v1 Improvement 12)
- **XC-003**: No sequenced first-run tutorial (overlaps v1 FE-004/Improvement 11 — merge, don't duplicate)

### P2 (Suboptimal but user can succeed — fix in 1-2 months)
- ING-002, ING-005, ING-006, ING-007, ING-008, ING-010
- KNW-003, KNW-004, KNW-005, KNW-006, KNW-007
- DOC-003, DOC-004, DOC-005, DOC-007, DOC-011
- EXT-004, EXT-005, EXT-007, EXT-011
- XC-002, XC-004, XC-005, XC-006

### P3 (Polish — backlog)
- ING-009
- KNW-001 (corrected: sources pre-selected, naming merged into KNW-010), KNW-008, KNW-009
- DOC-006, DOC-008, DOC-009, DOC-010
- EXT-008, EXT-009, EXT-010

---

## Part 4: Recommended Implementation Order

### Sprint 1: "Fix the broken pipeline + connect the islands" (ING-011 + DOC-002 + XC-001 + ING-004 + DOC-001)
The single highest-impact change: make documents ingested via QuickIngest actually openable in Document Workspace, THEN connect the pages with CTAs. These MUST be bundled — adding a "Open in Workspace" CTA without fixing `keep_original_file` would route users into a broken experience.

**Changes:**
1. **Fix `keep_original_file` in QuickIngest pipeline** — Either: (a) have `quick-ingest-batch.ts` send `keep_original_file: true` for document types (PDF, EPUB, DOCX), or (b) add it as a visible toggle in `WizardConfigureStep.tsx` defaulting to on, or (c) change server default in `media_request_models.py` for document MIME types
2. **Workspace: better error for documents without original file** — In `DocumentPickerModal.tsx`, detect documents where original file is unavailable and show "This document's original file was not preserved. Re-upload from the Upload tab or re-ingest with file storage enabled."
3. **Ingest Results step: Add next-step CTAs** — "Search in Knowledge QA" and "Open in Document Workspace" alongside existing Open/Chat buttons in `WizardResultsStep.tsx`
4. **Document Picker: "Recently ingested" section** — Show items from last hour at top of Library tab. Requires either polling media API with time filter or a shared event store between ingest and workspace.
5. **Knowledge source cards: "Open in Workspace"** — Per source card, add button for PDF/EPUB sources

**Implementation risk:** The "Recently ingested" section in the picker (change #4) requires a new data source. Current picker reads from localStorage `document-workspace-recent` which only tracks previously-opened documents, not recently ingested ones. May need to add a time-filtered media list API call or a cross-component event bus.

### Sprint 2: "Simplify first use" (ING-001 + ING-003)
Make the default path require zero configuration knowledge.

**Changes:**
1. **Ingest: Swap button priority** — "Use defaults & process" becomes primary (filled) button; "Configure" becomes secondary (outlined). Also extend "Use defaults & process" to work for multi-item queues (currently hidden when >1 item).
2. **Ingest Configure: Progressive disclosure** — Group options into "Common" (Analysis, Storage — visible) and "Advanced" (Chunking, Diarization, OCR, Review mode — collapsed). Add plain-language labels: "Chunking" → "Split into searchable sections", "Diarization" → "Identify different speakers".

**Implementation risk:** Changing button visibility (`queueItems.length <= 1` gate) and type (`type="primary"`) may break E2E test selectors. Check `QuickIngestWizardModal.integration.test.tsx`.

**Backward compatibility:** Users who've memorized the current configure layout will see different organization. Consider feature flag for staged rollout.

### Sprint 3: "Extension clarity" (EXT-001 + EXT-003 + EXT-006)
Make the extension approachable for non-technical users.

**Changes:**
1. **Group context menu into submenus** — "AI Actions" (Summarize/Explain/Rephrase/Translate/Custom), "Save" (Notes/Companion), "Process" (Send to library/Analyze without saving/Transcribe). Note: Chrome MV3 supports one level of sub-menus; verify WXT framework compatibility and Firefox MV2 support.
2. **Rename jargon labels** — "Send to tldw_server" → "Save to library", "Process with tldw_server (don't store)" → "Analyze without saving", "Narrate selection" → "Read aloud", "Contextual action" → "AI popup" (or merge with Custom)
3. **Add extension onboarding** — Merge with v1 audit Improvement 12 (already planned). 3-step tutorial after setup: "Right-click to use AI actions → Open sidepanel for chat → Send pages to your library"

**Implementation risk:** Renaming context menu labels is a breaking UX change for existing users. No migration path — users who learned current labels will be confused. Consider showing both old and new labels briefly, or adding a changelog notification.

### Sprint 4: "Missing tutorials & prereq checks" (DOC-007 + XC-003 + XC-004 + KNW-002)
Fill remaining gaps in guidance and prerequisite detection.

**Changes:**
1. **Create document-workspace tutorial** — 5-6 steps: open doc, navigate pages, annotate text, chat about document, generate insights, take quiz
2. **Create sequenced "Getting Started" meta-tutorial** — Merge with v1 FE-004/Improvement 11 (already planned). Chain: "Add content (ingest) → Search it (knowledge) → Study it (workspace)". Must coordinate with existing getting-started tutorial on home page.
3. **Add global LLM/embedding provider pre-checks** — On pages needing LLM (workspace insights, knowledge answer gen, chat), check provider availability and show "Configure an AI provider in Settings → Providers to enable AI features" banner. Overlaps v1 ERR-004/Improvement 10 — merge.
4. **Add indexing status feedback** — On Knowledge page, when a search returns no results, check if recently-ingested documents exist that may not be indexed yet. Show "Your recently added documents may still be indexing. Try again in a moment."

**Implementation risk:** Global provider pre-checks add latency to page loads if they require an API call. Consider caching the check result in sessionStorage with a TTL.

---

## Part 5: Additional Perspectives (from self-review)

### Mobile Users
- Document Workspace has responsive bottom-tab navigation — good
- Context menus don't exist on mobile browsers — extension is desktop-only
- Knowledge page search bar and settings panel need mobile verification
- Ingest modal on small screens may have overflow issues with the configure step

### Multi-User/Team Scenarios
- Second user joining existing instance sees different onboarding state
- Shared documents may already exist but source selections are per-user
- Not addressed in this review — recommend separate audit

### Extension-First Users
- Users who discover tldw through the extension (not web UI) have a reverse flow: install extension → setup → context menus → eventually web UI
- Extension setup asks for server URL but doesn't explain that a server must be running
- Already partially addressed by v1 Improvement 12 (extension-specific onboarding)

### Users Upgrading with Existing Content
- Existing documents ingested before ING-011 fix will not have original files stored
- Need migration strategy: either re-ingest or batch `keep_original_file` backfill
- Workspace should clearly explain the issue for legacy content

---

## Verification Plan

For each sprint:
1. **Persona A test:** Fresh browser (cleared localStorage), completed onboarding, no prior content. Walk through: ingest a PDF → search in knowledge → open in workspace. Verify the document actually opens in workspace (critical for Sprint 1). Time each step, note confusion points.
2. **Persona B test:** Same fresh state but also: batch ingest 5 mixed items (PDF + video + URL), use Expert RAG settings, trigger error cases (bad URL, large file, missing provider). Verify job IDs visible in errors.
3. **Extension test:** Fresh extension install, complete setup, right-click on a web page, verify context menu grouping (Sprint 3), use sidepanel, ingest from context menu, verify document appears in workspace.
4. **Regression:** Verify existing tutorials still trigger, empty states still display, error messages unchanged where not modified. Run existing E2E tests.
5. **Upgrade test (Sprint 1):** With existing ingested documents (pre-fix), verify workspace shows clear explanation for documents without original files, not just "file not available."

---

## Part 6: Issues Found in Second-Pass Code Verification

These issues were identified during code-level verification and are NOT in the original tables. Add to backlog.

### Internationalization (I18N)

| ID | P | Title | Fix Direction |
|----|---|-------|---------------|
| I18N-001 | P2 | 50+ hardcoded English strings in KnowledgeQA bypass i18n: `KnowledgeReadyState.tsx` (heading, subheading, "How it works" steps, warning), `SearchBar.tsx` (7 example queries, suggestion labels "History"/"Source"/"Example"), `AnswerPanel.tsx` (error classification), `errorMessages.ts` (12+ error strings), `KnowledgeQAProvider.tsx:50` ("Helpful AI Assistant") | Move all to locale files |
| I18N-002 | P2 | QuickIngest hardcoded strings: `FileDropZone.tsx` (size/type errors), `ProcessingStep.tsx` ("~Xs remaining"), `ReviewStep.tsx` (operation names "Transcribe", "OCR", "Extract") | Move to locale files |
| I18N-003 | P3 | DocumentWorkspace `"Loading..."` fallback at `DocumentWorkspacePage.tsx:99` not localized | Wrap in `t()` |

### Loading & Navigation (NAV)

| ID | P | Title | Fix Direction |
|----|---|-------|---------------|
| NAV-001 | P2 | DocumentWorkspace sidebar tabs show empty `<div>` during lazy load (`tabPanelFallback` at line 96) — no skeleton/spinner | Replace with skeleton loader |
| NAV-002 | P2 | Knowledge QA loses search state on browser back — no URL/sessionStorage preservation of query or filters | Persist in URL params or sessionStorage |
| NAV-003 | P2 | QuickIngest wizard loses ALL state on browser back during processing — no `beforeunload` warning | Add unload handler during active processing |
| NAV-004 | P3 | Knowledge QA route hydration has 1.5s retry delay (`ROUTE_HYDRATION_RETRY_DELAY_MS`) when restoring thread from URL — may appear broken | Add loading indicator during thread restoration |

### Accessibility (A11Y)

| ID | P | Title | Fix Direction |
|----|---|-------|---------------|
| A11Y-001 | P2 | No `aria-live` regions for dynamic content: Knowledge QA answers, QuickIngest processing progress, file upload status | Add `aria-live="polite"` to dynamic areas |
| A11Y-002 | P2 | Knowledge QA suggestion dropdown lacks focus trap — keyboard users can tab out without closing | Add focus management to `SearchBar.tsx` suggestion popover |
| A11Y-003 | P3 | Suggested prompt buttons in `KnowledgeReadyState.tsx` lack `aria-label` attributes | Add descriptive labels |

### Offline/Reconnection (NET)

| ID | P | Title | Fix Direction |
|----|---|-------|---------------|
| NET-001 | P2 | QuickIngest SSE has no explicit mid-ingest connection drop handling in `useIngestSSE.ts` — opaque status | Detect loss, show "Connection lost — ingest may still be running. Reconnecting..." |
| NET-002 | P3 | Knowledge QA doesn't explain reconnection timing when server goes offline mid-search | Add "Retrying in Xs..." to offline state |

### Updated Totals

**60 issues total** (50 original + 10 from code verification):
- P0: 3 (ING-011, DOC-002, XC-001)
- P1: 11
- P2: 30 (was 20, added I18N-001, I18N-002, NAV-001, NAV-002, NAV-003, A11Y-001, A11Y-002, NET-001, plus original XC-002 moved to P2)
- P3: 16 (was 13, added I18N-003, NAV-004, A11Y-003, NET-002)

The I18N issues are the most impactful new finding — 50+ hardcoded strings mean the FTUE is untranslatable. However, the project appears English-only currently, so P2 is appropriate.

NAV-001 through NAV-003 directly affect first-time user perception — empty divs and lost state create a "broken" impression. Bundle with Sprint 2 if possible.

---

## Relationship to Existing Audits

| This Review | Existing Audit | Action |
|-------------|---------------|--------|
| XC-003 (sequenced tutorial) | v1 FE-004 / Improvement 11 | Merge — extend existing plan scope to include workspace |
| EXT-006 (extension onboarding) | v1 Improvement 12 | Merge — this review adds specifics (3-step tutorial) |
| XC-004 (provider pre-checks) | v1 ERR-004 / Improvement 10 | Merge — this review adds page-specific banners |
| ING-008, ING-009 (error remediation) | v1 ERR-004 / Improvement 10 | Subset — already planned in v1 |
| ING-011 / DOC-002 (`keep_original_file`) | Not in v1/v2 | NEW — critical finding not in any existing audit |
| EXT-001-003 (context menu) | Not in v1/v2 | NEW — extension in-app UX not covered by existing audits |
| KNW-001 correction | v1 assumed it was broken | CORRECTED — sources are pre-selected, not broken |
