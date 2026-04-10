# UX Audit: /document-workspace Page

## Context

This is a comprehensive UX audit of the `/document-workspace` (Document Workspace) page, evaluated against Nielsen Norman Group's 10 usability heuristics from two persona perspectives:

- **Non-technical researcher / student** (competent user): Comfortable with Google Docs, Zotero, Mendeley, and typical PDF readers. Reads academic papers and annotates them. Does not understand RAG, embeddings, ingestion pipelines, or database migrations.
- **Technical power user**: Developer or sysadmin. Comfortable with keyboard shortcuts, config files, multi-document workflows, and technical terminology.

**Deliverable**: Heuristic scorecard with file:line references, persona journey maps, priority fix list, and implementation mapping.

**Scoring scale**: 1=Critical failures throughout, 2=Major usability issues, 3=Moderate issues (workable but painful), 4=Minor issues only, 5=Exemplary.

**Prior work**: This builds on `Docs/Plans/2026-04-08-knowledge-workspace-nng-ux-audit.md` (Knowledge QA and Workspace Playground audit). That audit covered `/knowledge` and `/workspace-playground` with NNG heuristics. This document applies the same methodology to the `/document-workspace` page, which serves a distinct purpose: reading, annotating, and chatting about individual documents.

**Caveat**: Line numbers were verified at time of writing but may drift with future commits. Use surrounding code context to locate if line numbers are stale.

---

## Part 1: NNG Heuristic Scorecard

### H1: Visibility of System Status -- Score: 3.5/5

**Strengths:**
- SyncStatusIndicator provides a visual dot indicating sync state (synced, syncing, error)
- Document viewer shows a loading overlay while the PDF renders
- Chat streaming indicator shows when the AI is generating a response
- Tab bar shows which documents are open and which is active

**Issues:**

| # | Issue | Sev | Personas | File:Line |
|---|-------|-----|----------|-----------|
| D1 | Sync status indicator shows "Syncing..." or "Synced" but never explains WHAT is being synced -- annotations? reading position? settings? User has no idea what data is at risk if they close the page during sync. | Med | Researcher | `SyncStatusIndicator.tsx:48-50` |
| D2 | Two independent loading indicators fire for the same document-open operation: an alert/banner in DocumentWorkspacePage AND an overlay spinner inside the DocumentViewer. User sees duplicate "loading" signals with no indication they represent the same operation. | Med | Researcher, Power User | `DocumentWorkspacePage.tsx:763-779` vs `DocumentViewer/index.tsx:396-403` |
| D3 | No download progress indicator for large documents. User clicks to open a 50MB PDF and sees only a generic spinner with no file size, percentage, or ETA. On slow connections this looks like a hang. | High | Researcher | `DocumentWorkspacePage.tsx:387,763-779` |

---

### H2: Match Between System and Real World -- Score: 2/5

**Strengths:**
- Document viewer uses familiar PDF-reader conventions (page navigation, zoom, scroll)
- Annotation highlighting uses standard color-picker and text-selection patterns
- Tab bar for multiple open documents follows browser-tab conventions

**Issues:**

| # | Issue | Sev | Personas | File:Line |
|---|-------|-----|----------|-----------|
| D4 | Feature discovery card says "RAG-powered answers" -- "RAG" is an AI/ML acronym that means nothing to a researcher reading a PDF. Should say "answers grounded in your document" or "cited answers from this document". | High | Researcher | `DocumentViewer/index.tsx:258` |
| D5 | Chat sidebar has a "Use document sources (RAG)" toggle. The word "RAG" appears as a user-facing label. Should be "Ground answers in this document" or "Use document as reference". | High | Researcher | `DocumentChat.tsx:377-379` |
| D6 | When the RAG toggle is off, the chat shows "RAG is off. Enable to ground answers in your document." -- mixes jargon ("RAG") with plain language ("ground answers") in the same sentence, creating confusion about whether these are the same thing. | Med | Researcher | `DocumentChat.tsx:456-458` |
| D7 | Document picker modal says documents are "ingested into your media library". "Ingested" is pipeline jargon. "Added to your library" or "imported" would be universally understood. | Med | Researcher | `DocumentPickerModal.tsx:509-510` |
| D8 | "Figures" tab in sidebar shows page thumbnails, not extracted figures/charts/diagrams. A researcher clicking "Figures" expects to see the charts and graphs from their paper, not miniature page previews. This is a significant expectation mismatch. | High | Researcher | `DocumentWorkspacePage.tsx:155-157`, `FiguresTab.tsx:11,87-88` |
| D9 | "Server connection required" message appears on Quick Insights tab with no explanation of what the server does, why it's needed, or how to connect. A researcher doesn't know what "server" means in this context. | Med | Researcher | `QuickInsightsTab.tsx:149-154` |
| D10 | "Recently ingested" label in the document picker. Same jargon issue as D7 -- should be "Recently added" or "Recently imported". | Low | Researcher | `DocumentPickerModal.tsx:342` |
| D11 | Error message says content is "not preserved during ingest" when a document fails to process. "Ingest" is pipeline terminology. "The document could not be fully processed" is clearer. | Med | Researcher | `DocumentWorkspacePage.tsx:649-654` |
| D12 | "Select a model in settings to chat" message appears when no LLM is configured, with no link to settings and no explanation of what a "model" is in this context. A researcher wants to ask questions about their PDF -- they don't know they need to configure an AI model first. | High | Researcher | `DocumentChat.tsx:384-389` |
| D13 | Health check warning tells users to "run the latest migrations to create missing tables" -- this is developer/DBA language that will alarm a non-technical user. Should say "The server needs to be updated. Contact your administrator." or auto-handle the migration. | High | Researcher | `DocumentWorkspacePage.tsx:817-819` |

---

### H3: User Control and Freedom -- Score: 3/5

**Strengths:**
- Users can close document tabs to dismiss them
- Annotation colors can be changed after creation
- Chat history is preserved per document session

**Issues:**

| # | Issue | Sev | Personas | File:Line |
|---|-------|-----|----------|-----------|
| D14 | Feature discovery cards in the document viewer can be permanently dismissed. Once dismissed, there is no way to see them again. New features or tips become permanently invisible. | Med | Researcher | `DocumentViewer/index.tsx:238-246` |
| D15 | Closing a document tab does not confirm whether the user has pending/unsaved annotations. User can lose work by accidentally closing a tab with unsaved highlights and notes. | High | Researcher | `DocumentTabBar.tsx:121-124` |
| D16 | No undo for closing a document tab. In browsers, Cmd+Shift+T reopens a closed tab. Document workspace has no equivalent -- once closed, the user must re-open from the picker and navigate back to their reading position. | Med | Researcher, Power User | `DocumentWorkspacePage.tsx:546-556` |

---

### H4: Consistency and Standards -- Score: 3/5

**Strengths:**
- PDF viewer controls (zoom, page navigation) follow standard PDF reader conventions
- Tab bar behavior matches browser tab conventions (click to switch, X to close)
- Annotation tools follow standard patterns (select text, choose color, add note)

**Issues:**

| # | Issue | Sev | Personas | File:Line |
|---|-------|-----|----------|-----------|
| D17 | Sidebar tab is labeled "Notes" but renders annotation content (highlights with attached comments). "Notes" implies freeform note-taking; "Annotations" is the correct term for highlights-with-comments on a document. Or the tab should contain freeform notes and annotations should have their own tab. | Med | Researcher | `DocumentWorkspacePage.tsx:239` vs `AnnotationsPanel.tsx:564` |
| D18 | Feature discovery cards use marketing-style copy ("Unlock powerful insights!", "AI-powered analysis") that clashes with the functional, tool-like tone of the rest of the UI (buttons say "Export", "Search", "Zoom"). The tonal inconsistency feels like two different products. | Low | Researcher | `DocumentViewer/index.tsx:249-269` |
| D19 | Keyboard shortcuts modal title is hardcoded in English. If the rest of the UI is ever localized, this modal will remain in English, breaking consistency. | Low | Power User | `DocumentShortcutsModal.tsx:195-196` |
| D20 | Desktop layout uses a resizable multi-pane layout (viewer + sidebar). Mobile layout uses a fundamentally different paradigm with bottom navigation tabs. Users switching between devices encounter different interaction patterns for the same feature -- sidebar panels become full-screen tabs, toolbar controls relocate. | Med | Researcher | `DocumentWorkspacePage.tsx:828-907,910-1026` |

---

### H5: Error Prevention -- Score: 3.5/5

**Strengths:**
- Document format validation prevents opening unsupported file types
- Chat input validation prevents submitting empty messages
- Annotation save operations are debounced to prevent rapid duplicate writes

**Issues:**

| # | Issue | Sev | Personas | File:Line |
|---|-------|-----|----------|-----------|
| D21 | No close confirmation when sync is pending (same underlying issue as D15 but from error prevention angle). User can close a tab or navigate away while SyncStatusIndicator shows "Syncing..." and silently lose data. | High | Researcher | `DocumentTabBar.tsx:121-124` |
| D22 | Chat input is not disabled when the server is offline or disconnected. User can type and submit a message that will fail, producing a confusing error instead of preventing the action upfront. | Med | Researcher | `DocumentChat.tsx:403` |
| D23 | "No matching media found" in the document picker does not distinguish between an empty library (user has never added documents) and a search with no results (user has documents but none match the query). The empty library case needs onboarding guidance; the no-results case needs search suggestions. | Med | Researcher | `DocumentPickerModal.tsx:393-399` |

---

### H6: Recognition Rather Than Recall -- Score: 3/5

**Strengths:**
- Recently opened documents appear in the picker for quick re-access
- Annotation colors are shown visually (not just named)
- Document tabs show truncated filenames for recognition

**Issues:**

| # | Issue | Sev | Personas | File:Line |
|---|-------|-----|----------|-----------|
| D24 | Keyboard shortcuts are only accessible via pressing the "?" key. There is no visible button, menu item, or hint anywhere in the UI indicating that keyboard shortcuts exist or how to access them. Users must already know the convention to discover it. | Med | Researcher, Power User | `DocumentWorkspacePage.tsx:354` |
| D25 | Text-to-speech (TTS) functionality is hidden behind a small icon with no text label. Users looking for "read aloud" functionality will not recognize the icon. Standard PDF readers label this "Read Aloud" with accompanying text. | Med | Researcher | `ViewerToolbar.tsx` (TTS button area) |
| D26 | Text highlighting assumes users know to select text first and then choose a color/action. There is no visible hint, tooltip, or onboarding prompt explaining the highlight workflow. First-time users stare at the toolbar looking for a "highlight" button that doesn't exist. | Med | Researcher | `AnnotationsPanel.tsx:599-601` |
| D27 | Export annotations is buried inside a dropdown menu. Users who want to export their highlights and notes to share with colleagues must discover the dropdown, recognize the export option, and understand the available formats. No visible "Export" button exists at the top level. | Low | Researcher, Power User | `AnnotationsPanel.tsx:566-574` |
| D28 | Quick Insights "detail level" selector (e.g., Brief / Standard / Detailed) has no explanation of what each level produces. Users must trial-and-error to understand the difference between levels. | Med | Researcher | `QuickInsightsTab.tsx:24-40` |
| D29 | Multi-document tab bar feature is invisible until a second document is opened. First-time users see a single document with no indication that the workspace supports multiple simultaneous documents. The capability is entirely hidden until organically discovered. | Low | Researcher | `DocumentTabBar.tsx:152-155` |

---

### H7: Flexibility and Efficiency of Use -- Score: 4/5

**Strengths:**
- Keyboard shortcuts exist for common actions (navigation, zoom, panel toggles)
- Panel resizing allows users to customize their layout
- Chat supports both quick questions and extended conversations
- TTS provides hands-free document consumption

**Issues:**

| # | Issue | Sev | Personas | File:Line |
|---|-------|-----|----------|-----------|
| D30 | No document-level bookmarks. Users reading long documents (100+ page PDFs) cannot bookmark specific pages to return to later. Must rely on annotations as pseudo-bookmarks, which conflates two different use cases. | Med | Researcher, Power User | (feature gap) |
| D31 | No batch annotation export beyond JSON format. Researchers who want to export highlights to Markdown, CSV, or BibTeX for use in other tools are limited to a single export format. | Low | Power User | (feature gap) |

---

### H8: Aesthetic and Minimalist Design -- Score: 2.5/5

**Strengths:**
- Document viewer itself is clean and focused on the content
- Mobile layout simplifies to tabbed navigation reducing simultaneous visual complexity
- Collapsed sidebar panels reduce noise when not needed

**Issues:**

| # | Issue | Sev | Personas | File:Line |
|---|-------|-----|----------|-----------|
| D32 | Nine sidebar tabs are visible simultaneously: Chat, Notes/Annotations, Figures, Quick Insights, References, Search, TTS, Bookmarks, Settings. This creates significant cognitive overload -- the user must scan and evaluate 9 options to find what they want. Standard PDF readers have 3-4 sidebar options. | High | Researcher | `DocumentWorkspacePage.tsx:148-174,231-252` |
| D33 | Viewer toolbar shows all controls simultaneously regardless of context: zoom, page navigation, rotation, fit modes, search, TTS, fullscreen, print, download. Many of these are rarely used but consume visual space constantly. Progressive disclosure (showing core controls with a "more" overflow) would reduce noise. | Med | Researcher | `ViewerToolbar.tsx:124-261` |
| D34 | Sidebar tab labels render at 11px font size. At this size, labels like "Quick Insights" and "Annotations" become difficult to read, especially on lower-resolution displays or for users with moderate vision impairment. Below WCAG recommended minimum for interactive element labels. | Med | Researcher | `TabIconLabel.tsx:16` |
| D35 | Mobile bottom navigation labels render at 10px font size. Even smaller than the desktop sidebar labels, creating readability issues on the platform where touch targets already need to be larger. | Med | Researcher | `DocumentWorkspacePage.tsx:901` |

---

### H9: Help Users Recognize, Diagnose, and Recover from Errors -- Score: 3/5

**Strengths:**
- Document load failures show an error state with a retry option
- Chat errors display inline with the conversation rather than in a disconnected modal
- Sync errors are surfaced in the SyncStatusIndicator

**Issues:**

| # | Issue | Sev | Personas | File:Line |
|---|-------|-----|----------|-----------|
| D36 | Health check warning tells users to "run the latest migrations to create missing tables" -- pure developer language. A non-technical user encountering this will be alarmed and unable to act. Should say "Something needs to be updated on the server. Please contact your administrator or check the setup guide." with a link to documentation. | High | Researcher | `DocumentWorkspacePage.tsx:817-819` |
| D37 | "Server connection required" appears on Quick Insights and References tabs with zero guidance on what to do. No "retry" button, no "check connection" link, no explanation of what the server provides. User is stuck with an opaque message and no path forward. | Med | Researcher | `QuickInsightsTab.tsx:149-154`, `ReferencesTab.tsx:342-355` |
| D38 | "Select a model in settings to chat" provides no link to settings. User must independently discover where "settings" lives (is it the gear icon? the sidebar? the main menu?). Adding a clickable link or button that navigates directly to the relevant setting would eliminate this friction. | Med | Researcher | `DocumentChat.tsx:384-389` |
| D39 | Server error messages in chat expose internal terminology: "RAG search failed" and references to "tldw server". Users should see "Could not search the document. Please check your connection and try again." -- not implementation details. | Med | Researcher | `DocumentChat.tsx:213-216` |

---

### H10: Help and Documentation -- Score: 2/5

**Strengths:**
- Feature discovery cards provide some initial guidance (when not dismissed)
- Keyboard shortcuts modal exists and is comprehensive
- Empty states provide minimal directional text

**Issues:**

| # | Issue | Sev | Personas | File:Line |
|---|-------|-----|----------|-----------|
| D40 | Feature discovery cards are permanently dismissible with no way to re-access them. First-time tips about chat, annotations, and AI features vanish forever after a single dismiss click. There should be a "Tips" or "What's New" menu item to re-surface these. | Med | Researcher | `DocumentViewer/index.tsx:238-246` |
| D41 | No onboarding tour or guided walkthrough for first-time users. The document workspace has 9 sidebar tabs, a multi-tool toolbar, annotation tools, chat, and TTS -- none of which are introduced to new users. They must explore and discover everything independently. | High | Researcher | (feature gap) |
| D42 | All sidebar panel empty states display a generic "Open a document to view X" message with no additional context. For example, the Chat empty state could explain what kinds of questions work well; the Annotations empty state could explain how to create a highlight; the Quick Insights empty state could explain what insights are generated and how. | Med | Researcher | All sidebar/panel empty states |
| D43 | No in-page help link, documentation link, or "Learn more" affordance anywhere in the document workspace. If a user is confused about any feature, there is no path to help without leaving the application entirely. | Med | Researcher, Power User | (feature gap) |
| D44 | The "?" keyboard shortcut that opens the shortcuts modal is itself undiscoverable. The only way to learn about keyboard shortcuts is to already know the universal "?" convention -- which most non-technical users do not. A small keyboard icon in the toolbar or a "Shortcuts" item in a help menu would solve this. | Med | Researcher | `DocumentShortcutsModal.tsx:234-236` |

---

## Part 2: Heuristic Summary Table

| # | Heuristic | Score | Top Issues | Worst Severity |
|---|-----------|-------|------------|----------------|
| H1 | Visibility of System Status | 3.5/5 | D1 (sync ambiguity), D2 (duplicate loaders), D3 (no download progress) | High |
| H2 | Match Between System and Real World | 2/5 | D4-D6 (RAG jargon), D8 (Figures misnomer), D12-D13 (opaque setup errors) | High |
| H3 | User Control and Freedom | 3/5 | D15 (no close confirmation), D16 (no undo close) | High |
| H4 | Consistency and Standards | 3/5 | D17 (Notes vs Annotations), D20 (desktop/mobile paradigm gap) | Med |
| H5 | Error Prevention | 3.5/5 | D21 (close during sync), D22 (chat while offline) | High |
| H6 | Recognition Rather Than Recall | 3/5 | D24 (shortcuts hidden), D26 (highlight workflow invisible) | Med |
| H7 | Flexibility and Efficiency of Use | 4/5 | D30 (no bookmarks), D31 (limited export formats) | Med |
| H8 | Aesthetic and Minimalist Design | 2.5/5 | D32 (9 simultaneous tabs), D33 (toolbar overload) | High |
| H9 | Help Users Recognize, Diagnose, and Recover from Errors | 3/5 | D36 (migrations language), D37 (no recovery guidance) | High |
| H10 | Help and Documentation | 2/5 | D41 (no onboarding tour), D43 (no help access) | High |

**Overall weighted score: 2.9/5** -- The page is functional for power users who already understand the interface, but presents significant barriers for non-technical researchers encountering it for the first time.

---

## Part 3: Priority Fix List

### P0 -- Critical (fix immediately, blocks core user journey)

| # | Issue | Heuristic | Rationale |
|---|-------|-----------|-----------|
| D13 | "Run the latest migrations" health warning | H2, H9 | Exposes raw developer language to end users. Alarming and unactionable. |
| D12 | "Select a model in settings" with no link | H2, H9 | Blocks the chat feature entirely with no path to resolution for non-technical users. |
| D15/D21 | No close confirmation with pending annotations or sync | H3, H5 | Data loss risk. User can lose annotation work with a single accidental click. |
| D32 | 9 sidebar tabs visible simultaneously | H8 | Cognitive overload is the first impression. Directly drives abandonment. |

### P1 -- High (fix in first sprint, significantly degrades experience)

| # | Issue | Heuristic | Rationale |
|---|-------|-----------|-----------|
| D4 | "RAG-powered answers" jargon in feature card | H2 | First impression jargon in a discovery context. |
| D5 | "Use document sources (RAG)" toggle label | H2 | Recurring jargon on a primary interaction control. |
| D8 | "Figures" tab shows page thumbnails | H2 | Significant expectation mismatch. Rename or fix content. |
| D36 | "Migrations" language in health warning | H9 | Duplicate of D13 from error recovery angle -- both must be fixed together. |
| D41 | No onboarding tour | H10 | 9 tabs, annotation tools, chat, TTS -- zero introduction. |
| D3 | No download progress for large documents | H1 | Looks like a hang on slow connections. |
| D26 | Highlight workflow invisible | H6 | Core annotation feature has no discoverability. |

### P2 -- Medium (fix in second sprint, causes friction but has workarounds)

| # | Issue | Heuristic | Rationale |
|---|-------|-----------|-----------|
| D1 | Sync status doesn't explain what's syncing | H1 | Ambiguity causes anxiety but doesn't block work. |
| D2 | Duplicate loading indicators | H1 | Confusing but not blocking. |
| D6 | "RAG is off" mixed jargon/plain language | H2 | Inconsistent but partially comprehensible. |
| D7 | "Ingested" jargon in picker | H2 | Simple word swap. |
| D9 | "Server connection required" no explanation | H2, H9 | Blocks AI features but document reading still works. |
| D11 | "Not preserved during ingest" error | H2 | Jargon in error path. |
| D14 | Feature cards permanently dismissible | H3, H10 | Lost tips, but features still accessible via UI. |
| D16 | No undo close document | H3 | Friction but user can re-open from picker. |
| D17 | "Notes" tab renders annotations | H4 | Terminology confusion but content is findable. |
| D20 | Desktop/mobile paradigm gap | H4 | Inherent in responsive design; manageable. |
| D22 | Chat input not disabled when offline | H5 | Leads to confusing error but not data loss. |
| D23 | Empty library vs no results indistinguishable | H5 | Missed onboarding opportunity. |
| D24 | Shortcuts only via "?" key | H6 | Hidden accelerator. |
| D25 | TTS hidden behind icon-only button | H6 | Discoverable by exploration. |
| D28 | Detail level selector unexplained | H6 | Trial-and-error works but wastes time. |
| D33 | Toolbar shows all controls | H8 | Visual noise but all controls accessible. |
| D34 | Tab labels at 11px | H8 | Readability concern, not blocking. |
| D35 | Mobile nav labels at 10px | H8 | Same as D34, mobile variant. |
| D37 | "Server connection required" no guidance | H9 | Blocks AI features, not document reading. |
| D38 | "Select a model" no link to settings | H9 | Related to D12; link would resolve both. |
| D39 | Server error exposes "RAG search" and "tldw server" | H9 | Jargon in error path. |
| D40 | Feature cards permanently dismissible (H10 angle) | H10 | Same as D14 from documentation perspective. |
| D42 | Generic empty states | H10 | Missed teaching opportunity. |
| D43 | No in-page help link | H10 | No path to help. |
| D44 | "?" shortcut undiscoverable | H10 | Chicken-and-egg discoverability problem. |

### P3 -- Low (fix when convenient, minor polish)

| # | Issue | Heuristic | Rationale |
|---|-------|-----------|-----------|
| D10 | "Recently ingested" label | H2 | Simple word swap, low impact. |
| D18 | Marketing tone in feature cards | H4 | Tonal inconsistency, not blocking. |
| D19 | Hardcoded English in shortcuts modal | H4 | Only matters if localization is pursued. |
| D27 | Export annotations buried in dropdown | H6 | Discoverable, just not prominent. |
| D29 | Multi-doc feature invisible until 2nd doc | H6 | Natural progressive disclosure, arguably correct. |
| D30 | No document-level bookmarks | H7 | Feature gap, annotations serve as workaround. |
| D31 | No batch annotation export beyond JSON | H7 | Feature gap, JSON export exists. |

---

## Part 4: Persona Journey Maps

### Journey A: Non-Technical Researcher -- "Read, annotate, and ask questions about a PDF"

**Task**: Open a research paper, highlight key findings, add margin notes, and ask the AI questions about methodology.

| Step | Action | Experience | Friction | Issue Refs |
|------|--------|------------|----------|------------|
| 1 | Navigate to /document-workspace from header navigation | Finds the link. Page loads. | None -- navigation label is clear enough. | -- |
| 2 | See empty workspace state | "Open a document to get started." Generic but functional. | No guidance on what the workspace can do. No tour or introduction. Researcher doesn't know about the 9 sidebar tabs, annotation tools, chat, or TTS. | D41, D42 |
| 3 | Click "Open Document" button. Document picker modal appears. | Sees "Recently ingested" section. | "Ingested"? Does that mean uploaded? Imported? Processed? Unclear but guesses correctly. | D7, D10 |
| 4 | Select a PDF. Document loads. | Sees a loading spinner. Waits. No progress indication. For a 30MB PDF on a slow connection, waits 20+ seconds with no feedback. Also notices both a banner alert AND a viewer overlay spinner. | Two loading indicators for one operation. No download progress for the large file. | D2, D3 |
| 5 | Document renders. Sees 9 sidebar tabs. | Overwhelmed. Chat, Notes, Figures, Quick Insights, References, Search, TTS, Bookmarks, Settings. Scans them all trying to understand the workspace. | 9 tabs is far more than standard PDF readers (Preview has 2, Adobe has 4). Cognitive overload on first impression. Tab labels are small (11px). | D32, D34 |
| 6 | Wants to highlight a passage. Looks for a "Highlight" button in the toolbar. | Doesn't find one. Toolbar has zoom, page nav, rotation, fit modes, search, fullscreen, print, download. No highlight button. | Highlighting requires selecting text first, then choosing from a popup. No hint or tooltip explains this workflow. Researcher is stuck. | D26, D33 |
| 7 | Eventually selects text by accident. Context menu appears with highlight options. | "Oh, THAT'S how it works." Creates a highlight. Wants to add a note to it. | Discoverable only by accident. First-time experience is frustrating. | D26 |
| 8 | Clicks "Notes" tab to see annotations. | Tab is labeled "Notes" but shows highlights with attached comments -- these are annotations, not freeform notes. | Terminology mismatch. Researcher expected a scratchpad for freeform notes. | D17 |
| 9 | Wants to ask the AI a question about the paper's methodology. Opens Chat tab. | Sees "Select a model in settings to chat." No link to settings. No explanation of what a "model" is. | Complete dead end. Researcher cannot use chat without configuring an LLM, but doesn't know what that means or where to do it. This is a wall. | D12, D38 |
| 10 | Explores settings independently. Finds model configuration. Returns to Chat. Sees "Use document sources (RAG)" toggle. | "RAG"? Toggles it on, guessing it means "use the document." Types a question. | Jargon on a primary control. Guessing works but confidence is low. | D5 |
| 11 | Gets an answer from the AI. Wants to see cited figures. Clicks "Figures" tab. | Sees page thumbnails, not the charts and graphs from the paper. | Major expectation mismatch. "Figures" in academic context means charts/graphs/diagrams. This shows page previews. | D8 |
| 12 | Wants to export annotations to share with a colleague. | Looks for "Export" button. Doesn't see one at the top level. Eventually finds it in a dropdown inside the Annotations panel. | Buried action for a common workflow. | D27 |
| 13 | Done reading. Closes the document tab. Had pending annotation sync. | No confirmation dialog. Sync indicator was showing "Syncing..." Tab closes. Annotations may be lost. | Data loss risk with no warning. | D15, D21 |
| 14 | Realizes annotations were lost. Tries to undo the close. | No undo capability. Must re-open the document and recreate annotations. | No recovery path. | D16 |

**Verdict**: The document reading experience is solid, but the surrounding features (chat, annotations, sidebar navigation) present significant barriers. The researcher's core journey -- read, annotate, ask questions -- is blocked or frustrated at multiple critical points.

---

### Journey B: Technical Power User -- "Multi-document research with keyboard efficiency"

**Task**: Open 3 related papers, use keyboard shortcuts for navigation, leverage RAG chat for cross-referencing, generate quick insights, and export annotations.

| Step | Action | Experience | Friction | Issue Refs |
|------|--------|------------|----------|------------|
| 1 | Navigate to /document-workspace. Press "?" to see shortcuts. | Keyboard shortcuts modal opens. Comprehensive list. | Smooth -- but only because this user already knows the "?" convention. Non-discoverable for others. | D24, D44 |
| 2 | Open 3 documents from picker. Tab bar appears. | Documents load. Tab bar shows 3 tabs with truncated filenames. | Good -- multi-document support works well once discovered. No indication it existed until the 2nd document was opened. | D29 |
| 3 | Use Cmd+[ and Cmd+] to toggle sidebar panels. | Panels resize smoothly. Keyboard shortcuts work as expected. | None -- well-implemented keyboard support. | -- |
| 4 | Use Cmd+F for in-document search. | Search panel opens in sidebar. Works well. | None -- standard behavior. | -- |
| 5 | Switch to Chat tab. Toggle "Use document sources (RAG)" on. Ask a technical question. | Chat responds with cited answer. References point to specific pages. | Smooth. The "RAG" label is acceptable jargon for this persona. | -- |
| 6 | Switch to Quick Insights tab. Select "Detailed" level. | Insights generate. Sees "Server connection required" if server is down. | Detail level selector has no explanation, but power user experiments freely. Server error message is opaque but this user checks connection status independently. | D28, D9, D37 |
| 7 | Generate insights for all 3 documents. Compare across tabs. | Tab switching works. Insights are per-document. | No cross-document comparison feature. Must manually switch tabs and compare. | -- |
| 8 | Export annotations from all 3 documents. | Opens each document's annotation panel. Finds export in dropdown. Exports JSON. | Wants Markdown or CSV export. Only JSON available. Must export 3 times (no batch). | D27, D31 |
| 9 | Try Quiz or Flashcard features. | Not available in document workspace (those are in /workspace-playground). | Feature gap for this page, but understandable scope boundary. | -- |
| 10 | Encounter a health check warning about migrations. | "Run the latest migrations to create missing tables." Understands this immediately. | No friction for this persona -- they know what migrations are. But acknowledges this would terrify a non-technical user. | D13, D36 |

**Verdict**: The power user journey is mostly smooth. Keyboard shortcuts, multi-document support, and RAG chat work well. Primary gaps are batch operations, cross-document features, and the single JSON export format. The experience is designed around this persona -- at the expense of the non-technical researcher.

---

## Part 5: Implementation Mapping

This section maps each D-issue to implementation stages for a fix plan. Stages are ordered by dependency and impact.

### Stage 0: Language and Copy Fixes (no logic changes)

These are pure text/label changes that can be done in a single pass with no behavioral changes.

| Issue | Fix Description | File(s) |
|-------|----------------|---------|
| D4 | Replace "RAG-powered answers" with "Answers grounded in your document" | `DocumentViewer/index.tsx:258` |
| D5 | Replace "Use document sources (RAG)" with "Ground answers in this document" | `DocumentChat.tsx:377-379` |
| D6 | Replace "RAG is off. Enable to ground answers." with "Document referencing is off. Enable to get answers based on this document." | `DocumentChat.tsx:456-458` |
| D7 | Replace "ingested into your media library" with "added to your library" | `DocumentPickerModal.tsx:509-510` |
| D10 | Replace "Recently ingested" with "Recently added" | `DocumentPickerModal.tsx:342` |
| D11 | Replace "not preserved during ingest" with "could not be fully processed" | `DocumentWorkspacePage.tsx:649-654` |
| D13 | Replace "run the latest migrations to create missing tables" with "The server needs to be updated. Please contact your administrator." | `DocumentWorkspacePage.tsx:817-819` |
| D18 | Tone down marketing-style feature card copy to match functional UI tone | `DocumentViewer/index.tsx:249-269` |
| D36 | Same as D13 (health warning language) | `DocumentWorkspacePage.tsx:817-819` |
| D39 | Replace "RAG search failed" / "tldw server" with "Could not search the document. Please check your connection and try again." | `DocumentChat.tsx:213-216` |

### Stage 1: Error Prevention and Data Safety

These changes prevent data loss and fix blocking error states.

| Issue | Fix Description | File(s) |
|-------|----------------|---------|
| D15/D21 | Add close confirmation dialog when document has pending annotations or sync is in progress. Check SyncStatusIndicator state before allowing tab close. | `DocumentTabBar.tsx:121-124`, `SyncStatusIndicator.tsx` |
| D22 | Disable chat input and show "Reconnecting..." message when server connection is lost. Gate on connection state. | `DocumentChat.tsx:403` |
| D12/D38 | Add a direct link/button to model settings in the "Select a model" message. Navigates to the settings panel or opens the settings modal. | `DocumentChat.tsx:384-389` |
| D23 | Distinguish empty library from no search results in DocumentPickerModal. Show onboarding guidance for empty library; show search suggestions for no results. | `DocumentPickerModal.tsx:393-399` |
| D37 | Add "Retry connection" button and brief explanation to "Server connection required" states. | `QuickInsightsTab.tsx:149-154`, `ReferencesTab.tsx:342-355` |

### Stage 2: Information Architecture and Progressive Disclosure

These changes reduce cognitive overload and improve discoverability.

| Issue | Fix Description | File(s) |
|-------|----------------|---------|
| D32 | Group 9 sidebar tabs into primary (Chat, Annotations, Search) and secondary (collapsed under "More": Figures, Quick Insights, References, TTS, Bookmarks, Settings). Show 3-4 primary tabs; secondary tabs in overflow menu. | `DocumentWorkspacePage.tsx:148-174,231-252` |
| D33 | Group toolbar into primary controls (zoom, page nav, search) visible by default and secondary controls (rotation, fit modes, fullscreen, print, download) in overflow dropdown. | `ViewerToolbar.tsx:124-261` |
| D17 | Rename "Notes" tab to "Annotations" to match its content. Or split into "Annotations" (highlights+comments) and "Notes" (freeform scratchpad). | `DocumentWorkspacePage.tsx:239`, `AnnotationsPanel.tsx:564` |
| D8 | Rename "Figures" tab to "Page Thumbnails" or "Pages". If figure extraction is planned, keep the label but implement actual figure extraction. | `DocumentWorkspacePage.tsx:155-157`, `FiguresTab.tsx:11,87-88` |
| D34/D35 | Increase tab label font size to minimum 12px (desktop) and 12px (mobile). Ensure WCAG AA compliance for interactive element labels. | `TabIconLabel.tsx:16`, `DocumentWorkspacePage.tsx:901` |

### Stage 3: Onboarding and Help System

These changes introduce guidance for first-time users.

| Issue | Fix Description | File(s) |
|-------|----------------|---------|
| D41 | Implement a guided onboarding tour (Joyride or similar) that highlights key features: open a document, create an annotation, use chat, explore sidebar tabs. Opt-in trigger, not auto-start. Dismissal should NOT be permanent. | New component + integration in `DocumentWorkspacePage.tsx` |
| D14/D40 | Add a "Tips" or "What's New" menu item that re-surfaces dismissed feature cards. Store dismissal per-card with a central re-show mechanism. | `DocumentViewer/index.tsx:238-246` |
| D26 | Add a first-time tooltip or inline hint near the document viewer: "Select text to highlight and annotate." Show once, then dismiss. Re-accessible from help menu. | `AnnotationsPanel.tsx:599-601` or viewer overlay |
| D42 | Enrich empty states with contextual guidance. Chat empty: "Ask questions about your document -- try 'Summarize the key findings'". Annotations empty: "Select text in the document to create highlights and notes." | All sidebar/panel empty states |
| D43 | Add a "Help" or "?" icon button in the toolbar/header that opens a help panel or links to documentation. | New button in `ViewerToolbar.tsx` or `DocumentWorkspacePage.tsx` header |
| D44 | Add visible keyboard icon in toolbar that opens the shortcuts modal. Remove dependency on knowing the "?" convention. | `DocumentShortcutsModal.tsx:234-236`, `ViewerToolbar.tsx` |
| D24 | Same as D44 -- making shortcuts discoverable via visible UI element. | `DocumentWorkspacePage.tsx:354` |

### Stage 4: Polish and Feature Gaps

These changes improve efficiency for power users and add missing capabilities.

| Issue | Fix Description | File(s) |
|-------|----------------|---------|
| D1 | Expand SyncStatusIndicator tooltip to explain what is being synced: "Saving annotations and reading position..." | `SyncStatusIndicator.tsx:48-50` |
| D2 | Consolidate duplicate loading indicators into a single unified loading state. Remove the alert banner and keep only the viewer overlay, or vice versa. | `DocumentWorkspacePage.tsx:763-779`, `DocumentViewer/index.tsx:396-403` |
| D3 | Add download progress indicator for large documents (file size + percentage or progress bar). | `DocumentWorkspacePage.tsx:387,763-779` |
| D9 | Expand "Server connection required" with brief explanation: "AI features need an active server connection. Check that your tldw server is running." | `QuickInsightsTab.tsx:149-154` |
| D16 | Implement undo-close-tab (Cmd+Shift+T or equivalent). Maintain a closed-tab stack with document ID and scroll position. | `DocumentWorkspacePage.tsx:546-556` |
| D19 | Externalize shortcuts modal title to i18n translation key. | `DocumentShortcutsModal.tsx:195-196` |
| D20 | Document the desktop/mobile paradigm differences. Consider adding a brief orientation message on mobile: "Swipe between tabs to access Chat, Annotations, and more." | `DocumentWorkspacePage.tsx:828-907,910-1026` |
| D25 | Add text label "Read Aloud" next to the TTS icon button, or add a tooltip with that label. | `ViewerToolbar.tsx` (TTS button area) |
| D27 | Promote "Export Annotations" to a visible button in the annotations panel header, outside the dropdown. | `AnnotationsPanel.tsx:566-574` |
| D28 | Add brief descriptions to each detail level option: "Brief: Key points only. Standard: Main findings and context. Detailed: Comprehensive analysis with examples." | `QuickInsightsTab.tsx:24-40` |
| D29 | Add a subtle hint on first document open: "Open multiple documents to compare side-by-side in tabs." | `DocumentTabBar.tsx:152-155` |
| D30 | Implement document-level bookmarks (separate from annotations). Allow bookmarking specific pages with optional labels. | New feature in sidebar |
| D31 | Add Markdown and CSV export options for annotations alongside existing JSON export. | `AnnotationsPanel.tsx` export logic |

---

## Appendix: Issue Index

Quick-reference table of all 44 issues sorted by ID.

| ID | Heuristic | Score Impact | Priority | Summary |
|----|-----------|-------------|----------|---------|
| D1 | H1 | 3.5 | P2 | Sync status doesn't explain what's syncing |
| D2 | H1 | 3.5 | P2 | Duplicate loading indicators for same operation |
| D3 | H1 | 3.5 | P1 | No download progress for large documents |
| D4 | H2 | 2.0 | P1 | "RAG-powered answers" jargon in feature card |
| D5 | H2 | 2.0 | P1 | "Use document sources (RAG)" toggle label |
| D6 | H2 | 2.0 | P2 | "RAG is off" mixed jargon message |
| D7 | H2 | 2.0 | P2 | "Ingested into your media library" jargon |
| D8 | H2 | 2.0 | P1 | "Figures" tab shows page thumbnails, not figures |
| D9 | H2 | 2.0 | P2 | "Server connection required" with no explanation |
| D10 | H2 | 2.0 | P3 | "Recently ingested" label |
| D11 | H2 | 2.0 | P2 | "Not preserved during ingest" error message |
| D12 | H2 | 2.0 | P0 | "Select a model in settings" with no link |
| D13 | H2 | 2.0 | P0 | "Run the latest migrations" developer language |
| D14 | H3 | 3.0 | P2 | Feature cards permanently dismissible |
| D15 | H3 | 3.0 | P0 | No close confirmation with pending annotations |
| D16 | H3 | 3.0 | P2 | No undo close document |
| D17 | H4 | 3.0 | P2 | "Notes" tab renders annotation content |
| D18 | H4 | 3.0 | P3 | Marketing tone in feature cards |
| D19 | H4 | 3.0 | P3 | Hardcoded English in shortcuts modal |
| D20 | H4 | 3.0 | P2 | Desktop/mobile paradigm gap |
| D21 | H5 | 3.5 | P0 | No close confirmation with pending sync |
| D22 | H5 | 3.5 | P2 | Chat input not disabled when offline |
| D23 | H5 | 3.5 | P2 | Empty library vs no results indistinguishable |
| D24 | H6 | 3.0 | P2 | Shortcuts only accessible via "?" key |
| D25 | H6 | 3.0 | P2 | TTS hidden behind icon-only button |
| D26 | H6 | 3.0 | P1 | Highlight workflow invisible to new users |
| D27 | H6 | 3.0 | P3 | Export annotations buried in dropdown |
| D28 | H6 | 3.0 | P2 | Detail level selector not explained |
| D29 | H6 | 3.0 | P3 | Multi-document feature invisible until 2nd doc |
| D30 | H7 | 4.0 | P3 | No document-level bookmarks |
| D31 | H7 | 4.0 | P3 | No batch annotation export beyond JSON |
| D32 | H8 | 2.5 | P0 | 9 sidebar tabs visible simultaneously |
| D33 | H8 | 2.5 | P2 | Toolbar shows all controls simultaneously |
| D34 | H8 | 2.5 | P2 | Tab labels at 11px font size |
| D35 | H8 | 2.5 | P2 | Mobile nav labels at 10px font size |
| D36 | H9 | 3.0 | P1 | Health warning uses "migrations" language |
| D37 | H9 | 3.0 | P2 | "Server connection required" with no guidance |
| D38 | H9 | 3.0 | P2 | "Select a model" with no link to settings |
| D39 | H9 | 3.0 | P2 | Server error mentions "RAG search" and "tldw server" |
| D40 | H10 | 2.0 | P2 | Feature cards permanently dismissible (help angle) |
| D41 | H10 | 2.0 | P1 | No onboarding tour |
| D42 | H10 | 2.0 | P2 | Generic empty states with no guidance |
| D43 | H10 | 2.0 | P2 | No in-page help link or documentation |
| D44 | H10 | 2.0 | P2 | "?" shortcut not discoverable |
