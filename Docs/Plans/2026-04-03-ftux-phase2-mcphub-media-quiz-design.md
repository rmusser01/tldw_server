# FTUX Audit Phase 2: MCPHub, Media, and Quizzes Pages

## Context

The previous FTUX audit (Phase 1) focused on onboarding, navigation, chat, moderation, and workspace. That work is implemented on `feature/ftux-phase1-squashed`. This audit covers the three remaining high-traffic pages — **MCPHub** (`/mcp-hub`), **Media** (`/media`), and **Quizzes** (`/quiz`) — plus their extension integration. Issues are evaluated from three personas: **family-safety parent** (non-technical), **content consumer** (tech-comfortable, non-developer), and **technical user**.

---

## Comprehensive Issues List

### A. MCPHub Page (`/mcp-hub`)

| # | Issue | Severity | Persona | File |
|---|-------|----------|---------|------|
| MH1 | **No page-level explanation of what MCP Hub is** — Title says "MCP Hub" with no subtitle, no acronym expansion, no description. First-time visitors of any skill level see an unexplained acronym followed by 11 tabs. | High | All | `McpHubPage.tsx:57-59` |
| MH2 | **No tutorial definition exists** — 12 other pages have Joyride tutorials. MCPHub (the most complex page) has none. No `data-testid` attributes on key elements to target. | High | All | `tutorials/registry.ts` (missing), `tutorials/definitions/` (no mcp-hub.ts) |
| MH3 | **11 tabs shown simultaneously with no grouping or progressive disclosure** — Profiles, Assignments, Path Scopes, Capability Mappings, Workspace Sets, Shared Workspaces, Audit, Governance Packs, Approvals, Catalog, Credentials. No "basic vs. advanced" separation. No indication of dependency order. | High | All | `McpHubPage.tsx:60-142` |
| MH4 | **Default tab "Profiles" is empty and unhelpful for new users** — Shows "No permission profiles yet" with jargon description. Should default to "Catalog" or "Credentials" (actionable tabs). | High | All | `McpHubPage.tsx:23`, `PermissionProfilesTab.tsx:450` |
| MH5 | **All empty states are bare `<Empty>` with no actionable guidance** — 15+ empty states across tabs, all "No X yet" with no explanation, no CTA, no prerequisite info, no documentation links. | Medium | All | All MCPHub tab files |
| MH6 | **Tab descriptions use unexplained domain jargon** — "tool allowlists", "runtime consent", "registry-backed tool metadata", "baseline restrictions" — undefined anywhere on the page. | Medium | Parent, Consumer | All tab files (subtitle Typography.Text) |
| MH7 | **No access gating for non-technical users** — Any user who navigates to `/mcp-hub` sees the full governance interface. No "this is advanced" interstitial or role-based visibility. | Medium | Parent | `option-mcp-hub.tsx` |
| MH8 | **Delete operations use bare `window.confirm()`** — 10 instances of native browser dialogs with no styling, no consequence warnings, no dependency info. | Medium | All | 10 files (PermissionProfilesTab, ExternalServersTab, etc.) |
| MH9 | **Error messages are generic** — All `catch` blocks show "Failed to X" with no status code, no server detail, no retry button, no documentation link. | Medium | All | All MCPHub tab files (40+ error strings) |
| MH10 | **No tooltips or contextual help on any form fields** — Zero instances of Tooltip, Popover, or help icons. Complex dropdowns (Owner Scope, Profile Mode, Secret Kind, etc.) have no explanation. | Medium | Consumer, Technical | All MCPHub form components |
| MH11 | **"Credentials" tab label is misleading** — Renders `ExternalServersTab` which manages servers, auth templates, and imports. "Credentials" suggests password management. | Low | Technical | `McpHubPage.tsx:133` |
| MH12 | **"Catalog" tab label is vague** — Shows tool registry metadata but "Catalog" doesn't communicate "Tool Catalog." This is the most useful first-visit tab but doesn't signal that. | Low | All | `McpHubPage.tsx:129` |
| MH13 | **Page reachable via two routes** — `/mcp-hub` and `/settings/mcp-hub` render same component with different layouts. Tutorial/state may not carry over. | Low | Technical | `option-mcp-hub.tsx`, `option-settings-mcp-hub.tsx` |

---

### B. Media Page (`/media`)

| # | Issue | Severity | Persona | File |
|---|-------|----------|---------|------|
| MD1 | **No first-time onboarding on the primary `/media` page** — Empty library shows generic "No results found" + "Try broader terms, or ingest new content to search." Small "Open Quick Ingest" button is the only action. Meanwhile, the legacy `/media-multi` page has a rich first-ingest tutorial with inline URL input, format guidance, and visual hierarchy. The primary route has none of this. | High | All | `ResultsList.tsx:284-334` |
| MD2 | **Inline URL input in first-ingest tutorial (on `/media-multi`) discards the typed URL** — User types YouTube URL, clicks "Ingest", but `requestQuickIngestOpen()` is called with no arguments. URL is ignored; Quick Ingest modal opens empty. | High | Consumer, Parent | `MediaReviewResultsList.tsx:196-209` |
| MD3 | **No post-ingest auto-refresh** — After Quick Ingest submission, results don't refresh. User must manually search or wait 30s for stale check. The critical "first ingest -> see content" moment has no automated bridge. | High | Consumer, Parent | `ViewMediaPage.tsx` (entire flow) |
| MD4 | **Ingest jobs panel requires manual batch ID entry** — Hidden in collapsed "Library tools" section. Requires pasting a "Batch ID from ingest response" — a concept foreign to non-technical users. No auto-linking from Quick Ingest. | High | Consumer, Parent | `MediaIngestJobsPanel.tsx:170-220` |
| MD5 | **Content viewer empty state misleading when library is empty** — Shows "Select a media item from the left sidebar" when there ARE no items. Also shows keyboard shortcut hints (j/k) prematurely. | Medium | Consumer, Parent | `ContentViewer.tsx:420-458` |
| MD6 | **Tutorial toast stays visible indefinitely** — `NOTIFICATION_DURATION = 0` means the "Take a quick tour" prompt never auto-dismisses. Partially obscures UI until manually closed. | Medium | All | `TutorialPrompt.tsx:24` |
| MD7 | **Filter panel sections all collapsed by default** — "Media types" filter only auto-expands when server returns types (won't happen on empty library). Filters are below the Search button — users may not scroll to find them. | Medium | Consumer, Technical | `FilterPanel.tsx:134-138` |
| MD8 | **Server offline states use identical copy for different scenarios** — Demo offline, generic offline, and "media unsupported" reuse same translation keys. Some states have retry buttons, others are dead ends. | Medium | All | `ViewMediaPage.tsx:193-261` |
| MD9 | **Two-column layout wastes screen on empty library** — Right pane (content viewer) takes 70%+ of screen but shows only "No media item selected" message. Left sidebar compresses the useful CTAs. | Medium | Consumer, Parent | `ViewMediaPage.tsx:1043-1057` |
| MD10 | **`media-basics` tutorial targets elements that may not exist in FTUX** — Steps 4-5 target `media-results-list` (empty) and `content-scroll-container` (not rendered when no item selected). Joyride may spotlight nothing. | Medium | All | `tutorials/definitions/media.ts:18-75` |
| MD11 | **"Dismiss" button in first-ingest tutorial leads to permanent dead end** — On `/media-multi`, dismissing tutorial sets localStorage flag forever. Post-dismissal shows bare `<Empty>` with no actions. No "show tutorial again" option. | Medium | Consumer, Parent | `MediaReviewResultsList.tsx:216-227` |
| MD12 | **Search help ("?") is tooltip-only** — 20x20px button, hover-only, no click behavior, no persistent reference. | Low | Technical | `SearchBar.tsx:86-102` |
| MD13 | **Search requires explicit submit — no auto-search on typing** — After changing filters/query, user must click Search or press Enter. No visual cue that results are stale. | Low | Consumer | `ViewMediaPage.tsx:1215-1221` |

---

### C. Quizzes Page (`/quiz`)

| # | Issue | Severity | Persona | File |
|---|-------|----------|---------|------|
| QZ1 | **No quiz tutorial definition exists** — Every other major feature has a Joyride tutorial. Quizzes have none. No `quiz.ts` in `tutorials/definitions/`, no quiz import in `registry.ts`. | High | All | `tutorials/registry.ts`, `tutorials/definitions/` |
| QZ2 | **Default tab is "Take Quiz" which shows empty state for new users** — Connected users with no quizzes land on "No quizzes available to take yet." Must navigate to Generate or Create themselves. Should default to Generate when `totalQuizzes === 0`. | High | Consumer, Parent | `QuizPlayground.tsx:53`, `TakeQuizTab.tsx:2282-2318` |
| QZ3 | **Generate tab shows no guidance when media library is empty** — Dropdown shows "No media found" with no link to `/media`, no explanation that media must be ingested first. "Generate Quiz" button disabled with no explanation. | High | All | `GenerateTab.tsx:958-1013, 1328-1338` |
| QZ4 | **No cross-navigation link from Generate tab to Media page** — The quiz feature's primary value depends on ingested media, but the Generate tab never points users to the ingestion pipeline when empty. | High | All | `GenerateTab.tsx:958-1013` |
| QZ5 | **Four source types presented simultaneously without explanation** — Media, Notes, Flashcard Decks, Flashcards shown in "Select Sources" card with no guidance on which to use or what the difference is. | Medium | Parent, Consumer | `GenerateTab.tsx:953-1099` |
| QZ6 | **Demo quizzes only accessible when disconnected** — `DemoQuizPreview` renders only when `!isOnline`. Connected users with zero quizzes cannot try the demo to learn the interface. | Medium | All | `QuizWorkspace.tsx:621-696` |
| QZ7 | **Demo does not teach Generate or Create workflows** — Demo covers catalog -> take -> results. Doesn't demonstrate the two most critical first-time actions (creating quizzes). | Medium | Consumer, Parent | `QuizWorkspace.tsx:94-421` |
| QZ8 | **Create tab opens with blank form, no orientation** — No guidance on question types available, recommended quiz length, or examples. Parent creating educational quizzes has no template. | Medium | Parent, Consumer | `CreateTab.tsx:968-1083` |
| QZ9 | **No visual indicator of recommended starting tab** — 5 tabs with equal visual weight. New users don't know where to start. No "Start here" badge on Generate tab. | Medium | All | `QuizPlayground.tsx:277-413` |
| QZ10 | **Beta tooltip has technical jargon** — "Score semantics may change" is meaningless to non-technical users. "Demo responses not saved" shows in all states, not just demo mode. | Medium | Parent, Consumer | `QuizWorkspace.tsx:42-92, 473-476` |
| QZ11 | **TakeQuizTab empty state doesn't explain media dependency** — Says "Generate one from media" but "media" is just text, not a link. No explanation of what media means here. | Medium | Parent, Consumer | `TakeQuizTab.tsx:2291-2316` |
| QZ12 | **Results tab empty state has no navigation CTAs** — Shows "No quiz attempts yet" with no button to navigate to Take Quiz or Generate. Dead end for explorers. | Medium | Consumer, Parent | `ResultsTab.tsx:1311-1330` |
| QZ13 | **No sidepanel/extension quiz access** — Quizzes are options-page only. No lightweight quiz widget in the sidepanel. Deep-linking exists (`/quiz?start_quiz_id=X`) but no extension surface. | Medium | Consumer, Technical | `quiz-flashcards-handoff.ts`, header shortcuts |
| QZ14 | **"Reset Current Tab" button visible with nothing to reset** — Adds cognitive noise for first-time users who have no activity. No tooltip explaining what it does. | Low | Parent, Consumer | `QuizPlayground.tsx:263-269` |
| QZ15 | **Mobile tab labels too abbreviated** — "Gen", "Build", "Stats" are ambiguous. "Stats" instead of "Results" may confuse users. | Low | Consumer, Parent | `QuizPlayground.tsx:121-141` |

---

### D. Extension Integration (Cross-Page)

| # | Issue | Severity | Persona | File |
|---|-------|----------|---------|------|
| EX1 | **Extension sidepanel has no Media page access** — Media browsing/search is options-page only. Sidepanel focuses on chat/companion. A content consumer who installed the extension to summarize content can't browse their library from the sidepanel. | Medium | Consumer | Extension sidepanel routes |
| EX2 | **Extension sidepanel has no Quiz access** — See QZ13. No quiz widget or link in sidepanel. | Medium | Consumer | Extension sidepanel routes |
| EX3 | **No "Open in full UI" links for Media/Quiz features** — Sidepanel has UI mode toggle but no feature-specific deep links ("Open Media Library" or "Take a Quiz" buttons that open the options page at the right route). | Medium | Consumer, Technical | `sidepanel-settings.tsx` |
| EX4 | **Extension Quick Ingest doesn't update Media page** — After using Quick Ingest from the extension sidepanel, the options-page Media view (if open) doesn't know content was ingested. No cross-tab event or polling. | Low | Consumer | Extension messaging architecture |

---

## Implementation Approach

### Phase 1: High-Severity Fixes (targeted friction removal)

These 12 items address the worst dead ends and broken flows:

**MCPHub (4 items):**
1. Add page subtitle + "What is MCP?" dismissible explainer card (`McpHubPage.tsx`)
2. Create `tutorials/definitions/mcp-hub.ts` with 5-step tour + add `data-testid` attrs to key elements
3. Group tabs into "Getting Started" (Catalog, Credentials) / "Policies" / "Advanced" categories; or add a visual divider
4. Change default tab from "Profiles" to "Catalog" when no profiles exist

**Media (4 items):**
5. Port first-ingest tutorial from `MediaReviewResultsList.tsx` into `ResultsList.tsx` for the primary `/media` route
6. Fix URL passthrough: wire input value to `requestQuickIngestOpen({ source: inputValue })`
7. Auto-refresh results after Quick Ingest completion (listen for `tldw:quick-ingest-complete` event)
8. Auto-link batch ID from Quick Ingest to `MediaIngestJobsPanel` (persist to storage key, auto-expand panel)

**Quizzes (4 items):**
9. Create `tutorials/definitions/quiz.ts` with a "quiz-basics" tutorial
10. When `totalQuizzes === 0`, default to "Generate" tab instead of "Take"
11. Add empty-media alert in GenerateTab with link to `/media`: "No content found. Go to Media to import videos, articles, or documents."
12. Add navigation CTA in Results tab empty state: "Take a Quiz" button

### Phase 2: Medium-Severity Improvements

**MCPHub:**
- Enrich empty states across all tabs with explanations, CTAs, and prerequisite notes
- Add tooltips to form fields (Owner Scope, Profile Mode, Secret Kind, etc.)
- Replace `window.confirm()` with styled Ant Design `Modal.confirm` including dependency warnings
- Add granular error messages with status codes and retry buttons

**Media:**
- Context-aware content viewer empty state (different message when library empty vs item not selected)
- Auto-dismiss tutorial toast after 15s
- Fix tutorial step targets for empty-library FTUX path
- Add retry/diagnostics to all offline states
- Improve two-column layout for empty library (single-column onboarding view)

**Quizzes:**
- Make demo accessible when connected but empty (`totalQuizzes === 0`)
- Add "Start here" badge to Generate tab when no quizzes exist
- Expand TakeQuizTab empty state with media library link
- Add orientation banner to Create tab explaining question types
- Split beta tooltip by context (connected vs demo)
- Add CTA to demo results screen bridging to real usage

**Extension:**
- Add "Open Media Library" and "Open Quizzes" deep links in sidepanel
- Surface lightweight quiz access from extension

---

## Key Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx` | 1 | Subtitle, default tab, tab grouping |
| `apps/packages/ui/src/tutorials/definitions/mcp-hub.ts` (new) | 1 | Tutorial definition |
| `apps/packages/ui/src/tutorials/registry.ts` | 1 | Register MCPHub + Quiz tutorials |
| `apps/packages/ui/src/components/Media/ResultsList.tsx` | 1 | First-ingest tutorial for empty library |
| `apps/packages/ui/src/components/Review/MediaReviewResultsList.tsx` | 1 | Fix URL passthrough |
| `apps/packages/ui/src/components/Review/ViewMediaPage.tsx` | 1 | Auto-refresh after ingest |
| `apps/packages/ui/src/components/Media/MediaIngestJobsPanel.tsx` | 1 | Auto-link batch ID |
| `apps/packages/ui/src/tutorials/definitions/quiz.ts` (new) | 1 | Tutorial definition |
| `apps/packages/ui/src/components/Quiz/QuizPlayground.tsx` | 1 | Default tab logic, "Start here" badge |
| `apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx` | 1 | Empty-media alert with link to /media |
| `apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx` | 1 | Navigation CTA in empty state |
| `apps/packages/ui/src/components/Option/MCPHub/PermissionProfilesTab.tsx` | 2 | Enriched empty state |
| `apps/packages/ui/src/components/Option/MCPHub/ExternalServersTab.tsx` | 2 | Enriched empty state, styled deletes |
| `apps/packages/ui/src/components/Media/ContentViewer.tsx` | 2 | Context-aware empty state |
| `apps/packages/ui/src/components/Common/TutorialPrompt.tsx` | 2 | Auto-dismiss timer |
| `apps/packages/ui/src/components/Quiz/QuizWorkspace.tsx` | 2 | Demo when connected, beta messaging |
| `apps/packages/ui/src/components/Quiz/tabs/TakeQuizTab.tsx` | 2 | Media library link in empty state |
| `apps/packages/ui/src/components/Quiz/tabs/CreateTab.tsx` | 2 | Orientation banner |

---

## Verification Plan

1. **Manual walkthrough per persona, per page:**
   - Clear localStorage, fresh browser session
   - Walk through: first visit -> first action -> first useful result
   - Count clicks from landing to productive outcome
   - Verify: MCPHub explainer is visible, Media ingest works end-to-end, Quiz generate flow has no dead ends

2. **Test coverage:**
   - Guard tests for each new empty state component
   - Tutorial step-target existence tests
   - URL passthrough test for first-ingest tutorial
   - Auto-refresh test for Quick Ingest completion event
   - Default-tab logic test for Quiz when `totalQuizzes === 0`

3. **Extension check:**
   - Verify deep links from sidepanel open correct options-page routes
   - Verify Quick Ingest from extension persists batch ID for Media page
