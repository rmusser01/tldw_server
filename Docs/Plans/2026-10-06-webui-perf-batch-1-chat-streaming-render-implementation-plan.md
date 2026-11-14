# WebUI Perf Batch W1 — Chat Streaming Render Path Implementation Plan

**Backlog task:** TASK-13521 · **Index:** [coordination index](2026-10-06-webui-perf-remediation-coordination-index.md)
**Verified against:** `dev` @ `7ba48f251e` (grep anchors inline; line numbers drift)
**Findings:** WP-01..WP-14

**Goal:** Stop re-rendering entire chat transcripts per streamed token: throttle sidepanel updates, memoize the row component, virtualize the two web transcripts, and remove per-row O(n) scans.

**Architecture:** The repo already contains every pattern needed — an 80ms streaming throttle (`STREAMING_UPDATE_INTERVAL_MS` in `hooks/chat/useChatActions.ts`, anchor: `grep -n "STREAMING_UPDATE_INTERVAL_MS" hooks/chat/useChatActions.ts`), a virtualized chat body (`components/Sidepanel/Chat/body.tsx` via `@tanstack/react-virtual`, anchor: `grep -n "useVirtualizer" components/Sidepanel/Chat/body.tsx`), and a memoized content child (`Common/Playground/MessageContent.tsx`). This batch ports those three proven patterns to the surfaces that lack them.

All paths below are relative to `apps/packages/ui/src` unless noted.

---

## Stage 1: Throttle + narrow sidepanel streaming updates (WP-01)

**Goal:** Per-token `setMessages` full-array clones replaced by an 80ms-coalesced update that touches only the last message.
**Files:** Modify `hooks/useMessage.tsx` (chunk handler, anchor: `grep -n "prev.map((m) =>" hooks/useMessage.tsx`); port scheduler from `hooks/chat/useChatActions.ts`.
**Approach:**
- Extract the existing `scheduleStreamingUpdate` throttle into a shared helper (e.g. `hooks/chat/streamingThrottle.ts`) and consume it from both paths.
- In the update itself, replace `prev.map(...)` with last-index replacement: `const next = prev.slice(); next[last] = { ...prev[last], message: fullText };` guarded to the streaming message id.
**Success Criteria:** W0 Stage-1 bench shows ≤ 12 store updates/sec during streaming (from per-token); array-clone count per token = 0.
**Tests:** Unit test: chunk handler fed 500 chunks <80ms apart produces ≤ ceil(elapsed/80)+1 updates; message content equals concatenation of all chunks; non-streaming messages keep referential identity.
**Status:** Not Started

## Stage 2: Memoize `PlaygroundMessage` + stable row props (WP-02, WP-03 partial, WP-09)

**Goal:** A memo boundary around the hottest row component; parents stop defeating it.
**Files:** Modify `components/Common/Playground/Message.tsx` (anchor: `grep -n "export const PlaygroundMessage" components/Common/Playground/Message.tsx`), `components/Option/Playground/PlaygroundChat.tsx` (row JSX, anchor: `grep -n "isStreaming={streaming}" components/Option/Playground/PlaygroundChat.tsx`), `components/Sidepanel/Chat/body.tsx` (row props).
**Approach:**
- Wrap export: `export const PlaygroundMessage = React.memo(function PlaygroundMessage(props: Props) { ... })`.
- Hoist per-row callbacks: parents pass `messageId` + stable `useCallback` handlers (dispatch-by-id), or a single `onRowAction(action, id)` discriminated callback. Replace `images={message.images || []}` with a module-level `EMPTY_IMAGES` constant.
- `isStreaming={streaming && block.index === messages.length - 1}` (ChatPane already does this correctly — copy it).
- Memoize search highlighting inside the row: `useMemo(() => searchQuery ? highlightText(message, searchQuery) : message, [message, searchQuery])` (anchor: `grep -n "highlightText" components/Common/Playground/Message.tsx`).
**Success Criteria:** React Profiler on a 100-message chat: a token flush re-renders 1 row, not 100.
**Tests:** Unit: render 50 rows, bump last-message text, assert memoized rows did not re-render (spy on inner render via `MessageContent` mock). Existing `Playground.cockpit-controls` / `Playground.coordinator` suites stay green.
**Status:** Not Started

## Stage 3: Virtualize the two web transcripts (WP-03, WP-04, WP-07)

**Goal:** Windowed rendering for `PlaygroundChat` and ResearchWorkspace `ChatPane`; bounded growth for `SharedWorkspaceChatPane`.
**Files:** Modify `components/Option/Playground/PlaygroundChat.tsx`, `components/Option/ResearchWorkspace/ChatPane/index.tsx` (anchor: `grep -n "messages.map((msg, idx)" components/Option/ResearchWorkspace/ChatPane/index.tsx`), `components/Option/ResearchWorkspace/SharedResearchWorkspace/SharedWorkspaceChatPane.tsx` (anchor: `grep -n "state.messages.map" ...SharedWorkspaceChatPane.tsx`).
**Approach:**
- Copy the `useVirtualizer` wiring from `Sidepanel/Chat/body.tsx:136-143` (overscan 3, `measureElement`, existing scroll-anchor mitigation at 211-236) into both panes.
- `ChatPane`: memoize `buildRetrievalDiagnostics` per message id (`useMemo` map over `msg.sources/generationInfo` — anchor: `grep -n "buildRetrievalDiagnostics" components/Option/ResearchWorkspace/ChatPane/index.tsx`); drop the `msg-${idx}` key fallback (generate ids at insert).
- Shared pane: cap rendered history to newest K=200 with "load older" (bidirectional window); virtualize if markdown cost still shows in profiler.
**Success Criteria:** 1,000-message transcript renders ≤ ~20 message shells; W0 e2e perf spec long-task count during streaming drops ≥ 50% from baseline.
**Tests:** Unit: virtualizer renders windowed subset (existing sidepanel body tests as template); e2e: extension/Playground scroll-through spec asserts no unbounded DOM (`document.querySelectorAll('[data-message]').length < 60`).
**Status:** Not Started

## Stage 4: Selector + context hygiene (WP-05, WP-06, WP-08, WP-10, WP-11)

**Goal:** Remove remaining per-render linear work and keystroke-wide context re-renders.
**Files / fixes (anchors in parens):**
- `components/Sidepanel/Chat/body.tsx` + `PlaygroundChat.tsx`: precompute previous-user-message `Map` per `messages` identity in `useMemo`; `getPreviousUserMessage` becomes O(1) (`grep -n "getPreviousUserMessage" components/Sidepanel/Chat/body.tsx`).
- `components/Sidepanel/Chat/CharacterSelect.tsx` + `Common/CharacterSelect.tsx`: subscribe to derived boolean `useStoreMessageOption((s) => s.messages.length > 0)` (`grep -n "state.messages)" components/Sidepanel/Chat/CharacterSelect.tsx`).
- `components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`: move raw search query to local state in `SearchBar.tsx`, dispatch debounced (250ms, matches Sidebar precedent) into the reducer; split context into state/actions halves so keystrokes don't re-render consumers (`grep -n "action.payload, queryWarning" components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`).
- `components/Layouts/Layout.tsx:789`: memoize `LayoutShellContext.Provider` value.
- `components/Media/ResultsList.tsx`: extract memoized row (`buildInspectorTooltip` inside the row's `useMemo`) (`grep -n "buildInspectorTooltip" components/Media/ResultsList.tsx`).
**Success Criteria:** KnowledgeQA typing 20 chars re-renders SearchBar only; profiler shows no transcript-wide renders on `isStreaming` toggle.
**Tests:** Unit: KnowledgeQA provider — 10 rapid query dispatches produce ≤ 1 reducer update after debounce; CharacterSelect does not re-render on last-message text change; `Layout` consumer does not re-render on sibling state change.
**Status:** Not Started

## Stage 5: Minor render items (WP-12, WP-13, WP-14)

**Files / fixes:**
- `apps/tldw-frontend/pages/research.tsx`: wrap `deriveTrustView`, `evaluateCheckpointEditor`, `loadedTrustArtifactNames` in `useMemo`; `textToStringList` dedupe via `new Set` (anchors: `grep -n "deriveTrustView\\|textToStringList" pages/research.tsx`).
- `components/Option/ChatWorkflows/ChatWorkflowsPage.tsx:1554` and `components/vn-scripts/VNScriptsWorkbench.tsx`: replace index keys with stable ids where lists can reorder.
- Flashcards panel (WP-12): documented as deferred in the index; no change.
**Success Criteria:** No keystroke-time O(n) recomputation in the research trust panel.
**Tests:** Existing research page suites green; add unit test for `textToStringList` (Set semantics).
**Status:** Not Started

## Verification & DoD

- [ ] Each stage: failing test first, then implementation, then `bun run vitest run <scoped>` green; commit referencing TASK-13521.
- [ ] W0 Stage-1 bench re-run; delta recorded in `Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md`.
- [ ] Full `apps/packages/ui` `vitest run` + `apps/tldw-frontend` `test:run` + extension `test:e2e:perf` green.
- [ ] `bun run lint` clean on touched paths.
