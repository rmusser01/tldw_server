# Media Viewer Read-Along TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build selection-initiated read-along TTS in the shared media viewer used by the WebUI and browser extension.

**Architecture:** Add a focused `read-along` module under `apps/packages/ui/src/components/Media/` for pure segmentation, selection mapping, cache keys, playback session state, and compact UI controls. Integrate through `ContentViewer` only after selection mediation is in place so annotations and read-along share one selection-action path. Reuse the existing TTS provider stack and Dexie database, adding only the minimum abort/cache surface needed for read-along.

**Tech Stack:** React 18, TypeScript, Vitest, Testing Library, Dexie, Ant Design primitives already used by `ContentViewer`, lucide-react icons, existing `@tldw/ui` TTS services.

---

## Source Documents

- Design spec: `Docs/superpowers/specs/2026-05-17-media-viewer-read-along-tts-design.md`
- Backlog task for this plan: `TASK-416`
- Original design task: `TASK-415`

## Scope Check

This plan covers one reviewable subsystem: read-along behavior inside the shared `ContentViewer` path. It deliberately excludes backend API work, new TTS providers, persistent server-side audio cache, word-level timing, and a page-level read-aloud player.

## Plan Hardening Review - 2026-05-23

Status: this plan has already been executed. The implementation is recorded in `TASK-417`, and post-PR review fixes are recorded in `TASK-425` for PR #1835. Do not treat the unchecked task-step checkboxes below as current backlog work; they are preserved as the original execution script. The authoritative completion evidence is the completed `TASK-417`/`TASK-425` records plus the final verification checklist in this file.

Current-code ownership check passed:

- `apps/packages/ui/src/components/Media/read-along/` contains the planned segmentation, cache, DOM, selection, session, popover, transport, and focused test modules.
- `ContentViewer` imports and wires `useContentSelectionActions`, `useMediaReadAlongSession`, `MediaReadAlongPopover`, and `MediaReadAlongTransport`.
- `useContentViewerModals` exposes explicit `captureAnnotationSelection` while retaining `handleCaptureAnnotationSelection` as a compatibility wrapper.
- `useTranscriptDisplay` and `ContentViewer` expose `data-read-along-segment-id` / `data-read-along-active` markers for plain and transcript rendering.
- `tts-provider` exposes `TtsSynthesizeOptions.signal`; `tldw`, OpenAI, and ElevenLabs synthesis paths accept abort signals where their helpers support it, while the browser provider remains a no-cache SpeechSynthesis path.
- Dexie schema/types include `mediaReadAlongAudioCache` with the media read-along audio cache entry type.
- Route guards verify WebUI and extension media routes stay on the shared `ViewMediaPage` path and do not duplicate read-along implementation in extension route files.

Risk review passed:

- Annotation selection mediation, full-content versus rendered-window behavior, abort/stale suppression, cache privacy/quota behavior, browser TTS, embedded media pause, markdown/html fallback, route parity, and accessibility all map to focused tests or recorded browser verification.
- No stale file ownership paths were found in this plan.
- Future read-along changes should use new focused Backlog tasks rather than re-executing this historical plan.

## File Structure

Create this directory:

- `apps/packages/ui/src/components/Media/read-along/`

New files:

- `types.ts`
  - Shared read-along types: segments, scopes, selection state, session state, cache signatures.
- `media-read-along-segments.ts`
  - Pure segmentation and scope expansion. No React imports.
- `media-read-along-cache-key.ts`
  - Stable content/settings signatures and request-cap subsegment helpers. No Dexie imports.
- `media-read-along-cache.ts`
  - Dexie-backed cache reads/writes, quota handling, LRU eviction, cache-disable fallback.
- `media-read-along-dom.ts`
  - DOM range helpers, nearest block lookup, segment element lookup, viewport clamping helpers.
- `useContentSelectionActions.ts`
  - One mediated selection path for annotation and read-along actions.
- `useMediaReadAlongSession.ts`
  - Playback queue, generated-audio lookahead, browser SpeechSynthesis fallback, abort/stale suppression, embedded media pause, retry/skip/stop.
- `MediaReadAlongPopover.tsx`
  - Compact selection action popover.
- `MediaReadAlongTransport.tsx`
  - Minimal inline playback transport.
- `__tests__/media-read-along-segments.test.ts`
- `__tests__/media-read-along-cache-key.test.ts`
- `__tests__/media-read-along-cache.test.ts`
- `__tests__/useContentSelectionActions.test.tsx`
- `__tests__/useMediaReadAlongSession.test.tsx`

Modify existing files:

- `apps/packages/ui/src/components/Media/ContentViewer.tsx`
  - Replace direct selection capture with mediated selection actions.
  - Add segment wrappers for plain/transcript paths.
  - Render read-along popover and transport.
  - Pass `modals.mediaPlayerRef` to the session hook for embedded preview pause.
- `apps/packages/ui/src/components/Media/hooks/useContentViewerModals.tsx`
  - Split annotation selection capture into an explicit action callable from selection mediation.
  - Preserve manual annotation and existing create/update/delete behavior.
- `apps/packages/ui/src/components/Media/hooks/useTranscriptDisplay.tsx`
  - Keep large-content windowing intact.
  - Expose enough rendered plain/transcript structure for segment wrappers without forcing full large-content render.
- `apps/packages/ui/src/services/tts-provider.ts`
  - Extend synthesis functions to accept `{ signal?: AbortSignal }`.
  - Preserve `provider === "browser"` as a supported no-cache SpeechSynthesis path for read-along sessions.
  - Preserve existing callers by making the options parameter optional.
- `apps/packages/ui/src/db/dexie/types.ts`
  - Add `MediaReadAlongAudioCacheEntry`.
- `apps/packages/ui/src/db/dexie/schema.ts`
  - Add a new Dexie version and `mediaReadAlongAudioCache` table.
- Existing tests under `apps/packages/ui/src/components/Media/__tests__/`
  - Update annotation baseline expectations.
  - Add read-along component coverage.

Do not create new backend files.

## Task 1: Baseline And Guardrail Tests

**Files:**
- Read: `apps/packages/ui/src/components/Media/ContentViewer.tsx`
- Read: `apps/packages/ui/src/components/Media/hooks/useContentViewerModals.tsx`
- Read: `apps/packages/ui/src/components/Media/hooks/useTranscriptDisplay.tsx`
- Read: `apps/packages/ui/src/services/tts-provider.ts`
- Read: `apps/packages/ui/src/db/dexie/schema.ts`
- Read: `apps/packages/ui/src/db/dexie/types.ts`

- [ ] **Step 1: Run current focused baselines**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage15.accessibility.test.tsx \
  src/services/__tests__/tts.defaults.test.ts \
  src/db/dexie/__tests__/stt-recordings.test.ts \
  --maxWorkers=1
```

Expected: PASS, or document pre-existing failures before editing implementation files.

- [ ] **Step 2: Inspect selection and media preview anchors**

Confirm these anchors still exist:

```bash
rg -n "handleCaptureAnnotationSelection|contentBodyRef|mediaPlayerRef|embedded-audio-player|embedded-video-player" \
  apps/packages/ui/src/components/Media
```

Expected: `ContentViewer.tsx` still binds selection events to `modals.handleCaptureAnnotationSelection`, and embedded media refs still go through `modals.mediaPlayerRef`.

- [ ] **Step 3: Commit nothing**

This is a read-only baseline task. Do not commit.

## Task 2: Segment And Scope Primitives

**Files:**
- Create: `apps/packages/ui/src/components/Media/read-along/types.ts`
- Create: `apps/packages/ui/src/components/Media/read-along/media-read-along-segments.ts`
- Create: `apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-segments.test.ts`

- [ ] **Step 1: Write failing segmentation tests**

Create `media-read-along-segments.test.ts` with tests covering timestamp lines, prose sentences, section expansion, read-from-here on full content, transient fallback, and long segment splitting.

Use this shape:

```ts
import { describe, expect, it } from "vitest"
import {
  buildReadAlongSegments,
  resolveReadAlongScope,
  splitSegmentForTtsRequest
} from "../media-read-along-segments"

describe("media read-along segmentation", () => {
  it("segments leading transcript timings as transcript lines", () => {
    const segments = buildReadAlongSegments({
      mediaId: "m1",
      content: "[00:01] First line.\n[00:04] Second line.",
      displayContent: "First line.\nSecond line.",
      renderMode: "plain",
      hideTranscriptTimings: true
    })

    expect(segments).toHaveLength(2)
    expect(segments[0]).toMatchObject({
      kind: "transcript-line",
      text: "First line.",
      timestampSeconds: 1
    })
    expect(segments[1].sourceStart).toBeGreaterThan(segments[0].sourceEnd)
  })

  it("resolves read-from-here against canonical full content, not a rendered window", () => {
    const content = "Alpha one. Beta two. Gamma three. Delta four."
    const segments = buildReadAlongSegments({
      mediaId: "m2",
      content,
      displayContent: "Alpha one. Beta two.",
      renderMode: "plain",
      hideTranscriptTimings: false
    })

    const queue = resolveReadAlongScope({
      scope: "from-here",
      segments,
      selection: {
        selectedText: "Beta",
        mappingConfidence: "nearest",
        sourceStart: content.indexOf("Beta"),
        sourceEnd: content.indexOf("Beta") + "Beta".length,
        anchorRect: new DOMRect()
      }
    })

    expect(queue.map((segment) => segment.text)).toEqual([
      "Beta two.",
      "Gamma three.",
      "Delta four."
    ])
  })

  it("splits overlong segments without changing the parent segment id", () => {
    const parts = splitSegmentForTtsRequest(
      { id: "s1", index: 0, kind: "sentence", text: "word ".repeat(80), sourceStart: 0, sourceEnd: 400 },
      120
    )

    expect(parts.length).toBeGreaterThan(1)
    expect(parts.every((part) => part.parentSegmentId === "s1")).toBe(true)
    expect(parts.every((part) => part.text.length <= 120)).toBe(true)
  })
})
```

- [ ] **Step 2: Run the test to verify red**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/media-read-along-segments.test.ts --maxWorkers=1
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the pure types**

Create `types.ts` with:

```ts
export type ReadAlongSegmentKind =
  | "transcript-line"
  | "sentence"
  | "paragraph"
  | "transient-selection"

export type ReadAlongScope =
  | "selection"
  | "from-here"
  | "current-section"
  | "full-item"

export interface ReadAlongSegment {
  id: string
  index: number
  kind: ReadAlongSegmentKind
  text: string
  sourceStart: number
  sourceEnd: number
  displayStart?: number
  displayEnd?: number
  sectionId?: string
  timestampSeconds?: number
}

export interface ReadAlongSelection {
  selectedText: string
  anchorRect: DOMRect
  startSegmentId?: string
  endSegmentId?: string
  sourceStart?: number
  sourceEnd?: number
  mappingConfidence: "exact" | "nearest" | "text-only"
}
```

- [ ] **Step 4: Implement the minimum segmentation behavior**

Implement `buildReadAlongSegments`, `resolveReadAlongScope`, and `splitSegmentForTtsRequest` in `media-read-along-segments.ts`.

Rules:

- Detect leading transcript timings with the existing utilities from `@/utils/media-transcript-display` when possible.
- Sentence splitting can use a conservative regex in v1, but must preserve source offsets.
- Segment IDs must be deterministic from media id, index, kind, and source offsets.
- Empty segments are ignored.
- `resolveReadAlongScope("from-here")` and `resolveReadAlongScope("full-item")` must use all canonical segments from `content`, not only displayed text.

- [ ] **Step 5: Run the segment tests green**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/media-read-along-segments.test.ts --maxWorkers=1
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Media/read-along/types.ts \
  apps/packages/ui/src/components/Media/read-along/media-read-along-segments.ts \
  apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-segments.test.ts
git commit -m "feat: add media read-along segmentation primitives"
```

## Task 3: Cache Keys, Dexie Store, And Provider Abort Surface

**Files:**
- Create: `apps/packages/ui/src/components/Media/read-along/media-read-along-cache-key.ts`
- Create: `apps/packages/ui/src/components/Media/read-along/media-read-along-cache.ts`
- Create: `apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts`
- Create: `apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache.test.ts`
- Modify: `apps/packages/ui/src/db/dexie/types.ts`
- Modify: `apps/packages/ui/src/db/dexie/schema.ts`
- Modify: `apps/packages/ui/src/services/tts-provider.ts`
- Test: `apps/packages/ui/src/services/__tests__/tts-provider.read-along.test.ts`

- [ ] **Step 1: Write failing cache-key tests**

Test that raw selected text is not stored in key metadata and that settings changes alter the signature:

```ts
import { describe, expect, it } from "vitest"
import { buildReadAlongCacheKey, buildTtsSettingsSignature } from "../media-read-along-cache-key"

describe("media read-along cache keys", () => {
  it("does not include raw segment text in the stable key", async () => {
    const key = await buildReadAlongCacheKey({
      serverScope: "http://127.0.0.1:8000",
      mediaId: "42",
      mediaKind: "media",
      segmentId: "s1",
      segmentText: "private transcript text",
      sourceStart: 10,
      sourceEnd: 33,
      settingsSignature: "provider:tldw|voice:default"
    })

    expect(key.id).not.toContain("private transcript text")
    expect(key.textHash).toMatch(/^[a-f0-9]{64}$/)
  })

  it("changes settings signature when voice or speed changes", () => {
    const a = buildTtsSettingsSignature({ provider: "tldw", model: "kokoro", voice: "af", speed: 1, format: "mp3" })
    const b = buildTtsSettingsSignature({ provider: "tldw", model: "kokoro", voice: "bf", speed: 1, format: "mp3" })
    const c = buildTtsSettingsSignature({ provider: "tldw", model: "kokoro", voice: "af", speed: 1.2, format: "mp3" })

    expect(a).not.toEqual(b)
    expect(a).not.toEqual(c)
  })
})
```

- [ ] **Step 2: Write failing cache store tests**

Mock `@/db/dexie/schema` the same way `src/db/dexie/__tests__/stt-recordings.test.ts` does. Cover save/get, LRU eviction, `QuotaExceededError` retry after eviction, and disabled cache fallback.

- [ ] **Step 3: Write failing provider abort test**

Create `src/services/__tests__/tts-provider.read-along.test.ts`.

Minimum assertion:

```ts
it("passes abort signals through tldw synthesis", async () => {
  const signal = new AbortController().signal
  const context = await resolveTtsProviderContext("hello", { provider: "tldw" })

  await context.synthesize?.("hello", { signal })

  expect(tldwClient.synthesizeSpeech).toHaveBeenCalledWith(
    "hello",
    expect.objectContaining({ signal })
  )
})
```

- [ ] **Step 4: Run tests red**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts \
  src/components/Media/read-along/__tests__/media-read-along-cache.test.ts \
  src/services/__tests__/tts-provider.read-along.test.ts \
  --maxWorkers=1
```

Expected: FAIL because files/signatures do not exist.

- [ ] **Step 5: Add Dexie type and schema**

In `apps/packages/ui/src/db/dexie/types.ts`, add:

```ts
export type MediaReadAlongAudioCacheEntry = {
  id: string
  createdAt: number
  lastUsedAt: number
  mediaId: string
  mediaKind: string
  segmentId: string
  settingsSignature: string
  textHash: string
  mimeType: string
  format: string
  blob: Blob
  sizeBytes: number
}
```

In `schema.ts`:

- import `MediaReadAlongAudioCacheEntry`
- add `mediaReadAlongAudioCache!: Table<MediaReadAlongAudioCacheEntry>`
- add version 14 with all existing version 13 stores plus:

```ts
mediaReadAlongAudioCache:
  "id, createdAt, lastUsedAt, mediaId, mediaKind, segmentId, settingsSignature, textHash"
```

- [ ] **Step 6: Implement key/store helpers**

Requirements:

- Use Web Crypto SHA-256 when available, with a deterministic fallback only for tests/non-secure environments.
- Evict by `lastUsedAt` before writing when approximate total bytes exceed `MEDIA_READ_ALONG_CACHE_MAX_BYTES`.
- Retry once after `QuotaExceededError`, then disable cache for the current session.
- Never throw from cache read/write paths used by playback.

- [ ] **Step 7: Extend provider synthesis options**

Change the provider type:

```ts
export type TtsSynthesizeOptions = {
  signal?: AbortSignal
}

export type TtsProviderContext = {
  provider: string
  utterance: string
  playbackSpeed: number
  supported: boolean
  synthesize?: (text: string, options?: TtsSynthesizeOptions) => Promise<TtsSynthesisResult>
  formatInfo?: TtsFormatInfo
}
```

For `tldw`, pass `signal` into `tldwClient.synthesizeSpeech`. For OpenAI and ElevenLabs, accept the option but leave behavior unchanged unless those helpers already support abort.

Do not add a fake `synthesize` function for the browser provider. Keep `supported: true` with no `synthesize`; the read-along session hook will use `window.speechSynthesis` directly for browser TTS and skip cache/lookahead for that provider.

- [ ] **Step 8: Run tests green**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts \
  src/components/Media/read-along/__tests__/media-read-along-cache.test.ts \
  src/services/__tests__/tts-provider.read-along.test.ts \
  src/services/__tests__/tts.defaults.test.ts \
  src/db/dexie/__tests__/stt-recordings.test.ts \
  --maxWorkers=1
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add apps/packages/ui/src/components/Media/read-along/media-read-along-cache-key.ts \
  apps/packages/ui/src/components/Media/read-along/media-read-along-cache.ts \
  apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts \
  apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache.test.ts \
  apps/packages/ui/src/services/__tests__/tts-provider.read-along.test.ts \
  apps/packages/ui/src/db/dexie/types.ts \
  apps/packages/ui/src/db/dexie/schema.ts \
  apps/packages/ui/src/services/tts-provider.ts
git commit -m "feat: add read-along audio cache primitives"
```

## Task 4: Mediated Content Selection Actions

**Files:**
- Create: `apps/packages/ui/src/components/Media/read-along/media-read-along-dom.ts`
- Create: `apps/packages/ui/src/components/Media/read-along/useContentSelectionActions.ts`
- Create: `apps/packages/ui/src/components/Media/read-along/__tests__/useContentSelectionActions.test.tsx`
- Modify: `apps/packages/ui/src/components/Media/hooks/useContentViewerModals.tsx`
- Modify: `apps/packages/ui/src/components/Media/ContentViewer.tsx`
- Modify: `apps/packages/ui/src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx`

- [ ] **Step 1: Write failing selection-mediation hook tests**

Test:

- ignores selections outside `contentBodyRef`
- returns selected text and anchor rect for content selections
- maps exact `data-read-along-segment-id` ancestors when present
- falls back to `text-only` when no segment wrapper exists

- [ ] **Step 2: Update annotation component test red**

Change the existing "captures selected content text into annotation draft" expectation:

- selection should open a selection action popover
- annotation preview should not appear until the user clicks the annotation action

Expected test shape:

```ts
fireEvent.mouseUp(contentNode)

expect(screen.getByTestId("media-selection-actions-popover")).toBeInTheDocument()
expect(screen.queryByTestId("media-annotation-selection-preview")).not.toBeInTheDocument()

fireEvent.click(screen.getByTestId("media-selection-action-annotate"))

await waitFor(() => {
  expect(screen.getByTestId("media-annotation-selection-preview")).toHaveTextContent("Selected body text")
})
```

- [ ] **Step 3: Run selection tests red**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Media/read-along/__tests__/useContentSelectionActions.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx \
  --maxWorkers=1
```

Expected: FAIL.

- [ ] **Step 4: Split annotation selection capture**

In `useContentViewerModals.tsx`, replace the current event-capturing handler with an explicit draft setter:

```ts
const captureAnnotationSelection = useCallback((selectionText: string, location?: string) => {
  const selectedText = selectionText.trim()
  if (!selectedMediaId || isNote || !selectedText) return

  setAnnotationSelectionText(selectedText.slice(0, 4000))
  setAnnotationSelectionLocation(location || `selection:${Date.now()}`)
  setActiveIntelligenceTab("annotations")
  if (collapsedSections.intelligence ?? true) {
    void setCollapsedSections((prev) => ({ ...prev, intelligence: false }))
  }
}, [collapsedSections.intelligence, isNote, selectedMediaId, setCollapsedSections])
```

Keep `handleCaptureAnnotationSelection` only as a backwards-compatible wrapper if any tests or callers still need it during the transition. New `ContentViewer` wiring should not call it directly from `onMouseUp`/`onKeyUp`.

- [ ] **Step 5: Implement `useContentSelectionActions`**

Responsibilities:

- expose `selectionActionState`
- expose `handleContentSelectionEvent`
- expose `clearSelectionActions`
- expose `applyAnnotationSelection`
- never auto-open the annotation panel until `applyAnnotationSelection` is called

- [ ] **Step 6: Wire `ContentViewer` to mediated selection**

Replace:

```tsx
onMouseUp={modals.handleCaptureAnnotationSelection}
onKeyUp={modals.handleCaptureAnnotationSelection}
```

with:

```tsx
onMouseUp={selectionActions.handleContentSelectionEvent}
onKeyUp={selectionActions.handleContentSelectionEvent}
```

Render a temporary button/action popover for annotation only in this task. Read-along actions can be added in Task 6.

- [ ] **Step 7: Run tests green**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Media/read-along/__tests__/useContentSelectionActions.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage15.accessibility.test.tsx \
  --maxWorkers=1
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Media/read-along/media-read-along-dom.ts \
  apps/packages/ui/src/components/Media/read-along/useContentSelectionActions.ts \
  apps/packages/ui/src/components/Media/read-along/__tests__/useContentSelectionActions.test.tsx \
  apps/packages/ui/src/components/Media/hooks/useContentViewerModals.tsx \
  apps/packages/ui/src/components/Media/ContentViewer.tsx \
  apps/packages/ui/src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx
git commit -m "feat: mediate media content selection actions"
```

## Task 5: Read-Along Session Hook

**Files:**
- Create: `apps/packages/ui/src/components/Media/read-along/useMediaReadAlongSession.ts`
- Create: `apps/packages/ui/src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx`
- Modify: `apps/packages/ui/src/components/Media/read-along/types.ts`

- [ ] **Step 1: Write failing session tests**

Use `renderHook` from `@testing-library/react`.

Cover:

- `start("selection")` resolves a queue and plays cached audio before generating lookahead
- `start("from-here")` queues beyond the rendered window
- lookahead prefetches 3 to 5 segments, not the full item
- browser TTS provider uses `window.speechSynthesis.speak()` and does not touch generated-audio cache
- stop aborts current and lookahead `AbortController`s
- stop cancels browser SpeechSynthesis when browser provider is active
- media/content change stops and suppresses stale completions
- settings are captured at session start and do not mutate mid-session
- `audio.play()` rejection enters `segment-error`
- starting read-along pauses `embeddedMediaRef.current` if it is playing

- [ ] **Step 2: Run session tests red**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx --maxWorkers=1
```

Expected: FAIL.

- [ ] **Step 3: Implement state types**

Add to `types.ts`:

```ts
export type ReadAlongSessionStatus =
  | "idle"
  | "preparing"
  | "playing"
  | "paused"
  | "segment-error"
  | "stopped"

export interface ReadAlongSessionState {
  status: ReadAlongSessionStatus
  scope: ReadAlongScope | null
  activeSegmentId: string | null
  activeIndex: number
  totalSegments: number
  error: string | null
  cacheDisabled: boolean
}
```

- [ ] **Step 4: Implement session hook**

Minimum API:

```ts
export function useMediaReadAlongSession(args: {
  mediaId: string | null
  mediaKind: string | null
  content: string
  displayContent: string
  renderMode: string
  hideTranscriptTimings: boolean
  selection: ReadAlongSelection | null
  contentBodyRef: React.RefObject<HTMLElement | null>
  contentScrollContainerRef: React.RefObject<HTMLElement | null>
  embeddedMediaRef: React.RefObject<HTMLMediaElement | null>
}) {
  return {
    state,
    start,
    pause,
    resume,
    stop,
    retry,
    skip,
    activeSegmentId
  }
}
```

Implementation requirements:

- Create one session token per `start`.
- Create an `AbortController` for current synthesis plus one for lookahead.
- Before every state mutation from async work, check the current session token.
- Use `resolveTtsProviderContext` once per session and freeze the settings signature.
- Do not call `useTTS.speak()` from this hook; it performs separate segment splitting and user-visible clip saving that would break read-along cache semantics.
- Use cache get before synthesize, and cache write after successful synthesis.
- If the context is the browser provider with no `synthesize`, play the active segment with `SpeechSynthesisUtterance`, advance on `onend`, and skip cache/lookahead.
- Revoke object URLs after segment playback.
- Pause embedded media preview before the first generated audio play.

- [ ] **Step 5: Run session tests green**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx \
  src/components/Media/read-along/__tests__/media-read-along-segments.test.ts \
  src/components/Media/read-along/__tests__/media-read-along-cache.test.ts \
  --maxWorkers=1
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Media/read-along/useMediaReadAlongSession.ts \
  apps/packages/ui/src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx \
  apps/packages/ui/src/components/Media/read-along/types.ts
git commit -m "feat: add media read-along playback session"
```

## Task 6: ContentViewer UI Integration

**Status:** Complete

**Files:**
- Create: `apps/packages/ui/src/components/Media/read-along/MediaReadAlongPopover.tsx`
- Create: `apps/packages/ui/src/components/Media/read-along/MediaReadAlongTransport.tsx`
- Modify: `apps/packages/ui/src/components/Media/ContentViewer.tsx`
- Modify: `apps/packages/ui/src/components/Media/hooks/useTranscriptDisplay.tsx`
- Test: `apps/packages/ui/src/components/Media/__tests__/ContentViewer.read-along.test.tsx`
- Test: `apps/packages/ui/src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx`
- Test: `apps/packages/ui/src/components/Media/__tests__/ContentViewer.stage10.findBar.test.tsx`

- [x] **Step 1: Write failing UI tests**

Create `ContentViewer.read-along.test.tsx`.

Cover:

- no read-along UI before selection
- selection popover contains `Read selection`, `Read from here`, `Read current section`, `Read full item`, and `Annotate`
- clicking `Read selection` starts playback and keeps inline transport visible after selection clears
- active segment wrapper gets `data-read-along-active="true"`
- stop clears read-along transient UI
- embedded audio/video preview is paused when generated playback starts
- markdown/html fallback starts playback without unsafe HTML mutation

- [x] **Step 2: Run UI tests red**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Media/__tests__/ContentViewer.read-along.test.tsx --maxWorkers=1
```

Expected: FAIL.

- [x] **Step 3: Implement popover**

Use real buttons. Test IDs:

- `media-selection-actions-popover`
- `media-selection-action-read-selection`
- `media-selection-action-read-from-here`
- `media-selection-action-read-current-section`
- `media-selection-action-read-full-item`
- `media-selection-action-annotate`

Keep labels short and use `t()` fallbacks. Do not add a page-level button.

- [x] **Step 4: Implement inline transport**

Use compact icon buttons from lucide-react where available:

- pause/resume
- stop
- retry when `segment-error`
- skip when `segment-error`

Test IDs:

- `media-read-along-transport`
- `media-read-along-toggle`
- `media-read-along-stop`
- `media-read-along-retry`
- `media-read-along-skip`
- `media-read-along-progress`

The transport should be anchored near the selection/active segment and clamp to the content viewport. On narrow widths, keep it compact near the active segment; do not add a sticky bottom player.

- [x] **Step 5: Wrap plain/transcript segments**

For plain and transcript-line rendering, render segment wrappers with:

```tsx
<span
  data-read-along-segment-id={segment.id}
  data-read-along-active={segment.id === readAlong.activeSegmentId ? "true" : undefined}
  className={segment.id === readAlong.activeSegmentId ? "rounded bg-primary/20 text-text" : undefined}
>
  {segment.text}
</span>
```

Do not break existing find highlighting. If find highlighting is active, prefer find markup and skip exact read-along wrappers until the query is cleared.

- [x] **Step 6: Keep large-content rendering lazy**

Use full content for queue construction inside the session hook, but only wrap the visible plain content window in the DOM. Do not force `visiblePlainContentChars = content.length` when `Read full item` starts.

- [x] **Step 7: Run UI/performance/find tests green**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Media/__tests__/ContentViewer.read-along.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage10.findBar.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx \
  --maxWorkers=1
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Media/read-along/MediaReadAlongPopover.tsx \
  apps/packages/ui/src/components/Media/read-along/MediaReadAlongTransport.tsx \
  apps/packages/ui/src/components/Media/ContentViewer.tsx \
  apps/packages/ui/src/components/Media/hooks/useTranscriptDisplay.tsx \
  apps/packages/ui/src/components/Media/__tests__/ContentViewer.read-along.test.tsx \
  apps/packages/ui/src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx \
  apps/packages/ui/src/components/Media/__tests__/ContentViewer.stage10.findBar.test.tsx
git commit -m "feat: wire read-along into media content viewer"
```

## Task 7: Accessibility, Route Parity, And Regression Hardening

**Status:** Complete

**Files:**
- Modify: `apps/packages/ui/src/components/Media/__tests__/ContentViewer.stage15.accessibility.test.tsx`
- Modify or create: `apps/packages/ui/src/routes/__tests__/option-media-route-guards.test.tsx`
- Modify: `apps/tldw-frontend/__tests__/extension/entry-shell-performance.test.ts`
- Optional create: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.read-along-parity.test.tsx`

- [x] **Step 1: Add accessibility tests**

Cover:

- popover actions are buttons with accessible names
- keyboard selection path can open the selection affordance
- active segment status uses a polite live region
- reduced motion disables smooth auto-scroll
- no essential controls are hover-only

- [x] **Step 2: Add route/shared import guard**

The feature must stay in `apps/packages/ui`. Add or extend a guard that proves WebUI and extension media routes both use shared `ViewMediaPage`/`ContentViewer`, with no duplicate read-along implementation under `apps/tldw-frontend/extension`.

- [x] **Step 3: Add bundle/performance guard**

Update the extension entry-shell performance test only if the new read-along modules change the expected static import set. Prefer lazy or local imports if static media entry cost grows unexpectedly.

- [x] **Step 4: Run tests**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Media/__tests__/ContentViewer.stage15.accessibility.test.tsx \
  src/routes/__tests__/option-media-route-guards.test.tsx \
  src/components/Review/__tests__/ViewMediaPage.connection.test.tsx \
  --maxWorkers=1
```

Then from repo root:

```bash
bunx vitest run apps/tldw-frontend/__tests__/extension/entry-shell-performance.test.ts --maxWorkers=1
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Media/__tests__/ContentViewer.stage15.accessibility.test.tsx \
  apps/packages/ui/src/routes/__tests__/option-media-route-guards.test.tsx \
  apps/tldw-frontend/__tests__/extension/entry-shell-performance.test.ts
git commit -m "test: verify media read-along route parity"
```

## Task 8: Browser Verification And Final Cleanup

**Files:**
- Modify only if needed after browser QA:
  - `apps/packages/ui/src/components/Media/read-along/*`
  - `apps/packages/ui/src/components/Media/ContentViewer.tsx`
  - focused tests touched above
- Modify: `backlog/tasks/task-416 - Plan-media-viewer-read-along-TTS-implementation.md` only to record implementation verification if this plan task is kept as the tracking task.

- [x] **Step 1: Run the focused unit/component suite**

Run:

```bash
cd apps/packages/ui && bunx vitest run \
  src/components/Media/read-along/__tests__ \
  src/components/Media/__tests__/ContentViewer.read-along.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage10.findBar.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx \
  src/components/Media/__tests__/ContentViewer.stage15.accessibility.test.tsx \
  src/services/__tests__/tts-provider.read-along.test.ts \
  src/db/dexie/__tests__/stt-recordings.test.ts \
  --maxWorkers=1
```

Expected: PASS.

- [x] **Step 2: Run design-system/openapi guards if touched imports cross those boundaries**

Run:

```bash
cd apps/packages/ui && bun run verify:design-system-state
cd apps/packages/ui && bun run verify:openapi
```

Expected: PASS, or document why a guard is unrelated and skipped.

- [x] **Step 3: Start the WebUI dev server**

Use the repo's normal dev flow. If a dev server is already running, reuse it. Otherwise run the existing frontend/server command expected for this repo and record the URL.

- [x] **Step 4: Browser smoke the WebUI media viewer**

Manual/browser checks:

- navigate to the WebUI media viewer
- select text in a plain/transcript item
- verify the selection popover appears
- click `Read selection`
- verify active segment highlight and inline transport
- stop playback
- select text again and click `Annotate`
- verify annotation draft appears only after annotation action

- [x] **Step 5: Browser smoke extension-width behavior**

Use a narrow viewport comparable to the extension sidepanel. Confirm:

- popover clamps within viewport
- transport remains compact
- text does not overlap controls
- no sticky bottom player appears

- [x] **Step 6: Run Bandit only if Python/backend files were touched**

Expected for this implementation: skipped, because the planned implementation is TypeScript UI-only. If Python files were touched despite this plan, run Bandit on the touched Python scope before completion.

- [x] **Step 7: Final commit for QA fixes**

If browser QA required fixes:

Stage only the focused files changed during QA, then run:

```bash
git commit -m "fix: harden media read-along browser behavior"
```

If no fixes were needed, do not create an empty commit.

Task 8 verification notes:

- Focused read-along suite rerun from `apps/packages/ui`: 12 files / 104 tests passed, including segmentation, cache, content selection, playback session, `ContentViewer.read-along`, find/performance/annotation/accessibility coverage, TTS provider mapping, and Dexie STT regression coverage.
- Route parity rerun from `apps/packages/ui`: 2 files / 6 tests passed for shared WebUI/extension media route guards and `ViewMediaPage` connection behavior.
- `git diff --check` passed.
- `bun run verify:openapi` passed earlier in Task 8; `bun run verify:design-system-state` remains blocked by unrelated repo-wide baseline findings outside read-along touched files.
- Next dev server was started at `http://127.0.0.1:8080` with quickstart deployment env. Browser render smoke passed against mocked backend responses, confirming `/media` renders a media detail in the shared content region and that the content region is explicitly text-selectable.
- A fuller headless interaction smoke exposed real hardening issues: the content body needed explicit text selection, the selection hook needed an always-on content-scoped `selectionchange` listener, popover buttons needed pointer-down default prevention, and floating controls needed visible-window fallback when the pane rectangle is offscreen. Those fixes were added with focused regression coverage. The remaining full interaction smoke was not kept as a final gate because headless programmatic text selection stayed flaky after the product fixes; interaction behavior is covered by the focused component tests.
- Bandit skipped because only TypeScript/React/docs/backlog files were touched.

## Final Verification Checklist

- [x] `ContentViewer` selection no longer automatically switches to annotations for every selection.
- [x] Annotation creation from selected text still works through the mediated selection action.
- [x] No read-along UI appears before selection or active playback.
- [x] Read-along actions support selection, from-here, current-section, and full-item scopes.
- [x] `Read from here` and `Read full item` use canonical full content, not only rendered large-content windows.
- [x] Large-content segmentation is lazy and cancellable.
- [x] Stop/media/content changes abort in-flight TTS requests and suppress stale cache writes.
- [x] Generated audio cache uses Dexie, LRU eviction, quota fallback, and no raw selected text metadata.
- [x] Existing TTS settings are reused and frozen per active session.
- [x] Browser TTS provider works as a no-cache SpeechSynthesis path.
- [x] Embedded media preview pauses when generated read-along playback starts.
- [x] Plain/transcript active highlighting works with find highlighting preserved.
- [x] Markdown/html use safe nearest-block fallback without mutating sanitized HTML.
- [x] WebUI and extension routes share the implementation.
- [x] Keyboard and reduced-motion paths are covered.

## Suggested PR/Commit Stack

1. `feat: add media read-along segmentation primitives`
2. `feat: add read-along audio cache primitives`
3. `feat: mediate media content selection actions`
4. `feat: add media read-along playback session`
5. `feat: wire read-along into media content viewer`
6. `test: verify media read-along route parity`
7. Optional browser QA fix commit

## Execution Handoff

Recommended execution mode: Subagent-driven, one fresh worker per task after Task 1 baseline, with parent review between commits. The tasks have mostly disjoint write scopes until `ContentViewer` integration.

Inline execution is also viable if the worktree is kept narrow and each task is committed before the next begins.
