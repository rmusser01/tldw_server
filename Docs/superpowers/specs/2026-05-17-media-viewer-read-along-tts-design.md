# Media Viewer Read-Along TTS Design

- Date: 2026-05-17
- Project: tldw_server
- Backlog: TASK-415
- Topic: Selection-initiated read-along in the shared media viewer
- Mode: Design for implementation

## 1. Objective

Add read-along functionality to the shared media viewer used by the WebUI and browser extension. A user should be able to select text in a media item and have that selection, the surrounding section, the rest of the item, or the full item read aloud through the existing TTS stack.

The v1 experience should stay content-native:

- no persistent page-level read-aloud player
- no standalone "read whole item" button
- no UI until the user selects text
- active text is highlighted during playback
- playback auto-scrolls only when the active segment leaves view
- voice, model, provider, format, and speed defaults come from existing TTS settings

The design is inspired by read-along patterns in OpenReader, especially segment-aware narration, active highlight, and local reuse of generated audio:

- https://github.com/richardr1126/openreader
- https://deepwiki.com/richardr1126/OpenReader-WebUI

## 2. Scope

### In Scope

- selection-first popover in `ContentViewer`
- actions for `Read selection`, `Read from here`, `Read current section`, and `Read full item`
- hybrid segmentation:
  - timestamped transcript lines when transcript timing lines exist
  - sentence segments for regular prose, markdown, and rich HTML
- active segment highlighting and viewport-aware auto-scroll
- chunked lookahead generation for long read sessions
- browser-local audio cache keyed by content and TTS settings
- reuse of existing TTS provider resolution and `/api/v1/audio/speech` client behavior
- shared WebUI and extension support through `apps/packages/ui`
- focused tests for segmentation, selection mapping, playback session state, cache behavior, and shared route parity

### Out Of Scope

- backend-persisted read-along audio cache
- generated TTS synchronization with embedded original audio or video
- word-level karaoke timing
- a persistent page-level audio player
- new backend TTS APIs
- new provider or voice-management behavior
- replacing the existing TTS playground or document workspace TTS panel

## 3. Current Repo Context

The feature should be implemented in the shared UI package, not separately in the Next app and extension.

Relevant shared surfaces:

- `apps/packages/ui/src/components/Review/ViewMediaPage.tsx`
  - shared media page shell
  - used by WebUI and extension routes
- `apps/packages/ui/src/components/Media/ContentViewer.tsx`
  - selected media detail viewer
  - owns content rendering, metadata controls, find, embedded media preview, and action menu
- `apps/packages/ui/src/components/Media/hooks/useContentRendering.tsx`
  - normalizes display content, render mode, transcript timing visibility, markdown/html/plain behavior
- `apps/packages/ui/src/components/Media/hooks/useTranscriptDisplay.tsx`
  - plain/transcript rendering helpers, find highlighting, large-content windowing, timestamp seeking
- `apps/packages/ui/src/components/Media/hooks/useReadingProgress.tsx`
  - scroll/progress restoration and navigation-target application
- `apps/packages/ui/src/hooks/useTTS.tsx`
  - existing chat/message TTS hook using provider resolution, segment splitting, and clip saving
- `apps/packages/ui/src/hooks/document-workspace/useDocumentTTS.ts`
  - existing document TTS playback hook
- `apps/packages/ui/src/services/tts-provider.ts`
  - provider context resolution and synthesis function construction
- `apps/packages/ui/src/services/tts.ts`
  - existing global TTS settings
- `apps/packages/ui/src/services/tldw/audio-voices.ts`
  - server voice catalog fetch
- `apps/packages/ui/src/db/dexie/tts-clips.ts`
  - user-visible saved TTS clip history; read-along cache should not reuse this table directly

## 4. Approved Decisions

1. V1 should combine basic read aloud with guided read-along.
2. The read unit should be hybrid:
   - transcript timestamp line for transcript-like content
   - sentence segment for regular content
3. The media viewer should use existing global TTS settings with compact controls only while reading.
4. Specific selections should support both transient text selection and segment/range expansion.
5. Generated audio should use a browser-local cache.
6. Long full-item reads should use chunked lookahead, not all-at-once generation.
7. None of the initial persistent placement options were acceptable:
   - header controls
   - sticky bottom player
   - right-side drawer
8. Read-along UI should be selection-initiated only.
9. When no text is selected and no read-along session is active, no read-along UI is visible.
10. Selecting text opens the read-along popover, including `Read full item` as an explicit scope expansion.
11. Once playback starts, the inline transport remains visible until the session is stopped, even if the original text selection is cleared.
12. `Read full item` segments from the canonical full media content already loaded in `ContentViewer`, not from the currently rendered large-content window.
13. Markdown and rich HTML use nearest-block highlighting/scroll fallback in v1 unless exact rendered text-node mapping is reliable.
14. Read-along generated audio is not saved into user-visible TTS clip history in v1.

## 5. Approaches Considered

### Recommended: Selection-First Popover

Read-along appears only after the user selects text in the content. The popover provides scope actions, and playback highlights the active segment in place.

Pros:

- keeps the normal media viewer clean
- keeps controls near the selected text
- supports quick selection and range expansion naturally
- avoids adding another persistent audio-player surface
- works in WebUI and extension because the interaction is inside the shared content component

Cons:

- discoverability depends on text selection
- needs careful accessibility treatment for keyboard users
- requires robust selection-to-segment mapping

### Alternative: Inline Segment Handles

Each sentence, transcript line, or paragraph exposes a subtle hover/focus read handle.

Pros:

- clear per-segment affordance
- direct read-from-here behavior

Cons:

- adds visible UI noise to dense content
- expensive to render for large documents
- awkward for markdown/rich HTML where generated layout may not map cleanly to source segments

### Alternative: Dedicated Read Mode

A user enters read mode, then content becomes explicitly segmented with range controls and a small active transport.

Pros:

- more discoverable once mode is active
- stronger keyboard and range workflow

Cons:

- adds a mode switch before the user can act
- feels heavier than the requested selection-first workflow
- risks becoming a separate reader UI rather than a media viewer enhancement

## 6. Interaction Design

### Entry

No read-along controls are visible by default.

When a user selects text within the content body, `ContentViewer` opens a small popover near the selection. The popover should include:

- `Read selection`
- `Read from here`
- `Read current section`
- `Read full item`

The popover should close when:

- selection is cleared
- user clicks outside
- media item changes
- playback is stopped and no selection remains
- the content is reloaded

### Scope Actions

`Read selection`:

- maps the selected text to intersecting segments
- if selection cannot map precisely, reads the selected text as a transient segment
- highlights only mapped segments when available

`Read from here`:

- maps the start of selection to the nearest segment
- queues from that segment through the end of the item using canonical full media content, not only the currently rendered large-content window

`Read current section`:

- expands to the nearest containing section:
  - transcript block if timestamped transcript
  - heading-delimited section for markdown/html where detectable
  - paragraph group fallback for plain prose

`Read full item`:

- queues segments from the canonical full media content using chunked lookahead
- does not limit itself to the currently rendered large-content window
- does not require a standalone page-level button

### Playback

After playback starts:

- active segment receives a visible highlight in the content
- completed segments may receive a subtle "read" state only if it does not add clutter
- the active segment scrolls into view only when outside the viewport
- a minimal inline transport appears near the active segment or previous selection anchor
- the inline transport remains visible while the session is preparing, playing, paused, or in a segment-error state

The inline transport should include:

- pause/resume
- stop
- current segment count, for example `4/22`
- cache/generation status only when useful, for example `Preparing...` or `Retry`

Voice, model, provider, response format, and default speed are read from existing TTS settings. V1 should avoid a provider picker in the media viewer.

## 7. Segmentation Design

Introduce a small utility module, for example:

- `apps/packages/ui/src/components/Media/read-along/media-read-along-segments.ts`

Core shape:

```ts
export type ReadAlongSegmentKind =
  | "transcript-line"
  | "sentence"
  | "paragraph"
  | "transient-selection"

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
```

Segmentation rules:

1. If content has leading transcript timings, segment by timestamped line.
2. If transcript timings are hidden in display mode, preserve mapping from original content offsets to displayed text offsets.
3. Otherwise segment prose into sentences.
4. Preserve paragraph and heading boundaries as section metadata for `Read current section`.
5. Keep segment IDs deterministic for a given media ID and content version.
6. Ignore empty or whitespace-only segments.
7. Cap pathological segment length; split very long segments by sentence or word boundary where possible.
8. For `Read full item`, build segments from the full source content string available to `ContentViewer`, even when the plain renderer is only showing a windowed subset.
9. If full source content is unavailable or cannot be segmented safely, disable `Read full item` for that item and explain that the user can read the selected or current rendered section instead.

The segmentation module should be independent of React so it can be unit tested heavily.

## 8. Selection Mapping

Introduce a hook, for example:

- `apps/packages/ui/src/components/Media/read-along/useMediaReadAlongSelection.ts`

Responsibilities:

- listen for selection changes scoped to `contentBodyRef`
- ignore selections outside the media content
- normalize selected text without losing source offsets
- map DOM ranges to segment IDs where rendered segment markers exist
- fall back to text search within display content when direct markers are unavailable
- emit a stable selection object for the popover

Selection shape:

```ts
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

The renderer should add `data-read-along-segment-id` wrappers for plain text and transcript-line modes. Markdown and rich HTML can start with nearest-block mapping and improve later.

## 9. Playback Session Design

Introduce a hook, for example:

- `apps/packages/ui/src/components/Media/read-along/useMediaReadAlongSession.ts`

Responsibilities:

- resolve selected action to a segment queue
- generate/read audio segment-by-segment
- prefetch a small lookahead window
- use local cache before calling TTS
- maintain active segment, queued scope, loading, paused, stopped, and error states
- expose commands for pause, resume, stop, retry failed segment, and skip failed segment
- stop playback on media/page/content change
- auto-scroll active segment only when needed

Recommended lookahead:

- start with current segment immediately
- prefetch 3 to 5 segments ahead
- for full-item reads, generate in batches of 10 to 20 queued segments
- never generate the entire item upfront

The session should treat audio generation and playback as separate phases so cached audio can play immediately while future segments generate.

## 10. Audio Generation And Cache

Reuse existing TTS provider resolution instead of calling `/api/v1/audio/speech` ad hoc.

Preferred implementation path:

- factor reusable synthesis/playback pieces out of `useTTS` or use `resolveTtsProviderContext()`
- keep media read-along-specific queue/session behavior in the new hook
- do not save read-along segments into user-visible TTS clip history in v1

Add a browser-local cache separate from `ttsClips`, for example:

- `apps/packages/ui/src/db/dexie/media-read-along-cache.ts`

Cache key should include:

- server scope/base URL or connection identity
- media ID
- media kind
- content hash or segment text hash
- segment source offsets
- provider
- model
- voice
- speed
- response format
- normalization/SSML-relevant settings if they affect generated audio

Cache record shape:

```ts
export interface MediaReadAlongAudioCacheEntry {
  id: string
  createdAt: number
  lastUsedAt: number
  mediaId: string
  segmentId: string
  settingsSignature: string
  textHash: string
  mimeType: string
  format: string
  blob: Blob
  sizeBytes: number
}
```

Eviction policy:

- cap total entries and approximate bytes globally for the read-along cache
- use a conservative v1 default of 200 entries and 250 MB approximate total cache size
- evict least-recently-used entries
- tolerate Dexie/private-window failure by disabling cache for the session

Cache failure must never block playback.

## 11. Rendering And Highlighting

### Plain And Transcript Rendering

Plain and transcript rendering should get first-class support in v1:

- wrap segments with `data-read-along-segment-id`
- apply active highlight class to current segment
- optionally apply a subtle completed state
- keep find highlighting compatible with read-along highlighting

Timestamp transcript lines should preserve existing timestamp seek buttons.

### Markdown And Rich HTML Rendering

Markdown and rich HTML can start with a conservative fallback:

- use selected text and nearest containing block for scroll
- highlight exact rendered segment only when mapping is reliable
- never mutate sanitized HTML with unsafe string replacement

Nearest-block highlighting/scroll fallback is acceptable for v1. Future hardening can add a render plugin or text-node walker that wraps exact text nodes after sanitization/rendering.

### Auto-Scroll

Auto-scroll rules:

- if the active segment is fully visible, do nothing
- if partly or fully outside the content viewport, scroll it into view
- respect reduced motion preferences
- avoid fighting user scrolling; if the user scrolls away manually, pause auto-follow until the next user action or segment boundary

## 12. Accessibility

Selection-first interaction must still be keyboard reachable.

Requirements:

- popover opens for keyboard text selection inside content
- popover actions are real buttons/menu items with accessible names
- transport buttons have labels and keyboard focus states
- active segment changes are announced politely, not on every word
- reduced-motion users get non-smooth scrolling
- color highlight must meet contrast against light and dark surfaces
- no essential control appears only on hover

The existing `content-selection-live-region` pattern can be extended for read-along status.

## 13. Error Handling

Expected states:

- idle
- selection-active
- preparing
- playing
- paused
- segment-error
- cache-disabled
- stopped

Behavior:

- TTS request failure marks the segment as failed and stops lookahead for that segment.
- Failed segment UI offers retry, skip, and stop.
- Unsupported response format falls back through existing TTS provider logic where possible.
- Browser cannot play generated MIME type: show a concise error and recommend changing TTS output format.
- Cache unavailable: show no blocking error; optionally expose "cache unavailable" in debug state only.
- Media/content changes stop playback and clear transient selection UI.
- TTS settings changes create a new cache signature; old cache entries remain for eviction.
- TTS settings changes during an active read-along session do not mutate the current queue; they apply to the next session or an explicit restart.

## 14. Rollout Plan

### Stage 1: Segmentation, Cache, And Session Primitives

Goal:

- add pure segmentation utilities
- add cache-key/cache-store helpers
- add session state tests without visible UI

Success criteria:

- transcript timing lines segment into transcript-line segments
- prose segments into sentence segments with section metadata
- cache keys change when relevant TTS settings change
- lookahead queue can play from cached and generated segments

### Stage 2: Selection Popover And Plain/Transcript Highlighting

Goal:

- add selection detection in `ContentViewer`
- add popover actions
- add active highlighting for plain and timestamped transcript rendering

Success criteria:

- selecting text opens read-along actions
- `Read selection`, `Read from here`, `Read current section`, and `Read full item` queue expected scopes
- active segment highlights and auto-scrolls correctly
- stop clears transient UI

### Stage 3: Markdown/HTML Fallback Mapping And Polish

Goal:

- support selection-first read-along in markdown and rich HTML render modes without unsafe DOM mutation
- add nearest-block scroll/highlight fallback where exact mapping is unavailable

Success criteria:

- playback works in markdown/html modes
- exact highlight is used only when reliable
- fallback behavior is understandable and non-disruptive

### Stage 4: Shared Route Parity And Accessibility

Goal:

- verify WebUI and extension shells
- harden keyboard and screen reader behavior

Success criteria:

- shared component tests pass
- one WebUI browser workflow passes
- one extension workflow passes
- keyboard selection path is usable
- reduced-motion and color contrast checks pass

## 15. Testing

Unit tests:

- timestamp-line segmentation
- sentence segmentation
- heading/paragraph section expansion
- selected text to segment mapping
- transient text-only selection fallback
- cache key construction
- cache LRU eviction
- queue lookahead behavior
- media/settings change invalidation

Component tests:

- popover appears only after valid content selection
- action menu queues the correct scope
- active segment highlight applies and clears
- pause/resume/stop update visible state
- TTS error exposes retry/skip/stop
- cache failure falls back to live generation
- find highlighting and read-along highlighting coexist

Browser/E2E tests:

- WebUI media viewer selection-to-read flow
- extension media viewer selection-to-read flow
- transcript-line flow with timestamped content
- long content full-item chunked generation does not request all segments upfront
- accessibility smoke for keyboard focus and status announcements

## 16. V1 Decisions Closed For Planning

These decisions should not be reopened by the first implementation plan unless new repo evidence makes them impossible:

- read-along entry remains selection-first
- no persistent page-level player
- active sessions keep a minimal inline transport visible until stopped
- `Read full item` uses canonical full source content, not the rendered window
- `Read from here` also continues through canonical full source content
- very large full-item reads use chunked lookahead after segmentation
- TTS settings changes apply to future sessions or explicit restarts, not mid-session mutation
- markdown/html exact inline highlighting is best effort; nearest-block fallback is acceptable
- cache eviction uses global LRU with default caps of 200 entries and 250 MB approximate total size
- generated read-along audio is not saved into user-visible TTS clip history in v1

## 17. Definition Of Done

- selection-initiated read-along works in shared `ContentViewer`
- WebUI and extension use the same implementation path
- no persistent page-level player is introduced
- existing TTS settings are reused
- transcript-line and sentence segmentation are covered by tests
- active segment highlighting works for plain/transcript content
- markdown/html have a safe fallback
- browser-local audio cache is opportunistic and separately evicted
- long reads use chunked lookahead
- error states include retry, skip, and stop
- route parity and accessibility checks are documented

## 18. Deferred Follow-Ups

These are not v1 blockers:

1. Expose read-along cache as clearable local data in settings.
2. Add an explicit user action to save a read-along segment or session into user-visible TTS clips.
3. Add exact markdown/rich HTML text-node highlighting after the fallback path is proven.
4. Add backend-persisted read-along cache if cross-device reuse becomes important.
5. Add word-level highlighting if provider timing metadata becomes available.
