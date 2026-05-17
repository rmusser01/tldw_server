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
- selection arbitration with the existing annotation capture flow in `ContentViewer`
- abortable synthesis/lookahead so stop, media changes, and stale sessions do not leak playback or cache writes
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
- `apps/packages/ui/src/components/Media/hooks/useContentViewerModals.tsx`
  - currently captures any content selection for annotations and switches the intelligence tab to annotations
  - read-along must mediate selection handling instead of adding a second independent selection listener that competes with annotations
- `apps/packages/ui/src/hooks/useTTS.tsx`
  - existing chat/message TTS hook using provider resolution, segment splitting, and clip saving
- `apps/packages/ui/src/hooks/document-workspace/useDocumentTTS.ts`
  - existing document TTS playback hook
- `apps/packages/ui/src/services/tts-provider.ts`
  - provider context resolution and synthesis function construction
  - existing resolver does not currently expose per-call abort to callers, while the lower-level `tldwClient.synthesizeSpeech` API already accepts an `AbortSignal`
- `apps/packages/ui/src/services/tts.ts`
  - existing global TTS settings
- `apps/packages/ui/src/services/tldw/audio-voices.ts`
  - server voice catalog fetch
- `apps/packages/ui/src/db/dexie/schema.ts` and `apps/packages/ui/src/db/dexie/types.ts`
  - any new read-along cache table needs an explicit Dexie version/type addition, not only a helper file
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
15. Text selection inside `ContentViewer` must have one coordinated consumer path so annotation capture and read-along actions do not both claim the same selection event unexpectedly.
16. Segmentation and lookahead are lazy, cancellable work that starts from an explicit read action, not page render.
17. Starting generated read-along playback pauses any embedded media preview that is already playing; v1 does not synchronize with or auto-resume that original media.

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

### Selection Consumer Arbitration

`ContentViewer` already calls `modals.handleCaptureAnnotationSelection` on content `onMouseUp` and `onKeyUp`. That handler captures selected text, stores an annotation selection, and activates the annotations tab. Adding read-along as a separate selection listener would create surprising behavior: the same drag could open read-along controls while also switching the intelligence panel to annotations.

V1 should introduce one shared selection mediation path scoped to `contentBodyRef`. That path can live in `ContentViewer` or a small `useContentSelectionActions` hook, but it should make the selection result available to both features. The user-facing result should be one compact selection affordance, not two competing popovers or an automatic tab switch plus a read-along menu.

Recommended behavior:

- valid content selection opens a selection action popover with read-along actions and an annotation action
- annotation text is captured only when the user chooses the annotation action or an existing annotation-focused shortcut, not merely because a selection occurred
- if the document intelligence panel already has an active annotation selection workflow, read-along should not clear it unless playback starts from a new selection
- keyboard selection follows the same mediated path

The implementation plan should explicitly preserve the annotation workflow while removing the current implicit "any selection opens annotations" coupling.

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
- abort in-flight synthesis and lookahead on stop, media/page/content change, and explicit restart
- suppress stale playback completions and cache writes by checking a session token before mutating state
- auto-scroll active segment only when needed

Recommended lookahead:

- start with current segment immediately
- prefetch 3 to 5 segments ahead
- for full-item reads, generate in batches of 10 to 20 queued segments
- never generate the entire item upfront

The session should treat audio generation and playback as separate phases so cached audio can play immediately while future segments generate.

Read-along should call the provider synthesis primitive directly, not `useTTS.speak()` as a queue driver. `useTTS` already performs its own punctuation-based segment splitting and user-visible clip saving, which would conflict with read-along's segment model and cache signature. One read-along segment should normally produce one TTS request. If a segment exceeds the provider/request cap, split it into deterministic sub-segments under that cap and keep the parent segment highlighted until all sub-segments finish.

Provider/request limits should be explicit in the implementation plan. V1 can use a conservative client-side maximum character count per request, overridable later if server/provider metadata becomes available.

Starting a read-along session should pause any embedded audio/video element in the media preview if it is playing. Stopping read-along should not auto-resume the original media because that would be surprising after a long generated narration session.

The first audio `play()` call is user-gesture-backed through the selection action. Subsequent segment playback should still handle `HTMLMediaElement.play()` promise rejection by moving to a clear segment-error state with retry/stop actions.

## 10. Audio Generation And Cache

Reuse existing TTS provider resolution instead of calling `/api/v1/audio/speech` ad hoc.

Preferred implementation path:

- factor reusable synthesis/playback pieces out of `useTTS` or use `resolveTtsProviderContext()`
- extend the provider synthesis path to accept `AbortSignal` because `tldwClient.synthesizeSpeech` already supports request aborts
- keep media read-along-specific queue/session behavior in the new hook
- do not save read-along segments into user-visible TTS clip history in v1

Add a browser-local cache separate from `ttsClips`, for example:

- `apps/packages/ui/src/db/dexie/media-read-along-cache.ts`

The cache also needs:

- a `MediaReadAlongAudioCacheEntry` type in `apps/packages/ui/src/db/dexie/types.ts`
- a Dexie schema version bump in `apps/packages/ui/src/db/dexie/schema.ts`
- migration-safe tests or store-helper tests that prove old databases can open and the new table can be read/written

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
- treat 250 MB as an upper bound, not guaranteed allocation; storage estimates may force a lower effective cap
- evict least-recently-used entries
- evict before writes when possible, and handle `QuotaExceededError` by retrying once after eviction before disabling cache
- tolerate Dexie/private-window failure by disabling cache for the session

Cache failure must never block playback.

Privacy rules:

- generated audio blobs stay browser-local and are never synced by v1
- cache metadata should not store raw selected text; use content hashes, segment IDs, media IDs, source offsets, and settings signatures
- cache debug UI, if any, must avoid exposing private transcript text in logs or persistent records

## 11. Rendering And Highlighting

### Plain And Transcript Rendering

Plain and transcript rendering should get first-class support in v1:

- wrap segments with `data-read-along-segment-id`
- apply active highlight class to current segment
- optionally apply a subtle completed state
- keep find highlighting compatible with read-along highlighting

Timestamp transcript lines should preserve existing timestamp seek buttons.

Segmentation should be lazy. Initial page render, scroll restoration, and find-in-content should not segment an entire large item merely because read-along is available. For full-item and read-from-here scopes on large content, build the queue in cancellable chunks and yield to the browser between batches so `ContentViewer` remains scrollable.

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

The popover and inline transport must clamp to the content viewport and sidepanel width. On constrained extension/mobile widths, prefer a compact anchored menu near the selection or active segment. Do not introduce a persistent bottom mini-player as a responsive fallback.

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
- Media/content changes abort in-flight synthesis/lookahead and ignore late async completions from older sessions.
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
- cache table schema/type/store helpers are migration-safe
- lookahead queue can play from cached and generated segments
- stop and media-change paths abort in-flight generation and suppress stale completions

### Stage 2: Selection Popover And Plain/Transcript Highlighting

Goal:

- replace direct annotation selection capture with mediated content selection actions in `ContentViewer`
- add popover actions
- add active highlighting for plain and timestamped transcript rendering

Success criteria:

- selecting text opens read-along actions
- annotation selection remains available from the same selection affordance without automatic tab switching on every selection
- `Read selection`, `Read from here`, `Read current section`, and `Read full item` queue expected scopes
- active segment highlights and auto-scrolls correctly
- stop clears transient UI
- generated read-along playback pauses an active embedded media preview without auto-resuming it on stop

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
- lazy large-content segmentation and cancellable queue construction
- stale session result suppression after stop/media change
- provider request cap fallback splitting

Component tests:

- popover appears only after valid content selection
- annotation and read-along selection actions coexist without competing UI state
- action menu queues the correct scope
- active segment highlight applies and clears
- pause/resume/stop update visible state
- stop/media change aborts in-flight TTS work and prevents stale cache writes
- TTS error exposes retry/skip/stop
- cache failure falls back to live generation
- storage quota failure evicts or disables cache without blocking playback
- find highlighting and read-along highlighting coexist
- TTS settings changed mid-session leave the active queue/cache signature stable until restart

Browser/E2E tests:

- WebUI media viewer selection-to-read flow
- extension media viewer selection-to-read flow
- transcript-line flow with timestamped content
- long content full-item chunked generation does not request all segments upfront
- `Read from here` on a windowed large item continues beyond the currently rendered window
- accessibility smoke for keyboard focus and status announcements

## 16. V1 Decisions Closed For Planning

These decisions should not be reopened by the first implementation plan unless new repo evidence makes them impossible:

- read-along entry remains selection-first
- no persistent page-level player
- active sessions keep a minimal inline transport visible until stopped
- `Read full item` uses canonical full source content, not the rendered window
- `Read from here` also continues through canonical full source content
- very large full-item reads use chunked lookahead after segmentation
- segmentation/lookahead starts lazily from a read action and must remain abortable
- annotation capture and read-along selection use one mediated selection path
- TTS settings changes apply to future sessions or explicit restarts, not mid-session mutation
- markdown/html exact inline highlighting is best effort; nearest-block fallback is acceptable
- cache eviction uses global LRU with default caps of 200 entries and 250 MB approximate total size
- cache metadata avoids raw selected text and treats browser storage quota as best effort
- generated read-along audio is not saved into user-visible TTS clip history in v1

## 17. Definition Of Done

- selection-initiated read-along works in shared `ContentViewer`
- WebUI and extension use the same implementation path
- no persistent page-level player is introduced
- existing TTS settings are reused
- selection handling is mediated so existing annotation workflows and read-along do not compete
- stop, media changes, and restarts abort in-flight TTS work and suppress stale completions
- transcript-line and sentence segmentation are covered by tests
- large-content segmentation is lazy and cancellable
- active segment highlighting works for plain/transcript content
- markdown/html have a safe fallback
- browser-local audio cache is opportunistic, separately evicted, quota-aware, and avoids raw text metadata
- long reads use chunked lookahead
- error states include retry, skip, and stop
- generated read-along playback pauses active embedded media preview without v1 sync/auto-resume behavior
- route parity and accessibility checks are documented

## 18. Deferred Follow-Ups

These are not v1 blockers:

1. Expose read-along cache as clearable local data in settings.
2. Add an explicit user action to save a read-along segment or session into user-visible TTS clips.
3. Add exact markdown/rich HTML text-node highlighting after the fallback path is proven.
4. Add backend-persisted read-along cache if cross-device reuse becomes important.
5. Add word-level highlighting if provider timing metadata becomes available.
