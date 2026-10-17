# WebUI Performance Baseline — 2026-10 (Program Batch W0)

This document is the frozen starting line that the WebUI/extension performance
remediation program (batches W0–W5) is judged against.

- Program index: `Docs/Plans/2026-10-06-webui-perf-remediation-coordination-index.md`
- W0 plan: `Docs/Plans/2026-10-06-webui-perf-batch-0-baseline-harness-implementation-plan.md`
- Backlog task: TASK-13520
- Baseline captured: 2026-10-07
- Sources: the W0 automated benches (commits `66c8a21c80`, `e645ed7a23`,
  `736b1a32a6`) and the production build output recorded verbatim in Family D.
- Precedence rule: the **measured** values in this document are authoritative.
  Any expectations or estimates written in bench-file headers or plan prose are
  non-normative commentary.

## Environment

| Item | Value |
|---|---|
| Machine | Apple silicon (`uname -m` → `arm64`) |
| OS | macOS 26.5.2 (Build 25F84, via `sw_vers`) |
| Date | 2026-10-07 |
| bun | 1.3.2 |
| node | v24.6.0 |
| Vitest | 4.0.18 (`apps/packages/ui` run banner) |
| Playwright | `@playwright/test` ^1.57.0, Chrome for Testing (extension e2e) |
| Frontend build | Next.js 16.1.4 (Turbopack) |
| Branch / HEAD | `codex/webui-perf-w0-baseline` @ `736b1a32a6` |
| Bench environment caveat | Vitest benches run under jsdom; its `localStorage` quota is 5,000,000 code units (affects the history bench, see Family C2) |

## How to re-run every number in this document

All commands run from the repo root of a worktree on the branch under test.
Each vitest/Playwright bench prints a one-line JSON blob with a stable prefix
(`[streaming-render.bench]`, `[performance-streaming]`,
`[workspace-persist.bench]`, `[history.bench]`); the build prints the
`[bundle-budget]` lines.

```bash
# Family A — unit streaming-render bench (sidepanel chat path)
cd apps/packages/ui && bun run vitest run src/hooks/__tests__/streaming-render.bench.test.ts

# Family B — e2e canned 100-chunk stream (packaged extension sidepanel)
# First run builds the production extension via globalSetup (~45 s); the five
# live-backend suites in the same file skip without TLDW_E2E_* env vars.
cd apps/extension && bunx playwright test tests/e2e/performance-streaming.spec.ts

# Family C1 — workspace persist bench (one trivial set() on a 10x50 fixture)
cd apps/packages/ui && bun run vitest run src/store/__tests__/workspace-persist.bench.test.ts

# Family C2 — request-history write amplification bench
cd apps/tldw-frontend && bun run vitest run lib/__tests__/history.bench.test.ts

# Family D — production build + bundle budget check
cd apps/tldw-frontend && bun run build:prod

# Family D — standalone re-check against an already-emitted .next (read-only)
cd apps/tldw-frontend && bun run check:bundle-budget
```

Note on `bun run build` vs `bun run build:prod`: see Family D — the default
`build` invocation fails off-`main` without `NEXT_PUBLIC_API_URL`; the baseline
numbers come from `build:prod`, the same pipeline CI's production gate runs.

## Family A — Unit streaming-render bench (sidepanel chat path)

Harness: `apps/packages/ui/src/hooks/__tests__/streaming-render.bench.test.ts`
(commit `66c8a21c80`). Renders the real `useMessage` hook over the real
`useStoreMessageOption` store seeded with 200 messages, mocks
`streamCharacterChatCompletion` as an async generator of 500 deterministic
token chunks, and counts store notifications and transcript-array `.map`
clones inside the first-chunk → persistence window.

| Metric | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| seedMessages | 200 | 200 | 200 | 200 |
| chunksStreamed | 500 | 500 | 500 | 500 |
| storeSubscriberNotifications | 502 | 502 | 502 | 502 |
| messageUpdateNotifications | 501 | 501 | 501 | 501 |
| transcriptArrayMapClones | 501 | 501 | 501 | 501 |
| streamPhaseMs | 21.2 | 14.9 | 13.5 | 13.18 |
| msPerChunk | 0.0424 | 0.0298 | 0.027 | 0.0264 |
| finalAssistantChars | 4890 | 4890 | 4890 | 4890 |

Interpretation (baseline behavior W1 is expected to change):

- Every streamed token triggers one full-transcript `.map` clone plus one
  messages-array store update: **500 chunks → 501 clones / 501 message
  updates / 502 store notifications** (the extra notification is
  `setIsProcessing(true)` on the first chunk; the 501st map is the post-loop
  final-text commit at `useMessage.tsx:1723`).
- Counts are deterministic (bit-identical across runs); only timings vary.

## Family B — E2e canned 100-chunk stream (Playwright, packaged sidepanel)

Harness: the "Canned 100-chunk stream (no live LLM)" suite in
`apps/extension/tests/e2e/performance-streaming.spec.ts` (commit `e645ed7a23`).
A loopback mock server serves a paced SSE stream of exactly 100 chunks
(**10 ms apart**) into the real sidepanel `/chat` send path; an in-page bench
on the real `useStoreMessageOption` store records marks/measures plus store,
DOM-mutation and long-task counts.

| Run | chunksServed | chunksApplied | storeSubscriberNotifications | messageUpdateNotifications | domMutations | longTasks | ttftMs | streamPhaseMs | msPerChunk | minChunkGapMs | maxChunkGapMs | domNodeCount |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 15 | 19 | 16 | 83 | 0 | 134.0 | 1187.2 | 79.15 | 74.4 | 98.3 | 196 |
| 2 | 100 | 15 | 19 | 16 | 81 | 0 | 96.5 | 1150.2 | 76.68 | 41.9 | 88.6 | 196 |
| 3 | 100 | 15 | 19 | 16 | 77 | 0 | 97.5 | 1152.1 | 76.81 | 33.2 | 89.1 | 215 |

**Label ambiguity:** `msPerChunk` in this family is **milliseconds per APPLIED
chunk** (store growth event), not per served SSE chunk. Quoted verbatim from
the Task 1B report's caveat:

> `chunksApplied` (15) < `chunksServed` (100) reflects real chunk coalescing
> in the sidepanel path; it is a measured observation, not a harness bug
> (delivery is asserted end-to-end via the final text). Baseline doc should
> note the 10 ms pacing when quoting msPerChunk.

Other observations: 100 served chunks coalesce to ~15 applied store-growth
events (network pacing of 10 ms/chunk is faster than the UI applies text);
store notifications (19) slightly exceed message-array identity updates (16)
because a few non-message `set` calls interleave in the window; **zero
main-thread long tasks (>50 ms)** in every run at this transcript size; the
stream phase is stable to within ~3% across runs.

## Family C — Persistence micro-benches

### C1. Workspace store: one trivial `set()` on a 10×50 fixture

Harness: `apps/packages/ui/src/store/__tests__/workspace-persist.bench.test.ts`
(commit `736b1a32a6`). Fixture: 10 workspaces × 50 generated artifacts ×
1,024-byte content = **595,571 B (~595.6 KB)** of snapshot JSON, plus 10 saved
workspaces. One trivial `set()` (`notes: "bench trivial set N"`) drives the
full zustand persist chain (`partialize` → JSON storage stringify → split-key
`setItem`). `indexedDbOffloadAvailable: false` (jsdom has no `indexedDB`), so
artifacts stay in localStorage for this measurement.

| Metric | Run 1 | Run 2 | Run 3 | Spread |
|---|---|---|---|---|
| JSON.stringify calls | 31 | 31 | 31 | 0% |
| JSON.stringify bytes | 2,451,167 (~2.45 MB) | same | same | 0% |
| localStorage.setItem calls | 2 | 2 | 2 | 0% |
| localStorage.setItem bytes | 121,470 (~121.5 KB) | same | same | 0% |
| durationMs (informational) | 10.65 | 12.15 | 10.01 | 21.38% |

**Dev-only diagnostics caveat (binding for W2 comparisons):** ~19 of the 31
`JSON.stringify` calls per trivial `set()` come from the dev-only
`recordWorkspacePersistenceDiagnostics`, which executes under
`NODE_ENV=test` (as in this bench) but **not in production builds**. The
absolute 2.45 MB serialize figure therefore **overstates production
serialization cost**; W2 before/after deltas remain valid because the offset
is constant across runs of the same harness.

Remaining interpretation: actual localStorage writes stay at 121,470 B (active
snapshot key + split index key; unchanged snapshot/chat keys are correctly
skipped). A one-field change to a ~595.6 KB persisted fixture re-serializes
~4.1× the fixture size across 31 stringify calls — zustand persist's
monolithic envelope is built and discarded even though split-key storage then
re-serializes all 10 snapshots to diff them, and the split index (which embeds
the active snapshot) is rebuilt.

### C2. Request history: `addRequestHistory` × 50 with 100 KiB bodies

Harness: `apps/tldw-frontend/lib/__tests__/history.bench.test.ts` (commit
`736b1a32a6`). Each entry carries a 102,400-char (100 KiB) `requestBody`;
count/byte metrics are identical across all 3 in-process runs.

| Metric | Value (runs 1–3 identical) |
|---|---|
| setItem calls attempted | 50 |
| setItem calls landed | 48 |
| setItem quota errors | 2 (jsdom 5,000,000-code-unit quota artifact; real-browser quota differs) |
| setItem bytes attempted | 130,757,040 B ≈ **130.8 MB** |
| setItem bytes landed | 120,698,808 B ≈ 120.7 MB |
| final history entries | 48 |
| final history bytes | 4,926,481 B ≈ 4.93 MB |
| bytes/request (attempted) | 2,615,141 B ≈ 2.62 MB |
| amplification vs final history | 26.54× (≈ 50× the average history size during the run) |
| durationMs (informational) | ~97–174 |

Interpretation: each add rewrites the ENTIRE accumulated history — 50 adds of
100 KiB bodies attempt ≈ 130.8 MB of `setItem` traffic for a 4.93 MB final
history, exactly the ≈50×-average-history-size quadratic-rewrite cost W2 is
targeted at. Attempted bytes is the primary metric (landed bytes are capped by
the jsdom quota artifact above).

## Family D — Frontend build & bundle-budget baseline

### Default invocation status (`bun run build`)

Fails **before compilation** on any non-`main` branch without
`NEXT_PUBLIC_API_URL` (branch-aware build profiles resolve non-`main` to the
development/advanced profile, and advanced mode requires an absolute
`NEXT_PUBLIC_API_URL`; this worktree has no `.env.local`). Captured verbatim:

```
$ bun run build
Invalid WebUI networking config: advanced mode requires NEXT_PUBLIC_API_URL to be an absolute browser API URL.
error: script "build" exited with code 1
```

This is a pre-existing environment precondition, unrelated to the W0 program
(no product code was changed), recorded here as the default invocation's
baseline status and deliberately not fixed in W0.

### Production build baseline (`bun run build:prod`, the CI production gate)

`TLDW_BUILD_PROFILE=production` → quickstart mode, default internal origin
`http://127.0.0.1:8000`. Succeeded on 2026-10-07:

```
▲ Next.js 16.1.4 (Turbopack)
✓ Compiled successfully in 35.5s
Generating static pages using 17 workers (154/154) in 209.8ms
[token-sync] OK: .next/static/chunks/b71c3e0a115c9212.css matches shared tokens
[bundle-budget] shared _app: 593.4 KB gzip (budget 600.0 KB, 34 files)
[bundle-budget] heaviest route: /__debug__/mermaid-chat-cards 897.2 KB gzip
[bundle-budget] ok
```

| Metric | Value | Budget | Headroom |
|---|---|---|---|
| Shared `_app` shell (gzip, 34 files) | 593.4 KB | 600.0 KB | 6.6 KB (98.9% of budget) |
| Heaviest route first-load JS (gzip) | 897.2 KB (`/__debug__/mermaid-chat-cards`) | 900.0 KB | 2.8 KB (99.7% of budget) |
| Bundle-budget check | **ok** (pass) | — | — |
| Static pages generated | 154 | — | — |

**Budget-headroom warning (binding for W1+):** headroom against the frozen
baseline is effectively zero — 6.6 KB / 1.1% on the shared `_app` budget and
2.8 KB / 0.3% on the heaviest route — so ANY byte-adding batch (W1 adds
memoization/virtualization code) can trip the hard `check-bundle-budget` gate
inside `build:prod`/`build`, and a failure there may be unrelated to that
batch's own success criteria. Byte-adding batches must run
`bun run check:bundle-budget` before and after their change and fund or
explicitly justify any budget increase. When the check is red, compare the
emitted numbers against this table to distinguish inherited saturation from
new growth introduced by the batch.

Notes:

- Turbopack prints **no per-route size table** in `next build`; the two
  `[bundle-budget]` lines above are the route-bundle size lines the build
  emits, produced by `apps/tldw-frontend/scripts/check-bundle-budget.mjs`
  reading `.next/build-manifest.json` (gzip level 6). Budgets live in that
  script: `SHARED_BUDGET_BYTES = 600 KB`, `ROUTE_BUDGET_BYTES = 900 KB`.
- `bun run check:bundle-budget` (standalone, read-only against the emitted
  `.next`) reproduced both numbers identically after the build.
- Raising either budget is a deliberate act and must be recorded in the
  committing message per the script's policy comment.

## Deferred manual measurements (PENDING-MANUAL)

Per the W0 controller ruling, the three manual metrics below are
**load-bearing-deferred**: they are placeholders with exact instructions, to
be filled in by the batch that consumes them (W0's load-bearing baseline is
the automated benches + build numbers above). Do not automate them with
browser automation when filling; record human DevTools observations.
Fill-in entry criteria: recorders must pin and record the exact Chrome
version used (e.g. from `chrome://version`) when filling M1–M3, so the W4/W5
before/after comparisons against these rows are like-for-like.

### M1. Sidepanel quick-ingest of a 10-item batch — HTTP request count — PENDING-MANUAL

- **What to measure:** total HTTP requests issued by the extension from
  submitting a 10-item quick-ingest batch until every item reaches a terminal
  state, split into: job-submission POSTs, job-poll GET rounds, and terminal
  result PATCHes.
- **DevTools steps:**
  1. `cd apps/extension && bun run build:chrome:prod`; load the unpacked
     extension in Chrome; configure the sidepanel against a running server.
  2. Open `chrome://extensions` → find the extension → click **service
     worker** to open its DevTools (ingest requests originate in the
     background worker).
  3. In that DevTools: Network tab, enable **Preserve log**, clear the view.
  4. From the sidepanel, quick-ingest a 10-URL batch sharing one
     source/config (identical `buildFields()` output).
  5. When all 10 items are terminal, record total requests and the three
     sub-counts.
- **Consumed by:** W4 Stage 1 (extension quick-ingest batching; success
  criterion: ≤ 4 submission requests from 10 and ≥ 60% wall-time drop against
  this row).

### M2. Page-load + idle request counts for `/` and `/chat` (buddy poll visible) — PENDING-MANUAL

- **What to measure:** request count at hard-load completion and after 15 s of
  an idle, visible tab on `/` and `/chat`, with the ~5 s repeating polls
  (PersonaBuddy refresh, notifications, research/workspace pollers) identified
  separately.
- **DevTools steps:**
  1. Serve the production webui: `cd apps/tldw-frontend && bun run build:prod
     && bun run start` (quickstart profile expects the backend on
     `http://127.0.0.1:8000`).
  2. Open `http://localhost:3000/` with DevTools → Network; enable
     **Disable cache**; clear; hard reload.
  3. Record the request count at load-complete, then keep the tab visible and
     idle and record the count again after 15 s; note every
     repeating-interval request and its period.
  4. Repeat steps 2–3 for `/chat`.
- **Consumed by:** W4 Stage 2 (polling → SSE/visibility gating; success
  criterion: idle visible tab on `/` + `/chat` at 0 background requests, from
  the ~2–4-per-5s baseline recorded here).

### M3. Copilot popup stream — host-page main-thread long tasks — PENDING-MANUAL

- **What to measure:** count and durations of main-thread long tasks (>50 ms)
  on the host page while the copilot popup streams a completion (the popup
  currently reassigns the full accumulated `responseEl.textContent` per
  chunk).
- **DevTools steps:**
  1. `cd apps/extension && bun run build:chrome:prod`; load the unpacked
     extension; open an ordinary content page.
  2. DevTools → Performance panel; start a recording.
  3. Invoke the copilot popup and run a streaming completion long enough to
     deliver ≥ 100 chunks.
  4. Stop the recording; in the timings/summary tracks count Long Tasks
     (>50 ms) inside the stream window and note their durations.
- **Consumed by:** W5 Stage 2 (copilot popup streaming + audio base64; its
  definition-of-done re-runs this measurement and appends before/after here).

## Change protocol (footer)

Every batch W1–W5 **must** append a dated before/after section to this
document, produced with the exact commands in "How to re-run", quoting deltas
against the tables above; the PENDING-MANUAL rows are filled in by their
consuming batches (M1/M2 → W4, M3 → W5). This file is the program's single
accumulating record — do not fork it per batch.

## Batch W1 (TASK-13521) — 2026-10-15

Appended per the change protocol above. Every number in this section comes
from an actual re-run at the W1 branch HEAD (no transcription from earlier
stage reports — the precedence rule applied to this batch too). The frozen
W0 tables above are untouched and remain the program baseline.

- Branch / HEAD: `codex/webui-perf-w1-chat-streaming` @ `07427bd044`
  (base: dev merge `3b68f2dbc2`, which includes the W0 harness)
- Backlog task: TASK-13521
- Run date: 2026-10-15 (e2e report timestamp); all commands from
  "How to re-run every number in this document"

### Family A re-run — and a binding drift note on the frozen 502 figure

Re-run of `streaming-render.bench.test.ts` at HEAD:

| Metric | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| seedMessages | 200 | 200 | 200 |
| chunksStreamed | 500 | 500 | 500 |
| storeSubscriberNotifications | 3 | 3 | 3 |
| messageUpdateNotifications | 2 | 2 | 2 |
| transcriptArrayMapClones | 1 | 1 | 1 |
| streamPhaseMs | 5.19 | 5.17 | 4.09 |
| msPerChunk | 0.0104 | 0.0103 | 0.0082 |
| finalAssistantChars | 4890 | 4890 | 4890 |

**Drift attribution (binding).** The frozen Family A figure of 502 store
notifications / 501 message updates / 501 map clones was **stale already at
W1's branch point**. Interim dev work on the same files (the TASK-13511
frontend-perf-arch lineage, merged into dev via PR #3210 before W1 branched)
had already added 80 ms streaming coalescing to the sidepanel path:
`STREAMING_UPDATE_INTERVAL_MS = 80` in `useChatActions.ts` and
`useCharacterChatMode.ts`, plus a `streamingScheduler` in `useMessage.tsx`
(verified in the tree at merge base `3b68f2dbc2`). At the branch point this
bench already read ~3 store notifications / 2 map clones — not 502/501.

**W1 Stage 1's own delta** (commit `52809b6571`; attribution verified
both-sides against the diff in stage review): transcript-array map clones
2 → 1, with per-token (and per-flush) clones 0 — the single surviving clone
is the once-per-stream final-text commit, reached via a last-index slice
replacement instead of a full-transcript `.map`. Stage 1 also unified the
inline throttles into the shared `streamingThrottle` helper adopted across
all three chat paths (`useMessage.tsx` ×4 scheduler sites,
`useChatActions.ts`, `useCharacterChatMode.ts`). Store notifications were
3 → 3 across the stage — the 502→~3 notification drop predates W1 entirely.

### Family B re-run at HEAD (canned 100-chunk sidepanel stream)

`bunx playwright test tests/e2e/performance-streaming.spec.ts` — 3 runs:

| Run | chunksServed | chunksApplied | storeSubscriberNotifications | messageUpdateNotifications | domMutations | longTasks | ttftMs | streamPhaseMs | msPerChunk | minChunkGapMs | maxChunkGapMs | domNodeCount |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 15 | 19 | 16 | 49 | 0 | 138.7 | 1196.8 | 79.79 | 55.9 | 123.5 | 196 |
| 2 | 100 | 15 | 19 | 16 | 51 | 0 | 145.0 | 1180.0 | 78.67 | 44.0 | 110.6 | 196 |
| 3 | 100 | 15 | 19 | 16 | 49 | 0 | 231.5 | 1232.4 | 82.16 | 75.5 | 152.7 | 196 |

(`msPerChunk` is per APPLIED chunk, per the label-ambiguity note in Family B
above.)

Comparison against the frozen Family B table:

- Count metrics are **identical** (100/15/19/16): the coalescing visible in
  the packaged sidepanel path predates W1 (it was already observable at W0
  baseline capture, per the baseline's own `chunksApplied` note), and W1 did
  not change the throttle cadence on this path.
- `domMutations` is the one metric that moved: **77–83 → 49–51** (~40% fewer
  mutation records inside the stream window), consistent with Stage 2's
  isStreaming last-row narrowing plus row memoization (unchanged rows no
  longer re-render per flush). Recorded as an observation — the unit suites,
  not this e2e, pin the render-count mechanics.
- `longTasks`: **0 in every baseline run and 0 in every HEAD run.**

**Stage 3's deferred success criterion, evaluated here (honest ruling).**
Stage 3 deferred its "long-task count ≥ 50% drop vs baseline during
streaming" criterion to this re-run. Verdict: **not-discriminating (0 → 0)**
— the canned stream at this transcript size produced zero main-thread long
tasks (> 50 ms) both before virtualization (frozen baseline, all 3 runs) and
at HEAD (all 3 runs). Per the batch-final ruling this is recorded as
not-discriminating rather than claimed as a pass. The virtualization work's
effect is instead pinned by unit window assertions: 1,000-message ChatPane
and 1,000-block PlaygroundChat transcripts render bounded windows (< 60
shells) in `ChatPane.virtualization.test.tsx` and
`VirtualChatTimeline.window.test.tsx`.

### Per-stage attribution (what THIS branch landed vs pre-landed interim work)

| Stage | Commit | Landed by W1 | Already at branch point (TASK-13511 lineage) |
|---|---|---|---|
| Task 0 | `c8c12161cb` | Extracted shared sidepanel mock-server helper for extension e2e specs (test-only; enables perf-spec reuse; no perf delta) | — |
| Stage 1 (WP-01) | `52809b6571` | Map clones 2 → 1 (per-token clones 0; survivor = once-per-stream final commit); unified `streamingThrottle` helper across all three chat paths (3 adoption sites + 4 scheduler sites in useMessage) | The 80 ms coalescing itself (502 → ~3 notifications, clones 501 → 2) |
| Stage 2 (WP-02/03/09) | `6747b1da64` | isStreaming narrowed to last row (both parents); highlightText memo; mount-time handler-cache guard fix | React.memo + comparator, per-row stable callbacks, EMPTY_IMAGES, previousUserMessageByIndex map (`dfa4dcc014`) |
| Stage 3 (WP-03/04/07) | `9d73e06724` | ChatPane virtualization (> 100-row gate) + stableMessageId + per-message retrieval-diagnostics cache; SharedWorkspaceChatPane K=200 render cap + window reveal | PlaygroundChat virtualization (VirtualChatTimeline, already on dev; pinned by new window tests) |
| Stage 4 (WP-05/06/08/10/11) | `4899049c07` | KnowledgeQA 250 ms query debounce + submit-flush; CharacterSelect derived-boolean selectors (both variants); LayoutShellContext value memo; ResultsList memoized row; PlaygroundChat O(1) prev-user lookup | body.tsx previousUserMessageByIndex O(1) map (pre-existing) |
| Stage 5 (WP-13/14) | `07427bd044` | Trust-panel derivation memos + Set-deduped `textToStringList` (research.tsx); WP-13 no-change dispositions verified + review-signed-off | — |

### Documented gaps (honest)

- **(a) No web-e2e scroll-through DOM-bound spec exists.** There is no web
  playground e2e infra in-repo (extension perf specs are sidepanel-only), so
  transcript virtualization on the web panes is covered by unit window
  assertions in both panes, not by a DOM-bound e2e.
- **(b) SharedWorkspaceChatPane virtualization deferred** per the cap-first
  ruling: the K=200 newest-message render cap bounds markdown shells to the
  virtualized panes' worst-case expanded window; evidence and the revisit
  trigger (profile markdown cost under the cap) are documented in-component.
- **(c) KnowledgeQA full state/actions context split deferred** (15 consumer
  files exceed the split threshold); the query debounce + submit-flush
  landed.
- **(d) WP-13 no-change dispositions** (ChatWorkflowsPage composite
  `role-step_index-index` key with load-bearing index suffix;
  VNScriptsWorkbench static-order id-less primitive lists) — verified and
  signed off in Stage 5 review; recorded here for whole-branch visibility.
- **(e) WP-12 (flashcards) deferred** per the program index; no W1 change.

### Other verification at HEAD (2026-10-15)

- Family C1 workspace-persist bench: green, **bit-identical** to the frozen
  baseline — 31 stringify calls / 2,451,167 B / 2 setItem calls / 121,470 B
  across 3 runs (durations 12.94 / 13.61 / 13.08 ms). W1 touched no
  persistence code (that is W2's scope).
- Family C2 history bench: not run — W2's bench per the program split.
- Family D `bun run build:prod` at HEAD: `[bundle-budget] shared _app:
  596.1 KB gzip (budget 600.0 KB, 35 files)`; `heaviest route:
  /__debug__/mermaid-chat-cards 875.8 KB gzip`; `[bundle-budget] ok`. Versus
  the frozen baseline: shared shell +2.7 KB (34 → 35 files, still inside
  budget — headroom now 3.9 KB), heaviest route −21.4 KB. Budget gate PASS;
  no budget raise performed.
- tsc: `apps/tldw-frontend` `tsc --noEmit` → 0 errors; `apps/packages/ui`
  scoped `tsc --noEmit` (8 GB heap) → 400 errors, the identical pre-existing
  set carried since Stage 1, with 0 errors in W1-touched files.

## Batch W2 (TASK-13522) — 2026-10-16

Persistence write-path batch: request-history sidecar (WP-15), workspace
persist chain (WP-16), ModelDb storage hygiene (WP-17/WP-61), legacy
`chrome.storage` blob-DB quarantine (WP-18), background session-state +
settings registry (WP-19/WP-20/WP-22), deep-clone cleanup (WP-21). Branch
`codex/webui-perf-w2-persistence`, range `3b68f2dbc2..aaf09f42ef`
(14 commits). All numbers below are from actual re-runs of the frozen
Family C benches at HEAD `aaf09f42ef` on 2026-10-14 (same machine and
environment as the frozen tables; the frozen tables above are untouched).
Note: W1's appended section is not present in this branch's copy of the
file — the branch base (`3b68f2dbc2`) predates it and W1 owns its own
append; nothing of W1's was modified here.
Coverage note (final-review round 1, 2026-10-16): the branch tip is now
`3c699d4814` — 16 commits. The 14-commit range above already includes the
Stage 2 backlog-note commits `fa77a7f56b` and `d1b91870b7`;
`55845325a3` (this delta append) and `3c699d4814` (import-suite test fix,
review round 2) extend it — all four are docs/test/backlog-only and change
no measured surface, so the attribution tables and HEAD-`aaf09f42ef` bench
numbers above remain the record for the perf range.

### Family C2 — request history (WP-15, commit `3958f1e8d3`)

**Like-for-like metrics (the binding framing for this family):**

| Metric | Baseline (frozen) | W2 @ HEAD |
|---|---|---|
| setItem calls attempted per run | 50 | **1** |
| setItem calls landed | 48 | 1 |
| setItem quota errors | 2 | **0** |
| Amplification vs final history | 26.54× | **1.0×** |
| final history entries | 48 | 50 |
| durationMs (informational) | ~97–174 | 2.79–4.84 |

**Byte totals (quoted alongside, with the conflation caveat):**

| Metric | Baseline (frozen) | W2 @ HEAD |
|---|---|---|
| setItem bytes attempted | 130,757,040 B ≈ 130.8 MB | 115,651 B |
| setItem bytes landed | 120,698,808 B ≈ 120.7 MB | 115,651 B |
| final history bytes | 4,926,481 B ≈ 4.93 MB | 115,651 B |
| bytes/request (attempted) | 2,615,141 B | 2,313 B |

The 130.8 MB → 115.7 KB byte delta is **not** like-for-like: it conflates
(a) batching — one debounced flush per run instead of 50 whole-history
rewrites, (b) 2 KB `responseBody` truncation per entry, and (c) the bench's
`requestBody` → `responseBody` payload retarget forced in Task 1: the plan
truncates only `responseBody` while the W0 bench carried its 100 KiB blob
in `requestBody`, and the untouched bench's quota math (untruncated 100 KB
bodies × whole-history rewrites against jsdom's 5,000,000-code-unit quota)
made it unpassable for any correct fix. The retarget and the strengthened
assertions were reviewed and approved (Task 1 review: 0 Critical/Important).
Ratified ruling: "2 KB" = **2048 UTF-16 code units**, as implemented for a
JS debug sidecar.

### Family C1 — workspace persist (WP-16)

| Metric | Baseline (frozen) | W2 @ HEAD |
|---|---|---|
| JSON.stringify calls | 31 | **22** |
| JSON.stringify bytes | 2,451,167 B (~2.45 MB) | **1,915,228 B (~1.92 MB)** |
| localStorage.setItem calls | 2 | 2 |
| localStorage.setItem bytes | 121,470 B | 121,600 B (+130 B) |
| durationMs (informational) | 10.01–12.15 | 12.99–21.46 |

The frozen table's dev-only diagnostics caveat still applies: ~19 of the
stringify calls per run are `NODE_ENV=test` diagnostics that do not execute
in production builds, so the absolute serialize figures overstate
production cost; the before/after delta remains valid because the offset
is constant across runs of the same harness.

Attribution: sub-steps 1–2 (coalescing `3751c03c22`, per-workspace revision
diff `df3db85514`) delivered the measurable drop (31→22 stringifies,
2.45 MB → 1.92 MB). Sub-step 3 (artifact offload dedup, `55b070bb90`) is
unobservable in this bench — jsdom has no IndexedDB
(`indexedDbOffloadAvailable: false`), so artifacts stay inline — and is
pinned by unit tests instead. The +130 B setItem delta is the per-workspace
revision/etag metadata riding the same two written keys (active snapshot +
split index; key set unchanged from baseline).

### Per-stage attribution (this branch vs pre-landed TASK-13511 work)

| Stage (WP) | Commits (this branch) | This branch delivered | Pre-landed overlap |
|---|---|---|---|
| Task 1 — request-history sidecar (WP-15) | `3958f1e8d3` | Whole sidecar: in-memory buffer, redact + truncate exactly once at insert, 500 ms trailing debounce + idle + lifecycle flushes, `flushRequestHistory()`, bench retarget + strengthened assertions | none |
| Stage 2 — workspace persist chain (WP-16) | `3751c03c22`, `df3db85514`, `55b070bb90`; fix round `426fc547af` (+ `d1b91870b7` backlog note) | Serialized coalescing writer (per-key pending map, literal 300 ms trailing flush, durability-mode lifecycle flush), per-workspace revision diff, artifact offload dedup with invalidation/failure handling | none |
| Stage 3 — ModelDb hygiene (WP-17 + WP-61 alarm) | `5b63d01050` | Ledger, verbatim: "~80% pre-landed by interim `ff8803237e` (namespacing, migration, batched createMany, warm Set, 6h alarm); this stage's delta = bulkAddModelsFB replaceAll (2 sets+1 remove → 1+1) + the pinning suite" | `ff8803237e` (~80% of the stage) |
| Stage 4 — legacy blob-DB quarantine (WP-18) | `d6421555d2`; fix round `3626e61179` | Batched legacy `deleteHistoriesByDateRange` (one `remove([...])` + one index rewrite), titles-only legacy `searchChatHistories`, module-header deprecation + complete fallback map (incl. the migration importer), quarantine/characterization/source-guard suites | The repoint was pre-landed: `messageHandlers` and both delete callers were already on Dexie at the branch base |
| Stage 5 — session state + settings registry (WP-19/20/22) | `174b7e7bfe`, `01edfa3178`; fix round `623f8b2896` | Serialize-on-drain sharded session writer (`tldw:ingest:*` + index, migration-on-read from the monolithic key), `result.data` strip at persist, registry value cache + write-only-on-change mirror + `onChanged` invalidation, default-off chat-request-debug gate | WP-22's clone removal was pre-landed (TASK-13511's lazy clone-on-read retained; W2 added the enabled gate) |
| Stage 6 — deep-clone cleanup (WP-21) | `aaf09f42ef` | `structuredClone` replaces JSON round-trips in store mutation paths | none |

Fix-round record: Stages 2, 4 and 5 each required one review round
(5 / 1 / 3 findings addressed respectively; 0 open after each). Task 1,
Stage 3 and Stage 6 were approved with 0 Critical/Important findings.
At HEAD there are zero unresolved Critical/Important review findings.

### Documented gaps / follow-ups

- **Import-suite residual — RESOLVED (round 2 of the batch-final task):
  branch-caused by Stage 2's debounce (test read persisted state
  synchronously); fixed test-side via reader-scoped flush; suite 68/68.**
  The Stage-2 fix-round report had recorded the residual failure in
  `apps/packages/ui/src/utils/__tests__/research-workspace-import.test.tsx`
  ("retains two note chunks, retries only unfinished snapshots, and reopens
  source provenance") as pre-existing; batch-final verification disproved
  that — the suite is 68/68 green at the dev merge-base `3b68f2dbc2` vs
  67/68 at HEAD in the same environment (merge-base run in a temporary
  worktree sharing this branch's `node_modules`; manifests unchanged).
  Mechanism, pinned empirically: the test snapshotted `localStorage`
  synchronously after its final mutation (`loadNote`) while the debounced
  write was still buffered (microtask/idle/300 ms trailing delivery), so
  the re-seeded rehydrate restored the pre-`loadNote` envelope. Fix:
  `await flushWorkspacePersistence()` immediately before the snapshot —
  the same reader-scoped flush pattern `use-research-workspace-prefill.ts`
  uses (Stage 2's Finding-4 resolution). Suite re-run 68/68 green at HEAD;
  the only other storage reads in the suite are out-of-band `setItem`
  seeding plus one rehydrate-after-awaits (microtask delivery lands writes;
  green), so no other site needed the flush.
- **Pre-existing split-storage test failure:** `workspace.split-storage.test.ts`
  (1/8) fails at the dev base as well — jsdom instance-level storage-spy
  pattern (the W0 ledger's jsdom note); documented since Stage 2, unchanged.
- **replaceAll crash-orphan reclamation:** a crash between ModelDb
  `replaceAll`'s set and remove leaves orphaned namespaced keys (storage
  bloat only, invisible to reads) — future cleanup-pass location noted in
  the Stage 3 review.
- **defineSetting options-drop trap:** `defineSetting` silently drops
  unknown options (Stage 5 review minor) — follow-up to validate or warn.
- **Multi-tab clobber:** the in-memory history buffer clobbers concurrent
  tab writes by design (Task 1 review minor); inherent to the sidecar
  design as approved.
- **In-flight-IDB-await pagehide gap (narrowed, not closed):** lifecycle
  flushes gate all IndexedDB awaits, but a pagehide arriving while an IDB
  await is already in flight on the flush-initiated path can still preempt;
  out-of-scope deferral from the Stage 2 review.
- **Registry absent-value / web-context `onChanged` limits:** the settings
  registry's cache invalidation has reduced fidelity for absent values and
  no `chrome.storage.onChanged` signal in web context — Stage 5 deferred
  minors.

### Verification runs (2026-10-14, HEAD `aaf09f42ef`)

- Family C2: `cd apps/tldw-frontend && bun run vitest run
  lib/__tests__/history.bench.test.ts` — passed as-is (the bench was
  legitimately modified in Task 1 per the reviewed ruling); numbers quoted
  above, 3 runs bit-identical on count/byte metrics.
- Family C1: `cd apps/packages/ui && bun run vitest run
  src/store/__tests__/workspace-persist.bench.test.ts` — passed untouched;
  numbers quoted above, 3 runs bit-identical on count/byte metrics.
- Scoped tsc baselines unchanged: `apps/packages/ui` `tsc --noEmit` → 400
  pre-existing errors (0 in touched files); `apps/tldw-frontend`
  `tsc --noEmit` → 0 errors.
- The W0 streaming benches (Families A/B) belong to W1's remediation scope
  and are skipped here per the batch brief.
