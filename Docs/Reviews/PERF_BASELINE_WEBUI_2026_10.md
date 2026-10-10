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
