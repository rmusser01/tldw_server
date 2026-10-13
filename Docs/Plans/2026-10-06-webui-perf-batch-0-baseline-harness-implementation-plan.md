# WebUI Perf Batch W0 — Baseline Harness Implementation Plan

**Backlog task:** TASK-13520 · **Index:** [coordination index](2026-10-06-webui-perf-remediation-coordination-index.md)
**Verified against:** `dev` @ `7ba48f251e`

**Goal:** Measurement infrastructure so every later batch can prove before/after deltas: a streaming-render benchmark, a persistence micro-benchmark, and a recorded baseline document.

**Architecture:** Vitest `bench`-style timing tests + Playwright perf specs (the extension already has `test:e2e:perf`) + a checked-in baseline markdown that later batches update. No product code changes.

---

## Stage 1: Chat streaming render benchmark

**Goal:** A deterministic, CI-runnable measurement of how much work a streamed token triggers in each chat surface.
**Files:** Create `apps/packages/ui/src/hooks/__tests__/streaming-render.bench.ts` (or `*.perf.test.ts` following existing naming), modeled on the streaming tests in `useChatActions` tests.
**Approach:**
- Seed a store with N=200 messages; simulate 500 token chunks through the sidepanel path (`useMessage.tsx` chunk handler, anchor: `grep -n "prev.map((m) =>" hooks/useMessage.tsx`) counting (a) store-update invocations, (b) array clones (spy on `Array.prototype.map` or count via subscriber notifications on `useStoreMessageOption`).
- Assert *measurement recorded*, not a threshold, in CI; thresholds live in the baseline doc.
- Add a Playwright perf spec (`apps/extension/tests/e2e/performance-streaming.spec.ts`, patterned on existing `performance-*.spec.ts`) that streams a canned response into the sidepanel and records `performance.mark` deltas for 100 chunks.
**Success Criteria:** `bun run test` runs the bench and prints numbers; e2e perf spec runs under `test:e2e:perf`.
**Tests:** The bench files are the tests.
**Status:** Not Started

## Stage 2: Persistence micro-benchmarks

**Goal:** Numbers for the two critical persistence paths before W2 touches them.
**Files:** Create `apps/packages/ui/src/store/__tests__/workspace-persist.bench.ts`, `apps/tldw-frontend/lib/__tests__/history.bench.ts`.
**Approach:**
- Workspace: build a store fixture with 10 workspaces × 50 artifacts; run one trivial `set()`; count `JSON.stringify` calls (spy) and total serialized bytes.
- History: call `addRequestHistory` 50× with 100 KB bodies against a jsdom localStorage; measure total `setItem` bytes written (target metric: bytes/request, expect ≈ 50× history size without fix).
**Success Criteria:** Both benches print reproducible numbers (±10% across 3 runs on the same machine).
**Tests:** The bench files are the tests.
**Status:** Not Started

## Stage 3: Bundle-budget baseline + baseline document

**Goal:** Freeze the starting line the program will be judged against.
**Files:** Create `Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md`; read-only use of `apps/tldw-frontend` `check-bundle-budget.mjs` (already wired into `build`).
**Approach:**
- Run `bun run build` in `apps/tldw-frontend`; record route bundle sizes it prints.
- Record Stage 1–2 numbers, plus manual numbers for: sidepanel ingest of a 10-item batch (request count via devtools), page-load request count for `/` and `/chat` (buddy poll visible), copilot popup stream (main-thread long tasks).
- Document environment (machine, browser, date) and re-run instructions.
**Success Criteria:** Baseline doc exists with all four measurement families; every W-batch plan references it.
**Tests:** N/A (documentation; numbers produced by Stages 1–2).
**Status:** Not Started

## Verification & DoD

- [ ] `bun run test` in `apps/packages/ui` green; `test:e2e:perf` green in `apps/extension`.
- [ ] `Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md` committed with numbers.
- [ ] TASK-13520 updated: notes, verification results, final summary.
