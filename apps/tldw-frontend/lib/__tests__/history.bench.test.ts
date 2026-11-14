// Request-history persistence micro-benchmark — addRequestHistory
// (TASK-13520, Batch W0 Stage 2; re-targeted in TASK-13522, Batch W2 Stage 1).
//
// Measures how much localStorage write traffic `addRequestHistory` produces
// when 50 requests with 100 KiB bodies are recorded against jsdom localStorage.
//
// Pre-W2 (recorded in Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md, Family C2):
// every add re-read, re-redacted (deep-copied) and re-serialized the ENTIRE
// accumulated history synchronously — 50 setItem calls attempting ≈ 130.8 MB
// for a 4.93 MB final history (quadratic rewrite), with the last two writes
// exceeding jsdom's 5,000,000-code-unit localStorage quota.
//
// Post-W2 (TASK-13522 Stage 1): adds buffer into a module-level array with
// redaction + 2 KB responseBody truncation applied once at insert, and persist
// in one batched setItem per FLUSH_DEBOUNCE_MS trailing-debounce window
// (requestIdleCallback when available; jsdom lacks it, so the setTimeout
// fallback fires directly). The harness therefore awaits one debounce window
// after the add loop to observe and measure the flush. Expected shape: ≤ 1
// small setItem per run (≈ 2 KB per entry, so no quota errors) and attempted ≈
// landed bytes.
//
// Method:
//   - 50 `addRequestHistory` calls, each carrying a 102,400-char responseBody
//     (all entries byte-length identical for deterministic totals).
//   - Passthrough spy on `Storage.prototype.setItem` (jsdom's Storage is a
//     WebIDL proxy; spying the `localStorage` instance stores the spy as a
//     named data entry instead of intercepting — see the sibling workspace
//     bench for the same finding).
//   - Repeated 3x in-process (localStorage cleared AND the history module's
//     buffer reset between runs) for reproducibility; counts/bytes are
//     deterministic, duration informational.
//
// This is a measurement harness, not a regression gate: assertions only
// sanity-check that counters are finite, non-negative and non-zero and that the
// harness drove the intended path. Perf thresholds live in
// Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md; batch W2 re-runs this bench
// after touching the history persistence path and records the before/after
// delta there.
//
// Run: cd apps/tldw-frontend && bun run vitest run lib/__tests__/history.bench.test.ts
import { describe, expect, it, vi } from "vitest"

import {
  FLUSH_DEBOUNCE_MS,
  addRequestHistory,
  clearRequestHistory,
  type RequestHistoryItem
} from "@web/lib/history"

const HISTORY_KEY = "tldw-request-history"
const REQUESTS = 50
const BODY_BYTES = 100 * 1024
const MEASUREMENT_RUNS = 3
// Real-time wait past the trailing debounce so the batched flush lands before
// metrics are collected (jsdom has no requestIdleCallback; the timer fallback
// fires directly).
const FLUSH_SETTLE_MS = FLUSH_DEBOUNCE_MS + 150

const textEncoder = new TextEncoder()
const utf8Bytes = (value: string): number => textEncoder.encode(value).length

const bodyBlob = "x".repeat(BODY_BYTES)

// All entries serialize to identical byte lengths (padded ids, fixed fields,
// fixed 2 KB truncation cap), so per-run totals are deterministic.
const makeRequest = (index: number): RequestHistoryItem => ({
  id: `bench-req-${String(index).padStart(2, "0")}`,
  method: "POST",
  url: "/api/v1/playground/completions",
  timestamp: "2026-10-06T00:00:00.000Z",
  requestHeaders: { "content-type": "application/json" },
  requestBody: { prompt: "bench" },
  responseBody: { blob: bodyBlob },
  status: 200,
  ok: true,
  duration_ms: 12
})

const spreadPercent = (values: number[]): number => {
  const min = Math.min(...values)
  const max = Math.max(...values)
  return min === 0 ? Number.POSITIVE_INFINITY : ((max - min) / min) * 100
}

type RunMetrics = {
  run: number
  setItemCallsAttempted: number
  setItemCallsLanded: number
  setItemQuotaErrors: number
  setItemBytesAttempted: number
  setItemBytesLanded: number
  finalHistoryEntries: number
  finalHistoryBytes: number
  bytesPerRequest: number
  amplificationVsFinalHistory: number
  durationMs: number
}

describe("request history bench (addRequestHistory 50x with 100 KiB bodies)", () => {
  it("records total setItem bytes written for 50 recorded requests", async () => {
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem")

    try {
      const runs: RunMetrics[] = []
      for (let run = 1; run <= MEASUREMENT_RUNS; run += 1) {
        localStorage.clear()
        // Reset the history module's in-memory buffer too, so each run starts
        // from the same empty state (the buffer survives localStorage.clear).
        clearRequestHistory()
        setItemSpy.mockClear()

        const startedAt = performance.now()
        for (let index = 1; index <= REQUESTS; index += 1) {
          addRequestHistory(makeRequest(index))
        }
        const durationMs = performance.now() - startedAt
        // Let the trailing-debounce flush land so it is measured.
        await new Promise((resolve) => setTimeout(resolve, FLUSH_SETTLE_MS))

        let setItemBytesLanded = 0
        let setItemCallsLanded = 0
        let setItemQuotaErrors = 0
        setItemSpy.mock.calls.forEach((call, callIndex) => {
          const result = setItemSpy.mock.results[callIndex]
          if (result.type === "throw") {
            setItemQuotaErrors += 1
            return
          }
          setItemCallsLanded += 1
          setItemBytesLanded += utf8Bytes(String(call[1]))
        })
        const setItemCallsAttempted = setItemSpy.mock.calls.length
        const setItemBytesAttempted = setItemSpy.mock.calls.reduce<number>(
          (total, call) => total + utf8Bytes(String(call[1])),
          0
        )

        const finalHistoryRaw = localStorage.getItem(HISTORY_KEY) || ""
        const finalHistoryEntries = finalHistoryRaw
          ? (JSON.parse(finalHistoryRaw) as unknown[]).length
          : 0
        const finalHistoryBytes = utf8Bytes(finalHistoryRaw)

        const metrics: RunMetrics = {
          run,
          setItemCallsAttempted,
          setItemCallsLanded,
          setItemQuotaErrors,
          setItemBytesAttempted,
          setItemBytesLanded,
          finalHistoryEntries,
          finalHistoryBytes,
          bytesPerRequest: Math.round(setItemBytesAttempted / REQUESTS),
          amplificationVsFinalHistory: Number(
            (setItemBytesAttempted / Math.max(1, finalHistoryBytes)).toFixed(2)
          ),
          durationMs: Number(durationMs.toFixed(2))
        }
        runs.push(metrics)
        console.log(
          `[history.bench] run ${run} ${JSON.stringify(metrics)}`
        )

        // The harness drove the intended path: the whole 50-request burst is
        // coalesced into exactly one batched setItem per debounce window (the
        // settle wait guarantees it landed), and stored response bodies are
        // truncated at insert (2 KB cap, head kept).
        expect(setItemCallsAttempted).toBe(1)
        expect(finalHistoryEntries).toBeGreaterThan(0)
        expect(finalHistoryEntries).toBeLessThanOrEqual(REQUESTS)
        const finalHistory = JSON.parse(
          localStorage.getItem(HISTORY_KEY) || "[]"
        ) as Array<{ responseBody?: unknown }>
        for (const entry of finalHistory) {
          const stored = String(entry.responseBody)
          expect(stored.startsWith('{"blob":"xxxx')).toBe(true)
          expect(stored).toMatch(/…\[truncated \d+ chars\]$/)
          expect(stored.length).toBeLessThanOrEqual(2048)
        }

        // Measurement-recorded sanity assertions: finite, non-negative, non-zero.
        expect(Number.isFinite(setItemBytesAttempted)).toBe(true)
        expect(Number.isFinite(setItemBytesLanded)).toBe(true)
        expect(Number.isFinite(finalHistoryBytes)).toBe(true)
        expect(Number.isFinite(durationMs)).toBe(true)
        expect(setItemCallsLanded).toBeGreaterThan(0)
        expect(setItemBytesAttempted).toBeGreaterThan(0)
        expect(setItemBytesLanded).toBeGreaterThan(0)
        expect(finalHistoryBytes).toBeGreaterThan(0)
        expect(durationMs).toBeGreaterThanOrEqual(0)
      }

      const summary = {
        requests: REQUESTS,
        bodyBytes: BODY_BYTES,
        runs: MEASUREMENT_RUNS,
        setItemCallsAttempted: runs.map((entry) => entry.setItemCallsAttempted),
        setItemCallsLanded: runs.map((entry) => entry.setItemCallsLanded),
        setItemQuotaErrors: runs.map((entry) => entry.setItemQuotaErrors),
        setItemBytesAttempted: runs.map((entry) => entry.setItemBytesAttempted),
        setItemBytesLanded: runs.map((entry) => entry.setItemBytesLanded),
        finalHistoryEntries: runs.map((entry) => entry.finalHistoryEntries),
        finalHistoryBytes: runs.map((entry) => entry.finalHistoryBytes),
        bytesPerRequest: runs.map((entry) => entry.bytesPerRequest),
        amplificationVsFinalHistory: runs.map(
          (entry) => entry.amplificationVsFinalHistory
        ),
        durationMs: runs.map((entry) => entry.durationMs),
        spreadPctSetItemBytesAttempted: Number(
          spreadPercent(runs.map((entry) => entry.setItemBytesAttempted)).toFixed(2)
        ),
        spreadPctFinalHistoryBytes: Number(
          spreadPercent(runs.map((entry) => entry.finalHistoryBytes)).toFixed(2)
        ),
        spreadPctDurationMs: Number(
          spreadPercent(runs.map((entry) => entry.durationMs)).toFixed(2)
        )
      }
      console.log(`[history.bench] summary ${JSON.stringify(summary)}`)

      // Reproducibility (informational, not a perf threshold): byte totals are
      // deterministic for this fixture, so the spread across runs must be 0.
      expect(summary.spreadPctSetItemBytesAttempted).toBe(0)
      expect(summary.spreadPctFinalHistoryBytes).toBe(0)
      expect(new Set(summary.setItemCallsAttempted).size).toBe(1)
      expect(new Set(summary.setItemCallsLanded).size).toBe(1)
    } finally {
      setItemSpy.mockRestore()
      localStorage.clear()
    }
  })
})
