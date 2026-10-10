// Request-history persistence micro-benchmark — addRequestHistory (TASK-13520, Batch W0 Stage 2).
//
// Measures how much localStorage write traffic `addRequestHistory` produces when
// 50 requests with 100 KiB bodies are recorded against jsdom localStorage:
// every add re-reads, re-redacts (deep-copies) and re-serializes the ENTIRE
// accumulated history, so total `setItem` bytes grow quadratically
// (sum_{i=1..50} i * ~100 KiB ≈ 127 MiB attempted for a ~5 MiB final history).
//
// jsdom caps localStorage at 5,000,000 code units, so with 100 KiB bodies the
// final two writes exceed the quota and `addRequestHistory` swallows the
// QuotaExceededError. The bench therefore reports BOTH attempted bytes (the
// metric that captures the quadratic rewrite) and landed bytes, plus the
// deterministic quota-error count; this is an environment artifact, not part of
// the measured code path.
//
// Method:
//   - 50 `addRequestHistory` calls, each carrying a 102,400-char requestBody
//     (all entries byte-length identical for deterministic totals).
//   - Passthrough spy on `Storage.prototype.setItem` (jsdom's Storage is a
//     WebIDL proxy; spying the `localStorage` instance stores the spy as a
//     named data entry instead of intercepting — see the sibling workspace
//     bench for the same finding).
//   - Repeated 3x in-process (localStorage cleared between runs) for
//     reproducibility; counts/bytes are deterministic, duration informational.
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

import { addRequestHistory, type RequestHistoryItem } from "@web/lib/history"

const HISTORY_KEY = "tldw-request-history"
const REQUESTS = 50
const BODY_BYTES = 100 * 1024
const MEASUREMENT_RUNS = 3

const textEncoder = new TextEncoder()
const utf8Bytes = (value: string): number => textEncoder.encode(value).length

const bodyBlob = "x".repeat(BODY_BYTES)

// All entries serialize to identical byte lengths (padded ids, fixed fields),
// so per-run totals are deterministic.
const makeRequest = (index: number): RequestHistoryItem => ({
  id: `bench-req-${String(index).padStart(2, "0")}`,
  method: "POST",
  url: "/api/v1/playground/completions",
  timestamp: "2026-10-06T00:00:00.000Z",
  requestHeaders: { "content-type": "application/json" },
  requestBody: { blob: bodyBlob },
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
        setItemSpy.mockClear()

        const startedAt = performance.now()
        for (let index = 1; index <= REQUESTS; index += 1) {
          addRequestHistory(makeRequest(index))
        }
        const durationMs = performance.now() - startedAt

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

        // The harness drove the intended path: one setItem attempt per add,
        // history persisted under the expected key with full-size bodies.
        expect(setItemCallsAttempted).toBe(REQUESTS)
        expect(finalHistoryEntries).toBeGreaterThan(0)
        expect(finalHistoryEntries).toBeLessThanOrEqual(REQUESTS)
        const finalHistory = JSON.parse(
          localStorage.getItem(HISTORY_KEY) || "[]"
        ) as Array<{ requestBody?: { blob?: string } }>
        for (const entry of finalHistory) {
          expect(entry.requestBody?.blob).toBe(bodyBlob)
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
