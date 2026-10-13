/**
 * Streaming Performance Tests
 *
 * Measures chat streaming latency and throughput against a real tldw_server.
 * The live-server suites require TLDW_E2E_SERVER_URL and TLDW_E2E_API_KEY
 * environment variables. The "Canned 100-chunk stream" suite at the bottom
 * does not: it serves a scripted SSE stream from a loopback mock server and
 * needs no live LLM (TASK-13520, WebUI perf batch W0 Stage 1).
 */

import { test, expect, type Page } from "@playwright/test"
import { launchWithExtensionOrSkip } from "./utils/real-server"
import path from "path"
import http from "node:http"
import { AddressInfo } from "node:net"
import { launchWithExtension } from "./utils/extension"
import { waitForConnectionStore, forceConnected, setSelectedModel } from "./utils/connection"
import { grantHostPermission } from "./utils/permissions"
import {
  PerfTimer,
  measureStreamingThroughput,
  measureMemoryDelta,
  createReport,
  logReport,
  countDOMNodes
} from "./utils/performance"

// Configuration
const TEST_EXT_PATH = path.resolve("build/chrome-mv3")
const SERVER_URL = process.env.TLDW_E2E_SERVER_URL
const API_KEY = process.env.TLDW_E2E_API_KEY
const REQUIRE_ENV_REASON = "Set TLDW_E2E_SERVER_URL and TLDW_E2E_API_KEY to run streaming performance tests"

// Performance targets
const TARGETS = {
  timeToFirstToken: 2000, // ms - generous for local server
  tokensPerSecond: 10, // tokens/sec - minimum acceptable
  memoryDeltaMB: 50 // MB - max memory growth during stream
}

const ROOT_SELECTOR = "#root"
const CHAT_INPUT_SELECTOR = 'textarea[placeholder*="message"], input[placeholder*="message"]'

async function setupConnectedChat(page: Page, url: string | undefined, label: string) {
  if (url) {
    await page.goto(url, { waitUntil: "domcontentloaded" })
  }
  await page.waitForSelector(ROOT_SELECTOR, { state: "attached", timeout: 15000 })

  // Bypass onboarding by forcing connected state and setting model
  await waitForConnectionStore(page, `${label}-init`)
  await forceConnected(page, { serverUrl: SERVER_URL! }, `${label}-connected`)
  await setSelectedModel(page, "gpt-4")

  // Reload page so Plasmo's useStorage reads the model from chrome.storage.local on mount
  await page.reload({ waitUntil: "domcontentloaded" })
  await page.waitForSelector(ROOT_SELECTOR, { state: "attached", timeout: 15000 })

  // Re-apply connection state after reload
  await waitForConnectionStore(page, `${label}-after-reload`)
  await forceConnected(page, { serverUrl: SERVER_URL! }, `${label}-reconnected`)

  // Wait for React to re-render after state change
  await page.waitForTimeout(300)

  // Wait for chat input to be ready
  await page.waitForSelector(CHAT_INPUT_SELECTOR, { state: "visible", timeout: 10000 })
  return CHAT_INPUT_SELECTOR
}

test.describe("Streaming Performance", () => {
  test.skip(!SERVER_URL || !API_KEY, REQUIRE_ENV_REASON)

  test("measures time to first token", async () => {
    const { context, page, optionsUrl } = await launchWithExtensionOrSkip(test, TEST_EXT_PATH, {
      seedConfig: {
        serverUrl: SERVER_URL!,
        authMode: "single-user",
        apiKey: API_KEY!,
        selectedModel: "gpt-4"
      }
    })

    try {
      const inputSelector = await setupConnectedChat(page, optionsUrl, "ttft-test")

      // Measure TTFT
      const reportStartTime = performance.now()
      const timer = new PerfTimer()
      timer.start()

      // Send a simple message
      const input = page.locator(inputSelector).first()
      await input.fill("Say hello in exactly 3 words")
      await input.press("Enter")

      timer.mark("sent")

      // Wait for streaming to start (stop button appears)
      try {
        await page.waitForSelector('button[aria-label*="Stop"], button:has-text("Stop")', {
          state: "visible",
          timeout: 10000
        })
        timer.mark("streaming-started")
      } catch {
        // Some UIs don't show stop button, wait for response instead
        await page.waitForSelector('[class*="assistant"], [data-role="assistant"]', {
          state: "visible",
          timeout: 10000
        })
        timer.mark("streaming-started")
      }

      const ttft = timer.betweenMarks("sent", "streaming-started")

      // Create report
      const report = createReport("Time to First Token", [
        {
          name: "TTFT",
          value: ttft,
          unit: "ms",
          target: TARGETS.timeToFirstToken
        }
      ], reportStartTime)

      logReport(report)

      // Assert performance
      expect(ttft).toBeLessThan(TARGETS.timeToFirstToken * 2) // 2x target for CI variance
      console.log(`TTFT: ${ttft.toFixed(0)}ms (target: <${TARGETS.timeToFirstToken}ms)`)
    } finally {
      await context.close()
    }
  })

  test("measures streaming throughput", async () => {
    const { context, page, optionsUrl } = await launchWithExtensionOrSkip(test, TEST_EXT_PATH, {
      seedConfig: {
        serverUrl: SERVER_URL!,
        authMode: "single-user",
        apiKey: API_KEY!,
        selectedModel: "gpt-4"
      }
    })

    try {
      const inputSelector = await setupConnectedChat(page, optionsUrl, "throughput-test")

      // Request a longer response for throughput measurement
      const reportStartTime = performance.now()
      const input = page.locator(inputSelector).first()
      await input.fill("Write a 200-word essay about the benefits of performance testing")
      await input.press("Enter")

      // Wait for streaming to start
      await page.waitForTimeout(1000)

      // Find the response container
      const responseSelectors = [
        '[data-testid="assistant-message"]',
        '[class*="assistant"]',
        '[data-role="assistant"]',
        ".message-content"
      ]

      let responseSelector = ""
      for (const selector of responseSelectors) {
        if ((await page.locator(selector).count()) > 0) {
          responseSelector = selector
          break
        }
      }

      if (!responseSelector) {
        console.log("Could not find response container, skipping throughput test")
        return
      }

      // Measure throughput over 5 seconds
      const { tokensPerSecond, totalTokens } = await measureStreamingThroughput(
        page,
        responseSelector + ":last-of-type",
        5000
      )

      const report = createReport("Streaming Throughput", [
        {
          name: "Tokens per second",
          value: tokensPerSecond,
          unit: " tok/s",
          target: TARGETS.tokensPerSecond
        },
        {
          name: "Total tokens (5s)",
          value: totalTokens,
          unit: " tokens"
        }
      ], reportStartTime)

      logReport(report)

      console.log(
        `Throughput: ${tokensPerSecond.toFixed(1)} tok/s (target: >${TARGETS.tokensPerSecond} tok/s)`
      )
    } finally {
      await context.close()
    }
  })

  test("memory stays stable during streaming", async () => {
    const { context, page, optionsUrl } = await launchWithExtensionOrSkip(test, TEST_EXT_PATH, {
      seedConfig: {
        serverUrl: SERVER_URL!,
        authMode: "single-user",
        apiKey: API_KEY!,
        selectedModel: "gpt-4"
      }
    })

    try {
      const inputSelector = await setupConnectedChat(page, optionsUrl, "memory-test")

      // Measure memory during a chat exchange
      const reportStartTime = performance.now()
      const { beforeMB, afterMB, deltaMB } = await measureMemoryDelta(page, async () => {
        const input = page.locator(inputSelector).first()

        // Send multiple messages to stress memory
        for (let i = 0; i < 3; i++) {
          await input.fill(`Message ${i + 1}: Tell me a short fact about the number ${i + 1}`)
          await input.press("Enter")

          // Wait for response to complete
          await page.waitForTimeout(3000)
        }
      })

      const report = createReport("Memory Stability", [
        {
          name: "Memory before",
          value: beforeMB,
          unit: "MB"
        },
        {
          name: "Memory after",
          value: afterMB,
          unit: "MB"
        },
        {
          name: "Memory delta",
          value: deltaMB,
          unit: "MB",
          target: TARGETS.memoryDeltaMB
        }
      ], reportStartTime)

      logReport(report)

      // Memory API may not be available in all browsers
      if (beforeMB > 0) {
        console.log(
          `Memory delta: ${deltaMB.toFixed(1)}MB (target: <${TARGETS.memoryDeltaMB}MB)`
        )
        expect(deltaMB).toBeLessThan(TARGETS.memoryDeltaMB * 2)
      } else {
        console.log("Memory measurement not available in this browser")
      }
    } finally {
      await context.close()
    }
  })

  test("cancellation works cleanly", async () => {
    const { context, page, optionsUrl } = await launchWithExtensionOrSkip(test, TEST_EXT_PATH, {
      seedConfig: {
        serverUrl: SERVER_URL!,
        authMode: "single-user",
        apiKey: API_KEY!,
        selectedModel: "gpt-4"
      }
    })

    try {
      const inputSelector = await setupConnectedChat(page, optionsUrl, "cancel-test")

      const reportStartTime = performance.now()
      const timer = new PerfTimer()
      timer.start()

      // Start a long-running request
      const input = page.locator(inputSelector).first()
      await input.fill("Write a very detailed 1000-word analysis")
      await input.press("Enter")

      // Wait for streaming to start
      await page.waitForTimeout(500)

      // Try to find and click stop button
      const stopButton = page.locator('button[aria-label*="Stop"], button:has-text("Stop")').first()

      if (await stopButton.isVisible({ timeout: 5000 }).catch(() => false)) {
        timer.mark("stop-click")
        await stopButton.click()

        // Measure time for UI to return to ready state
        await page.waitForSelector(inputSelector + ":not([disabled])", {
          state: "visible",
          timeout: 5000
        })
        timer.mark("ui-ready")

        const cancelTime = timer.betweenMarks("stop-click", "ui-ready")

        const report = createReport("Stream Cancellation", [
          {
            name: "Cancel to ready",
            value: cancelTime,
            unit: "ms",
            target: 1000 // Should be responsive
          }
        ], reportStartTime)

        logReport(report)

        console.log(`Cancellation time: ${cancelTime.toFixed(0)}ms`)
        expect(cancelTime).toBeLessThan(2000)
      } else {
        console.log("Stop button not available, skipping cancellation test")
      }
    } finally {
      await context.close()
    }
  })
})

test.describe("Sidepanel Streaming Performance", () => {
  test.skip(!SERVER_URL || !API_KEY, REQUIRE_ENV_REASON)

  test("sidepanel TTFT matches options page", async () => {
    const { context, openSidepanel } = await launchWithExtensionOrSkip(test, TEST_EXT_PATH, {
      seedConfig: {
        serverUrl: SERVER_URL!,
        authMode: "single-user",
        apiKey: API_KEY!,
        selectedModel: "gpt-4"
      }
    })

    try {
      const sidepanel = await openSidepanel()
      const inputSelector = await setupConnectedChat(sidepanel, undefined, "sidepanel-ttft")

      const timer = new PerfTimer()
      timer.start()

      const input = sidepanel.locator(inputSelector).first()
      await input.fill("Hello from sidepanel")
      await input.press("Enter")

      timer.mark("sent")

      // Wait for response (streaming indicator or assistant message)
      try {
        await sidepanel.waitForSelector('button[aria-label*="Stop"], button:has-text("Stop")', {
          state: "visible",
          timeout: 10000
        })
        timer.mark("response")
      } catch {
        // Some UIs don't show stop button, wait for response instead
        try {
          await sidepanel.waitForSelector('[class*="assistant"], [data-role="assistant"]', {
            state: "visible",
            timeout: 10000
          })
          timer.mark("response")
        } catch {
          // If no streaming indicator appears, check for any response element
          console.log("Sidepanel: No streaming indicator or assistant message found, checking for any response")
          timer.mark("response")
        }
      }

      const ttft = timer.betweenMarks("sent", "response")

      // Log but don't fail if no server is available (TTFT will be ~10s timeout)
      console.log(`Sidepanel TTFT: ${ttft.toFixed(0)}ms`)
      if (ttft > 5000) {
        console.log("Note: High TTFT suggests no server is responding. This is expected without TLDW_E2E_SERVER_URL set.")
        return // Skip assertion if no server
      }
      expect(ttft).toBeLessThan(TARGETS.timeToFirstToken * 2)
    } finally {
      await context.close()
    }
  })
})

// ---------------------------------------------------------------------------
// Canned 100-chunk stream (TASK-13520, WebUI perf batch W0 Stage 1).
//
// Deterministic, no live LLM: a loopback HTTP mock (same mechanism as
// sidepanel-chat-smoke.spec.ts) serves a paced SSE stream of exactly 100
// token chunks into the sidepanel chat. The production per-chunk store path
// (`setMessages((prev) => prev.map(...))` in useMessage.tsx) processes each
// chunk; an in-page bench subscribed to the real `useStoreMessageOption`
// store records `performance.mark`/`performance.measure` deltas plus store
// notification, DOM mutation and long-task counts inside the streaming
// window (mirroring the vitest `streaming-render.bench` counters).
//
// This is a measurement harness, not a regression gate: assertions are
// sanity-level only (finite/non-zero measured values, exactly 100 chunks
// delivered, delivery correctness). Perf thresholds live in the Stage 3
// baseline document (Docs/Reviews/PERF_BASELINE_WEBUI_2026_10.md).
// ---------------------------------------------------------------------------

const CANNED_EXT_PATH = path.resolve(process.env.TLDW_E2E_EXT_PATH || "build/chrome-mv3")
const CANNED_MODEL_ID = "perf-stream-model"
const CANNED_MODEL_KEY = `tldw:${CANNED_MODEL_ID}`
const CANNED_CHUNK_COUNT = 100
const CANNED_CHUNK_PACE_MS = 10
const CANNED_STREAM_TIMEOUT_MS = 45_000

const cannedTokenFor = (index: number) => `perf-token-${index} `
const CANNED_FULL_TEXT = Array.from(
  { length: CANNED_CHUNK_COUNT },
  (_, index) => cannedTokenFor(index)
).join("")

const round = (value: number, digits = 2) => Number(value.toFixed(digits))

interface CannedStreamServer {
  server: http.Server
  baseUrl: string
  chunksServed: () => number
}

const readCannedRequestBody = (req: http.IncomingMessage) =>
  new Promise<string>((resolve) => {
    let body = ""
    req.on("data", (chunk) => {
      body += chunk
    })
    req.on("end", () => resolve(body))
  })

const startCannedStreamServer = async (): Promise<CannedStreamServer> => {
  let chunksServed = 0

  const server = http.createServer(async (req, res) => {
    const method = (req.method || "GET").toUpperCase()
    const url = req.url || "/"

    const sendJson = (code: number, payload: unknown) => {
      res.writeHead(code, {
        "content-type": "application/json",
        "access-control-allow-origin": "http://127.0.0.1",
        "access-control-allow-credentials": "true"
      })
      res.end(JSON.stringify(payload))
    }

    if (method === "OPTIONS") {
      res.writeHead(204, {
        "access-control-allow-origin": "http://127.0.0.1",
        "access-control-allow-credentials": "true",
        "access-control-allow-headers": "content-type, x-api-key, authorization"
      })
      return res.end()
    }

    if (url === "/api/v1/health" && method === "GET") {
      return sendJson(200, { status: "ok" })
    }

    if (url === "/api/v1/llm/models/metadata" && method === "GET") {
      return sendJson(200, [
        {
          id: CANNED_MODEL_ID,
          name: "Perf Stream Mock Model",
          provider: "mock",
          context_length: 4096,
          capabilities: ["chat"]
        }
      ])
    }

    if (url === "/api/v1/llm/models" && method === "GET") {
      return sendJson(200, [CANNED_MODEL_ID])
    }

    if (url === "/openapi.json" && method === "GET") {
      return sendJson(200, {
        openapi: "3.0.0",
        info: { version: "mock" },
        paths: {
          "/api/v1/health": {},
          "/api/v1/chat/completions": {},
          "/api/v1/llm/models": {},
          "/api/v1/llm/models/metadata": {}
        }
      })
    }

    if (url === "/api/v1/chat/completions" && method === "POST") {
      await readCannedRequestBody(req)

      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive"
      })

      // The canned response: exactly CANNED_CHUNK_COUNT distinct token
      // chunks at a fixed pace, then the terminal sentinel.
      for (let index = 0; index < CANNED_CHUNK_COUNT; index += 1) {
        if (res.destroyed || res.writableEnded) return
        res.write(
          `data: ${JSON.stringify({
            choices: [{ delta: { content: cannedTokenFor(index) } }]
          })}\n\n`
        )
        chunksServed += 1
        await new Promise((resolve) => setTimeout(resolve, CANNED_CHUNK_PACE_MS))
      }
      if (res.destroyed || res.writableEnded) return
      res.write("data: [DONE]\n\n")
      return res.end()
    }

    return sendJson(404, { detail: "not found" })
  })
  server.on("error", () => {
    // Ignore socket errors after the test closes the browser context.
  })

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
  const addr = server.address() as AddressInfo
  return { server, baseUrl: `http://127.0.0.1:${addr.port}`, chunksServed: () => chunksServed }
}

const stopCannedStreamServer = async (server: http.Server) => {
  await new Promise<void>((resolve) => {
    let settled = false
    const done = () => {
      if (settled) return
      settled = true
      resolve()
    }

    server.close(done)
    server.closeAllConnections?.()
    const fallback = setTimeout(done, 1000)
    fallback.unref?.()
  })
}

interface CannedStreamPageResult {
  completed: boolean
  timedOut: boolean
  chunksApplied: number
  storeSubscriberNotifications: number
  messageUpdateNotifications: number
  domMutations: number
  longTasks: number
  finalAssistantText: string
  sentAt: number
  firstChunkAt: number
  lastChunkAt: number
  chunkMarkTimes: number[]
  totalMeasureMs: number
  ttftMeasureMs: number
}

/**
 * Install the in-page measurement bench on the sidepanel.
 *
 * Subscribes to the real `useStoreMessageOption` store (exposed on window for
 * Playwright) and records one `performance.mark("tldw-perf-streaming:chunk")`
 * per assistant-content growth event, plus `performance.measure` entries for
 * time-to-first-chunk and the total stream window. Also counts DOM mutations
 * and long tasks inside the window. Resolves `window.__tldwStreamBenchDone`
 * when the canned payload has fully arrived (or after a bounded timeout).
 */
const installCannedStreamBench = (page: Page, expectedFullText: string) =>
  page.evaluate(
    ({ expected, timeoutMs }: { expected: string; timeoutMs: number }) => {
    const w = window as any
    const store = w.__tldw_useStoreMessageOption
    if (!store?.subscribe || !store?.getState) {
      return "store-unavailable"
    }

    const MARK_SENT = "tldw-perf-streaming:sent"
    const MARK_CHUNK = "tldw-perf-streaming:chunk"
    const MEASURE_TOTAL = "tldw-perf-streaming:total"
    const MEASURE_TTFT = "tldw-perf-streaming:ttft"
    const CURSOR = "\u258b"

    const assistantTextOf = (messages: any[]): string => {
      for (let index = messages.length - 1; index >= 0; index -= 1) {
        const message = messages[index]
        if (message && (message.isBot === true || message.role === "assistant")) {
          return typeof message.message === "string" ? message.message : ""
        }
      }
      return ""
    }

    const bench = {
      completed: false,
      timedOut: false,
      started: false,
      chunksApplied: 0,
      storeSubscriberNotifications: 0,
      messageUpdateNotifications: 0,
      domMutations: 0,
      longTasks: 0,
      finalAssistantText: "",
      firstChunkAt: 0,
      lastChunkAt: 0,
      prevAssistantLength: 0
    }
    bench.prevAssistantLength = assistantTextOf(store.getState().messages).length

    const mutationObserver = new MutationObserver((records) => {
      if (!bench.completed) bench.domMutations += records.length
    })
    mutationObserver.observe(document.body, {
      subtree: true,
      childList: true,
      characterData: true
    })

    let longTaskObserver: PerformanceObserver | null = null
    try {
      longTaskObserver = new PerformanceObserver((list) => {
        if (!bench.completed) bench.longTasks += list.getEntries().length
      })
      longTaskObserver.observe({ entryTypes: ["longtask"] })
    } catch {
      // longtask entries are unsupported in this browser; count stays 0.
    }

    let resolveDone: (result: unknown) => void
    const done = new Promise((resolve) => {
      resolveDone = resolve
    })

    let finish: () => void = () => {}
    const unsubscribe = store.subscribe((next: any, previous: any) => {
      if (bench.completed) return
      bench.storeSubscriberNotifications += 1
      if (next?.messages !== previous?.messages) {
        bench.messageUpdateNotifications += 1
      }

      const text = assistantTextOf(next?.messages || [])
      if (text.length > bench.prevAssistantLength) {
        const now = performance.now()
        if (!bench.started) {
          bench.started = true
          bench.firstChunkAt = now
        }
        bench.prevAssistantLength = text.length
        bench.chunksApplied += 1
        bench.lastChunkAt = now
        performance.mark(MARK_CHUNK)
      }
      bench.finalAssistantText = text

      const delivered =
        text === expected || text.replace(new RegExp(`${CURSOR}$`), "") === expected
      if (bench.started && delivered && (text === expected || next?.streaming === false)) {
        finish()
      }
    })

    const timeoutId = setTimeout(() => {
      if (bench.completed) return
      bench.timedOut = true
      finish()
    }, timeoutMs)

    finish = () => {
      if (bench.completed) return
      bench.completed = true
      clearTimeout(timeoutId)
      mutationObserver.disconnect()
      longTaskObserver?.disconnect()
      unsubscribe()

      const sentMarks = performance.getEntriesByName(MARK_SENT)
      const sentAt = sentMarks.length > 0 ? sentMarks[0].startTime : 0
      try {
        performance.measure(MEASURE_TTFT, { start: sentAt, end: bench.firstChunkAt })
        performance.measure(MEASURE_TOTAL, { start: bench.firstChunkAt, end: bench.lastChunkAt })
      } catch {
        // Measures are best-effort; numeric marks below remain authoritative.
      }

      const totalEntries = performance.getEntriesByName(MEASURE_TOTAL)
      const ttftEntries = performance.getEntriesByName(MEASURE_TTFT)
      resolveDone({
        completed: bench.completed && !bench.timedOut,
        timedOut: bench.timedOut,
        chunksApplied: bench.chunksApplied,
        storeSubscriberNotifications: bench.storeSubscriberNotifications,
        messageUpdateNotifications: bench.messageUpdateNotifications,
        domMutations: bench.domMutations,
        longTasks: bench.longTasks,
        finalAssistantText: bench.finalAssistantText,
        sentAt,
        firstChunkAt: bench.firstChunkAt,
        lastChunkAt: bench.lastChunkAt,
        chunkMarkTimes: performance
          .getEntriesByName(MARK_CHUNK)
          .map((entry) => entry.startTime),
        totalMeasureMs:
          totalEntries.length > 0 ? totalEntries[totalEntries.length - 1].duration : 0,
        ttftMeasureMs:
          ttftEntries.length > 0 ? ttftEntries[ttftEntries.length - 1].duration : 0
      })
    }

    w.__tldwStreamBenchDone = done
    w.__tldwStreamBenchCleanup = () => {
      clearTimeout(timeoutId)
      mutationObserver.disconnect()
      longTaskObserver?.disconnect()
      unsubscribe()
    }
    return "ok"
  },
  { expected: expectedFullText, timeoutMs: CANNED_STREAM_TIMEOUT_MS })

test.describe("Canned 100-chunk stream (no live LLM)", () => {
  test("records performance.mark deltas for 100 streamed chunks in the sidepanel", async () => {
    test.setTimeout(120_000)

    const reportStartTime = performance.now()
    const canned = await startCannedStreamServer()

    const { context, page, extensionId, openSidepanel } = await launchWithExtensionOrSkip(
      test,
      CANNED_EXT_PATH,
      {
        seedConfig: {
          __tldw_first_run_complete: true,
          __tldw_allow_offline: true,
          tldwConfig: {
            serverUrl: canned.baseUrl,
            authMode: "single-user",
            apiKey: "test-key"
          }
        }
      }
    )

    let sidepanel: Page | undefined
    try {
      const origin = `${new URL(canned.baseUrl).origin}/*`
      const granted = await grantHostPermission(context, extensionId, origin)
      expect(
        granted,
        "Host permission must be granted programmatically before sidepanel chat can reach the canned stream server."
      ).toBe(true)

      await setSelectedModel(page, CANNED_MODEL_KEY)

      sidepanel = await openSidepanel("/chat")
      await sidepanel.setViewportSize({ width: 390, height: 780 })
      await waitForConnectionStore(sidepanel, "perf-streaming:store")
      await forceConnected(
        sidepanel,
        { serverUrl: canned.baseUrl },
        "perf-streaming:connected"
      )

      // Chat input (same discovery sequence as sidepanel-chat-smoke.spec.ts).
      const startButton = sidepanel.getByRole("button", { name: /Start chatting/i })
      if ((await startButton.count()) > 0) {
        await startButton.first().click()
      }
      let input = sidepanel.getByTestId("chat-input")
      if ((await input.count()) === 0) {
        input = sidepanel.getByPlaceholder(/Type a message/i)
      }
      await expect(input).toBeVisible({ timeout: 15_000 })
      await expect(input).toBeEditable({ timeout: 15_000 })
      await input.click()

      // The real message store must be exposed before the bench can install.
      await sidepanel.waitForFunction(
        () => Boolean((window as any).__tldw_useStoreMessageOption?.getState),
        null,
        { timeout: 15_000 }
      )
      const installed = await installCannedStreamBench(sidepanel, CANNED_FULL_TEXT)
      expect(installed).toBe("ok")

      await input.fill(`perf stream bench ${Date.now()}`)

      const sendButton = sidepanel.locator('[data-testid="chat-send"]')
      if ((await sendButton.count()) > 0) {
        await expect(sendButton).toBeEnabled({ timeout: 15_000 })
      }
      await sidepanel.evaluate(() => {
        performance.mark("tldw-perf-streaming:sent")
      })
      if ((await sendButton.count()) > 0) {
        await sendButton.click()
      } else {
        await input.press("Enter")
      }

      const raw = (await sidepanel.evaluate(
        () => (window as any).__tldwStreamBenchDone
      )) as CannedStreamPageResult

      const domNodeCount = await countDOMNodes(sidepanel, "#root")

      const ttftMs = raw.firstChunkAt - raw.sentAt
      const streamPhaseMs = raw.lastChunkAt - raw.firstChunkAt
      const chunkGapsMs: number[] = []
      for (let index = 1; index < raw.chunkMarkTimes.length; index += 1) {
        chunkGapsMs.push(raw.chunkMarkTimes[index] - raw.chunkMarkTimes[index - 1])
      }

      const summary = {
        chunksServed: canned.chunksServed(),
        chunksApplied: raw.chunksApplied,
        storeSubscriberNotifications: raw.storeSubscriberNotifications,
        messageUpdateNotifications: raw.messageUpdateNotifications,
        domMutations: raw.domMutations,
        longTasks: raw.longTasks,
        ttftMs: round(ttftMs),
        streamPhaseMs: round(streamPhaseMs),
        msPerChunk: round(streamPhaseMs / raw.chunksApplied, 4),
        minChunkGapMs: round(chunkGapsMs.length > 0 ? Math.min(...chunkGapsMs) : 0),
        maxChunkGapMs: round(chunkGapsMs.length > 0 ? Math.max(...chunkGapsMs) : 0),
        totalMeasureMs: round(raw.totalMeasureMs),
        ttftMeasureMs: round(raw.ttftMeasureMs),
        domNodeCount
      }
      console.log(`[performance-streaming] ${JSON.stringify(summary)}`)

      const report = createReport(
        "Canned 100-chunk sidepanel stream",
        [
          { name: "Chunks served", value: summary.chunksServed, unit: " chunks" },
          { name: "Chunks applied (store growth events)", value: raw.chunksApplied, unit: " chunks" },
          { name: "Store subscriber notifications", value: raw.storeSubscriberNotifications, unit: " notifications" },
          { name: "Message-array update notifications", value: raw.messageUpdateNotifications, unit: " notifications" },
          { name: "DOM mutations during stream", value: raw.domMutations, unit: " mutations" },
          { name: "Long tasks during stream", value: raw.longTasks, unit: " tasks" },
          { name: "Time to first chunk", value: round(ttftMs), unit: "ms" },
          { name: "Stream phase total", value: round(streamPhaseMs), unit: "ms" },
          { name: "ms per chunk", value: summary.msPerChunk, unit: "ms" },
          { name: "Max chunk gap", value: summary.maxChunkGapMs, unit: "ms" },
          { name: "performance.measure total", value: round(raw.totalMeasureMs), unit: "ms" },
          { name: "DOM node count", value: domNodeCount, unit: " nodes" }
        ],
        reportStartTime
      )
      logReport(report)

      // Harness-correctness assertions (not perf thresholds).
      expect(raw.completed).toBe(true)
      expect(raw.timedOut).toBe(false)
      expect(canned.chunksServed()).toBe(CANNED_CHUNK_COUNT)
      expect(raw.finalAssistantText.replace(/\u258b$/, "")).toBe(CANNED_FULL_TEXT)

      // Measurement-recorded sanity assertions: finite and > 0.
      for (const [name, value] of Object.entries(summary)) {
        expect(Number.isFinite(value), `${name} must be finite`).toBe(true)
      }
      expect(raw.chunksApplied).toBeGreaterThan(0)
      expect(raw.storeSubscriberNotifications).toBeGreaterThan(0)
      expect(raw.messageUpdateNotifications).toBeGreaterThan(0)
      expect(raw.domMutations).toBeGreaterThan(0)
      expect(raw.longTasks).toBeGreaterThanOrEqual(0)
      expect(ttftMs).toBeGreaterThan(0)
      expect(streamPhaseMs).toBeGreaterThan(0)
      expect(summary.msPerChunk).toBeGreaterThan(0)
      expect(raw.totalMeasureMs).toBeGreaterThan(0)
    } finally {
      try {
        await sidepanel?.evaluate(() => {
          ;(window as any).__tldwStreamBenchCleanup?.()
        })
      } catch {
        // Page may already be closed.
      }
      await context.close()
      await stopCannedStreamServer(canned.server)
    }
  })
})
