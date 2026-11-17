import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  FLUSH_DEBOUNCE_MS,
  addRequestHistory,
  clearRequestHistory,
  getRequestHistory,
  historyInternals,
} from "@web/lib/history"

const KEY = "tldw-request-history"

const makeItem = (overrides: Record<string, unknown> = {}) => ({
  id: `req-${Math.random().toString(36).slice(2, 8)}`,
  method: "GET",
  url: "/api/v1/config",
  timestamp: "2026-06-30T00:00:00.000Z",
  ...overrides,
})

describe("request history", () => {
  let setItemSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    localStorage.clear()
    // Reset the module-level history cache and cancel any pending debounce
    // so tests are isolated from each other.
    clearRequestHistory()
    // Spy the Storage prototype: jsdom's Storage is a WebIDL proxy, so spying
    // the localStorage instance stores the spy as a data entry instead of
    // intercepting (W0 bench finding).
    setItemSpy = vi.spyOn(Storage.prototype, "setItem")
  })

  afterEach(() => {
    // Cancel any pending debounce (under whatever timer mode is active) so a
    // straggler flush can never leak into the next test.
    clearRequestHistory()
    setItemSpy.mockRestore()
    vi.useRealTimers()
  })

  it("omits credential headers before writing request history", () => {
    vi.useFakeTimers()
    addRequestHistory({
      id: "req-1",
      method: "GET",
      url: "/api/v1/config",
      timestamp: "2026-06-30T00:00:00.000Z",
      requestHeaders: {
        Authorization: "Bearer secret-token",
        "x-api-key": "secret-api-key",
        "content-type": "application/json"
      }
    })

    // Nothing is written synchronously; land the debounced flush so the
    // persisted payload itself is asserted, not an empty pre-flush read.
    expect(setItemSpy).not.toHaveBeenCalled()
    vi.advanceTimersByTime(FLUSH_DEBOUNCE_MS)

    const raw = localStorage.getItem(KEY) || ""
    expect(raw).not.toContain("secret-token")
    expect(raw).not.toContain("secret-api-key")

    const [item] = getRequestHistory()
    expect(item.requestHeaders).toEqual({
      "content-type": "application/json"
    })
  })

  it("migrates existing unredacted request history on read", () => {
    localStorage.setItem(KEY, JSON.stringify([
      {
        id: "req-1",
        method: "GET",
        url: "/api/v1/config",
        timestamp: "2026-06-30T00:00:00.000Z",
        requestHeaders: {
          Authorization: "Bearer old-secret",
          Cookie: "sid=old-cookie",
          Accept: "application/json"
        }
      }
    ]))

    const [item] = getRequestHistory()

    expect(item.requestHeaders).toEqual({
      Accept: "application/json"
    })
    const raw = localStorage.getItem(KEY) || ""
    expect(raw).not.toContain("old-secret")
    expect(raw).not.toContain("old-cookie")
  })

  it("coalesces 50 sequential adds into at most one setItem per debounce window", () => {
    vi.useFakeTimers()

    for (let index = 0; index < 50; index += 1) {
      addRequestHistory(makeItem({
        id: `req-${String(index).padStart(2, "0")}`,
        responseBody: { note: `payload-${index}` },
      }))
    }

    // No synchronous write on the request path: the entire burst is buffered.
    expect(setItemSpy).not.toHaveBeenCalled()

    vi.advanceTimersByTime(FLUSH_DEBOUNCE_MS)

    // One debounced flush for the whole window, newest-first, all 50 kept.
    expect(setItemSpy).toHaveBeenCalledTimes(1)
    const stored = JSON.parse(localStorage.getItem(KEY) || "[]") as Array<{ id: string }>
    expect(stored).toHaveLength(50)
    expect(stored[0].id).toBe("req-49")
    expect(stored[49].id).toBe("req-00")

    // A later burst gets its own window (trailing debounce re-arms).
    setItemSpy.mockClear()
    addRequestHistory(makeItem({ id: "req-50" }))
    vi.advanceTimersByTime(FLUSH_DEBOUNCE_MS)
    expect(setItemSpy).toHaveBeenCalledTimes(1)
  })

  it("truncates stored responseBody to 2 KB at insert, exactly once", () => {
    vi.useFakeTimers()

    const bigString = "y".repeat(100 * 1024)
    addRequestHistory(makeItem({
      id: "big-string",
      status: 200,
      ok: true,
      duration_ms: 42,
      responseBody: bigString,
    }))
    const bigObject = { blob: "z".repeat(100 * 1024) }
    addRequestHistory(makeItem({
      id: "big-object",
      responseBody: bigObject,
    }))
    addRequestHistory(makeItem({
      id: "small-object",
      responseBody: { kept: "small-value" },
    }))
    addRequestHistory(makeItem({
      id: "auth-route",
      url: "/api/v1/auth/login",
      responseBody: { access_token: "token-value" },
    }))

    vi.advanceTimersByTime(FLUSH_DEBOUNCE_MS)

    const history = getRequestHistory()
    const byId = new Map(history.map((item) => [item.id, item]))

    const truncatedString = byId.get("big-string")?.responseBody
    expect(typeof truncatedString).toBe("string")
    // Stored value keeps the head of the body and is capped at 2 KB total.
    expect(truncatedString).toMatch(/^y{100}/)
    expect(truncatedString).toMatch(/\[truncated \d+ chars\]$/)
    expect(String(truncatedString).length).toBe(2048)
    // status/duration/url stay full.
    expect(byId.get("big-string")?.status).toBe(200)
    expect(byId.get("big-string")?.duration_ms).toBe(42)
    expect(byId.get("big-string")?.url).toBe("/api/v1/config")

    const truncatedObject = byId.get("big-object")?.responseBody
    expect(typeof truncatedObject).toBe("string")
    expect(String(truncatedObject)).toMatch(/^\{"blob":"z{50}/)
    expect(String(truncatedObject)).toMatch(/\[truncated \d+ chars\]$/)
    expect(String(truncatedObject).length).toBe(2048)

    // Small bodies are stored verbatim with their type intact.
    expect(byId.get("small-object")?.responseBody).toEqual({ kept: "small-value" })

    // Auth-route bodies are dropped entirely by redaction before truncation.
    expect(byId.get("auth-route")?.responseBody).toBe("[REDACTED]")

    // Exactly-once: a second flush window must not re-truncate (the marker
    // prefix would move deeper into the payload if it re-sliced).
    addRequestHistory(makeItem({ id: "one-more" }))
    vi.advanceTimersByTime(FLUSH_DEBOUNCE_MS)
    const after = new Map(getRequestHistory().map((item) => [item.id, item]))
    expect(after.get("big-string")?.responseBody).toBe(truncatedString)
  })

  it("applies redaction exactly once per entry at insert", () => {
    vi.useFakeTimers()
    const redactSpy = vi.spyOn(historyInternals, "redactHistoryItem")

    for (let index = 0; index < 50; index += 1) {
      addRequestHistory(makeItem({ id: `req-${index}` }))
    }

    // Once per inserted entry — never re-redacting the accumulated array
    // (the old quadratic path redacted 1+2+...+50 = 1275 times).
    expect(redactSpy).toHaveBeenCalledTimes(50)

    // Flushing and adding one more entry redacts only the new entry.
    vi.advanceTimersByTime(FLUSH_DEBOUNCE_MS)
    addRequestHistory(makeItem({ id: "req-50" }))
    expect(redactSpy).toHaveBeenCalledTimes(51)
  })

  it("flushes pending entries on visibilitychange and beforeunload", () => {
    addRequestHistory(makeItem({ id: "req-1", responseBody: { a: 1 } }))
    addRequestHistory(makeItem({ id: "req-2" }))

    // Nothing written yet (debounce pending).
    expect(localStorage.getItem(KEY)).toBeNull()

    const visibilitySpy = vi
      .spyOn(document, "visibilityState", "get")
      .mockReturnValue("hidden")
    try {
      document.dispatchEvent(new Event("visibilitychange"))
    } finally {
      visibilitySpy.mockRestore()
    }

    // The visibility flush preserved both pending entries, newest first.
    const onHidden = JSON.parse(localStorage.getItem(KEY) || "[]") as Array<{ id: string }>
    expect(onHidden.map((item) => item.id)).toEqual(["req-2", "req-1"])

    // beforeunload also flushes immediately.
    localStorage.clear()
    clearRequestHistory()
    addRequestHistory(makeItem({ id: "req-3" }))
    window.dispatchEvent(new Event("beforeunload"))
    const onUnload = JSON.parse(localStorage.getItem(KEY) || "[]") as Array<{ id: string }>
    expect(onUnload.map((item) => item.id)).toEqual(["req-3"])
  })

  it("keeps legacy un-truncated entries readable via the getter", () => {
    const legacyBody = { full: "x".repeat(64 * 1024) }
    localStorage.setItem(KEY, JSON.stringify([
      {
        id: "legacy-1",
        method: "POST",
        url: "/api/v1/media/search",
        timestamp: "2026-01-01T00:00:00.000Z",
        requestHeaders: { Authorization: "Bearer legacy-secret" },
        responseBody: legacyBody,
        status: 200,
      },
      {
        id: "legacy-2",
        method: "GET",
        url: "/api/v1/media/search",
        timestamp: "2026-01-02T00:00:00.000Z",
        responseBody: legacyBody,
      },
    ]))

    const history = getRequestHistory()
    expect(history).toHaveLength(2)
    expect(history[0].id).toBe("legacy-1")
    // Legacy bodies are NOT truncated on read; they still parse and display.
    expect(history[0].responseBody).toEqual(legacyBody)
    expect(history[1].responseBody).toEqual(legacyBody)

    // A new insert hydrates the legacy entries instead of clobbering them.
    vi.useFakeTimers()
    addRequestHistory(makeItem({ id: "fresh-1", responseBody: "n".repeat(100 * 1024) }))
    vi.advanceTimersByTime(FLUSH_DEBOUNCE_MS)

    const merged = JSON.parse(localStorage.getItem(KEY) || "[]") as Array<{
      id: string
      responseBody?: unknown
    }>
    expect(merged.map((item) => item.id)).toEqual(["fresh-1", "legacy-1", "legacy-2"])
    expect(merged[1].responseBody).toEqual(legacyBody)
    // While the new entry's body is truncated at insert.
    expect(String(merged[0].responseBody)).toMatch(/\[truncated \d+ chars\]$/)
  })
})
