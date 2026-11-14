import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  SIDEPANEL_OPEN_GUARD_CAPACITY,
  createBackgroundStatusBroadcaster,
  createSidepanelOpenOnceGuard,
} from "@/services/background-helpers"

describe("createSidepanelOpenOnceGuard", () => {
  it("allows the first open attempt for a funnel and suppresses the rest", () => {
    const guard = createSidepanelOpenOnceGuard()

    // First status emit of a funnel must still reach sidePanel.open — this is
    // the "still opens from closed" regression pin (WP-60).
    expect(guard("ingest-1")).toBe(true)
    expect(guard("ingest-1")).toBe(false)
    expect(guard("ingest-1")).toBe(false)
  })

  it("grants each funnel exactly one open attempt", () => {
    const guard = createSidepanelOpenOnceGuard()

    expect(guard("ingest-1")).toBe(true)
    expect(guard("ingest-2")).toBe(true)
    expect(guard("ingest-1")).toBe(false)
    expect(guard("ingest-2")).toBe(false)
    expect(guard("ingest-3")).toBe(true)
  })

  it("keeps the legacy always-open behavior for uncorrelated broadcasts", () => {
    const guard = createSidepanelOpenOnceGuard()

    expect(guard(undefined)).toBe(true)
    expect(guard(undefined)).toBe(true)
    expect(guard(null)).toBe(true)
    expect(guard("")).toBe(true)
    expect(guard("   ")).toBe(true)
  })

  it("treats funnel ids as case- and whitespace-sensitive trimmed strings", () => {
    const guard = createSidepanelOpenOnceGuard()

    expect(guard("ingest-1")).toBe(true)
    expect(guard(" ingest-1 ")).toBe(true)
    expect(guard("INGEST-1")).toBe(true)
    expect(guard("ingest-1")).toBe(false)
  })

  it("resets the remembered funnels once the capacity is exceeded", () => {
    const guard = createSidepanelOpenOnceGuard(2)

    expect(guard("ingest-1")).toBe(true)
    expect(guard("ingest-2")).toBe(true)
    expect(guard("ingest-1")).toBe(false)

    // Capacity exceeded: the guard forgets everything, so a long-lived worker
    // degrades to at most one extra open attempt per capacity window.
    expect(guard("ingest-3")).toBe(true)
    expect(guard("ingest-1")).toBe(true)
    expect(guard("ingest-3")).toBe(false)
  })

  it("defaults the capacity to the documented constant", () => {
    const guard = createSidepanelOpenOnceGuard()

    for (let index = 0; index < SIDEPANEL_OPEN_GUARD_CAPACITY; index += 1) {
      expect(guard(`ingest-${index}`)).toBe(true)
    }
    expect(guard(`ingest-0`)).toBe(false)

    expect(guard(`ingest-${SIDEPANEL_OPEN_GUARD_CAPACITY}`)).toBe(true)
    expect(guard(`ingest-0`)).toBe(true)
  })
})

describe("createBackgroundStatusBroadcaster", () => {
  const deps = vi.hoisted(() => ({
    openSidepanel: vi.fn(async () => undefined),
    sendMessage: vi.fn(async () => ({ handled: true })),
    logError: vi.fn()
  }))

  let broadcaster: (
    tabId: number | undefined,
    type: string,
    payload?: Record<string, unknown>
  ) => Promise<void>

  beforeEach(() => {
    vi.clearAllMocks()
    deps.sendMessage.mockResolvedValue({ handled: true })
    deps.openSidepanel.mockResolvedValue(undefined)
    broadcaster = createBackgroundStatusBroadcaster({
      openSidepanel: (tabId) => deps.openSidepanel(tabId),
      sendMessage: (message) => deps.sendMessage(message),
      logError: deps.logError
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("opens the sidepanel once per funnel across repeated status broadcasts", async () => {
    await broadcaster(8, "media-ingest-status", { funnelId: "ingest-1" })
    await broadcaster(8, "media-ingest-status", { funnelId: "ingest-1" })
    await broadcaster(8, "media-ingest-status", { funnelId: "ingest-1" })

    expect(deps.openSidepanel).toHaveBeenCalledTimes(1)
    expect(deps.openSidepanel).toHaveBeenCalledWith(8)

    // The broadcast itself is NEVER suppressed — only the redundant
    // sidePanel.setOptions/open calls are.
    expect(deps.sendMessage).toHaveBeenCalledTimes(3)
    expect(deps.sendMessage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        from: "background",
        type: "media-ingest-status",
        payload: expect.objectContaining({ funnelId: "ingest-1" })
      })
    )
  })

  it("still opens the sidepanel on the first status emit of each funnel", async () => {
    await broadcaster(8, "media-ingest-status", { funnelId: "ingest-1" })
    await broadcaster(9, "media-ingest-status", { funnelId: "ingest-2" })

    expect(deps.openSidepanel).toHaveBeenCalledTimes(2)
    expect(deps.openSidepanel).toHaveBeenNthCalledWith(1, 8)
    expect(deps.openSidepanel).toHaveBeenNthCalledWith(2, 9)
  })

  it("opens once per quick-ingest session for session-correlated broadcasts", async () => {
    await broadcaster(undefined, "tldw:quick-ingest/progress", {
      sessionId: "qi-1",
      processedCount: 1
    })
    await broadcaster(undefined, "tldw:quick-ingest/progress", {
      sessionId: "qi-1",
      processedCount: 2
    })
    await broadcaster(undefined, "tldw:quick-ingest/completed", {
      sessionId: "qi-1"
    })

    expect(deps.openSidepanel).toHaveBeenCalledTimes(1)
    expect(deps.sendMessage).toHaveBeenCalledTimes(3)
  })

  it("keeps opening on every uncorrelated broadcast (legacy behavior)", async () => {
    await broadcaster(undefined, "transcription", { text: "one" })
    await broadcaster(undefined, "transcription", { text: "two" })

    expect(deps.openSidepanel).toHaveBeenCalledTimes(2)
  })

  it("still broadcasts when opening the sidepanel is rejected", async () => {
    deps.openSidepanel.mockRejectedValueOnce(new Error("denied"))

    await broadcaster(8, "media-ingest-status", { funnelId: "ingest-1" })

    expect(deps.sendMessage).toHaveBeenCalledTimes(1)
    expect(deps.logError).toHaveBeenCalledWith(
      "open sidebar",
      expect.any(Error)
    )
  })

  it("retries a failed runtime send once after 500ms and delivers it", async () => {
    vi.useFakeTimers()
    deps.sendMessage
      .mockRejectedValueOnce(
        new Error(
          "Could not establish connection. Receiving end does not exist."
        )
      )
      .mockResolvedValue({ handled: true })

    const sent = broadcaster(8, "media-ingest-status", {
      funnelId: "ingest-1"
    })
    await vi.advanceTimersByTimeAsync(0)
    expect(deps.sendMessage).toHaveBeenCalledTimes(1)
    // The first failure is logged immediately (legacy behavior).
    expect(deps.logError).toHaveBeenCalledWith(
      "send media-ingest-status",
      expect.any(Error)
    )

    await vi.advanceTimersByTimeAsync(500)
    await sent

    expect(deps.sendMessage).toHaveBeenCalledTimes(2)
    expect(deps.sendMessage).toHaveBeenLastCalledWith(
      expect.objectContaining({
        from: "background",
        type: "media-ingest-status",
        payload: expect.objectContaining({ funnelId: "ingest-1" })
      })
    )
    expect(deps.logError).not.toHaveBeenCalledWith(
      "send media-ingest-status (retry)",
      expect.any(Error)
    )
  })

  it("logs when the retried runtime send also fails", async () => {
    vi.useFakeTimers()
    deps.sendMessage.mockRejectedValue(new Error("still no receiver"))

    const sent = broadcaster(8, "media-ingest-status", {
      funnelId: "ingest-1"
    })
    await vi.advanceTimersByTimeAsync(0)
    await vi.advanceTimersByTimeAsync(500)
    await sent

    expect(deps.sendMessage).toHaveBeenCalledTimes(2)
    expect(deps.logError).toHaveBeenCalledWith(
      "send media-ingest-status (retry)",
      expect.any(Error)
    )
  })
})
