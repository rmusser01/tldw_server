import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("wxt/browser", () => ({ browser: {} }))

import { openSidepanel } from "../sidepanel"

afterEach(() => vi.unstubAllGlobals())

describe("sidebar acknowledgment", () => {
  it("starts opening synchronously and waits for the browser acknowledgment", async () => {
    let opened: () => void = () => {}
    const open = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          opened = resolve
        }),
    )
    vi.stubGlobal("chrome", {
      sidePanel: { setOptions: vi.fn().mockResolvedValue(undefined), open },
    })
    let acknowledged = false

    const opening = openSidepanel(8).then(() => {
      acknowledged = true
    })

    expect(open).toHaveBeenCalledWith({ tabId: 8 })
    await Promise.resolve()
    expect(acknowledged).toBe(false)
    opened()
    await opening
    expect(acknowledged).toBe(true)
  })

  it("propagates browser denial so the caller can explain the failure", async () => {
    vi.stubGlobal("chrome", {
      sidePanel: {
        open: vi.fn().mockRejectedValue(new Error("gesture required")),
      },
    })

    await expect(openSidepanel(8)).rejects.toThrow("gesture required")
  })
})
