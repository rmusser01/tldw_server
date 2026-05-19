import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/config/platform", () => ({
  isChromiumTarget: true
}))

import { getScreenshotFromCurrentTab } from "@/libs/get-screenshot"

describe("getScreenshotFromCurrentTab", () => {
  afterEach(() => {
    Reflect.deleteProperty(globalThis, "chrome")
  })

  it("returns a failure when Chromium captureVisibleTab reports no data", async () => {
    Object.defineProperty(globalThis, "chrome", {
      configurable: true,
      value: {
        runtime: {
          lastError: {
            message: "capture blocked"
          }
        },
        tabs: {
          query: (_query: unknown, callback: (tabs: Array<{ id: number }>) => void) =>
            callback([{ id: 1 }]),
          captureVisibleTab: (
            _windowId: number | null,
            _options: unknown,
            callback: (dataUrl?: string) => void
          ) => callback(undefined)
        }
      }
    })

    const result = await getScreenshotFromCurrentTab()

    expect(result).toEqual({
      success: false,
      screenshot: null,
      error: "capture blocked"
    })
  })
})
