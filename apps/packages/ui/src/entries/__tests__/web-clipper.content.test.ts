import { describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  addListener: vi.fn(),
  listener: undefined as
    | ((message: { type?: string; requestedType?: string }) => unknown)
    | undefined
}))

vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineContentScript", {
    configurable: true,
    value: (options: unknown) => options
  })
  return {}
})

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      onMessage: {
        addListener: (listener: unknown) => {
          mocks.listener = listener as typeof mocks.listener
          return mocks.addListener(listener)
        }
      }
    }
  }
}))

import entry from "@/entries/web-clipper.content"

describe("web clipper content script entry", () => {
  it("registers for regular web pages so WXT can build the extension", () => {
    const contentScript = entry as {
      allFrames?: boolean
      matches?: string[]
    }

    expect(contentScript.allFrames).toBe(false)
    expect(contentScript.matches).toEqual(["http://*/*", "https://*/*"])
  })

  it("ignores unrelated runtime messages without claiming them", () => {
    const contentScript = entry as {
      main?: () => void
    }

    contentScript.main?.()

    expect(typeof mocks.listener).toBe("function")
    expect(mocks.listener?.({ type: "other-message" })).toBeUndefined()
  })

  it("prefers the selection text captured by the context menu payload", async () => {
    document.body.innerHTML = "<article><p>Article fallback body</p></article>"
    document.title = "Example Story"

    const contentScript = entry as {
      main?: () => void
    }

    contentScript.main?.()

    const result = await mocks.listener?.({
      type: "capture-web-clipper",
      requestedType: "selection",
      selectionText: "Captured selection"
    })

    expect(result).toMatchObject({
      requestedType: "selection",
      selectionText: "Captured selection",
      visibleBody: "Captured selection",
      captureMetadata: {
        fallbackPath: ["selection"]
      }
    })
  })
})
