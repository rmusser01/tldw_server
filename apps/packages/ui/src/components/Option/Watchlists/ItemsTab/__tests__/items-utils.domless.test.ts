import { describe, expect, it, vi } from "vitest"

describe("stripHtmlToText without a browser DOM", () => {
  it("fails closed for nonempty untrusted HTML", async () => {
    const originalWindow = globalThis.window
    vi.stubGlobal("window", undefined)
    vi.resetModules()

    try {
      const { stripHtmlToText } = await import("../items-utils")

      expect(stripHtmlToText("<p>Hello<script>alert(1)</script></p>")).toBe("")
    } finally {
      vi.stubGlobal("window", originalWindow)
      vi.resetModules()
    }
  })
})
