// @vitest-environment jsdom
/**
 * A clipboard write that the browser refuses must reach the caller, so the
 * copy control can say it failed instead of confirming a copy that did not
 * happen.
 */
import { afterEach, describe, expect, it, vi } from "vitest"

import { copyToClipboard } from "../clipboard"

vi.mock("@/services/app", () => ({
  isRemoveReasoningTagFromCopy: async () => false
}))

const setClipboard = (value: unknown) =>
  Object.defineProperty(navigator, "clipboard", { configurable: true, value })

afterEach(() => {
  setClipboard(undefined)
})

describe("copyToClipboard", () => {
  it("resolves once the text is on the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    setClipboard({ writeText })
    await expect(copyToClipboard({ text: "Saved answer" })).resolves.toBeUndefined()
    expect(writeText).toHaveBeenCalledWith("Saved answer")
  })

  it("rejects when the browser refuses the write", async () => {
    setClipboard({ writeText: vi.fn().mockRejectedValue(new DOMException("Denied", "NotAllowedError")) })
    await expect(copyToClipboard({ text: "Saved answer" })).rejects.toThrow("Denied")
  })

  it("rejects when there is no clipboard to write to", async () => {
    setClipboard(undefined)
    await expect(copyToClipboard({ text: "Saved answer" })).rejects.toThrow()
  })
})
