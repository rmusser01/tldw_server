import { describe, expect, it, vi } from "vitest"
import {
  decodeText,
  readBoundedResponse,
  validateRepoPath
} from "../exportPolicy"

describe("exportPolicy", () => {
  it.each([
    "../outside.md",
    "src/../../outside.md",
    "C:/outside.md",
    "src\\outside.md",
    "/outside.md",
    "src//file.md",
    "src/./file.md"
  ])("rejects unsafe path %s", (path) => {
    expect(() => validateRepoPath(path)).toThrow(/path/i)
  })
  it("rejects invalid UTF8", () => {
    expect(() => decodeText(new Uint8Array([0xff, 0xfe]), "binary.md")).toThrow(
      /UTF-8/
    )
  })
  it("bounds an unadvertised streamed response and cancels the reader", async () => {
    const cancel = vi.fn()
    const response = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new Uint8Array(4))
          controller.enqueue(new Uint8Array(4))
        },
        cancel
      })
    )
    await expect(readBoundedResponse(response, 5)).rejects.toThrow(/size limit/)
    expect(cancel).toHaveBeenCalledOnce()
  })
})
