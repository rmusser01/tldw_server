// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest"

describe("tldw:quick-ingest-complete event contract", () => {
  it("can be dispatched and received with batchId detail", () => {
    const handler = vi.fn()
    window.addEventListener("tldw:quick-ingest-complete", handler)

    window.dispatchEvent(new CustomEvent("tldw:quick-ingest-complete", {
      detail: { batchId: "test-123", successCount: 3, failCount: 0 }
    }))

    expect(handler).toHaveBeenCalledTimes(1)
    const event = handler.mock.calls[0][0] as CustomEvent
    expect(event.detail.batchId).toBe("test-123")
    expect(event.detail.successCount).toBe(3)

    window.removeEventListener("tldw:quick-ingest-complete", handler)
  })
})
