import { afterEach, describe, expect, it, vi } from "vitest"

import {
  captureChatRequestDebugSnapshot,
  getLastChatCompletionDebugSnapshot,
  getLastChatRequestDebugSnapshot,
  setChatRequestDebugEnabled
} from "@/services/tldw/chat-request-debug"

afterEach(() => {
  setChatRequestDebugEnabled(false)
})

describe("chat-request-debug flag gate (WP-22)", () => {
  it("does not snapshot when the debug flag is off", () => {
    setChatRequestDebugEnabled(false)
    const stringifySpy = vi.spyOn(JSON, "stringify")
    try {
      captureChatRequestDebugSnapshot({
        endpoint: "/api/v1/chat/completions",
        method: "POST",
        mode: "stream",
        body: { messages: new Array(1000).fill({ role: "user", content: "x" }) }
      })
      // No clone at capture time…
      expect(stringifySpy).not.toHaveBeenCalled()
      // …and none at read time either: no snapshot was taken at all.
      expect(getLastChatRequestDebugSnapshot()).toBeNull()
      expect(getLastChatCompletionDebugSnapshot()).toBeNull()
      expect(stringifySpy).not.toHaveBeenCalled()
    } finally {
      stringifySpy.mockRestore()
    }
  })

  it("snapshots (and lazily clones) when the debug flag is on", () => {
    setChatRequestDebugEnabled(true)
    const body = { model: "gpt-test", messages: [{ role: "user", content: "hello" }] }
    const stringifySpy = vi.spyOn(JSON, "stringify")
    try {
      captureChatRequestDebugSnapshot({
        endpoint: "/api/v1/chat/completions",
        method: "POST",
        mode: "non-stream",
        body
      })
      // Still lazy: capture itself does not clone.
      expect(stringifySpy).not.toHaveBeenCalled()

      const snapshot = getLastChatRequestDebugSnapshot()
      expect(snapshot).not.toBeNull()
      // The snapshot body is an isolated clone, not the live request reference.
      expect(snapshot?.body).not.toBe(body)
      expect(snapshot?.body).toEqual(body)
      expect(stringifySpy).toHaveBeenCalled()
    } finally {
      stringifySpy.mockRestore()
    }
  })
})
