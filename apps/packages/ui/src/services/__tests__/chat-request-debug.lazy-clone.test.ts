import { describe, expect, it, vi } from "vitest"

import {
  captureChatRequestDebugSnapshot,
  getLastChatCompletionDebugSnapshot,
  getLastChatRequestDebugSnapshot
} from "../tldw/chat-request-debug"

describe("chat-request-debug lazy cloning (TASK-13511)", () => {
  it("does not deep-clone the payload at capture time", () => {
    const stringifySpy = vi.spyOn(JSON, "stringify")
    try {
      captureChatRequestDebugSnapshot({
        endpoint: "/api/v1/chat/completions",
        method: "POST",
        mode: "stream",
        body: { messages: new Array(1000).fill({ role: "user", content: "x" }) }
      })
      expect(stringifySpy).not.toHaveBeenCalled()
    } finally {
      stringifySpy.mockRestore()
    }
  })

  it("clones on first read and caches the clone for subsequent reads", () => {
    const body = { messages: [{ role: "user", content: "hello" }] }
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "non-stream",
      body
    })

    const first = getLastChatRequestDebugSnapshot()
    expect(first).not.toBeNull()
    // The snapshot body is an isolated clone, not the live request reference.
    expect(first?.body).not.toBe(body)
    expect(first?.body).toEqual(body)

    // Second read returns the same cached clone (no re-clone).
    const stringifySpy = vi.spyOn(JSON, "stringify")
    try {
      const second = getLastChatRequestDebugSnapshot()
      expect(second?.body).toBe(first?.body)
      expect(stringifySpy).not.toHaveBeenCalled()
    } finally {
      stringifySpy.mockRestore()
    }
  })

  it("keeps the chat-completions helper contract on the lazy snapshot", () => {
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/chat/completions",
      method: "POST",
      mode: "stream",
      body: { model: "gpt-test", messages: [] },
      metadata: { model: "gpt-test", toolCounts: { total: 1, kept: 1 } }
    })
    const snapshot = getLastChatCompletionDebugSnapshot()
    expect(snapshot).not.toBeNull()
    expect(snapshot?.endpoint).toBe("/api/v1/chat/completions")
    expect(snapshot?.request.model).toBe("gpt-test")
    expect(snapshot?.metadata?.toolCounts).toEqual({ total: 1, kept: 1 })
  })

  it("returns null from the helper for other endpoints", () => {
    captureChatRequestDebugSnapshot({
      endpoint: "/api/v1/other",
      method: "POST",
      mode: "stream",
      body: {}
    })
    expect(getLastChatCompletionDebugSnapshot()).toBeNull()
  })
})
