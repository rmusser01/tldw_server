import { describe, expect, it, vi } from "vitest"
import { createRegenerateLastMessage } from "../messageHandlers"

describe("same-parent regeneration capability gate", () => {
  it("captures the selected assistant boundary and never invokes the old copy/resubmit callback", async () => {
    const setHistory = vi.fn()
    const setMessages = vi.fn()
    const onSubmit = vi.fn()
    const beforeSubmit = vi.fn()
    const regenerate = createRegenerateLastMessage({
      validateBeforeSubmitFn: () => true,
      history: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi" }
      ],
      messages: [
        {
          id: "user",
          isBot: false,
          name: "You",
          message: "Hello",
          sources: []
        },
        {
          id: "selected-assistant",
          parentMessageId: "user",
          isBot: true,
          name: "Assistant",
          message: "Hi",
          sources: []
        }
      ],
      setHistory,
      setMessages,
      onSubmit,
      beforeSubmit
    })
    await expect(regenerate()).rejects.toMatchObject({
      message: "unsupported_history_regeneration",
      boundary: { kind: "before_message", message_id: "selected-assistant" }
    })
    expect(beforeSubmit).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(setHistory).not.toHaveBeenCalled()
    expect(setMessages).not.toHaveBeenCalled()
  })
})
