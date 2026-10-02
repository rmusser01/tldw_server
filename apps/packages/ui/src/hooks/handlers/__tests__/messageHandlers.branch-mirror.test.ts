import { describe, expect, it, vi } from "vitest"
import type { ChatHistory, Message } from "@/store/option"
import { createRegenerateLastMessage } from "../messageHandlers"

// The real mirror/source/image regression now exercises useChatActions in its
// Character integration suite. This checks the submission handoff it consumes.
describe("ordinary Retry branch snapshot", () => {
  it.each([false, true])("preserves copied receipts and source variants for branch=%s", async branched => {
    const user: Message = {
      id: "source-user", serverMessageId: "parent-user", isBot: false,
      role: "user", name: "You", message: "Question", sources: []
    }
    const answer: Message = {
      id: "source-answer", serverMessageId: "parent-answer", isBot: true,
      role: "assistant", name: "Character", message: "Answer", sources: [],
      variants: [{ id: "parent-old", message: "Old answer" }, { id: "source-answer", message: "Answer" }],
      activeVariantIndex: 1
    }
    const rows = [user, answer]
    const original = structuredClone(rows)
    const copied: Message[] = [{ ...user, id: "child-user", serverMessageId: "child-user" }]
    const history: ChatHistory = [{ role: "user", content: "Question" }, { role: "assistant", content: "Answer" }]
    const onSubmit = vi.fn(async () => ({ status: "submitted" as const }))
    const setMessages = vi.fn()
    await createRegenerateLastMessage({
      allowOrdinaryRetry: true,
      validateBeforeSubmitFn: () => true,
      messages: rows, history, setHistory: vi.fn(), setMessages, onSubmit,
      beforeSubmit: async () => branched ? {
        messages: copied,
        submitExtras: { serverChatIdOverride: "child-chat", historyIdOverride: "child-history" }
      } : undefined
    })()
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      messages: branched ? copied : [user],
      regenerateFromMessage: branched ? undefined : answer,
      ...(branched ? { serverChatIdOverride: "child-chat", historyIdOverride: "child-history" } : {})
    }))
    expect(setMessages).toHaveBeenCalledWith(branched ? copied : [user])
    expect(rows).toEqual(original)
  })
})
