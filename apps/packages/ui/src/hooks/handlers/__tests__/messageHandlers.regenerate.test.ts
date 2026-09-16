import { describe, expect, it, vi } from "vitest"
import type { ChatHistory, Message } from "@/store/option"
import { createRegenerateLastMessage } from "../messageHandlers"

const buildMessages = (): Message[] => [
  {
    isBot: false,
    name: "You",
    message: "Hello",
    sources: []
  },
  {
    isBot: true,
    name: "Assistant",
    message: "Hi there",
    sources: []
  }
]

const buildHistory = (): ChatHistory => [
  {
    role: "user",
    content: "Hello"
  },
  {
    role: "assistant",
    content: "Hi there"
  }
]

describe("createRegenerateLastMessage", () => {
  it("removes the latest assistant turn and resubmits the previous user turn", async () => {
    const history = buildHistory()
    const messages = buildMessages()
    const setHistory = vi.fn()
    const setMessages = vi.fn()
    const onSubmit = vi.fn().mockResolvedValue(undefined)

    const regenerate = createRegenerateLastMessage({
      validateBeforeSubmitFn: () => true,
      history,
      messages,
      setHistory,
      setMessages,
      onSubmit
    })

    await regenerate()

    expect(setHistory).toHaveBeenCalledWith([])
    expect(setMessages).toHaveBeenCalledWith([messages[0]])
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Hello",
        image: "",
        isRegenerate: true,
        memory: [],
        messages: [messages[0]],
        messageType: undefined,
        regenerateFromMessage: messages[1]
      })
    )
  })

  it.each([
    { content: "  Preserve spaces  ", image: "", expected: true },
    { content: "", image: "data:image/png;base64,aW1hZ2U=", expected: true },
    { content: "   ", image: "", expected: false },
    { content: "", image: "", expected: false }
  ])("retains exact Retry input and admits only nonempty text or an image: $content / $image", async ({ content, image, expected }) => {
    const messages = buildMessages()
    messages[0] = { ...messages[0], message: content, images: image ? [image] : [] }
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    await createRegenerateLastMessage({ validateBeforeSubmitFn: () => true,
      history: [{ role: "user", content, image }], messages,
      setHistory: vi.fn(), setMessages: vi.fn(), onSubmit })()
    if (expected) expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ message: content, image }))
    else expect(onSubmit).not.toHaveBeenCalled()
  })

  it("bails safely when setHistory is not callable", async () => {
    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined)
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    const setMessages = vi.fn()

    const regenerate = createRegenerateLastMessage({
      validateBeforeSubmitFn: () => true,
      history: buildHistory(),
      messages: buildMessages(),
      setHistory: null as unknown as (history: ChatHistory) => void,
      setMessages,
      onSubmit
    })

    await expect(regenerate()).resolves.toBeUndefined()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(setMessages).not.toHaveBeenCalled()
    expect(consoleErrorSpy).toHaveBeenCalled()
    consoleErrorSpy.mockRestore()
  })

  it("allows pre-submit hooks to override regenerate payload", async () => {
    const history = buildHistory()
    const messages = buildMessages()
    const setHistory = vi.fn()
    const setMessages = vi.fn()
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    const beforeSubmit = vi.fn().mockResolvedValue({
      memory: [
        {
          role: "user",
          content: "Override memory"
        }
      ] satisfies ChatHistory,
      messages: [messages[0]],
      submitExtras: {
        serverChatIdOverride: "branched-chat-id"
      }
    })

    const regenerate = createRegenerateLastMessage({
      validateBeforeSubmitFn: () => true,
      history,
      messages,
      setHistory,
      setMessages,
      onSubmit,
      beforeSubmit
    })

    await regenerate()

    expect(beforeSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        memory: [{ role: "user", content: "Override memory" }],
        messages: [messages[0]],
        serverChatIdOverride: "branched-chat-id"
      })
    )
  })
})
