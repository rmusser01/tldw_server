import { describe, expect, it, vi } from "vitest"
import { useStoreMessageOption, type ChatHistory, type Message } from "@/store/option"
import type { ActorSceneSend } from "@/hooks/chat/actor-scene-source"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import { freshChatDraftAddress } from "@/utils/chat-draft-address"
import { createDefaultActorSettings } from "@/types/actor"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
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

// The helper forwards an issued private Scene as an opaque command proof.
// It creates no selected-history owner/view from transcript or server IDs.
const originalSceneProof = (isCurrent = () => true): ActorSceneSend => {
  const requestScope = { config: {
    serverUrl: "https://chat.test", authMode: "single-user" as const,
    expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope("single-user", "synthetic-a")!
  }, userId: null }
  const target = captureActorSettingsTarget({ address: freshChatDraftAddress(false),
    requestScope, accountIsCurrent: isCurrent, isCurrent })!
  const settings = createDefaultActorSettings()
  const proof: ActorSceneSend = { scene: { target, settings,
    originalRecord: { value: settings, updatedAt: 1700000000000 },
    origin: "scene", profileUnavailable: false }, requestScope, isCurrent,
    commandIsCurrent: isCurrent }
  return Object.freeze(proof)
}

describe("createRegenerateLastMessage", () => {
  it("forwards the exact original SDK Scene through held regenerate preparation", async () => {
    const original = originalSceneProof()
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    const onSubmit = vi.fn().mockResolvedValue({ status: "submitted" })
    const regenerate = createRegenerateLastMessage({
      allowOrdinaryRetry: true, validateBeforeSubmitFn: () => true,
      history: buildHistory(), messages: buildMessages(),
      setHistory: vi.fn(), setMessages: vi.fn(), onSubmit,
      beforeSubmit: async () => { await pending }
    })
    const result = regenerate({ actorSceneSend: original })
    expect(onSubmit).not.toHaveBeenCalled()
    release()
    await result
    expect(onSubmit.mock.calls[0][0].actorSceneSend).toBe(original)
    expect(onSubmit.mock.calls[0][0].actorSceneSend.scene.originalRecord).toBe(original.scene.originalRecord)
  })

  it.each(["entry", "preparation"] as const)(
    "refuses a retired supplied SDK Scene during %s before truncation or submission",
    async boundary => {
      let current = true
      const original = originalSceneProof(() => current)
      let release!: () => void
      const pending = new Promise<void>(resolve => { release = resolve })
      const onSubmit = vi.fn()
      const setHistory = vi.fn(), setMessages = vi.fn()
      const regenerate = createRegenerateLastMessage({
        allowOrdinaryRetry: true, validateBeforeSubmitFn: () => true,
        history: buildHistory(), messages: buildMessages(), setHistory, setMessages, onSubmit,
        beforeSubmit: async () => { await pending }
      })
      if (boundary === "entry") current = false
      const result = regenerate({ actorSceneSend: original })
      current = false
      release()
      expect(await result).toMatchObject({ status: "skipped" })
      expect(onSubmit).not.toHaveBeenCalled()
      expect(setHistory).not.toHaveBeenCalled()
      expect(setMessages).not.toHaveBeenCalled()
    }
  )
  it("rechecks the original SDK Scene after a synchronous Retry setter subscriber replaces work", async () => {
    const before = useStoreMessageOption.getState()
    let current = true
    const original = originalSceneProof(() => current)
    const history = buildHistory(), messages = buildMessages()
    const replacement = [{ ...messages[0], message: "Keep replacement work" }]
    useStoreMessageOption.setState({ history, messages, streaming: false, isProcessing: false })
    const state = useStoreMessageOption.getState()
    const onSubmit = vi.fn(), setMessages = vi.fn(state.setMessages)
    const regenerate = createRegenerateLastMessage({
      allowOrdinaryRetry: true, validateBeforeSubmitFn: () => true,
      history, messages, setHistory: state.setHistory, setMessages, onSubmit
    })
    const unsubscribe = useStoreMessageOption.subscribe(next => {
      if (next.history === history) return
      unsubscribe()
      current = false
      next.startNewChatDraft(false)
      useStoreMessageOption.setState({ messages: replacement })
    })
    try {
      expect(await regenerate({ actorSceneSend: original })).toMatchObject({ status: "skipped" })
      expect(useStoreMessageOption.getState().messages).toBe(replacement)
      expect(setMessages).not.toHaveBeenCalled()
      expect(onSubmit).not.toHaveBeenCalled()
    } finally {
      unsubscribe()
      useStoreMessageOption.setState(before, true)
    }
  })

  it("retirement during regenerate preparation prevents truncation and submission", async () => {
    let release!: () => void
    const pending = new Promise<void>((resolve) => { release = resolve })
    const onSubmit = vi.fn()
    const setHistory = vi.fn()
    const setMessages = vi.fn()
    const regenerate = createRegenerateLastMessage({
      allowOrdinaryRetry: true, validateBeforeSubmitFn: () => true,
      history: buildHistory(), messages: buildMessages(), setHistory, setMessages, onSubmit,
      beforeSubmit: async () => { await pending }
    })
    const controller = new AbortController()
    const result = regenerate({ controller }).catch((error: Error) => error)
    controller.abort()
    release()
    await result
    expect(onSubmit).not.toHaveBeenCalled()
    expect(setHistory).not.toHaveBeenCalled()
    expect(setMessages).not.toHaveBeenCalled()
  })

  it("rejects before truncation when selection exists but has no captured path", async () => {
    const setHistory = vi.fn()
    const setMessages = vi.fn()
    const onSubmit = vi.fn()
    const regenerate = createRegenerateLastMessage({
      allowOrdinaryRetry: true,
      historySelection: { getCurrent: () => ({ status: "idle", capture: null }) } as any,
      validateBeforeSubmitFn: () => true,
      history: buildHistory(),
      messages: buildMessages(),
      setHistory,
      setMessages,
      onSubmit
    })

    await expect(regenerate()).rejects.toThrow("unsupported_history_regeneration")
    expect(setHistory).not.toHaveBeenCalled()
    expect(setMessages).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  })

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
      beforeSubmit,
      historySelection: { getCurrent: () => ({ status: "ready", capture: { status: "captured" } }) } as any
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

  it("removes the latest assistant turn and resubmits the previous user turn", async () => {
    const history = buildHistory()
    const messages = buildMessages()
    const setHistory = vi.fn()
    const setMessages = vi.fn()
    const onSubmit = vi.fn().mockResolvedValue(undefined)

    const regenerate = createRegenerateLastMessage({ allowOrdinaryRetry: true,
      validateBeforeSubmitFn: () => true,
      history,
      messages,
      setHistory,
      setMessages,
      onSubmit
    })

    const controller = new AbortController()
    const assertCurrent = vi.fn()
    await regenerate({ controller, assertCurrent })

    expect(assertCurrent).toHaveBeenCalledTimes(2)
    expect(setHistory).toHaveBeenCalledWith([])
    expect(setMessages).toHaveBeenCalledWith([messages[0]])
    expect(onSubmit).toHaveBeenCalledTimes(1)
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Hello",
        image: "",
        isRegenerate: true,
        controller,
        assertCurrent,
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
    await createRegenerateLastMessage({ allowOrdinaryRetry: true, validateBeforeSubmitFn: () => true,
      history: [{ role: "user", content, image }], messages,
      setHistory: vi.fn(), setMessages: vi.fn(), onSubmit })()
    if (expected) expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ message: content, image }))
    else expect(onSubmit).not.toHaveBeenCalled()
  })

  it("retries a text-only local diagnostic without borrowing an older turn's image or type", async () => {
    const generationInfo = { mode: "rag", grounded: false, reason: "selected_source_retrieval_failed" }
    const history: ChatHistory = [{ role: "user", content: "Older image question", image: "data:image/png;base64,older", messageType: "older-type" }, { role: "assistant", content: "Older answer" }]
    const user: Message = { id: "local-user", isBot: false, name: "You", message: "New source question", sources: [], images: [], generationInfo }
    const assistant: Message = { id: "local-diagnostic", isBot: true, name: "Assistant", message: "Retrieval failed", sources: [], parentMessageId: user.id, generationInfo }
    const onSubmit = vi.fn()
    await createRegenerateLastMessage({ allowOrdinaryRetry: true, validateBeforeSubmitFn: () => true, history,
      messages: [...buildMessages(), user, assistant], setHistory: vi.fn(), setMessages: vi.fn(), onSubmit })()
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ message: user.message, image: "", messageType: undefined, memory: history }))
  })

  it("bails safely when setHistory is not callable", async () => {
    const consoleErrorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined)
    const onSubmit = vi.fn().mockResolvedValue(undefined)
    const setMessages = vi.fn()

    const regenerate = createRegenerateLastMessage({ allowOrdinaryRetry: true,
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

    const regenerate = createRegenerateLastMessage({ allowOrdinaryRetry: true,
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
