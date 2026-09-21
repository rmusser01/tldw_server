import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useStoreMessageOption } from "@/store/option"
import { useSidepanelChatMetadata, type QueuedDispatchGuard } from "../useSidepanelChatMetadata"
import { useComposerQueue } from "@/components/Chat/composer/hooks/useComposerQueue"
import { buildQueuedRequest } from "@/utils/chat-request-queue"

const io = vi.hoisted(() => ({
  getChat: vi.fn(), getCharacter: vi.fn(), getPersonaProfile: vi.fn(),
  load: vi.fn(), selection: vi.fn(), selectionRevision: 0
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  getChat: io.getChat, getCharacter: io.getCharacter, getPersonaProfile: io.getPersonaProfile
} }))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: io.load }))
vi.mock("@/hooks/useSelectedAssistant", () => ({
  getSelectedAssistantOperationRevision: () => io.selectionRevision
}))

const held = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
const queued = () => buildQueuedRequest({
  id: "queue-original", clientRequestId: "original-client", conversationId: "saved-chat",
  promptText: "Queued private request", snapshot: { selectedModel: "tldw:model", chatMode: "normal" }
})
const characterChat = { id: "saved-chat", title: "Owned Character", assistant_kind: "character", assistant_id: "42", character_id: 42 }
const initialState = useStoreMessageOption.getState()
let scope: AbortController
let release: ReturnType<typeof vi.fn>

beforeEach(() => {
  vi.clearAllMocks()
  useStoreMessageOption.setState(initialState, true)
  useStoreMessageOption.getState().setServerChatId("saved-chat")
  useStoreMessageOption.getState().setQueuedMessages([queued()])
  scope = new AbortController()
  release = vi.fn()
  io.selectionRevision = 0
  io.selection.mockImplementation(async () => { io.selectionRevision++ })
  io.load.mockResolvedValue({
    requestScope: { config: { serverUrl: "https://owned.test", authMode: "multi-user" }, userId: "alice" },
    scopeSignal: scope.signal, scopeInvalidatedSignal: scope.signal, release
  })
  io.getChat.mockResolvedValue(characterChat)
  io.getCharacter.mockResolvedValue({ id: 42, name: "Owned Character" })
  io.getPersonaProfile.mockResolvedValue({ id: "persona-1", name: "Owned Persona" })
})
afterEach(() => { useStoreMessageOption.setState(initialState, true) })

// Exercise the actual queue primitive and Zustand mutations used by the form.
const mountQueue = (prepare = async () => {}, connected = true, complete: (guard: QueuedDispatchGuard) => Promise<void> = async () => {}) => {
  const dispatched: unknown[] = []
  const hook = renderHook(() => {
    const metadata = useSidepanelChatMetadata(io.selection)
    const { captureQueuedDispatchGuard } = metadata
    const state = useStoreMessageOption()
    const send = React.useCallback(async (item: ReturnType<typeof queued>) => {
      const assertCurrent = captureQueuedDispatchGuard(item.conversationId)
      await prepare()
      assertCurrent()
      const current = useStoreMessageOption.getState()
      dispatched.push({ item, actor: [current.serverChatAssistantKind, current.serverChatAssistantId] })
      await complete(assertCurrent)
    }, [captureQueuedDispatchGuard])
    const queue = useComposerQueue({
      isConnectionReady: connected && metadata.isReady, isStreaming: false,
      queuedMessages: state.queuedMessages, setQueuedMessages: state.setQueuedMessages,
      sendQueuedRequest: send, stopStreamingRequest: () => {},
      canCommitDispatchResult: metadata.isQueuedCompletionCurrent,
      resolveConversationId: () => state.serverChatId, buildQueuedDocuments: () => [],
      buildQueuedRequestSnapshot: () => ({}), isQueuedDispatchBlocked: false,
      cancelCurrentAndRunDisabledReasonText: null
    })
    return { metadata, queue }
  })
  return { ...hook, dispatched }
}

describe("saved sidepanel conversation readiness", () => {
  it.each(["preflight", "completion"])("does not mutate Alice's restored queue when old %s settles after remount", async phase => {
    const pending = held<void>()
    const originalQueue = useStoreMessageOption.getState().queuedMessages
    const hook = phase === "preflight"
      ? mountQueue(() => pending.promise)
      : mountQueue(undefined, true, () => pending.promise)
    await waitFor(() => expect(useStoreMessageOption.getState().queuedMessages[0].status).toBe("sending"))
    if (phase === "completion") await waitFor(() => expect(hook.dispatched).toHaveLength(1))
    hook.unmount()
    await act(async () => {
      window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
      useStoreMessageOption.getState().setServerChatId("saved-chat")
      useStoreMessageOption.getState().setQueuedMessages(originalQueue)
      pending.resolve()
    })
    expect(useStoreMessageOption.getState().queuedMessages).toEqual(originalQueue)
  })

  it("removes a sent new-chat queue item after its expected server promotion", async () => {
    useStoreMessageOption.getState().setServerChatId(null)
    useStoreMessageOption.getState().setQueuedMessages([{ ...queued(), conversationId: null }])
    const hook = mountQueue(undefined, true, async guard => {
      guard.publishServerChatId("promoted-chat", useStoreMessageOption.getState().setServerChatId)
      guard.publishHistoryId("promoted-history", id => useStoreMessageOption.getState().setHistoryId(id, { preserveServerChatId: true }))
    })
    await waitFor(() => expect(hook.dispatched).toHaveLength(1))
    await waitFor(() => expect(useStoreMessageOption.getState().queuedMessages).toEqual([]))
  })

  it("holds automatic and manual queue replay until the canonical Character is ready", async () => {
    const read = held<typeof characterChat>()
    io.getChat.mockReturnValue(read.promise)
    const hook = mountQueue()
    await waitFor(() => expect(io.getChat).toHaveBeenCalled())
    await act(async () => {
      await hook.result.current.queue.handleRunNextQueuedRequest()
      await hook.result.current.queue.handleRunQueuedRequest("queue-original")
    })
    expect(hook.dispatched).toEqual([])
    expect(useStoreMessageOption.getState().queuedMessages[0]).toMatchObject({
      id: "queue-original", clientRequestId: "original-client", status: "queued", promptText: "Queued private request"
    })
    await act(async () => { read.resolve(characterChat) })
    await waitFor(() => expect(hook.dispatched).toHaveLength(1))
    expect(hook.dispatched[0]).toMatchObject({ actor: ["character", "42"], item: { id: "queue-original", clientRequestId: "original-client" } })
    expect(useStoreMessageOption.getState().queuedMessages).toEqual([])
    expect(io.getChat.mock.calls[0][1]).toMatchObject({ requestScope: { userId: "alice" }, signal: scope.signal })
    expect(io.getCharacter.mock.calls[0][1]).toMatchObject({ requestScope: { userId: "alice" }, signal: scope.signal })
  })

  it("retains a failed metadata queue and retries the same request once details recover", async () => {
    const originalQueue = useStoreMessageOption.getState().queuedMessages
    io.getChat.mockRejectedValueOnce(new Error("secret transport failure"))
    const hook = mountQueue()
    await waitFor(() => expect(hook.result.current.metadata.failed).toBe(true))
    expect(hook.dispatched).toEqual([])
    expect(useStoreMessageOption.getState().serverChatLoadError).not.toContain("secret")
    expect(useStoreMessageOption.getState().queuedMessages).toEqual(originalQueue)
    await act(async () => { hook.result.current.metadata.retryMetadata() })
    await waitFor(() => expect(hook.dispatched).toHaveLength(1))
    expect(hook.dispatched[0]).toMatchObject({ item: { clientRequestId: "original-client", promptText: "Queued private request" } })
  })

  it("waits for saved persona presentation before enabling replay", async () => {
    io.getChat.mockResolvedValue({ id: "saved-chat", assistant_kind: "persona", assistant_id: "persona-1" })
    const profile = held<{ id: string; name: string }>()
    io.getPersonaProfile.mockReturnValue(profile.promise)
    const hook = mountQueue()
    await waitFor(() => expect(io.getPersonaProfile).toHaveBeenCalled())
    expect(hook.dispatched).toEqual([])
    await act(async () => { profile.resolve({ id: "persona-1", name: "Owned Persona" }) })
    await waitFor(() => expect(hook.dispatched).toHaveLength(1))
    expect(hook.dispatched[0]).toMatchObject({ actor: ["persona", "persona-1"] })
  })

  it.each(["unmount", "account", "conversation"])("refuses held presentation after %s", async (boundary) => {
    const profile = held<{ id: number; name: string }>()
    io.getCharacter.mockReturnValue(profile.promise)
    const hook = mountQueue()
    await waitFor(() => expect(io.getCharacter).toHaveBeenCalled())
    if (boundary === "unmount") hook.unmount()
    await act(async () => {
      if (boundary === "account") window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
      if (boundary === "conversation") useStoreMessageOption.getState().setServerChatId(null)
      profile.resolve({ id: 42, name: "Stale private Character" })
    })
    expect(io.selection).not.toHaveBeenCalled()
    expect(useStoreMessageOption.getState().serverChatTitle).not.toBe("Owned Character")
    expect(hook.dispatched).toEqual([])
    expect(release).toHaveBeenCalled()
    hook.unmount()
  })

  it.each(["unmount", "account", "conversation"])("refuses a queued preflight completed after %s", async (boundary) => {
    const preparation = held<void>()
    const hook = mountQueue(() => preparation.promise)
    await waitFor(() => expect(useStoreMessageOption.getState().queuedMessages[0].status).toBe("sending"))
    if (boundary === "unmount") hook.unmount()
    await act(async () => {
      if (boundary === "account") window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
      if (boundary === "conversation") {
        useStoreMessageOption.getState().setServerChatId("other-chat")
        useStoreMessageOption.getState().setServerChatId("saved-chat")
      }
      preparation.resolve()
    })
    expect(hook.dispatched).toEqual([])
    hook.unmount()
  })

  it("keeps new local and successfully restored plain conversations usable", async () => {
    io.getChat.mockResolvedValue({ id: "saved-chat", title: "Plain" })
    const saved = mountQueue()
    await waitFor(() => expect(saved.dispatched).toHaveLength(1))
    expect(saved.dispatched[0]).toMatchObject({ actor: [null, null] })
    saved.unmount()
    useStoreMessageOption.getState().setServerChatId(null)
    useStoreMessageOption.getState().setQueuedMessages([{ ...queued(), conversationId: null }])
    const local = mountQueue()
    await waitFor(() => expect(local.dispatched).toHaveLength(1))
    expect(io.getChat).toHaveBeenCalledTimes(1)
  })
})

describe("real-store queued Persona creation", () => {
  it.each(["none", "foreign", "ABA", "account", "tab", "unmount"])("owns only its own promotion after %s", async boundary => {
    const { ensurePersonaServerChat } = await import("../personaServerChat")
    const { useSidepanelChatTabsStore } = await import("@/store/sidepanel-chat-tabs")
    useStoreMessageOption.getState().setServerChatId(null)
    useStoreMessageOption.getState().setHistoryId(null)
    const hook = renderHook(() => useSidepanelChatMetadata(io.selection))
    const guard = hook.result.current.captureQueuedDispatchGuard(null)
    const completion = hook.result.current.isQueuedCompletionCurrent
    const state = useStoreMessageOption.getState()
    const creation = held<unknown>()
    const invalidate = vi.fn()
    const run = ensurePersonaServerChat({
      ...state,
      assistant: { kind: "persona", id: "p1", name: "Persona", metadata: { selectionMode: "tracked" } },
      createChat: async () => creation.promise,
      ensureServerChatHistoryId: async () => state.historyId,
      invalidateServerChatHistory: invalidate,
      setServerChatId: id => guard.publishServerChatId(id, state.setServerChatId)
    }).then(value => { guard(); return { ok: true, value } }, error => ({ ok: false, error }))
    if (boundary === "unmount") hook.unmount()
    await act(async () => {
      if (boundary === "foreign" || boundary === "ABA") {
        useStoreMessageOption.getState().setServerChatId("foreign-chat")
        if (boundary === "ABA") useStoreMessageOption.getState().setServerChatId(null)
      }
      if (boundary === "account") window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
      if (boundary === "tab") useSidepanelChatTabsStore.getState().setActiveTabId("different-tab")
      creation.resolve({ id: "created-persona", assistant_kind: "persona", assistant_id: "p1" })
    })
    const outcome = await run
    if (boundary === "none") {
      expect(outcome).toMatchObject({ ok: true, value: { chatId: "created-persona" } })
      expect(useStoreMessageOption.getState()).toMatchObject({
        serverChatId: "created-persona", serverChatMetaLoaded: true, serverChatAssistantKind: "persona", serverChatAssistantId: "p1"
      })
      expect(completion()).toBe(true)
      expect(() => guard()).not.toThrow()
    } else {
      expect(outcome.ok).toBe(false)
      expect(useStoreMessageOption.getState().serverChatId).toBe(boundary === "foreign" ? "foreign-chat" : null)
      expect(invalidate).not.toHaveBeenCalled()
      expect(completion()).toBe(false)
    }
    hook.unmount()
  })
})

it("UAT375: an old null-chat dispatch must not change a foreign restored queue", async () => {
  useStoreMessageOption.getState().setServerChatId(null)
  useStoreMessageOption.getState().setHistoryId(null)
  useStoreMessageOption.getState().setQueuedMessages([{ ...queued(), conversationId: null }])
  const preflight = held<void>()
  const hook = mountQueue(() => preflight.promise)
  await waitFor(() => expect(useStoreMessageOption.getState().queuedMessages[0].status).toBe("sending"))
  let foreignQueue = [{ ...queued(), conversationId: "foreign-chat", promptText: "Restored foreign queue", status: "blocked" as const, blockedReason: "Foreign queue's own retained error" }]
  await act(async () => {
    useStoreMessageOption.getState().setServerChatId("foreign-chat")
    useStoreMessageOption.getState().setQueuedMessages(foreignQueue)
    foreignQueue = useStoreMessageOption.getState().queuedMessages as typeof foreignQueue
    preflight.resolve()
  })
  expect(hook.dispatched).toEqual([])
  expect(useStoreMessageOption.getState().queuedMessages).toEqual(foreignQueue)
  hook.unmount()
})

it("UAT375: an old null-chat completion must not remove a foreign restored queue", async () => {
  useStoreMessageOption.getState().setServerChatId(null)
  useStoreMessageOption.getState().setHistoryId(null)
  useStoreMessageOption.getState().setQueuedMessages([{ ...queued(), conversationId: null }])
  const completion = held<void>()
  const hook = mountQueue(undefined, true, () => completion.promise)
  await waitFor(() => expect(hook.dispatched).toHaveLength(1))
  let foreignQueue: ReturnType<typeof useStoreMessageOption.getState>["queuedMessages"]
  await act(async () => {
    useStoreMessageOption.getState().setServerChatId("foreign-chat")
    useStoreMessageOption.getState().setQueuedMessages([{ ...queued(), conversationId: "foreign-chat", promptText: "Restored foreign queue", status: "blocked", blockedReason: "Foreign queue's own retained error" }])
    foreignQueue = useStoreMessageOption.getState().queuedMessages
    completion.resolve()
  })
  expect(useStoreMessageOption.getState().queuedMessages).toEqual(foreignQueue!)
  hook.unmount()
})
