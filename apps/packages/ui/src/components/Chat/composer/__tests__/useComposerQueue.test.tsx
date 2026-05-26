import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useComposerQueue } from "../hooks/useComposerQueue"
import type { QueuedRequest } from "@/utils/chat-request-queue"
import type { QueuedRequestSnapshot } from "@/utils/chat-request-queue"
import type { ChatDocuments } from "@/models/ChatTypes"

// Mock the lower-level primitive so we can inspect the orchestration layer
// without touching real queue mutations.
const mockActions = {
  clear: vi.fn(),
  enqueue: vi.fn(),
  flushNext: vi.fn(async () => null),
  markBlocked: vi.fn(),
  move: vi.fn(),
  remove: vi.fn(),
  update: vi.fn(),
  runNow: vi.fn(async () => null)
}

vi.mock("@/hooks/chat/useQueuedRequests", () => ({
  useQueuedRequests: vi.fn(() => mockActions)
}))

const makeQueuedRequest = (
  overrides: Partial<QueuedRequest> = {}
): QueuedRequest => ({
  id: "req-1",
  clientRequestId: "client-1",
  conversationId: "conv-1",
  promptText: "hello",
  message: "hello",
  image: "",
  attachments: [],
  sourceContext: null,
  snapshot: {
    selectedModel: "gpt-4",
    chatMode: "normal",
    webSearch: false,
    compareMode: false,
    compareSelectedModels: [],
    selectedSystemPrompt: null,
    selectedQuickPrompt: null,
    toolChoice: null,
    useOCR: false
  },
  status: "queued",
  blockedReason: null,
  attemptCount: 0,
  createdAt: 0,
  updatedAt: 0,
  ...overrides
})

type BaseProps = {
  queuedMessages?: QueuedRequest[]
  isStreaming?: boolean
  isConnectionReady?: boolean
  isQueuedDispatchBlocked?: boolean
  cancelCurrentAndRunDisabledReasonText?: string | null
  onEnqueueBlocked?: (reason: string) => void
  onEnqueueSuccess?: (isStreaming: boolean, item: QueuedRequest) => void
  resolveConversationId?: () => string | null
  buildQueuedDocuments?: () => ChatDocuments
  buildQueuedRequestSnapshot?: () => Partial<QueuedRequestSnapshot>
  sendQueuedRequest?: (item: QueuedRequest) => Promise<void>
  stopStreamingRequest?: (options?: { discardTurn?: boolean }) => void
}

const renderComposerQueue = (props: BaseProps = {}) => {
  const setQueuedMessages = vi.fn()
  const hook = renderHook(() =>
    useComposerQueue({
      isConnectionReady: props.isConnectionReady ?? true,
      isStreaming: props.isStreaming ?? false,
      queuedMessages: props.queuedMessages ?? [],
      setQueuedMessages,
      sendQueuedRequest: props.sendQueuedRequest ?? vi.fn(async () => {}),
      stopStreamingRequest: props.stopStreamingRequest ?? vi.fn(),
      resolveConversationId:
        props.resolveConversationId ?? (() => "conv-1"),
      buildQueuedDocuments:
        props.buildQueuedDocuments ?? vi.fn(() => [] as any),
      buildQueuedRequestSnapshot:
        props.buildQueuedRequestSnapshot ??
        vi.fn(() => ({
          selectedModel: "gpt-4",
          chatMode: "normal" as const,
          webSearch: false,
          compareMode: false,
          compareSelectedModels: [],
          selectedSystemPrompt: null,
          selectedQuickPrompt: null,
          toolChoice: null,
          useOCR: false
        })),
      isQueuedDispatchBlocked: props.isQueuedDispatchBlocked ?? false,
      onEnqueueBlocked: props.onEnqueueBlocked,
      onEnqueueSuccess: props.onEnqueueSuccess,
      cancelCurrentAndRunDisabledReasonText:
        props.cancelCurrentAndRunDisabledReasonText ?? null
    })
  )
  return { hook, setQueuedMessages }
}

describe("useComposerQueue", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockActions.flushNext.mockImplementation(async () => null)
    mockActions.runNow.mockImplementation(async () => null)
    mockActions.enqueue.mockImplementation((input: any) =>
      makeQueuedRequest({
        id: "new-item",
        promptText: input.promptText ?? "",
        image: input.image ?? "",
        sourceContext: input.sourceContext ?? null,
        snapshot: {
          ...makeQueuedRequest().snapshot,
          ...(input.snapshot ?? {})
        }
      })
    )
  })

  it("returns null and fires onEnqueueBlocked when blockedReason is set", () => {
    const onEnqueueBlocked = vi.fn()
    const { hook } = renderComposerQueue({ onEnqueueBlocked })

    let result: QueuedRequest | null = null
    act(() => {
      result = hook.result.current.enqueue({
        promptText: "hi",
        image: "",
        blockedReason: "attachments-conflict"
      })
    })

    expect(result).toBeNull()
    expect(onEnqueueBlocked).toHaveBeenCalledWith("attachments-conflict")
    expect(mockActions.enqueue).not.toHaveBeenCalled()
  })

  it("forwards docs/snapshot/source-context into queuedRequestActions.enqueue", () => {
    const buildQueuedDocuments = vi.fn(() => [
      { type: "tab", tabId: "t1", title: "T", url: "u", favIconUrl: "f" }
    ]) as any
    const buildQueuedRequestSnapshot = vi.fn(() => ({
      selectedModel: "gpt-5",
      chatMode: "normal" as const,
      webSearch: true,
      compareMode: false,
      compareSelectedModels: [],
      selectedSystemPrompt: null,
      selectedQuickPrompt: null,
      toolChoice: null,
      useOCR: false
    }))
    const resolveConversationId = vi.fn(() => "hist-42")

    const { hook } = renderComposerQueue({
      buildQueuedDocuments,
      buildQueuedRequestSnapshot,
      resolveConversationId
    })

    act(() => {
      hook.result.current.enqueue({
        promptText: "hello",
        image: "data:image/png;base64,AAA",
        sourceContext: { isImageCommand: false, documents: [] }
      })
    })

    expect(mockActions.enqueue).toHaveBeenCalledTimes(1)
    const input = mockActions.enqueue.mock.calls[0][0]
    expect(input).toMatchObject({
      conversationId: "hist-42",
      promptText: "hello",
      image: "data:image/png;base64,AAA",
      sourceContext: { isImageCommand: false, documents: [] }
    })
    expect(input.snapshot).toMatchObject({
      selectedModel: "gpt-5",
      webSearch: true
    })
    expect(buildQueuedDocuments).toHaveBeenCalled()
    expect(buildQueuedRequestSnapshot).toHaveBeenCalled()
  })

  it("fires onEnqueueSuccess with the isStreaming flag and the new item", () => {
    const onEnqueueSuccess = vi.fn()
    const { hook } = renderComposerQueue({
      isStreaming: true,
      onEnqueueSuccess
    })

    act(() => {
      hook.result.current.enqueue({ promptText: "queued", image: "" })
    })

    expect(onEnqueueSuccess).toHaveBeenCalledTimes(1)
    const [streamingFlag, item] = onEnqueueSuccess.mock.calls[0]
    expect(streamingFlag).toBe(true)
    expect(item.id).toBe("new-item")
  })

  it("handleRunQueuedRequest calls runNow then flushNext when idle + connected", async () => {
    const { hook } = renderComposerQueue({
      isStreaming: false,
      isConnectionReady: true
    })

    await act(async () => {
      await hook.result.current.handleRunQueuedRequest("req-1")
    })

    expect(mockActions.runNow).toHaveBeenCalledWith("req-1")
    expect(mockActions.flushNext).toHaveBeenCalledTimes(1)
  })

  it("handleRunQueuedRequest bails without calling runNow when streaming and a disabled reason is set", async () => {
    const { hook } = renderComposerQueue({
      isStreaming: true,
      cancelCurrentAndRunDisabledReasonText: "cannot-cancel"
    })

    await act(async () => {
      await hook.result.current.handleRunQueuedRequest("req-1")
    })

    expect(mockActions.runNow).not.toHaveBeenCalled()
    expect(mockActions.flushNext).not.toHaveBeenCalled()
  })

  it("handleRunNextQueuedRequest routes blocked head items through handleRunQueuedRequest", async () => {
    const blocked = makeQueuedRequest({
      id: "blk-1",
      status: "blocked",
      blockedReason: "boom"
    })

    const { hook } = renderComposerQueue({
      queuedMessages: [blocked]
    })

    await act(async () => {
      await hook.result.current.handleRunNextQueuedRequest()
    })

    // blocked head → runNow(id), then flushNext (since idle + connected)
    expect(mockActions.runNow).toHaveBeenCalledWith("blk-1")
    expect(mockActions.flushNext).toHaveBeenCalledTimes(1)
  })

  it("handleRunNextQueuedRequest calls flushNext directly when head is queued", async () => {
    const queued = makeQueuedRequest({ id: "q-1", status: "queued" })

    // Auto-drain runs on mount when head is queued + all gates open, so we
    // start with the blocked flag set to isolate the handleRunNext call
    // path. Flipping it off would still work, but this is tighter.
    const { hook } = renderComposerQueue({
      queuedMessages: [queued],
      isQueuedDispatchBlocked: true
    })

    // Sanity: auto-drain suppressed.
    expect(mockActions.flushNext).not.toHaveBeenCalled()

    await act(async () => {
      await hook.result.current.handleRunNextQueuedRequest()
    })

    expect(mockActions.runNow).not.toHaveBeenCalled()
    expect(mockActions.flushNext).toHaveBeenCalledTimes(1)
  })

  it("derives cancelCurrentAndRunDisabledReason: non-null when streaming + text provided, null when idle", () => {
    const { hook: streaming } = renderComposerQueue({
      isStreaming: true,
      cancelCurrentAndRunDisabledReasonText: "cannot-cancel"
    })
    expect(streaming.result.current.cancelCurrentAndRunDisabledReason).toBe(
      "cannot-cancel"
    )

    const { hook: idle } = renderComposerQueue({
      isStreaming: false,
      cancelCurrentAndRunDisabledReasonText: "cannot-cancel"
    })
    expect(idle.result.current.cancelCurrentAndRunDisabledReason).toBeNull()

    const { hook: streamingButAllowed } = renderComposerQueue({
      isStreaming: true,
      cancelCurrentAndRunDisabledReasonText: null
    })
    expect(
      streamingButAllowed.result.current.cancelCurrentAndRunDisabledReason
    ).toBeNull()
  })

  it("auto-drain: does NOT call flushNext when isQueuedDispatchBlocked is true", () => {
    const queued = makeQueuedRequest({ id: "q-1", status: "queued" })
    renderComposerQueue({
      queuedMessages: [queued],
      isQueuedDispatchBlocked: true
    })

    expect(mockActions.flushNext).not.toHaveBeenCalled()
  })

  it("auto-drain: calls flushNext on mount when head is queued and all gates are open", async () => {
    const queued = makeQueuedRequest({ id: "q-1", status: "queued" })
    renderComposerQueue({
      queuedMessages: [queued],
      isConnectionReady: true,
      isStreaming: false,
      isQueuedDispatchBlocked: false
    })

    // The effect fires synchronously after render; flushNext is called.
    expect(mockActions.flushNext).toHaveBeenCalledTimes(1)
  })
})
