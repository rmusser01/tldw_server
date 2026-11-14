import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { useStoreMessageOption } from "@/store/option"
import { useQueuedRequests } from "@/hooks/chat/useQueuedRequests"
import { buildQueuedRequest, type QueuedRequest } from "@/utils/chat-request-queue"

const createQueueSetter = (queueRef: { current: QueuedRequest[] }) =>
  vi.fn((nextQueueOrUpdater: QueuedRequest[] | ((prev: QueuedRequest[]) => QueuedRequest[])) => {
    queueRef.current =
      typeof nextQueueOrUpdater === "function"
        ? nextQueueOrUpdater(queueRef.current)
        : nextQueueOrUpdater
  })

describe("useQueuedRequests", () => {
  it("promotes a queued request to the front and stops streaming before run-now", async () => {
    const stopStreamingRequest = vi.fn()
    const first = buildQueuedRequest({ promptText: "one" })
    const second = buildQueuedRequest({ promptText: "two" })
    const queueRef = {
      current: [first, second]
    }
    const setQueue = createQueueSetter(queueRef)

    const { result } = renderHook(() =>
      useQueuedRequests({
        isConnectionReady: true,
        isStreaming: true,
        queue: queueRef.current,
        setQueue,
        sendQueuedRequest: vi.fn(),
        stopStreamingRequest
      })
    )

    await act(async () => {
      await result.current.runNow(second.id)
    })

    expect(stopStreamingRequest).toHaveBeenCalledTimes(1)
    expect(stopStreamingRequest).toHaveBeenCalledWith({
      discardTurn: true
    })
    expect(queueRef.current).toEqual([
      expect.objectContaining({
        id: second.id,
        status: "queued",
        blockedReason: null
      }),
      expect.objectContaining({
        id: first.id
      })
    ])
  })

  it("refuses offline enqueue and only flushes after an explicit connected call", async () => {
    const sendQueuedRequest = vi.fn().mockResolvedValue(undefined)
    const queueRef = {
      current: [buildQueuedRequest({ promptText: "existing" })]
    }
    const setQueue = createQueueSetter(queueRef)

    const { result, rerender } = renderHook(
      ({
        ready,
        streaming
      }: {
        ready: boolean
        streaming: boolean
      }) =>
        useQueuedRequests({
          isConnectionReady: ready,
          isStreaming: streaming,
          queue: queueRef.current,
          setQueue,
          sendQueuedRequest,
          stopStreamingRequest: vi.fn()
        }),
      {
        initialProps: { ready: false, streaming: false }
      }
    )

    await act(async () => {
      expect(result.current.enqueue({ message: "queued later" })).toBeNull()
    })

    expect(queueRef.current.map(item => item.promptText)).toEqual(["existing"])

    await act(async () => {
      await result.current.flushNext()
    })

    expect(sendQueuedRequest).not.toHaveBeenCalled()

    rerender({ ready: true, streaming: false })

    await act(async () => {
      result.current.enqueue({ message: "queued later" })
      await result.current.flushNext()
    })
    expect(queueRef.current.at(-1)?.promptText).toBe("queued later")

    expect(sendQueuedRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        promptText: "existing",
        status: "sending"
      }),
      expect.any(Function)
    )
  })

  it("refuses source recapture after a synchronous New subscriber sees sending publication", async () => {
    const store = useStoreMessageOption
    store.getState().startNewChatDraft()
    const first = buildQueuedRequest({ promptText: "Original source question" })
    store.getState().setQueuedMessages([first])
    const source = store.getState()
    const originalSourceIsCurrent = () => store.getState().chatDraftAddress === source.chatDraftAddress &&
      store.getState().chatDraftRevision === source.chatDraftRevision
    const sendQueuedRequest = vi.fn(async (_item: QueuedRequest, assertReady: () => void) => { assertReady() })
    let replaced = false
    const unsubscribe = store.subscribe(state => {
      if (!replaced && state.queuedMessages[0]?.status === "sending") {
        replaced = true
        state.startNewChatDraft()
        store.getState().setQueuedMessages([buildQueuedRequest({ promptText: "Replacement source question" })])
      }
    })
    const view = renderHook(() => useQueuedRequests({ isConnectionReady: true, isStreaming: false,
      queue: source.queuedMessages, setQueue: source.setQueuedMessages, sendQueuedRequest,
      canCommitDispatchResult: originalSourceIsCurrent, stopStreamingRequest: vi.fn() }))
    try {
      await act(async () => { expect(await view.result.current.flushNext()).toBeNull() })
      expect(sendQueuedRequest).not.toHaveBeenCalled()
      expect(store.getState().queuedMessages.map(item => item.promptText)).toEqual(["Replacement source question"])
    } finally { unsubscribe(); view.unmount() }
  })

  it("removes the flushed request after a successful send", async () => {
    const first = buildQueuedRequest({ promptText: "first" })
    const second = buildQueuedRequest({ promptText: "second" })
    const queueRef = {
      current: [first, second]
    }
    const setQueue = createQueueSetter(queueRef)
    const sendQueuedRequest = vi.fn().mockResolvedValue(undefined)

    const { result } = renderHook(() =>
      useQueuedRequests({
        isConnectionReady: true,
        isStreaming: false,
        queue: queueRef.current,
        setQueue,
        sendQueuedRequest,
        stopStreamingRequest: vi.fn()
      })
    )

    await act(async () => {
      await result.current.flushNext()
    })

    expect(sendQueuedRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        id: first.id,
        status: "sending"
      }),
      expect.any(Function)
    )
    expect(queueRef.current.map((item) => item.id)).toEqual([second.id])
  })

  it("retains a queued request while its source is not ready and dispatches after a ready render", async () => {
    const first = buildQueuedRequest({ promptText: "waiting for draft hydration" })
    const initialQueue = [first]
    const queueRef = { current: initialQueue }
    const setQueue = createQueueSetter(queueRef)
    const sendQueuedRequest = vi.fn().mockResolvedValue(undefined)

    const { result, rerender } = renderHook(
      ({ ready }: { ready: boolean }) =>
        useQueuedRequests({
          isConnectionReady: true,
          isStreaming: false,
          queue: queueRef.current,
          setQueue,
          sendQueuedRequest,
          canCommitDispatchResult: () => ready,
          stopStreamingRequest: vi.fn()
        }),
      { initialProps: { ready: false } }
    )

    await act(async () => {
      expect(await result.current.flushNext()).toBeNull()
    })

    expect(sendQueuedRequest).not.toHaveBeenCalled()
    expect(setQueue).not.toHaveBeenCalled()
    expect(queueRef.current).toBe(initialQueue)

    rerender({ ready: true })

    await act(async () => {
      expect(await result.current.flushNext()).toMatchObject({
        id: first.id,
        promptText: "waiting for draft hydration",
        status: "sending"
      })
    })

    expect(sendQueuedRequest).toHaveBeenCalledTimes(1)
    expect(queueRef.current).toEqual([])
  })

  it.each(["success", "failure"] as const)(
    "keeps the captured source predicate through an in-flight %s after a ready rerender",
    async (outcome) => {
      const first = buildQueuedRequest({ promptText: "original source" })
      const queueRef = { current: [first] }
      const setQueue = createQueueSetter(queueRef)
      let sourceCurrent = true
      const originalPredicate = () => sourceCurrent
      let resolveSend: (() => void) | undefined
      let rejectSend: ((error: Error) => void) | undefined
      const sendQueuedRequest = vi.fn(
        () =>
          new Promise<void>((resolve, reject) => {
            resolveSend = resolve
            rejectSend = reject
          })
      )

      const { result, rerender } = renderHook(
        ({ canCommit }: { canCommit: () => boolean }) =>
          useQueuedRequests({
            isConnectionReady: true,
            isStreaming: false,
            queue: queueRef.current,
            setQueue,
            sendQueuedRequest,
            canCommitDispatchResult: canCommit,
            stopStreamingRequest: vi.fn()
          }),
        { initialProps: { canCommit: originalPredicate } }
      )

      let flushPromise: Promise<QueuedRequest | null> | undefined
      await act(async () => {
        flushPromise = result.current.flushNext()
        await Promise.resolve()
      })

      expect(queueRef.current[0]?.status).toBe("sending")
      sourceCurrent = false
      const replacementQueue = [{ ...first, promptText: "replacement source" }]
      queueRef.current = replacementQueue
      rerender({ canCommit: () => true })

      await act(async () => {
        if (outcome === "success") {
          resolveSend?.()
        } else {
          rejectSend?.(new Error("original dispatch failed"))
        }
        expect(await flushPromise).toBeNull()
      })

      expect(queueRef.current).toBe(replacementQueue)
      expect(sendQueuedRequest).toHaveBeenCalledTimes(1)
    }
  )

  it("does not restore queued items after clear-all during an in-flight send", async () => {
    const first = buildQueuedRequest({ promptText: "first" })
    const second = buildQueuedRequest({ promptText: "second" })
    const queueRef = {
      current: [first, second]
    }
    const setQueue = createQueueSetter(queueRef)
    let resolveSend: (() => void) | null = null
    const sendQueuedRequest = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveSend = resolve
        })
    )

    const { result } = renderHook(() =>
      useQueuedRequests({
        isConnectionReady: true,
        isStreaming: false,
        queue: queueRef.current,
        setQueue,
        sendQueuedRequest,
        stopStreamingRequest: vi.fn()
      })
    )

    let flushPromise: Promise<QueuedRequest | null> | null = null
    await act(async () => {
      flushPromise = result.current.flushNext()
      await Promise.resolve()
    })

    expect(queueRef.current[0]).toMatchObject({
      id: first.id,
      status: "sending"
    })

    await act(async () => {
      result.current.clear()
    })

    expect(queueRef.current).toEqual([])

    await act(async () => {
      resolveSend?.()
      await flushPromise
    })

    expect(queueRef.current).toEqual([])
  })

  it("blocks the next queued request when dispatch fails", async () => {
    const first = buildQueuedRequest({ promptText: "first" })
    const second = buildQueuedRequest({ promptText: "second" })
    const queueRef = {
      current: [first, second]
    }
    const setQueue = createQueueSetter(queueRef)
    const sendQueuedRequest = vi
      .fn()
      .mockRejectedValue(new Error("selected model unavailable"))

    const { result } = renderHook(() =>
      useQueuedRequests({
        isConnectionReady: true,
        isStreaming: false,
        queue: queueRef.current,
        setQueue,
        sendQueuedRequest,
        stopStreamingRequest: vi.fn()
      })
    )

    await act(async () => {
      await result.current.flushNext()
    })

    expect(queueRef.current[0]).toMatchObject({
      id: first.id,
      status: "blocked",
      blockedReason: "selected model unavailable"
    })
    expect(queueRef.current[1]?.id).toBe(second.id)
  })

  it("does not edit or reorder requests that are already sending", async () => {
    const sending = buildQueuedRequest({
      promptText: "in flight",
      status: "sending"
    })
    const queued = buildQueuedRequest({ promptText: "later" })
    const queueRef = {
      current: [sending, queued]
    }
    const setQueue = createQueueSetter(queueRef)

    const { result } = renderHook(() =>
      useQueuedRequests({
        isConnectionReady: true,
        isStreaming: false,
        queue: queueRef.current,
        setQueue,
        sendQueuedRequest: vi.fn(),
        stopStreamingRequest: vi.fn()
      })
    )

    await act(async () => {
      result.current.update(sending.id, "edited in flight")
      result.current.move(sending.id, "down")
    })

    expect(queueRef.current).toEqual([
      expect.objectContaining({
        id: sending.id,
        promptText: "in flight",
        status: "sending"
      }),
      expect.objectContaining({
        id: queued.id,
        promptText: "later"
      })
    ])
  })
})
