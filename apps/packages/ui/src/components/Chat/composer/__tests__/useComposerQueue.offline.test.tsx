import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { useComposerQueue } from "../hooks/useComposerQueue"
import { buildQueuedRequest, restoreQueuedRequests, type QueuedRequest } from "@/utils/chat-request-queue"

const mountQueue = (ready: boolean, streaming = false, initial: QueuedRequest[] = []) => {
  let queue = initial
  const send = vi.fn(async () => {})
  const onSuccess = vi.fn()
  const stop = vi.fn()
  const setter = (value: QueuedRequest[] | ((prev: QueuedRequest[]) => QueuedRequest[])) => { queue = typeof value === "function" ? value(queue) : value }
  const hook = renderHook(({ ready, streaming }) => useComposerQueue({
    isConnectionReady: ready, isStreaming: streaming, queuedMessages: queue, setQueuedMessages: setter,
    sendQueuedRequest: send, stopStreamingRequest: stop, resolveConversationId: () => "owned-chat",
    buildQueuedDocuments: () => [], buildQueuedRequestSnapshot: () => ({}), isQueuedDispatchBlocked: false,
    onEnqueueSuccess: onSuccess, cancelCurrentAndRunDisabledReasonText: null
  }), { initialProps: { ready, streaming } })
  return { hook, send, onSuccess, stop, readQueue: () => queue }
}

describe("real shared queue without an offline outbox", () => {
  it.each([false, true])("refuses disconnected enqueue while streaming=%s without a success reset", streaming => {
    const test = mountQueue(false, streaming)
    let result: unknown
    act(() => { result = test.hook.result.current.enqueue({ promptText: "KEEP ORIGINAL", image: "KEEP IMAGE" }) })
    expect(result).toBeNull()
    expect(test.readQueue()).toEqual([])
    expect(test.onSuccess).not.toHaveBeenCalled()
    expect(test.send).not.toHaveBeenCalled()
  })
  it("rejects a previously captured enqueue and run callback after disconnect", async () => {
    const test = mountQueue(true, true)
    const old = test.hook.result.current
    test.hook.rerender({ ready: false, streaming: true })
    await act(async () => {
      expect(old.enqueue({ promptText: "STALE", image: "" })).toBeNull()
      await old.handleRunQueuedRequest("absent")
    })
    expect(test.readQueue()).toEqual([])
    expect(test.stop).not.toHaveBeenCalled()
    expect(test.send).not.toHaveBeenCalled()
  })
  it("pauses a connected busy queue on disconnect and reconnect requires explicit run", async () => {
    const test = mountQueue(true, true)
    act(() => { test.hook.result.current.enqueue({ promptText: "BUSY REQUEST", image: "" }) })
    test.hook.rerender({ ready: false, streaming: false })
    expect(test.readQueue()).toMatchObject([{ promptText: "BUSY REQUEST", status: "blocked", blockedReason: "connection_unavailable" }])
    test.hook.rerender({ ready: true, streaming: false })
    await act(async () => {})
    expect(test.send).not.toHaveBeenCalled()
    await act(async () => { await test.hook.result.current.handleRunNextQueuedRequest() })
    expect(test.send).toHaveBeenCalledOnce()
    expect(test.readQueue()).toEqual([])
  })
  it("does not automatically send a legacy queued item loaded while disconnected", async () => {
    const test = mountQueue(false, false, [buildQueuedRequest({ promptText: "OLD OFFLINE OUTBOX" })])
    test.hook.rerender({ ready: true, streaming: false })
    await act(async () => {})
    expect(test.send).not.toHaveBeenCalled()
    expect(test.readQueue()).toMatchObject([{ promptText: "OLD OFFLINE OUTBOX", status: "blocked" }])
  })
  it("still runs the next connected request when the active response finishes", async () => {
    const test = mountQueue(true, true)
    act(() => { test.hook.result.current.enqueue({ promptText: "AFTER THIS RESPONSE", image: "" }) })
    expect(test.send).not.toHaveBeenCalled()
    test.hook.rerender({ ready: true, streaming: false })
    await act(async () => {})
    expect(test.send).toHaveBeenCalledOnce()
    expect(test.readQueue()).toEqual([])
  })
})

describe("restored requests require review even when already connected", () => {
  it.each(["queued", "sending"] as const)("does not auto-send a restored %s request", async status => {
    const restored = restoreQueuedRequests([buildQueuedRequest({ promptText: "REVIEW RESTORED", status })])
    const test = mountQueue(true, false, restored)
    await act(async () => {})
    expect(test.send).not.toHaveBeenCalled()
    expect(test.readQueue()).toMatchObject([{ promptText: "REVIEW RESTORED", status: "blocked", blockedReason: "restored_request" }])
    await act(async () => { await test.hook.result.current.handleRunNextQueuedRequest() })
    expect(test.send).toHaveBeenCalledOnce()
    expect(test.readQueue()).toEqual([])
  })
})


describe("automatic queue dispatch ownership", () => {
  it.each(["success", "failure"] as const)("runs a ready replacement queue while the old source is held and ignores its late %s", async outcome => {
    type Actor = "A" | "B"
    const actor = { current: "A" as Actor }
    const firstB = buildQueuedRequest({ conversationId: "B", promptText: "B first" })
    const queues: Record<Actor, QueuedRequest[]> = {
      A: [buildQueuedRequest({ conversationId: "A", promptText: "A held" })],
      B: [firstB, buildQueuedRequest({ conversationId: "B", promptText: "B next" })]
    }
    let resolveA!: () => void, rejectA!: (error: Error) => void, resolveB!: () => void
    const heldA = new Promise<void>((resolve, reject) => { resolveA = resolve; rejectA = reject })
    const heldB = new Promise<void>(resolve => { resolveB = resolve })
    const wire: string[] = []
    const sendQueuedRequest = async (item: QueuedRequest) => {
      wire.push(item.promptText)
      if (item.promptText === "A held") await heldA
      if (item.promptText === "B first") await heldB
    }
    let publishRender: (() => void) | undefined
    const publishQueue: UseQueueSetter = value => {
      queues[actor.current] = typeof value === "function" ? value(queues[actor.current]) : value
      publishRender?.()
    }
    const setterA = vi.fn(publishQueue), setterB = vi.fn(publishQueue)
    const view = renderHook(({ source, ready }: { source: Actor; ready: boolean }) => {
      const [, publish] = React.useState(0)
      publishRender = () => publish(revision => revision + 1)
      const canCommit = React.useMemo(() => () => ready && actor.current === source, [source, ready])
      return useComposerQueue({
        isConnectionReady: true, isStreaming: false,
        queuedMessages: queues[source], setQueuedMessages: source === "A" ? setterA : setterB,
        sendQueuedRequest, canCommitDispatchResult: canCommit,
        stopStreamingRequest: () => {}, resolveConversationId: () => source,
        buildQueuedDocuments: () => [], buildQueuedRequestSnapshot: () => ({}),
        isQueuedDispatchBlocked: false, cancelCurrentAndRunDisabledReasonText: null
      })
    }, { initialProps: { source: "A" as Actor, ready: true } })

    try {
      await waitFor(() => expect(wire).toEqual(["A held"]))
      const originalWrites = setterA.mock.calls.length
      actor.current = "B"
      view.rerender({ source: "B", ready: false })
      await act(async () => {})
      expect(setterB).not.toHaveBeenCalled()
      expect(queues.B.map(item => item.status)).toEqual(["queued", "queued"])

      view.rerender({ source: "B", ready: true })
      await waitFor(() => expect(wire).toEqual(["A held", "B first"]))
      await act(async () => {
        if (outcome === "success") resolveA()
        else rejectA(new Error("retired A failed"))
        await heldA.catch(() => {})
      })
      expect(setterA).toHaveBeenCalledTimes(originalWrites)

      // An old finally must not unlock B while B's own dispatch is still held.
      act(() => view.result.current.queuedRequestActions.remove(firstB.id))
      await act(async () => {})
      expect(wire).toEqual(["A held", "B first"])
      expect(queues.B.map(item => item.promptText)).toEqual(["B next"])
      await act(async () => { resolveB(); await heldB })
      await waitFor(() => expect(wire).toEqual(["A held", "B first", "B next"]))
      await waitFor(() => expect(queues.B).toEqual([]))
    } finally {
      await act(async () => { resolveA(); resolveB(); await Promise.allSettled([heldA, heldB]) })
    }
  })
})

type UseQueueSetter = (value: QueuedRequest[] | ((previous: QueuedRequest[]) => QueuedRequest[])) => void
