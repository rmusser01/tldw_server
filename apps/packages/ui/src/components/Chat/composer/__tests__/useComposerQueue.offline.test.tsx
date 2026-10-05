import { act, renderHook } from "@testing-library/react"
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
