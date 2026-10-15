import React from "react"

import {
  blockQueuedRequest,
  buildQueuedRequest,
  moveQueuedRequestToFront,
  type QueuedRequest,
  type QueuedRequestInput
} from "@/utils/chat-request-queue"

type UseQueuedRequestsOptions = {
  isConnectionReady: boolean
  isStreaming: boolean
  queue: QueuedRequest[]
  setQueue: (
    queueOrUpdater:
      | QueuedRequest[]
      | ((prev: QueuedRequest[]) => QueuedRequest[])
  ) => void
  sendQueuedRequest: (item: QueuedRequest, assertDispatchReady: () => void) => Promise<void>
  canCommitDispatchResult?: () => boolean
  stopStreamingRequest: (options?: { discardTurn?: boolean }) => void
}

export function useQueuedRequests({
  isConnectionReady,
  isStreaming,
  queue,
  setQueue,
  sendQueuedRequest,
  canCommitDispatchResult,
  stopStreamingRequest
}: UseQueuedRequestsOptions) {
  const connectionReadyRef = React.useRef(isConnectionReady)
  const connectionGenerationRef = React.useRef(0)
  React.useLayoutEffect(() => {
    if (connectionReadyRef.current && !isConnectionReady) connectionGenerationRef.current += 1
    connectionReadyRef.current = isConnectionReady
  }, [isConnectionReady])

  const enqueue = React.useCallback(
    (partial: QueuedRequestInput) => {
      if (!connectionReadyRef.current) return null
      const nextItem = buildQueuedRequest(partial)
      setQueue((prev) => [...prev, nextItem])
      return nextItem
    },
    [setQueue]
  )

  const remove = React.useCallback(
    (requestId: string) => {
      setQueue((prev) => prev.filter((item) => item.id !== requestId))
    },
    [setQueue]
  )

  const clear = React.useCallback(() => {
    setQueue([])
  }, [setQueue])

  const update = React.useCallback(
    (requestId: string, promptText: string) => {
      setQueue(
        (prev) =>
          prev.map((item) =>
            item.id === requestId && item.status !== "sending"
              ? {
                  ...item,
                  promptText,
                  message: promptText,
                  updatedAt: Date.now()
                }
              : item
          )
      )
    },
    [setQueue]
  )

  const move = React.useCallback(
    (requestId: string, direction: "up" | "down") => {
      setQueue((prev) => {
        const currentIndex = prev.findIndex((item) => item.id === requestId)
        if (currentIndex === -1) return prev
        if (prev[currentIndex]?.status === "sending") return prev
        const targetIndex =
          direction === "up" ? currentIndex - 1 : currentIndex + 1
        if (targetIndex < 0 || targetIndex >= prev.length) return prev
        const reordered = [...prev]
        const [target] = reordered.splice(currentIndex, 1)
        reordered.splice(targetIndex, 0, target)
        return reordered
      })
    },
    [setQueue]
  )

  const markBlocked = React.useCallback(
    (requestId: string, blockedReason: string) => {
      setQueue(
        (prev) =>
          prev.map((item) =>
            item.id === requestId ? blockQueuedRequest(item, blockedReason) : item
          )
      )
    },
    [setQueue]
  )

  const runNow = React.useCallback(
    async (requestId: string) => {
      if (!connectionReadyRef.current) return null
      let promotedItem: QueuedRequest | null = null
      setQueue((prev) => {
        const reordered: QueuedRequest[] = moveQueuedRequestToFront(
          prev,
          requestId
        ).map((item, index) =>
          index === 0
            ? {
                ...item,
                status: "queued" as const,
                blockedReason: null,
                updatedAt: Date.now()
              }
            : item
        )
        promotedItem = reordered[0] ?? null
        return reordered
      })
      if (isStreaming) {
        stopStreamingRequest({ discardTurn: true })
      }
      return promotedItem
    },
    [isStreaming, setQueue, stopStreamingRequest]
  )

  const flushNext = React.useCallback(async () => {
    if (
      !connectionReadyRef.current ||
      isStreaming ||
      (canCommitDispatchResult && !canCommitDispatchResult())
    ) {
      return null
    }

    const connectionGeneration = connectionGenerationRef.current
    const assertDispatchReady = () => {
      if (!connectionReadyRef.current || connectionGenerationRef.current !== connectionGeneration) {
        throw new Error("connection_unavailable")
      }
      if (canCommitDispatchResult?.() === false) throw new Error("chat_source_changed")
    }
    let sendingItem: QueuedRequest | null = null
    setQueue((prev) => {
      const next = prev[0]
      if (
        !next ||
        next.status === "blocked" ||
        next.status === "sending"
      ) {
        return prev
      }

      sendingItem = {
        ...next,
        status: "sending",
        blockedReason: null,
        updatedAt: Date.now()
      }

      return [sendingItem, ...prev.slice(1)] as QueuedRequest[]
    })

    if (!sendingItem) {
      return null
    }

    try {
      assertDispatchReady()
      await sendQueuedRequest(sendingItem, assertDispatchReady)
      if (canCommitDispatchResult && !canCommitDispatchResult()) return null
      setQueue((prev) => prev.filter((item) => item.id !== sendingItem?.id))
      return sendingItem
    } catch (error) {
      if (canCommitDispatchResult && !canCommitDispatchResult()) return null
      const blockedReason =
        error instanceof Error ? error.message : "dispatch_failed"
      setQueue((prev) =>
        prev.map((item) =>
          item.id === sendingItem?.id
            ? blockQueuedRequest(item, blockedReason)
            : item
        )
      )
      return null
    }
  }, [canCommitDispatchResult, isStreaming, sendQueuedRequest, setQueue])

  return {
    clear,
    enqueue,
    flushNext,
    markBlocked,
    move,
    remove,
    update,
    runNow
  }
}
