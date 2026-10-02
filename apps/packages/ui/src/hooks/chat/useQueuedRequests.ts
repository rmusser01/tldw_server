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
  sendQueuedRequest: (item: QueuedRequest) => Promise<void>
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
  const enqueue = React.useCallback(
    (partial: QueuedRequestInput) => {
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
    if (!isConnectionReady || isStreaming) {
      return null
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
      await sendQueuedRequest(sendingItem)
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
  }, [canCommitDispatchResult, isConnectionReady, isStreaming, sendQueuedRequest, setQueue])

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
