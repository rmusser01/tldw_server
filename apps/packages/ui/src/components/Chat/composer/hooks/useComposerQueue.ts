import React from "react"

import { useQueuedRequests } from "@/hooks/chat/useQueuedRequests"
import type {
  QueuedRequest,
  QueuedRequestInput,
  QueuedRequestSnapshot
} from "@/utils/chat-request-queue"
import type { ChatDocuments } from "@/models/ChatTypes"

/**
 * Shared queue-orchestration primitive for both composer surfaces.
 *
 * Both Playground and Sidepanel wrap `useQueuedRequests` with near-identical
 * enqueue + run-now + run-next + auto-drain orchestration. This primitive
 * deduplicates that wrapper. The surface-specific bits (token-budget
 * projection warnings, compare-mode validation, image-gen metadata,
 * attachment-type rules, i18n strings) stay in the caller and are either
 * applied before calling `enqueue` or threaded through the callback options.
 *
 * What this primitive OWNS:
 *   - the `useQueuedRequests` instance + its pass-through actions
 *   - the enqueue flow (block-guard → build input → notify success)
 *   - the `cancelCurrentAndRunDisabledReason` derivation (both surfaces
 *     currently use the same rule: streaming + server-backed conversation)
 *   - `handleRunQueuedRequest` and `handleRunNextQueuedRequest`
 *   - the auto-drain effect that pulls items off the queue when the
 *     connection is ready and the composer is not blocked
 *
 * What STAYS in surface wrappers (enforce before calling `enqueue`):
 *   - Token-budget projection warnings (Playground only)
 *   - Compare-mode + model-availability validation (Playground only)
 *   - Image-generation metadata routing (Playground, via its `sendMessage`)
 *   - Draft-attachment rules (each surface decides what blocks the queue)
 *   - All translated strings — this hook never calls `t()`
 *
 * The `enqueue` callback accepts a `blockedReason` so callers can short-
 * circuit for their own reasons (e.g. Playground's
 * `isQueuedDispatchBlockedByComposerState` guard) without duplicating the
 * notify-and-return-null pattern.
 */

/** Arguments passed to the primitive's `enqueue(...)` callback. */
export interface ComposerQueueEnqueueArgs {
  promptText: string
  image: string
  /**
   * Surface-specific flavour info (e.g. `isImageCommand`, documents) — the
   * primitive doesn't introspect this. Forwarded verbatim to the underlying
   * `enqueue`'s `sourceContext` field.
   */
  sourceContext?: Record<string, unknown> | null
  /** Attachments (documents) forwarded verbatim. */
  attachments?: unknown[]
  /**
   * If non-null, `onEnqueueBlocked(reason)` fires and `enqueue` returns null
   * without touching the queue. Use for caller-specific guards like
   * "cannot queue while the current draft has attached files".
   */
  blockedReason?: string | null
}

export interface UseComposerQueueOptions {
  /** True when the server connection is ready to dispatch. */
  isConnectionReady: boolean
  /** True when a request is actively streaming. */
  isStreaming: boolean
  /** Current queue state. */
  queuedMessages: QueuedRequest[]
  /** Setter for the queue (supports value or updater function). */
  setQueuedMessages: (
    value: QueuedRequest[] | ((prev: QueuedRequest[]) => QueuedRequest[])
  ) => void

  /**
   * Surface-specific sender that dispatches a single queued item through
   * its own `sendMessage`. Called by `flushNext` from `useQueuedRequests`.
   * May throw — the underlying primitive converts a throw into a blocked
   * state on the queue head item.
   */
  sendQueuedRequest: (item: QueuedRequest) => Promise<void>

  /**
   * Cancel the in-flight streaming turn. Called by `useQueuedRequests.runNow`
   * to preempt an active stream when a user promotes a queued item.
   */
  stopStreamingRequest: (options?: { discardTurn?: boolean }) => void

  /**
   * Resolve the conversation id to attach to the new queued item. Playground
   * uses `historyId`; Sidepanel uses `historyId ?? serverChatId`.
   */
  resolveConversationId: () => string | null

  /** Build the documents list at enqueue time (typically selected tabs). */
  buildQueuedDocuments: () => ChatDocuments

  /**
   * Build the snapshot at enqueue time. Each surface returns its own shape
   * (Playground includes compare-mode fields; Sidepanel always passes false).
   */
  buildQueuedRequestSnapshot: () => Partial<QueuedRequestSnapshot>

  /**
   * True if this surface's current draft blocks queue dispatch (e.g. has
   * attachments that can't be queued). When true, the auto-drain effect
   * will not run and `handleRunNextQueuedRequest` will still route normally
   * (it's up to the surface's pre-flight guard to stop before calling).
   */
  isQueuedDispatchBlocked: boolean

  /**
   * Fires when `enqueue` is called with a non-null `blockedReason`. Use to
   * show a surface-specific toast/notification. The hook itself emits no UI.
   */
  onEnqueueBlocked?: (reason: string) => void

  /**
   * Fires after the underlying `queuedRequestActions.enqueue` has pushed
   * a new item. `isStreaming` is the value at the time of enqueue — both
   * surfaces use it to pick between "will run after current response" vs
   * "will send when reconnected" copy.
   */
  onEnqueueSuccess?: (isStreaming: boolean, item: QueuedRequest) => void

  /**
   * Translated string for the "cancel current & run now not allowed"
   * disabled reason, shown to users when they try to promote a queued
   * item during a server-backed turn. Pass `null` to allow that action.
   */
  cancelCurrentAndRunDisabledReasonText: string | null
}

export interface UseComposerQueueResult {
  /** Pass-through to the underlying `useQueuedRequests` actions. */
  queuedRequestActions: ReturnType<typeof useQueuedRequests>
  /**
   * Enqueue a new item. Returns null if `blockedReason` was set;
   * otherwise returns the newly built `QueuedRequest`.
   */
  enqueue: (args: ComposerQueueEnqueueArgs) => QueuedRequest | null
  /**
   * Promote `requestId` to the head of the queue and run it. No-op when
   * streaming is active and the caller has passed a disabled reason.
   */
  handleRunQueuedRequest: (requestId: string) => Promise<void>
  /**
   * Run the next item in the queue. Routes blocked head items through
   * `handleRunQueuedRequest` so they get promoted + unblocked first.
   */
  handleRunNextQueuedRequest: () => Promise<void>
  /**
   * Non-null when "cancel current & run now" should be disabled. Mirrors
   * the `cancelCurrentAndRunDisabledReasonText` input when streaming; else
   * null. Provided as a convenience so callers can pass it to button UIs.
   */
  cancelCurrentAndRunDisabledReason: string | null
}

export function useComposerQueue(
  options: UseComposerQueueOptions
): UseComposerQueueResult {
  const {
    isConnectionReady,
    isStreaming,
    queuedMessages,
    setQueuedMessages,
    sendQueuedRequest,
    stopStreamingRequest,
    resolveConversationId,
    buildQueuedDocuments,
    buildQueuedRequestSnapshot,
    isQueuedDispatchBlocked,
    onEnqueueBlocked,
    onEnqueueSuccess,
    cancelCurrentAndRunDisabledReasonText
  } = options

  const queuedRequestActions = useQueuedRequests({
    isConnectionReady,
    isStreaming,
    queue: queuedMessages,
    setQueue: setQueuedMessages,
    sendQueuedRequest,
    stopStreamingRequest
  })

  const cancelCurrentAndRunDisabledReason = isStreaming
    ? cancelCurrentAndRunDisabledReasonText
    : null

  const enqueue = React.useCallback(
    (args: ComposerQueueEnqueueArgs): QueuedRequest | null => {
      if (args.blockedReason) {
        onEnqueueBlocked?.(args.blockedReason)
        return null
      }

      const documents = buildQueuedDocuments()
      const snapshot = buildQueuedRequestSnapshot()

      const input: QueuedRequestInput = {
        conversationId: resolveConversationId(),
        promptText: args.promptText,
        image: args.image,
        attachments: args.attachments ?? documents,
        sourceContext: args.sourceContext ?? null,
        snapshot
      }

      const queuedItem = queuedRequestActions.enqueue(input)
      onEnqueueSuccess?.(isStreaming, queuedItem)
      return queuedItem
    },
    [
      buildQueuedDocuments,
      buildQueuedRequestSnapshot,
      isStreaming,
      onEnqueueBlocked,
      onEnqueueSuccess,
      queuedRequestActions,
      resolveConversationId
    ]
  )

  const handleRunQueuedRequest = React.useCallback(
    async (requestId: string) => {
      if (isStreaming && cancelCurrentAndRunDisabledReason) {
        return
      }
      await queuedRequestActions.runNow(requestId)
      if (!isStreaming && isConnectionReady) {
        await queuedRequestActions.flushNext()
      }
    },
    [
      cancelCurrentAndRunDisabledReason,
      isConnectionReady,
      isStreaming,
      queuedRequestActions
    ]
  )

  const handleRunNextQueuedRequest = React.useCallback(async () => {
    const next = queuedMessages[0]
    if (!next) return
    if (isStreaming && cancelCurrentAndRunDisabledReason) {
      return
    }
    if (next.status === "blocked") {
      await handleRunQueuedRequest(next.id)
      return
    }
    await queuedRequestActions.flushNext()
  }, [
    cancelCurrentAndRunDisabledReason,
    handleRunQueuedRequest,
    isStreaming,
    queuedMessages,
    queuedRequestActions
  ])

  // Auto-drain queued requests when the composer is ready and idle.
  const autoDrainingRef = React.useRef(false)
  React.useEffect(() => {
    const next = queuedMessages[0]
    if (
      autoDrainingRef.current ||
      !next ||
      !isConnectionReady ||
      isStreaming ||
      next.status !== "queued" ||
      isQueuedDispatchBlocked
    ) {
      return
    }

    autoDrainingRef.current = true
    void queuedRequestActions.flushNext().finally(() => {
      autoDrainingRef.current = false
    })
  }, [
    isConnectionReady,
    isQueuedDispatchBlocked,
    isStreaming,
    queuedMessages,
    queuedRequestActions
  ])

  return {
    queuedRequestActions,
    enqueue,
    handleRunQueuedRequest,
    handleRunNextQueuedRequest,
    cancelCurrentAndRunDisabledReason
  }
}
