import React from "react"
import {
  getFeedbackSessionId,
  submitExplicitFeedback,
  updateChatRating,
  type ExplicitFeedbackRequest
} from "@/services/feedback"
import {
  useFeedbackStore,
  type FeedbackThumb,
  type FeedbackDetail
} from "@/store/feedback"
import { useStoreMessageOption } from "@/store/option"
import { shallow } from "zustand/shallow"
import { useAntdNotification } from "./useAntdNotification"
import { extractSourceFeedbackIds } from "@/utils/feedback"
import { captureChatDraftSource } from "./chat/chat-draft-transition"
import { useHistorySelectionContext } from "./chat/useHistorySelection"
import { useTranslation } from "react-i18next"

type FeedbackContext = {
  messageKey: string
  conversationId?: string | null
  messageId?: string | null
  query?: string | null
}

type FeedbackDetailInput = {
  rating?: number | null
  issues?: string[]
  notes?: string
}

type SourceFeedbackInput = {
  sourceKey: string
  source: any
  thumb: FeedbackThumb
}

const normalizeQuery = (value?: string | null) => {
  if (!value) return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

export const useFeedback = ({
  messageKey,
  conversationId,
  messageId,
  query
}: FeedbackContext) => {
  const selection = useHistorySelectionContext()
  const renderedOwner = selection?.getCurrent?.().owner
  const owner = renderedOwner?.kind === "native" && renderedOwner.conversation_id === conversationId ? renderedOwner : null
  const mounted = React.useRef(true)
  const ratingFence = React.useMemo(() => {
    const fence = selection?.fence?.() ?? (() => false)
    return () => fence() && selection?.getCurrent().owner === owner
  }, [selection, owner])
  const ratingSignal = selection?.getSignal?.()
  const ratingSource = React.useMemo(() => captureChatDraftSource(ratingFence), [ratingFence])
  const notification = useAntdNotification()
  const { serverChatVersion, setServerChatVersion } = useStoreMessageOption(
    (state) => ({
      serverChatVersion: state.serverChatVersion,
      setServerChatVersion: state.setServerChatVersion
    }),
    shallow
  )
  const { t } = useTranslation(["common", "playground"])
  const entry = useFeedbackStore((state) => state.entries[messageKey])
  const setThumb = useFeedbackStore((state) => state.setThumb)
  const setThumbSubmittedAt = useFeedbackStore(
    (state) => state.setThumbSubmittedAt
  )
  const setDetail = useFeedbackStore((state) => state.setDetail)
  const setDetailSubmittedAt = useFeedbackStore(
    (state) => state.setDetailSubmittedAt
  )
  const setDetailIdempotencyKey = useFeedbackStore(
    (state) => state.setDetailIdempotencyKey
  )
  const clearDetailIdempotencyKey = useFeedbackStore(
    (state) => state.clearDetailIdempotencyKey
  )
  const setSourceThumb = useFeedbackStore((state) => state.setSourceThumb)
  const setSourceSubmittedAt = useFeedbackStore(
    (state) => state.setSourceSubmittedAt
  )

  const [isSubmitting, setIsSubmitting] = React.useState(false)
  const [showThanks, setShowThanks] = React.useState(false)
  const [sourceInFlight, setSourceInFlight] = React.useState<Record<string, boolean>>({})
  const thanksTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(
    null
  )
  const sessionId = React.useMemo(() => getFeedbackSessionId(), [])

  const canSubmit = Boolean(messageId || normalizeQuery(query))

  const triggerThanks = React.useCallback(() => {
    setShowThanks(true)
    if (thanksTimerRef.current) {
      clearTimeout(thanksTimerRef.current)
    }
    thanksTimerRef.current = setTimeout(() => {
      setShowThanks(false)
      thanksTimerRef.current = null
    }, 2500)
  }, [])

  const createIdempotencyKey = React.useCallback(() => {
    try {
      return crypto.randomUUID()
    } catch {
      return `idem_${Date.now()}_${Math.random().toString(16).slice(2)}`
    }
  }, [])

  React.useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      if (thanksTimerRef.current) {
        clearTimeout(thanksTimerRef.current)
      }
    }
  }, [])

  const buildBasePayload = React.useCallback(
    (): Pick<
      ExplicitFeedbackRequest,
      "conversation_id" | "message_id" | "query" | "session_id"
    > => ({
      conversation_id: conversationId || undefined,
      message_id: messageId || undefined,
      query: messageId ? undefined : normalizeQuery(query),
      session_id: sessionId
    }),
    [conversationId, messageId, query, sessionId]
  )

  const updateRating = React.useCallback(
    async (rating: number) => {
      const isCurrent = () => mounted.current && ratingSource.isCurrent() && !ratingSignal?.aborted &&
        owner?.validate_lease() === true && selection?.getCurrent().owner === owner &&
        selection?.settingsMode?.(conversationId) === "ordinary"
      if (!conversationId || !owner || !isCurrent() || !Number.isSafeInteger(serverChatVersion) || (serverChatVersion ?? 0) <= 0) return
      try {
        const updated = await updateChatRating(conversationId, rating, serverChatVersion!, {
          requestScope: owner.request_scope, scope: owner.scope, signal: ratingSignal
        })
        const live = useStoreMessageOption.getState()
        if (isCurrent() && live.serverChatId === conversationId && updated.id === conversationId &&
          Number.isSafeInteger(updated.version) && updated.version! > serverChatVersion! &&
          (live.serverChatVersion == null || live.serverChatVersion <= updated.version!))
          setServerChatVersion(updated.version!)
      } catch {
        // Explicit message feedback succeeded; its advisory conversation rating
        // must not replay against a newer version or another owner (TASK-13549).
      }
    },
    [conversationId, owner, ratingSource, ratingSignal, selection, serverChatVersion, setServerChatVersion]
  )

  const submitThumb = React.useCallback(
    async (thumb: FeedbackThumb) => {
      if (!thumb || !canSubmit) return false
      setThumb(messageKey, thumb)
      setThumbSubmittedAt(messageKey, Date.now())
      setIsSubmitting(true)
      try {
        await submitExplicitFeedback({
          ...buildBasePayload(),
          feedback_type: "helpful",
          helpful: thumb === "up"
        })
        triggerThanks()
      } catch (e: any) {
        notification.error({
          message: t("playground:feedback.errorTitle", "Feedback failed"),
          description:
            e?.message ||
            t("playground:feedback.errorBody", "Unable to submit feedback.")
        })
        setIsSubmitting(false)
        return false
      }
      setIsSubmitting(false)
      if (conversationId) {
        await updateRating(thumb === "up" ? 5 : 1)
      }
      return true
    },
    [
      buildBasePayload,
      canSubmit,
      conversationId,
      messageKey,
      notification,
      setThumb,
      setThumbSubmittedAt,
      t,
      triggerThanks,
      updateRating
    ]
  )

  const sendDetailedFeedback = React.useCallback(
    async (
      kind: "report" | "relevance",
      payload: ExplicitFeedbackRequest,
      currentKey?: string
    ) => {
      const key = currentKey || createIdempotencyKey()
      setDetailIdempotencyKey(messageKey, kind, key)
      await submitExplicitFeedback({
        ...payload,
        idempotency_key: key
      })
      clearDetailIdempotencyKey(messageKey, kind)
    },
    [
      clearDetailIdempotencyKey,
      createIdempotencyKey,
      messageKey,
      setDetailIdempotencyKey
    ]
  )

  const submitDetail = React.useCallback(
    async ({ rating, issues, notes }: FeedbackDetailInput) => {
      const hasRating = typeof rating === "number" && rating > 0
      const hasIssues = Array.isArray(issues) && issues.length > 0
      const hasNotes = Boolean(notes && notes.trim().length > 0)
      if (!canSubmit || (!hasRating && !hasIssues && !hasNotes)) {
        return false
      }

      setIsSubmitting(true)
      try {
        const base = buildBasePayload()
        const detailKind: "relevance" | "report" = hasRating
          ? "relevance"
          : "report"
        const currentKey =
          detailKind === "relevance"
            ? entry?.detailIdempotencyKeys?.relevance
            : entry?.detailIdempotencyKeys?.report
        await sendDetailedFeedback(
          detailKind,
          {
            ...base,
            feedback_type: detailKind,
            relevance_score: hasRating ? rating || undefined : undefined,
            issues: hasIssues ? issues : undefined,
            user_notes: hasNotes ? notes?.trim() : undefined
          },
          currentKey
        )

        const detail: FeedbackDetail = {
          rating: hasRating ? rating : null,
          issues: hasIssues ? issues : [],
          notes: hasNotes ? notes?.trim() : undefined
        }
        setDetail(messageKey, detail)
        setDetailSubmittedAt(messageKey, Date.now())
        triggerThanks()
      } catch (e: any) {
        notification.error({
          message: t("playground:feedback.errorTitle", "Feedback failed"),
          description:
            e?.message ||
            t("playground:feedback.errorBody", "Unable to submit feedback.")
        })
        setIsSubmitting(false)
        return false
      }
      setIsSubmitting(false)
      return true
    },
    [
      buildBasePayload,
      canSubmit,
      entry?.detailIdempotencyKeys?.relevance,
      entry?.detailIdempotencyKeys?.report,
      messageKey,
      notification,
      sendDetailedFeedback,
      setDetail,
      setDetailSubmittedAt,
      t,
      triggerThanks
    ]
  )

  const submitSourceThumb = React.useCallback(
    async ({ sourceKey, source, thumb }: SourceFeedbackInput) => {
      if (!thumb || !canSubmit) return false

      // Prevent concurrent submissions for the same source
      if (sourceInFlight[sourceKey]) {
        return false
      }

      // Mark this source as in-flight
      setSourceInFlight((prev) => ({ ...prev, [sourceKey]: true }))

      // Optimistic update
      setSourceThumb(messageKey, sourceKey, thumb)
      setSourceSubmittedAt(messageKey, sourceKey, Date.now())

      try {
        const { documentIds, chunkIds, corpus } =
          extractSourceFeedbackIds(source)
        await submitExplicitFeedback({
          ...buildBasePayload(),
          feedback_type: "helpful",
          helpful: thumb === "up",
          document_ids: documentIds.length > 0 ? documentIds : undefined,
          chunk_ids: chunkIds.length > 0 ? chunkIds : undefined,
          corpus
        })
        return true
      } catch (e: any) {
        // Rollback optimistic update on error
        setSourceThumb(messageKey, sourceKey, null)
        setSourceSubmittedAt(messageKey, sourceKey, 0)

        notification.error({
          message: t("playground:feedback.errorTitle", "Feedback failed"),
          description:
            e?.message ||
            t(
              "playground:feedback.errorSource",
              "Unable to submit source feedback."
            )
        })
        return false
      } finally {
        // Always clean up in-flight tracking
        setSourceInFlight((prev) => {
          const next = { ...prev }
          delete next[sourceKey]
          return next
        })
      }
    },
    [
      buildBasePayload,
      canSubmit,
      messageKey,
      notification,
      setSourceSubmittedAt,
      setSourceThumb,
      sourceInFlight,
      t
    ]
  )

  return {
    entry,
    thumb: entry?.thumb ?? null,
    detail: entry?.detail ?? null,
    sourceFeedback: entry?.sourceFeedback ?? {},
    canSubmit,
    isSubmitting,
    showThanks,
    submitThumb,
    submitDetail,
    submitSourceThumb
  }
}
