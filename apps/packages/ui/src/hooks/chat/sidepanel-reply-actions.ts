/**
 * Reply actions the side panel's `useMessage` offers on its last reply:
 * Continue (CM-03), Retry of a kept turn, and Retry of a saved question.
 *
 * Continue follows the full page's rules. A saved server chat continues its
 * last reply through the server (D7), so the original reply is kept and the
 * continuation is saved as an alternative to it; any other chat (local or
 * temporary) extends the reply with the ordinary continue mode.
 *
 * These live here so the frozen `useMessage.tsx` doesn't grow.
 */
import React from "react"
import { chatSubmitFailed, type ChatSubmitResult } from "@/hooks/chat/chat-action-utils"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { createContinueSavedReply, savedReplyActionFeedback } from "@/hooks/chat/native-history-continue"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { HistoryTurnRecoveryEntry } from "@/services/history-turn-keep"
import {
  loadHistoryRecoveryRetryContext,
  retryHistoryTurnRecovery,
  type HistoryRecoveryRetryContext,
  type SavedQuestionReview
} from "@/services/history-turn-retry"
import { useStoreMessageOption } from "@/store/option"

type SubmitResult = Promise<ChatSubmitResult | void | undefined>

type ContinueSavedReplySubmit = Parameters<typeof createContinueSavedReply>[0]["onSubmit"]
type RecoveryRetrySubmit = Parameters<typeof retryHistoryTurnRecovery>[3]

/** The side panel's `onSubmit`, as these actions call it. */
export type SidepanelReplySubmit = ContinueSavedReplySubmit &
  RecoveryRetrySubmit &
  ((params: {
    message: string
    image: string
    isContinue?: true
    requestOverrides?: { chatMode: "normal" }
    historyServerExistingInput?: SavedQuestionReview
  }) => SubmitResult)

export const useSidepanelReplyActions = ({
  historySelection,
  onSubmit,
  actionsMountedRef,
  retrySource,
  t,
  notification
}: {
  historySelection: HistorySelectionController | null | undefined
  onSubmit: SidepanelReplySubmit
  actionsMountedRef: React.MutableRefObject<boolean>
  /** The composer settings a retry is reviewed against. */
  retrySource: { model: string; selectedSystemPrompt: string | null; systemPrompt?: string | null }
  t: (key: string, fallback: string) => string
  notification: { error: (options: { message: string; description?: string }) => void }
}) => {
  const recoveryRetrySource = React.useRef(retrySource)
  recoveryRetrySource.current = retrySource
  const getRecoveryRetryContext = () => {
    const source = { ...recoveryRetrySource.current }
    return loadHistoryRecoveryRetryContext(source, () => JSON.stringify(source) === JSON.stringify(recoveryRetrySource.current))
  }
  const retryRecoveredHistoryTurn = (
    entry: HistoryTurnRecoveryEntry,
    review: { contextReviewed: boolean; context: HistoryRecoveryRetryContext }
  ) => retryHistoryTurnRecovery(historySelection, entry, review, onSubmit, getRecoveryRetryContext)

  const getSavedQuestionReview = async () => {
    const { reviewServerSavedQuestion } = await import("@/services/history-turn-retry")
    return reviewServerSavedQuestion(historySelection, await getRecoveryRetryContext())
  }
  const retrySavedQuestion = async (review: SavedQuestionReview): Promise<ChatSubmitResult> => {
    if (!review.isCurrent()) return chatSubmitFailed("history_selection_changed")
    const submitted = (await onSubmit({ message: review.text, image: "", historyServerExistingInput: review })) as
      ChatSubmitResult | undefined
    return submitted ?? chatSubmitFailed("native_history_existing_input_not_submitted")
  }

  const continueSavedReply = createContinueSavedReply({
    historySelection,
    onSubmit,
    validateBeforeSubmitFn: () => true,
    getMessages: () => useStoreMessageOption.getState().messages,
    captureSourceFence: () => captureChatDraftSource(() => actionsMountedRef.current).isCurrent,
    captureViewFence: () => captureChatDraftSource(historySelection?.fence()).isCurrent
  })
  /**
   * Continue the last reply. A chat that isn't a saved server chat continues
   * in the normal chat mode: the continuation extends the reply's own text,
   * so knowledge search and vision don't apply to it.
   */
  const continueLastMessage = async () => {
    try {
      // `null` means "not a saved server chat"; a saved chat's own result, even
      // an empty one, never falls through to a local continuation.
      const saved = await continueSavedReply()
      if (saved !== null) return saved
      return await onSubmit({ message: "", image: "", isContinue: true, requestOverrides: { chatMode: "normal" } })
    } catch (error) {
      // A refusal before dispatch would otherwise be silent: say it in Continue wording.
      const reason = error instanceof Error ? error.message : String(error)
      notification.error({
        message: t("error", "Error"),
        description: savedReplyActionFeedback(error, t, undefined, true) ?? reason
      })
      return chatSubmitFailed(reason)
    }
  }

  return {
    continueLastMessage,
    getRecoveryRetryContext,
    retryRecoveredHistoryTurn,
    getSavedQuestionReview,
    retrySavedQuestion
  }
}
