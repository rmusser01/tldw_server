/**
 * Native Continue for saved chats (CM-01, #3106).
 *
 * Under D7 a saved chat's turns are settled by the server. Continue on a saved
 * reply therefore runs the same server-owned operation as Regenerate: it reuses
 * the saved question's admission, and the server saves the original reply plus
 * the new text as an alternative reply to that question. Nothing is written
 * locally, so the original exchange is kept whatever happens.
 */
import type { ActorSceneSend } from "@/hooks/chat/actor-scene-source"
import { chatSubmitSkipped, type ChatSubmitResult } from "@/hooks/chat/chat-action-utils"
import { nativeRegenerationFeedback } from "@/hooks/chat/native-character-preflight"
import { nativeSavedReplyTarget } from "@/hooks/chat/saved-reply-actions"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { nativeRegenerateView } from "@/services/chat-history-selection"
import { useStoreMessageOption, type Message } from "@/store/option"
import type { NativeHistoryRegenerate } from "@/types/chat-modes"

type Dispatch = {
  controller?: AbortController
  assertCurrent?: () => void
  actorSceneSend?: ActorSceneSend
}

/** True when the selected history is a ready, server-owned saved chat. */
export const isNativeSavedSelection = (historySelection?: HistorySelectionController | null): boolean => {
  const origin = historySelection?.getCurrent()
  return Boolean(
    origin?.status === "ready" &&
      origin.owner?.kind === "native" &&
      origin.view &&
      origin.capture?.status === "captured"
  )
}

/**
 * Build the Continue action for the last saved reply. The returned function
 * resolves `null` when the chat is not a saved server chat, so the caller keeps
 * its ordinary (local or temporary) Continue.
 */
export const createContinueSavedReply = ({
  historySelection,
  getMessages,
  captureViewFence,
  captureSourceFence,
  validateBeforeSubmitFn,
  onSubmit
}: {
  historySelection?: HistorySelectionController | null
  getMessages: () => Message[]
  captureViewFence?: () => () => boolean
  captureSourceFence?: () => () => boolean
  validateBeforeSubmitFn: () => boolean
  onSubmit: (params: {
    message: string
    image: string
    isRegenerate: true
    historyRegenerate: NativeHistoryRegenerate
    controller: AbortController
    actorSceneSend?: ActorSceneSend
    assertCurrent: () => void
  }) => Promise<ChatSubmitResult | void>
}) => {
  const current = captureViewFence?.() ?? historySelection?.fence?.() ?? (() => true)
  const sourceCurrent = captureSourceFence?.() ?? (() => true)
  return async (dispatch?: Dispatch): Promise<ChatSubmitResult | void | null> => {
    if (!isNativeSavedSelection(historySelection)) return null
    dispatch?.controller?.signal.throwIfAborted()
    dispatch?.assertCurrent?.()
    const isCurrent = () => {
      try {
        dispatch?.assertCurrent?.()
        return (
          sourceCurrent() &&
          !dispatch?.controller?.signal.aborted &&
          dispatch?.actorSceneSend?.isCurrent() !== false
        )
      } catch {
        return false
      }
    }
    if (!isCurrent()) return chatSubmitSkipped("actor_settings_scope_changed")
    if (!current()) return chatSubmitSkipped("stale_selection")
    const busy = useStoreMessageOption.getState()
    if (busy.streaming || busy.isProcessing) return chatSubmitSkipped("response_in_progress")
    const target = nativeSavedReplyTarget(historySelection, getMessages())
    if (!target) return chatSubmitSkipped("message_changed")
    const { origin, capture, input, reply, inputContent } = target
    const intent: NativeHistoryRegenerate = {
      owner_key: origin.view.owner_key,
      conversation_id: origin.view.conversation_id,
      message_id: reply.id,
      message_revision: reply.revision,
      input_message_id: input.id,
      input_message_revision: input.revision,
      action: "continue",
      isCurrent,
      isOriginCurrent: () => current() && isCurrent()
    }
    // Validates the same saved question/reply pair Regenerate requires.
    nativeRegenerateView(origin.owner, origin.view, capture, intent)
    if (!validateBeforeSubmitFn()) return chatSubmitSkipped("validation_failed")
    return await onSubmit({
      message: inputContent.message,
      image: inputContent.images[0] ?? "",
      isRegenerate: true,
      historyRegenerate: intent,
      controller: dispatch?.controller ?? new AbortController(),
      actorSceneSend: dispatch?.actorSceneSend,
      assertCurrent: () => {
        if (!intent.isOriginCurrent()) throw new Error("stale_selection")
      }
    })
  }
}

const SAVED_HISTORY_REFUSAL = /^(unsupported_history_|native_history_)/

/**
 * Plain copy for a failed saved-reply action, or `null` to keep the caller's
 * ordinary error text. Continue on a saved chat never shows a raw history code.
 */
export const savedReplyActionFeedback = (
  error: unknown,
  t: (key: string, fallback: string) => string,
  historyRegenerate: NativeHistoryRegenerate | undefined,
  isContinue: boolean | undefined
): string | null => {
  if (historyRegenerate)
    return nativeRegenerationFeedback(error, t, historyRegenerate.action === "continue" ? "continue" : "regenerate")
  if (!isContinue) return null
  const message = error instanceof Error ? error.message : typeof error === "string" ? error : ""
  return SAVED_HISTORY_REFUSAL.test(message) ? nativeRegenerationFeedback(error, t, "continue") : null
}
