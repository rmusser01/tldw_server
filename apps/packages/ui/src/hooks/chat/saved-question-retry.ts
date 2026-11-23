/**
 * Retry of a server-saved question that has no reply (B02; D7 P5, Wave 8 lane 8B, #3101).
 *
 * A turn can end with its question saved on the server and no reply: a Stop
 * before the first token, or a failure after the question was admitted.
 * "Review saved question" then answers that question through
 * `tldw_history_server_existing_input_v1`, so it is never saved twice and the
 * server settles the reply with its model and status. The rules that need no
 * store live in `@/utils/saved-question-retry`.
 *
 * These live here so the frozen `useChatActions.ts` and `useMessage.tsx` keep
 * only routing lines.
 */
import { canonicalHistoryJson } from "@/db/dexie/history-selection"
import { useStoreMessageOption } from "@/store/option"
import {
  INSTRUCTIONS_UNSUPPORTED,
  SAVED_QUESTION_RETRY_IN_PROGRESS,
  savedQuestionChatKindOf,
  type SavedQuestionChatKind,
  type SavedQuestionInstructions
} from "@/utils/saved-question-retry"
import { chatSubmitFailed, type ChatSubmitResult } from "./chat-action-utils"

/** The kind of the open server chat `conversationId`, or null when another chat is open. */
export const savedQuestionChatKind = (conversationId: string): SavedQuestionChatKind | null =>
  savedQuestionChatKindOf(useStoreMessageOption.getState(), conversationId)

type ReviewedSavedQuestion = SavedQuestionInstructions & {
  id: string
  text: string
  instructionsSupported?: boolean
  isCurrent: () => boolean
}

// One Retry per saved question at a time: a second click must not answer it twice.
const inFlight = new Set<string>()

/**
 * Submit a reviewed saved question. A plain review is refused while its
 * instructions can't be carried, or once they differ from the current ones
 * (for example, an edited Prompt or default instruction).
 */
export const submitSavedQuestionRetry = async <T extends ReviewedSavedQuestion>(
  review: T,
  getCurrentContext: () => Promise<{ systemPrompts: readonly string[] }>,
  submit: (submission: {
    message: string
    image: string
    historyServerExistingInput: T
  }) => Promise<ChatSubmitResult | void | undefined>
): Promise<ChatSubmitResult> => {
  if (inFlight.has(review.id)) return chatSubmitFailed(SAVED_QUESTION_RETRY_IN_PROGRESS)
  inFlight.add(review.id)
  try {
    if (!review.isCurrent()) return chatSubmitFailed("history_selection_changed")
    if (review.instructions) {
      if (review.instructions.length && review.instructionsSupported !== true)
        return chatSubmitFailed(INSTRUCTIONS_UNSUPPORTED)
      let current: readonly string[]
      try {
        current = (await getCurrentContext()).systemPrompts
      } catch (error) {
        return chatSubmitFailed(error instanceof Error ? error.message : "retry_context_changed")
      }
      if (!review.isCurrent() || canonicalHistoryJson(current) !== canonicalHistoryJson(review.instructions))
        return chatSubmitFailed("retry_context_changed")
    }
    // A submit that returns nothing did not report a send.
    const submitted = (await submit({ message: review.text, image: "", historyServerExistingInput: review })) as
      ChatSubmitResult | undefined
    return submitted ?? chatSubmitFailed("native_history_existing_input_not_submitted")
  } finally {
    inFlight.delete(review.id)
  }
}
