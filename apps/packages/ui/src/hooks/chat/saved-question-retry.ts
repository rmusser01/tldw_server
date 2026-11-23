/**
 * Retry of a server-saved question that has no reply (B02; D7 P5, Wave 8 lane 8B, #3101).
 *
 * A turn can end with its question saved on the server and no reply: a Stop
 * before the first token, or a provider error after the question was admitted.
 * "Review saved question" then answers that question through
 * `tldw_history_server_existing_input_v1`, so it is never saved twice and the
 * server settles the reply with its model and status.
 *
 * A character chat answers under its saved character context. A plain chat
 * carries its active System/Prompt instructions as system-only messages (owner
 * decision 22). A server that doesn't advertise
 * `native_completion_v1.existing_input_system_messages_supported` can't take
 * them: while an instruction is active, the Retry is refused before anything is
 * sent and the question is kept.
 *
 * These rules live here so the frozen `useChatActions.ts` and `useMessage.tsx`
 * keep only routing lines.
 */
import { canonicalHistoryJson } from "@/db/dexie/history-selection"
import { useStoreMessageOption } from "@/store/option"
import type { HistorySelectionCaptureV1 } from "@/types/history-selection"
import { chatSubmitFailed, type ChatSubmitResult } from "./chat-action-utils"

/** Refusal code: the server can't carry the active instructions on a Retry. */
export const INSTRUCTIONS_UNSUPPORTED = "native_history_existing_input_instructions_unsupported"

/** What a saved-question review records about a plain chat's instructions. */
export type SavedQuestionInstructions = {
  /**
   * Plain chats only: the reviewed System/Prompt instructions, in the order a
   * normal send uses them. Absent for a character chat.
   */
  readonly instructions?: readonly string[]
}

/** A plain-chat review carries its instructions (possibly none); a character review doesn't. */
export const isPlainSavedQuestion = (input?: SavedQuestionInstructions | null): boolean =>
  Array.isArray(input?.instructions)

/**
 * The kind of the open server chat `conversationId`, or null when another chat
 * is open. Persona chats have no saved-question Retry.
 */
export const savedQuestionChatKind = (
  conversationId: string
): "plain" | "character" | "persona" | null => {
  const state = useStoreMessageOption.getState()
  if (state.serverChatId !== conversationId) return null
  if (state.serverChatAssistantKind === "persona") return "persona"
  return state.serverChatAssistantKind === "character" ||
    (state.serverChatAssistantKind == null && state.serverChatCharacterId != null)
    ? "character"
    : "plain"
}

/**
 * The current messages an existing-input completion sends, checked against the
 * capture taken just before dispatch. A character Retry sends none. A plain
 * Retry needs a server that settles turns itself (Sync v2 owners and older
 * servers stay client-managed), and sends instructions only to a server that
 * accepts them.
 */
export const existingInputCurrentMessages = (
  capture: HistorySelectionCaptureV1,
  input: SavedQuestionInstructions
): Array<{ role: "system"; content: string }> => {
  if (!input.instructions) return []
  const native = capture.native_completion_v1
  if (native?.supported !== true) throw new Error("unsupported_history_capability")
  if (input.instructions.length && native.existing_input_system_messages_supported !== true)
    throw new Error(INSTRUCTIONS_UNSUPPORTED)
  return input.instructions.map((content) => ({ role: "system" as const, content }))
}

type ReviewedSavedQuestion = SavedQuestionInstructions & {
  text: string
  instructionsSupported?: boolean
  isCurrent: () => boolean
}

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
}
