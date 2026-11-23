/**
 * Rules for retrying a server-saved question that has no reply (B02; D7 P5,
 * Wave 8 lane 8B, #3101). This is a leaf module: it reads no store and no
 * local database, so services and the send path can share it.
 *
 * A character chat answers the saved question under its saved character
 * context. A plain chat carries its active System/Prompt instructions as
 * system-only messages (owner decision 22), only on a server that settles
 * turns itself and advertises
 * `native_completion_v1.existing_input_system_messages_supported`.
 *
 * A plain chat whose selected history has saved system rows, error replies or
 * reasoning the server can't keep stays client-settled, like an ordinary send
 * (`normalChatMode`): the server would build a different model context.
 */
import { removeReasoning } from "@/libs/reasoning"
import type { HistoryCaptureResultV1, HistorySelectionCaptureV1 } from "@/types/history-selection"
import { decodeChatErrorPayload } from "@/utils/chat-error-message"

/** The server can't carry the active instructions on a Retry. */
export const INSTRUCTIONS_UNSUPPORTED = "native_history_existing_input_instructions_unsupported"
/** The plain chat's selected history must stay client-settled (system, error or reasoning rows). */
export const SAVED_QUESTION_CLIENT_MANAGED = "native_history_existing_input_client_managed"
/** The same saved question is already being retried. */
export const SAVED_QUESTION_RETRY_IN_PROGRESS = "history_retry_in_progress"

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

export type SavedQuestionChatKind = "plain" | "character" | "persona"

/**
 * The kind of the open server chat `conversationId`, from the chat store's
 * fields, or null when another chat is open.
 */
export const savedQuestionChatKindOf = (
  state: {
    serverChatId?: string | null
    serverChatAssistantKind?: string | null
    serverChatCharacterId?: string | number | null
  },
  conversationId: string
): SavedQuestionChatKind | null => {
  if (state.serverChatId !== conversationId) return null
  if (state.serverChatAssistantKind === "persona") return "persona"
  return state.serverChatAssistantKind === "character" ||
    (state.serverChatAssistantKind == null && state.serverChatCharacterId != null)
    ? "character"
    : "plain"
}

/**
 * Whether a plain chat's saved question must stay client-settled: the server
 * doesn't settle turns for this owner (Sync v2, older servers), or the
 * selected history has rows an ordinary send also keeps client-settled.
 */
export const plainSavedQuestionClientManaged = (capture: HistoryCaptureResultV1 | null | undefined): boolean => {
  if (capture?.status !== "captured" || capture.native_completion_v1?.supported !== true) return true
  const reasoningKept = capture.native_completion_v1.reasoning_history_supported === true
  const text = new Map(capture.selected_content.map((row) => [row.id, row.message]))
  return capture.rows.some((row) => {
    if (row.role === "system") return true
    if (row.role !== "assistant") return false
    const content = text.get(row.id) ?? ""
    return Boolean(decodeChatErrorPayload(content)) || (!reasoningKept && removeReasoning(content) !== content)
  })
}

/** Whether to offer "Review saved question" for this chat kind and view capture. */
export const savedQuestionReviewOffered = (
  kind: SavedQuestionChatKind | null,
  capture: HistoryCaptureResultV1 | null | undefined
): boolean => kind === "character" || (kind === "plain" && !plainSavedQuestionClientManaged(capture))

/**
 * The current messages an existing-input completion sends, checked against the
 * capture taken just before dispatch. A character Retry sends none. A plain
 * Retry is refused when it must stay client-settled, and sends instructions
 * only to a server that accepts them.
 */
export const existingInputCurrentMessages = (
  capture: HistorySelectionCaptureV1,
  input: SavedQuestionInstructions
): Array<{ role: "system"; content: string }> => {
  if (!input.instructions) return []
  if (capture.native_completion_v1?.supported !== true) throw new Error("unsupported_history_capability")
  if (plainSavedQuestionClientManaged(capture)) throw new Error(SAVED_QUESTION_CLIENT_MANAGED)
  if (input.instructions.length && capture.native_completion_v1.existing_input_system_messages_supported !== true)
    throw new Error(INSTRUCTIONS_UNSUPPORTED)
  return input.instructions.map((content) => ({ role: "system" as const, content }))
}

/**
 * A plain Retry refused before dispatch by the native send's settings checks.
 * Its message is the original code; the copy for it is plain-chat wording.
 */
export class PlainSavedQuestionRefusal extends Error {
  constructor(code: string, options?: { cause?: unknown }) {
    super(code, options)
    this.name = "PlainSavedQuestionRefusal"
  }
}

type Translate = (key: string, fallback: string, options?: { setting: string }) => string

/** Where the review is shown: the full page's instructions also carry the output formatting guide. */
export type SavedQuestionSurface = "playground" | "sidepanel"

const PLAIN_REFUSAL_COPY: Record<string, string> = {
  [INSTRUCTIONS_UNSUPPORTED]:
    "This server can't keep System/Prompt instructions on a retry. To retry here, clear this chat's System prompt and Prompt, and the default system prompt and output formatting guide in Settings, then review again. Or update the server. Your question is kept.",
  [SAVED_QUESTION_CLIENT_MANAGED]:
    "This chat's history can't be answered by the server. Your question is kept; use Retry on the stopped or failed reply instead.",
  [SAVED_QUESTION_RETRY_IN_PROGRESS]: "This saved question is already being retried.",
  retry_context_changed:
    "The System/Prompt instructions changed after your review. Review the saved question again to see what the retry will use.",
  retry_prompt_unavailable:
    "The selected Prompt could not be read. Choose it again or clear it, then review the saved question again.",
  history_retry_input_changed:
    "The saved question changed or now has a reply. Load the latest messages; nothing was sent.",
  native_history_existing_input_changed:
    "The saved question changed or now has a reply. Load the latest messages; nothing was sent.",
  native_history_client_tools_unsupported:
    "A saved question can't be retried with client tools on. Set tools to None and retry; your question is kept.",
  native_history_reasoning_effort_unsupported:
    "Choose Low, Medium or High reasoning effort to retry this saved question; your question is kept.",
  native_history_extra_parameters_unsupported:
    "A saved question can't be retried with extra request parameters. Clear them in the model settings and retry; your question is kept.",
  native_history_prompt_template_unsupported:
    "A saved question can't be retried with a behavior template. Clear the template and retry; your question is kept.",
  native_history_actor_context_unsupported:
    "A saved question can't be retried with scene overrides on. Turn them off and retry; your question is kept.",
  native_history_explicit_model_provider_required:
    "Choose a specific model and provider (not Auto) to retry this saved question; your question is kept.",
  native_history_explicit_context_unsupported:
    "A saved question can't be retried with web search, research, files or other added context. Turn them off and retry; your question is kept.",
  native_history_assets_unsupported:
    "A saved question with an image can't be retried this way. Your question is kept; use Retry on the stopped or failed reply instead.",
  temporary_history_unavailable:
    "A temporary chat has no saved question to retry. Your question is kept.",
  unsupported_history_action_context:
    "This saved question can't be retried with the chat's current setup, such as a selected assistant or character, a reply target or added context. Clear them, then review it again; your question is kept."
}

/** The side panel sends no output formatting guide, so its copy doesn't mention one. */
const SIDEPANEL_INSTRUCTIONS_UNSUPPORTED =
  "This server can't keep System/Prompt instructions on a retry. To retry here, clear this chat's System prompt and Prompt, and the default system prompt in Settings, then review again. Or update the server. Your question is kept."

/** Plain-chat copy for a refused saved-question Retry, or null for an unknown reason. */
export const plainSavedQuestionRefusalMessage = (
  code: string,
  t: Translate,
  surface: SavedQuestionSurface = "playground"
): string | null => {
  const prefix = "native_history_setting_unsupported:"
  if (code.startsWith(prefix)) {
    const key = code.slice(prefix.length)
    // The model settings' own label where it has one ("Top K"), else the key.
    const setting = t(`common:modelSettings.form.${key}.label`, key)
    // Fill the placeholder here too, whatever the translator does with options.
    return t(
      "playground:historySelection.savedQuestionSettingUnsupported",
      "The {{setting}} setting can't be used when retrying a saved question. Clear it in the model settings and retry; your question is kept.",
      { setting }
    ).replaceAll("{{setting}}", setting)
  }
  if (code === INSTRUCTIONS_UNSUPPORTED && surface === "sidepanel")
    return t("playground:historySelection.savedQuestionInstructionsUnsupportedSidepanel", SIDEPANEL_INSTRUCTIONS_UNSUPPORTED)
  return PLAIN_REFUSAL_COPY[code]
    ? t(`playground:historySelection.savedQuestionRefusal.${code}`, PLAIN_REFUSAL_COPY[code])
    : null
}
