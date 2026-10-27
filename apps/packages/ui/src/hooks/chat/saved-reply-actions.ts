/**
 * Which reply actions can run on a selected saved chat (CM-01, #3106).
 *
 * Under D7 a saved chat's turns are settled by the server, so Regenerate and
 * Continue on a saved reply run as server-owned operations on the selected
 * question and reply. This module answers, from what the client already
 * knows, whether those operations can start, so the Playground can disable or
 * hide an action instead of showing it and refusing it on click.
 *
 * The server stays the authority: a reply that looks eligible here can still
 * be refused (for example a question saved before server admissions existed),
 * and that refusal is explained in plain words by the caller.
 */
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { Message } from "@/store/option"
import type { NativeHistoryRegenerate } from "@/types/chat-modes"
import type {
  HistoryNodeV1,
  HistorySelectedContentV1,
  HistorySelectionCaptureV1,
  HistoryViewSelectionV1
} from "@/types/history-selection"

type SelectionState = ReturnType<HistorySelectionController["getCurrent"]>

/** The saved question and reply that native Regenerate or Continue would act on. */
export type NativeSavedReplyTarget = {
  origin: SelectionState & {
    owner: NonNullable<SelectionState["owner"]>
    view: HistoryViewSelectionV1
    capture: HistorySelectionCaptureV1
  }
  capture: HistorySelectionCaptureV1
  input: HistoryNodeV1
  reply: HistoryNodeV1
  inputContent: HistorySelectedContentV1
  replyContent: HistorySelectedContentV1
}

const rowId = (row?: Message) => row?.serverMessageId ?? row?.id

const sameImages = (row: Message, content: HistorySelectedContentV1) =>
  JSON.stringify(row.images ?? []) === JSON.stringify(content.images ?? [])

/**
 * Resolve the last saved question and reply of a ready server-owned chat, or
 * `null` when the selection is not a ready server chat or the visible thread no
 * longer shows exactly that saved pair.
 */
export const nativeSavedReplyTarget = (
  historySelection: HistorySelectionController | null | undefined,
  visible: readonly Message[]
): NativeSavedReplyTarget | null => {
  const origin = historySelection?.getCurrent()
  if (
    origin?.status !== "ready" ||
    origin.owner?.kind !== "native" ||
    !origin.view ||
    origin.capture?.status !== "captured"
  )
    return null
  const capture = origin.capture
  if (
    origin.view.interpretation.kind !== "parent_graph_v1" ||
    capture.view.interpretation.kind !== "parent_graph_v1"
  )
    return null
  const reply = capture.rows.at(-1)
  const input = capture.rows.at(-2)
  const replyContent = capture.selected_content.at(-1)
  const inputContent = capture.selected_content.at(-2)
  if (
    !reply ||
    !input ||
    !replyContent ||
    !inputContent ||
    reply.role !== "assistant" ||
    input.role !== "user" ||
    !reply.settled ||
    !input.settled ||
    reply.parent_id !== input.id ||
    reply.comparison ||
    input.comparison ||
    replyContent.id !== reply.id ||
    inputContent.id !== input.id
  )
    return null
  const target = visible.at(-1)
  const question = visible.at(-2)
  if (
    !target?.isBot ||
    !question ||
    question.isBot ||
    rowId(target) !== reply.id ||
    rowId(question) !== input.id ||
    visible.filter((row) => rowId(row) === reply.id).length !== 1 ||
    visible.filter((row) => rowId(row) === input.id).length !== 1 ||
    target.message !== replyContent.message ||
    question.message !== inputContent.message ||
    !sameImages(question, inputContent)
  )
    return null
  return {
    origin: origin as NativeSavedReplyTarget["origin"],
    capture,
    input,
    reply,
    inputContent,
    replyContent
  }
}

/** A continuation extends plain saved text; tool calls and images have none. */
export const isContinuableReply = (content: HistorySelectedContentV1): boolean => {
  const extra = content.extra_metadata ?? {}
  return (
    content.message.trim().length > 0 &&
    !content.images.length &&
    content.tool_calls == null &&
    extra.function_call == null &&
    extra.tool_call_id == null &&
    extra.content_placeholder_reason == null
  )
}

type SavedReplyAction = "regenerate" | "continue"

/**
 * Saved replies the server refused to regenerate or continue in this session,
 * by owner, conversation, reply and revision. A question saved before server
 * admissions existed cannot be told apart up front; after one definitive
 * refusal its actions say so instead of failing again.
 */
const refusedSavedReplies = new Map<string, Set<SavedReplyAction>>()

const savedReplyKey = (reply: {
  owner_key: string
  conversation_id: string
  message_id: string
  message_revision: string
}) =>
  JSON.stringify([reply.owner_key, reply.conversation_id, reply.message_id, reply.message_revision])

/**
 * True for the server's definitive refusal of a saved-reply Regenerate or
 * Continue. The server sends it before admitting anything or calling the
 * model, so nothing was saved and nothing needs recovering.
 */
export const isSavedReplyActionRefusal = (error: unknown): boolean => {
  const response = error as { status?: unknown; details?: { detail?: { status?: unknown; code?: unknown } } } | null
  const detail = response?.details?.detail
  return (
    response?.status === 409 &&
    detail?.status === "unsupported_history_capability" &&
    (detail.code === "unsupported_history_regeneration" || detail.code === "unsupported_history_continuation")
  )
}

/** Remember a definitive refusal; a refused Regenerate also rules out Continue. */
export const rememberSavedReplyRefusal = (intent: NativeHistoryRegenerate) => {
  const key = savedReplyKey(intent)
  const refused = refusedSavedReplies.get(key) ?? new Set<SavedReplyAction>()
  refused.add("continue")
  if (intent.action !== "continue") refused.add("regenerate")
  refusedSavedReplies.set(key, refused)
}

/** Test seam: forget remembered refusals. */
export const resetSavedReplyRefusals = () => refusedSavedReplies.clear()

const refusedActions = (target: NativeSavedReplyTarget) =>
  refusedSavedReplies.get(
    savedReplyKey({
      owner_key: target.origin.view.owner_key,
      conversation_id: target.origin.view.conversation_id,
      message_id: target.reply.id,
      message_revision: target.reply.revision
    })
  )

/** Why a reply action cannot start right now. */
export type SavedReplyActionBlock = "loading" | "unsupported"

export type SavedReplyActionAvailability = {
  regenerate: SavedReplyActionBlock | null
  continue: SavedReplyActionBlock | null
}

const AVAILABLE: SavedReplyActionAvailability = Object.freeze({
  regenerate: null,
  continue: null
})

/**
 * A reply that stopped early in a saved chat is retried from the question
 * before it (CS-04); that path does not need the reply to be saved.
 */
/**
 * Whether a device history (local owner) can regenerate its last answer. The
 * send path regenerates it through its own boundary, as long as the captured
 * history ends with an answer to the question before it (any order for a
 * legacy linear history). Native histories need a saved-reply target instead.
 */
// Mirrors the local regenerate path in handlers/messageHandlers.ts, which
// resends the question before the answer with at most one image.
const localHistoryRegenerateReady = (state: SelectionState): boolean => {
  if (state?.status !== "ready" || state.owner?.kind !== "local" || !state.view ||
      state.capture?.status !== "captured")
    return false
  const answer = state.capture.rows.at(-1)
  const question = state.capture.rows.at(-2)
  const content = state.capture.selected_content.at(-2)
  return (
    answer?.role === "assistant" &&
    question?.role === "user" &&
    (state.view.interpretation.kind === "legacy_linear_v1" || answer.parent_id === question.id) &&
    Boolean(content) &&
    content!.images.length <= 1
  )
}

const isRetryableInterruptedReply = (visible: readonly Message[]) => {
  const reply = visible.at(-1)
  const question = visible.at(-2)
  const info = reply?.generationInfo as Record<string, unknown> | undefined
  return Boolean(
    reply?.isBot && info?.interrupted && question && !question.isBot && question.id
  )
}

/**
 * Which of Regenerate and Continue can start on the last reply.
 *
 * A chat without a selected saved history (a new or local chat, or a
 * temporary chat that has no owner) keeps its ordinary actions.
 */
export const savedReplyActionAvailability = ({
  historySelection,
  messages,
  temporaryChat = false
}: {
  historySelection: HistorySelectionController | null | undefined
  messages: readonly Message[]
  temporaryChat?: boolean
}): SavedReplyActionAvailability => {
  const state = historySelection?.getCurrent()
  if (!state || state.status === "idle" || (temporaryChat && !state.owner))
    return AVAILABLE
  if (state.status === "loading")
    return { regenerate: "loading", continue: "loading" }
  const target = nativeSavedReplyTarget(historySelection, messages)
  const refused = target ? refusedActions(target) : undefined
  return {
    regenerate:
      refused?.has("regenerate")
        ? "unsupported"
        : target || localHistoryRegenerateReady(state) || isRetryableInterruptedReply(messages)
          ? null
          : "unsupported",
    continue:
      target && !refused?.has("continue") && isContinuableReply(target.replyContent)
        ? null
        : "unsupported"
  }
}

/**
 * Whether the last reply's recorded status (Output limit reached, Interrupted)
 * may offer its Continue and Retry. Those actions run on the server's saved
 * reply (D7), so they are offered only when the selection shows that reply as
 * the last saved reply of a server chat; the recorded status grants nothing by
 * itself (CM-03).
 */
export const savedReplyStatusOffersActions = (
  historySelection: HistorySelectionController | null | undefined,
  messages: readonly Message[]
): boolean => nativeSavedReplyTarget(historySelection, messages) !== null

/** Plain explanation shown on a disabled Regenerate control. */
export const savedReplyRegenerateUnavailableReason = (
  block: SavedReplyActionBlock | null,
  t: (key: string, fallback: string) => string
): string | null => {
  if (block === "loading")
    return t(
      "playground:savedReplyActions.loading",
      "This saved chat is still loading. Try again in a moment."
    )
  if (block === "unsupported")
    return t(
      "playground:savedReplyActions.regenerateUnavailable",
      "This saved reply can't be regenerated here. Use New Branch to try again from this point."
    )
  return null
}
