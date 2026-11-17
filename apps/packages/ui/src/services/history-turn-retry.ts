import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { chatSubmitFailed, type ChatSubmitResult } from "@/hooks/chat/chat-action-utils"
import { canonicalHistoryJson, loadHistoryTurnRecoveries } from "@/db/dexie/history-selection"
import { getPromptById } from "@/db/dexie/helpers"
import { systemPromptForNonRagOption } from "@/services/tldw-server"
import { captureHistorySnapshot } from "@/services/chat-history-selection"
import { isHistoryTurnLive, type HistoryTurnRecoveryEntry } from "@/services/history-turn-keep"
import type { HistoryAdmissionReferenceV1 } from "@/types/history-selection"
import { appendSystemPromptSuffix } from "@/utils/output-formatting-guide"

export type HistoryRecoveryRetryContext = {
  model: string
  systemPrompts: string[]
  /** The review belongs to these current composer/model settings. */
  isCurrent: () => boolean
}

export const loadHistoryRecoveryRetryContext = async (
  source: { model: string; selectedSystemPrompt: string | null; systemPrompt?: string | null; systemPromptAppendix?: string | null },
  isCurrent: () => boolean
): Promise<HistoryRecoveryRetryContext> => {
  const [defaultPrompt, selectedPrompt] = await Promise.all([
    systemPromptForNonRagOption(), getPromptById(source.selectedSystemPrompt ?? "")
  ])
  if (!isCurrent()) throw new Error("retry_context_changed")
  if (source.selectedSystemPrompt && !selectedPrompt)
    throw new Error("retry_prompt_unavailable")
  // Match normalChatMode's default/selected/inline system instruction order.
  const systemPrompts: string[] = []
  const addPrompt = (content?: string | null) => {
    const resolved = appendSystemPromptSuffix(content || "", source.systemPromptAppendix)
    if (resolved) systemPrompts.push(resolved)
  }
  if (!selectedPrompt) addPrompt(defaultPrompt)
  if (source.systemPrompt?.trim()) addPrompt(source.systemPrompt)
  else if (selectedPrompt) {
    const content = selectedPrompt.system_prompt ?? selectedPrompt.content
    addPrompt(content)
  }
  return { model: source.model, systemPrompts, isCurrent }
}

/** Only client-managed no-output failures can reuse their saved admission here. */
export const canRetryHistoryTurnRecovery = (entry: HistoryTurnRecoveryEntry) => {
  const turn = entry.turn
  return turn.persistence !== "server" && turn.state === "accepted_unsent" &&
    !!turn.admission && turn.input_id === turn.admission.input_message_id &&
    !turn.result_text && !turn.outcome && !turn.settled_message_id &&
    !turn.existing_selection && !isHistoryTurnLive(turn.operation_id)
}

export type HistoryRecoveryRetrySubmit = {
  message: string
  image: string
  historyRetryAdmission: HistoryAdmissionReferenceV1
  historyRetryRecoveries: HistoryTurnRecoveryEntry[]
  historyRecoveryRetry: true
  historyRecoveryRetryIsCurrent: () => boolean
  historyRecoveryRetrySystemPrompts: string[]
}

const inFlight = new Set<string>()

/** Reuse the original accepted input; never turn a refused recovery into a new send. */
export const retryHistoryTurnRecovery = async (
  controller: HistorySelectionController | null | undefined,
  entry: HistoryTurnRecoveryEntry,
  review: { contextReviewed: boolean; context: HistoryRecoveryRetryContext },
  onSubmit: (submission: HistoryRecoveryRetrySubmit) => Promise<ChatSubmitResult | void>,
  getCurrentContext?: () => Promise<HistoryRecoveryRetryContext>
): Promise<ChatSubmitResult> => {
  const failed = (reason: string) => chatSubmitFailed(reason)
  const operation = canonicalHistoryJson([entry.scope, entry.turn.owner_key, entry.turn.conversation_id, entry.turn.operation_id])
  if (inFlight.has(operation)) return failed("history_retry_in_progress")
  inFlight.add(operation)
  try {
    if (!review.contextReviewed || !review.context.isCurrent()) return failed("retry_context_review_required")
    const reviewedSystemPrompts = [...review.context.systemPrompts]
    if (getCurrentContext) {
      const now = await getCurrentContext()
      if (!review.context.isCurrent() || canonicalHistoryJson([now.model, now.systemPrompts]) !==
          canonicalHistoryJson([review.context.model, reviewedSystemPrompts]))
        return failed("retry_context_changed")
    }
    if (!controller || !canRetryHistoryTurnRecovery(entry)) return failed("history_recovery_not_retryable")
    const isCurrent = controller.fence()
    const current = controller.getCurrent()
    const { owner, view, bookmarkScope: scope } = current
    if (current.status !== "ready" || !owner || owner.kind === "unavailable" || !view || !scope)
      return failed("history_selection_not_ready")
    const valid = () => isCurrent() && review.context.isCurrent() &&
      (owner.kind !== "native" || owner.validate_lease())
    if (owner.kind === "native" && (!current.settingsQualified || !owner.validate_lease()))
      return failed("request_config_scope_changed")
    if (entry.scope.profile_id !== scope.profile_id || entry.turn.owner_key !== view.owner_key ||
        entry.turn.conversation_id !== view.conversation_id)
      return failed("owner_conversation_mismatch")
    const fresh = (await loadHistoryTurnRecoveries(scope, view)).find(candidate =>
      candidate.turn.operation_id === entry.turn.operation_id &&
      candidate.scope.client_session_id === entry.scope.client_session_id)
    if (!valid()) return failed("history_selection_changed")
    if (!fresh || !canRetryHistoryTurnRecovery(fresh) ||
        canonicalHistoryJson(fresh.turn) !== canonicalHistoryJson(entry.turn))
      return failed("history_recovery_changed")
    const turn = fresh.turn
    const admission = turn.admission!
    if (admission.owner_key !== view.owner_key || admission.conversation_id !== view.conversation_id ||
        view.cursor.kind !== "after_message" || view.cursor.message_id !== admission.input_message_id)
      return failed("invalid_history_retry")
    const capture = await captureHistorySnapshot(
      owner, view, "send", undefined, undefined, owner.kind === "native" ? admission : undefined
    )
    if (!valid()) return failed("history_selection_changed")
    if (capture.status !== "captured") return failed(capture.code)
    const question = capture.rows.at(-1)
    const content = capture.selected_content.at(-1)
    if (!question || question.role !== "user" || question.id !== admission.input_message_id ||
        (owner.kind === "native"
          ? !capture.validated_admission ||
            canonicalHistoryJson(capture.validated_admission.admission) !== canonicalHistoryJson(admission) ||
            question.revision !== capture.validated_admission.input_revision
          : question.revision !== admission.input_message_revision) || !content ||
        content.message !== turn.input_text ||
        canonicalHistoryJson(content.images) !== canonicalHistoryJson(turn.input_images) ||
        content.images.length > 1 ||
        capture.snapshot.nodes.some(node => node.role === "assistant" && node.parent_id === question.id))
      return failed("history_retry_input_changed")
    const choosing = controller.choose({ kind: "before_message", message_id: question.id })
    const adoptedIsCurrent = controller.fence()
    const moved = await choosing
    const retryViewIsCurrent = () => adoptedIsCurrent() &&
      controller.getCurrent().owner === owner &&
      (owner.kind !== "native" || owner.validate_lease())
    const retryIsCurrent = () => retryViewIsCurrent() && review.context.isCurrent()
    if (!moved) return failed("history_selection_changed")
    if (!retryIsCurrent()) {
      if (retryViewIsCurrent()) await controller.choose({ kind: "after_message", message_id: question.id })
      return failed("history_selection_changed")
    }
    let result: ChatSubmitResult
    try {
      const submitted = await onSubmit({
        message: content.message,
        image: content.images[0] ?? "",
        historyRetryAdmission: structuredClone(admission),
        historyRetryRecoveries: [structuredClone(fresh)],
        historyRecoveryRetry: true,
        historyRecoveryRetryIsCurrent: retryIsCurrent,
        historyRecoveryRetrySystemPrompts: reviewedSystemPrompts
      })
      result = typeof submitted === "undefined" ? failed("history_recovery_retry_not_submitted") : submitted
    } catch (error) {
      result = failed(error instanceof Error ? error.message : "history_recovery_retry_failed")
    }
    // A failed retry keeps the saved question selected, so it remains visible
    // and a later deliberate retry can use the same admission. Never move a new view.
    if (result.status !== "submitted" && retryViewIsCurrent())
      await controller.choose({ kind: "after_message", message_id: question.id })
    return result
  } catch (error) {
    return failed(error instanceof Error ? error.message : "history_recovery_retry_failed")
  } finally {
    inFlight.delete(operation)
  }
}


export type SavedQuestionReview = {
  id: string
  revision: string
  text: string
  model: string
  isCurrent: () => boolean
}

/** A new explicit answer intent; never reinterpret or dismiss an old unknown receipt. */
export const reviewServerSavedQuestion = async (
  controller: HistorySelectionController | null | undefined,
  context: HistoryRecoveryRetryContext
): Promise<SavedQuestionReview> => {
  if (!controller) throw new Error("history_selection_not_ready")
  const current = controller.getCurrent()
  const {owner, view} = current
  if (current.status !== "ready" || !current.settingsQualified || owner?.kind !== "native" ||
      !view || view.cursor.kind !== "after_message") throw new Error("history_selection_not_ready")
  const fence = controller.fence()
  const isCurrent = () => fence() && context.isCurrent() && owner.validate_lease()
  if (!isCurrent()) throw new Error("history_selection_changed")
  const captured = await captureHistorySnapshot(owner, view, "send", undefined, "tldw_history_server_existing_input_v1")
  if (!isCurrent()) throw new Error("history_selection_changed")
  if (captured.status !== "captured") throw new Error(captured.code)
  const question = captured.rows.at(-1)
  const content = captured.selected_content.at(-1)
  if (!question || question.role !== "user" || question.id !== view.cursor.message_id ||
      !content || content.id !== question.id || content.images.length || !content.message.trim() ||
      captured.snapshot.nodes.some(node => node.role === "assistant" && node.parent_id === question.id))
    throw new Error("history_retry_input_changed")
  return {id: question.id, revision: question.revision, text: content.message, model: context.model, isCurrent}
}
