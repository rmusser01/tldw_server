/**
 * Keep interrupted and orphaned chat turns in the transcript (CS-04, #3104).
 *
 * A selected-history send writes the user's question to its owner before the
 * reply streams. If the reply then ends early (Stop, dropped connection,
 * reload) or finishes after its view went away (the user left /chat), the turn
 * is recorded as a retained recovery with an `outcome`. This module puts such
 * a turn back into the selected history instead of parking it for review:
 * - the reply is written to the owner (marked interrupted or stopped for a
 *   local chat), and the view follows it so the question and answer show;
 * - a server chat cannot carry the interrupted marker, so a partial server
 *   reply stays in the retained record on this device and only the question
 *   is followed. Saving partial replies on the server belongs to the
 *   resumable-generation design (D7).
 * - a reply that never started because the user stopped it, or because the
 *   server answered with an error (CM-02, CC-02, #3107), is kept the same way:
 *   the question is followed and the record is shown after it as a stopped
 *   reply or an error reply with Retry, instead of waiting in review.
 *
 * It only uses the controller's public API and the owner write that a normal
 * turn uses, so ownership and lease checks stay where they are.
 */
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { followStoppedChatTurnSelection } from "@/hooks/chat/abort-turn-cleanup"
import {
  canonicalHistoryJson,
  dismissHistoryTurnRecovery,
  loadHistoryTurnRecoveries
} from "@/db/dexie/history-selection"
import type {
  HistoryBookmarkScope,
  HistoryTurnOutcome,
  HistoryTurnRecovery,
  Message as StoredMessage
} from "@/db/dexie/types"
import type { Message } from "@/store/option"
import { settleAcceptedAssistant } from "@/services/chat-history-selection"
import { buildSendFailureContent } from "@/utils/chat-error-message"
import { beginServerChatWrite } from "@/store/server-chat-save-status"
import type { HistoryViewSelectionV1 } from "@/types/history-selection"

export type HistoryTurnRecoveryEntry = {
  scope: HistoryBookmarkScope
  turn: HistoryTurnRecovery
}

export type HistoryTurnKeepResult =
  | {
      status: "kept"
      /** The message the view now ends at: the reply, or the question. */
      resultId: string
      followed: boolean
      /** A partial server reply kept only in the retained record. */
      retained: boolean
    }
  | { status: "skipped"; reason: string }

/** The reason shown on a reply the user did not stop. */
export const DEFAULT_INTERRUPTION_REASON =
  "The reply was interrupted before it finished."
export const STOPPED_INTERRUPTION_REASON = "Stopped"

// ---------------------------------------------------------------------------
// Turns still running in this page
// ---------------------------------------------------------------------------

const liveTurns = new Set<string>()
const releaseListeners = new Set<() => void>()

/** A running turn owns its record; nothing else may keep it until it ends. */
export const markHistoryTurnLive = (operationId: string) => {
  liveTurns.add(operationId)
}

export const releaseHistoryTurn = (operationId: string) => {
  if (!liveTurns.delete(operationId)) return
  for (const listener of [...releaseListeners]) listener()
}

export const isHistoryTurnLive = (operationId: string) =>
  liveTurns.has(operationId)

/** Called after any running turn ends, e.g. one that finished off-screen. */
export const onHistoryTurnReleased = (listener: () => void) => {
  releaseListeners.add(listener)
  return () => {
    releaseListeners.delete(listener)
  }
}

/**
 * Forget the turns of this page. A reload starts with none; tests that model
 * a reload in one process call this, since requests the old page left in
 * flight never end there.
 */
export const resetHistoryTurnRegistry = () => {
  liveTurns.clear()
}

// ---------------------------------------------------------------------------
// Keeping a retained turn
// ---------------------------------------------------------------------------

/** The marker a kept reply carries; a finished reply carries none. */
export const interruptionGenerationInfo = (
  outcome: HistoryTurnOutcome | undefined,
  reason?: string
): Record<string, unknown> | undefined => {
  if (!outcome || outcome === "complete") return undefined
  const stopped = outcome === "stopped"
  return {
    interrupted: true,
    ...(stopped ? { stopped: true } : {}),
    interruptionReason:
      reason?.trim() ||
      (stopped ? STOPPED_INTERRUPTION_REASON : DEFAULT_INTERRUPTION_REASON)
  }
}

/**
 * A client-only reply bubble for a reply kept on this device: a partial
 * server reply, a reply stopped before it started, or the error the server
 * answered with (shown as an error reply with Retry and Switch model).
 */
export const retainedReplyMessage = (turn: HistoryTurnRecovery): Message => ({
  isBot: true,
  role: "assistant",
  id: turn.assistant_id,
  name: turn.model_name || "Assistant",
  modelName: turn.model_name,
  message:
    turn.result_text.trim() || !turn.failure
      ? turn.result_text
      : buildSendFailureContent(turn.failure),
  sources: [],
  createdAt: turn.created_at,
  parentMessageId: turn.admission?.input_message_id ?? turn.input_id ?? null,
  generationInfo: interruptionGenerationInfo(
    turn.outcome,
    turn.interruption_reason
  )
})

/**
 * Kept outcomes that remain as a retained record after keeping, shown as a
 * client-only reply after their question:
 * - a partial reply of a server chat, which the server does not hold;
 * - a reply the user stopped before it started, or one the server refused
 *   with an error (CM-02, CC-02). A server-settled turn qualifies once the
 *   server confirmed its question.
 */
export const isRetainedPartialReply = (
  turn: HistoryTurnRecovery,
  ownerKind: "local" | "native" | undefined
) => {
  if (
    !ownerKind ||
    !turn.outcome ||
    turn.outcome === "complete" ||
    turn.settled_message_id
  )
    return false
  const partial = turn.result_text.trim().length > 0
  const settled = turn.outcome === "stopped" || Boolean(turn.failure)
  if (turn.persistence === "server")
    return ownerKind === "native" && Boolean(turn.admission) && settled
  // A local chat stores a partial reply with its marker instead.
  return ownerKind === "native" ? partial || settled : !partial && settled
}

const sameBoundary = (
  current: HistoryViewSelectionV1 | null,
  origin: HistoryViewSelectionV1
) =>
  !!current &&
  canonicalHistoryJson(current.cursor) === canonicalHistoryJson(origin.cursor) &&
  canonicalHistoryJson(current.interpretation) ===
    canonicalHistoryJson(origin.interpretation)

const errorCode = (error: unknown) =>
  error instanceof Error ? error.message : String(error)

const inFlight = new Map<string, Promise<HistoryTurnKeepResult>>()

const keep = async (
  controller: HistorySelectionController,
  entry: HistoryTurnRecoveryEntry,
  stopSignal?: AbortSignal
): Promise<HistoryTurnKeepResult> => {
  const skipped = (reason: string): HistoryTurnKeepResult => ({
    status: "skipped",
    reason
  })
  const isCurrent = controller.fence()
  const current = controller.getCurrent()
  const owner = current.owner
  const view = current.view
  const scope = current.bookmarkScope
  if (
    current.status !== "ready" ||
    !owner ||
    owner.kind === "unavailable" ||
    !view ||
    !scope
  )
    return skipped("not_ready")
  if (
    owner.kind === "native" &&
    (!current.settingsQualified || !owner.validate_lease())
  )
    return skipped("not_ready")
  const listed = entry.turn
  if (
    listed.action !== undefined ||
    !listed.outcome ||
    !listed.admission ||
    listed.owner_key !== view.owner_key ||
    listed.conversation_id !== view.conversation_id ||
    entry.scope.profile_id !== scope.profile_id
  )
    return skipped("not_keepable")
  if (isHistoryTurnLive(listed.operation_id)) return skipped("live")
  // The record may have been completed or dismissed since it was listed.
  const fresh = (await loadHistoryTurnRecoveries(scope, view)).find(
    (candidate) =>
      candidate.turn.operation_id === listed.operation_id &&
      candidate.scope.client_session_id === entry.scope.client_session_id
  );
  if (!isCurrent()) return skipped('view_changed');
  if (!fresh?.turn.outcome || !fresh.turn.admission) return skipped('gone');
  if (fresh.turn.action !== undefined) return skipped('explicit_action_requires_review');
  const turn = fresh.turn;
  const retained = isRetainedPartialReply(turn, owner.kind);
  if (turn.persistence === 'server' && !turn.settled_message_id && !retained)
    return skipped('server_outcome_unknown');
  const admission = turn.admission!;
  const native = owner.kind === 'native';
  const content = turn.result_text;
  let resultId = admission.input_message_id;
  if (turn.settled_message_id) {
    resultId = turn.settled_message_id
  } else if (content.trim() && !retained) {
    const generationInfo = native
      ? undefined
      : interruptionGenerationInfo(turn.outcome, turn.interruption_reason)
    const reply: StoredMessage = {
      id: turn.assistant_id,
      history_id: owner.conversation_id,
      role: "assistant",
      name: turn.model_name || "Assistant",
      content,
      createdAt: turn.created_at,
      images: [],
      parent_message_id: admission.input_message_id,
      ...(generationInfo ? { generationInfo } : {}),
      ...(!native && turn.model_id ? { modelId: turn.model_id } : {})
    }
    try {
      await settleAcceptedAssistant(owner, admission, reply)
    } catch (error) {
      // An earlier attempt already wrote this reply (e.g. the page closed
      // before its record was cleared): follow it rather than write again.
      if (errorCode(error) !== "message_id_conflict")
        return skipped(errorCode(error))
    }
    resultId = turn.assistant_id
  }
  if (retained) {
    // The server holds the question but not this partial reply (CS-04): the
    // chat is not fully saved there, whatever the last acknowledged write
    // was. A reply that never started leaves nothing unsaved.
    if (native && turn.persistence !== 'server' && content.trim())
      beginServerChatWrite(owner.conversation_id)('failed');
  } else if (turn.persistence !== 'server') {
    await dismissHistoryTurnRecovery(
      fresh.scope,
      { owner_key: view.owner_key, conversation_id: view.conversation_id },
      turn.operation_id,
      "completed"
    )
  }
  let followed = false
  if (
    isCurrent() &&
    sameBoundary(controller.getCurrent().view, turn.origin_view)
  ) {
    followed = await followStoppedChatTurnSelection(stopSignal, controller, {
      kind: "after_message",
      message_id: resultId
    })
  }

  if (turn.persistence === 'server' && !retained) {
    if (!followed) return skipped('saved_reply_requires_open');
    await dismissHistoryTurnRecovery(fresh.scope, view, turn.operation_id, 'completed');
  }
  await controller.refreshRecovery?.()
  return { status: "kept", resultId, followed, retained }
}

/**
 * Put a retained turn with an `outcome` back into the selected history. Safe
 * to call more than once for the same operation: concurrent calls share one
 * attempt and a finished one finds nothing left to do.
 */
export const keepHistoryTurnRecovery = (
  controller: HistorySelectionController,
  entry: HistoryTurnRecoveryEntry,
  stopSignal?: AbortSignal
): Promise<HistoryTurnKeepResult> => {
  const operationId = entry.turn.operation_id
  const running = inFlight.get(operationId)
  if (running) return running
  const attempt = keep(controller, entry, stopSignal)
    .catch((error): HistoryTurnKeepResult => ({
      status: "skipped",
      reason: errorCode(error)
    }))
    .finally(() => {
      inFlight.delete(operationId)
    })
  inFlight.set(operationId, attempt)
  return attempt
}
