/**
 * How a selected-history turn that ended without a saved reply is settled
 * (CC-02, CM-02, #3107).
 *
 * - A turn whose request the server answered with an error before any reply
 *   arrived has no reply to recover. It is shown as an error reply under its
 *   question, with the usual Retry and Switch model actions, instead of being
 *   parked in "review".
 * - A server-settled turn learns which question the server saved from the
 *   first frame of its stream. When the request fails before that frame, the
 *   server names the saved question in its error instead
 *   (`detail.tldw_history_admission_v1`), so the question stays in view and
 *   Retry answers it without saving it twice.
 * - Outcomes the client cannot know (a stream that dropped after the server
 *   took the question, a save that may have happened) stay in review.
 */
import { NativeHistoryBranchChoice } from "@/hooks/chat/native-character-preflight"
import { parseNativeHistoryAdmission } from "@/services/chat-history-selection"
import type { HistoryTurnFailure } from "@/db/dexie/types"
import type { HistoryAdmissionV1 } from "@/types/history-selection"
import type { HistorySendTurn } from "@/types/chat-modes"
import {
  describeSendFailure,
  isDefinitiveSendFailure,
  sendFailureTarget
} from "@/utils/chat-send-failure"

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null

/** The admission a failed server-settled request names, if any. */
const failureAdmissionValue = (error: unknown): unknown => {
  const seen = new Set<unknown>()
  let current: unknown = error
  while (current && typeof current === "object" && !seen.has(current)) {
    seen.add(current)
    const entry = current as Record<string, unknown>
    const details = asRecord(entry.details)
    for (const body of [asRecord(details?.detail), details, asRecord(entry.detail)]) {
      if (body && body.tldw_history_admission_v1 !== undefined)
        return body.tldw_history_admission_v1
    }
    current = entry.cause
  }
  return undefined
}

/**
 * The question a server-settled request saved before it failed, when its
 * error names it and the name matches this turn. Anything that does not match
 * is ignored: the turn then stays unconfirmed, as before.
 */
export const adoptFailureAdmission = (
  historyTurn: Pick<
    HistorySendTurn,
    "owner" | "persistence" | "admission" | "selection" | "capture"
  >,
  error: unknown
): HistoryAdmissionV1 | undefined => {
  if (
    historyTurn.persistence !== "server" ||
    historyTurn.admission ||
    historyTurn.owner.kind !== "native" ||
    !historyTurn.selection
  )
    return undefined
  const value = failureAdmissionValue(error)
  if (value === undefined) return undefined
  try {
    const admission = parseNativeHistoryAdmission(
      historyTurn.owner,
      historyTurn.selection,
      value
    )
    // A saved question is new: it can never be a message the view already had.
    if (
      historyTurn.capture.snapshot.nodes.some(
        (row) => row.id === admission.input_message_id
      )
    )
      return undefined
    return admission
  } catch {
    return undefined
  }
}

/**
 * Why the turn got no reply, when the server answered with an error before
 * any reply arrived (or, for a reply only this client saves, it gave up
 * waiting for the first token). Undefined when the outcome is not known.
 */
export const definitiveHistoryTurnFailure = ({
  error,
  historyTurn,
  aborted,
  scopeInvalidated,
  replyCompleted,
  fullText,
  selectedModel
}: {
  error: unknown
  historyTurn: Pick<HistorySendTurn, "persistence" | "admission" | "resultId">
  aborted: boolean
  scopeInvalidated: boolean
  replyCompleted: boolean
  fullText: string
  selectedModel: string
}): HistoryTurnFailure | undefined => {
  if (
    aborted ||
    scopeInvalidated ||
    replyCompleted ||
    fullText.trim().length > 0 ||
    historyTurn.resultId ||
    error instanceof NativeHistoryBranchChoice
  )
    return undefined
  // A client-settled turn saves its question first; a failure before that is a
  // failed save, which stays in review.
  if (historyTurn.persistence !== "server" && !historyTurn.admission) return undefined
  const failure = describeSendFailure(error, sendFailureTarget(selectedModel))
  return isDefinitiveSendFailure(
    failure,
    historyTurn.persistence === "server" ? "server" : "client"
  )
    ? failure
    : undefined
}

/**
 * What is known about why a turn that stays in review got no reply, such as
 * no response before the startup limit (CM-06). Shown with its review.
 */
export const historyTurnFailureCause = (
  error: unknown,
  selectedModel: string
): HistoryTurnFailure | undefined =>
  describeSendFailure(error, sendFailureTarget(selectedModel))
