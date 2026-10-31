import type {
  PromptImproveErrorCode,
  PromptImproveModelSelection,
  PromptImproveResponse,
  PromptImproveTarget
} from "@/services/prompt-improvement"

export type PromptAssistMode = "improve_now" | "review_changes"

export type PromptAssistOperation = {
  operationId: string
  target: PromptImproveTarget
  mode: PromptAssistMode
  originalText: string
  revision: string
  route: PromptImproveModelSelection
}

export type PromptUndoSnapshot = unknown

export type PromptUndoEntry = {
  target: PromptImproveTarget
  snapshot: PromptUndoSnapshot
  operationId: string
  candidate: string
}

export type PromptAssistNotice =
  | "no_change"
  | "review_required"
  | "draft_changed"
  | "route_changed"
  | null

export type PromptAssistFailure = {
  code: PromptImproveErrorCode
  message: string
  retryable: boolean
  retryAfterSeconds?: number | null
  requestId?: string | null
}

type PromptAssistBase = {
  undo: PromptUndoEntry | null
}

export type PromptAssistIdleState = PromptAssistBase & {
  status: "idle"
  notice: "no_change" | null
}

export type PromptAssistAnalyzingState = PromptAssistBase & {
  status: "analyzing"
  operation: PromptAssistOperation
}

export type PromptAssistReviewingState = PromptAssistBase & {
  status: "reviewing"
  operation: PromptAssistOperation
  response: PromptImproveResponse
  candidate: string
  notice: Exclude<PromptAssistNotice, "no_change">
  replaceConfirmationRequired: boolean
}

export type PromptAssistAppliedState = PromptAssistBase & {
  status: "applied"
  operation: PromptAssistOperation
  response: PromptImproveResponse
  candidate: string
}

export type PromptAssistFailedState = PromptAssistBase & {
  status: "failed"
  operation: PromptAssistOperation
  error: PromptAssistFailure
}

export type PromptAssistState =
  | PromptAssistIdleState
  | PromptAssistAnalyzingState
  | PromptAssistReviewingState
  | PromptAssistAppliedState
  | PromptAssistFailedState

export type PromptAssistAction =
  | { type: "request_started"; operation: PromptAssistOperation }
  | {
      type: "response_received"
      operationId: string
      response: PromptImproveResponse
      liveText: string
      liveRevision: string
      liveRoute: PromptImproveModelSelection
      autoApplied: boolean
      undoSnapshot?: PromptUndoSnapshot
    }
  | {
      type: "request_failed"
      operationId: string
      error: PromptAssistFailure
    }
  | { type: "candidate_edited"; candidate: string }
  | {
      type: "review_apply_requested"
      liveText: string
      fresh: boolean
      applied: boolean
      undoSnapshot?: PromptUndoSnapshot
    }
  | { type: "replace_confirmed"; undoSnapshot: PromptUndoSnapshot }
  | { type: "target_edited" }
  | { type: "undo_restored" }
  | { type: "lifecycle_cleared" }

export const createPromptAssistInitialState = (): PromptAssistIdleState => ({
  status: "idle",
  notice: null,
  undo: null
})

export const promptRoutesEqual = (
  left: PromptImproveModelSelection,
  right: PromptImproveModelSelection
): boolean =>
  left.selected_model === right.selected_model &&
  (left.provider_hint ?? null) === (right.provider_hint ?? null)

export const promptOperationIsFresh = (
  operation: PromptAssistOperation,
  live: {
    text: string
    revision: string
    route: PromptImproveModelSelection
  }
): boolean =>
  operation.originalText === live.text &&
  operation.revision === live.revision &&
  promptRoutesEqual(operation.route, live.route)

const reviewState = (
  state: PromptAssistAnalyzingState,
  response: PromptImproveResponse,
  notice: PromptAssistReviewingState["notice"]
): PromptAssistReviewingState => ({
  status: "reviewing",
  operation: state.operation,
  response,
  candidate: response.improved_text as string,
  notice,
  replaceConfirmationRequired: notice === "draft_changed",
  undo: null
})

const appliedState = (
  operation: PromptAssistOperation,
  response: PromptImproveResponse,
  candidate: string,
  snapshot: PromptUndoSnapshot
): PromptAssistAppliedState => ({
  status: "applied",
  operation,
  response,
  candidate,
  undo: {
    target: operation.target,
    snapshot,
    operationId: operation.operationId,
    candidate
  }
})

export const reducePromptAssist = (
  state: PromptAssistState,
  action: PromptAssistAction
): PromptAssistState => {
  switch (action.type) {
    case "request_started":
      return {
        status: "analyzing",
        operation: action.operation,
        undo: null
      }
    case "response_received": {
      if (
        state.status !== "analyzing" ||
        state.operation.operationId !== action.operationId ||
        action.response.operation_id !== action.operationId
      ) {
        return state
      }
      if (action.response.status === "no_change") {
        return { status: "idle", notice: "no_change", undo: null }
      }
      if (!action.response.improved_text) {
        return {
          status: "failed",
          operation: state.operation,
          error: {
            code: "invalid_model_output",
            message: "The model returned an invalid improvement result.",
            retryable: false
          },
          undo: null
        }
      }
      if (
        state.operation.originalText !== action.liveText ||
        state.operation.revision !== action.liveRevision
      ) {
        return reviewState(state, action.response, "draft_changed")
      }
      if (!promptRoutesEqual(state.operation.route, action.liveRoute)) {
        return reviewState(state, action.response, "route_changed")
      }
      if (state.operation.mode === "review_changes") {
        return reviewState(state, action.response, null)
      }
      if (action.response.review_required || !action.autoApplied) {
        return reviewState(state, action.response, "review_required")
      }
      return appliedState(
        state.operation,
        action.response,
        action.response.improved_text,
        action.undoSnapshot
      )
    }
    case "request_failed":
      if (
        state.status !== "analyzing" ||
        state.operation.operationId !== action.operationId
      ) {
        return state
      }
      return {
        status: "failed",
        operation: state.operation,
        error: action.error,
        undo: null
      }
    case "candidate_edited":
      return state.status === "reviewing"
        ? { ...state, candidate: action.candidate }
        : state
    case "review_apply_requested":
      if (state.status !== "reviewing") return state
      if (!action.fresh || action.liveText !== state.operation.originalText) {
        return {
          ...state,
          notice: "draft_changed",
          replaceConfirmationRequired: true
        }
      }
      return action.applied
        ? appliedState(
            state.operation,
            state.response,
            state.candidate,
            action.undoSnapshot
          )
        : state
    case "replace_confirmed":
      if (state.status !== "reviewing" || !state.replaceConfirmationRequired) {
        return state
      }
      return appliedState(
        state.operation,
        state.response,
        state.candidate,
        action.undoSnapshot
      )
    case "target_edited":
      return state.undo ? { ...state, undo: null } : state
    case "undo_restored":
    case "lifecycle_cleared":
      return createPromptAssistInitialState()
    default:
      return state
  }
}
