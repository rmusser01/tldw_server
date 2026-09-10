import { useCallback, useLayoutEffect, useReducer, useRef } from "react"
import {
  PromptImprovementApiError,
  collectProtectedTokens,
  improvePrompt,
  promptPreservationIsSafe,
  type PromptImproveErrorCode,
  type PromptImproveModelSelection,
  type PromptImproveResponse,
  type PromptImproveTarget,
  type RecognizedPromptToken
} from "@/services/prompt-improvement"
import type { PromptImprovementLimits } from "@/services/prompts-api"
import {
  createPromptAssistInitialState,
  promptOperationIsFresh,
  promptRoutesEqual,
  reducePromptAssist,
  type PromptAssistFailure,
  type PromptAssistMode,
  type PromptAssistOperation,
  type PromptUndoSnapshot
} from "./prompt-assist-state"

export type PromptTargetAdapter = {
  target: PromptImproveTarget
  read: () => string
  readRevision: () => string
  apply: (candidate: string) => void
  captureUndo: () => PromptUndoSnapshot
  restoreUndo: (snapshot: PromptUndoSnapshot) => void
}

export type UsePromptAssistOptions = {
  adapter: PromptTargetAdapter
  readActiveRoute: () => PromptImproveModelSelection
  readRecognizedTokens?: () => readonly RecognizedPromptToken[]
  limits?: PromptImprovementLimits | null
  contextKey: string
  surfaceOpen: boolean
}

type CommittedPromptTarget = UsePromptAssistOptions & {
  text: string
  revision: string
  route: PromptImproveModelSelection
}

type PromptOperationGuard = {
  operationId: string
  epoch: number
  contextKey: string
}

const readCommittedTarget = (
  options: UsePromptAssistOptions
): CommittedPromptTarget => ({
  ...options,
  text: options.adapter.read(),
  revision: options.adapter.readRevision(),
  route: options.readActiveRoute()
})

const sameCommittedTarget = (
  left: CommittedPromptTarget,
  right: CommittedPromptTarget
): boolean =>
  left.adapter.target === right.adapter.target &&
  left.text === right.text &&
  left.revision === right.revision &&
  promptRoutesEqual(left.route, right.route) &&
  left.contextKey === right.contextKey &&
  left.surfaceOpen === right.surfaceOpen

const isPromptImproveErrorCode = (value: unknown): value is PromptImproveErrorCode =>
  typeof value === "string" &&
  [
    "invalid_input",
    "missing_model",
    "unsupported_model",
    "provider_not_configured",
    "draft_too_large",
    "provider_rate_limited",
    "provider_timeout",
    "provider_unavailable",
    "model_refusal",
    "invalid_model_output",
    "preservation_failed",
    "internal_error"
  ].includes(value)

const toFailure = (error: unknown): PromptAssistFailure => {
  if (error instanceof PromptImprovementApiError) {
    return {
      code: error.code,
      message: error.message,
      retryable: error.retryable,
      retryAfterSeconds: error.retryAfterSeconds,
      requestId: error.requestId
    }
  }
  const value = error as Partial<PromptAssistFailure> | null
  const code = isPromptImproveErrorCode(value?.code)
    ? value.code
    : "provider_unavailable"
  return {
    code,
    message:
      code === "provider_unavailable"
        ? "The prompt improvement service is unavailable."
        : "Prompt improvement failed.",
    retryable: typeof value?.retryable === "boolean" ? value.retryable : true
  }
}

const requireClientReview = (
  response: PromptImproveResponse
): PromptImproveResponse => ({
  schema_version: response.schema_version,
  operation_id: response.operation_id,
  status: response.status,
  improved_text: response.improved_text,
  findings: response.findings,
  review_required: true,
  warnings: response.warnings,
  resolved_model: response.resolved_model,
  meta_prompt_version: response.meta_prompt_version
})

export function usePromptAssist({
  adapter,
  readActiveRoute,
  readRecognizedTokens,
  limits,
  contextKey,
  surfaceOpen
}: UsePromptAssistOptions) {
  const [state, dispatch] = useReducer(
    reducePromptAssist,
    undefined,
    createPromptAssistInitialState
  )
  const mountedRef = useRef(true)
  const inFlightRef = useRef(false)
  const operationIdRef = useRef<string | null>(null)
  const operationGuardRef = useRef<PromptOperationGuard | null>(null)
  const lastModeRef = useRef<PromptAssistMode>("improve_now")
  const targetEpochRef = useRef(0)
  const latestRef = useRef<CommittedPromptTarget | null>(null)
  if (!latestRef.current) {
    latestRef.current = readCommittedTarget({
      adapter,
      readActiveRoute,
      readRecognizedTokens,
      limits,
      contextKey,
      surfaceOpen
    })
  }

  const clearLifecycle = useCallback(() => {
    targetEpochRef.current += 1
    operationIdRef.current = null
    operationGuardRef.current = null
    inFlightRef.current = false
    dispatch({ type: "lifecycle_cleared" })
  }, [])

  useLayoutEffect(() => {
    const previous = latestRef.current as CommittedPromptTarget
    const next = readCommittedTarget({
      adapter,
      readActiveRoute,
      readRecognizedTokens,
      limits,
      contextKey,
      surfaceOpen
    })
    const changed = !sameCommittedTarget(previous, next)
    if (changed) targetEpochRef.current += 1
    latestRef.current = next
    if (
      previous.contextKey !== next.contextKey ||
      (previous.surfaceOpen && !next.surfaceOpen)
    ) {
      operationIdRef.current = null
      operationGuardRef.current = null
      inFlightRef.current = false
      dispatch({ type: "lifecycle_cleared" })
    }
  })

  useLayoutEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      targetEpochRef.current += 1
      operationIdRef.current = null
      operationGuardRef.current = null
      inFlightRef.current = false
    }
  }, [])

  const run = useCallback(
    async (mode: PromptAssistMode): Promise<void> => {
      const committed = latestRef.current as CommittedPromptTarget
      if (inFlightRef.current || !committed.surfaceOpen) return
      const originalText = committed.adapter.read()
      if (!originalText.trim()) return
      const route = committed.readActiveRoute()
      if (!route.selected_model?.trim()) return

      const operationId = globalThis.crypto.randomUUID()
      const operationEpoch = targetEpochRef.current
      const operationContextKey = committed.contextKey
      const operation: PromptAssistOperation = {
        operationId,
        target: committed.adapter.target,
        mode,
        originalText,
        revision: committed.adapter.readRevision(),
        route
      }
      operationIdRef.current = operationId
      operationGuardRef.current = {
        operationId,
        epoch: operationEpoch,
        contextKey: operationContextKey
      }
      inFlightRef.current = true
      lastModeRef.current = mode
      dispatch({ type: "request_started", operation })

      try {
        const protectedTokens = collectProtectedTokens(
          originalText,
          committed.readRecognizedTokens?.() ?? [],
          committed.limits ?? undefined
        )
        const request = {
          operation_id: operationId,
          target: committed.adapter.target,
          text: originalText,
          model_selection: route,
          protected_tokens: protectedTokens
        }
        const response = committed.limits
          ? await improvePrompt(request, committed.limits)
          : await improvePrompt(request)
        const guardedResponse =
          response.status === "improved" &&
          response.improved_text &&
          !promptPreservationIsSafe(
            originalText,
            response.improved_text,
            protectedTokens
          )
            ? requireClientReview(response)
            : response
        const latest = latestRef.current as CommittedPromptTarget
        if (
          !mountedRef.current ||
          operationIdRef.current !== operationId ||
          !latest.surfaceOpen ||
          latest.contextKey !== operationContextKey
        ) {
          return
        }

        const live = {
          text: latest.adapter.read(),
          revision: latest.adapter.readRevision(),
          route: latest.readActiveRoute()
        }
        const autoApply =
          targetEpochRef.current === operationEpoch &&
          mode === "improve_now" &&
          guardedResponse.status === "improved" &&
          Boolean(guardedResponse.improved_text) &&
          !guardedResponse.review_required &&
          promptOperationIsFresh(operation, live)
        let undoSnapshot: PromptUndoSnapshot
        if (autoApply) {
          undoSnapshot = latest.adapter.captureUndo()
          latest.adapter.apply(guardedResponse.improved_text as string)
        }
        dispatch({
          type: "response_received",
          operationId,
          response: guardedResponse,
          liveText: live.text,
          liveRevision: live.revision,
          liveRoute: live.route,
          autoApplied: autoApply,
          undoSnapshot
        })
      } catch (error) {
        if (!mountedRef.current || operationIdRef.current !== operationId) return
        dispatch({
          type: "request_failed",
          operationId,
          error: toFailure(error)
        })
      } finally {
        if (operationIdRef.current === operationId) {
          inFlightRef.current = false
        }
      }
    }, [])

  const improveNow = useCallback(() => run("improve_now"), [run])
  const reviewChanges = useCallback(() => run("review_changes"), [run])
  const retry = useCallback(() => run(lastModeRef.current), [run])

  const editCandidate = useCallback((candidate: string) => {
    dispatch({ type: "candidate_edited", candidate })
  }, [])

  const applyCandidate = useCallback(() => {
    if (state.status !== "reviewing") return
    const latest = latestRef.current as CommittedPromptTarget
    const guard = operationGuardRef.current
    const live = {
      text: latest.adapter.read(),
      revision: latest.adapter.readRevision(),
      route: latest.readActiveRoute()
    }
    const fresh = Boolean(
      mountedRef.current &&
        latest.surfaceOpen &&
        guard?.operationId === state.operation.operationId &&
        guard.contextKey === latest.contextKey &&
        guard.epoch === targetEpochRef.current &&
        promptOperationIsFresh(state.operation, live)
    )
    if (!fresh) {
      dispatch({
        type: "review_apply_requested",
        liveText: live.text,
        fresh: false,
        applied: false
      })
      return
    }
    const undoSnapshot = latest.adapter.captureUndo()
    latest.adapter.apply(state.candidate)
    dispatch({
      type: "review_apply_requested",
      liveText: live.text,
      fresh: true,
      applied: true,
      undoSnapshot
    })
  }, [state])

  const confirmReplaceCurrent = useCallback(() => {
    if (state.status !== "reviewing" || !state.replaceConfirmationRequired) return
    const latest = latestRef.current as CommittedPromptTarget
    const guard = operationGuardRef.current
    if (
      !mountedRef.current ||
      !latest.surfaceOpen ||
      guard?.operationId !== state.operation.operationId ||
      guard.contextKey !== latest.contextKey ||
      latest.adapter.target !== state.operation.target
    ) {
      return
    }
    const undoSnapshot = latest.adapter.captureUndo()
    latest.adapter.apply(state.candidate)
    dispatch({ type: "replace_confirmed", undoSnapshot })
  }, [state])

  const undo = useCallback(() => {
    const latest = latestRef.current as CommittedPromptTarget
    if (!state.undo || state.undo.target !== latest.adapter.target) return
    latest.adapter.restoreUndo(state.undo.snapshot)
    dispatch({ type: "undo_restored" })
  }, [state.undo])

  const notifyTargetEdited = useCallback(() => {
    dispatch({ type: "target_edited" })
  }, [])

  const notifySendOrSave = clearLifecycle
  const dismiss = clearLifecycle

  return {
    state,
    improveNow,
    reviewChanges,
    retry,
    editCandidate,
    applyCandidate,
    confirmReplaceCurrent,
    undo,
    notifyTargetEdited,
    notifySendOrSave,
    dismiss
  }
}
