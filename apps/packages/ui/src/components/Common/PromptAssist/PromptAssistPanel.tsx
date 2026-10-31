import { Button } from "@/components/Common/Button"
import type { PromptImproveErrorCode } from "@/services/prompt-improvement"
import { LoaderCircle } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"

import { PromptReviewSurface } from "./PromptReviewSurface"
import type { PromptAssistState } from "./prompt-assist-state"

export type PromptAssistPanelProps = {
  state: PromptAssistState
  onCancel: () => void
  onRetry: () => void
  onSelectModel?: () => void
  onCandidateChange: (candidate: string) => void
  onApply: () => void
  onConfirmReplace: () => void
  onUndo: () => void
  onRequestReturnFocus?: () => void
  inspectionOpen?: boolean
  onInspectionOpenChange?: (open: boolean) => void
}

const modelRecoveryCodes = new Set([
  "missing_model",
  "unsupported_model",
  "provider_not_configured"
])

const errorFallback: Record<PromptImproveErrorCode, string> = {
  invalid_input: "Check the draft and try again.",
  missing_model: "Select a chat model and try again.",
  unsupported_model: "Choose a supported chat model and try again.",
  provider_not_configured: "Configure the selected provider and try again.",
  draft_too_large: "Shorten the draft and try again.",
  provider_rate_limited: "The active provider is temporarily rate limited.",
  provider_timeout: "The prompt improvement request timed out.",
  provider_unavailable: "The prompt improvement service is unavailable.",
  model_refusal: "The active model declined to improve this prompt.",
  invalid_model_output: "The model returned an invalid improvement result.",
  preservation_failed:
    "The candidate could not safely preserve protected content.",
  internal_error: "Prompt improvement failed."
}

export function PromptAssistPanel({
  state,
  onCancel,
  onRetry,
  onSelectModel,
  onCandidateChange,
  onApply,
  onConfirmReplace,
  onUndo,
  onRequestReturnFocus,
  inspectionOpen: controlledInspectionOpen,
  onInspectionOpenChange
}: PromptAssistPanelProps) {
  const { t } = useTranslation(["common"])
  const [internalInspectionOpen, setInternalInspectionOpen] = useState(false)
  const inspectionOpen = controlledInspectionOpen ?? internalInspectionOpen
  const setInspectionOpen = (open: boolean) => {
    onInspectionOpenChange?.(open)
    if (controlledInspectionOpen === undefined) {
      setInternalInspectionOpen(open)
    }
  }
  const [announcement, setAnnouncement] = useState("")
  const pendingFocusOperationRef = useRef<string | null>(null)
  const currentOperationId =
    state.status === "idle" ? null : state.operation.operationId
  const retryIdentity =
    state.status === "failed"
      ? `${state.operation.operationId}:${state.error.code}:${state.error.requestId ?? ""}:${state.error.retryAfterSeconds ?? 0}`
      : ""
  const retryAfterSeconds =
    state.status === "failed"
      ? Math.max(0, state.error.retryAfterSeconds ?? 0)
      : 0
  const [retryRemaining, setRetryRemaining] = useState(retryAfterSeconds)

  const close = () => {
    pendingFocusOperationRef.current = null
    onCancel()
    onRequestReturnFocus?.()
  }

  useEffect(() => {
    const pendingOperationId = pendingFocusOperationRef.current
    if (!pendingOperationId) return
    if (
      state.status === "applied" &&
      currentOperationId === pendingOperationId
    ) {
      pendingFocusOperationRef.current = null
      onRequestReturnFocus?.()
      return
    }
    if (
      state.status !== "reviewing" ||
      currentOperationId !== pendingOperationId
    ) {
      pendingFocusOperationRef.current = null
    }
  }, [currentOperationId, onRequestReturnFocus, state.status])

  useEffect(
    () => () => {
      pendingFocusOperationRef.current = null
    },
    []
  )

  useEffect(() => {
    if (!retryIdentity || retryAfterSeconds <= 0) {
      setRetryRemaining(0)
      return
    }
    const eligibleAt = Date.now() + retryAfterSeconds * 1_000
    const update = () => {
      const remaining = Math.max(
        0,
        Math.ceil((eligibleAt - Date.now()) / 1_000)
      )
      setRetryRemaining(remaining)
    }
    update()
    const timer = window.setInterval(update, 250)
    return () => window.clearInterval(timer)
  }, [retryAfterSeconds, retryIdentity])

  const undo = () => {
    if (state.status !== "applied" || !state.undo) return
    onUndo()
    setAnnouncement(t("common:promptAssist.undone", "Improvement undone."))
  }

  const requestApply = () => {
    if (state.status !== "reviewing") return
    pendingFocusOperationRef.current = state.operation.operationId
    onApply()
  }

  const requestReplace = () => {
    if (state.status !== "reviewing") return
    pendingFocusOperationRef.current = state.operation.operationId
    onConfirmReplace()
  }

  return (
    <section
      role="region"
      aria-label={t("common:promptAssist.region", "Prompt improvement")}
      tabIndex={-1}
      className="min-w-0 space-y-5"
      onKeyDown={(event) => {
        if (
          event.key !== "Escape" ||
          state.status === "reviewing" ||
          (state.status === "applied" && inspectionOpen)
        ) {
          return
        }
        event.preventDefault()
        event.stopPropagation()
        close()
      }}>
      <div role="status" aria-live="polite" aria-atomic="true">
        {announcement ||
          (state.status === "analyzing"
            ? t("common:promptAssist.analyzing", "Analyzing with {{model}}", {
                model:
                  state.operation.route.selected_model.toLowerCase() === "auto"
                    ? t("common:promptAssist.autoModel", "Auto")
                    : state.operation.route.selected_model
              })
            : state.status === "applied"
              ? t("common:promptAssist.applied", "Improvement applied.")
              : state.status === "idle" && state.notice === "no_change"
                ? t(
                    "common:promptAssist.noChange",
                    "No useful improvement found."
                  )
                : state.status === "failed" && retryRemaining > 0
                  ? t(
                      "common:promptAssist.retryCountdown",
                      "Retry available in {{count}} seconds.",
                      { count: retryRemaining }
                    )
                  : "")}
      </div>

      {state.status === "analyzing" ? (
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <LoaderCircle
              data-testid="prompt-assist-spinner"
              aria-hidden="true"
              className="h-4 w-4 animate-spin motion-reduce:animate-none"
            />
            {t(
              "common:promptAssist.analyzingHelp",
              "Checking clarity, structure, and protected content."
            )}
          </div>
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-muted/30 p-3 text-sm">
            {state.operation.originalText}
          </pre>
          <Button variant="outline" size="lg" onClick={close}>
            {t("common:cancel", "Cancel")}
          </Button>
        </div>
      ) : null}

      {state.status === "failed" ? (
        <div className="space-y-4">
          <div
            role="alert"
            className="rounded-md border border-danger/30 bg-danger/10 p-3 text-sm text-text">
            {t(
              `common:promptAssist.errors.${state.error.code}`,
              errorFallback[state.error.code]
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {modelRecoveryCodes.has(state.error.code) && onSelectModel ? (
              <Button variant="primary" size="lg" onClick={onSelectModel}>
                {t("common:promptAssist.selectModel", "Select model")}
              </Button>
            ) : null}
            {state.error.retryable ? (
              <Button
                variant="primary"
                size="lg"
                disabled={retryRemaining > 0}
                onClick={onRetry}>
                {t("common:retry", "Retry")}
              </Button>
            ) : null}
            <Button variant="outline" size="lg" onClick={close}>
              {t("common:cancel", "Cancel")}
            </Button>
          </div>
        </div>
      ) : null}

      {state.status === "reviewing" ? (
        <PromptReviewSurface
          original={state.operation.originalText}
          candidate={state.candidate}
          findings={state.response.findings}
          warnings={state.response.warnings}
          notice={state.notice}
          resolvedModel={state.response.resolved_model}
          replaceConfirmationRequired={state.replaceConfirmationRequired}
          onCandidateChange={onCandidateChange}
          onApply={requestApply}
          onConfirmReplace={requestReplace}
          onCancel={close}
          onEscape={close}
        />
      ) : null}

      {state.status === "applied" && inspectionOpen ? (
        <PromptReviewSurface
          mode="inspection"
          original={state.operation.originalText}
          candidate={state.candidate}
          findings={state.response.findings}
          warnings={state.response.warnings}
          notice={null}
          resolvedModel={state.response.resolved_model}
          onCandidateChange={onCandidateChange}
          onApply={requestApply}
          onConfirmReplace={requestReplace}
          onUndo={state.undo ? undo : undefined}
          onCancel={() => setInspectionOpen(false)}
          onEscape={() => setInspectionOpen(false)}
        />
      ) : null}

      {state.status === "applied" && !inspectionOpen ? (
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="lg"
            onClick={() => setInspectionOpen(true)}>
            {t("common:promptAssist.viewChanges", "View changes")}
          </Button>
          {state.undo ? (
            <Button variant="outline" size="lg" onClick={undo}>
              {t("common:promptAssist.undo", "Undo improvement")}
            </Button>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}
