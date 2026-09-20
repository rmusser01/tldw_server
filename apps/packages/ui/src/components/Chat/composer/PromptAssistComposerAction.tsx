import { Button } from "@/components/Common/Button"
import { PromptAssistMenu } from "@/components/Common/PromptAssist/PromptAssistMenu"
import { PromptAssistPanel } from "@/components/Common/PromptAssist/PromptAssistPanel"
import {
  type PromptTargetAdapter,
  usePromptAssist
} from "@/components/Common/PromptAssist/usePromptAssist"
import type { useSimpleForm } from "@/hooks/useSimpleForm"
import type { PromptImproveModelSelection } from "@/services/prompt-improvement"
import { fetchPromptCapabilities } from "@/services/prompts-api"
import { useQuery } from "@tanstack/react-query"
import { Drawer } from "antd"
import React from "react"
import { useTranslation } from "react-i18next"

import type { ComposerPromptAssistMutation } from "./hooks/useComposerText"

type ComposerForm = Pick<
  ReturnType<typeof useSimpleForm<{ message: string; image: string }>>,
  "values" | "setFieldValue"
>

type ControllerMutation = {
  fromRevision: number
  expectedValue: string
}

export type PromptAssistComposerActionProps = {
  form: ComposerForm
  messageRevision: number
  promptAssistMutation: ComposerPromptAssistMutation
  promptAssistSavedAttemptId: number | null
  modelSelection: PromptImproveModelSelection | null
  promptAssistContextKey: string
  promptAssistBackendKey?: string | null
  sending?: boolean
  surfaceOpen?: boolean
  narrow?: boolean
  onSelectModel?: () => void
  onReturnFocus?: () => void
}

export function PromptAssistComposerAction({
  form,
  messageRevision,
  promptAssistMutation,
  promptAssistSavedAttemptId,
  modelSelection,
  promptAssistContextKey,
  promptAssistBackendKey = null,
  sending = false,
  surfaceOpen = true,
  narrow = false,
  onSelectModel,
  onReturnFocus
}: PromptAssistComposerActionProps) {
  const { t } = useTranslation(["common"])
  const [panelOpen, setPanelOpen] = React.useState(false)
  const [inspectionOpen, setInspectionOpen] = React.useState(false)
  const normalizedBackendKey = promptAssistBackendKey?.trim() || null
  const modelSelectionRef = React.useRef(modelSelection)
  const controllerMutationRef = React.useRef<ControllerMutation | null>(null)
  const observedRevisionRef = React.useRef(promptAssistMutation.revision)
  const observedSavedAttemptRef = React.useRef(promptAssistSavedAttemptId)
  const pendingResetAttemptRef = React.useRef<number | null>(null)
  const pendingUndoFocusRef = React.useRef(false)
  const pendingDrawerFocusRef = React.useRef(false)
  const message = form.values.message
  const setFieldValue = form.setFieldValue
  modelSelectionRef.current = modelSelection

  const { data: promptCapabilities } = useQuery({
    queryKey: ["promptCapabilities", normalizedBackendKey],
    queryFn: fetchPromptCapabilities,
    enabled: Boolean(normalizedBackendKey),
    retry: false
  })
  const capability = !promptCapabilities
    ? "unknown"
    : promptCapabilities.availability === "available" &&
        promptCapabilities.prompt_improvement_v1.supported
      ? "supported"
      : "unsupported"

  const adapter = React.useMemo<PromptTargetAdapter>(
    () => ({
      target: "user_message",
      read: () => message,
      readRevision: () => String(messageRevision),
      apply: (candidate) => {
        controllerMutationRef.current = {
          fromRevision: messageRevision,
          expectedValue: candidate
        }
        setFieldValue("message", candidate)
      },
      captureUndo: () => message,
      restoreUndo: (snapshot) => {
        if (typeof snapshot === "string") {
          controllerMutationRef.current = {
            fromRevision: messageRevision,
            expectedValue: snapshot
          }
          setFieldValue("message", snapshot)
        }
      }
    }),
    [message, messageRevision, setFieldValue]
  )
  const lifecycleKey = JSON.stringify([
    promptAssistContextKey,
    modelSelection?.selected_model.trim() ?? "",
    modelSelection?.provider_hint?.trim() ?? "",
    normalizedBackendKey ?? ""
  ])
  const promptAssist = usePromptAssist({
    adapter,
    readActiveRoute: () => modelSelectionRef.current ?? { selected_model: "" },
    limits:
      promptCapabilities?.prompt_improvement_v1.supported === true
        ? promptCapabilities.prompt_improvement_v1.limits
        : null,
    contextKey: lifecycleKey,
    surfaceOpen
  })
  const {
    dismiss: dismissPromptAssist,
    notifySendOrSave,
    notifyTargetEdited,
    state: promptAssistState
  } = promptAssist

  React.useLayoutEffect(() => {
    pendingDrawerFocusRef.current = false

    return () => {
      pendingDrawerFocusRef.current = false
    }
  }, [lifecycleKey, surfaceOpen])

  React.useLayoutEffect(() => {
    if (observedRevisionRef.current === promptAssistMutation.revision) return
    observedRevisionRef.current = promptAssistMutation.revision
    const controllerMutation = controllerMutationRef.current
    if (
      controllerMutation &&
      controllerMutation.fromRevision !== messageRevision &&
      controllerMutation.expectedValue === message
    ) {
      controllerMutationRef.current = null
      pendingResetAttemptRef.current = null
      return
    }
    controllerMutationRef.current = null
    if (promptAssistMutation.source === "optimistic_reset") {
      if (promptAssistSavedAttemptId === promptAssistMutation.attemptId) {
        pendingResetAttemptRef.current = null
        notifySendOrSave()
      } else {
        pendingResetAttemptRef.current = promptAssistMutation.attemptId
      }
      return
    }
    pendingResetAttemptRef.current = null
    notifyTargetEdited()
  }, [
    message,
    messageRevision,
    notifySendOrSave,
    notifyTargetEdited,
    promptAssistMutation,
    promptAssistSavedAttemptId
  ])

  React.useLayoutEffect(() => {
    if (observedSavedAttemptRef.current === promptAssistSavedAttemptId) return
    observedSavedAttemptRef.current = promptAssistSavedAttemptId
    if (
      promptAssistSavedAttemptId === null ||
      pendingResetAttemptRef.current !== promptAssistSavedAttemptId
    ) {
      return
    }
    pendingResetAttemptRef.current = null
    controllerMutationRef.current = null
    notifySendOrSave()
  }, [notifySendOrSave, promptAssistSavedAttemptId])

  React.useLayoutEffect(() => {
    if (promptAssistState.status === "applied") {
      pendingDrawerFocusRef.current = true
      setPanelOpen(false)
      setInspectionOpen(false)
      return
    }
    if (promptAssistState.status === "idle" && pendingUndoFocusRef.current) {
      pendingUndoFocusRef.current = false
      onReturnFocus?.()
    }
  }, [onReturnFocus, promptAssistState])

  const closePanel = React.useCallback(() => {
    dismissPromptAssist()
    setPanelOpen(false)
    setInspectionOpen(false)
  }, [dismissPromptAssist])
  const returnFocus = React.useCallback(() => {
    pendingDrawerFocusRef.current = true
    setPanelOpen(false)
    setInspectionOpen(false)
  }, [])
  const closeDrawer = React.useCallback(() => {
    if (promptAssistState.status !== "applied") {
      dismissPromptAssist()
    }
    pendingDrawerFocusRef.current = true
    setPanelOpen(false)
    setInspectionOpen(false)
  }, [dismissPromptAssist, promptAssistState.status])
  const handleDrawerAfterOpenChange = React.useCallback(
    (open: boolean) => {
      if (open || !pendingDrawerFocusRef.current) return
      pendingDrawerFocusRef.current = false
      onReturnFocus?.()
    },
    [onReturnFocus]
  )
  const start = React.useCallback((operation: () => Promise<void>) => {
    setPanelOpen(true)
    void operation()
  }, [])
  const undoAndReturnFocus = React.useCallback(() => {
    pendingUndoFocusRef.current = true
    promptAssist.undo()
  }, [promptAssist])

  const panel = (
    <PromptAssistPanel
      state={promptAssist.state}
      onCancel={closePanel}
      onRetry={() => start(promptAssist.retry)}
      onSelectModel={onSelectModel}
      onCandidateChange={promptAssist.editCandidate}
      onApply={promptAssist.applyCandidate}
      onConfirmReplace={promptAssist.confirmReplaceCurrent}
      onUndo={undoAndReturnFocus}
      onRequestReturnFocus={returnFocus}
      inspectionOpen={inspectionOpen}
      onInspectionOpenChange={(open) => {
        if (!open) pendingDrawerFocusRef.current = true
        setInspectionOpen(open)
        setPanelOpen(open)
      }}
    />
  )

  if (!surfaceOpen) return null

  return (
    <div className="min-w-0">
      <PromptAssistMenu
        draft={form.values.message}
        capability={capability}
        modelSelection={modelSelection}
        onImproveNow={() => start(promptAssist.improveNow)}
        onReviewChanges={() => start(promptAssist.reviewChanges)}
        onSelectModel={onSelectModel}
        disabled={sending || promptAssist.state.status === "analyzing"}
      />

      {promptAssist.state.status === "applied" ? (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <span role="status" className="text-xs text-muted-foreground">
            {t("common:promptAssist.applied", "Improvement applied.")}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setInspectionOpen(true)
              setPanelOpen(true)
            }}>
            {t("common:promptAssist.viewChanges", "View changes")}
          </Button>
          {promptAssist.state.undo ? (
            <Button variant="outline" size="sm" onClick={undoAndReturnFocus}>
              {t("common:promptAssist.undo", "Undo improvement")}
            </Button>
          ) : null}
        </div>
      ) : null}

      <Drawer
        placement="right"
        open={
          panelOpen &&
          promptAssist.state.status !== "idle" &&
          (promptAssist.state.status !== "applied" || inspectionOpen)
        }
        onClose={closeDrawer}
        afterOpenChange={handleDrawerAfterOpenChange}
        focusable={{ focusTriggerAfterClose: false }}
        size={narrow ? "100%" : 480}
        title={t("common:promptAssist.region", "Prompt improvement")}>
        <div onKeyDown={(event) => event.stopPropagation()}>{panel}</div>
      </Drawer>
    </div>
  )
}
