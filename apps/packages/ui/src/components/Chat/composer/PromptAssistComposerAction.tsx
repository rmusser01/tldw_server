import { Button } from "@/components/Common/Button"
import { PromptAssistMenu } from "@/components/Common/PromptAssist/PromptAssistMenu"
import { PromptAssistPanel } from "@/components/Common/PromptAssist/PromptAssistPanel"
import {
  type PromptTargetAdapter,
  usePromptAssist
} from "@/components/Common/PromptAssist/usePromptAssist"
import { useRecipePersistenceOwner } from "@/hooks/useRecipePersistenceOwner"
import type { useSimpleForm } from "@/hooks/useSimpleForm"
import type { PromptImproveModelSelection } from "@/services/prompt-improvement"
import {
  fetchPromptCapabilities,
  revalidatePromptCapabilities
} from "@/services/prompts-api"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Drawer } from "antd"
import React from "react"
import { createPortal } from "react-dom"
import { useTranslation } from "react-i18next"

import type { ComposerPromptAssistMutation } from "./hooks/useComposerText"

const PromptRecipeBuilder = React.lazy(() =>
  import("@/components/Common/PromptAssist/recipes/PromptRecipeBuilder").then(
    ({ PromptRecipeBuilder }) => ({ default: PromptRecipeBuilder })
  )
)

type ComposerForm = Pick<
  ReturnType<typeof useSimpleForm<{ message: string; image: string }>>,
  "values" | "setFieldValue"
>

type ControllerMutation = {
  fromRevision: number
  expectedValue: string
}

type FeedbackPosition = {
  left: number
  top: number
  visible: boolean
}

function PromptAssistFeedbackOverlay({
  anchorRef,
  children
}: {
  anchorRef: React.RefObject<HTMLElement | null>
  children: React.ReactNode
}) {
  const overlayRef = React.useRef<HTMLDivElement>(null)
  const [portalTarget, setPortalTarget] = React.useState<HTMLElement | null>(
    null
  )
  const [position, setPosition] = React.useState<FeedbackPosition | null>(null)

  React.useLayoutEffect(() => {
    const body = typeof document === "undefined" ? null : document.body
    setPortalTarget(body)
  }, [])

  React.useLayoutEffect(() => {
    if (!portalTarget) return
    let animationFrame: number | null = null

    const updatePosition = () => {
      const anchor = anchorRef.current
      const overlay = overlayRef.current
      if (!anchor || !overlay) return
      const inset = 8
      const gap = 8
      const visualViewport = window.visualViewport
      const viewportLeft = visualViewport?.offsetLeft ?? 0
      const viewportTop = visualViewport?.offsetTop ?? 0
      const viewportRight =
        viewportLeft + (visualViewport?.width ?? window.innerWidth)
      const viewportBottom =
        viewportTop + (visualViewport?.height ?? window.innerHeight)
      const anchorRect = anchor.getBoundingClientRect()
      let draftRect: DOMRect | null = null
      let ancestor = anchor.parentElement
      while (ancestor && ancestor !== document.body) {
        const draft = Array.from(
          ancestor.querySelectorAll<HTMLElement>(
            'textarea, [contenteditable="true"]'
          )
        ).find((candidate) => {
          const rect = candidate.getBoundingClientRect()
          const style = window.getComputedStyle(candidate)
          return (
            rect.width > 0 &&
            rect.height > 0 &&
            style.display !== "none" &&
            style.visibility !== "hidden"
          )
        })
        if (draft) {
          draftRect = draft.getBoundingClientRect()
          break
        }
        ancestor = ancestor.parentElement
      }
      const controlsRect = anchor
        .closest<HTMLElement>(
          '[data-testid="sidepanel-send-action-cluster"], [data-testid="composer-inline-send-control"]'
        )
        ?.getBoundingClientRect()
      overlay.style.maxWidth = `${Math.max(
        0,
        viewportRight - viewportLeft - inset * 2
      )}px`
      const overlayRect = overlay.getBoundingClientRect()
      const minimumTop = viewportTop + inset
      const maximumBottom = viewportBottom - inset
      const blockedRects = [draftRect, controlsRect].filter(
        (rect): rect is DOMRect =>
          Boolean(rect && rect.bottom > minimumTop && rect.top < maximumBottom)
      )
      const fitsAt = (top: number) =>
        top >= minimumTop &&
        top + overlayRect.height <= maximumBottom &&
        blockedRects.every(
          (rect) =>
            top + overlayRect.height <= rect.top - gap ||
            top >= rect.bottom + gap
        )
      const candidates = [
        draftRect ? draftRect.top - gap - overlayRect.height : null,
        controlsRect ? controlsRect.top - gap - overlayRect.height : null,
        draftRect ? draftRect.bottom + gap : null,
        controlsRect ? controlsRect.bottom + gap : null,
        minimumTop,
        maximumBottom - overlayRect.height
      ].filter((top): top is number => top !== null)
      const top = candidates.find(fitsAt)
      setPosition({
        left: Math.min(
          Math.max(viewportLeft + inset, anchorRect.right - overlayRect.width),
          Math.max(
            viewportLeft + inset,
            viewportRight - inset - overlayRect.width
          )
        ),
        top: top ?? minimumTop,
        visible: top !== undefined
      })
    }

    const schedulePosition = () => {
      if (animationFrame !== null) cancelAnimationFrame(animationFrame)
      animationFrame = requestAnimationFrame(() => {
        animationFrame = null
        updatePosition()
      })
    }

    updatePosition()
    window.addEventListener("resize", schedulePosition)
    window.addEventListener("scroll", schedulePosition, true)
    window.visualViewport?.addEventListener("resize", schedulePosition)
    window.visualViewport?.addEventListener("scroll", schedulePosition)
    return () => {
      window.removeEventListener("resize", schedulePosition)
      window.removeEventListener("scroll", schedulePosition, true)
      window.visualViewport?.removeEventListener("resize", schedulePosition)
      window.visualViewport?.removeEventListener("scroll", schedulePosition)
      if (animationFrame !== null) cancelAnimationFrame(animationFrame)
    }
  }, [anchorRef, portalTarget])

  if (!portalTarget) return null
  return createPortal(
    <div
      ref={overlayRef}
      style={{
        left: position?.left ?? 0,
        top: position?.top ?? 0,
        visibility: position?.visible ? undefined : "hidden"
      }}
      className="fixed z-50 flex w-max max-w-[calc(100vw-1rem)] flex-wrap items-center gap-2 rounded-lg border border-border bg-popover p-2 text-popover-foreground shadow-lg">
      {children}
    </div>,
    portalTarget
  )
}

export type PromptAssistComposerActionProps = {
  form: ComposerForm
  messageRevision: number
  promptAssistMutation: ComposerPromptAssistMutation
  promptAssistSavedAttemptId: number | null
  modelSelection: PromptImproveModelSelection | null
  promptAssistContextKey: string
  promptAssistBackendKey?: string | null
  promptAssistAuthorizationRevision?: string | null
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
  promptAssistAuthorizationRevision = null,
  sending = false,
  surfaceOpen = true,
  narrow = false,
  onSelectModel,
  onReturnFocus
}: PromptAssistComposerActionProps) {
  const { t } = useTranslation(["common"])
  const [panelOpen, setPanelOpen] = React.useState(false)
  const [inspectionOpen, setInspectionOpen] = React.useState(false)
  const [recipeOpen, setRecipeOpen] = React.useState(false)
  const [promptActionsOpen, setPromptActionsOpen] = React.useState(false)
  const [drawerPresented, setDrawerPresented] = React.useState(false)
  const { owner: recipeOwner, loading: recipeOwnerLoading } =
    useRecipePersistenceOwner(recipeOpen && surfaceOpen)
  const [authorizedRecipeOwner, setAuthorizedRecipeOwner] =
    React.useState<typeof recipeOwner>(null)
  const [recipeUndo, setRecipeUndo] = React.useState<{ draft: string } | null>(
    null
  )
  const normalizedBackendKey = promptAssistBackendKey?.trim() || null
  const normalizedAuthorizationRevision =
    promptAssistAuthorizationRevision?.trim() || null
  const modelSelectionRef = React.useRef(modelSelection)
  const controllerMutationRef = React.useRef<ControllerMutation | null>(null)
  const observedRevisionRef = React.useRef(promptAssistMutation.revision)
  const observedSavedAttemptRef = React.useRef(promptAssistSavedAttemptId)
  const pendingResetAttemptRef = React.useRef<number | null>(null)
  const pendingUndoFocusRef = React.useRef(false)
  const pendingDrawerFocusRef = React.useRef(false)
  const pendingRecipeTriggerFocusRef = React.useRef(false)
  const promptAssistTriggerRef = React.useRef<HTMLButtonElement>(null)
  const message = form.values.message
  const setFieldValue = form.setFieldValue
  modelSelectionRef.current = modelSelection

  const { data: promptCapabilities } = useQuery({
    queryKey: [
      "promptCapabilities",
      normalizedBackendKey,
      normalizedAuthorizationRevision
    ],
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
  const {
    data: resolvedRecipeCapabilities,
    isFetching: recipeCapabilitiesFetching,
    isError: recipeCapabilitiesError,
    refetch: refetchRecipeCapabilities
  } = useQuery({
    queryKey: [
      "promptCapabilities",
      recipeOwner?.ownerId ?? null,
      recipeOwner?.authorizationRevision ?? null
    ],
    queryFn: revalidatePromptCapabilities,
    enabled: false,
    retry: false
  })
  const recipeQueryClient = useQueryClient()
  React.useEffect(() => {
    if (!recipeOwner) return
    let current = true
    // An earlier open's initial fetch may still be in flight. Do not deduplicate
    // this open's authorization check onto that older request.
    void recipeQueryClient
      .cancelQueries({
        queryKey: [
          "promptCapabilities",
          recipeOwner.ownerId,
          recipeOwner.authorizationRevision
        ],
        exact: true
      })
      .then(async () => {
        if (!current) return
        const result = await refetchRecipeCapabilities()
        if (current && result.isSuccess) setAuthorizedRecipeOwner(recipeOwner)
      })
    return () => {
      current = false
    }
  }, [recipeOwner, recipeQueryClient, refetchRecipeCapabilities])
  const recipeCapabilities =
    recipeOwner &&
    !recipeOwnerLoading &&
    authorizedRecipeOwner === recipeOwner &&
    !recipeCapabilitiesFetching &&
    !recipeCapabilitiesError
      ? resolvedRecipeCapabilities
      : undefined

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
    pendingRecipeTriggerFocusRef.current = false
    setDrawerPresented(false)
    setRecipeOpen(false)
    setRecipeUndo(null)

    return () => {
      pendingDrawerFocusRef.current = false
      pendingRecipeTriggerFocusRef.current = false
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
    setRecipeUndo(null)
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
    setRecipeUndo(null)
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
    pendingRecipeTriggerFocusRef.current = recipeOpen
    setPanelOpen(false)
    setInspectionOpen(false)
    setRecipeOpen(false)
  }, [dismissPromptAssist, promptAssistState.status, recipeOpen])
  const handleDrawerAfterOpenChange = React.useCallback(
    (open: boolean) => {
      setDrawerPresented(open)
      if (open || !pendingDrawerFocusRef.current) return
      pendingDrawerFocusRef.current = false
      if (pendingRecipeTriggerFocusRef.current) {
        pendingRecipeTriggerFocusRef.current = false
        promptAssistTriggerRef.current?.focus()
        return
      }
      onReturnFocus?.()
    },
    [onReturnFocus]
  )
  const start = React.useCallback((operation: () => Promise<void>) => {
    setRecipeUndo(null)
    setPanelOpen(true)
    void operation()
  }, [])
  const undoAndReturnFocus = React.useCallback(() => {
    pendingUndoFocusRef.current = true
    promptAssist.undo()
  }, [promptAssist])
  const openRecipeBuilder = React.useCallback(() => {
    dismissPromptAssist()
    setPanelOpen(false)
    setInspectionOpen(false)
    pendingRecipeTriggerFocusRef.current = false
    setRecipeOpen(true)
  }, [dismissPromptAssist])
  const closeRecipeBuilder = React.useCallback(() => {
    pendingDrawerFocusRef.current = true
    pendingRecipeTriggerFocusRef.current = true
    setRecipeOpen(false)
  }, [])
  const applyRecipe = React.useCallback(
    (compiledText: string) => {
      setRecipeUndo({ draft: message })
      controllerMutationRef.current = {
        fromRevision: messageRevision,
        expectedValue: compiledText
      }
      setFieldValue("message", compiledText)
      pendingDrawerFocusRef.current = true
      setRecipeOpen(false)
    },
    [message, messageRevision, setFieldValue]
  )
  const undoRecipe = React.useCallback(() => {
    if (!recipeUndo) return
    controllerMutationRef.current = {
      fromRevision: messageRevision,
      expectedValue: recipeUndo.draft
    }
    setFieldValue("message", recipeUndo.draft)
    setRecipeUndo(null)
    onReturnFocus?.()
  }, [messageRevision, onReturnFocus, recipeUndo, setFieldValue])

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

  const drawerOpen =
    recipeOpen ||
    (panelOpen &&
      promptAssist.state.status !== "idle" &&
      (promptAssist.state.status !== "applied" || inspectionOpen))
  const feedbackVisible = !promptActionsOpen && !drawerOpen && !drawerPresented

  return (
    <div className="relative min-w-0">
      <PromptAssistMenu
        triggerRef={promptAssistTriggerRef}
        draft={form.values.message}
        capability={capability}
        modelSelection={modelSelection}
        onImproveNow={() => start(promptAssist.improveNow)}
        onReviewChanges={() => start(promptAssist.reviewChanges)}
        onBuildFromRecipe={openRecipeBuilder}
        onSelectModel={onSelectModel}
        onOpenChange={setPromptActionsOpen}
        disabled={sending || promptAssist.state.status === "analyzing"}
        compact
        placement="top"
      />

      {promptAssist.state.status === "applied" && feedbackVisible ? (
        <PromptAssistFeedbackOverlay anchorRef={promptAssistTriggerRef}>
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
        </PromptAssistFeedbackOverlay>
      ) : null}

      {recipeUndo && feedbackVisible ? (
        <PromptAssistFeedbackOverlay anchorRef={promptAssistTriggerRef}>
          <span role="status" className="text-xs text-muted-foreground">
            {t("common:promptAssist.recipeApplied", "Recipe applied.")}
          </span>
          <Button variant="outline" size="sm" onClick={undoRecipe}>
            {t("common:promptAssist.undoRecipe", "Undo recipe")}
          </Button>
        </PromptAssistFeedbackOverlay>
      ) : null}

      <Drawer
        placement="right"
        open={drawerOpen}
        onClose={closeDrawer}
        afterOpenChange={handleDrawerAfterOpenChange}
        focusable={{ focusTriggerAfterClose: false }}
        size={narrow ? "100vw" : 480}
        title={
          recipeOpen
            ? t("common:promptAssist.recipeTitle", "Build from recipe")
            : t("common:promptAssist.region", "Prompt improvement")
        }>
        <div
          onKeyDown={(event) => {
            if (recipeOpen && event.key === "Escape") {
              event.preventDefault()
              closeDrawer()
            }
            event.stopPropagation()
          }}>
          {recipeOpen ? (
            <React.Suspense
              fallback={
                <p role="status">
                  {t(
                    "common:promptAssist.recipeLoading",
                    "Loading recipe builder…"
                  )}
                </p>
              }>
              <PromptRecipeBuilder
                target="user_message"
                capabilities={recipeCapabilities}
                persistenceScope={recipeOwner?.ownerId ?? null}
                onApply={applyRecipe}
                onBack={closeRecipeBuilder}
              />
            </React.Suspense>
          ) : (
            panel
          )}
        </div>
      </Drawer>
    </div>
  )
}
