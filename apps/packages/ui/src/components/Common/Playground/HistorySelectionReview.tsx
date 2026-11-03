import { captureSavedComposerInput } from "@/services/chat-composer-settlement"
import { usePageAssist } from "@/context"
import { useStoreMessageOption } from "@/store/option"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { focusComposer } from "@/hooks/useComposerFocus"
import { classifyCharacterChatFailureRecovery } from "@/hooks/chat/useCharacterChatMode"
import { isHiddenBySelfOrAncestor } from "@/utils/focus-return"
import { Button } from "@/components/Common/Button"
import { openModelSelector } from "@/components/Option/Playground/playground-cockpit-actions"
import { RecoveryCallout, StatePanel } from "@/components/ui/state"
import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { keepHistoryTurnRecovery } from "@/services/history-turn-keep"
import { canRetryHistoryTurnRecovery, type HistoryRecoveryRetryContext } from "@/services/history-turn-retry"
import type { ChatSubmitResult } from "@/hooks/chat/chat-action-utils"
import type { HistoryCursorV1 } from "@/types/history-selection"
import {
  chronologicalHistoryNodes,
  hasReviewableHistory,
  partialHistoryCoverage,
  plainHistoryPreview
} from "./history-review-presentation"
import { useVirtualizer } from "@tanstack/react-virtual"
import React from "react"
import { useTranslation } from "react-i18next"

// Never expose arbitrary provider/transport exception text in recovery details.
const retryRefusalCodes = new Set([
  "retry_context_changed", "retry_prompt_unavailable", "retry_context_review_required",
  "history_retry_in_progress", "history_recovery_not_retryable", "history_selection_not_ready",
  "request_config_scope_changed", "owner_conversation_mismatch", "history_selection_changed",
  "history_recovery_changed", "invalid_history_retry", "history_retry_input_changed",
  "history_recovery_retry_not_submitted", "history_recovery_retry_failed"
])

const forkCopy = {
  prepared: [
    "Fork prepared",
    "This fork has not been sent. The app will not start another attempt automatically."
  ],
  dispatching: [
    "Fork in progress",
    "The copy outcome is not confirmed yet. The app will not start another attempt automatically. A connection failure may have left multiple server copies; inspect the server before starting a new copy."
  ],
  unknown: [
    "Fork status unknown",
    "A copy may have been created, but its outcome could not be confirmed. The app will not start another attempt automatically. A connection failure may have left multiple server copies; inspect the server before starting a new copy."
  ],
  partial: [
    "Fork incomplete",
    "A copy was created. Some content may be missing because copying could not be confirmed as complete. The app will not start another attempt automatically. A connection failure may have left multiple server copies; inspect the server before starting a new copy."
  ],
  completed: [
    "Fork saved",
    "The selected supported content was copied. You can inspect the saved copy."
  ],
  rejected: [
    "Fork not started",
    "No copy was created by this operation. Review the details before starting a new fork."
  ]
} as const

/** Complete-manifest review; virtualization affects presentation, never consent. */
/** A hidden list longer than this shows only its latest messages (CC-04). */
const HIDDEN_PREVIEW_LIMIT = 4
const HIDDEN_PREVIEW_SHOWN = 3

export function HistorySelectionReview({
  selection,
  onExpand,
  onBind,
  hideRecovery,
  getRetryContext,
  onRetryRecovery,
  getSavedQuestionReview,
  onRetrySavedQuestion,
  onSwitchModel = () => openModelSelector()
}: {
  selection: HistorySelectionController
  /**
   * Open the model picker, offered when a send failed because of its model
   * (CC-02). Defaults to the shared model-selector event, which the full page
   * and the side panel both answer.
   */
  onSwitchModel?: () => void
  onExpand?: () => void
  onBind?: () => void
  /** Leave out retained turns the transcript already shows or still handles (CS-04). */
  hideRecovery?: (
    entry: NonNullable<HistorySelectionController["recoveries"]>[number]
  ) => boolean
  getSavedQuestionReview?: () => Promise<import("@/services/history-turn-retry").SavedQuestionReview>
  onRetrySavedQuestion?: (review: import("@/services/history-turn-retry").SavedQuestionReview) => Promise<ChatSubmitResult>
  getRetryContext?: () => Promise<HistoryRecoveryRetryContext>
  onRetryRecovery?: (
    entry: NonNullable<HistorySelectionController["recoveries"]>[number],
    review: { contextReviewed: boolean; context: HistoryRecoveryRetryContext }
  ) => Promise<ChatSubmitResult>
}) {
  const { t } = useTranslation("playground")
  const text = (key: string, fallback: string) =>
    t(`historySelection.${key}`, fallback)
  // Fill {{name}} placeholders here, so counts read the same whatever t() does.
  const fill = (template: string, values: Record<string, string | number>) =>
    template.replace(/\{\{(\w+)\}\}/g, (match, name: string) =>
      name in values ? String(values[name]) : match)
  const roleLabel = (role: string) =>
    role === "user"
      ? text("roleUser", "You")
      : role === "assistant"
        ? text("roleAssistant", "Assistant")
        : role
  const savedSetup = useStoreMessageOption(
    (state) => state.serverChatRoleplaySetup
  )
  const savedChatId = useStoreMessageOption((state) => state.serverChatId)
  const sendInProgress = useStoreMessageOption((state) => Boolean(state.isLoading || state.streaming || state.isProcessing))
  const { setController } = usePageAssist()
  const [retrying, setRetrying] = React.useState<string | null>(null)
  const retryAbort = React.useRef<AbortController | null>(null)
  const [openingRecovery, setOpeningRecovery] = React.useState<string | null>(null)
  const [openRecoveryError, setOpenRecoveryError] = React.useState<string | null>(null)
  const renderedSource = captureChatDraftSource()
  const renderedSelectionIsCurrent = selection.fence?.() ?? (() => true)
  const openingRecoveryRef = React.useRef<{
    operationId: string
    owner: HistorySelectionController["owner"]
    view: HistorySelectionController["view"]
    sourceIsCurrent: () => boolean
  } | null>(null)
  const [recoveryAction, setRecoveryAction] = React.useState<{
    operationId: string
    action: "copy" | "dismiss"
    pending: boolean
    feedback?: string
    failed?: boolean
  } | null>(null)
  const recoveryActionRef = React.useRef<{
    pending: boolean
    isCurrent: () => boolean
    cancel: () => void
  } | null>(null)
  const latestAction = React.useRef<HTMLButtonElement>(null)
  const branchAction = React.useRef<HTMLButtonElement>(null)
  const refreshAction = React.useRef<HTMLButtonElement>(null)
  const recoveryFocus = React.useRef<(() => void) | null>(null)
  const mounted = React.useRef(true)
  React.useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      retryAbort.current?.abort()
      recoveryFocus.current?.()
      recoveryActionRef.current?.cancel()
    }
  }, [])
  const editorOpen = () =>
    !!document.querySelector("form[data-chat-message-editor]")
  const acknowledgeRecovery = async (
    action: "copy" | "dismiss",
    entry: HistorySelectionController["recoveries"][number],
    initiator: HTMLButtonElement
  ) => {
    const owner = selection.owner,
      view = selection.view
    const sourceIsCurrent = () => {
      const current = selection.getCurrent?.()
      return (
        mounted.current &&
        renderedSource.isCurrent() &&
        renderedSelectionIsCurrent() &&
        owner?.kind !== "unavailable" &&
        owner?.validate_lease?.() !== false &&
        (!current || (current.owner === owner && current.view === view))
      )
    }
    if (!sourceIsCurrent() || retryAbort.current) return
    const previous = recoveryActionRef.current
    if (previous?.pending && previous.isCurrent()) return
    previous?.cancel()
    let focusOwned = document.activeElement === initiator
    let frame: number | null = null
    const leaveFocus = (event: Event) => {
      if (event.type === "input" || event.target !== initiator) focusOwned = false
    }
    const cleanup = () => {
      if (frame !== null) cancelAnimationFrame(frame)
      document.removeEventListener("focusin", leaveFocus)
      document.removeEventListener("pointerdown", leaveFocus)
      document.removeEventListener("input", leaveFocus)
    }
    const attempt = {
      pending: true,
      isCurrent: () => recoveryActionRef.current === attempt && sourceIsCurrent(),
      cancel: () => {
        cleanup()
        if (recoveryActionRef.current === attempt)
          recoveryActionRef.current = null
      }
    }
    recoveryActionRef.current = attempt
    setRecoveryAction({
      operationId: entry.turn.operation_id,
      action,
      pending: true
    })
    if (action === "dismiss") {
      document.addEventListener("focusin", leaveFocus)
      document.addEventListener("pointerdown", leaveFocus)
      document.addEventListener("input", leaveFocus)
    }
    const failed = () =>
      text(
        action === "copy" ? "copyRecoveryFailed" : "dismissRecoveryFailed",
        action === "copy"
          ? "Could not copy recovered text. Select it in the details or try Copy again."
          : "The recovery could not be dismissed. Your text is still available; try again."
      )
    try {
      if (action === "copy")
        await navigator.clipboard.writeText(
          [...(entry.turn.retained_result_texts ?? []), entry.turn.result_text || entry.turn.input_text].join("\n\n")
        )
      else if (!(await selection.dismissRecovery(entry, attempt.isCurrent)))
        throw new Error("recovery_dismissal_not_committed")
      if (!attempt.isCurrent()) return
      attempt.pending = false
      setRecoveryAction({
        operationId: entry.turn.operation_id,
        action,
        pending: false,
        feedback: text(
          action === "copy"
            ? "copyRecoverySucceeded"
            : "dismissRecoverySucceeded",
          action === "copy" ? "Recovered text copied." : "Recovery dismissed."
        )
      })
      if (action === "dismiss")
        frame = requestAnimationFrame(() => {
          const owned =
            attempt.isCurrent() &&
            focusOwned &&
            !initiator.isConnected &&
            document.activeElement === document.body &&
            document.visibilityState === "visible" &&
            !editorOpen() &&
            !Array.from(
              document.querySelectorAll<HTMLElement>(
                '[role="dialog"], [aria-modal="true"]'
              )
            ).some((dialog) => !isHiddenBySelfOrAncestor(dialog))
          cleanup()
          if (owned) focusComposer()
        })
    } catch {
      if (attempt.isCurrent()) {
        attempt.pending = false
        setRecoveryAction({
          operationId: entry.turn.operation_id,
          action,
          pending: false,
          feedback: failed(),
          failed: true
        })
      }
    } finally {
      if (frame === null) cleanup()
      if (recoveryActionRef.current === attempt && !attempt.isCurrent()) {
        attempt.cancel()
        if (mounted.current) setRecoveryAction(null)
      }
    }
  }
  React.useEffect(() => {
    const attempt = recoveryActionRef.current
    if (attempt && !attempt.isCurrent()) {
      attempt.cancel()
      setRecoveryAction(null)
    }
  }, [selection, renderedSource])
  const recoverSendChoice = async (choice: "latest" | "branch" | "refresh") => {
    const origin = selection.view
    if (
      !mounted.current ||
      !origin ||
      !renderedSource.isCurrent() ||
      !renderedSelectionIsCurrent()
    )
      return
    if (editorOpen()) {
      setOpenRecoveryError(
        text(
          "finishEditBeforeOpening",
          "Finish or cancel your current message edit before opening the saved reply."
        )
      )
      return
    }
    setOpenRecoveryError(null)
    recoveryFocus.current?.()
    const initiator =
      choice === "latest" ? latestAction.current : choice === "branch" ? branchAction.current : refreshAction.current
    let focusOwned = Boolean(initiator && document.activeElement === initiator)
    let frame: number | null = null
    const leaveFocus = (event: Event) => {
      if (event.target !== initiator) focusOwned = false
    }
    const cancel = () => {
      if (frame !== null) cancelAnimationFrame(frame)
      document.removeEventListener("focusin", leaveFocus)
      document.removeEventListener("pointerdown", leaveFocus)
      if (recoveryFocus.current === cancel) recoveryFocus.current = null
    }
    recoveryFocus.current = cancel
    document.addEventListener("focusin", leaveFocus)
    document.addEventListener("pointerdown", leaveFocus)
    const onAccepted = (isCurrent: () => boolean) => {
      if (recoveryFocus.current !== cancel) return
      frame = requestAnimationFrame(() => {
        const owned =
          recoveryFocus.current === cancel &&
          mounted.current &&
          renderedSource.isCurrent() &&
          isCurrent() &&
          !editorOpen() &&
          document.visibilityState === "visible" &&
          focusOwned &&
          initiator &&
          (document.activeElement === initiator ||
            (!initiator.isConnected && document.activeElement === document.body))
        cancel()
        if (owned) {
          if (choice === "refresh") (trigger.current ?? refreshAction.current ?? errorRegion.current)?.focus()
          else focusComposer()
        }
      })
    }
    try {
      if (choice === "latest")
        await selection.loadLatest(origin, renderedSource.isCurrent, onAccepted)
      else if (choice === "refresh") await selection.refresh(onAccepted)
      else selection.branchHere(origin, renderedSource.isCurrent, onAccepted)
    } finally {
      if (frame === null) cancel()
    }
  }

  const openSavedReply = async (
    entry: HistorySelectionController["recoveries"][number]
  ) => {
    if (
      openingRecoveryRef.current || retryAbort.current ||
      !renderedSource.isCurrent() ||
      !renderedSelectionIsCurrent()
    )
      return
    if (editorOpen()) {
      setOpenRecoveryError(
        text(
          "finishEditBeforeOpening",
          "Finish or cancel your current message edit before opening the saved reply."
        )
      )
      return
    }
    const attempt = {
      operationId: entry.turn.operation_id,
      owner: selection.owner,
      view: selection.view,
      sourceIsCurrent: renderedSource.isCurrent
    }
    openingRecoveryRef.current = attempt
    const current = () =>
      mounted.current &&
      openingRecoveryRef.current === attempt &&
      attempt.sourceIsCurrent() &&
      !editorOpen()
    setOpeningRecovery(entry.turn.operation_id)
    setOpenRecoveryError(null)
    try {
      if (!(await selection.openRecoveryResult(entry, current)) && current())
        setOpenRecoveryError(
          text(
            "openSavedReplyFailed",
            "Could not open the saved reply. Your current view and saved reply are retained."
          )
        )
    } catch {
      if (current())
        setOpenRecoveryError(
          text(
            "openSavedReplyFailed",
            "Could not open the saved reply. Your current view and saved reply are retained."
          )
        )
    } finally {
      if (openingRecoveryRef.current === attempt) {
        openingRecoveryRef.current = null
        if (mounted.current) setOpeningRecovery(null)
      }
    }
  }
  React.useEffect(() => {
    const attempt = openingRecoveryRef.current
    if (
      attempt &&
      (!attempt.sourceIsCurrent() ||
        selection.owner !== attempt.owner ||
        selection.view !== attempt.view ||
        selection.status !== "ready")
    ) {
      openingRecoveryRef.current = null
      setOpeningRecovery(null)
      setOpenRecoveryError(null)
    }
  })
  const [savedQuestionReview, setSavedQuestionReview] = React.useState<import("@/services/history-turn-retry").SavedQuestionReview | null>(null)
  const [savedQuestionBusy, setSavedQuestionBusy] = React.useState(false)
  const [savedQuestionError, setSavedQuestionError] = React.useState(false)
  const [expanded, setExpanded] = React.useState(false)
  const [included, setIncluded] = React.useState<string[]>([])
  const [cursor, setCursor] = React.useState<HistoryCursorV1>({ kind: "empty" })
  const [focusedId, setFocusedId] = React.useState<string | null>(null)
  const [failedKeep, setFailedKeep] = React.useState<string | null>(null)
  const [retryReview, setRetryReview] = React.useState<{ operation: string; context: HistoryRecoveryRetryContext } | null>(null)
  const [failedRetry, setFailedRetry] = React.useState<string | null>(null)
  const [retryRefusal, setRetryRefusal] = React.useState<string | null>(null)
  const trigger = React.useRef<HTMLButtonElement>(null)
  // "Start fresh" moves focus to "Restore all", which is its undo (CC-04):
  // it returns to the choice Start fresh replaced, on the same branch.
  const restoreAction = React.useRef<HTMLButtonElement>(null)
  const coverageBlock = React.useRef<HTMLDivElement>(null)
  const focusRestoreAfterChoice = React.useRef(false)
  const [startFreshFrom, setStartFreshFrom] = React.useState<{
    owner: string | null
    cursor: HistoryCursorV1
  } | null>(null)
  const scroller = React.useRef<HTMLDivElement>(null)
  const errorRegion = React.useRef<HTMLDivElement>(null)
  const identity = React.useId()
  const reviewOwner =
    selection.owner && selection.owner.kind !== "unavailable"
      ? `${selection.owner.kind}:${selection.owner.owner_key}:${selection.owner.conversation_id}`
      : null
  const previousReviewOwner = React.useRef(reviewOwner)
  const [recoveryDisclosure, setRecoveryDisclosure] = React.useState<{
    owner: string | null
    open: boolean
  }>({ owner: null, open: false })
  const recoveries = (selection.recoveries ?? [])
    .filter((entry) => !hideRecovery?.(entry))
    .sort(
      (a, b) =>
        b.turn.created_at - a.turn.created_at ||
        a.turn.operation_id.localeCompare(b.turn.operation_id)
    )
  // Retained timestamps alone cannot identify an operation from this writer and view.
  const currentRecovery = recoveries.find((entry) => {
    const origin = entry.turn.origin_view,
      view = selection.view,
      scope = selection.bookmarkScope
    return (
      origin &&
      view &&
      scope &&
      entry.scope.profile_id === scope.profile_id &&
      entry.scope.client_session_id === scope.client_session_id &&
      origin.owner_key === view.owner_key &&
      origin.conversation_id === view.conversation_id &&
      origin.view_session_id === view.view_session_id &&
      origin.selection_revision === view.selection_revision &&
      JSON.stringify(origin.cursor) === JSON.stringify(view.cursor) &&
      JSON.stringify(origin.interpretation) ===
        JSON.stringify(view.interpretation)
    )
  })
  const olderRecoveries = recoveries.filter((entry) => entry !== currentRecovery)

  const sourceNodes = selection.capture?.snapshot.nodes
  // Owner order, but a reply never comes before its question (CC-04).
  const nodes = React.useMemo(
    () => chronologicalHistoryNodes(sourceNodes || []),
    [sourceNodes]
  )
  const rowId = (id: string) => `${identity}-${encodeURIComponent(id)}`
  const virtualizer = useVirtualizer({
    count: nodes.length,
    getScrollElement: () => scroller.current,
    estimateSize: () => 132,
    getItemKey: (index) => nodes[index].id,
    overscan: 3,
    initialRect: { width: 320, height: 480 }
  })
  const pending =
    selection.status === "pending" || selection.status === "pending_unknown"
  const stale =
    selection.status === "stale_selection" || selection.status === "error"
  // How much of the conversation the AI sees, when it isn't all of it (CC-04).
  const restorePreference =
    startFreshFrom?.owner === reviewOwner ? startFreshFrom.cursor : null
  const coverage = React.useMemo(
    () =>
      selection.status === "ready"
        ? partialHistoryCoverage(selection.capture, selection.view, restorePreference)
        : null,
    [selection.status, selection.capture, selection.view, restorePreference]
  )
  // The review is offered once there is something to choose between, or
  // something it must explain (CC-04): not for a single exchange.
  const reviewAvailable =
    nodes.length > 0 &&
    (expanded ||
      selection.status === "legacy_review_required" ||
      stale ||
      Boolean(coverage) ||
      recoveries.length > 0 ||
      hasReviewableHistory(nodes))
  React.useEffect(() => {
    if (!coverage || !focusRestoreAfterChoice.current) return
    focusRestoreAfterChoice.current = false
    // A disabled "Restore all" (a send in progress) can't take focus.
    if (restoreAction.current && !restoreAction.current.disabled) restoreAction.current.focus()
    else coverageBlock.current?.focus()
  }, [coverage])
  React.useEffect(() => {
    if (previousReviewOwner.current === reviewOwner) return
    previousReviewOwner.current = reviewOwner
    setRecoveryDisclosure({ owner: reviewOwner, open: false })
    setOpenRecoveryError(null)
    setExpanded(false)
    setIncluded([])
    setCursor({ kind: "empty" })
    setFocusedId(null)
    setFailedKeep(null)
    setStartFreshFrom(null)
  }, [reviewOwner])
  React.useEffect(() => {
    setRetryReview(null)
    setSavedQuestionReview(null)
    setSavedQuestionError(false)
    setFailedRetry(null)
    setRetryRefusal(null)
  }, [reviewOwner, selection.view?.view_session_id, selection.view?.selection_revision])
  React.useEffect(() => {
    if (stale && expanded && !selection.freshnessConflict)
      errorRegion.current?.focus()
  }, [stale, expanded, selection.freshnessConflict])

  const focusRow = (index: number) => {
    const node = nodes[Math.max(0, Math.min(index, nodes.length - 1))]
    if (!node) return
    setFocusedId(node.id)
    virtualizer.scrollToIndex(nodes.indexOf(node), { align: "auto" })
    requestAnimationFrame(() =>
      document.getElementById(rowId(node.id))?.focus()
    )
  }
  const beginReview = () => {
    // Start from the complete owner manifest, including offscreen alternatives.
    if (!expanded && included.length === 0) {
      const ids = nodes.map((node) => node.id)
      setIncluded(ids)
      setCursor(
        ids.length
          ? { kind: "after_message", message_id: ids[ids.length - 1] }
          : { kind: "empty" }
      )
    }
    setExpanded(true)
    setFocusedId(nodes[0]?.id ?? null)
  }
  const changeIncluded = (id: string) => {
    setIncluded((previous) =>
      previous.includes(id)
        ? previous.filter((item) => item !== id)
        : [...previous, id]
    )
    if (
      cursor.kind !== "empty" &&
      cursor.message_id === id &&
      included.includes(id)
    )
      setCursor({ kind: "empty" })
  }
  const move = (id: string, direction: -1 | 1) => {
    setIncluded((previous) => {
      const index = previous.indexOf(id)
      const destination = index + direction
      if (index < 0 || destination < 0 || destination >= previous.length)
        return previous
      const next = [...previous]
      ;[next[index], next[destination]] = [next[destination], next[index]]
      return next
    })
  }
  const retryRoleplay = async (
    entry: HistorySelectionController["recoveries"][number]
  ) => {
    if (
      retryAbort.current ||
      editorOpen() ||
      !renderedSource.isCurrent() ||
      !renderedSelectionIsCurrent() ||
      useStoreMessageOption.getState().streaming ||
      useStoreMessageOption.getState().isProcessing
    )
      return
    const acknowledgeInput = captureSavedComposerInput(entry.turn.input_text)
    const abort = new AbortController()
    retryAbort.current = abort
    const activity = {
      originalReply: useStoreMessageOption.getState().messages.at(-1)
    }
    useStoreMessageOption.setState({
      detachedHistoryResponse: activity,
      isProcessing: true,
      streaming: true
    })
    setController(abort)
    const originIsCurrent = selection.fence()
    const sourceIsCurrent = () =>
      mounted.current &&
      retryAbort.current === abort &&
      renderedSource.isCurrent() &&
      !editorOpen()
    setRetrying(entry.turn.operation_id)
    setOpenRecoveryError(null)
    try {
      const { sendSavedRoleplay } = await import(
        "@/hooks/chat/roleplay-resume-send"
      )
      await sendSavedRoleplay({
        controller: selection,
        signal: abort.signal,
        originIsCurrent,
        sourceIsCurrent,
        message: entry.turn.input_text,
        acknowledgeInput,
        retry: entry,
        serverChatId: entry.turn.conversation_id,
        setMessages: useStoreMessageOption.getState().setMessages
      })
    } catch (error) {
      if (sourceIsCurrent()) {
        const recovery = classifyCharacterChatFailureRecovery(error, t)
        setOpenRecoveryError(recovery.kind === "transient"
          ? text("roleplayRetryFailed", "The original request could not be confirmed. Its input and any reply are still retained here.")
          : recovery.message)
      }
    } finally {
      setController((current) => (current === abort ? null : current))
      if (useStoreMessageOption.getState().detachedHistoryResponse === activity)
        useStoreMessageOption.setState({
          detachedHistoryResponse: null,
          isProcessing: false,
          streaming: false
        })
      if (retryAbort.current === abort) {
        retryAbort.current = null
        if (mounted.current) setRetrying(null)
      }
    }
  }
  const renderRecovery = (
    entry: HistorySelectionController["recoveries"][number]
  ) => (
    <div
      key={entry.turn.operation_id}
      className="mb-3 rounded border p-3"
      role="status">
      {entry.turn.roleplay_resume ? (
        <>
          <p className="font-medium">{entry.turn.roleplay_resume.completion?.saved
            ? text("savedRoleplayReplyTitle", "Reply saved")
            : entry.turn.state === "refused" ? text("roleplayMessageRefusedTitle", "Message not added")
            : entry.turn.roleplay_resume.completion && entry.turn.state !== "unknown" ? text("roleplayReplyUnsavedTitle", "Reply not saved")
            : entry.turn.roleplay_resume.append ? text("roleplayReplyUnconfirmedTitle", "Reply save unconfirmed")
            : text("roleplayInputUnconfirmedTitle", "Message save unconfirmed")}</p>
          <p>{entry.turn.roleplay_resume.completion?.saved
            ? text("savedRoleplayReplyMessage", "This reply is saved in the original conversation. Open it when you are ready.")
            : entry.turn.state === "refused" ? text("roleplayMessageRefused", "The server refused this request before adding your message. Your input is retained here.")
            : entry.turn.roleplay_resume.completion && entry.turn.state !== "unknown" ? text("roleplayReplyUnsaved", "Your message is saved. The reply was generated but could not be saved. Copy it or explicitly retry the original request; earlier replies will remain available.")
            : entry.turn.roleplay_resume.append
              ? text("roleplayReplyUnconfirmedMessage", "Your message is saved. Any reply that arrived is retained here. Explicit Retry may generate another reply if none is already saved, including while an earlier request is still running. It does not add another user message.")
              : text("roleplayInputUnconfirmedMessage", "The server may have received this message. Explicit Retry may generate a reply if none is already saved. It keeps the original message identity and your retained input.")}</p>
        </>
      ) : (entry.turn.action === "edit_resend" ||
        entry.turn.action === "regenerate") &&
      entry.turn.outcome === "complete" &&
      entry.turn.settled_message_id ? (
        <>
          <p className="font-medium">
            {entry.turn.action === "regenerate"
              ? text("savedRegeneratedReplyTitle", "New reply saved")
              : text("savedEditedReplyTitle", "Edited reply saved")}
          </p>
          <p>
            {entry.turn.action === "regenerate"
              ? text(
                  "savedRegeneratedReplyMessage",
                  "The new reply is saved as an alternative to the original reply. Your original question is kept once. Open the saved reply when you are ready."
                )
              : text(
                  "savedEditedReplyMessage",
                  "Your edited question and its reply are saved as a separate branch. Open the saved reply when you are ready."
                )}
          </p>
        </>
      ) : entry.turn.persistence === 'server' &&
        entry.turn.action === undefined &&
        entry.turn.settled_message_id ? (
        <>
          <p className="font-medium">{text('savedNativeReplyTitle', 'Reply saved')}</p>
          <p>
            {text(
              'savedNativeReplyMessage',
              'The server saved this reply. Open it to review its status and continue from the saved history.'
            )}
          </p>
        </>
      ) : entry.turn.action === "regenerate" ? (
        <>
          <p className="font-medium">
            {entry.turn.state === "refused"
              ? text("regenerationRefusedTitle", "New reply refused")
              : text("regenerationUnknownTitle", "New reply unconfirmed")}
          </p>
          <p>
            {entry.turn.state === "refused"
              ? text(
                  "regenerationRefusedMessage",
                  "This saved chat uses unsupported context. Your original exchange and draft are kept. Review the saved setup before regenerating again."
                )
              : text(
                  "regenerationUnknownMessage",
                  "Your original exchange is kept. Any new text that arrived is retained here; the server may also have saved a new reply. Reload this chat before regenerating again. Recovery will not send it again automatically."
                )}
          </p>
        </>
      ) : entry.turn.persistence !== 'server' &&
        (entry.turn.outcome === 'stopped' || entry.turn.outcome === 'interrupted') ? (
        <>
          {/* A partial reply kept on this device only (CS-04, #3104). */}
          <p className="font-medium">
            {entry.turn.outcome === "stopped"
              ? text("keptReplyStoppedTitle", "Reply stopped")
              : text("keptReplyInterruptedTitle", "Reply interrupted")}
          </p>
          <p>
            {text(
              "keptReplyMessage",
              "Your question is saved. The part of the reply that arrived is kept on this device only; it was not saved to your server."
            )}
          </p>
        </>
      ) : entry.turn.state === "refused" ? (
        <>
          <p className="font-medium">
            {text("refusedTitle", "Message not added")}
          </p>
          <p>
            {text(
              "sendRefusedMessage",
              "Your original input is kept here; copy it and review the conversation before trying again. It will not be sent again automatically."
            )}
          </p>
        </>
      ) : (
        <>
          {/* The problem, then its cause, then what is kept (CC-02). */}
          <p className="font-medium">
            {entry.turn.state === "generated_unsaved"
              ? text("recoveryUnsavedReplyTitle", "Reply not saved")
              : entry.turn.persistence === "server" && entry.turn.admission
                ? text("recoveryUnconfirmedTitle", "Reply not confirmed")
                : text("recoveryTitle", "No reply to this message")}
          </p>
          {entry.turn.failure_reason && <p role="alert">{entry.turn.failure_reason}</p>}
          <p>
            {entry.turn.persistence === "server" && entry.turn.admission
              ? text(
                  "recoveryState.accepted_response_unknown",
                  "Your message was saved. Not known whether a reply was saved"
                )
              : text(`recoveryState.${entry.turn.state}`, entry.turn.state)}
          </p>
          <p>
            {text(
              entry.turn.persistence === 'server' ? 'nativeRecoveryMessage' : 'recoveryMessage',
              entry.turn.persistence === 'server'
                ? 'The server may have saved a reply. Any text that arrived is kept here. Reload this chat to check, and choose where to continue in Review conversation history before trying again. Nothing is sent again automatically.'
                : 'Your message is kept. Nothing is sent again unless you choose to retry.'
            )}
          </p>
          {entry.turn.failure_action === "open-model-selector" && (
            <Button
              variant="text"
              disabled={sendInProgress}
              onClick={() => {
                if (mounted.current && renderedSource.isCurrent() && renderedSelectionIsCurrent())
                  onSwitchModel()
              }}>
              {text("switchModel", "Switch model")}
            </Button>
          )}
        </>
      )}
      {entry.turn.recovery_storage === "memory" && <p role="alert">{text("recoveryStorageFull", "Local storage could not keep all recovered text. Some text is retained in this window only; copy it before closing or reloading.")}</p>}
      <details>
        <summary>
          {text("inspectRecovery", "Show what was sent and received")}
        </summary>
        <pre className="whitespace-pre-wrap break-words">
          {entry.turn.input_text}
        </pre>
        <pre className="whitespace-pre-wrap break-words">
          {entry.turn.result_text}
        </pre>
        {entry.turn.retained_result_texts?.map((value,index)=><pre key={index} className="whitespace-pre-wrap break-words">{value}</pre>)}
        {entry.turn.input_images
          ?.filter((image) =>
            /^data:image\/(?:png|jpeg|gif|webp|bmp|x-icon);base64,[A-Za-z0-9+/]*={0,2}$/.test(
              image
            )
          )
          .map((image, index) => (
            <a
              key={index}
              className="block text-primary underline"
              href={image}
              download={`recovered-image-${index + 1}`}
              onClick={(event) => {
                if (
                  !mounted.current ||
                  !renderedSource.isCurrent() ||
                  !renderedSelectionIsCurrent()
                )
                  event.preventDefault()
              }}>
              {text("downloadRecoveredImage", "Download retained image")}
            </a>
          ))}
      </details>
      {entry.turn.outcome === "complete" && entry.turn.existing_selection && !entry.turn.settled_message_id && (
        <Button disabled={sendInProgress || retrying !== null || openingRecovery !== null || recoveryAction?.pending}
          loading={openingRecovery === entry.turn.operation_id}
          onClick={async () => {
            if (!mounted.current || !renderedSource.isCurrent() || !renderedSelectionIsCurrent()) return
            const current = () => mounted.current && renderedSource.isCurrent() && renderedSelectionIsCurrent()
            setOpeningRecovery(entry.turn.operation_id)
            setFailedKeep(null)
            try {
              const result = await keepHistoryTurnRecovery(selection, entry)
              if (current() && result.status === "skipped") setFailedKeep(entry.turn.operation_id)
            } catch {
              if (current()) setFailedKeep(entry.turn.operation_id)
            } finally {
              if (mounted.current) setOpeningRecovery(value => value === entry.turn.operation_id ? null : value)
            }
          }}>
          {text("keepRecovery", "Keep recovered reply")}
        </Button>
      )}
      {failedKeep === entry.turn.operation_id && (
        <p role="alert">{text("keepRecoveryFailed", "Could not keep this reply. The original recovery is retained for review.")}</p>
      )}
      {onRetryRecovery && getRetryContext && canRetryHistoryTurnRecovery(entry) && (
        <>
          <Button disabled={sendInProgress || retrying !== null || openingRecovery !== null || recoveryAction?.pending} onClick={async () => {
            const fence = selection.fence()
                  const isCurrent = () => mounted.current && renderedSource.isCurrent() && renderedSelectionIsCurrent() && fence()
            setFailedRetry(null)
            setRetryRefusal(null)
            try {
              const context = await getRetryContext()
              if (isCurrent() && context.isCurrent())
                setRetryReview({ operation: entry.turn.operation_id, context })
            } catch {
              if (isCurrent()) setFailedRetry(entry.turn.operation_id)
            }
          }}>
            {text("reviewRetry", "Review retry")}
          </Button>
          {retryReview?.operation === entry.turn.operation_id && (
            <section aria-label={text("reviewRetry", "Review retry")} className="my-2 rounded border p-3">
              <p>{text("retryPromptReview", "This recovery cannot restore the original prompt settings. Review or reapply the original System/Prompt settings using the chat controls, then review this retry again. Retry uses the model and instructions shown below.")}</p>
              <p>{text("retrySavedQuestion", "Saved question")}</p>
              <pre className="whitespace-pre-wrap break-words">{entry.turn.input_text}</pre>
              <p>{text("retryCurrentModel", "Current model")}</p>
              <p>{retryReview.context.model}</p>
              <p>{text("retryCurrentPrompt", "Current System/Prompt instructions")}</p>
              {retryReview.context.systemPrompts.length
                ? retryReview.context.systemPrompts.map((prompt, index) => <pre key={index} className="whitespace-pre-wrap break-words">{prompt}</pre>)
                : <p>{text("retryNoPrompt", "No System/Prompt instruction is currently applied.")}</p>}
              <Button disabled={sendInProgress || retrying !== null || openingRecovery !== null || recoveryAction?.pending} onClick={async () => {
                setRetrying(entry.turn.operation_id)
                setFailedRetry(null)
                setRetryRefusal(null)
                try {
                  const result = await onRetryRecovery(entry, { contextReviewed: true, context: retryReview.context })
                  if (result.status !== "submitted") {
                    setFailedRetry(entry.turn.operation_id)
                    setRetryRefusal(result.status === "failed" && retryRefusalCodes.has(result.errorMessage || "") ? result.errorMessage! : null)
                  }
                  setRetryReview(null)
                } catch {
                  setFailedRetry(entry.turn.operation_id)
                } finally {
                  setRetrying(null)
                }
              }}>
                {text("retrySavedQuestionAction", "Retry saved question")}
              </Button>
              <Button disabled={sendInProgress || retrying !== null || openingRecovery !== null || recoveryAction?.pending} onClick={() => setRetryReview(null)}>
                {text("cancel", "Cancel review")}
              </Button>
            </section>
          )}
        </>
      )}
      {failedRetry === entry.turn.operation_id && (
        <div>
          <p role="alert">{text("retryRecoveryFailed", "Could not retry this question. Review the current chat and prompt settings; the original recovery is retained for inspection.")}</p>
          {retryRefusal && <details className="text-sm text-text-muted">
            <summary>{text("retryDetails", "Retry details")}</summary>
            <p>{retryRefusal}</p>
          </details>}
        </div>
      )}
      <Button
        disabled={retrying !== null || recoveryAction?.pending || openingRecovery !== null}
        loading={recoveryAction?.pending && recoveryAction.action === "copy" && recoveryAction.operationId === entry.turn.operation_id}
        onClick={(event) => void acknowledgeRecovery("copy", entry, event.currentTarget)}>
        {text("copyRecovery", "Copy recovered text")}
      </Button>
      {(((entry.turn.action === 'edit_resend' ||
        entry.turn.action === 'regenerate' ||
        entry.turn.roleplay_resume?.completion?.saved) &&
        entry.turn.outcome === 'complete') ||
        (entry.turn.persistence === 'server' &&
          entry.turn.action === undefined &&
          !entry.turn.roleplay_resume)) &&
        entry.turn.settled_message_id && (
          <Button
            disabled={retrying !== null || openingRecovery !== null || recoveryAction?.pending || selection.status !== "ready"}
            loading={openingRecovery === entry.turn.operation_id}
            onClick={() => void openSavedReply(entry)}>
            {text("openSavedReply", "Open saved reply")}
          </Button>
        )}
      {entry.turn.roleplay_resume && entry.turn.state !== "refused" && !entry.turn.roleplay_resume.completion?.saved && (
        <Button disabled={retrying !== null || openingRecovery !== null || recoveryAction?.pending || selection.status !== "ready"}
          loading={retrying === entry.turn.operation_id} onClick={() => void retryRoleplay(entry)}>
          {text("retryOriginalRoleplay", "Retry original request")}
        </Button>
      )}
      {retrying === entry.turn.operation_id && <Button onClick={() => retryAbort.current?.abort()}>
        {text("cancelRoleplayRetry", "Stop retry")}
      </Button>}
      <Button
        disabled={retrying !== null || openingRecovery !== null || recoveryAction?.pending}
        loading={recoveryAction?.pending && recoveryAction.action === "dismiss" && recoveryAction.operationId === entry.turn.operation_id}
        onClick={(event) => void acknowledgeRecovery("dismiss", entry, event.currentTarget)}>
        {text("dismissRecovery", "Dismiss recovery")}
      </Button>
    </div>
  )

  if (selection.status === "idle") return null
  return (
    <section
      className="w-full min-w-0 px-3 py-2"
      aria-label={text("region", "Conversation history selection")}>
      {savedSetup && savedChatId === selection.view?.conversation_id && (
        <p className="mb-2 text-sm text-text-muted">{text("savedRoleplaySetup", "Saved roleplay setup")}: {savedSetup.name}
          {savedSetup.provider && savedSetup.model ? ` · ${savedSetup.provider} / ${savedSetup.model}` : ""}.
          {" "}{text("savedRoleplaySetupHelp", "This conversation keeps its saved character and settings.")}</p>
      )}
      {getSavedQuestionReview && onRetrySavedQuestion && selection.status === "ready" &&
        selection.owner?.kind === "native" && selection.settingsQualified &&
        selection.capture?.status === "captured" && selection.capture.rows.at(-1)?.role === "user" && (
        <div className="mb-3 rounded border p-3">
          <Button disabled={savedQuestionBusy} onClick={async () => {
            setSavedQuestionBusy(true)
            setSavedQuestionError(false)
            try { setSavedQuestionReview(await getSavedQuestionReview()) }
            catch { setSavedQuestionError(true) }
            finally { setSavedQuestionBusy(false) }
          }}>{text("reviewSavedQuestion", "Review saved question")}</Button>
          {savedQuestionReview && <div role="group" aria-label={text("savedQuestionReview", "Saved question review")}>
            <p>{text("savedQuestionExplanation", "Answer this saved question using the selected model and saved character context. This does not send another copy of your question. Earlier uncertain attempts remain available for inspection.")}</p>
            <pre className="whitespace-pre-wrap break-words">{savedQuestionReview.text}</pre>
            <p>{savedQuestionReview.model}</p>
            <Button disabled={savedQuestionBusy} onClick={async () => {
              if (!savedQuestionReview.isCurrent()) { setSavedQuestionError(true); return }
              setSavedQuestionBusy(true)
              setSavedQuestionError(false)
              try {
                const result = await onRetrySavedQuestion(savedQuestionReview)
                if (result.status !== "submitted") setSavedQuestionError(true)
                else setSavedQuestionReview(null)
              } catch { setSavedQuestionError(true) }
              finally { setSavedQuestionBusy(false) }
            }}>{text("retrySavedQuestion", "Retry saved question")}</Button>
          </div>}
          {savedQuestionError && <p role="alert">{text("savedQuestionRefused", "Could not retry. Review the saved question again after checking the model, connection and conversation.")}</p>}
        </div>
      )}
      {selection.forkOperationsError && (
        <p role="alert">
          {text("forkLoadError", "Could not load retained fork outcomes.")}
        </p>
      )}
      {selection.forkOperations?.map((entry) => (
        <div
          key={entry.operation_id}
          role="status"
          className="mb-3 rounded border p-3">
          <p className="font-medium">
            {text(`forkStates.${entry.state}.title`, forkCopy[entry.state][0])}
          </p>
          <p>
            {text(
              `forkStates.${entry.state}.message`,
              forkCopy[entry.state][1]
            )}
          </p>
          <details className="text-sm text-text-muted">
            <summary>{text("forkDetails", "Operation details")}</summary>
            <p className="break-words">{entry.operation_id}</p>
            {entry.result && "code" in entry.result && (
              <p>{entry.result.code}</p>
            )}
          </details>
          {entry.candidate_child_id && (
            <Button onClick={() => void selection.inspectForkOperation(entry)}>
              {text("forkInspect", "Inspect copy")}: {entry.candidate_child_id}
            </Button>
          )}
          {entry.active_intent &&
            (entry.state === "prepared" ||
              entry.state === "unknown" ||
              entry.state === "partial") && (
              <Button onClick={() => void selection.allowNewFork(entry)}>
                {text("forkAllowNew", "Allow a new fork action")}
              </Button>
            )}
          {(entry.state === "prepared" ||
            entry.state === "unknown" ||
            entry.state === "partial") && (
            <p>
              {text(
                "forkNewExplanation",
                "Allowing a new action permits a separate copy and keeps this result. It does not retry this operation."
              )}
            </p>
          )}
        </div>
      ))}
      {selection.recoveryError && (
        <p role="alert">
          {text(
            "recoveryLoadError",
            "Could not load retained send outcomes. Refresh history to try again."
          )}
        </p>
      )}
      {openRecoveryError && <p role="alert">{openRecoveryError}</p>}
      {recoveryAction?.feedback && <p role={recoveryAction.failed ? "alert" : "status"}>{recoveryAction.feedback}</p>}
      {selection.status === "loading" && (
        <StatePanel
          state="loading"
          title={text("loading", "Loading selected history")}
          aria-live="polite"
        />
      )}
      {pending && (
        <StatePanel
          state="loading"
          title={text("pending", "History confirmation pending")}
          message={text(
            "pendingMessage",
            "The original choices are retained. Check the same confirmation before creating another interpretation."
          )}
          primaryAction={
            selection.status === "pending_unknown"
              ? {
                  label: text(
                    "checkConfirmation",
                    "Check original confirmation"
                  ),
                  onClick: () => void selection.confirm([], { kind: "empty" })
                }
              : undefined
          }
        />
      )}
      {selection.branchPending && (
        <div role="status">
          <p>
            {text(
              "branchPending",
              "Your next message will branch from this point. The existing conversation is kept."
            )}
          </p>
          <Button
            onClick={() => {
              if (
                mounted.current &&
                selection.view &&
                renderedSource.isCurrent() &&
                renderedSelectionIsCurrent()
              )
                void selection.choose(selection.view.cursor)
            }}>
            {text("cancelBranch", "Cancel branch")}
          </Button>
        </div>
      )}
      {selection.sendConflict ? (
        <div ref={errorRegion} tabIndex={-1}>
          <RecoveryCallout
            state="blocked"
            role={selection.freshnessConflict ? "status" : "alert"}
            title={text("sendConflictTitle", "Updated in another window or tab")}
            message={
              selection.freshnessConflict
                ? text(
                    "freshnessConflictMessage",
                    "This conversation changed. Your draft and the messages shown here are kept. Load the latest messages to continue, or branch from this point."
                  )
                : text(
                    "sendConflictMessage",
                    "Your message was not added. Load the latest messages to continue, or branch from the point shown here. Your input is kept."
                  )
            }
            primaryAction={{
              label: text("loadLatest", "Load latest messages"),
              ref: latestAction,
              onClick: () => { void recoverSendChoice("latest") }
            }}
            secondaryActions={[
              {
                label: text("branchHere", "Branch here"),
                ref: branchAction,
                onClick: () => { void recoverSendChoice("branch") }
              }
            ]}
          />
        </div>
      ) : (
        stale && (
          <div ref={errorRegion} tabIndex={-1}>
            <RecoveryCallout
              state="blocked"
              title={text("changed", "Conversation history changed")}
              role="alert"
              message={text(
                "changedMessage",
                "Your choices are retained. Refresh the source, then review missing messages or choose a new boundary."
              )}
              diagnostics={
                selection.error
                  ? [{ label: text("reason", "Reason"), value: selection.error }]
                  : undefined
              }
              primaryAction={{
                label: text("refresh", "Refresh history"),
                onClick: () => void selection.refresh()
              }}
            />
          </div>
        )
      )}
      {selection.status === "unsupported_history_capability" && (
        <RecoveryCallout
          state="unavailable"
          title={text("unavailable", "Selected history unavailable")}
          message={selection.error === "unbound_local_history" ? text("localBindingMessage", "This imported local conversation has no trusted owner. Use it on this browser to connect it to this browser profile, then review its history before continuing.") : text(
            "unavailableMessage",
            "You can still read this conversation. A supported, verified history owner is required to continue from a selection."
          )}
          diagnostics={
            selection.error
              ? [{ label: text("reason", "Reason"), value: selection.error }]
              : undefined
          }
          primaryAction={
            (selection.error === "unbound_server_mirror" || selection.error === "unbound_local_history") && selection.canBindUnavailable && onBind
              ? {
                  label: selection.error === "unbound_local_history" ? text("bindLocal", "Use this local conversation") : text("bind", "Connect this conversation"),
                  onClick: onBind
                }
              : undefined
          }
        />
      )}
      {currentRecovery && renderRecovery(currentRecovery)}
      {olderRecoveries.length > 0 && (
        <details
          open={
            recoveryDisclosure.owner === reviewOwner && recoveryDisclosure.open
          }
          onToggle={(event) => {
            if (
              mounted.current &&
              renderedSource.isCurrent() &&
              renderedSelectionIsCurrent()
            )
              setRecoveryDisclosure({
                owner: reviewOwner,
                open: event.currentTarget.open
              })
          }}>
          <summary className="cursor-pointer py-2 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus">
            {`${text("retainedOutcomes", "Earlier messages to review")} (${olderRecoveries.length})`}
          </summary>
          {olderRecoveries.map(renderRecovery)}
        </details>
      )}
      {coverage && (!sendInProgress || coverage.seen === 0) && (
        <div ref={coverageBlock} tabIndex={-1} className="mb-2 rounded border border-border p-2 text-sm focus:outline-none">
          <div className="flex flex-wrap items-center gap-2">
            <p role="status">
              {fill(text("coverage", "AI sees {{seen}} of {{total}} messages"), {
                seen: coverage.seen,
                total: coverage.total
              })}
            </p>
            <Button
              ref={restoreAction}
              variant="text"
              size="sm"
              disabled={
                sendInProgress ||
                pending ||
                retrying !== null ||
                openingRecovery !== null ||
                Boolean(recoveryAction?.pending)
              }
              onClick={() => {
                if (mounted.current && renderedSource.isCurrent() && renderedSelectionIsCurrent())
                  void selection.choose(coverage.restore)
              }}>
              {text("restoreAll", "Restore all")}
            </Button>
          </div>
          {coverage.seen === 0 && (
            <>
              {coverage.hidden.length > HIDDEN_PREVIEW_LIMIT && (
                <p className="mt-1 text-xs text-text-muted">
                  {fill(text("hiddenMore", "and {{count}} earlier messages"), {
                    count: (coverage.hidden.length - HIDDEN_PREVIEW_SHOWN).toLocaleString()
                  })}
                </p>
              )}
              {/* Shown for reference only: the AI doesn't see these (ADR-049
                  keeps them out of the transcript and its actions). */}
              <ul
                aria-label={text("hiddenEarlier", "Earlier messages, hidden from the AI")}
                className="mt-1 space-y-0.5 text-xs text-text-muted">
                {(coverage.hidden.length > HIDDEN_PREVIEW_LIMIT
                  ? coverage.hidden.slice(-HIDDEN_PREVIEW_SHOWN)
                  : coverage.hidden
                ).map((node) => (
                  <li key={node.id} className="truncate">
                    {`${roleLabel(node.role)} · ${
                      plainHistoryPreview(node.preview) ||
                      (node.assets?.length
                        ? text("imageOnly", "Image or attachment only")
                        : text("emptyMessage", "Empty message"))
                    }`}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {selection.status === "ready" && selection.owner?.kind === "native" &&
          selection.capture?.status === "captured" && selection.capture.view === selection.view &&
          selection.bookmarkScope && selection.view?.conversation_id === selection.owner.conversation_id &&
          nodes.length === 0 && recoveries.some(entry =>
            entry.turn.persistence === "server" && entry.turn.owner_key === selection.view?.owner_key &&
            entry.turn.conversation_id === selection.view?.conversation_id) && (
          <Button ref={refreshAction} variant="text" disabled={sendInProgress || Boolean(retrying || openingRecovery || recoveryAction?.pending)}
            onClick={() => {
              const state = useStoreMessageOption.getState()
              if (mounted.current && !state.isLoading && !state.streaming && !state.isProcessing && !editorOpen() &&
                renderedSource.isCurrent() && renderedSelectionIsCurrent() && selection.owner?.kind === "native" && selection.owner.validate_lease())
                void recoverSendChoice("refresh")
            }}>
            {text("refresh", "Refresh history")}
          </Button>
        )}
        {reviewAvailable && (
          <Button
            ref={trigger}
            variant="text"
            onClick={beginReview}
            disabled={pending}>
            {text("review", "Review conversation history")}
          </Button>
        )}
        {onExpand && (
          <Button variant="ghost" size="sm" onClick={onExpand}>
            {text("expand", "Expand in full page")}
          </Button>
        )}
      </div>
      {expanded && (
        <div className="mt-2 border-t border-border pt-3">
          <h3 className="text-base font-semibold">
            {text("review", "Review conversation history")}
          </h3>
          <p className="text-sm text-text-muted">
            {fill(text("includedCount", "{{included}} of {{total}} messages included"), {
              included: included.length,
              total: nodes.length
            })}
          </p>
          <p className="my-2 text-sm text-text-muted">
            {text(
              "instructions",
              "Tick the messages the AI should see, in the order it should see them. Arrow keys move between messages; Alt + arrow keys reorder them."
            )}
          </p>
          <Button
            variant="secondary"
            size="sm"
            disabled={pending}
            onClick={() => {
              // Hides every earlier message from the AI now; "Restore all" undoes it.
              const previous = selection.view?.cursor
              if (previous) setStartFreshFrom({ owner: reviewOwner, cursor: previous })
              focusRestoreAfterChoice.current = true
              setExpanded(false)
              void Promise.resolve(selection.choose({ kind: "empty" })).then(
                (chosen) => {
                  if (chosen === false) focusRestoreAfterChoice.current = false
                },
                () => {
                  focusRestoreAfterChoice.current = false
                }
              )
            }}>
            {text("startFresh", "Start fresh")}
          </Button>
          <div
            ref={scroller}
            className="max-h-[30rem] overflow-auto"
            style={{ height: Math.min(480, nodes.length * 132) }}
            role="list"
            aria-label={text("source", "All messages in this chat")}>
            <div
              style={{
                height: virtualizer.getTotalSize(),
                position: "relative"
              }}>
              {virtualizer.getVirtualItems().map((item) => {
                const node = nodes[item.index]
                const position = included.indexOf(node.id)
                const preview =
                  plainHistoryPreview(node.preview) ||
                  (node.assets?.length
                    ? text("imageOnly", "Image or attachment only")
                    : text("emptyMessage", "Empty message"))
                return (
                  <div
                    key={node.id}
                    id={rowId(node.id)}
                    data-history-message-id={node.id}
                    role="listitem"
                    aria-setsize={nodes.length}
                    aria-posinset={item.index + 1}
                    tabIndex={focusedId === node.id ? 0 : -1}
                    onFocus={() => setFocusedId(node.id)}
                    className="absolute left-0 top-0 w-full overflow-hidden border-b border-border p-2 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-focus"
                    style={{
                      transform: `translateY(${item.start}px)`,
                      minHeight: item.size
                    }}
                    onKeyDown={(event) => {
                      if (event.key !== "ArrowDown" && event.key !== "ArrowUp")
                        return
                      event.preventDefault()
                      const direction = event.key === "ArrowDown" ? 1 : -1
                      if (event.altKey) move(node.id, direction)
                      else focusRow(item.index + direction)
                    }}>
                    <label className="flex items-start gap-2">
                      <input
                        type="checkbox"
                        checked={position >= 0}
                        disabled={pending}
                        onChange={() => changeIncluded(node.id)}
                        aria-label={`${text("include", "Include")} ${roleLabel(node.role)}: ${preview}`}
                      />
                      <span className="line-clamp-2">
                        {`${roleLabel(node.role)} · ${preview}`}
                      </span>
                    </label>
                    <div className="mt-1 flex flex-wrap items-center gap-2">
                      <span className="text-text-muted">
                        {position < 0
                          ? text("omitted", "Hidden from the AI")
                          : fill(text("includedPosition", "AI sees this as message {{position}}"), {
                              position: position + 1
                            })}
                      </span>
                      {position >= 0 && (
                        <>
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={pending || position === 0}
                            onClick={() => move(node.id, -1)}
                            ariaLabel={`${text("up", "Move earlier")}: ${preview}`}>
                            {text("up", "Move earlier")}
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={
                              pending || position === included.length - 1
                            }
                            onClick={() => move(node.id, 1)}
                            ariaLabel={`${text("down", "Move later")}: ${preview}`}>
                            {text("down", "Move later")}
                          </Button>
                        </>
                      )}
                      <Button
                        size="sm"
                        variant="text"
                        disabled={pending}
                        onClick={() =>
                          void selection.choose({
                            kind: "after_message",
                            message_id: node.id
                          })
                        }>
                        {text("chooseBoundary", "Continue after this message")}
                      </Button>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
          <div className="my-3 flex flex-wrap items-center gap-2">
            <Button
              variant="secondary"
              disabled={pending || !included.length}
              onClick={() =>
                setCursor({ kind: "before_message", message_id: included[0] })
              }>
              {text("beforeFirst", "Before first included message")}
            </Button>
            <Button
              variant="secondary"
              disabled={pending}
              onClick={() => setCursor({ kind: "empty" })}>
              {text("empty", "Use no prior messages")}
            </Button>
            <Button
              variant="secondary"
              disabled={pending || !included.length}
              onClick={() =>
                setCursor({
                  kind: "after_message",
                  message_id: included[included.length - 1]
                })
              }>
              {text("throughLast", "Through last included message")}
            </Button>
          </div>
          <p className="my-2 text-sm text-text-muted">
            {text(
              "confirmEffect",
              "Confirming saves this choice for the chat. Nothing is deleted: messages you leave out are only hidden from the AI."
            )}
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={pending || selection.status === "loading"}
              onClick={() => void selection.confirm(included, cursor)}>
              {text("confirm", "Confirm selected history")}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setExpanded(false)
                trigger.current?.focus()
              }}>
              {text("cancel", "Cancel review")}
            </Button>
          </div>
        </div>
      )}
    </section>
  )
}
