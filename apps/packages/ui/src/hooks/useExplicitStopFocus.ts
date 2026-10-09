import { useEffect, useLayoutEffect, useRef } from "react"
import { focusComposer } from "@/hooks/useComposerFocus"
import { usePageAssist } from "@/context"
import { useStoreMessageOption } from "@/store/option"
import { captureChatDraftSource } from "@/hooks/chat/chat-draft-transition"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { isHiddenBySelfOrAncestor } from "@/utils/focus-return"

/** Only an explicit UI Stop may return a removed focused control to the composer. */
export const useExplicitStopFocus = (
  stop: () => void,
  captureCurrent?: () => () => boolean
) => {
  const busy = useStoreMessageOption(
    (state) => state.isLoading || state.streaming || state.isProcessing
  )
  const selection = useHistorySelectionContext()
  const { controller, getController } = usePageAssist()
  const response = getController ? getController() : controller
  const fence = selection?.fence() ?? (() => true)
  const owner = selection?.owner
  const source = captureCurrent?.() ?? captureChatDraftSource().isCurrent
  const mounted = useRef(true)
  const pending = useRef<{
    isCurrent: () => boolean
    cancel: () => void
    finish: () => void
  } | null>(null)
  const responseIsCurrent = () => {
    const current = getController ? getController() : controller
    return current === response || current === null
  }
  const sourceIsCurrent = () =>
    mounted.current &&
    source() &&
    fence() &&
    owner?.kind !== "unavailable" &&
    owner?.validate_lease?.() !== false &&
    responseIsCurrent()
  useLayoutEffect(() => {
    const attempt = pending.current
    if (!attempt) return
    if (!attempt.isCurrent()) attempt.cancel()
    else if (!busy) attempt.finish()
  })
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      pending.current?.cancel()
    }
  }, [])
  return (event?: { currentTarget: EventTarget | null }) => {
    const activity = useStoreMessageOption.getState()
    if (
      !sourceIsCurrent() ||
      !busy ||
      !(activity.isLoading || activity.streaming || activity.isProcessing) ||
      response?.signal.aborted ||
      (getController && getController() !== response)
    )
      return
    pending.current?.cancel()
    const initiator = event?.currentTarget
    if (
      !(initiator instanceof HTMLButtonElement) ||
      document.activeElement !== initiator ||
      isHiddenBySelfOrAncestor(initiator)
    ) {
      stop()
      return
    }
    let frame: number | null = null
    let becameIdle = false
    const cancel = () => {
      if (frame !== null) cancelAnimationFrame(frame)
      document.removeEventListener("focusin", leaveFocus)
      document.removeEventListener("pointerdown", leaveFocus)
      document.removeEventListener("input", leaveFocus)
      unsubscribe()
      if (pending.current === attempt) pending.current = null
    }
    const leaveFocus = (event: Event) => {
      if (event.type === "input" || event.target !== initiator) cancel()
    }
    const unsubscribe = useStoreMessageOption.subscribe((state) => {
      const active = state.isLoading || state.streaming || state.isProcessing
      if (!sourceIsCurrent() || (becameIdle && active)) cancel()
      if (!active) becameIdle = true
    })
    const attempt = {
      isCurrent: () => pending.current === attempt && sourceIsCurrent(),
      cancel,
      finish: () => {
        if (frame !== null) return
        frame = requestAnimationFrame(() => {
          const owned =
            attempt.isCurrent() &&
            !initiator.isConnected &&
            document.activeElement === document.body &&
            document.visibilityState === "visible" &&
            !document.querySelector("form[data-chat-message-editor]") &&
            !Array.from(
              document.querySelectorAll<HTMLElement>(
                '[role="dialog"], [aria-modal="true"]'
              )
            ).some((dialog) => !isHiddenBySelfOrAncestor(dialog))
          cancel()
          if (owned) focusComposer()
        })
      }
    }
    pending.current = attempt
    document.addEventListener("focusin", leaveFocus)
    document.addEventListener("pointerdown", leaveFocus)
    document.addEventListener("input", leaveFocus)
    try {
      stop()
    } catch (error) {
      cancel()
      throw error
    }
  }
}
