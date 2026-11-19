import React from "react"
import { X } from "lucide-react"
import { useTranslation } from "react-i18next"

import { cn } from "@/libs/utils"
import { focusRing } from "@/styles/tokens/focus"
import type { InlineUndoController } from "./useInlineUndo"

type InlineUndoBannerProps = {
  undo: InlineUndoController
  /** Placement for the visible banner, for example a floating position. */
  className?: string
  testId?: string
  /**
   * Moves focus somewhere sensible, such as the list, when the banner closes
   * while it holds focus. Without it focus would fall to the page body.
   */
  onReturnFocus?: () => void
}

const isTextEntry = (element: Element): boolean => {
  if (element instanceof HTMLTextAreaElement) return true
  if (element instanceof HTMLInputElement) {
    return !["button", "checkbox", "radio", "submit", "reset", "file", "range", "color"].includes(element.type)
  }
  return element instanceof HTMLElement && element.isContentEditable
}

const actionClass = cn(
  "inline-flex min-h-8 items-center rounded px-1.5 font-medium text-primary hover:underline aria-disabled:cursor-wait aria-disabled:opacity-70",
  focusRing.default
)

/**
 * The one Undo pattern for soft deletes (XP-17, AX-06; #3101 decision 8).
 *
 * The offer sits in the page, in reading order, instead of in a toast at the
 * end of the document. It has no timer: it stays until the user undoes or
 * dismisses it, a newer delete replaces it, or the page that owns it goes
 * away. Its message is a polite status, and Undo takes focus when the delete
 * left focus nowhere, so keyboard users can reach it at once.
 */
export const InlineUndoBanner: React.FC<InlineUndoBannerProps> = ({
  undo,
  className,
  testId = "inline-undo",
  onReturnFocus
}) => {
  const { t } = useTranslation(["common"])
  const { offer, dismiss } = undo
  const messageId = React.useId()
  const containerRef = React.useRef<HTMLDivElement>(null)
  const undoButtonRef = React.useRef<HTMLButtonElement>(null)
  const offerIdRef = React.useRef<number | null>(null)
  offerIdRef.current = offer?.id ?? null
  const pendingRef = React.useRef<number | null>(null)
  const [pendingId, setPendingId] = React.useState<number | null>(null)
  const [announcement, setAnnouncement] = React.useState("")
  const returnFocusRequestedRef = React.useRef(false)
  const offerId = offer?.id ?? null

  React.useEffect(() => {
    if (offerId == null) return
    setAnnouncement("")
    const active = document.activeElement
    // Someone typing keeps their place; anywhere else, Undo is next.
    if (active && active !== document.body && active.isConnected && isTextEntry(active)) return
    undoButtonRef.current?.focus()
  }, [offerId])

  const onReturnFocusRef = React.useRef(onReturnFocus)
  onReturnFocusRef.current = onReturnFocus
  React.useEffect(() => {
    if (offerId != null || !returnFocusRequestedRef.current) return
    returnFocusRequestedRef.current = false
    onReturnFocusRef.current?.()
  }, [offerId])

  const close = React.useCallback(() => {
    if (containerRef.current?.contains(document.activeElement)) {
      returnFocusRequestedRef.current = true
    }
    dismiss()
  }, [dismiss])

  const runUndo = async () => {
    if (!offer || pendingRef.current === offer.id) return
    const { id, onUndo, restoredMessage } = offer
    pendingRef.current = id
    setPendingId(id)
    let restored = false
    try {
      restored = (await onUndo()) !== false
    } catch {
      restored = false
    }
    if (pendingRef.current === id) pendingRef.current = null
    setPendingId((current) => (current === id ? null : current))
    if (!restored || offerIdRef.current !== id) return
    setAnnouncement(
      restoredMessage ?? t("common:undo.restoredStatus", { defaultValue: "Restored." })
    )
    close()
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "Escape") return
    event.stopPropagation()
    close()
  }

  const pending = offer != null && pendingId === offer.id

  return (
    <div
      ref={containerRef}
      data-testid={testId}
      className={
        offer
          ? cn(
              "flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text shadow-sm",
              className
            )
          : "sr-only"
      }
      onKeyDown={offer ? handleKeyDown : undefined}
    >
      <span
        id={messageId}
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className={offer ? "min-w-0 flex-1 break-words" : undefined}
      >
        {/* A new node per offer, so a repeated message is announced again. */}
        {offer ? <span key={offer.id}>{offer.message}</span> : announcement}
      </span>
      {offer && (
        <>
          <button
            ref={undoButtonRef}
            type="button"
            className={actionClass}
            aria-describedby={messageId}
            aria-disabled={pending || undefined}
            aria-busy={pending || undefined}
            onClick={() => {
              void runUndo()
            }}
            data-testid={`${testId}-undo`}
          >
            {t("common:undo.action", { defaultValue: "Undo" })}
          </button>
          {offer.secondaryAction && (
            <button
              type="button"
              className={actionClass}
              onClick={offer.secondaryAction.onClick}
              data-testid={`${testId}-secondary`}
            >
              {offer.secondaryAction.label}
            </button>
          )}
          <button
            type="button"
            className={cn(
              "inline-flex size-8 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text",
              focusRing.default
            )}
            aria-label={t("common:undo.dismiss", { defaultValue: "Dismiss" })}
            title={t("common:undo.dismiss", { defaultValue: "Dismiss" })}
            onClick={close}
            data-testid={`${testId}-dismiss`}
          >
            <X className="size-4" aria-hidden="true" />
          </button>
        </>
      )}
    </div>
  )
}

export default InlineUndoBanner
