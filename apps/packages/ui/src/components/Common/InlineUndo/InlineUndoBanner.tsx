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

const focusIsLost = (active: Element | null): boolean =>
  !active || active === document.body || !active.isConnected

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
 * left focus nowhere (or still on the control that started it), so keyboard
 * users can reach it at once. A failed Undo says so in the same status.
 */
export const InlineUndoBanner: React.FC<InlineUndoBannerProps> = ({
  undo,
  className,
  testId = "inline-undo",
  onReturnFocus
}) => {
  const { t } = useTranslation(["common"])
  const { offer, dismiss } = undo
  const descriptionId = React.useId()
  const containerRef = React.useRef<HTMLDivElement>(null)
  const undoButtonRef = React.useRef<HTMLButtonElement>(null)
  const offerIdRef = React.useRef<number | null>(null)
  offerIdRef.current = offer?.id ?? null
  const pendingRef = React.useRef<number | null>(null)
  const [pendingId, setPendingId] = React.useState<number | null>(null)
  const [announcement, setAnnouncement] = React.useState("")
  const [failedId, setFailedId] = React.useState<number | null>(null)
  const returnFocusRequestedRef = React.useRef(false)
  const offerId = offer?.id ?? null
  const initiatorRef = React.useRef<Element | null | undefined>(null)
  initiatorRef.current = offer?.initiator

  React.useEffect(() => {
    if (offerId == null) return
    setAnnouncement("")
    const focusUndo = () => undoButtonRef.current?.focus()
    const active = document.activeElement
    if (focusIsLost(active) || (initiatorRef.current && active === initiatorRef.current)) {
      focusUndo()
      return
    }
    // Focus stays where the user is. If that element leaves the page later
    // (its row is removed when the list refetches), Undo takes focus instead
    // of <body>.
    const watched = active as Element
    const observer = new MutationObserver(() => {
      if (watched.isConnected) return
      observer.disconnect()
      if (focusIsLost(document.activeElement)) focusUndo()
    })
    observer.observe(document.body, { childList: true, subtree: true })
    return () => observer.disconnect()
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
    setFailedId((current) => (current === id ? null : current))
    let outcome: Awaited<ReturnType<typeof onUndo>> = false
    try {
      outcome = await onUndo()
    } catch {
      outcome = false
    }
    if (pendingRef.current === id) pendingRef.current = null
    setPendingId((current) => (current === id ? null : current))
    // A newer offer, or the host's own follow-up offer, owns the banner now.
    if (offerIdRef.current !== id) return
    if (outcome === false) {
      setFailedId(id)
      return
    }
    setAnnouncement(
      typeof outcome === "string" && outcome
        ? outcome
        : restoredMessage ?? t("common:undo.restoredStatus", { defaultValue: "Restored." })
    )
    close()
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "Escape") return
    event.stopPropagation()
    close()
  }

  const pending = offer != null && pendingId === offer.id
  const failed = offer != null && failedId === offer.id

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
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className={offer ? "min-w-0 flex-1 break-words" : undefined}
      >
        {/* A new node per offer, so a repeated message is announced again. */}
        {offer ? (
          <span key={offer.id}>
            {offer.message}
            {failed && (
              <>
                {" "}
                <span className="font-medium text-danger">
                  {t("common:undo.failed", { defaultValue: "Couldn't restore. Try again." })}
                </span>
              </>
            )}
          </span>
        ) : (
          announcement
        )}
      </span>
      {offer && (
        <>
          {/* Undo's description: a quiet copy, so the live message isn't read twice. */}
          <span id={descriptionId} hidden>
            {offer.message}
          </span>
          <button
            ref={undoButtonRef}
            type="button"
            className={actionClass}
            aria-describedby={descriptionId}
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
              onClick={() => {
                offer.secondaryAction?.onClick()
                close()
              }}
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
