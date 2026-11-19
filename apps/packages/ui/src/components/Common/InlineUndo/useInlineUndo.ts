import React from "react"

/**
 * One Undo offer after a soft delete (XP-17, AX-06; #3101 decision 8).
 *
 * A soft delete (Move to Trash) happens at once and offers Undo inline,
 * instead of asking first. Only permanent deletes ask for confirmation.
 */
export type InlineUndoOffer = {
  /** What was deleted. `dismiss(key)` retires the offer once that is back. */
  key: string
  /** What happened, for example `Moved "Plan" to Trash.` */
  message: string
  /**
   * Puts the deleted item back. Resolve `false` (or throw) when nothing was
   * restored: the offer stays, says so, and the user can try again or
   * dismiss it. Resolve a string to announce what was restored, for example
   * "Restored 1 of 2 notes." when part of a bulk Undo failed.
   */
  onUndo: () => Promise<UndoOutcome> | UndoOutcome
  /** Announced after a successful Undo that returned no message. */
  restoredMessage?: string
  /** A second action next to Undo, such as "View Trash". The offer closes after it. */
  secondaryAction?: { label: string; onClick: () => void }
  /**
   * The control that started the delete. Undo takes focus from it, or when
   * focus is nowhere, but not from anything the user moved to meanwhile.
   */
  initiator?: Element | null
}

export type UndoOutcome = boolean | string | void

export type InlineUndoShownOffer = InlineUndoOffer & { id: number }

export type InlineUndoController = {
  /** The offer on screen, or null. */
  offer: InlineUndoShownOffer | null
  /** Shows an offer. It replaces any older one and has no timer. Stable. */
  show: (offer: InlineUndoOffer) => void
  /** Closes the offer; with a key, only when the offer is for that key. Stable. */
  dismiss: (key?: string) => void
}

/**
 * State for one `InlineUndoBanner`. The offer lives as long as the component
 * that owns it, so it ends when the user dismisses it or leaves the page.
 */
export function useInlineUndo(): InlineUndoController {
  const [offer, setOffer] = React.useState<InlineUndoShownOffer | null>(null)
  const sequenceRef = React.useRef(0)

  const show = React.useCallback((next: InlineUndoOffer) => {
    sequenceRef.current += 1
    setOffer({ ...next, id: sequenceRef.current })
  }, [])

  const dismiss = React.useCallback((key?: string) => {
    setOffer((current) => {
      if (!current) return current
      if (key !== undefined && current.key !== key) return current
      return null
    })
  }, [])

  return React.useMemo(() => ({ offer, show, dismiss }), [dismiss, offer, show])
}
