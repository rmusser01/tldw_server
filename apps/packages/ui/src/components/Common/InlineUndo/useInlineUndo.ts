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
   * Puts the deleted item back. Resolve `false` (or throw) when the restore
   * failed, so the offer stays and the user can try again or dismiss it.
   */
  onUndo: () => Promise<boolean | void> | boolean | void
  /** Announced after a successful Undo. */
  restoredMessage?: string
  /** A second action next to Undo, such as "View Trash". */
  secondaryAction?: { label: string; onClick: () => void }
}

export type InlineUndoShownOffer = InlineUndoOffer & { id: number }

export type InlineUndoController = {
  /** The offer on screen, or null. */
  offer: InlineUndoShownOffer | null
  /** Shows an offer. It replaces any older one and has no timer. */
  show: (offer: InlineUndoOffer) => void
  /** Closes the offer; with a key, only when the offer is for that key. */
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
