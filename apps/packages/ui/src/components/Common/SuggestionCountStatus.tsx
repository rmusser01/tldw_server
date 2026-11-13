import React from "react"

/** How long the count waits for typing to settle before it is read. */
export const SUGGESTION_COUNT_DELAY_MS = 750

/**
 * Says how many suggestions a composer menu lists (AX-12). It stays mounted
 * while the menu is closed, so the count is read when the menu opens or the
 * list changes. The count waits for typing to settle, so each key press does
 * not queue an announcement; closing the menu empties it at once. A polite
 * region rather than role=status, so it does not compete with a surface's
 * own status notices.
 */
export const SuggestionCountStatus = ({
  open,
  text
}: {
  open: boolean
  text: string
}) => {
  const [announced, setAnnounced] = React.useState("")
  React.useEffect(() => {
    if (!open) {
      setAnnounced("")
      return
    }
    const timer = setTimeout(() => setAnnounced(text), SUGGESTION_COUNT_DELAY_MS)
    return () => clearTimeout(timer)
  }, [open, text])
  return (
    <span
      aria-live="polite"
      aria-atomic="true"
      className="sr-only"
      data-testid="suggestion-count-status"
    >
      {open ? announced : ""}
    </span>
  )
}
