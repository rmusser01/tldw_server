import React from "react"

/**
 * Says how many suggestions a composer menu lists (AX-12). It stays mounted
 * while the menu is closed, so the count is read when the menu opens or the
 * list changes. A polite region rather than role=status, so it does not
 * compete with a surface's own status notices.
 */
export const SuggestionCountStatus = ({
  open,
  text
}: {
  open: boolean
  text: string
}) => (
  <span
    aria-live="polite"
    aria-atomic="true"
    className="sr-only"
    data-testid="suggestion-count-status"
  >
    {open ? text : ""}
  </span>
)
