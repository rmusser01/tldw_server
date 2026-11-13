import React from "react"

/**
 * Name the browser tab for as long as something is on screen (an admin
 * module, a live voice chat), and return a release that gives back the title
 * it replaced. The release leaves the tab alone when someone else has named
 * it since: on a route change the route title hook (XP-19) names the new
 * route before the old page's effects are cleaned up, and restoring the old
 * title would undo that.
 */
export const holdDocumentTitle = (title: string): (() => void) => {
  if (typeof document === "undefined") return () => undefined
  const previous = document.title
  document.title = title
  return () => {
    if (document.title === title) document.title = previous
  }
}

/** Hold `title` (see `holdDocumentTitle`) while it is set; null holds nothing. */
export const useHeldDocumentTitle = (title: string | null) => {
  React.useEffect(() => (title ? holdDocumentTitle(title) : undefined), [title])
}
