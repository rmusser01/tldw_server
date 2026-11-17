import React from "react"

type TitleHold = { title: string; previous: string }

/** Titles held now, innermost last. */
const holds: TitleHold[] = []

/**
 * Name the browser tab for as long as something is on screen (an admin
 * module, a live voice chat), and return a release that gives back the title
 * it replaced. The release leaves the tab alone when someone else has named
 * it since: on a route change the route title hook (XP-19) names the new
 * route before the old page's effects are cleaned up, and restoring the old
 * title would undo that. Holds may be released in any order: one released
 * under another hands its saved title to the hold above it.
 */
export const holdDocumentTitle = (title: string): (() => void) => {
  if (typeof document === "undefined") return () => undefined
  const hold: TitleHold = { title, previous: document.title }
  holds.push(hold)
  document.title = title
  return () => {
    const index = holds.indexOf(hold)
    if (index < 0) return
    holds.splice(index, 1)
    if (index < holds.length) holds[index].previous = hold.previous
    else if (document.title === title) document.title = hold.previous
  }
}

/** Hold `title` (see `holdDocumentTitle`) while it is set; null holds nothing. */
export const useHeldDocumentTitle = (title: string | null) => {
  React.useEffect(() => (title ? holdDocumentTitle(title) : undefined), [title])
}
