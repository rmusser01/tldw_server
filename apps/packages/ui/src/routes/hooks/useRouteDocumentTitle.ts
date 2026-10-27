import React from "react"
import { useLocation } from "react-router-dom"
import { getRouteDocumentTitle } from "@/routes/route-document-title"

/**
 * Name the browser tab after the route on every page change (XP-19), so a
 * chat's title never sticks after leaving it. This runs before the page's
 * own effects, so a page can still name what it shows ("<note> · Notes |
 * tldw", a chat's title).
 */
export const useRouteDocumentTitle = (fallbackTitle: string) => {
  const { pathname } = useLocation()
  React.useLayoutEffect(() => {
    document.title = getRouteDocumentTitle(pathname) ?? fallbackTitle
  }, [fallbackTitle, pathname])
}
