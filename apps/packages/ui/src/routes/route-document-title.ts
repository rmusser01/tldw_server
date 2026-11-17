import { useSyncExternalStore } from "react"
import { getRouteMetadata } from "./route-metadata"

/**
 * Browser tab titles. Pages are named "<Page> | tldw" on the WebUI and in the
 * extension, and a page showing one item names it first:
 * "<note title> · Notes | tldw" (XP-05, XP-19).
 */
export const DOCUMENT_TITLE_SUFFIX = "tldw"

export const formatPageDocumentTitle = (pageLabel: string): string =>
  `${pageLabel} | ${DOCUMENT_TITLE_SUFFIX}`

/** "<Page> | tldw" for a known route, from the same labels the WebUI uses. */
export const getRouteDocumentTitle = (pathname: string): string | null => {
  const label = getRouteMetadata(pathname)?.label
  return label ? formatPageDocumentTitle(label) : null
}

export const formatNotesDocumentTitle = (noteTitle: string | null | undefined): string => {
  const trimmed = noteTitle?.trim() ?? ""
  return trimmed
    ? `${trimmed} · ${formatPageDocumentTitle("Notes")}`
    : formatPageDocumentTitle("Notes")
}

// The WebUI's Notes page owns its <title> through next/head, so the open
// note's title is published here for it to read.
let openNoteTitle: string | null = null
const listeners = new Set<() => void>()

export const publishOpenNoteTitle = (title: string | null) => {
  const next = title?.trim() || null
  if (next === openNoteTitle) return
  openNoteTitle = next
  listeners.forEach((listener) => listener())
}

const subscribe = (listener: () => void) => {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/** The open note's title, for the WebUI Notes page's <title>. */
export const useOpenNoteTitle = (): string | null =>
  useSyncExternalStore(subscribe, () => openNoteTitle, () => null)
