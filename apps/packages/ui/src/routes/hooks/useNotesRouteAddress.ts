import React from "react"
import { useLocation, useNavigate } from "react-router-dom"
import type { NotesSelectionReport } from "@/components/Notes/hooks/useNotesSelectionReport"
import { formatNotesDocumentTitle, publishOpenNoteTitle } from "@/routes/route-document-title"
import { updatePageTitle } from "@/utils/update-page-title"

/**
 * The Notes page's address names the open note: /notes?note=<id> (XP-05).
 * `source_ref_id` is still read, as links saved before XP-05 use it.
 */
export const NOTE_ROUTE_PARAM = "note"
export const LEGACY_NOTE_ROUTE_PARAM = "source_ref_id"
/**
 * `/notes?new=<any value>` asks the Notes page for a new note, then leaves the
 * address (XP-06). An address that also names a note opens that note instead.
 */
export const NEW_NOTE_ROUTE_PARAM = "new"
export const NEW_NOTE_PATH = `/notes?${NEW_NOTE_ROUTE_PARAM}=1`

let newNoteRequestCount = 0
/**
 * A new-note address unique to this request. The web build's router keys a
 * location by its address, so repeated requests must not share one.
 */
export const buildNewNotePath = (): string => {
  newNoteRequestCount += 1
  return `/notes?${NEW_NOTE_ROUTE_PARAM}=${Date.now().toString(36)}${newNoteRequestCount.toString(36)}`
}

const toSearchParams = (search: string | null | undefined) =>
  new URLSearchParams((search ?? "").replace(/^\?/, ""))

/** The new-note request an address carries, or null (none, or the address names a note). */
const readNewNoteRequest = (search: string | null | undefined): string | null => {
  const params = toSearchParams(search)
  const request = params.get(NEW_NOTE_ROUTE_PARAM)?.trim() ?? ""
  if (!request || params.get(NOTE_ROUTE_PARAM)?.trim() || params.get(LEGACY_NOTE_ROUTE_PARAM)?.trim()) return null
  return request
}

export const readNoteRouteId = (search: string | null | undefined): string | null => {
  const params = toSearchParams(search)
  const raw = params.get(NOTE_ROUTE_PARAM) ?? params.get(LEGACY_NOTE_ROUTE_PARAM)
  const trimmed = raw?.trim() ?? ""
  return trimmed || null
}

/** `search` naming `noteId` (or no note); a new-note request never carries over. */
export const withNoteRouteId = (search: string | null | undefined, noteId: string | null): string => {
  const params = toSearchParams(search)
  params.delete(LEGACY_NOTE_ROUTE_PARAM)
  params.delete(NEW_NOTE_ROUTE_PARAM)
  if (noteId) params.set(NOTE_ROUTE_PARAM, noteId)
  else params.delete(NOTE_ROUTE_PARAM)
  const next = params.toString()
  return next ? `?${next}` : ""
}

export const buildNotePath = (noteId: string): string => `/notes${withNoteRouteId("", noteId)}`

/**
 * Keep the Notes address, browser history and page title in step with the
 * open note.
 *
 * - Opening a note from another note adds a history entry, so Back reopens
 *   the previous note. Giving the page its first note (a link, a restore, a
 *   new note's first save) replaces the current entry instead.
 * - When the address changes (Back, Forward, a link) to a note other than
 *   the open one, `sourceNoteId` asks the page to open it.
 * - /notes without a note reopens the note opened last.
 * - /notes?new=<value> asks for a new note instead (`newNoteRequestKey`,
 *   once per navigation). `onNewNoteRequestHandled` then drops the request
 *   from the address without a history entry; if no draft was started (the
 *   user stayed on unsaved edits) the address names the open note again.
 */
export const useNotesRouteAddress = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const addressNoteId = readNoteRouteId(location.search)
  const locationRef = React.useRef(location)
  locationRef.current = location
  const openNoteIdRef = React.useRef<string | null>(null)
  const restoredRef = React.useRef(false)
  const handledLocationKeyRef = React.useRef<string | null>(null)
  const [sourceNoteId, setSourceNoteId] = React.useState<string | null>(addressNoteId)

  React.useEffect(() => {
    if (handledLocationKeyRef.current === location.key) return
    handledLocationKeyRef.current = location.key
    // The address naming the note already open (after this hook wrote it)
    // asks for nothing; any other note is a request to open it.
    setSourceNoteId(addressNoteId && addressNoteId !== openNoteIdRef.current ? addressNoteId : null)
  }, [addressNoteId, location.key])

  React.useEffect(() => () => publishOpenNoteTitle(null), [])

  const onNoteSelectionChange = React.useCallback(
    (report: NotesSelectionReport) => {
      const previousNoteId = openNoteIdRef.current
      openNoteIdRef.current = report.noteId
      const noteTitle = report.noteId ? report.title : null
      updatePageTitle(formatNotesDocumentTitle(noteTitle))
      publishOpenNoteTitle(noteTitle)

      const current = locationRef.current
      const address = readNoteRouteId(current.search)
      const go = (noteId: string | null, replace: boolean) =>
        navigate(
          {
            pathname: current.pathname,
            search: withNoteRouteId(current.search, noteId),
            hash: current.hash
          },
          replace ? { replace: true } : undefined
        )

      if (report.noteId) {
        if (report.noteId === address) {
          // A legacy source_ref_id link, or one that also asked for a new
          // note: name the note the current way.
          const params = toSearchParams(current.search)
          if (
            params.get(NOTE_ROUTE_PARAM) !== report.noteId ||
            params.has(LEGACY_NOTE_ROUTE_PARAM) ||
            params.has(NEW_NOTE_ROUTE_PARAM)
          ) {
            go(report.noteId, true)
          }
          return
        }
        go(report.noteId, previousNoteId === null)
        return
      }
      if (previousNoteId !== null) {
        // The note was closed (a new note, a deletion): the address follows.
        if (address) go(null, true)
        return
      }
      if (!address && !readNewNoteRequest(current.search) && report.lastOpenedNoteId && !restoredRef.current) {
        restoredRef.current = true
        go(report.lastOpenedNoteId, true)
      }
    },
    [navigate]
  )

  const onNewNoteRequestHandled = React.useCallback((started: boolean) => {
    // A new note replaces any restore of the note opened last.
    restoredRef.current = true
    const current = locationRef.current
    const params = toSearchParams(current.search)
    if (!params.has(NEW_NOTE_ROUTE_PARAM)) return
    params.delete(NEW_NOTE_ROUTE_PARAM)
    // No draft was started: the note still open keeps its address (XP-05).
    if (!started && openNoteIdRef.current && !params.get(NOTE_ROUTE_PARAM)) {
      params.set(NOTE_ROUTE_PARAM, openNoteIdRef.current)
    }
    const search = params.toString()
    navigate(
      { pathname: current.pathname, search: search ? `?${search}` : "", hash: current.hash },
      { replace: true }
    )
  }, [navigate])

  const newNoteRequest = readNewNoteRequest(location.search)
  const newNoteRequestKey = newNoteRequest ? `${location.key}|${newNoteRequest}` : null

  return { sourceNoteId, onNoteSelectionChange, newNoteRequestKey, onNewNoteRequestHandled }
}
