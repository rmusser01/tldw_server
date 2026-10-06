import React from "react"
import { useLocation, useNavigate } from "react-router-dom"

import {
  consumeSidepanelChatHandoff,
  readSidepanelChatHandoff,
  type SidepanelChatHandoffPackage,
  type SidepanelChatHandoffPageContext,
} from "@/services/sidepanel-chat-handoff"

type SetMessageValue = (
  value: string,
  options?: { collapseLarge?: boolean; forceCollapse?: boolean },
) => void

type NotificationApi = {
  warning?: (args: { message: string; description?: string }) => void
}

export type SidepanelChatHandoffConflict = {
  handoff: SidepanelChatHandoffPackage
}

type UseSidepanelChatHandoffImportDeps = {
  draftValue: string
  getDraftRevision: () => number
  setMessageValue: SetMessageValue
  notificationApi?: NotificationApi
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
}

const appendImportedDraft = (currentDraft: string, importedDraft: string) => {
  if (!currentDraft) return importedDraft
  if (!importedDraft) return currentDraft
  return `${currentDraft}\n\n${importedDraft}`
}

export function useSidepanelChatHandoffImport({
  draftValue,
  getDraftRevision,
  setMessageValue,
  notificationApi,
  t,
}: UseSidepanelChatHandoffImportDeps) {
  const location = useLocation()
  const navigate = useNavigate()
  const [importedContext, setImportedContext] =
    React.useState<SidepanelChatHandoffPageContext | null>(null)
  const [conflict, setConflict] =
    React.useState<SidepanelChatHandoffConflict | null>(null)
  const [importPending, setImportPending] = React.useState(false)
  const [conflictFeedback, setConflictFeedback] = React.useState<string | null>(null)
  const pendingChoiceRef = React.useRef<{ location: typeof location; draft: string; revision: number } | null>(null)
  const locationRef = React.useRef(location)
  locationRef.current = location
  const draftValueRef = React.useRef(draftValue)
  draftValueRef.current = draftValue
  const handledHandoffIdsRef = React.useRef(new Set<string>())

  React.useLayoutEffect(() => {
    pendingChoiceRef.current = null
    setImportPending(false)
    setConflict(null)
    setConflictFeedback(null)
    return () => { pendingChoiceRef.current = null }
  }, [location.key, location.pathname, location.search, location.hash])

  const cleanRouteHandoff = React.useCallback(() => {
    const params = new URLSearchParams(location.search)
    if (!params.has("handoff")) return
    params.delete("handoff")
    const search = params.toString()
    navigate(
      {
        pathname: location.pathname,
        search: search ? `?${search}` : "",
        hash: location.hash,
      },
      { replace: true },
    )
  }, [location.hash, location.pathname, location.search, navigate])

  const showInvalidFeedback = React.useCallback(() => {
    notificationApi?.warning?.({
      message: t(
        "playground:sidepanelHandoff.unavailableTitle",
        "Sidepanel handoff unavailable",
      ),
      description: t(
        "playground:sidepanelHandoff.unavailableCopyDescription",
        "The copy expired or could not be read. Your destination draft remains here. Copy again from the sidepanel to try again.",
      ),
    })
  }, [notificationApi, t])

  const consumeQuietly = React.useCallback((handoffId: string) => {
    void consumeSidepanelChatHandoff(handoffId).catch(() => undefined)
  }, [])

  const applyImportedHandoff = React.useCallback(
    (
      handoff: SidepanelChatHandoffPackage,
      nextDraft: string,
      options?: { consume?: boolean },
    ) => {
      setMessageValue(nextDraft, { collapseLarge: true })
      setImportedContext(handoff.pageContext ?? null)
      setConflict(null)
      cleanRouteHandoff()
      handledHandoffIdsRef.current.add(handoff.id)
      if (options?.consume !== false) {
        queueMicrotask(() => consumeQuietly(handoff.id))
      }
    },
    [cleanRouteHandoff, consumeQuietly, setMessageValue],
  )

  React.useEffect(() => {
    const params = new URLSearchParams(location.search)
    const handoffId = params.get("handoff")
    if (!handoffId || handledHandoffIdsRef.current.has(handoffId)) return

    let cancelled = false

    void (async () => {
      const handoff = await readSidepanelChatHandoff(handoffId)
      if (cancelled) return

      if (!handoff) {
        handledHandoffIdsRef.current.add(handoffId)
        setConflict(null)
        showInvalidFeedback()
        cleanRouteHandoff()
        return
      }

      if (draftValueRef.current.length > 0) {
        handledHandoffIdsRef.current.add(handoff.id)
        setConflict({ handoff })
        return
      }

      applyImportedHandoff(handoff, handoff.draft.text)
    })()

    return () => {
      cancelled = true
    }
  }, [applyImportedHandoff, cleanRouteHandoff, location.search, showInvalidFeedback])

  const removeImportedContext = React.useCallback(() => {
    setImportedContext(null)
  }, [])

  const acceptHandoffDraft = React.useCallback(async (action: "insert" | "replace") => {
    if (!conflict || pendingChoiceRef.current) return
    const attempt = { location: locationRef.current, draft: draftValueRef.current, revision: getDraftRevision() }
    pendingChoiceRef.current = attempt
    setImportPending(true)
    setConflictFeedback(null)
    const isCurrent = () => pendingChoiceRef.current === attempt &&
      locationRef.current.key === attempt.location.key &&
      locationRef.current.pathname === attempt.location.pathname &&
      locationRef.current.search === attempt.location.search &&
      locationRef.current.hash === attempt.location.hash &&
      new URLSearchParams(locationRef.current.search).get("handoff") === conflict.handoff.id
    try {
      const handoff = await readSidepanelChatHandoff(conflict.handoff.id, { throwOnReadError: true })
      if (!isCurrent()) return
      if (!handoff) {
        setConflict(null)
        showInvalidFeedback()
        cleanRouteHandoff()
        return
      }
      if (getDraftRevision() !== attempt.revision || draftValueRef.current !== attempt.draft) {
        setConflictFeedback(t(
          "playground:sidepanelHandoff.draftChanged",
          "Your draft changed while the copy was checked. Review it, then choose Insert or Replace again.",
        ))
        return
      }
      applyImportedHandoff(handoff, action === "insert"
        ? appendImportedDraft(draftValueRef.current, handoff.draft.text)
        : handoff.draft.text)
    } catch {
      if (isCurrent()) setConflictFeedback(t(
        "playground:sidepanelHandoff.readFailed",
        "The copy could not be checked. Your draft remains here. Try Insert or Replace again.",
      ))
    } finally {
      if (pendingChoiceRef.current === attempt) {
        pendingChoiceRef.current = null
        setImportPending(false)
      }
    }
  }, [applyImportedHandoff, cleanRouteHandoff, conflict, getDraftRevision, showInvalidFeedback, t])

  const insertHandoffDraft = React.useCallback(
    () => acceptHandoffDraft("insert"), [acceptHandoffDraft],
  )
  const replaceWithHandoffDraft = React.useCallback(
    () => acceptHandoffDraft("replace"), [acceptHandoffDraft],
  )

  const cancelHandoffImport = React.useCallback(() => {
    if (!conflict) return
    pendingChoiceRef.current = null
    setImportPending(false)
    setConflictFeedback(null)
    setConflict(null)
    cleanRouteHandoff()
    handledHandoffIdsRef.current.add(conflict.handoff.id)
    consumeQuietly(conflict.handoff.id)
  }, [cleanRouteHandoff, conflict, consumeQuietly])

  return {
    importedContext,
    removeImportedContext,
    conflict,
    importPending,
    conflictFeedback,
    insertHandoffDraft,
    replaceWithHandoffDraft,
    cancelHandoffImport,
  }
}
