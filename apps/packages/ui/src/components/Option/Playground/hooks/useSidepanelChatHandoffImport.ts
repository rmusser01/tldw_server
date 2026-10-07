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
  ready: boolean
  destinationKey: string
  captureGuard: () => () => boolean
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
  ready,
  destinationKey,
  captureGuard,
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
  const pendingChoiceRef = React.useRef<{ location: typeof location; draft: string; revision: number; isOwned: () => boolean } | null>(null)
  const locationRef = React.useRef(location)
  locationRef.current = location
  const draftValueRef = React.useRef(draftValue)
  draftValueRef.current = draftValue
  const handledHandoffIdsRef = React.useRef(new Map<string, { destinationKey: string; settled: boolean; review: boolean }>())
  const conflictOwnerRef = React.useRef<{ conflict: SidepanelChatHandoffConflict; isOwned: () => boolean } | null>(null)
  const readyRef = React.useRef(ready)
  readyRef.current = ready

  const retireConflict = React.useCallback(() => {
    const conflictId = conflictOwnerRef.current?.conflict.handoff.id
    const retired = conflictId ? handledHandoffIdsRef.current.get(conflictId) : undefined
    if (retired) { retired.settled = false; retired.review = true }
    pendingChoiceRef.current = null
    conflictOwnerRef.current = null
    setImportPending(false)
    setConflict(null)
    setConflictFeedback(null)
  }, [])

  React.useLayoutEffect(() => {
    retireConflict()
    setImportedContext(null)
  }, [destinationKey, ready, retireConflict])

  React.useLayoutEffect(() => {
    retireConflict()
    return retireConflict
  }, [location, retireConflict])

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

  const consumeQuietly = React.useCallback((handoffId: string, isOwned: () => boolean) => {
    void consumeSidepanelChatHandoff(handoffId, isOwned).catch(() => undefined)
  }, [])

  const applyImportedHandoff = React.useCallback(
    (
      handoff: SidepanelChatHandoffPackage,
      nextDraft: string,
      options: { isOwned: () => boolean },
    ) => {
      if (!readyRef.current || !options.isOwned()) return
      setMessageValue(nextDraft, { collapseLarge: true })
      setImportedContext(handoff.pageContext ?? null)
      setConflict(null)
      conflictOwnerRef.current = null
      cleanRouteHandoff()
      handledHandoffIdsRef.current.set(handoff.id, { destinationKey, settled: true, review: false })
      queueMicrotask(() => {
        if (options.isOwned()) consumeQuietly(handoff.id, options.isOwned)
      })
    },
    [cleanRouteHandoff, consumeQuietly, destinationKey, setMessageValue],
  )

  React.useEffect(() => {
    const params = new URLSearchParams(location.search)
    const handoffId = params.get("handoff")
    const previous = handoffId ? handledHandoffIdsRef.current.get(handoffId) : undefined
    if (!ready || location.pathname !== "/chat" || !handoffId || (previous?.destinationKey === destinationKey && previous.settled && (!conflictOwnerRef.current || conflictOwnerRef.current.isOwned()))) return
    const isOwned = captureGuard()
    if (!isOwned()) return
    if (conflictOwnerRef.current && !conflictOwnerRef.current.isOwned()) retireConflict()
    const attempt = { destinationKey, settled: false, review: previous?.review === true || (previous !== undefined && previous.destinationKey !== destinationKey) }
    handledHandoffIdsRef.current.set(handoffId, attempt)

    let cancelled = false

    void (async () => {
      const handoff = await readSidepanelChatHandoff(handoffId)
      if (cancelled || handledHandoffIdsRef.current.get(handoffId) !== attempt || !readyRef.current || !isOwned()) return
      attempt.settled = true

      if (!handoff) {
        setConflict(null)
        showInvalidFeedback()
        cleanRouteHandoff()
        return
      }

      if (attempt.review || draftValueRef.current.length > 0) {
        attempt.review = true
        const nextConflict = { handoff }
        conflictOwnerRef.current = { conflict: nextConflict, isOwned }
        setConflict(nextConflict)
        return
      }

      applyImportedHandoff(handoff, handoff.draft.text, { isOwned })
    })()

    return () => {
      cancelled = true
    }
  }, [applyImportedHandoff, captureGuard, cleanRouteHandoff, destinationKey, location, ready, retireConflict, showInvalidFeedback])

  const removeImportedContext = React.useCallback(() => {
    setImportedContext(null)
  }, [])

  const isConflictCurrent = React.useCallback(() => readyRef.current && conflict !== null &&
    conflictOwnerRef.current?.conflict === conflict && conflictOwnerRef.current.isOwned() &&
    new URLSearchParams(locationRef.current.search).get("handoff") === conflict.handoff.id, [conflict])

  const acceptHandoffDraft = React.useCallback(async (action: "insert" | "replace") => {
    if (!isConflictCurrent() || !conflict || pendingChoiceRef.current) return
    const attempt = { location: locationRef.current, draft: draftValueRef.current, revision: getDraftRevision(), isOwned: captureGuard() }
    if (!attempt.isOwned()) return
    pendingChoiceRef.current = attempt
    setImportPending(true)
    setConflictFeedback(null)
    const isCurrent = () => readyRef.current && attempt.isOwned() && pendingChoiceRef.current === attempt &&
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
        : handoff.draft.text, { isOwned: attempt.isOwned })
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
  }, [applyImportedHandoff, captureGuard, cleanRouteHandoff, conflict, getDraftRevision, isConflictCurrent, showInvalidFeedback, t])

  const insertHandoffDraft = React.useCallback(
    () => acceptHandoffDraft("insert"), [acceptHandoffDraft],
  )
  const replaceWithHandoffDraft = React.useCallback(
    () => acceptHandoffDraft("replace"), [acceptHandoffDraft],
  )

  const cancelHandoffImport = React.useCallback(() => {
    if (!isConflictCurrent() || !conflict) return
    const isOwned = conflictOwnerRef.current!.isOwned
    conflictOwnerRef.current = null
    pendingChoiceRef.current = null
    setImportPending(false)
    setConflictFeedback(null)
    setConflict(null)
    cleanRouteHandoff()
    handledHandoffIdsRef.current.set(conflict.handoff.id, { destinationKey, settled: true, review: false })
    consumeQuietly(conflict.handoff.id, isOwned)
  }, [cleanRouteHandoff, conflict, consumeQuietly, destinationKey, isConflictCurrent])

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
