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
  setMessageValue: SetMessageValue
  notificationApi?: NotificationApi
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
}

const appendImportedDraft = (currentDraft: string, importedDraft: string) => {
  if (!currentDraft.trim()) return importedDraft
  if (!importedDraft.trim()) return currentDraft
  return `${currentDraft.trimEnd()}\n\n${importedDraft}`
}

export function useSidepanelChatHandoffImport({
  draftValue,
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
  const draftValueRef = React.useRef(draftValue)
  const handledHandoffIdsRef = React.useRef(new Set<string>())

  React.useEffect(() => {
    draftValueRef.current = draftValue
  }, [draftValue])

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
        "playground:sidepanelHandoff.unavailableDescription",
        "The imported sidepanel draft expired or could not be read. You can keep chatting normally.",
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

      if (draftValueRef.current.trim().length > 0) {
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

  const insertHandoffDraft = React.useCallback(() => {
    if (!conflict) return
    applyImportedHandoff(
      conflict.handoff,
      appendImportedDraft(draftValueRef.current, conflict.handoff.draft.text),
    )
  }, [applyImportedHandoff, conflict])

  const replaceWithHandoffDraft = React.useCallback(() => {
    if (!conflict) return
    applyImportedHandoff(conflict.handoff, conflict.handoff.draft.text)
  }, [applyImportedHandoff, conflict])

  const cancelHandoffImport = React.useCallback(() => {
    if (!conflict) return
    setConflict(null)
    cleanRouteHandoff()
    handledHandoffIdsRef.current.add(conflict.handoff.id)
    consumeQuietly(conflict.handoff.id)
  }, [cleanRouteHandoff, conflict, consumeQuietly])

  return {
    importedContext,
    removeImportedContext,
    conflict,
    insertHandoffDraft,
    replaceWithHandoffDraft,
    cancelHandoffImport,
  }
}
