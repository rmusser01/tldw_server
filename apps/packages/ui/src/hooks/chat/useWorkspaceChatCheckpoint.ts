import { createContext, useCallback, useContext, useLayoutEffect, useRef, useState, type Dispatch, type SetStateAction } from "react"
import type { useMessageOption } from "@/hooks/useMessageOption"
import { parseHistorySelectionHandoff, useHistorySelectionContext, type HistoryLoadReceipt, type HistorySelectionController } from "./useHistorySelection"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { resolveServicePromptScope, subscribeToServicePromptConfigChanges } from "@/services/service-prompts"
import { useWorkspaceStore, type WorkspaceChatSession, type WorkspaceChatSessionQualification } from "@/store/workspace"
import { buildWorkspaceChatSessionKey } from "@/store/workspace-chat-session-key"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { CHAT_ROUTE_REPLACEMENT_EVENT, normalizeCharacterChatSessionId } from "@/utils/character-chat-mode-intent"

export const WorkspaceChatRouteSearchContext = createContext<string | undefined>(undefined)

type CheckpointChat = Pick<ReturnType<typeof useMessageOption>,
  "messages" | "history" | "historyId" | "serverChatId" | "temporaryChat" |
  "setMessages" | "setHistory" | "setHistoryId" | "setServerChatId" |
  "stopStreamingRequest" | "setStreaming" | "setIsProcessing" | "setIsLoading"
>

type CheckpointOptions = {
  workspaceId: string | null | undefined
  workspaceReady: boolean
  draft: string
  setDraft: Dispatch<SetStateAction<string>>
  chat: CheckpointChat
  legacySessionKey?: string
  routeSearch?: string
  replaceRouteSearch?: (search: string) => void
}

type CheckpointLease = {
  key: string
  generation: number
  qualification: WorkspaceChatSessionQualification
  valid: boolean
  settled: boolean
  writable: boolean
  allowEmpty: boolean
  snapshot: WorkspaceChatSession | null
  pendingUndo?: () => boolean
}

const hasRouteIntent = (search: string) => {
  const params = new URLSearchParams(search)
  return ["historySelection", "chatId", "chat_id", "serverChatId", "server_chat_id", "historyId"].some(key => params.has(key))
}

const parseWorkspaceRoute = (search: string) => {
  const params = new URLSearchParams(search)
  const ids = ["chatId", "chat_id", "serverChatId", "server_chat_id"]
    .flatMap(key => params.getAll(key)).map(normalizeCharacterChatSessionId)
  if (ids.some(id => !id) || new Set(ids).size > 1 || params.has("historyId")) throw new Error("invalid_history_reference")
  if (params.has("historySelection")) {
    const reference = parseHistorySelectionHandoff(search)
    if (!reference || reference.owner_kind !== "native" || params.getAll("historySelection").length !== 1 ||
      (ids.length && ids[0] !== reference.conversation_id)) throw new Error("invalid_history_reference")
    return { serverChatId: reference.conversation_id, reference }
  }
  if (!ids[0]) throw new Error("invalid_history_reference")
  return { serverChatId: ids[0], reference: undefined }
}

const qualifiedReference = (control: HistorySelectionController, chat: Pick<CheckpointChat, "serverChatId" | "historyId">, qualification: WorkspaceChatSessionQualification) => {
  const current = control.getCurrent()
  const reference = control.getReference()
  const owner = current.owner
  if (!reference || !owner || owner.kind === "unavailable" ||
    current.status !== "ready" || current.capture?.status !== "captured" ||
    current.capture.snapshot.owner_key !== reference.owner_key ||
    current.capture.snapshot.conversation_id !== reference.conversation_id ||
    reference.conversation_id !== (chat.serverChatId ?? chat.historyId)) return null
  if (owner.kind === "native" && (!owner.validate_lease() ||
    owner.scope?.type !== "workspace" || owner.scope.workspaceId !== qualification.workspaceId ||
    serverChatMirrorOwnerKey({ requestScope: owner.request_scope }) !== qualification.ownerKey)) return null
  return reference
}

/** Workspace bookmarks/drafts only; the H1 controller owns all history authority. */
export function useWorkspaceChatCheckpoint(options: CheckpointOptions) {
  const control = useHistorySelectionContext()
  const active = control !== null
  const contextSearch = useContext(WorkspaceChatRouteSearchContext)
  const suppliedSearch = options.routeSearch ?? contextSearch
  const routeSearch = suppliedSearch ?? (typeof window === "undefined" ? "" : window.location.search)
  const route = useRef({ search: routeSearch, supplied: suppliedSearch !== undefined })
  route.current = { search: routeSearch, supplied: suppliedSearch !== undefined }
  const retiredRoute = useRef<string | null>(null)
  if (retiredRoute.current !== routeSearch) retiredRoute.current = null
  const [routeRevision, setRouteRevision] = useState(0)
  const { chat, draft, legacySessionKey } = options
  const referenceId = useWorkspaceStore(state => state.workspaceChatReferenceId)
  const hydrated = useWorkspaceStore(state => state.storeHydrated)
  const saveSession = useWorkspaceStore(state => state.saveWorkspaceChatSession)
  const getSession = useWorkspaceStore(state => state.getWorkspaceChatSession)
  const latest = useRef(options)
  latest.current = options
  const selection = useRef(control)
  selection.current = control
  const lease = useRef<CheckpointLease | null>(null)
  const clearRoute = useRef<{ from: string; to: string; origin: CheckpointLease; isCurrent: () => boolean } | null>(null)
  const legacy = useRef<{ key: string; snapshot: WorkspaceChatSession | null; skipCapture: boolean } | null>(null)
  const generation = useRef(0)
  const draftRevision = useRef(0)
  const draftScope = useRef<{ key: string; revision: number } | null>(null)
  const [scopeRevision, setScopeRevision] = useState(0)
  const [restoring, setRestoring] = useState(false)
  const [restoreError, setRestoreError] = useState<{ message: string; isCurrent: () => boolean } | null>(null)
  const workspaceId = options.workspaceId?.trim() || null
  const ready = Boolean(control && hydrated && options.workspaceReady && workspaceId && referenceId)
  const setDraft = useCallback<Dispatch<SetStateAction<string>>>((value) => {
    draftRevision.current += 1
    latest.current.setDraft(value)
  }, [])
  const clearView = useCallback((clearComposer = true) => {
    const { chat, setDraft: clearDraft } = latest.current
    chat.stopStreamingRequest()
    chat.setMessages([])
    chat.setHistory([])
    chat.setHistoryId(null, { preserveServerChatId: true })
    chat.setServerChatId(null)
    chat.setStreaming(false)
    chat.setIsProcessing(false)
    chat.setIsLoading(false)
    if (clearComposer) clearDraft("")
  }, [])
  const fence = useCallback(() => {
    if (!selection.current) return () => true
    const token = generation.current
    const owned = lease.current
    return () => {
      const state = useWorkspaceStore.getState()
      const owner = selection.current?.getCurrent().owner
      return Boolean(owned?.valid && owned.settled && token === generation.current &&
        state.storeHydrated && latest.current.workspaceReady &&
        latest.current.workspaceId?.trim() === owned.qualification.workspaceId &&
        state.workspaceId === owned.qualification.workspaceId &&
        state.workspaceChatReferenceId === owned.qualification.referenceId &&
        owner?.kind !== "unavailable" && (owner?.kind !== "native" || (
          owner.validate_lease() && owner.scope?.type === "workspace" &&
          owner.scope.workspaceId === owned.qualification.workspaceId &&
          serverChatMirrorOwnerKey({ requestScope: owner.request_scope }) === owned.qualification.ownerKey)))
    }
  }, [])

  const clearChat = useCallback(() => {
    const controller = selection.current
    const owned = lease.current
    const { chat, draft } = latest.current
    if (!controller || !owned?.writable || chat.temporaryChat || !fence()()) return null
    const reference = qualifiedReference(controller, chat, owned.qualification)
    if (!reference) return null
    const search = route.current.search
    if (hasRouteIntent(search) && !latest.current.replaceRouteSearch) return null
    const target = { historyId: chat.historyId, serverChatId: chat.serverChatId }
    controller.reset()
    const cleared = controller.fence()
    clearView(false)
    owned.allowEmpty = true
    owned.snapshot = {
      messages: [], history: [], historyId: null, serverChatId: null,
      checkpoint: { version: 1, ...owned.qualification, historySelectionReference: null, draft }
    }
    saveSession(owned.key, owned.snapshot)
    if (hasRouteIntent(search)) {
      const params = new URLSearchParams(search)
      for (const key of ["historySelection", "chatId", "chat_id", "serverChatId", "server_chat_id", "historyId"]) params.delete(key)
      const next = params.size ? `?${params}` : ""
      clearRoute.current = { from: search, to: next, origin: owned, isCurrent: cleared }
      latest.current.replaceRouteSearch!(next)
    }
    return async () => {
      const restoringLease = lease.current
      if (!cleared() || restoringLease !== owned ||
        restoringLease.generation !== generation.current || !fence()() || latest.current.chat.temporaryChat) return false
      let undoOwner: HistoryLoadReceipt["owner"] | null = null
      let undoFence: (() => boolean) | undefined
      let preparingFence: (() => boolean) | undefined
      let completed = false
      // H1's owner lease calls this predicate; keep it independent of that lease.
      const contextCurrent = () => {
        const state = useWorkspaceStore.getState()
        const owner = controller.getCurrent().owner
        return state.storeHydrated && latest.current.workspaceReady && !latest.current.chat.temporaryChat &&
          latest.current.workspaceId?.trim() === owned.qualification.workspaceId &&
          state.workspaceId === owned.qualification.workspaceId &&
          state.workspaceChatReferenceId === owned.qualification.referenceId &&
          (!undoOwner || owner === undoOwner) &&
          (owner?.kind !== "native" || (owner.conversation_id === reference.conversation_id &&
            serverChatMirrorOwnerKey({ requestScope: owner.request_scope }) === owned.qualification.ownerKey))
      }
      const isCurrent = () => lease.current === restoringLease && restoringLease.valid &&
        restoringLease.generation === generation.current && contextCurrent()
      const loadIsCurrent = () => {
        const owner = controller.getCurrent().owner
        if (!undoOwner && owner && (owner.kind === "unavailable" || owner.conversation_id === reference.conversation_id)) {
          undoOwner = owner
          undoFence = controller.fence()
        }
        return isCurrent()
      }
      const operationCurrent = () => (undoFence ?? preparingFence ?? cleared)()
      if (clearRoute.current?.origin === owned) clearRoute.current.isCurrent = () =>
        lease.current === owned && owned.valid && contextCurrent() && operationCurrent()
      owned.pendingUndo = () => isCurrent() && operationCurrent()
      setRestoring(true)
      setRestoreError(null)
      try {
        let receipt: HistoryLoadReceipt | undefined
        const loading = controller.loadConversation({ ...target, temporary: false,
          scope: { type: "workspace", workspaceId: owned.qualification.workspaceId }, isCurrent: loadIsCurrent
        }, reference, result => { receipt = result })
        preparingFence = controller.fence()
        const loaded = await loading
        const current = controller.getCurrent()
        if (!isCurrent() || !loaded || (undoFence && !undoFence())) return false
        if (!receipt || current.owner !== receipt.owner || current.view !== receipt.view ||
          !qualifiedReference(controller, target, owned.qualification)) return "failed"
        latest.current.chat.setHistoryId(target.historyId, { preserveServerChatId: true })
        latest.current.chat.setServerChatId(target.serverChatId)
        completed = true
        return true
      } catch {
        return isCurrent() && (!undoFence || undoFence()) ? "failed" : false
      } finally {
        if (!completed && isCurrent() && undoFence?.() && controller.getCurrent().owner === undoOwner) {
          controller.reset()
          clearView(false)
          undoOwner = null
          undoFence = controller.fence()
        }
        owned.pendingUndo = undefined
        if (lease.current === restoringLease && restoringLease.generation === generation.current) setRestoring(false)
      }
    }
  }, [clearView, fence, saveSession])

  useLayoutEffect(() => {
    if (!active) return
    const invalidateScope = () => {
      generation.current += 1
      if (lease.current) lease.current.valid = false
      selection.current?.reset()
      clearView()
      setRestoring(false)
      setRestoreError(null)
      setScopeRevision(value => value + 1)
    }
    const unsubscribe = subscribeToServicePromptConfigChanges(invalidateScope)
    window.addEventListener("tldw:auth-principal-changed", invalidateScope)
    return () => {
      unsubscribe()
      window.removeEventListener("tldw:auth-principal-changed", invalidateScope)
    }
  }, [active, clearView])

  useLayoutEffect(() => {
    const controller = selection.current
    if (!ready || !controller || !workspaceId || !referenceId) return
    if (retiredRoute.current === routeSearch) return
    const token = ++generation.current
    const request = new AbortController()
    const key = buildWorkspaceChatSessionKey(workspaceId, referenceId)
    const draftKey = JSON.stringify([key, scopeRevision, routeSearch, routeRevision])
    if (draftScope.current?.key !== draftKey) draftScope.current = { key: draftKey, revision: draftRevision.current }
    const revision = draftScope.current.revision
    const temporary = chat.temporaryChat
    const replacingClearRoute = clearRoute.current
    const continuingClear = Boolean(replacingClearRoute && replacingClearRoute.to === routeSearch &&
      replacingClearRoute.origin === lease.current && replacingClearRoute.origin.valid && replacingClearRoute.isCurrent() &&
      replacingClearRoute.origin.qualification.workspaceId === workspaceId &&
      replacingClearRoute.origin.qualification.referenceId === referenceId && !temporary)
    const intent = usePlaygroundSessionStore.getState()
    const explicitRoute = hasRouteIntent(routeSearch)
    const existing = controller.getCurrent()
    const liveWorkspace = existing.capture?.status === "captured" && existing.owner?.kind === "native" &&
      existing.owner.scope?.type === "workspace" && existing.owner.scope.workspaceId === workspaceId
    if (!continuingClear && (temporary || explicitRoute || !liveWorkspace)) {
      controller.reset()
      clearView(draftRevision.current === revision)
    }
    const beforeLoad = controller.fence()
    let owned: CheckpointLease | null = continuingClear ? replacingClearRoute!.origin : null
    if (owned) {
      // A qualified local clear consumes only its route; it needs no new history/scope read.
      owned.generation = token
      clearRoute.current = null
    }
    let capturedOwnerKey: string | null = null
    const matchesOwner = () => {
      const owner = controller.getCurrent().owner
      return !capturedOwnerKey || owner?.kind !== "native" ||
        serverChatMirrorOwnerKey({ requestScope: owner.request_scope }) === capturedOwnerKey
    }
    const isCurrent = () => token === generation.current && !request.signal.aborted &&
      matchesOwner() &&
      latest.current.chat.temporaryChat === temporary &&
      (route.current.supplied ? route.current.search : window.location.search) === routeSearch &&
      latest.current.workspaceId?.trim() === workspaceId &&
      useWorkspaceStore.getState().workspaceId === workspaceId &&
      useWorkspaceStore.getState().workspaceChatReferenceId === referenceId &&
      useWorkspaceStore.getState().storeHydrated &&
      usePlaygroundSessionStore.getState().restoreRevision === intent.restoreRevision &&
      usePlaygroundSessionStore.getState().serverChatSelectionIntent === intent.serverChatSelectionIntent
    const routeChanged = (event: Event) => {
      if (event.type === CHAT_ROUTE_REPLACEMENT_EVENT) retiredRoute.current = routeSearch
      generation.current += 1
      request.abort()
      controller.beginLoad()
      setRestoring(false)
      setRestoreError(null)
      if (event.type === "popstate") setRouteRevision(value => value + 1)
    }
    window.addEventListener(CHAT_ROUTE_REPLACEMENT_EVENT, routeChanged)
    window.addEventListener("popstate", routeChanged)
    if (!continuingClear) setRestoring(true)
    setRestoreError(null)
    if (!continuingClear) void (async () => {
      try {
        const scope = await resolveServicePromptScope({ signal: request.signal })
        if (!isCurrent() || !scope.clientPrincipalVerified) return
        const qualification = { ownerKey: serverChatMirrorOwnerKey({ requestScope: scope }), workspaceId, referenceId }
        const stored = getSession(key, qualification)
        // Rejected exact records are retained, never adopted or overwritten by an empty view.
        owned = { key, generation: token, qualification, valid: true, settled: false,
          writable: Boolean(stored) || !useWorkspaceStore.getState().workspaceChatSessions[key], allowEmpty: !explicitRoute, snapshot: null }
        lease.current = owned
        if (explicitRoute) {
          let target: ReturnType<typeof parseWorkspaceRoute>
          try { target = parseWorkspaceRoute(routeSearch) }
          catch {
            owned.writable = false
            await controller.open({ kind: "unavailable", code: "invalid_history_reference" })
            return
          }
          if (!beforeLoad()) return
          controller.activate(`workspace-checkpoint:${qualification.ownerKey}:${key}`)
          controller.reset()
          capturedOwnerKey = qualification.ownerKey
          let receipt: HistoryLoadReceipt | undefined
          const loaded = await controller.loadConversation({
            serverChatId: target.serverChatId, temporary, scope: { type: "workspace", workspaceId }, isCurrent
          }, target.reference, result => { receipt = result })
          if (!isCurrent() || !loaded || !receipt) return
          const current = controller.getCurrent()
          if (current.owner !== receipt.owner || current.view !== receipt.view ||
            current.capture?.status !== "captured" || receipt.owner.kind !== "native" ||
            !receipt.owner.validate_lease() || receipt.owner.conversation_id !== target.serverChatId ||
            receipt.owner.scope?.type !== "workspace" || receipt.owner.scope.workspaceId !== workspaceId ||
            serverChatMirrorOwnerKey({ requestScope: receipt.owner.request_scope }) !== qualification.ownerKey) return
          const savedReference = stored?.checkpoint?.historySelectionReference
          const currentReference = controller.getReference()
          if (savedReference && currentReference &&
            savedReference.profile_id === currentReference.profile_id &&
            savedReference.owner_key === currentReference.owner_key &&
            savedReference.conversation_id === currentReference.conversation_id &&
            draftRevision.current === revision) latest.current.setDraft(stored!.checkpoint!.draft)
          latest.current.chat.setHistoryId(null, { preserveServerChatId: true })
          latest.current.chat.setServerChatId(target.serverChatId)
          owned.settled = true
          return
        }
        if (!temporary && qualifiedReference(controller, latest.current.chat, qualification)) {
          owned.settled = true
          return
        }
        if (!beforeLoad()) return
        controller.activate(`workspace-checkpoint:${qualification.ownerKey}:${key}`)
        controller.reset()
        capturedOwnerKey = qualification.ownerKey
        clearView(draftRevision.current === revision)
        if (!stored) { owned.settled = true; return }
        const reference = stored.checkpoint!.historySelectionReference
        if (reference) {
          let receipt: HistoryLoadReceipt | undefined
          const loaded = await controller.loadConversation({
            historyId: stored.historyId, serverChatId: stored.serverChatId,
            temporary, scope: { type: "workspace", workspaceId }, isCurrent
          }, reference, result => { receipt = result })
          if (!isCurrent() || !loaded || !receipt) return
          const current = controller.getCurrent()
          if (current.owner !== receipt.owner || current.view !== receipt.view ||
            current.capture?.status !== "captured" ||
            (receipt.owner.kind === "native" && (!receipt.owner.validate_lease() ||
              serverChatMirrorOwnerKey({ requestScope: receipt.owner.request_scope }) !== qualification.ownerKey))) return
          latest.current.chat.setHistoryId(stored.historyId, { preserveServerChatId: true })
          latest.current.chat.setServerChatId(stored.serverChatId)
        }
        if (!isCurrent()) return
        if (draftRevision.current === revision) latest.current.setDraft(stored.checkpoint!.draft)
        owned.settled = true
      } catch (error) {
        // Keep a rejected checkpoint intact; a failed read never grants save authority.
        const cancelled = typeof error === "object" && error !== null && "name" in error && error.name === "AbortError"
        if (isCurrent() && !cancelled && (capturedOwnerKey !== null || beforeLoad())) {
          const failedCapture = controller.getCurrent().capture
          const errorFence = controller.fence()
          setRestoreError({ message: "Workspace chat restoration failed", isCurrent: () => {
            const current = controller.getCurrent()
            return errorFence() || current.capture === failedCapture || current.owner?.kind !== "native" ||
              !qualifiedReference(controller, latest.current.chat, {
                ownerKey: serverChatMirrorOwnerKey({ requestScope: current.owner.request_scope }), workspaceId, referenceId
              })
          } })
        }
      } finally {
        if (token === generation.current) setRestoring(false)
      }
    })()
    return () => {
      // Use only the last qualified copy, never the incoming surface's current rows.
      if (owned?.valid && owned.snapshot && useWorkspaceStore.getState().storeHydrated) saveSession(owned.key, owned.snapshot)
      const replacement = clearRoute.current
      const consumingClear = owned?.valid && replacement?.origin === owned && replacement.from === routeSearch &&
        replacement.to === route.current.search && replacement.isCurrent() && !latest.current.chat.temporaryChat &&
        latest.current.workspaceId?.trim() === owned.qualification.workspaceId &&
        useWorkspaceStore.getState().workspaceChatReferenceId === owned.qualification.referenceId
      if (lease.current === owned && !consumingClear) lease.current = null
      generation.current += 1
      request.abort()
      if (!consumingClear) controller.beginLoad()
      window.removeEventListener(CHAT_ROUTE_REPLACEMENT_EVENT, routeChanged)
      window.removeEventListener("popstate", routeChanged)
    }
  }, [ready, workspaceId, referenceId, scopeRevision, routeSearch, routeRevision, chat.temporaryChat, control?.activate, clearView, getSession, saveSession])

  useLayoutEffect(() => {
    const owned = lease.current
    if (!control || !owned?.valid || owned.generation !== generation.current || !owned.writable ||
      workspaceId !== owned.qualification.workspaceId || referenceId !== owned.qualification.referenceId ||
      chat.temporaryChat) return
    if (owned.pendingUndo?.() && owned.settled && owned.allowEmpty &&
      owned.snapshot?.checkpoint?.historySelectionReference === null && !chat.historyId && !chat.serverChatId) {
      // Undo's partial capture grants no row/reference authority; retain only the cleared draft.
      owned.snapshot = { ...owned.snapshot, checkpoint: { ...owned.snapshot.checkpoint, draft } }
      saveSession(owned.key, owned.snapshot)
      return
    }
    const reference = qualifiedReference(control, { serverChatId: chat.serverChatId, historyId: chat.historyId }, owned.qualification)
    if (!owned.settled) {
      // A newer authorized H1 capture can win while the checkpoint read is still pending.
      if (!reference || control.getCurrent().owner?.kind !== "native") return
      owned.settled = true
      setRestoring(false)
      setRestoreError(null)
    }
    if (!reference && (!owned.allowEmpty || control.getReference() || chat.messages.length || chat.history.length || chat.historyId || chat.serverChatId)) return
    const session: WorkspaceChatSession = {
      messages: chat.messages, history: chat.history, historyId: chat.historyId, serverChatId: chat.serverChatId,
      checkpoint: { version: 1, ...owned.qualification, historySelectionReference: reference, draft }
    }
    saveSession(owned.key, session)
    owned.snapshot = getSession(owned.key, owned.qualification)
  }, [control, workspaceId, referenceId, draft, chat.messages, chat.history, chat.historyId, chat.serverChatId, chat.temporaryChat, restoring, getSession, saveSession])

  useLayoutEffect(() => {
    if (active || !legacySessionKey) return
    const key = legacySessionKey
    const stored = getSession(key)
    if (stored?.checkpoint) return
    const { chat } = latest.current
    chat.setMessages(stored?.messages ?? [])
    chat.setHistory(stored?.history ?? [])
    chat.setHistoryId(stored?.historyId ?? null, { preserveServerChatId: true })
    chat.setServerChatId(stored?.serverChatId ?? null)
    chat.setStreaming(false)
    chat.setIsProcessing(false)
    const owned = { key, snapshot: stored, skipCapture: true }
    legacy.current = owned
    return () => {
      if (owned.snapshot && !getSession(key)?.checkpoint) saveSession(key, owned.snapshot)
      if (legacy.current === owned) legacy.current = null
    }
  }, [active, legacySessionKey, getSession, saveSession])

  useLayoutEffect(() => {
    const owned = legacy.current
    if (active || !owned || owned.key !== legacySessionKey || getSession(owned.key)?.checkpoint) return
    if (owned.skipCapture) { owned.skipCapture = false; return }
    saveSession(owned.key, { messages: chat.messages, history: chat.history, historyId: chat.historyId, serverChatId: chat.serverChatId })
    owned.snapshot = getSession(owned.key)
  }, [active, legacySessionKey, chat.messages, chat.history, chat.historyId, chat.serverChatId, getSession, saveSession])

  return { active, restoring, restoreError: restoreError?.isCurrent() ? restoreError.message : null,
    setDraft, controller: control, referenceId, fence, clearChat }
}
