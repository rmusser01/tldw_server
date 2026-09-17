import {
  createElement,
  Fragment,
  type ReactNode,
  createContext,
  useContext,
  useCallback,
  useEffect,
  useRef,
  useState
} from "react"
import {
  ensureLocalProfileId,
  loadHistoryBookmark,
  saveHistoryBookmark
} from "@/db/dexie/history-selection"
import type { HistoryBookmarkScope } from "@/db/dexie/types"
import {
  captureHistorySnapshot,
  confirmLegacyHistoryProjection,
  type HistoryOwnerV1
} from "@/services/chat-history-selection"
import type {
  HistoryCaptureRequestV1,
  HistoryCaptureResultV1,
  HistoryCursorV1,
  HistoryViewSelectionV1,
  LegacyHistoryProjectionConfirmV1
} from "@/types/history-selection"

/** A bookmark address, never an owner credential or a shared live cursor. */
export type HistorySelectionReference = HistoryBookmarkScope & {
  owner_key: string
  conversation_id: string
  pending_confirmation_reference?: {
    client_session_id: string
    view_session_id: string
    projection_id: string
  }
}
type PendingConfirmation = {
  scope: HistoryBookmarkScope
  view: HistoryViewSelectionV1
  intent: LegacyHistoryProjectionConfirmV1
}
type SelectionState = {
  owner: HistoryOwnerV1 | null
  bookmarkScope: HistoryBookmarkScope | null
  view: HistoryViewSelectionV1 | null
  capture: HistoryCaptureResultV1 | null
  status:
    | "idle"
    | "loading"
    | "ready"
    | "pending"
    | "pending_unknown"
    | "error"
    | "legacy_review_required"
    | "stale_selection"
    | "invalid_history"
    | "unsupported_history_capability"
  error: string | null
  pending: PendingConfirmation | null
}
const initialState = (): SelectionState => ({
  owner: null,
  bookmarkScope: null,
  view: null,
  capture: null,
  status: "idle",
  error: null,
  pending: null
})
const pendingReference = (pending: PendingConfirmation | null) =>
  pending
    ? {
        client_session_id: pending.scope.client_session_id,
        view_session_id: pending.view.view_session_id,
        projection_id: pending.intent.projection_id
      }
    : undefined
const sameView = (
  a: HistoryViewSelectionV1 | null,
  b: HistoryViewSelectionV1
) =>
  Boolean(
    a &&
    a.owner_key === b.owner_key &&
    a.conversation_id === b.conversation_id &&
    a.view_session_id === b.view_session_id &&
    a.selection_revision === b.selection_revision
  )
const errorCode = (error: unknown) =>
  String(
    (error as { code?: string })?.code ||
      (error instanceof Error ? error.message : "history_selection_failed")
  )

/** One controller per mounted view. Reopen references initialize a fresh writer. */
export function useHistorySelection(
  options: {
    storageKey?: string
    onCapture?: (
      capture: import("@/types/history-selection").HistorySelectionCaptureV1
    ) => void
  } = {}
) {
  const onCapture = useRef(options.onCapture)
  onCapture.current = options.onCapture
  const [state, setState] = useState<SelectionState>(initialState)
  const live = useRef(state)
  const identity = useRef<{ client: string; view: string; revision: number }>()
  if (!identity.current)
    identity.current = {
      client: crypto.randomUUID(),
      view: crypto.randomUUID(),
      revision: -1
    }
  const logicalViews = useRef(
    new Map<
      string,
      {
        identity: { client: string; view: string; revision: number }
        state: SelectionState
      }
    >()
  )
  const activeLogicalView = useRef("default")
  const releaseOwnerLease = useRef<(() => void) | null>(null)
  const epoch = useRef(0)
  const request = useRef<AbortController | null>(null)
  const mounted = useRef(true)
  const saves = useRef(Promise.resolve())
  const storageKey = useRef(options.storageKey)
  const publish = useCallback((next: SelectionState) => {
    live.current = next
    if (mounted.current) setState(next)
  }, [])
  const invalidate = useCallback(() => {
    epoch.current += 1
    request.current?.abort()
    request.current = new AbortController()
    return { epoch: epoch.current, signal: request.current.signal }
  }, [])
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      invalidate()
      releaseOwnerLease.current?.()
    }
  }, [invalidate])
  const persist = useCallback(
    async (scope: HistoryBookmarkScope, view: HistoryViewSelectionV1) => {
      const operation = saves.current
        .catch(() => {})
        .then(async () => {
          await saveHistoryBookmark(scope, view)
          if (storageKey.current && sameView(live.current.view, view)) {
            sessionStorage.setItem(
              storageKey.current,
              JSON.stringify({
                ...scope,
                owner_key: view.owner_key,
                conversation_id: view.conversation_id,
                owner_kind: live.current.owner?.kind,
                pending_confirmation_reference: pendingReference(
                  live.current.pending
                )
              })
            )
          }
        })
      saves.current = operation
      await operation
    },
    []
  )
  const install = useCallback(
    async (
      result: HistoryCaptureResultV1,
      owner: HistoryOwnerV1,
      scope: HistoryBookmarkScope,
      token: number,
      pending: PendingConfirmation | null = null
    ) => {
      if (!mounted.current || token !== epoch.current) return false
      publish({
        owner,
        bookmarkScope: scope,
        view: result.view,
        capture: result,
        pending,
        status: pending
          ? "pending_unknown"
          : result.status === "captured"
            ? "ready"
            : result.status,
        error: result.status === "captured" ? null : result.code
      })
      if (result.status === "captured") onCapture.current?.(result)
      await persist(scope, result.view)
      return token === epoch.current
    },
    [persist, publish]
  )

  const open = useCallback(
    async (
      owner: HistoryOwnerV1,
      reference?: HistorySelectionReference | null
    ) => {
      const operation = invalidate()
      if (owner.kind !== "native") {
        releaseOwnerLease.current?.()
        releaseOwnerLease.current = null
      }
      publish({ ...initialState(), owner, status: "loading" })
      try {
        // Unsupported temporary owners must not create even a profile/bookmark.
        if (owner.kind === "unavailable") throw new Error(owner.code)
        const profile = await ensureLocalProfileId()
        if (operation.epoch !== epoch.current) return false
        const scope = {
          profile_id: profile,
          client_session_id: identity.current!.client
        }
        let source = reference
        if (source === undefined && storageKey.current) {
          const stored = sessionStorage.getItem(storageKey.current)
          source = stored
            ? parseHistorySelectionHandoff(
                "historySelection=" + encodeURIComponent(stored)
              )
            : null
        }
        if (
          reference === undefined &&
          source?.conversation_id !== owner.conversation_id
        )
          source = null
        if (
          source &&
          (source.profile_id !== profile ||
            source.conversation_id !== owner.conversation_id ||
            (owner.owner_key && source.owner_key !== owner.owner_key))
        ) {
          throw new Error("owner_conversation_mismatch")
        }
        const ownBookmark = owner.owner_key
          ? await loadHistoryBookmark(scope, {
              owner_key: owner.owner_key,
              conversation_id: owner.conversation_id
            })
          : null
        const bookmark =
          ownBookmark ||
          (source
            ? await loadHistoryBookmark(source, {
                owner_key: source.owner_key,
                conversation_id: source.conversation_id
              })
            : null)
        if (source && !bookmark) throw new Error("stale_bookmark")
        let pendingBookmark = bookmark
        const pointer = source?.pending_confirmation_reference
        if (pointer) {
          pendingBookmark = await loadHistoryBookmark(
            {
              profile_id: profile,
              client_session_id: pointer.client_session_id
            },
            {
              owner_key: source!.owner_key,
              conversation_id: source!.conversation_id
            }
          )
          if (!pendingBookmark)
            throw new Error("pending_confirmation_reference_missing")
          if (
            pendingBookmark?.pending_confirmation &&
            (pendingBookmark.pending_confirmation.projection_id !==
              pointer.projection_id ||
              pendingBookmark.pending_view_session_id !==
                pointer.view_session_id)
          )
            throw new Error("pending_confirmation_reference_mismatch")
        }
        const revision = ++identity.current!.revision
        let view: HistoryCaptureRequestV1["view"] = bookmark
          ? {
              ...bookmark.view,
              view_session_id: identity.current!.view,
              selection_revision: revision
            }
          : {
              view_session_id: identity.current!.view,
              owner_key: owner.owner_key,
              conversation_id: owner.conversation_id,
              interpretation: { kind: "parent_graph_v1" },
              cursor: { kind: "empty" },
              selection_revision: revision
            }
        let result = await captureHistorySnapshot(
          owner,
          view,
          "send",
          operation.signal
        )
        // A fresh graph uses its last owner-ordered tip. An explicit empty bookmark never does.
        if (
          !bookmark &&
          result.status === "captured" &&
          result.snapshot.nodes.length
        ) {
          const parents = new Set(
            result.snapshot.nodes.map((node) => node.parent_id)
          )
          const tip = [...result.snapshot.nodes]
            .reverse()
            .find((node) => !parents.has(node.id))
          if (tip) {
            view = {
              ...result.view,
              cursor: { kind: "after_message", message_id: tip.id }
            }
            result = await captureHistorySnapshot(
              owner,
              view,
              "send",
              operation.signal
            )
          }
        }
        const pending = pendingBookmark?.pending_confirmation
          ? {
              scope: {
                profile_id: pendingBookmark.profile_id,
                client_session_id: pendingBookmark.client_session_id
              },
              view: {
                ...pendingBookmark.view,
                view_session_id:
                  pendingBookmark.pending_view_session_id ||
                  pendingBookmark.view.view_session_id,
                selection_revision:
                  pendingBookmark.pending_confirmation.selection_revision
              },
              intent: pendingBookmark.pending_confirmation
            }
          : null
        return await install(result, owner, scope, operation.epoch, pending)
      } catch (error) {
        if (operation.epoch !== epoch.current) return false
        publish({
          ...live.current,
          status:
            errorCode(error) === "stale_bookmark"
              ? "stale_selection"
              : "unsupported_history_capability",
          error: errorCode(error)
        })
        return true
      }
    },
    [install, invalidate, publish]
  )

  const choose = useCallback(
    async (cursor: HistoryCursorV1) => {
      const current = live.current
      if (!current.owner || !current.view || !current.bookmarkScope)
        return false
      const operation = invalidate()
      const view = {
        ...current.view,
        cursor,
        selection_revision: ++identity.current!.revision
      }
      publish({ ...current, view, status: "loading", error: null })
      try {
        const result = await captureHistorySnapshot(
          current.owner,
          view,
          "send",
          operation.signal
        )
        return await install(
          result,
          current.owner,
          current.bookmarkScope,
          operation.epoch,
          current.pending
        )
      } catch (error) {
        if (operation.epoch === epoch.current)
          publish({ ...live.current, status: "error", error: errorCode(error) })
        return false
      }
    },
    [install, invalidate, publish]
  )
  const refresh = useCallback(async () => {
    const current = live.current
    if (!current.owner || !current.view || !current.bookmarkScope) return false
    const operation = invalidate()
    publish({ ...current, status: "loading", error: null })
    try {
      const result = await captureHistorySnapshot(
        current.owner,
        current.view,
        "send",
        operation.signal
      )
      return await install(
        result,
        current.owner,
        current.bookmarkScope,
        operation.epoch,
        current.pending
      )
    } catch (error) {
      if (operation.epoch === epoch.current)
        publish({ ...live.current, status: "error", error: errorCode(error) })
      return false
    }
  }, [install, invalidate, publish])
  const confirm = useCallback(
    async (orderedPathIds: readonly string[], cursor: HistoryCursorV1) => {
      const current = live.current
      if (
        current.status === "pending" ||
        !current.owner ||
        !current.view ||
        !current.bookmarkScope ||
        !current.capture
      )
        return false
      const { snapshot } = current.capture
      const pending: PendingConfirmation = current.pending || {
        scope: current.bookmarkScope,
        view: current.view,
        intent: {
          version: 1,
          projection_id: crypto.randomUUID(),
          owner_key: snapshot.owner_key,
          conversation_id: snapshot.conversation_id,
          source_digest: snapshot.source_digest,
          fences: snapshot.fences,
          source_members: snapshot.nodes.map(({ id, revision }) => ({
            id,
            revision
          })),
          ordered_path_ids: [...orderedPathIds],
          cursor,
          selection_revision: current.view.selection_revision
        }
      }
      publish({ ...current, status: "pending", pending, error: null })
      try {
        // Confirmation is immutable once dispatched. A cursor change cannot cancel owner outcome.
        const projection = await confirmLegacyHistoryProjection(
          current.owner,
          pending.scope,
          pending.intent,
          pending.view
        )
        if (!sameView(live.current.view, current.view)) {
          if (
            live.current.pending?.intent.projection_id ===
            pending.intent.projection_id
          ) {
            publish({
              ...live.current,
              pending: null,
              status:
                live.current.capture?.status === "captured"
                  ? "ready"
                  : live.current.capture?.status || "idle"
            })
          }
          return false
        }
        if (current.pending && !sameView(current.view, pending.view)) {
          publish({
            ...live.current,
            pending: null,
            status: "stale_selection",
            error: "confirmation_completed_in_original_view"
          })
          return true
        }
        publish({
          ...live.current,
          pending: null,
          view: {
            ...current.view,
            selection_revision: ++identity.current!.revision,
            interpretation: {
              kind: "legacy_linear_v1",
              projection_id: projection.projection_id
            },
            cursor: projection.cursor
          }
        })
        return await refresh()
      } catch (error) {
        let stored
        try {
          stored = await loadHistoryBookmark(pending.scope, pending.view)
        } catch {
          if (sameView(live.current.view, current.view))
            publish({
              ...live.current,
              pending,
              status: "pending_unknown",
              error: errorCode(error)
            })
          return false
        }
        if (!sameView(live.current.view, current.view)) return false
        publish({
          ...live.current,
          pending: stored?.pending_confirmation ? pending : null,
          status: stored?.pending_confirmation
            ? "pending_unknown"
            : "stale_selection",
          error: errorCode(error)
        })
        return false
      }
    },
    [publish, refresh]
  )
  const followResult = useCallback(
    async (origin: HistoryViewSelectionV1, messageId: string) => {
      if (!sameView(live.current.view, origin)) return false
      return choose({ kind: "after_message", message_id: messageId })
    },
    [choose]
  )
  const activate = useCallback(
    (key: string) => {
      if (activeLogicalView.current === key) return
      logicalViews.current.set(activeLogicalView.current, {
        identity: identity.current!,
        state: live.current
      })
      invalidate()
      releaseOwnerLease.current?.()
      activeLogicalView.current = key
      const previous = logicalViews.current.get(key)
      identity.current = previous?.identity || {
        client: crypto.randomUUID(),
        view: crypto.randomUUID(),
        revision: -1
      }
      publish(previous?.state || initialState())
    },
    [invalidate, publish]
  )
  const loadConversation = useCallback(
    async (
      target: {
        historyId?: string | null
        serverChatId?: string | null
        temporary?: boolean
        scope?: import("@/types/chat-scope").ChatScope
        bindUnbound?: boolean
      },
      reference?: HistorySelectionReference | null
    ) => {
      const operation = invalidate()
      releaseOwnerLease.current?.()
      releaseOwnerLease.current = null
      if (target.temporary || target.historyId === "temp")
        return open({
          kind: "unavailable",
          code: "temporary_history_unavailable"
        })
      if (!target.historyId && !target.serverChatId) {
        publish(initialState())
        return false
      }
      try {
        const { PageAssistDatabase } = await import("@/db/dexie/chat")
        const details = target.historyId
          ? await new PageAssistDatabase().getHistoryInfo(target.historyId)
          : null
        if (operation.epoch !== epoch.current) return false
        const chatId = target.serverChatId || details?.server_chat_id
        if (!chatId) {
          const { getLocalHistoryOwner } =
            await import("@/db/dexie/history-selection")
          const owner = await getLocalHistoryOwner(target.historyId!)
          if (operation.epoch !== epoch.current) return false
          return open(owner, reference)
        }
        if (
          details?.server_chat_id &&
          !details.server_scope_key &&
          !target.bindUnbound
        )
          return open({ kind: "unavailable", code: "unbound_server_mirror" })
        const {
          resolveServicePromptScope,
          subscribeToServicePromptConfigChanges
        } = await import("@/services/service-prompts")
        let valid = true
        let capturedOwner: HistoryOwnerV1 | null = null
        const unsubscribe = subscribeToServicePromptConfigChanges(() => {
          valid = false
          if (
            operation.epoch !== epoch.current &&
            live.current.owner !== capturedOwner
          )
            return
          invalidate()
          publish({
            ...live.current,
            status: "unsupported_history_capability",
            error: "request_config_scope_changed"
          })
        })
        releaseOwnerLease.current = () => {
          valid = false
          unsubscribe()
        }
        const requestScope = await resolveServicePromptScope({
          signal: operation.signal
        })
        if (!valid || operation.epoch !== epoch.current) return false
        const { serverChatMirrorOwnerKey, linkServerChatMirror } =
          await import("@/db/dexie/server-chat-mirror")
        const mirrorKey = serverChatMirrorOwnerKey({ requestScope })
        if (
          details?.server_scope_key &&
          (details.server_scope_key !== mirrorKey ||
            details.server_chat_id !== chatId)
        )
          throw new Error("owner_conversation_mismatch")
        const owner: HistoryOwnerV1 = {
          kind: "native",
          conversation_id: String(chatId),
          request_scope: requestScope,
          scope: target.scope,
          validate_lease: () => valid
        }
        capturedOwner = owner
        const result = await open(owner, reference)
        // Only an explicit action plus an authorized owner capture can bind an old mirror.
        if (
          target.bindUnbound &&
          result &&
          live.current.capture &&
          details &&
          valid
        ) {
          const bindingEpoch = epoch.current
          try {
            await linkServerChatMirror({
              chatId: String(chatId),
              title: details.title,
              ownerKey: mirrorKey,
              currentHistoryId: details.id,
              legacyHistoryId: details.id,
              signal: request.current?.signal
            })
          } catch (error) {
            if (bindingEpoch !== epoch.current) return false
            publish({
              ...live.current,
              status: "error",
              error: errorCode(error)
            })
            return true
          }
        }
        return result
      } catch (error) {
        if (operation.epoch !== epoch.current) return false
        return open({ kind: "unavailable", code: errorCode(error) })
      }
    },
    [invalidate, open, publish]
  )
  const reset = useCallback(() => {
    invalidate()
    releaseOwnerLease.current?.()
    releaseOwnerLease.current = null
    publish(initialState())
    if (storageKey.current) sessionStorage.removeItem(storageKey.current)
  }, [invalidate, publish])
  const fence = useCallback(() => {
    const token = epoch.current
    return () => mounted.current && token === epoch.current
  }, [])
  const getStoredReference = useCallback((): HistorySelectionHandoff | null => {
    if (!storageKey.current) return null
    const stored = sessionStorage.getItem(storageKey.current)
    return stored
      ? parseHistorySelectionHandoff(
          "historySelection=" + encodeURIComponent(stored)
        )
      : null
  }, [])
  const getReference = useCallback((): HistorySelectionReference | null => {
    const current = live.current
    return current.view && current.bookmarkScope
      ? {
          ...current.bookmarkScope,
          owner_key: current.view.owner_key,
          conversation_id: current.view.conversation_id,
          pending_confirmation_reference: pendingReference(current.pending)
        }
      : null
  }, [])
  const prepareExpansionPath = useCallback(async (): Promise<string | null> => {
    const current = live.current
    if (current.status === "pending") return null
    if (
      !current.view ||
      !current.bookmarkScope ||
      !current.owner ||
      current.owner.kind === "unavailable"
    )
      return "/chat"
    const view = structuredClone(current.view)
    const scope = {
      ...current.bookmarkScope,
      client_session_id: crypto.randomUUID()
    }
    const reference: HistorySelectionReference = {
      ...scope,
      owner_key: view.owner_key,
      conversation_id: view.conversation_id,
      pending_confirmation_reference: pendingReference(current.pending)
    }
    try {
      // A fresh address is write-once initialization data; source swipes cannot change it.
      await saveHistoryBookmark(scope, view)
      return historySelectionExpansionPath({ reference, owner: current.owner })
    } catch (error) {
      if (sameView(live.current.view, view))
        publish({ ...live.current, status: "error", error: errorCode(error) })
      return null
    }
  }, [publish])
  const reference: HistorySelectionReference | null =
    state.view && state.bookmarkScope
      ? {
          ...state.bookmarkScope,
          owner_key: state.view.owner_key,
          conversation_id: state.view.conversation_id,
          pending_confirmation_reference: pendingReference(state.pending)
        }
      : null
  return {
    ...state,
    reference,
    getReference,
    getStoredReference,
    prepareExpansionPath,
    open,
    loadConversation,
    activate,
    choose,
    refresh,
    confirm,
    followResult,
    reset,
    fence,
    beginLoad: invalidate,
    getSignal: () => request.current?.signal,
    getCurrent: () => live.current
  }
}
export type HistorySelectionController = ReturnType<typeof useHistorySelection>

export const HistorySelectionContext =
  createContext<HistorySelectionController | null>(null)
export const useHistorySelectionContext = () =>
  useContext(HistorySelectionContext)

/** Only a scoped address crosses the extension full-page URL. */
export type HistorySelectionHandoff = HistorySelectionReference & {
  owner_kind: "local" | "native"
}
export const historySelectionExpansionPath = (
  selection: Pick<HistorySelectionController, "reference" | "owner">
): string => {
  if (
    !selection.reference ||
    !selection.owner ||
    selection.owner.kind === "unavailable"
  )
    return "/chat"
  return `/chat?historySelection=${encodeURIComponent(JSON.stringify({ ...selection.reference, owner_kind: selection.owner.kind }))}`
}
export const parseHistorySelectionHandoff = (
  search: string
): HistorySelectionHandoff | null => {
  const value = new URLSearchParams(search.replace(/^.*\?/, "")).get(
    "historySelection"
  )
  if (!value) return null
  let parsed: unknown
  try {
    parsed = JSON.parse(value)
  } catch {
    throw new Error("invalid_history_reference")
  }
  const keys = [
    "profile_id",
    "client_session_id",
    "owner_key",
    "conversation_id",
    "owner_kind"
  ]
  if (
    !parsed ||
    typeof parsed !== "object" ||
    Array.isArray(parsed) ||
    Object.keys(parsed).some(
      (key) => !keys.includes(key) && key !== "pending_confirmation_reference"
    ) ||
    keys.some(
      (key) =>
        typeof (parsed as any)[key] !== "string" || !(parsed as any)[key].length
    ) ||
    !["local", "native"].includes((parsed as any).owner_kind)
  )
    throw new Error("invalid_history_reference")
  const pointer = (parsed as HistorySelectionReference)
    .pending_confirmation_reference
  if (
    pointer !== undefined &&
    (!pointer ||
      typeof pointer !== "object" ||
      Array.isArray(pointer) ||
      Object.keys(pointer).length !== 3 ||
      ["client_session_id", "view_session_id", "projection_id"].some(
        (key) => typeof pointer[key] !== "string" || !pointer[key]
      ))
  )
    throw new Error("invalid_history_reference")
  return parsed as HistorySelectionHandoff
}

/** Reuses an ancestor controller; a standalone surface allocates exactly one. */
export function HistorySelectionProvider({
  children,
  ...options
}: Parameters<typeof useHistorySelection>[0] & { children: ReactNode }) {
  const existing = useHistorySelectionContext()
  return existing
    ? createElement(Fragment, null, children)
    : createElement(HistorySelectionOwner, options, children)
}
function HistorySelectionOwner({
  children,
  ...options
}: Parameters<typeof useHistorySelection>[0] & { children?: ReactNode }) {
  const selection = useHistorySelection(options)
  return createElement(
    HistorySelectionContext.Provider,
    { value: selection },
    children
  )
}
