import { useStoreMessageOption } from "@/store/option"
import type { ActorSceneSend } from "./actor-scene-source"
import { useHistorySelectionContext } from "./useHistorySelection"
import React from "react"
import type { ChatScope } from "@/types/chat-scope"
import type { TFunction } from "i18next"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { linkServerChatMirror, serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { usePlaygroundSessionStore } from "@/store/playground-session"

export type VerifiedHistoryMirrorOwner = {
  ownerKey: string
  validateLease: () => boolean
  scope?: ChatScope
}

type UseServerChatHistoryIdOptions = {
  serverChatId: string | null
  historyId: string | null
  setHistoryId: (
    historyId: string | null,
    options?: { preserveServerChatId?: boolean }
  ) => void
  temporaryChat: boolean
  t: TFunction
}

export const useServerChatHistoryId = ({
  serverChatId,
  historyId,
  setHistoryId,
  temporaryChat,
  t
}: UseServerChatHistoryIdOptions) => {
  const selection = useHistorySelectionContext()
  const scopedCache = React.useRef(new Map<string, string>())
  const historyIdRef = React.useRef(historyId)
  const serverChatHistoryIdRef = React.useRef<{
    chatId: string | null
    historyId: string | null
    ownerKey?: string
    scopeKey?: string
  }>({ chatId: null, historyId: null })

  React.useEffect(() => {
    historyIdRef.current = historyId
  }, [historyId])

  React.useEffect(() => {
    if (serverChatHistoryIdRef.current.chatId !== serverChatId) {
      serverChatHistoryIdRef.current = {
        chatId: serverChatId ?? null,
        historyId: null
      }
    }
  }, [serverChatId])

  const ensureServerChatHistoryId = React.useCallback(
    async (
      chatId: string,
      title?: string,
      scopeInvalidatedSignal?: AbortSignal,
      ownerOrSnapshot?: VerifiedHistoryMirrorOwner | ServicePromptSnapshot,
      sceneSend?: ActorSceneSend
    ) => {
      if (!chatId || temporaryChat) return null
      const snapshot = ownerOrSnapshot && "requestScope" in ownerOrSnapshot ? ownerOrSnapshot : undefined
      const verifiedOwner = ownerOrSnapshot && "validateLease" in ownerOrSnapshot ? ownerOrSnapshot : undefined
      if (!snapshot && !verifiedOwner && !selection) throw createServicePromptScopeChangedError()
      const throwIfScopeInvalidated = () => {
        if (scopeInvalidatedSignal?.aborted) {
          throw createServicePromptScopeChangedError()
        }
      }
      throwIfScopeInvalidated()
      const currentHistoryId = historyIdRef.current
      const selectedOwner = selection?.getCurrent().owner
      const owner = verifiedOwner ||
        (selectedOwner?.kind === "native" &&
        selectedOwner.conversation_id === chatId
          ? {
              ownerKey: serverChatMirrorOwnerKey({
                requestScope: selectedOwner.request_scope
              }),
              validateLease: selectedOwner.validate_lease,
              scope: selectedOwner.scope
            }
          : null)
      if (selection && !owner) throw new Error("unbound_server_mirror")
      if (owner) {
        const assertCurrent = () => {
          throwIfScopeInvalidated()
          if (!owner.validateLease())
            throw createServicePromptScopeChangedError()
        }
        assertCurrent()
        if (snapshot && serverChatMirrorOwnerKey(snapshot) !== owner.ownerKey)
          throw createServicePromptScopeChangedError()
        const key = JSON.stringify([owner.ownerKey, chatId, owner.scope?.type ?? "global", owner.scope?.type === "workspace" ? owner.scope.workspaceId : null])
        const localId =
          scopedCache.current.get(key) ||
          (await linkServerChatMirror({
            chatId,
            title:
              title?.trim() ||
              t("common:untitled", { defaultValue: "Untitled" }),
            ownerKey: owner.ownerKey,
            scope: owner.scope,
            currentHistoryId,
            signal: scopeInvalidatedSignal
          }))
        assertCurrent()
        // A concurrent bind for this same verified owner may have installed
        // the exact mirror while this request was awaiting storage.
        if (historyIdRef.current !== currentHistoryId && historyIdRef.current !== localId)
          throw createServicePromptScopeChangedError()
        if (sceneSend && sceneSend.acceptHistoryMirror?.(localId) !== true)
          throw createServicePromptScopeChangedError()
        scopedCache.current.set(key, localId)
        if (
          historyIdRef.current === currentHistoryId &&
          currentHistoryId !== localId
        )
          setHistoryId(localId, { preserveServerChatId: true })
        assertCurrent()
        if (sceneSend && (!sceneSend.isCurrent() || useStoreMessageOption.getState().historyId !== localId))
          throw createServicePromptScopeChangedError()
        return localId
      }
      if (!snapshot) throw createServicePromptScopeChangedError()
      const assertLegacyCurrent = () => {
        throwIfScopeInvalidated()
        if (sceneSend && !sceneSend.isCurrent()) throw createServicePromptScopeChangedError()
      }
      const publishLegacyMirror = (localId: string) => {
        assertLegacyCurrent()
        if (sceneSend && sceneSend.acceptHistoryMirror?.(localId) !== true) throw createServicePromptScopeChangedError()
        if (currentHistoryId !== localId) setHistoryId(localId, { preserveServerChatId: true })
        assertLegacyCurrent()
        if (sceneSend && useStoreMessageOption.getState().historyId !== localId) throw createServicePromptScopeChangedError()
      }
      assertLegacyCurrent()
      const ownerKey = serverChatMirrorOwnerKey(snapshot)
      const mirrorScope = sceneSend?.scene.target.address?.kind === "local" ? undefined : sceneSend?.scene.target.address?.scope
      const mirrorScopeKey = JSON.stringify(mirrorScope ?? { type: "global" })
      const session = usePlaygroundSessionStore.getState()
      const legacyHistoryId = session.isSessionValid(snapshot.scopeKey) &&
        session.historyId === currentHistoryId &&
        (!session.serverChatId || session.serverChatId === chatId)
        ? currentHistoryId : null
      if (
        serverChatHistoryIdRef.current.chatId === chatId &&
        serverChatHistoryIdRef.current.ownerKey === ownerKey &&
        serverChatHistoryIdRef.current.scopeKey === mirrorScopeKey &&
        serverChatHistoryIdRef.current.historyId
      ) {
        const existingId = serverChatHistoryIdRef.current.historyId
        publishLegacyMirror(existingId)
        return existingId
      }

      let linkedHistoryId: string
      try {
        linkedHistoryId = await linkServerChatMirror({
          chatId,
          title: title?.trim() || t("common:untitled", { defaultValue: "Untitled" }),
          ownerKey,
          scope: mirrorScope,
          currentHistoryId,
          legacyHistoryId,
          signal: scopeInvalidatedSignal
        })
      } catch (error) {
        if (scopeInvalidatedSignal?.aborted) {
          throw createServicePromptScopeChangedError()
        }
        throw error
      }
      if (historyIdRef.current !== currentHistoryId) {
        throw createServicePromptScopeChangedError()
      }

      assertLegacyCurrent()
      publishLegacyMirror(linkedHistoryId)
      serverChatHistoryIdRef.current = {
        chatId,
        historyId: linkedHistoryId,
        ownerKey,
        scopeKey: mirrorScopeKey
      }
      return linkedHistoryId
    },
    [selection, setHistoryId, t, temporaryChat]
  )

  return {
    ensureServerChatHistoryId
  }
}
