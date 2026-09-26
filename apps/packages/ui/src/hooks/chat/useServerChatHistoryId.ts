import { useHistorySelectionContext } from "./useHistorySelection"
import React from "react"
import type { TFunction } from "i18next"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { linkServerChatMirror, serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { usePlaygroundSessionStore } from "@/store/playground-session"

export type VerifiedHistoryMirrorOwner = {
  ownerKey: string
  validateLease: () => boolean
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
      ownerOrSnapshot?: VerifiedHistoryMirrorOwner | ServicePromptSnapshot
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
              validateLease: selectedOwner.validate_lease
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
        const key = JSON.stringify([owner.ownerKey, chatId])
        const localId =
          scopedCache.current.get(key) ||
          (await linkServerChatMirror({
            chatId,
            title:
              title?.trim() ||
              t("common:untitled", { defaultValue: "Untitled" }),
            ownerKey: owner.ownerKey,
            currentHistoryId,
            signal: scopeInvalidatedSignal
          }))
        assertCurrent()
        // A concurrent bind for this same verified owner may have installed
        // the exact mirror while this request was awaiting storage.
        if (historyIdRef.current !== currentHistoryId && historyIdRef.current !== localId)
          throw createServicePromptScopeChangedError()
        scopedCache.current.set(key, localId)
        if (
          historyIdRef.current === currentHistoryId &&
          currentHistoryId !== localId
        )
          setHistoryId(localId, { preserveServerChatId: true })
        return localId
      }
      if (!snapshot) throw createServicePromptScopeChangedError()
      const ownerKey = serverChatMirrorOwnerKey(snapshot)
      const session = usePlaygroundSessionStore.getState()
      const legacyHistoryId = session.isSessionValid(snapshot.scopeKey) &&
        session.historyId === currentHistoryId &&
        (!session.serverChatId || session.serverChatId === chatId)
        ? currentHistoryId : null
      if (
        serverChatHistoryIdRef.current.chatId === chatId &&
        serverChatHistoryIdRef.current.ownerKey === ownerKey &&
        serverChatHistoryIdRef.current.historyId
      ) {
        const existingId = serverChatHistoryIdRef.current.historyId
        if (currentHistoryId !== existingId) {
          setHistoryId(existingId, { preserveServerChatId: true })
        }
        return existingId
      }

      let linkedHistoryId: string
      try {
        linkedHistoryId = await linkServerChatMirror({
          chatId,
          title: title?.trim() || t("common:untitled", { defaultValue: "Untitled" }),
          ownerKey,
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

      throwIfScopeInvalidated()
      serverChatHistoryIdRef.current = {
        chatId,
        historyId: linkedHistoryId,
        ownerKey
      }
      if (currentHistoryId !== linkedHistoryId) {
        setHistoryId(linkedHistoryId, { preserveServerChatId: true })
      }
      return linkedHistoryId
    },
    [selection, setHistoryId, t, temporaryChat]
  )

  return {
    ensureServerChatHistoryId
  }
}
