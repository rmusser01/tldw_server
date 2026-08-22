import React from "react"
import type { TFunction } from "i18next"
import {
  getHistoryByServerChatId,
  saveHistory,
  setHistoryServerChatId,
  updateHistory
} from "@/db/dexie/helpers"
import { runChatPersistenceTransaction } from "@/db/dexie/chat-persistence-transaction"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"

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
  const historyIdRef = React.useRef(historyId)
  const serverChatHistoryIdRef = React.useRef<{
    chatId: string | null
    historyId: string | null
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
      scopeInvalidatedSignal?: AbortSignal
    ) => {
      if (!chatId || temporaryChat) return null
      const throwIfScopeInvalidated = () => {
        if (scopeInvalidatedSignal?.aborted) {
          throw createServicePromptScopeChangedError()
        }
      }
      throwIfScopeInvalidated()
      const currentHistoryId = historyIdRef.current
      if (
        serverChatHistoryIdRef.current.chatId === chatId &&
        serverChatHistoryIdRef.current.historyId
      ) {
        const existingId = serverChatHistoryIdRef.current.historyId
        if (currentHistoryId !== existingId) {
          setHistoryId(existingId, { preserveServerChatId: true })
        }
        return existingId
      }

      const linkHistory = async () => {
        const existing = await getHistoryByServerChatId(chatId)
        const trimmedTitle = (title || existing?.title || "").trim()
        const resolvedTitle =
          trimmedTitle ||
          t("common:untitled", { defaultValue: "Untitled" })

        if (existing) {
          if (resolvedTitle && resolvedTitle !== existing.title) {
            await updateHistory(existing.id, resolvedTitle)
          }
          return {
            historyId: existing.id,
            shouldSetHistoryId: currentHistoryId !== existing.id
          }
        }

        if (currentHistoryId && currentHistoryId !== "temp") {
          await setHistoryServerChatId(currentHistoryId, chatId)
          if (resolvedTitle) {
            await updateHistory(currentHistoryId, resolvedTitle)
          }
          return {
            historyId: currentHistoryId,
            shouldSetHistoryId: false
          }
        }

        const newHistory = await saveHistory(
          resolvedTitle,
          false,
          "server",
          undefined,
          chatId
        )
        return {
          historyId: newHistory.id,
          shouldSetHistoryId: true
        }
      }

      let linkedHistory: Awaited<ReturnType<typeof linkHistory>>
      try {
        linkedHistory = scopeInvalidatedSignal
          ? await runChatPersistenceTransaction(
              scopeInvalidatedSignal,
              linkHistory
            )
          : await linkHistory()
      } catch (error) {
        if (scopeInvalidatedSignal?.aborted) {
          throw createServicePromptScopeChangedError()
        }
        throw error
      }

      throwIfScopeInvalidated()
      serverChatHistoryIdRef.current = {
        chatId,
        historyId: linkedHistory.historyId
      }
      if (linkedHistory.shouldSetHistoryId) {
        setHistoryId(linkedHistory.historyId, { preserveServerChatId: true })
      }
      return linkedHistory.historyId
    },
    [setHistoryId, t, temporaryChat]
  )

  return {
    ensureServerChatHistoryId
  }
}
