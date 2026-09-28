import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import React from "react"
import { message } from "antd"

import { PageAssistDatabase } from "@/db/dexie/chat"
import {
  formatToChatHistory,
  formatToMessage,
  getPromptById,
  getSessionFiles
} from "@/db/dexie/helpers"
import { lastUsedChatModelEnabled } from "@/services/model-settings"
import { updatePageTitle } from "@/utils/update-page-title"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"

interface LoadLocalConversationDeps {
  setServerChatId: (id: string | null) => void
  setHistoryId: (id: string) => void
  setHistory: (history: any) => void
  setMessages: (messages: any[]) => void
  setSelectedModel: (id: string) => void
  setSelectedSystemPrompt: (id: string | null) => void
  setSystemPrompt: (prompt: string) => void
  setContextFiles: (files: any[]) => void
}

interface UseLoadLocalConversationOptions {
  t: (key: string, options?: any) => string
  errorLogPrefix: string
  errorDefaultMessage: string
}

export function useLoadLocalConversation(
  deps: LoadLocalConversationDeps,
  options: UseLoadLocalConversationOptions
) {
  const {
    setServerChatId,
    setHistoryId,
    setHistory,
    setMessages,
    setSelectedModel,
    setSelectedSystemPrompt,
    setSystemPrompt,
    setContextFiles
  } = deps

  const { t, errorLogPrefix, errorDefaultMessage } = options

  const { beginLoad, fence, loadConversation, getCurrent } =
    useHistorySelectionContext() ?? {}
  const dbRef = React.useRef<PageAssistDatabase | null>(null)
  const mountedRef = React.useRef(false)
  const loadGenerationRef = React.useRef(0)

  React.useEffect(() => {
    mountedRef.current = true
    const invalidate = () => { loadGenerationRef.current += 1 }
    const stop = watchChatAccountChanges(invalidated => { if (invalidated) invalidate() })
    return () => {
      mountedRef.current = false
      invalidate()
      stop()
    }
  }, [])

  if (!dbRef.current) {
    dbRef.current = new PageAssistDatabase()
  }

  return React.useCallback(
    async (conversationId: string): Promise<boolean> => {
      const generation = ++loadGenerationRef.current
      const restoreRevision = usePlaygroundSessionStore.getState().restoreRevision
      if (!mountedRef.current) return false
      beginLoad?.()
      let selectionCurrent = fence?.() || (() => true)
      let snapshot: ServicePromptSnapshot | undefined
      // The controller advances its own fence while opening. This caller lease
      // must track navigation/account changes independently of that fence.
      const isCurrentLoad = () => mountedRef.current &&
        !snapshot?.scopeSignal.aborted && !snapshot?.scopeInvalidatedSignal.aborted &&
        generation === loadGenerationRef.current &&
        restoreRevision === usePlaygroundSessionStore.getState().restoreRevision
      const isCurrent = () => isCurrentLoad() && selectionCurrent()
      try {
        const db = dbRef.current!
        const historyDetails = await db.getHistoryInfo(conversationId)
        if (!isCurrent() || !historyDetails) return false
        // Profile-owned forks browse independently of the inference account.
        // The controller still validates the local profile before publishing.
        const profileOwned = loadConversation && historyDetails.local_owner_key &&
          !historyDetails.server_scope_key && !historyDetails.server_chat_id &&
          historyDetails.message_source !== "server"
        if (!profileOwned) {
          snapshot = await loadServicePromptSnapshot([])
          if (!isCurrent() || historyDetails.server_scope_key !== serverChatMirrorOwnerKey(snapshot)) return false
        }
        const history = await db.getChatHistory(conversationId)
        if (!isCurrent()) return false
        if (loadConversation && fence) {
          if (!await loadConversation({ historyId: conversationId, isCurrent: isCurrentLoad })) return false
          if (generation !== loadGenerationRef.current) return false
          selectionCurrent = fence()
        }
        if (!isCurrent()) return false
        const selected = getCurrent?.()
        if (selected?.owner?.kind === "unavailable") return false
        setServerChatId(selected?.owner?.kind === "native" ? selected.owner.conversation_id : null)
        setHistoryId(conversationId)
        if (!selected || selected.capture?.status !== "captured") {
          setHistory(formatToChatHistory(history))
          setMessages(formatToMessage(history))
        }

        if (!isCurrent()) return false
        const isLastUsedChatModel = await lastUsedChatModelEnabled()
        if (!isCurrent()) return false
        if (isLastUsedChatModel && historyDetails?.model_id) {
          setSelectedModel(historyDetails.model_id)
        }

        if (!isCurrent()) return false
        const lastUsedPrompt = historyDetails?.last_used_prompt
        if (lastUsedPrompt) {
          let promptContent = lastUsedPrompt.prompt_content ?? ""
          if (lastUsedPrompt.prompt_id) {
            const prompt = await getPromptById(lastUsedPrompt.prompt_id)
            if (!isCurrent()) return false
            if (prompt) {
              setSelectedSystemPrompt(prompt.id)
              if (!promptContent.trim()) {
                promptContent = prompt.content
              }
            }
          }
          if (!isCurrent()) return false
          setSystemPrompt(promptContent)
        }

        if (!isCurrent()) return false
        const session = await getSessionFiles(conversationId)
        if (!isCurrent()) return false
        setContextFiles(session)

        if (!isCurrent()) return false
        updatePageTitle(
          historyDetails?.title || t("common:untitled", { defaultValue: "Untitled" })
        )
        return true
      } catch (error) {
        if (!isCurrent()) return false
        // eslint-disable-next-line no-console
        console.error(`${errorLogPrefix}:`, error)
        message.error(
          t("common:error.friendlyLocalHistorySummary", {
            defaultValue: errorDefaultMessage
          })
        )
        return false
      } finally {
        snapshot?.release()
      }
    },
    [
      beginLoad,
      fence,
      loadConversation,
      getCurrent,
      errorDefaultMessage,
      errorLogPrefix,
      setContextFiles,
      setHistory,
      setHistoryId,
      setMessages,
      setSelectedModel,
      setSelectedSystemPrompt,
      setServerChatId,
      setSystemPrompt,
      t
    ]
  )
}

/** Read-only comparison presentation; this never upgrades an unsupported send capture. */
export async function restoreReadableLocalComparison(
  controller: import('@/hooks/chat/useHistorySelection').HistorySelectionController,
  display: (value: { history: ReturnType<typeof formatToChatHistory>; messages: ReturnType<typeof formatToMessage> }) => void,
  supplied?: Awaited<ReturnType<typeof import('@/db/dexie/helpers').getFullChatData>>
): Promise<boolean> {
  const initial = controller.getCurrent()
  const owner = initial.owner
  const view = initial.view
  if (owner?.kind !== 'local' || !view || initial.error !== 'unsupported_comparison_history' || initial.capture?.snapshot.owner_key !== owner.owner_key) return false
  const current = controller.fence()
  const { getFullChatData } = await import('@/db/dexie/helpers')
  const data = supplied === undefined ? await getFullChatData(owner.conversation_id) : supplied
  if (!current() || !data || data.historyInfo.id !== owner.conversation_id) return false
  const { getLocalHistoryOwner } = await import('@/db/dexie/history-selection')
  let verified
  try { verified = await getLocalHistoryOwner(owner.conversation_id) } catch (error) {
    console.warn("Failed to verify readable comparison owner", error)
    return false
  }
  if (!current() || controller.getCurrent().owner !== owner || controller.getCurrent().view !== view || verified.owner_key !== owner.owner_key || verified.profile_id !== owner.profile_id) return false
  display({ history: formatToChatHistory(data.messages), messages: formatToMessage(data.messages) })
  return true
}
