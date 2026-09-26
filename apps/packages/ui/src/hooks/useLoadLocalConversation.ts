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

  const selection = useHistorySelectionContext()
  const dbRef = React.useRef<PageAssistDatabase | null>(null)
  const mountedRef = React.useRef(false)
  const loadGenerationRef = React.useRef(0)

  React.useEffect(() => {
    mountedRef.current = true
    const invalidate = () => { loadGenerationRef.current += 1 }
    window.addEventListener("tldw:auth-principal-changed", invalidate)
    return () => {
      mountedRef.current = false
      invalidate()
      window.removeEventListener("tldw:auth-principal-changed", invalidate)
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
      selection?.beginLoad()
      let selectionCurrent = selection?.fence() || (() => true)
      const isCurrent = () => mountedRef.current &&
        generation === loadGenerationRef.current &&
        restoreRevision === usePlaygroundSessionStore.getState().restoreRevision &&
        selectionCurrent()
      try {
        const db = dbRef.current!
        const [history, historyDetails] = await Promise.all([
          db.getChatHistory(conversationId),
          db.getHistoryInfo(conversationId)
        ])
        if (!isCurrent()) return false

        if (!isCurrent()) return false
        if (selection) {
          if (!await selection.loadConversation({ historyId: conversationId })) return false
          if (generation !== loadGenerationRef.current) return false
          selectionCurrent = selection.fence()
        }
        if (!isCurrent()) return false
        const selected = selection?.getCurrent()
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
      }
    },
    [
      selection,
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
