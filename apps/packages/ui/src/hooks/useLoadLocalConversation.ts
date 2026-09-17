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
  const loadEpoch = React.useRef(0)
  React.useEffect(() => () => { loadEpoch.current += 1 }, [])
  const dbRef = React.useRef<PageAssistDatabase | null>(null)

  if (!dbRef.current) {
    dbRef.current = new PageAssistDatabase()
  }

  return React.useCallback(
    async (conversationId: string) => {
      const token = ++loadEpoch.current
      selection?.beginLoad()
      let selectionCurrent = selection?.fence() || (() => true)
      const isCurrent = () => token === loadEpoch.current && selectionCurrent()
      try {
        const db = dbRef.current!
        const [history, historyDetails] = await Promise.all([
          db.getChatHistory(conversationId),
          db.getHistoryInfo(conversationId)
        ])

        if (!isCurrent()) return
        if (selection) {
          if (!await selection.loadConversation({ historyId: conversationId })) return
          if (token !== loadEpoch.current) return
          selectionCurrent = selection.fence()
        }
        if (!isCurrent()) return
        const selected = selection?.getCurrent()
        setServerChatId(selected?.owner?.kind === "native" ? selected.owner.conversation_id : null)
        setHistoryId(conversationId)
        if (!selected || selected.capture?.status !== "captured") {
          setHistory(formatToChatHistory(history))
          setMessages(formatToMessage(history))
        }

        const isLastUsedChatModel = await lastUsedChatModelEnabled()
        if (!isCurrent()) return
        if (isLastUsedChatModel && historyDetails?.model_id) {
          setSelectedModel(historyDetails.model_id)
        }

        const lastUsedPrompt = historyDetails?.last_used_prompt
        if (lastUsedPrompt) {
          let promptContent = lastUsedPrompt.prompt_content ?? ""
          if (lastUsedPrompt.prompt_id) {
            const prompt = await getPromptById(lastUsedPrompt.prompt_id)
            if (!isCurrent()) return
            if (prompt) {
              setSelectedSystemPrompt(prompt.id)
              if (!promptContent.trim()) {
                promptContent = prompt.content
              }
            }
          }
          setSystemPrompt(promptContent)
        }

        const session = await getSessionFiles(conversationId)
        if (!isCurrent()) return
        setContextFiles(session)

        updatePageTitle(
          historyDetails?.title || t("common:untitled", { defaultValue: "Untitled" })
        )
      } catch (error) {
        if (!isCurrent()) return
        // eslint-disable-next-line no-console
        console.error(`${errorLogPrefix}:`, error)
        message.error(
          t("common:error.friendlyLocalHistorySummary", {
            defaultValue: errorDefaultMessage
          })
        )
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
