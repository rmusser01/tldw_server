import React from "react"
import { useNavigate } from "react-router-dom"
import { Modal } from "antd"
import { shallow } from "zustand/shallow"
import { useChatBaseState } from "@/hooks/chat/useChatBaseState"
import { useStoreMessageOption } from "@/store/option"
import { cleanupAntOverlays } from "@/utils/cleanup-ant-overlays"
import { normalizeConversationState } from "@/utils/conversation-state"
import { updatePageTitle } from "@/utils/update-page-title"
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient"
import { resolveServerChatAssistantIdentity } from "@/hooks/chat/useServerChatLoader"

const resolveCharacterId = (value: unknown): string | number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (trimmed.length > 0) return trimmed
  }
  return null
}

export const useSelectServerChat = () => {
  const navigate = useNavigate()
  const {
    setHistory,
    setHistoryId,
    setMessages,
    setIsLoading,
    setIsProcessing,
    setStreaming,
    setIsEmbedding
  } = useChatBaseState(useStoreMessageOption)
  const {
    setIsSearchingInternet,
    clearReplyTarget,
    setServerChatId,
    setServerChatTitle,
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatPersonaMemoryMode,
    setServerChatVersion,
    setServerChatLoadState,
    setServerChatLoadError,
    setServerChatState,
    setServerChatTopic,
    setServerChatClusterId,
    setServerChatSource,
    setServerChatExternalRef,
    setServerChatMetaLoaded,
    setWebSearch,
    setSelectedSystemPrompt,
    setSelectedQuickPrompt,
    setContextFiles,
    setSelectedKnowledge,
    setRagMediaIds
  } = useStoreMessageOption(
    (state) => ({
      setIsSearchingInternet: state.setIsSearchingInternet,
      clearReplyTarget: state.clearReplyTarget,
      setServerChatId: state.setServerChatId,
      setServerChatTitle: state.setServerChatTitle,
      setServerChatCharacterId: state.setServerChatCharacterId,
      setServerChatAssistantKind: state.setServerChatAssistantKind,
      setServerChatAssistantId: state.setServerChatAssistantId,
      setServerChatPersonaMemoryMode: state.setServerChatPersonaMemoryMode,
      setServerChatVersion: state.setServerChatVersion,
      setServerChatLoadState: state.setServerChatLoadState,
      setServerChatLoadError: state.setServerChatLoadError,
      setServerChatState: state.setServerChatState,
      setServerChatTopic: state.setServerChatTopic,
      setServerChatClusterId: state.setServerChatClusterId,
      setServerChatSource: state.setServerChatSource,
      setServerChatExternalRef: state.setServerChatExternalRef,
      setServerChatMetaLoaded: state.setServerChatMetaLoaded,
      setWebSearch: state.setWebSearch,
      setSelectedSystemPrompt: state.setSelectedSystemPrompt,
      setSelectedQuickPrompt: state.setSelectedQuickPrompt,
      setContextFiles: state.setContextFiles,
      setSelectedKnowledge: state.setSelectedKnowledge,
      setRagMediaIds: state.setRagMediaIds
    }),
    shallow
  )

  return React.useCallback(
    (chat: ServerChatSummary) => {
      if (typeof window !== "undefined") {
        Modal.destroyAll()
        cleanupAntOverlays()
      }
      setIsLoading(true)
      setHistoryId(null)
      setHistory([])
      setMessages([])
      setWebSearch(false)
      setSelectedSystemPrompt("")
      setSelectedQuickPrompt(null)
      setContextFiles([])
      setSelectedKnowledge(null)
      setRagMediaIds(null)
      setServerChatId(chat.id)
      setServerChatTitle(chat.title || "")
      const assistantIdentity = resolveServerChatAssistantIdentity(
        chat as unknown as Record<string, unknown>
      )
      const characterId = resolveCharacterId(assistantIdentity.characterId)
      setServerChatCharacterId(characterId)
      setServerChatAssistantKind(assistantIdentity.assistantKind)
      setServerChatAssistantId(assistantIdentity.assistantId)
      setServerChatPersonaMemoryMode(assistantIdentity.personaMemoryMode)
      setServerChatLoadState("loading")
      setServerChatLoadError(null)
      setIsProcessing(false)
      setStreaming(false)
      setIsEmbedding(false)
      setIsSearchingInternet(false)
      clearReplyTarget()
      setServerChatVersion(chat.version ?? null)
      setServerChatState(normalizeConversationState(chat.state))
      setServerChatTopic(chat.topic_label ?? null)
      setServerChatClusterId(chat.cluster_id ?? null)
      setServerChatSource(chat.source ?? null)
      setServerChatExternalRef(chat.external_ref ?? null)
      setServerChatMetaLoaded(true)
      updatePageTitle(chat.title)
      const chatPath = typeof window !== "undefined" &&
        window.location.pathname.toLowerCase().endsWith("sidepanel.html")
        ? "/"
        : "/chat"
      navigate(chatPath)
    },
    [
      clearReplyTarget,
      navigate,
      setHistory,
      setHistoryId,
      setIsEmbedding,
      setIsLoading,
      setIsProcessing,
      setIsSearchingInternet,
      setMessages,
      setContextFiles,
      setRagMediaIds,
      setSelectedKnowledge,
      setSelectedQuickPrompt,
      setSelectedSystemPrompt,
      setServerChatAssistantId,
      setServerChatAssistantKind,
      setServerChatCharacterId,
      setServerChatClusterId,
      setServerChatExternalRef,
      setServerChatId,
      setServerChatLoadError,
      setServerChatLoadState,
      setServerChatMetaLoaded,
      setServerChatPersonaMemoryMode,
      setServerChatSource,
      setServerChatState,
      setServerChatTitle,
      setServerChatTopic,
      setServerChatVersion,
      setStreaming,
      setWebSearch
    ]
  )
}
