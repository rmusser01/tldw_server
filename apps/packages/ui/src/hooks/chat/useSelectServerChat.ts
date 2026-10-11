import React from "react"
import { captureChatDraftLoad, captureChatDraftSource, confirmTemporaryChatDiscard } from "@/hooks/chat/chat-draft-transition"
import { useNavigate } from "react-router-dom"
import { Modal } from "antd"
import { shallow } from "zustand/shallow"
import { useChatBaseState } from "@/hooks/chat/useChatBaseState"
import { useHistorySelectionContext } from "@/hooks/chat/useHistorySelection"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { cleanupAntOverlays } from "@/utils/cleanup-ant-overlays"
import { normalizeConversationState } from "@/utils/conversation-state"
import { updatePageTitle } from "@/utils/update-page-title"
import { requestSettingsNavigation } from "@/utils/settings-return"
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
  const captureSelectionFence = useHistorySelectionContext()?.fence
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
    (chat: ServerChatSummary, options?: {
      discardTemporaryConfirmed?: true
      settingsNavigationConfirmed?: true
      temporaryChat?: false
      isCurrent?: () => boolean
    }) => {
      const callerIsCurrent = options?.isCurrent ?? (() => true)
      if (!callerIsCurrent()) return false
      const selectionIsCurrent = captureSelectionFence?.() ?? (() => true)
      const source = captureChatDraftSource(selectionIsCurrent)
      const chatPath = typeof window !== "undefined" &&
        window.location.pathname.toLowerCase().endsWith("sidepanel.html") ? "/" : "/chat"
      if (options?.settingsNavigationConfirmed !== true && !requestSettingsNavigation(chatPath)) return false
      if (!source.isCurrent() || !callerIsCurrent()) return false
      if (options?.discardTemporaryConfirmed !== true && !confirmTemporaryChatDiscard()) return false
      if (!source.isCurrent() || !callerIsCurrent()) return false
      // A mode publication must not adopt a New/load published by a synchronous subscriber.
      if (options?.temporaryChat === false) {
        const before = useStoreMessageOption.getState()
        const session = usePlaygroundSessionStore.getState()
        before.setTemporaryChat(false)
        const after = useStoreMessageOption.getState()
        const changed = before.temporaryChat !== false
        const addressCurrent = !changed || before.chatDraftAddress?.kind !== "fresh"
          ? after.chatDraftAddress === before.chatDraftAddress
          : after.chatDraftAddress?.kind === "fresh" && after.chatDraftAddress.id === before.chatDraftAddress.id &&
            after.chatDraftAddress.scope === before.chatDraftAddress.scope && after.chatDraftAddress.temporary === false
        if (!callerIsCurrent() || !selectionIsCurrent() || !addressCurrent ||
          after.chatDraftRevision !== before.chatDraftRevision + (changed ? 1 : 0) ||
          after.historyId !== before.historyId || after.serverChatId !== before.serverChatId ||
          usePlaygroundSessionStore.getState().restoreRevision !== session.restoreRevision ||
          usePlaygroundSessionStore.getState().sourceSelectionRevision !== session.sourceSelectionRevision) return false
      }
      if (!callerIsCurrent()) return false
      const acceptDraft = captureChatDraftLoad("switch")
      usePlaygroundSessionStore.getState().cancelPendingRestore()
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
      setServerChatMetaLoaded(assistantIdentity.assistantKind !== "character")
      usePlaygroundSessionStore.getState().requestServerChatSelection(chat.id, acceptDraft)
      updatePageTitle(chat.title)
      navigate(chatPath)
      return true
    },
    [
      clearReplyTarget,
      captureSelectionFence,
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
