import React from "react"
import { useNavigate } from "react-router-dom"
import { Modal } from "antd"
import { shallow } from "zustand/shallow"
import { useChatBaseState } from "@/hooks/chat/useChatBaseState"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { useStoreMessageOption } from "@/store/option"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { cleanupAntOverlays } from "@/utils/cleanup-ant-overlays"
import { normalizeConversationState } from "@/utils/conversation-state"
import { updatePageTitle } from "@/utils/update-page-title"
import { collectGreetings } from "@/utils/character-greetings"
import type { ServerChatSummary } from "@/services/tldw/TldwApiClient"
import {
  characterToAssistantSelection,
  personaToAssistantSelection
} from "@/types/assistant-selection"
import type { Character } from "@/types/character"
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

const normalizeSelectedCharacter = (raw: unknown): Character | null => {
  if (!raw || typeof raw !== "object") return null
  const candidate = raw as Record<string, unknown>
  const idSource = resolveCharacterId(
    candidate.id ?? candidate.slug ?? candidate.name ?? candidate.title
  )
  const nameSource = (
    candidate.name ??
    candidate.title ??
    candidate.slug
  ) as string | undefined
  if (!idSource || typeof nameSource !== "string" || nameSource.trim().length === 0) {
    return null
  }
  const greetings = collectGreetings(candidate as any)
  return {
    id: String(idSource),
    name: nameSource.trim(),
    avatar_url:
      typeof candidate.avatar_url === "string" ? candidate.avatar_url : null,
    image_base64:
      typeof candidate.image_base64 === "string" ? candidate.image_base64 : null,
    image_mime:
      typeof candidate.image_mime === "string" ? candidate.image_mime : null,
    system_prompt:
      typeof candidate.system_prompt === "string"
        ? candidate.system_prompt
        : typeof candidate.systemPrompt === "string"
          ? candidate.systemPrompt
          : typeof candidate.instructions === "string"
            ? candidate.instructions
            : null,
    greeting: greetings[0] ?? null,
    slug: typeof candidate.slug === "string" ? candidate.slug : null,
    title: typeof candidate.title === "string" ? candidate.title : null,
    tags: Array.isArray(candidate.tags)
      ? candidate.tags
          .map((tag) => (typeof tag === "string" ? tag.trim() : ""))
          .filter((tag) => tag.length > 0)
      : undefined,
    extensions:
      candidate.extensions &&
      typeof candidate.extensions === "object" &&
      !Array.isArray(candidate.extensions)
        ? (candidate.extensions as Record<string, unknown>)
        : null,
    version:
      typeof candidate.version === "number" && Number.isFinite(candidate.version)
        ? candidate.version
        : undefined
  }
}

export const useSelectServerChat = () => {
  const navigate = useNavigate()
  const [, setSelectedAssistant] = useSelectedAssistant(null)
  const assistantSyncRequestRef = React.useRef(0)
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
      setSelectedQuickPrompt(null as unknown as string)
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
      const syncRequestId = assistantSyncRequestRef.current + 1
      assistantSyncRequestRef.current = syncRequestId
      const syncSelectedAssistant = async () => {
        if (assistantIdentity.assistantKind === "persona" && assistantIdentity.assistantId) {
          try {
            await tldwClient.initialize().catch(() => null)
            const persona = await tldwClient.getPersonaProfile(
              assistantIdentity.assistantId
            )
            if (assistantSyncRequestRef.current !== syncRequestId) return
            await setSelectedAssistant(
              personaToAssistantSelection({
                ...persona,
                id: assistantIdentity.assistantId,
                name: persona?.name || "Persona"
              })
            )
          } catch (error) {
            if (assistantSyncRequestRef.current !== syncRequestId) return
            console.warn("[useSelectServerChat] Failed to sync persona", {
              chatId: chat.id,
              assistantId: assistantIdentity.assistantId,
              error
            })
            await setSelectedAssistant(
              personaToAssistantSelection({
                id: assistantIdentity.assistantId,
                name: "Persona"
              })
            )
          }
          return
        }
        if (characterId == null) {
          await setSelectedAssistant(null)
          return
        }
        try {
          await tldwClient.initialize().catch(() => null)
          const character = await tldwClient.getCharacter(characterId)
          if (assistantSyncRequestRef.current !== syncRequestId) return
          const normalized = normalizeSelectedCharacter(character)
          await setSelectedAssistant(characterToAssistantSelection(normalized))
        } catch (error) {
          if (assistantSyncRequestRef.current !== syncRequestId) return
          console.warn("[useSelectServerChat] Failed to sync character", {
            chatId: chat.id,
            characterId,
            error
          })
          await setSelectedAssistant(null)
        }
      }
      void syncSelectedAssistant()
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
      setSelectedAssistant,
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
      setContextFiles,
      setRagMediaIds,
      setStreaming,
      setWebSearch
    ]
  )
}
