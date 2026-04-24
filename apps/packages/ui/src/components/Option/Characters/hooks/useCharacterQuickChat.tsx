import React from "react"
import {
  tldwClient,
  type ServerChatSummary
} from "@/services/tldw/TldwApiClient"
import { useStoreMessageOption } from "@/store/option"
import { shallow } from "zustand/shallow"
import { useNavigate } from "react-router-dom"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { focusComposer } from "@/hooks/useComposerFocus"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import { buildCharacterSelectionPayload } from "../utils"

type CharacterQuickChatMessage = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: number
}

const makeQuickChatMessageId = (): string => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

const resolveCharacterNumericId = (record: any): number | null => {
  const raw = record?.id ?? record?.character_id ?? record?.characterId
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null
  }
  return parsed
}

export interface UseCharacterQuickChatDeps {
  /** i18n translator */
  t: (key: string, opts?: Record<string, any>) => string
  /** Active model for quick chat (resolved from overrides/defaults) */
  activeQuickChatModel: string | null
}

export function useCharacterQuickChat(deps: UseCharacterQuickChatDeps) {
  const { t, activeQuickChatModel } = deps

  const navigate = useNavigate()
  const [, setSelectedCharacter] = useSelectedCharacter<any>(null)

  const {
    setHistory,
    setMessages,
    setHistoryId,
    setServerChatId,
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatMetaLoaded,
    setServerChatState,
    setServerChatTopic,
    setServerChatClusterId,
    setServerChatSource,
    setServerChatExternalRef
  } = useStoreMessageOption(
    (state) => ({
      setHistory: state.setHistory,
      setMessages: state.setMessages,
      setHistoryId: state.setHistoryId,
      setServerChatId: state.setServerChatId,
      setServerChatCharacterId: state.setServerChatCharacterId,
      setServerChatAssistantKind: state.setServerChatAssistantKind,
      setServerChatAssistantId: state.setServerChatAssistantId,
      setServerChatMetaLoaded: state.setServerChatMetaLoaded,
      setServerChatState: state.setServerChatState,
      setServerChatTopic: state.setServerChatTopic,
      setServerChatClusterId: state.setServerChatClusterId,
      setServerChatSource: state.setServerChatSource,
      setServerChatExternalRef: state.setServerChatExternalRef
    }),
    shallow
  )

  const [quickChatCharacter, setQuickChatCharacter] = React.useState<any | null>(null)
  const [quickChatMessages, setQuickChatMessages] = React.useState<CharacterQuickChatMessage[]>([])
  const [quickChatDraft, setQuickChatDraft] = React.useState("")
  const [quickChatSessionId, setQuickChatSessionId] = React.useState<string | null>(null)
  const [quickChatSending, setQuickChatSending] = React.useState(false)
  const [quickChatError, setQuickChatError] = React.useState<string | null>(null)

  const closeQuickChat = React.useCallback(
    async (options?: { preserveSession?: boolean }) => {
      const chatIdToDelete = quickChatSessionId
      const shouldDeleteSession =
        Boolean(chatIdToDelete) && options?.preserveSession !== true

      setQuickChatCharacter(null)
      setQuickChatMessages([])
      setQuickChatDraft("")
      setQuickChatSending(false)
      setQuickChatError(null)
      setQuickChatSessionId(null)

      if (shouldDeleteSession && chatIdToDelete) {
        try {
          await tldwClient.deleteChat(chatIdToDelete, { hardDelete: true })
        } catch {
          // Best-effort cleanup of ephemeral quick-chat session.
        }
      }
    },
    [quickChatSessionId]
  )

  const openQuickChat = React.useCallback((record: any) => {
    const characterSelection = buildCharacterSelectionPayload(record)
    setQuickChatCharacter(record)
    setQuickChatDraft("")
    setQuickChatError(null)
    setQuickChatSessionId(null)
    const greeting = characterSelection.greeting?.trim()
    setQuickChatMessages(
      greeting
        ? [
            {
              id: makeQuickChatMessageId(),
              role: "assistant",
              content: greeting,
              timestamp: Date.now()
            }
          ]
        : []
    )
  }, [])

  const sendQuickChatMessage = React.useCallback(async () => {
    const trimmed = quickChatDraft.trim()
    if (!trimmed || quickChatSending || !quickChatCharacter) return
    if (!activeQuickChatModel) {
      setQuickChatError(
        t("settings:manageCharacters.quickChat.modelRequired", {
          defaultValue: "Select a model to start quick chat."
        })
      )
      return
    }

    const userMessage: CharacterQuickChatMessage = {
      id: makeQuickChatMessageId(),
      role: "user",
      content: trimmed,
      timestamp: Date.now()
    }

    const nextHistory = [...quickChatMessages, userMessage]
    setQuickChatMessages(nextHistory)
    setQuickChatDraft("")
    setQuickChatSending(true)
    setQuickChatError(null)

    try {
      let sessionId = quickChatSessionId
      if (!sessionId) {
        const characterId = resolveCharacterNumericId(quickChatCharacter)
        if (!characterId) {
          throw new Error(
            t("settings:manageCharacters.quickChat.unsupportedCharacter", {
              defaultValue:
                "Quick chat is only available for server-synced characters."
            })
          )
        }
        const created = await tldwClient.createChat({
          character_id: characterId,
          state: "in-progress",
          source: "characters-quick-chat",
          title: t("settings:manageCharacters.quickChat.sessionTitle", {
            defaultValue: "{{name}} quick chat",
            name:
              quickChatCharacter?.name ||
              quickChatCharacter?.title ||
              quickChatCharacter?.slug ||
              t("settings:manageCharacters.preview.untitled", {
                defaultValue: "Untitled character"
              })
          })
        })
        const rawId = (created as any)?.id ?? (created as any)?.chat_id ?? created
        sessionId = rawId != null ? String(rawId) : ""
        if (!sessionId) {
          throw new Error(
            t("settings:manageCharacters.quickChat.sessionCreateFailed", {
              defaultValue: "Unable to start a quick chat session."
            })
          )
        }
        setQuickChatSessionId(sessionId)
      }

      const payload = await tldwClient.completeCharacterChatTurn(sessionId, {
        append_user_message: trimmed,
        include_character_context: true,
        limit: 100,
        save_to_db: true,
        stream: false,
        model: activeQuickChatModel
      })
      const assistantContent = (
        typeof payload?.assistant_content === "string"
          ? payload.assistant_content
          : typeof payload?.content === "string"
            ? payload.content
            : typeof payload?.text === "string"
              ? payload.text
              : ""
      ).trim()

      if (!assistantContent) {
        throw new Error(
          t("settings:manageCharacters.quickChat.emptyResponse", {
            defaultValue: "No response received from the model."
          })
        )
      }

      setQuickChatMessages((previous) => [
        ...previous,
        {
          id: makeQuickChatMessageId(),
          role: "assistant",
          content: assistantContent,
          timestamp: Date.now()
        }
      ])
    } catch (error: any) {
      const message =
        error?.message ||
        t("settings:manageCharacters.quickChat.error", {
          defaultValue: "Quick chat failed. Please try again."
        })
      setQuickChatError(message)
    } finally {
      setQuickChatSending(false)
    }
  }, [
    activeQuickChatModel,
    quickChatCharacter,
    quickChatDraft,
    quickChatMessages,
    quickChatSessionId,
    quickChatSending,
    t
  ])

  const handlePromoteQuickChat = React.useCallback(async () => {
    if (!quickChatCharacter) return

    const characterSelection = buildCharacterSelectionPayload(quickChatCharacter)
    setSelectedCharacter(characterSelection)

    const assistantName =
      characterSelection.name ||
      t("common:assistant", {
        defaultValue: "Assistant"
      })
    const history = quickChatMessages.map((message) => ({
      role: message.role,
      content: message.content
    }))
    const mappedMessages = quickChatMessages.map((message) => ({
      createdAt: message.timestamp,
      isBot: message.role === "assistant",
      role: message.role,
      name:
        message.role === "assistant"
          ? assistantName
          : t("common:you", { defaultValue: "You" }),
      message: message.content,
      sources: [],
      images: []
    }))
    const normalizedAssistantId =
      characterSelection.id == null ? null : String(characterSelection.id)

    setHistoryId(null)
    setServerChatId(quickChatSessionId)
    setServerChatCharacterId(normalizedAssistantId)
    setServerChatAssistantKind("character")
    setServerChatAssistantId(normalizedAssistantId)
    setServerChatMetaLoaded(false)
    setServerChatState("in-progress")
    setServerChatTopic(null)
    setServerChatClusterId(null)
    setServerChatSource("characters-quick-chat")
    setServerChatExternalRef(null)
    setHistory(history)
    setMessages(mappedMessages)

    await closeQuickChat({ preserveSession: true })
    navigate("/")
    setTimeout(() => {
      focusComposer()
    }, 0)
  }, [
    closeQuickChat,
    navigate,
    quickChatCharacter,
    quickChatMessages,
    quickChatSessionId,
    setHistory,
    setHistoryId,
    setMessages,
    setSelectedCharacter,
    setServerChatAssistantId,
    setServerChatAssistantKind,
    setServerChatCharacterId,
    setServerChatClusterId,
    setServerChatExternalRef,
    setServerChatId,
    setServerChatMetaLoaded,
    setServerChatSource,
    setServerChatState,
    setServerChatTopic,
    t
  ])

  return {
    // state
    quickChatCharacter,
    setQuickChatCharacter,
    quickChatMessages,
    setQuickChatMessages,
    quickChatDraft,
    setQuickChatDraft,
    quickChatSessionId,
    setQuickChatSessionId,
    quickChatSending,
    quickChatError,
    setQuickChatError,
    // callbacks
    openQuickChat,
    closeQuickChat,
    sendQuickChatMessage,
    handlePromoteQuickChat
  }
}
