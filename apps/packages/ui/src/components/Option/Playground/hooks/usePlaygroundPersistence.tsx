import React from "react"
import { Modal } from "antd"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { usePersistenceMode } from "@/hooks/playground"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import type { Character } from "@/types/character"
import {
  characterToAssistantSelection,
  type AssistantSelectionMode
} from "@/types/assistant-selection"
import {
  WEBUI_CHARACTER_CHAT_SOURCE,
  WEBUI_CHAT_SOURCE,
  buildCharacterChatSessionTitle
} from "@/utils/character-chat-session"

// ---------------------------------------------------------------------------
// Deps interface
// ---------------------------------------------------------------------------

export interface UsePlaygroundPersistenceDeps {
  isFireFoxPrivateMode: boolean
  isConnectionReady: boolean
  temporaryChat: boolean
  setTemporaryChat: (value: boolean) => void
  serverChatId: string | null
  setServerChatId: (id: string) => void
  historyId: string | null
  serverChatState: string | null
  setServerChatState: (state: string) => void
  serverChatSource: string | null
  setServerChatSource: (source: string | null) => void
  setServerChatVersion: (version: number | null) => void
  setServerChatCharacterId: (id: string | number | null) => void
  setServerChatAssistantKind: (kind: "character" | "persona" | null) => void
  setServerChatAssistantId: (id: string | number | null) => void
  setServerChatPersonaMemoryMode: (
    mode: "read_only" | "read_write" | null
  ) => void
  history: Array<{ role: string; content?: string; image?: string }>
  clearChat: () => void
  selectedCharacter: Character | null
  selectedAssistantMode: AssistantSelectionMode | null
  assistantOverlayActive: boolean
  serverPersistenceHintSeen: boolean
  setServerPersistenceHintSeen: (value: boolean) => void
  invalidateServerChatHistory: () => void
  navigate: (path: string) => void
  notificationApi: {
    error: (opts: Record<string, any>) => void
    warning: (opts: Record<string, any>) => void
    info: (opts: Record<string, any>) => void
    success: (opts: Record<string, any>) => void
  }
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
}

const persistTrackedCharacterPlaygroundSession = async ({
  chatId,
  historyId,
  character
}: {
  chatId: string
  historyId: string | null
  character: Character
}) => {
  const selection = characterToAssistantSelection(
    character as Character & Record<string, unknown>
  )
  if (!selection) return

  const trackedSelection = {
    ...selection,
    metadata: {
      ...(selection.metadata ?? {}),
      selectionMode: "tracked" as const
    }
  }

  try {
    const config = await tldwClient.getConfig().catch(() => null)
    const scopeKey = buildChatSurfaceScopeKeyFromConfig(config)
    usePlaygroundSessionStore.getState().saveSession({
      historyId,
      serverChatId: chatId,
      trackedAssistantSelection: trackedSelection,
      trackedAssistantKind: "character",
      trackedAssistantId: trackedSelection.id,
      trackedCharacterId: trackedSelection.id,
      trackedAssistantDisplayName: trackedSelection.name ?? null,
      trackedAssistantAvatarUrl: trackedSelection.avatar_url ?? null,
      serverChatPersonaMemoryMode: null,
      scopeKey
    })
  } catch (error) {
    console.warn(
      "[usePlaygroundPersistence] Failed to persist tracked character session",
      error
    )
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function usePlaygroundPersistence(deps: UsePlaygroundPersistenceDeps) {
  const {
    isFireFoxPrivateMode,
    isConnectionReady,
    temporaryChat,
    setTemporaryChat,
    serverChatId,
    setServerChatId,
    historyId,
    serverChatState,
    setServerChatState,
    serverChatSource,
    setServerChatSource,
    setServerChatVersion,
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatPersonaMemoryMode,
    history,
    clearChat,
    selectedCharacter,
    selectedAssistantMode,
    assistantOverlayActive,
    serverPersistenceHintSeen,
    setServerPersistenceHintSeen,
    invalidateServerChatHistory,
    notificationApi,
    t
  } = deps

  const [showServerPersistenceHint, setShowServerPersistenceHint] =
    React.useState(false)
  const serverSaveInFlightRef = React.useRef(false)
  const historyRef = React.useRef(history)
  const selectedCharacterRef = React.useRef(selectedCharacter)
  const selectedAssistantModeRef = React.useRef(selectedAssistantMode)
  const assistantOverlayActiveRef = React.useRef(assistantOverlayActive)
  const serverChatStateRef = React.useRef(serverChatState)
  const serverChatSourceRef = React.useRef(serverChatSource)
  const serverPersistenceHintSeenRef = React.useRef(serverPersistenceHintSeen)

  React.useEffect(() => {
    historyRef.current = history
  }, [history])

  React.useEffect(() => {
    selectedCharacterRef.current = selectedCharacter
  }, [selectedCharacter])

  React.useEffect(() => {
    selectedAssistantModeRef.current = selectedAssistantMode
  }, [selectedAssistantMode])

  React.useEffect(() => {
    assistantOverlayActiveRef.current = assistantOverlayActive
  }, [assistantOverlayActive])

  React.useEffect(() => {
    serverChatStateRef.current = serverChatState
  }, [serverChatState])

  React.useEffect(() => {
    serverChatSourceRef.current = serverChatSource
  }, [serverChatSource])

  React.useEffect(() => {
    serverPersistenceHintSeenRef.current = serverPersistenceHintSeen
  }, [serverPersistenceHintSeen])

  const {
    persistenceTooltip,
    focusConnectionCard,
    getPersistenceModeLabel
  } = usePersistenceMode({
    temporaryChat,
    serverChatId,
    isConnectionReady
  })

  const privateChatLocked = temporaryChat && history.length > 0

  const handleToggleTemporaryChat = React.useCallback(
    (next: boolean) => {
      if (isFireFoxPrivateMode) {
        notificationApi.error({
          message: t(
            "common:privateModeSaveErrorTitle",
            "tldw Assistant can't save data"
          ),
          description: t(
            "playground:errors.privateModeDescription",
            "Firefox Private Mode does not support saving chat. Temporary chat is enabled by default. More fixes coming soon."
          )
        })
        return
      }

      const hasExistingHistory = history.length > 0

      if (!next && temporaryChat && hasExistingHistory) {
        notificationApi.warning({
          message: t(
            "playground:composer.privateChatLockedTitle",
            "Private chat is locked"
          ),
          description: t(
            "playground:composer.privateChatLockedBody",
            "Start a new chat to switch back to saved conversations."
          )
        })
        return
      }

      if (next && hasExistingHistory) {
        Modal.confirm({
          title: t(
            "playground:composer.tempChatConfirmTitle",
            "Enable temporary mode?"
          ),
          content: t(
            "playground:composer.tempChatConfirmContent",
            "This will clear your current conversation. Messages won't be saved."
          ),
          okText: t("common:confirm", "Confirm"),
          cancelText: t("common:cancel", "Cancel"),
          onOk: () => {
            setTemporaryChat(next)
            clearChat()
            const modeLabel = getPersistenceModeLabel(
              t,
              next,
              isConnectionReady,
              serverChatId
            )
            notificationApi.info({
              message: modeLabel,
              placement: "bottomRight",
              duration: 2.5
            })
          }
        })
        return
      }

      setTemporaryChat(next)
      if (hasExistingHistory) {
        clearChat()
      }

      const modeLabel = getPersistenceModeLabel(
        t,
        next,
        isConnectionReady,
        serverChatId
      )

      notificationApi.info({
        message: modeLabel,
        placement: "bottomRight",
        duration: 2.5
      })
    },
    [
      clearChat,
      history.length,
      isConnectionReady,
      notificationApi,
      serverChatId,
      setTemporaryChat,
      t,
      temporaryChat,
      getPersistenceModeLabel
    ]
  )

  const handleSaveChatToServer = React.useCallback(async () => {
    if (serverSaveInFlightRef.current) return
    serverSaveInFlightRef.current = true
    try {
      const snapshot = [...historyRef.current]
      if (
        !isConnectionReady ||
        temporaryChat ||
        serverChatId ||
        snapshot.length === 0
      ) {
        return
      }
      await tldwClient.initialize()
      const firstUser = snapshot.find((m) => m.role === "user")
      const isOverlaySelection =
        assistantOverlayActiveRef.current ||
        selectedAssistantModeRef.current === "overlay"
      const selectedAssistantMode = selectedAssistantModeRef.current
      const shouldPersistTrackedCharacter =
        !isOverlaySelection && selectedAssistantMode === "tracked"
      const trackedCharacterSnapshot =
        shouldPersistTrackedCharacter && selectedCharacterRef.current?.id != null
          ? selectedCharacterRef.current
          : null
      let characterId: string | number | null = shouldPersistTrackedCharacter
        ? ((trackedCharacterSnapshot as any)?.id ?? null)
        : null
      const selectedCharacterName =
        shouldPersistTrackedCharacter &&
        typeof trackedCharacterSnapshot?.name === "string"
          ? trackedCharacterSnapshot.name.trim()
          : ""
      const explicitSource =
        serverChatSourceRef.current &&
        serverChatSourceRef.current.trim().length > 0
          ? serverChatSourceRef.current.trim()
          : null
      const isCharacterPersistence = Boolean(characterId)
      const title =
        isCharacterPersistence && selectedCharacterName
          ? buildCharacterChatSessionTitle({
              characterName: selectedCharacterName,
              firstUserMessage: firstUser?.content,
              fallbackTitle: String(
                t(
                  "playground:composer.persistence.characterFallbackTitle",
                  "{{name}} role-play",
                  { name: selectedCharacterName }
                )
              )
            })
          : (() => {
              const fallbackTitle =
                explicitSource === "extension"
                  ? t(
                      "playground:composer.persistence.serverDefaultTitle",
                      "Extension chat"
                    )
                  : t(
                      "playground:composer.persistence.serverWebUiDefaultTitle",
                      "WebUI chat"
                    )
              const titleSource =
                typeof firstUser?.content === "string" &&
                firstUser.content.trim().length > 0
                  ? firstUser.content.trim()
                  : fallbackTitle
              return titleSource.length > 80
                ? `${titleSource.slice(0, 77)}…`
                : titleSource
            })()

      if (selectedAssistantMode === "tracked" && characterId == null) {
        notificationApi.error({
          key: "playground-server-character-required",
          message: t("error"),
          description: t(
            "playground:composer.persistence.serverCharacterRequired",
            "Unable to save this tracked character chat because the selected character is missing. Re-select the character and try again."
          ),
          duration: 6
        })
        return
      }

      const createPayload = {
        title,
        ...(characterId != null ? { character_id: characterId } : {}),
        state: serverChatStateRef.current || "in-progress",
        source:
          explicitSource ||
          (isCharacterPersistence
            ? WEBUI_CHARACTER_CHAT_SOURCE
            : WEBUI_CHAT_SOURCE)
      }
      const created = await tldwClient.createChat(createPayload)
      const rawId = (created as any)?.id ?? (created as any)?.chat_id ?? created
      const cid = rawId != null ? String(rawId) : ""
      if (!cid) {
        throw new Error("Failed to create server chat")
      }
      setServerChatId(cid)
      setServerChatState(
        (created as any)?.state ??
          (created as any)?.conversation_state ??
          serverChatStateRef.current ??
          "in-progress"
      )
      setServerChatSource(
        (created as any)?.source ?? serverChatSourceRef.current ?? null
      )
      setServerChatVersion((created as any)?.version ?? null)
      if (isCharacterPersistence && trackedCharacterSnapshot) {
        setServerChatCharacterId(characterId)
        setServerChatAssistantKind("character")
        setServerChatAssistantId(characterId)
        setServerChatPersonaMemoryMode(null)
        await persistTrackedCharacterPlaygroundSession({
          chatId: cid,
          historyId,
          character: trackedCharacterSnapshot
        })
      }
      invalidateServerChatHistory()

      for (const msg of snapshot) {
        const content = (msg.content || "").trim()
        if (!content) continue
        const role =
          msg.role === "system" ||
          msg.role === "assistant" ||
          msg.role === "user"
            ? msg.role
            : "user"
        await tldwClient.addChatMessage(cid, {
          role,
          content
        })
      }

      if (!serverPersistenceHintSeenRef.current) {
        serverPersistenceHintSeenRef.current = true
        setServerPersistenceHintSeen(true)
        setShowServerPersistenceHint(true)
      }
    } catch (e: any) {
      notificationApi.error({
        message: t("error"),
        description: e?.message || t("somethingWentWrong")
      })
    } finally {
      serverSaveInFlightRef.current = false
    }
  }, [
    invalidateServerChatHistory,
    isConnectionReady,
    notificationApi,
    temporaryChat,
    serverChatId,
    setServerChatId,
    historyId,
    setServerPersistenceHintSeen,
    t,
    setServerChatState,
    setServerChatSource,
    setServerChatVersion,
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatPersonaMemoryMode
  ])

  // Auto-save to server
  React.useEffect(() => {
    if (
      !isConnectionReady ||
      temporaryChat ||
      serverChatId ||
      history.length === 0
    ) {
      return
    }
    void handleSaveChatToServer()
  }, [
    handleSaveChatToServer,
    history.length,
    isConnectionReady,
    selectedCharacter?.id,
    serverChatId,
    temporaryChat
  ])

  const persistChatMetadata = React.useCallback(
    async (patch: Record<string, any>) => {
      if (!serverChatId) return
      try {
        const updated = await tldwClient.updateChat(serverChatId, patch)
        setServerChatState(
          (updated as any)?.state ??
            (updated as any)?.conversation_state ??
            "in-progress"
        )
        setServerChatSource((updated as any)?.source ?? null)
        setServerChatVersion((updated as any)?.version ?? null)
        invalidateServerChatHistory()
      } catch (e: any) {
        notificationApi.error({
          message: t("error", { defaultValue: "Error" }),
          description:
            e?.message ||
            t("somethingWentWrong", { defaultValue: "Something went wrong" })
        })
      }
    },
    [
      invalidateServerChatHistory,
      notificationApi,
      serverChatId,
      setServerChatSource,
      setServerChatState,
      setServerChatVersion,
      t
    ]
  )

  const handleDismissServerPersistenceHint = React.useCallback(() => {
    setShowServerPersistenceHint(false)
  }, [setShowServerPersistenceHint])

  return {
    persistenceTooltip,
    focusConnectionCard,
    getPersistenceModeLabel,
    privateChatLocked,
    showServerPersistenceHint,
    handleToggleTemporaryChat,
    handleSaveChatToServer,
    persistChatMetadata,
    handleDismissServerPersistenceHint
  }
}

export type UsePlaygroundPersistenceReturn = ReturnType<typeof usePlaygroundPersistence>
