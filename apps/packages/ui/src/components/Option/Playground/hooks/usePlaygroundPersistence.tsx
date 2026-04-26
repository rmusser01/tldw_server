import React from "react"
import { Button, Modal } from "antd"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { usePersistenceMode } from "@/hooks/playground"
import type { Character } from "@/types/character"

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
  serverChatState: string | null
  setServerChatState: (state: string) => void
  serverChatSource: string | null
  setServerChatSource: (source: string | null) => void
  setServerChatVersion: (version: string | null) => void
  history: Array<{ role: string; content?: string; image?: string }>
  clearChat: () => void
  selectedCharacter: Character | null
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
    serverChatState,
    setServerChatState,
    serverChatSource,
    setServerChatSource,
    setServerChatVersion,
    history,
    clearChat,
    selectedCharacter,
    serverPersistenceHintSeen,
    setServerPersistenceHintSeen,
    invalidateServerChatHistory,
    navigate,
    notificationApi,
    t
  } = deps

  const [showServerPersistenceHint, setShowServerPersistenceHint] =
    React.useState(false)
  const serverSaveInFlightRef = React.useRef(false)
  const historyRef = React.useRef(history)
  const selectedCharacterRef = React.useRef(selectedCharacter)
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
      const fallbackTitle = t(
        "playground:composer.persistence.serverDefaultTitle",
        "Extension chat"
      )
      const titleSource =
        typeof firstUser?.content === "string" &&
        firstUser.content.trim().length > 0
          ? firstUser.content.trim()
          : fallbackTitle
      const title =
        titleSource.length > 80 ? `${titleSource.slice(0, 77)}…` : titleSource

      let characterId: string | number | null =
        (selectedCharacterRef.current as any)?.id ?? null

      if (!characterId) {
        const DEFAULT_NAME = "Helpful AI Assistant"
        const normalizeName = (value: unknown) =>
          String(value || "").trim().toLowerCase()
        const findByName = (list: any[]) =>
          (list || []).find(
            (c: any) => normalizeName(c?.name) === normalizeName(DEFAULT_NAME)
          )
        const findDefaultCharacter = async () => {
          try {
            const results = await tldwClient.searchCharacters(DEFAULT_NAME, {
              limit: 50
            })
            const match = findByName(results)
            if (match) return match
          } catch {}
          try {
            const results = await tldwClient.listCharacters({ limit: 200 })
            const match = findByName(results)
            if (match) return match
          } catch {}
          return null
        }
        try {
          let target = await findDefaultCharacter()
          if (!target) {
            try {
              target = await tldwClient.createCharacter({
                name: DEFAULT_NAME
              })
            } catch (error: any) {
              if (error?.status === 409) {
                target = await findDefaultCharacter()
              } else {
                throw error
              }
            }
          }
          characterId =
            target && typeof target.id !== "undefined" ? target.id : null
        } catch {
          characterId = null
        }
      }

      if (characterId == null) {
        notificationApi.error({
          key: "playground-server-character-required",
          message: t("error"),
          description: t(
            "playground:composer.persistence.serverCharacterRequired",
            "Unable to find or create a default assistant character on the server. Try again from the Characters page."
          ),
          btn: (
            <Button
              type="primary"
              size="small"
              title={t(
                "playground:composer.persistence.serverCharacterCta",
                "Open Characters workspace"
              ) as string}
              onClick={() => {
                navigate("/characters?from=server-chat-persistence-error")
              }}>
              {t(
                "playground:composer.persistence.serverCharacterCta",
                "Open Characters workspace"
              )}
            </Button>
          ),
          duration: 6
        })
        return
      }

      const created = await tldwClient.createChat({
        title,
        character_id: characterId,
        state: serverChatStateRef.current || "in-progress",
        source:
          serverChatSourceRef.current &&
          serverChatSourceRef.current.trim().length > 0
            ? serverChatSourceRef.current.trim()
            : undefined
      })
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
        notificationApi.success({
          message: t(
            "playground:composer.persistence.serverSavedTitle",
            "Chat now saved on server"
          ),
          description:
            t(
              "playground:composer.persistence.serverSaved",
              "Future messages in this chat will sync to your tldw server."
            ) +
            " " +
            t(
              "playground:composer.persistence.serverBenefits",
              "This keeps a durable record in server history so you can reopen the conversation later, access it from other browsers, and run server-side analytics over your chats."
            )
        })
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
    navigate,
    setServerPersistenceHintSeen,
    t,
    setServerChatState,
    setServerChatSource,
    setServerChatVersion
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
