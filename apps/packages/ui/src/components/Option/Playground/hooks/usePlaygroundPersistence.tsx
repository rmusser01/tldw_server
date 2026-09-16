import React from "react"
import type { Message } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { acknowledgePromotedChatMessage, serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { Modal } from "antd"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { usePersistenceMode } from "@/hooks/playground"
import type { Character } from "@/types/character"
import { type AssistantSelectionMode } from "@/types/assistant-selection"
import { WEBUI_CHAT_SOURCE } from "@/utils/character-chat-session"
import {
  loadServicePromptSnapshot,
  type ServicePromptSnapshot
} from "@/services/service-prompts"
import { isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import {
  clearChatPromotion,
  isChatPromotionIncompleteError,
  retryChatPromotion,
  trackChatPromotion
} from "@/services/pending-chat-promotion"

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
  messages?: Message[]
  setMessages?: (updater: (messages: Message[]) => Message[]) => void
  history: Array<{ role: string; content?: string; image?: string }>
  clearChat: () => void
  selectedCharacter: Character | null
  selectedAssistantMode: AssistantSelectionMode | null
  characterWorkflowActive?: boolean
  assistantOverlayActive: boolean
  serverPersistenceHintSeen: boolean
  setServerPersistenceHintSeen: (value: boolean) => void
  invalidateServerChatHistory: () => void
  navigate: (path: string) => void
  notificationApi: {
    destroy?: (key?: string) => void
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
    selectedAssistantMode,
    characterWorkflowActive = false,
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
  const activeSaveControllerRef = React.useRef<AbortController | null>(null)
  const latestDepsRef = React.useRef(deps)
  latestDepsRef.current = deps
  React.useEffect(() => () => activeSaveControllerRef.current?.abort(), [])
  React.useEffect(
    () => () => clearChatPromotion(setServerChatId, deps.historyId),
    [setServerChatId, deps.historyId, temporaryChat]
  )
  const historyRef = React.useRef(history)
  const selectedAssistantModeRef = React.useRef(selectedAssistantMode)
  const characterWorkflowActiveRef = React.useRef(characterWorkflowActive)
  const assistantOverlayActiveRef = React.useRef(assistantOverlayActive)
  const serverChatStateRef = React.useRef(serverChatState)
  const serverChatSourceRef = React.useRef(serverChatSource)
  const serverPersistenceHintSeenRef = React.useRef(serverPersistenceHintSeen)

  React.useEffect(() => {
    historyRef.current = history
  }, [history])

  React.useEffect(() => {
    selectedAssistantModeRef.current = selectedAssistantMode
  }, [selectedAssistantMode])

  React.useEffect(() => {
    characterWorkflowActiveRef.current = characterWorkflowActive
  }, [characterWorkflowActive])

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

  const { persistenceTooltip, focusConnectionCard, getPersistenceModeLabel } =
    usePersistenceMode({
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
    const controller = new AbortController()
    activeSaveControllerRef.current = controller
    const capturedHistoryId = latestDepsRef.current.historyId
    const capturedRestoreRevision = usePlaygroundSessionStore.getState().restoreRevision
    let createdChatId: string | null = null
    const acknowledgedIds: string[] = []
    let requestSnapshot: ServicePromptSnapshot | undefined
    const isCurrentSave = () => {
      const current = latestDepsRef.current
      return (
        !controller.signal.aborted &&
        !requestSnapshot?.scopeSignal.aborted &&
        !current.temporaryChat &&
        current.historyId === capturedHistoryId &&
        usePlaygroundSessionStore.getState().restoreRevision === capturedRestoreRevision &&
        current.history.length > 0 &&
        (!current.serverChatId || current.serverChatId === createdChatId)
      )
    }
    const requireConnectionReady = () => {
      if (!latestDepsRef.current.isConnectionReady) {
        throw new Error(
          "Connection unavailable. Reconnect, then retry saving chat."
        )
      }
    }
    try {
      const retry = retryChatPromotion(setServerChatId, capturedHistoryId)
      if (retry) {
        await retry
        return
      }
      const capturedMessages = (latestDepsRef.current.messages || []).map(message => ({ ...message, images: [...(message.images || [])] }))
      const snapshot = historyRef.current
        .map((msg) => ({
          role: ["system", "assistant", "user"].includes(msg.role)
            ? msg.role
            : "user",
          content: (msg.content || "").trim()
        }))
        .filter((msg) => msg.content)
      // Pair the immutable, ordered transcript once. Never search equal text in
      // a changing array or assign one receipt to several identical messages.
      const visible = capturedMessages.filter(message => (message.role || (message.isBot ? "assistant" : "user")) !== "system" && message.message.trim())
      const transcript = snapshot.filter(message => message.role !== "system")
      const aligned = new Set(visible.map(message => message.id)).size === visible.length &&
        visible.length === transcript.length && visible.every((message, index) =>
        Boolean(message.id) && (message.role || (message.isBot ? "assistant" : "user")) === transcript[index].role &&
        message.message.trim() === transcript[index].content && !message.images?.some(Boolean))
      let visibleIndex = 0
      const sources = snapshot.map(message => message.role === "system" ? undefined : aligned ? visible[visibleIndex++] : undefined)
      if (
        !isConnectionReady ||
        temporaryChat ||
        serverChatId ||
        snapshot.length === 0
      ) {
        return
      }
      const isOverlaySelection =
        assistantOverlayActiveRef.current ||
        selectedAssistantModeRef.current === "overlay"
      const selectedAssistantMode = selectedAssistantModeRef.current
      if (!isOverlaySelection && selectedAssistantMode === "tracked") {
        return
      }
      const characterWorkflowNeedsTrackedCharacter =
        characterWorkflowActiveRef.current && !isOverlaySelection
      if (characterWorkflowNeedsTrackedCharacter) {
        return
      }
      await trackChatPromotion(
        setServerChatId,
        capturedHistoryId,
        loadServicePromptSnapshot([], { signal: controller.signal }),
        async (scopeSnapshot) => {
          requestSnapshot = scopeSnapshot
          if (!isCurrentSave()) return
          requireConnectionReady()
          const firstUser = snapshot.find((m) => m.role === "user")
          const explicitSource =
            serverChatSourceRef.current &&
            serverChatSourceRef.current.trim().length > 0
              ? serverChatSourceRef.current.trim()
              : null
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
          const title =
            titleSource.length > 80
              ? `${titleSource.slice(0, 77)}…`
              : titleSource

          const createPayload = {
            title,
            state: serverChatStateRef.current || "in-progress",
            source: explicitSource || WEBUI_CHAT_SOURCE
          }
          const requestOptions = {
            signal: requestSnapshot.scopeSignal,
            requestScope: requestSnapshot.requestScope
          }
          // Every resumed conversation needs a fresh prefix check, including
          // readiness pauses after an acknowledged create or message write.
          const reconcileExistingChat = createdChatId !== null
          if (!createdChatId) {
            const created = await tldwClient.createChat(
              createPayload,
              requestOptions
            )
            if (!isCurrentSave()) return
            const rawId =
              (created as any)?.id ?? (created as any)?.chat_id ?? created
            const cid = rawId != null ? String(rawId) : ""
            if (!cid) {
              throw new Error("Failed to create server chat")
            }
            createdChatId = cid
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
          }

          const cid = createdChatId
          const acknowledge = async (index: number, saved: { id: string; version?: number }) => {
            const source = sources[index]
            if (!source?.id || !isCurrentSave()) return
            await acknowledgePromotedChatMessage({
              historyId: capturedHistoryId, chatId: cid,
              ownerKey: serverChatMirrorOwnerKey(scopeSnapshot), source,
              serverMessageId: String(saved.id), version: saved.version,
              signal: scopeSnapshot.scopeSignal, isCurrent: isCurrentSave
            })
            if (!isCurrentSave()) return
            latestDepsRef.current.setMessages?.(current => current.map(message => {
              if (message.id !== source.id) return message
              if (message.serverMessageId && message.serverMessageId !== String(saved.id))
                throw new Error("The local saved message identity changed. Keep this chat and resolve its history before retrying.")
              return { ...message, serverMessageId: String(saved.id),
                serverMessageVersion: message.serverMessageVersion ?? (message.message === source.message ? saved.version : undefined) }
            }))
          }
          if (reconcileExistingChat) {
            const stored = []
            for (let offset = 0; ; offset += 200) {
              requireConnectionReady()
              const batch = await tldwClient.listChatMessages(
                cid,
                { limit: 200, offset, render_placeholders: false },
                { ...requestOptions, fresh: true }
              )
              if (!isCurrentSave()) return
              stored.push(...batch)
              if (stored.length > snapshot.length || batch.length < 200) break
            }
            const prefixMatches =
              stored.length >= acknowledgedIds.length &&
              stored.length <= snapshot.length &&
              stored.every(
                (row, index) =>
                  Boolean(row.id) &&
                  (row.role || row.sender) === snapshot[index]?.role &&
                  row.content === snapshot[index]?.content &&
                  (!acknowledgedIds[index] ||
                    String(row.id) === acknowledgedIds[index])
              )
            if (!prefixMatches)
              throw new Error(
                "Saved messages changed during recovery. Keep this local chat and resolve the server history before retrying."
              )
            for (let index = 0; index < stored.length; index++) {
              await acknowledge(index, stored[index])
              if (!isCurrentSave()) return
            }
            acknowledgedIds.splice(
              0,
              acknowledgedIds.length,
              ...stored.map((row) => String(row.id))
            )
          }
          for (const msg of snapshot.slice(acknowledgedIds.length)) {
            if (!isCurrentSave()) return
            requireConnectionReady()
            const saved = await tldwClient.addChatMessage(
              cid,
              msg,
              requestOptions
            )
            if (!isCurrentSave()) return
            if (!saved?.id)
              throw new Error(
                "The server did not confirm the saved message identity."
              )
            const index = acknowledgedIds.length
            acknowledgedIds.push(String(saved.id))
            await acknowledge(index, saved)
          }

          if (!isCurrentSave()) return
          if (!serverPersistenceHintSeenRef.current) {
            serverPersistenceHintSeenRef.current = true
            setServerPersistenceHintSeen(true)
            setShowServerPersistenceHint(true)
          }
        },
        {
          abort: () => controller.abort(),
          onComplete: () =>
            notificationApi.destroy?.(`chat-promotion-${capturedHistoryId}`),
          onFailure: (error, retrying) => {
            if (!isCurrentSave()) return
            notificationApi.error({
              key: `chat-promotion-${capturedHistoryId}`,
              duration: 0,
              title: t(
                "playground:composer.persistence.incompleteTitle",
                "Chat saving is incomplete"
              ),
              description:
                t(
                  "playground:composer.persistence.incompleteBody",
                  "Earlier messages are not fully saved. Retry saving chat before continuing."
                ) + (error instanceof Error ? ` ${error.message}` : ""),
              actions: (
                <button
                  type="button"
                  disabled={retrying}
                  onClick={() => {
                    if (isCurrentSave()) void handleSaveChatToServer()
                  }}
                >
                  {t(
                    "playground:composer.persistence.retrySave",
                    "Retry saving chat"
                  )}
                </button>
              )
            })
          }
        }
      )
    } catch (e: any) {
      if (
        controller.signal.aborted ||
        requestSnapshot?.scopeSignal.aborted ||
        isRequestConfigScopeChangedError(e) ||
        isChatPromotionIncompleteError(e)
      )
        return
      notificationApi.error({
        message: t("error"),
        description: e?.message || t("somethingWentWrong")
      })
    } finally {
      if (activeSaveControllerRef.current === controller) {
        activeSaveControllerRef.current = null
      }
      serverSaveInFlightRef.current = false
    }
  }, [
    invalidateServerChatHistory,
    isConnectionReady,
    notificationApi,
    temporaryChat,
    serverChatId,
    setServerChatId,
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
    characterWorkflowActive,
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

export type UsePlaygroundPersistenceReturn = ReturnType<
  typeof usePlaygroundPersistence
>
