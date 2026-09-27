import React from "react"
import { useStoreMessageOption, type Message } from "@/store/option"
import { generateID } from "@/db/dexie/helpers"
import { createSaveMessageOnSuccess } from "@/hooks/utils/messageHelpers"
import { updateActiveVariant } from "@/utils/message-variants"
import { useVoiceChatSettings } from "@/hooks/useVoiceChatSettings"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"

export const useVoiceChatMessages = () => {
  const {
    messages,
    setMessages,
    history,
    setHistory,
    historyId,
    setHistoryId,
    temporaryChat
  } = useStoreMessageOption()
  const { voiceChatModel } = useVoiceChatSettings()
  const { selectedModel } = useSelectedModel()

  const saveMessageOnSuccess = React.useMemo(
    () => createSaveMessageOnSuccess(temporaryChat, setHistoryId),
    [temporaryChat, setHistoryId]
  )

  const currentTurnRef = React.useRef<{
    userId: string
    assistantId: string
    userText: string
    assistantText: string
    modelName: string
  } | null>(null)
  const [activeAssistantId, setActiveAssistantId] = React.useState<string | null>(null)

  const resetTurn = React.useCallback(() => {
    currentTurnRef.current = null
    setActiveAssistantId(null)
  }, [])

  const beginTurn = React.useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed) return
      const userId = generateID()
      const assistantId = generateID()
      const createdAt = Date.now()
      const modelName = String(voiceChatModel || selectedModel || "Assistant")

      const userMessage: Message = {
        isBot: false,
        name: "You",
        role: "user",
        message: trimmed,
        sources: [],
        createdAt,
        id: userId
      }

      const assistantMessage: Message = {
        isBot: true,
        name: modelName,
        role: "assistant",
        message: "▋",
        sources: [],
        createdAt: createdAt + 1,
        id: assistantId,
        modelName
      }

      currentTurnRef.current = {
        userId,
        assistantId,
        userText: trimmed,
        assistantText: "",
        modelName
      }
      setActiveAssistantId(assistantId)

      setMessages((prev) => [...prev, userMessage, assistantMessage])
      setHistory((prev) => [...prev, { role: "user", content: trimmed }])
    },
    [selectedModel, setHistory, setMessages, voiceChatModel]
  )

  const appendAssistantDelta = React.useCallback((delta: string) => {
    const turn = currentTurnRef.current
    if (!turn) return
    const nextText = `${turn.assistantText}${delta}`
    turn.assistantText = nextText

    setMessages((prev) =>
      prev.map((message) =>
        message.id === turn.assistantId
          ? updateActiveVariant(message, { message: `${nextText}▋` })
          : message
      )
    )
  }, [setMessages])

  const finalizeAssistant = React.useCallback(
    async (text: string) => {
      const turn = currentTurnRef.current
      if (!turn) return
      const finalText = text || turn.assistantText

      setMessages((prev) =>
        prev.map((message) =>
          message.id === turn.assistantId
            ? updateActiveVariant(message, { message: finalText })
            : message
        )
      )

      setHistory((prev) => [...prev, { role: "assistant", content: finalText }])

      await saveMessageOnSuccess({
        historyId,
        setHistoryId,
        isRegenerate: false,
        selectedModel: turn.modelName,
        message: turn.userText,
        image: "",
        fullText: finalText,
        source: [],
        message_source: "server",
        userMessageId: turn.userId,
        assistantMessageId: turn.assistantId
      })

      if (currentTurnRef.current === turn) resetTurn()
    },
    [historyId, resetTurn, saveMessageOnSuccess, setHistory, setHistoryId, setMessages]
  )

  const failTurn = React.useCallback(
    async (reason: string) => {
      const turn = currentTurnRef.current
      if (!turn) return

      const finalText = turn.assistantText.trim()
      const interruptedAt = Date.now()
      if (!finalText) {
        setMessages((prev) =>
          prev.filter((message) => message.id !== turn.assistantId)
        )
        resetTurn()
        return
      }

      setMessages((prev) =>
        prev.map((message) =>
          message.id === turn.assistantId
            ? updateActiveVariant(message, {
                message: finalText,
                generationInfo: {
                  ...(message.generationInfo || {}),
                  interrupted: true,
                  interruptionReason: reason,
                  interruptedAt
                }
              })
            : message
        )
      )

      setHistory((prev) => {
        const last = prev[prev.length - 1]
        if (last?.role === "assistant" && last.content === finalText) {
          return prev
        }
        return [...prev, { role: "assistant", content: finalText }]
      })

      await saveMessageOnSuccess({
        historyId,
        setHistoryId,
        isRegenerate: false,
        selectedModel: turn.modelName,
        message: turn.userText,
        image: "",
        fullText: finalText,
        source: [],
        message_source: "server",
        generationInfo: {
          interrupted: true,
          interruptionReason: reason,
          interruptedAt
        },
        userMessageId: turn.userId,
        assistantMessageId: turn.assistantId
      })

      if (currentTurnRef.current === turn) resetTurn()
    },
    [historyId, resetTurn, saveMessageOnSuccess, setHistory, setHistoryId, setMessages]
  )

  const abandonTurn = React.useCallback(
    (options?: { includeInHistory?: boolean }) => {
      const turn = currentTurnRef.current
      if (!turn) return
      const finalText = turn.assistantText.trim()
      if (!finalText) {
        setMessages((prev) =>
          prev.filter((message) => message.id !== turn.assistantId)
        )
      } else {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === turn.assistantId
              ? updateActiveVariant(message, { message: finalText })
              : message
          )
        )
        if (options?.includeInHistory ?? true) {
          setHistory((prev) => {
            const last = prev[prev.length - 1]
            if (last?.role === "assistant" && last.content === finalText) {
              return prev
            }
            return [...prev, { role: "assistant", content: finalText }]
          })
        }
      }
      resetTurn()
    },
    [resetTurn, setHistory, setMessages]
  )

  return {
    beginTurn,
    appendAssistantDelta,
    finalizeAssistant,
    failTurn,
    resetTurn,
    abandonTurn,
    activeAssistantId
  }
}
