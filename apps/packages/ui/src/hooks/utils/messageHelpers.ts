import type { NotificationInstance } from "antd/es/notification/interface"
import { useTranslation } from "react-i18next"
import {
  saveMessageOnError as saveError,
  saveMessageOnSuccess as saveSuccess
} from "../chat-helper"

export const focusTextArea = (textareaRef?: React.RefObject<HTMLTextAreaElement>) => {
  try {
    if (textareaRef?.current) {
      textareaRef.current.focus()
    } else {
      const textareaElement = document.getElementById(
        "textarea-message"
      ) as HTMLTextAreaElement
      if (textareaElement) {
        textareaElement.focus()
      }
    }
  } catch (e) {
    // best-effort focus — failure is harmless if element is not in DOM
  }
}

export const validateBeforeSubmit = (
  selectedModel: string,
  t: any,
  notification: NotificationInstance
) => {
  if (!selectedModel || selectedModel?.trim()?.length === 0) {
    notification.error({
      message: t("error"),
      description: t("validationSelectModel")
    })
    return false
  }

  return true
}

export const createSaveMessageOnSuccess = (
  temporaryChat: boolean,
  setHistoryId: (id: string, options?: { preserveServerChatId?: boolean }) => void,
  options?: {
    onServerConversationLinked?: (conversationId: string) => void
  }
) => {
  return async (e: any): Promise<string | null> => {
    if (!temporaryChat) {
      const historyId = await saveSuccess({
        ...e,
        setHistoryId: e?.setHistoryId ?? setHistoryId
      })
      const conversationId =
        typeof e?.conversationId === "string"
          ? e.conversationId.trim()
          : e?.conversationId != null
            ? String(e.conversationId).trim()
            : ""
      if (conversationId.length > 0) {
        options?.onServerConversationLinked?.(conversationId)
      }
      return historyId
    } else {
      setHistoryId("temp")
      return null
    }
  }
}

export const createSaveMessageOnError = (
  temporaryChat: boolean,
  history: any,
  setHistory: (history: any) => void,
  setHistoryId: (id: string, options?: { preserveServerChatId?: boolean }) => void
) => {
  return async (e: any): Promise<string | null> => {
    if (!temporaryChat) {
      return await saveError({
        ...e,
        history: e?.history ?? history,
        setHistory: e?.setHistory ?? setHistory,
        setHistoryId: e?.setHistoryId ?? setHistoryId
      })
    } else {
      const historyToUpdate = Array.isArray(e?.history) ? e.history : history
      const setHistoryTarget =
        typeof e?.setHistory === "function" ? e.setHistory : setHistory
      const setHistoryIdTarget =
        typeof e?.setHistoryId === "function" ? e.setHistoryId : setHistoryId
      setHistoryTarget([
        ...historyToUpdate,
        {
          role: "user",
          content: e.userMessage,
          image: e.image,
          messageType: e.userMessageType ?? e.message_type
        },
        {
          role: "assistant",
          content: e.botMessage,
          messageType: e.assistantMessageType ?? e.message_type
        }
      ])

      setHistoryIdTarget("temp")
      return null
    }
  }
}
