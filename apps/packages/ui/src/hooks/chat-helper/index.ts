import {
  getLastChatHistory,
  saveHistory,
  saveMessage,
  updateMessage,
  updateLastUsedModel as setLastUsedChatModel,
  updateLastUsedPrompt as setLastUsedChatSystemPrompt,
  updateChatHistoryCreatedAt
} from "@/db/dexie/helpers"
import { ChatDocuments } from "@/models/ChatTypes"
import { generateTitle } from "@/services/title"
import { ChatHistory, useStoreMessageOption } from "@/store/option"
import { updatePageTitle } from "@/utils/update-page-title"
import { buildAssistantErrorContent } from "@/utils/chat-error-message"

let didLogSetHistoryMissing = false

const buildFallbackHistoryTitle = (userMessage: string): string => {
  const normalized = userMessage.replace(/\s+/g, " ").trim()
  return normalized.length > 0 ? normalized.slice(0, 80) : "Untitled Chat"
}

const generateTitleWithFallback = async (
  selectedModel: string,
  userMessage: string
): Promise<string> => {
  try {
    const title = await generateTitle(selectedModel, userMessage, userMessage)
    const trimmed = typeof title === "string" ? title.trim() : ""
    return trimmed.length > 0 ? trimmed : buildFallbackHistoryTitle(userMessage)
  } catch {
    return buildFallbackHistoryTitle(userMessage)
  }
}

const resolveHistorySetter = (
  candidate: unknown
): ((history: ChatHistory) => void) | null => {
  if (typeof candidate === "function") {
    return candidate as (history: ChatHistory) => void
  }

  const fallback = useStoreMessageOption.getState().setHistory
  if (typeof fallback === "function") {
    return fallback
  }

  if (!didLogSetHistoryMissing) {
    didLogSetHistoryMissing = true
    console.error("[chat] saveMessageOnError could not resolve setHistory setter", {
      setHistoryType: typeof candidate
    })
  }
  return null
}

export const saveMessageOnError = async ({
  e,
  history,
  setHistory,
  image,
  userMessage,
  botMessage,
  historyId,
  selectedModel,
  setHistoryId,
  isRegenerating,
  message_source = "web-ui",
  message_type,
  userMessageType,
  assistantMessageType,
  clusterId,
  modelId,
  userModelId,
  userMessageId,
  assistantMessageId,
  userParentMessageId,
  assistantParentMessageId,
  prompt_content,
  prompt_id,
  isContinue,
  documents = []
}: {
  e: any
  setHistory: (history: ChatHistory) => void
  history: ChatHistory
  userMessage: string
  image: string
  botMessage: string
  historyId: string | null
  selectedModel: string
  setHistoryId: (
    historyId: string,
    options?: { preserveServerChatId?: boolean }
  ) => void
  isRegenerating: boolean
  message_source?: "copilot" | "web-ui" | "server" | "branch"
  message_type?: string
  userMessageType?: string
  assistantMessageType?: string
  clusterId?: string
  modelId?: string
  userModelId?: string
  userMessageId?: string
  assistantMessageId?: string
  userParentMessageId?: string | null
  assistantParentMessageId?: string | null
  prompt_id?: string
  prompt_content?: string
  isContinue?: boolean
  documents?: ChatDocuments
}) => {
  const isAbort = (
    e?.name === "AbortError" ||
    e?.message === "AbortError" ||
    e?.name?.includes?.("AbortError") ||
    e?.message?.includes?.("AbortError")
  )

  const assistantContent = buildAssistantErrorContent(botMessage, e)
  const safeSetHistory = resolveHistorySetter(setHistory)

  if (isAbort) {
    safeSetHistory?.([
      ...history,
      {
        role: "user",
        content: userMessage,
        image,
        messageType: userMessageType ?? message_type
      },
      {
        role: "assistant",
        content: assistantContent,
        messageType: assistantMessageType ?? message_type
      }
    ])

    if (historyId) {
      if (!isRegenerating && !isContinue) {
        await saveMessage({
          id: userMessageId,
          history_id: historyId,
          name: selectedModel,
          role: "user",
          content: userMessage,
          images: [image],
          time: 1,
          message_type: userMessageType ?? message_type,
          clusterId,
          modelId: userModelId,
          parent_message_id: userParentMessageId ?? null,
          documents
        })
      }

      if (isContinue) {
        console.log("Saving Last Message")
        const lastMessage = await getLastChatHistory(historyId)
        await updateMessage(historyId, lastMessage.id, botMessage)
      } else {
        await saveMessage({
          id: assistantMessageId,
          history_id: historyId,
          name: selectedModel,
          role: "assistant",
          content: assistantContent,
          images: [],
          source: [],
          time: 2,
          message_type: assistantMessageType ?? message_type,
          clusterId,
          modelId,
          parent_message_id: assistantParentMessageId ?? null
        })
      }
      await setLastUsedChatModel(historyId, selectedModel)
      if (prompt_id || prompt_content) {
        await setLastUsedChatSystemPrompt(historyId, {
          prompt_content,
          prompt_id
        })
      }

      return historyId
    } else {
      const title = await generateTitleWithFallback(selectedModel, userMessage)
      const newHistoryId = await saveHistory(title, false, message_source)
      updatePageTitle(title)
      if (!isRegenerating) {
        await saveMessage({
          id: userMessageId,
          history_id: newHistoryId.id,
          name: selectedModel,
          role: "user",
          content: userMessage,
          images: [image],
          time: 1,
          message_type: userMessageType ?? message_type,
          clusterId,
          modelId: userModelId,
          parent_message_id: userParentMessageId ?? null,
          documents
        })
      }

      await saveMessage({
        id: assistantMessageId,
        history_id: newHistoryId.id,
        name: selectedModel,
        role: "assistant",
        content: assistantContent,
        images: [],
        source: [],
        time: 2,
        message_type: assistantMessageType ?? message_type,
        clusterId,
        modelId,
        parent_message_id: assistantParentMessageId ?? null
      })
      setHistoryId(newHistoryId.id)
      await setLastUsedChatModel(newHistoryId.id, selectedModel)
      if (prompt_id || prompt_content) {
        await setLastUsedChatSystemPrompt(newHistoryId.id, {
          prompt_content,
          prompt_id
        })
      }

      return newHistoryId.id
    }
  }

  // Non-abort errors: append user + assistant with error content as well
  safeSetHistory?.([
    ...history,
    {
      role: "user",
      content: userMessage,
      image,
      messageType: userMessageType ?? message_type
    },
    {
      role: "assistant",
      content: assistantContent,
      messageType: assistantMessageType ?? message_type
    }
  ])

  if (historyId) {
    try {
      // Save user message if not regenerating
      if (!isRegenerating) {
        await saveMessage({
          id: userMessageId,
          history_id: historyId,
          name: selectedModel,
          role: "user",
          content: userMessage,
          images: [image],
          time: 1,
          message_type: userMessageType ?? message_type,
          clusterId,
          modelId: userModelId,
          parent_message_id: userParentMessageId ?? null,
          documents
        })
      }
      // Save assistant error message
      await saveMessage({
        id: assistantMessageId,
        history_id: historyId,
        name: selectedModel,
        role: "assistant",
        content: assistantContent,
        images: [],
        source: [],
        time: 2,
        message_type: assistantMessageType ?? message_type,
        clusterId,
        modelId,
        parent_message_id: assistantParentMessageId ?? null
      })
    } catch {}
    return historyId
  } else {
    // Create new history on error
    const title = await generateTitleWithFallback(selectedModel, userMessage)
    const newHistoryId = await saveHistory(title, false, message_source)
    updatePageTitle(title)
    try {
      if (!isRegenerating) {
        await saveMessage({
          id: userMessageId,
          history_id: newHistoryId.id,
          name: selectedModel,
          role: "user",
          content: userMessage,
          images: [image],
          time: 1,
          message_type: userMessageType ?? message_type,
          clusterId,
          modelId: userModelId,
          parent_message_id: userParentMessageId ?? null,
          documents
        })
      }
      await saveMessage({
        id: assistantMessageId,
        history_id: newHistoryId.id,
        name: selectedModel,
        role: "assistant",
        content: assistantContent,
        images: [],
        source: [],
        time: 2,
        message_type: assistantMessageType ?? message_type,
        clusterId,
        modelId,
        parent_message_id: assistantParentMessageId ?? null
      })
    } catch {}
    setHistoryId(newHistoryId.id)
    return newHistoryId.id
  }
}

export const saveMessageOnSuccess = async ({
  historyId,
  setHistoryId,
  isRegenerate,
  selectedModel,
  message,
  image,
  fullText,
  source,
  assistantImages,
  message_source = "web-ui",
  message_type,
  userMessageType,
  assistantMessageType,
  clusterId,
  modelId,
  userModelId,
  userMessageId,
  assistantMessageId,
  userParentMessageId,
  assistantParentMessageId,
  generationInfo,
  prompt_id,
  prompt_content,
  reasoning_time_taken = 0,
  isContinue,
  documents = []
}: {
  historyId: string | null
  setHistoryId: (
    historyId: string,
    options?: { preserveServerChatId?: boolean }
  ) => void
  isRegenerate: boolean
  selectedModel: string | null
  message: string
  image: string
  fullText: string
  source: any[]
  assistantImages?: string[]
  message_source?: "copilot" | "web-ui" | "server" | "branch"
  message_type?: string
  userMessageType?: string
  assistantMessageType?: string
  clusterId?: string
  modelId?: string
  userModelId?: string
  userMessageId?: string
  assistantMessageId?: string
  userParentMessageId?: string | null
  assistantParentMessageId?: string | null
  generationInfo?: any
  prompt_id?: string
  prompt_content?: string
  reasoning_time_taken?: number
  isContinue?: boolean
  documents?: ChatDocuments
}) => {
  if (historyId) {
    if (!isRegenerate && !isContinue) {
      await saveMessage({
        id: userMessageId,
        history_id: historyId,
        name: selectedModel,
        role: "user",
        content: message,
        images: [image],
        time: 1,
        message_type: userMessageType ?? message_type,
        clusterId,
        modelId: userModelId,
        parent_message_id: userParentMessageId ?? null,
        generationInfo,
        reasoning_time_taken,
        documents
      })
    }

    if (isContinue) {
      console.log("Saving Last Message")
      const lastMessage = await getLastChatHistory(historyId)
      console.log("lastMessage", lastMessage)
      await updateMessage(historyId, lastMessage.id, fullText)
    } else {
      await saveMessage(
        {
          id: assistantMessageId,
          history_id: historyId,
          name: selectedModel,
          role: "assistant",
          content: fullText,
          images: assistantImages ?? [],
          source,
          time: 2,
          message_type: assistantMessageType ?? message_type,
          clusterId,
          modelId,
          parent_message_id: assistantParentMessageId ?? null,
          generationInfo,
          reasoning_time_taken
        }
        // historyId,
        // selectedModel!,
        // "assistant",
        // fullText,
        // [],
        // source,
        // 2,
        // message_type,
        // generationInfo,
        // reasoning_time_taken
      )
    }

    await setLastUsedChatModel(historyId, selectedModel!)
    if (prompt_id || prompt_content) {
      await setLastUsedChatSystemPrompt(historyId, {
        prompt_content,
        prompt_id
      })
    }

    await updateChatHistoryCreatedAt(historyId)

    return historyId
  } else {
    const title = await generateTitle(selectedModel, message, message)
    updatePageTitle(title)
    const newHistoryId = await saveHistory(title, false, message_source)

    await saveMessage(
      {
        id: userMessageId,
        history_id: newHistoryId.id,
        name: selectedModel,
        role: "user",
        content: message,
        images: [image],
        time: 1,
        message_type: userMessageType ?? message_type,
        clusterId,
        modelId: userModelId,
        parent_message_id: userParentMessageId ?? null,
        generationInfo,
        reasoning_time_taken,
        documents
      }
      // newHistoryId.id,
      // selectedModel,
      // "user",
      // message,
      // [image],
      // [],
      // 1,
      // message_type,
      // generationInfo,
      // reasoning_time_taken
    )

    await saveMessage(
      {
        id: assistantMessageId,
        history_id: newHistoryId.id,
        name: selectedModel,
        role: "assistant",
        content: fullText,
        images: assistantImages ?? [],
        source,
        time: 2,
        message_type: assistantMessageType ?? message_type,
        clusterId,
        modelId,
        parent_message_id: assistantParentMessageId ?? null,
        generationInfo,
        reasoning_time_taken
      }
      // newHistoryId.id,
      // selectedModel!,
      // "assistant",
      // fullText,
      // [],
      // source,
      // 2,
      // message_type,
      // generationInfo,
      // reasoning_time_taken
    )
    setHistoryId(newHistoryId.id)
    await setLastUsedChatModel(newHistoryId.id, selectedModel!)
    if (prompt_id || prompt_content) {
      await setLastUsedChatSystemPrompt(newHistoryId.id, {
        prompt_content,
        prompt_id
      })
    }

    return newHistoryId.id
  }
}
