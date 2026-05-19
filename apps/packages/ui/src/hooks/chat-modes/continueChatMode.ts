import { systemPromptForNonRagOption } from "~/services/tldw-server"
import { type ChatHistory, type Message, type ToolChoice } from "~/store/option"
import { getPromptById } from "@/db/dexie/helpers"
import { generateHistory } from "@/utils/generate-history"
import { systemPromptFormatter } from "@/utils/system-message"
import type { ActorSettings } from "@/types/actor"
import { maybeInjectActorMessage } from "@/utils/actor"
import { updateActiveVariant } from "@/utils/message-variants"
import type { SaveMessageData, SaveMessageErrorData } from "@/types/chat-modes"
import {
  runChatPipeline,
  type ChatModeDefinition
} from "./chatModePipeline"
import { appendSystemPromptSuffix } from "@/utils/output-formatting-guide"
import type { ChatSubmitResult } from "@/hooks/chat/chat-action-utils"

type ContinueChatModeParams = {
  selectedModel: string
  useOCR: boolean
  selectedSystemPrompt: string
  currentChatModelSettings: any
  toolChoice?: ToolChoice
  setMessages: (messages: Message[] | ((prev: Message[]) => Message[])) => void
  saveMessageOnSuccess: (data: SaveMessageData) => Promise<string | null>
  saveMessageOnError: (data: SaveMessageErrorData) => Promise<string | null>
  setHistory: (history: ChatHistory) => void
  setIsProcessing: (value: boolean) => void
  setStreaming: (value: boolean) => void
  setAbortController: (controller: AbortController | null) => void
  historyId: string | null
  setHistoryId: (id: string) => void
  actorSettings?: ActorSettings
  systemPromptAppendix?: string
  assistantMessageId?: string
  assistantParentMessageId?: string | null
}

const continueChatModeDefinition: ChatModeDefinition<ContinueChatModeParams> = {
  id: "continue",
  isContinue: true,
  setupMessages: (ctx) => {
    const lastMessage = ctx.messages[ctx.messages.length - 1]
    if (lastMessage) {
      const resolvedId = lastMessage.id ?? ctx.resolvedAssistantMessageId
      const updatedLast = updateActiveVariant(lastMessage, {
        message: `${lastMessage.message}▋`,
        id: resolvedId
      })
      ctx.setMessages((prev) => {
        const next = [...prev]
        next[next.length - 1] = updatedLast
        return next
      })
      return {
        targetMessageId: resolvedId,
        initialFullText: lastMessage.message
      }
    }
    return {
      targetMessageId: ctx.resolvedAssistantMessageId,
      initialFullText: ""
    }
  },
  preparePrompt: async (ctx) => {
    const prompt = await systemPromptForNonRagOption()
    const selectedPrompt = await getPromptById(ctx.selectedSystemPrompt)
    let promptContent: string | undefined = undefined

    const applicationChatHistory = generateHistory(
      ctx.history,
      ctx.selectedModel
    )
    const resolvePromptWithAppendix = (content?: string | null) =>
      appendSystemPromptSuffix(content || "", ctx.systemPromptAppendix)

    if (!selectedPrompt) {
      const resolvedDefaultPrompt = resolvePromptWithAppendix(prompt)
      if (resolvedDefaultPrompt) {
        applicationChatHistory.unshift(
          await systemPromptFormatter({
            content: resolvedDefaultPrompt
          })
        )
      }
    }

    const isTempSystemprompt =
      ctx.currentChatModelSettings.systemPrompt &&
      ctx.currentChatModelSettings.systemPrompt?.trim().length > 0

    if (!isTempSystemprompt && selectedPrompt) {
      const selectedPromptContent =
        selectedPrompt.system_prompt ?? selectedPrompt.content
      const resolvedSelectedPrompt =
        resolvePromptWithAppendix(selectedPromptContent)
      applicationChatHistory.unshift(
        await systemPromptFormatter({
          content: resolvedSelectedPrompt
        })
      )
      promptContent = resolvedSelectedPrompt
    }

    if (isTempSystemprompt) {
      const resolvedTemporaryPrompt = resolvePromptWithAppendix(
        ctx.currentChatModelSettings.systemPrompt
      )
      applicationChatHistory.unshift(
        await systemPromptFormatter({
          content: resolvedTemporaryPrompt
        })
      )
      promptContent = resolvedTemporaryPrompt
    }

    const templatesActive = !!ctx.selectedSystemPrompt
    const nextHistory = await maybeInjectActorMessage(
      applicationChatHistory,
      ctx.actorSettings || null,
      templatesActive
    )
    applicationChatHistory.length = 0
    applicationChatHistory.push(...nextHistory)

    return {
      chatHistory: applicationChatHistory,
      promptId: ctx.selectedSystemPrompt,
      promptContent
    }
  },
  updateHistory: (ctx, fullText) => {
    const nextHistory = [...ctx.history]
    if (nextHistory.length > 0) {
      nextHistory[nextHistory.length - 1] = {
        ...nextHistory[nextHistory.length - 1],
        content: fullText
      }
    }
    return nextHistory
  }
}

export const continueChatMode = async (
  messages: Message[],
  history: ChatHistory,
  signal: AbortSignal,
  params: ContinueChatModeParams
): Promise<ChatSubmitResult> => {
  console.log("Using continueChatMode")
  const lastMessage = messages[messages.length - 1]
  const assistantMessageId = lastMessage?.id

  return runChatPipeline(
    continueChatModeDefinition,
    "",
    "",
    false,
    messages,
    history,
    signal,
    {
      ...params,
      assistantMessageId,
      assistantParentMessageId: lastMessage?.parentMessageId ?? null
    }
  )
}
