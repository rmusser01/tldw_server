import { promptForRag } from "~/services/tldw-server"
import { type ChatHistory, type Message, type ToolChoice } from "~/store/option"
import { generateHistory } from "@/utils/generate-history"
import { humanMessageFormatter } from "@/utils/human-message"
import { extractGenerationInfo } from "@/utils/llm-helpers"
import { getTabContents } from "@/libs/get-tab-contents"
import { ChatDocuments } from "@/models/ChatTypes"
import type { ActorSettings } from "@/types/actor"
import { maybeInjectActorMessage } from "@/utils/actor"
import type { SaveMessageData, SaveMessageErrorData } from "@/types/chat-modes"
import {
  runChatPipeline,
  type ChatModeDefinition
} from "./chatModePipeline"
import { appendSystemPromptSuffix } from "@/utils/output-formatting-guide"
import type { ChatSubmitResult } from "@/hooks/chat/chat-action-utils"

type TabChatModeParams = {
  selectedModel: string
  useOCR: boolean
  selectedSystemPrompt: string
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
  systemPromptAppendix?: string
  actorSettings?: ActorSettings
  clusterId?: string
  userMessageType?: string
  assistantMessageType?: string
  modelIdOverride?: string
  userMessageId?: string
  assistantMessageId?: string
  userParentMessageId?: string | null
  assistantParentMessageId?: string | null
  historyForModel?: ChatHistory
  regenerateFromMessage?: Message
  documents: ChatDocuments
}

const tabChatModeDefinition: ChatModeDefinition<TabChatModeParams> = {
  id: "tab",
  buildUserMessage: (ctx) => ({
    isBot: false,
    name: "You",
    message: ctx.message,
    sources: [],
    images: ctx.image ? [ctx.image] : [],
    createdAt: ctx.createdAt,
    documents: ctx.documents,
    id: ctx.resolvedUserMessageId,
    messageType: ctx.userMessageType,
    clusterId: ctx.clusterId,
    modelId: ctx.userModelId,
    parentMessageId: ctx.userParentMessageId ?? null
  }),
  buildAssistantMessage: (ctx) => ({
    isBot: true,
    name: ctx.selectedModel,
    message: "▋",
    sources: [],
    createdAt: ctx.createdAt,
    id: ctx.resolvedAssistantMessageId,
    modelImage: ctx.modelInfo?.model_avatar,
    modelName: ctx.modelInfo?.model_name || ctx.selectedModel,
    messageType: ctx.assistantMessageType,
    clusterId: ctx.clusterId,
    modelId: ctx.resolvedModelId,
    parentMessageId: ctx.resolvedAssistantParentMessageId ?? null
  }),
  preparePrompt: async (ctx) => {
    const { ragPrompt: systemPrompt } = await promptForRag()
    const resolvedSystemPrompt = appendSystemPromptSuffix(
      systemPrompt,
      ctx.systemPromptAppendix
    )
    const context = await getTabContents(ctx.documents)

    let humanMessage = await humanMessageFormatter({
      content: [
        {
          text: resolvedSystemPrompt
            .replace("{context}", context)
            .replace("{question}", ctx.message),
          type: "text"
        }
      ],
      model: ctx.selectedModel,
      useOCR: ctx.useOCR
    })

    if (ctx.image.length > 0) {
      humanMessage = await humanMessageFormatter({
        content: [
          {
            text: ctx.message,
            type: "text"
          },
          {
            image_url: ctx.image,
            type: "image_url"
          }
        ],
        model: ctx.selectedModel,
        useOCR: ctx.useOCR
      })
    }

    let applicationChatHistory = generateHistory(
      ctx.historyForModel ?? ctx.history,
      ctx.selectedModel
    )

    const templatesActive = !!ctx.selectedSystemPrompt
    applicationChatHistory = await maybeInjectActorMessage(
      applicationChatHistory,
      ctx.actorSettings || null,
      templatesActive
    )

    return {
      chatHistory: applicationChatHistory,
      humanMessage,
      sources: []
    }
  },
  extractGenerationInfo: (output) => extractGenerationInfo(output)
}

export const tabChatMode = async (
  message: string,
  image: string,
  documents: ChatDocuments,
  isRegenerate: boolean,
  messages: Message[],
  history: ChatHistory,
  signal: AbortSignal,
  params: Omit<TabChatModeParams, "documents">
): Promise<ChatSubmitResult> => {
  console.log("Using tabChatMode")
  return runChatPipeline(
    tabChatModeDefinition,
    message,
    image,
    isRegenerate,
    messages,
    history,
    signal,
    {
      ...params,
      documents
    }
  )
}
