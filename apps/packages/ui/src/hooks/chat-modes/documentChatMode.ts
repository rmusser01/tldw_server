import { promptForRag } from "~/services/tldw-server"
import { type ChatHistory, type Message, type ToolChoice } from "~/store/option"
import { addFileToSession, getSessionFiles } from "@/db/dexie/helpers"
import { generateHistory } from "@/utils/generate-history"
import { pageAssistModel } from "@/models"
import { humanMessageFormatter } from "@/utils/human-message"
import { removeReasoning } from "@/libs/reasoning"
import { formatDocs } from "@/utils/format-docs"
import { getAllDefaultModelSettings } from "@/services/model-settings"
import { getNoOfRetrievedDocs } from "@/services/app"
import { UploadedFile } from "@/db/dexie/types"
import { getMaxContextSize } from "@/services/kb"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ActorSettings } from "@/types/actor"
import { maybeInjectActorMessage } from "@/utils/actor"
import type { ChatModelSettings } from "@/store/model"
import { extractGenerationInfo } from "@/utils/llm-helpers"
import type { SaveMessageData, SaveMessageErrorData } from "@/types/chat-modes"
import {
  runChatPipeline,
  type ChatModeDefinition
} from "./chatModePipeline"
import { appendSystemPromptSuffix } from "@/utils/output-formatting-guide"
import type { ChatSubmitResult } from "@/hooks/chat/chat-action-utils"

interface RagDocumentMetadata {
  filename?: string
  title?: string
  type?: string
  url?: string
  [key: string]: unknown
}

interface RagDocument {
  content?: string
  text?: string
  chunk?: string
  metadata?: RagDocumentMetadata
}

interface RagResponse {
  results?: RagDocument[]
  documents?: RagDocument[]
  docs?: RagDocument[]
}

type DocumentChatModeParams = {
  selectedModel: string
  useOCR: boolean
  currentChatModelSettings: ChatModelSettings | null
  toolChoice?: ToolChoice
  setMessages: (
    messages: Message[] | ((prev: Message[]) => Message[])
  ) => void
  saveMessageOnSuccess: (data: SaveMessageData) => Promise<string | null>
  saveMessageOnError: (data: SaveMessageErrorData) => Promise<string | null>
  setHistory: (history: ChatHistory) => void
  setIsProcessing: (value: boolean) => void
  setStreaming: (value: boolean) => void
  setAbortController: (controller: AbortController | null) => void
  historyId: string | null
  setHistoryId: (id: string) => void
  fileRetrievalEnabled: boolean
  setActionInfo: (actionInfo: string | null) => void
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
  uploadedFiles: UploadedFile[]
  newFiles: UploadedFile[]
  allFiles: UploadedFile[]
  documents?: any[]
}

const documentChatModeDefinition: ChatModeDefinition<DocumentChatModeParams> = {
  id: "document",
  buildUserMessage: (ctx) => ({
    isBot: false,
    name: "You",
    message: ctx.message,
    sources: [],
    images: ctx.image ? [ctx.image] : [],
    createdAt: ctx.createdAt,
    documents: ctx.newFiles.map((file) => ({
      type: "file",
      filename: file.filename,
      fileSize: file.size
    })),
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
    let query = ctx.message
    const { ragPrompt: systemPrompt, ragQuestionPrompt: questionPrompt } =
      await promptForRag()
    const resolvedSystemPrompt = appendSystemPromptSuffix(
      systemPrompt,
      ctx.systemPromptAppendix
    )
    const contextMessages = ctx.isRegenerate
      ? ctx.messages
      : [
          ...ctx.messages,
          {
            isBot: false,
            name: "You",
            message: ctx.message,
            sources: [],
            images: ctx.image ? [ctx.image] : []
          }
        ]

    let context = ""
    let source: any[] = []
    const docSize = await getNoOfRetrievedDocs()

    if (contextMessages.length > 2) {
      const lastTenMessages = contextMessages.slice(-10)
      lastTenMessages.pop()
      const chat_history = lastTenMessages
        .map((message) => {
          return `${message.isBot ? "Assistant: " : "Human: "}${message.message}`
        })
        .join("\n")
      const promptForQuestion = questionPrompt
        .replaceAll("{chat_history}", chat_history)
        .replaceAll("{question}", ctx.message)
      const questionMessage = await humanMessageFormatter({
        content: [
          {
            text: promptForQuestion,
            type: "text"
          }
        ],
        model: ctx.selectedModel,
        useOCR: ctx.useOCR
      })
      const questionOllama = await pageAssistModel({
        model: ctx.selectedModel,
        toolChoice: ctx.toolChoice
      })
      const response = await questionOllama.invoke([questionMessage])
      query = response.content.toString()
      query = removeReasoning(query)
    }

    try {
      const keyword_filter = ctx.uploadedFiles.map((f) => f.filename).slice(0, 10)
      if (ctx.uploadedFiles.length > 10) {
        ctx.setActionInfo("Only the first 10 uploaded files are used for filtering")
      }
      const ragPayload = {
        query,
        sources: ["media_db"],
        search_mode: "hybrid",
        hybrid_alpha: 0.7,
        top_k: docSize,
        min_score: 0,
        enable_reranking: true,
        reranking_strategy: "flashrank",
        rerank_top_k: Math.max(docSize * 2, 10),
        keyword_filter,
        enable_cache: true,
        adaptive_cache: true,
        enable_chunk_citations: true,
        enable_generation: false
      } as const
      const ragRes = (await tldwClient.ragSearch(
        query,
        ragPayload
      )) as RagResponse
      const docs: RagDocument[] =
        ragRes?.results || ragRes?.documents || ragRes?.docs || []
      if (docs.length > 0) {
        context += formatDocs(
          docs.map((doc) => ({
            pageContent: doc.content || doc.text || doc.chunk || "",
            metadata: doc.metadata || {}
          }))
        )
        source = [
          ...source,
          ...docs.map((doc) => ({
            name: doc.metadata?.filename || doc.metadata?.title || "media",
            type: doc.metadata?.type || "media",
            mode: "rag",
            url: doc.metadata?.url || "",
            pageContent: doc.content || doc.text || doc.chunk || "",
            metadata: doc.metadata || {}
          }))
        ]
      }
    } catch (e) {
      console.error("media_db RAG failed; will fallback to inline context", e)
    }

    if (ctx.uploadedFiles.length > 0 && context.length === 0) {
      const maxContextSize = await getMaxContextSize()
      context += ctx.allFiles
        .map((file) => `File: ${file.filename}\nContent: ${file.content}\n---\n`)
        .join("")
        .substring(0, maxContextSize)
      source = [
        ...source,
        ...ctx.allFiles.map((file) => ({
          pageContent: file.content.substring(0, 200) + "...",
          metadata: {
            source: file.filename,
            type: file.type,
            mode: "rag"
          },
          name: file.filename,
          type: file.type,
          mode: "rag",
          url: ""
        }))
      ]
    }

    if (ctx.uploadedFiles.length === 0 && context.length === 0) {
      context += "No documents uploaded for this conversation."
    }

    const humanMessage = await humanMessageFormatter({
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

    let applicationChatHistory = generateHistory(
      ctx.historyForModel ?? ctx.history,
      ctx.selectedModel
    )

    const templatesActive = false
    applicationChatHistory = await maybeInjectActorMessage(
      applicationChatHistory,
      ctx.actorSettings || null,
      templatesActive
    )

    return {
      chatHistory: applicationChatHistory,
      humanMessage,
      sources: source
    }
  },
  extractGenerationInfo: (output) => extractGenerationInfo(output)
}

export const documentChatMode = async (
  message: string,
  image: string,
  isRegenerate: boolean,
  messages: Message[],
  history: ChatHistory,
  signal: AbortSignal,
  uploadedFiles: UploadedFile[],
  params: Omit<DocumentChatModeParams, "uploadedFiles" | "newFiles" | "allFiles">
): Promise<ChatSubmitResult> => {
  await getAllDefaultModelSettings()

  let sessionFiles: UploadedFile[] = []
  const currentFiles: UploadedFile[] = uploadedFiles

  if (params.historyId) {
    sessionFiles = await getSessionFiles(params.historyId)
  }

  const newFiles = currentFiles.filter(
    (file) => !sessionFiles.some((sf) => sf.id === file.id)
  )

  const allFiles = [...sessionFiles, ...newFiles]
  const documentsForSave: any[] = uploadedFiles.map((file) => ({
    type: "file",
    filename: file.filename,
    fileSize: file.size,
    processed: file.processed
  }))

  const saveMessageOnSuccess = async (data: SaveMessageData) => {
    const chatHistoryId = await params.saveMessageOnSuccess(data)
    if (chatHistoryId) {
      for (const file of newFiles) {
        await addFileToSession(chatHistoryId, file)
      }
    }
    return chatHistoryId
  }

  const modeParams: DocumentChatModeParams = {
    ...params,
    saveMessageOnSuccess,
    documents: documentsForSave,
    uploadedFiles,
    newFiles,
    allFiles
  }

  return runChatPipeline(
    documentChatModeDefinition,
    message,
    image,
    isRegenerate,
    messages,
    history,
    signal,
    modeParams
  )
}
