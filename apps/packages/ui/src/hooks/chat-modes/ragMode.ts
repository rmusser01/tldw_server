import { promptForRag } from "~/services/tldw-server" // Reuse prompts storage for now
import {
  type ChatHistory,
  type Message,
  type ToolChoice,
  type Knowledge
} from "~/store/option"
import { generateHistory } from "@/utils/generate-history"
import { pageAssistModel } from "@/models"
import { humanMessageFormatter } from "@/utils/human-message"
import { removeReasoning } from "@/libs/reasoning"
import { formatDocs } from "@/utils/format-docs"
import { getNoOfRetrievedDocs } from "@/services/app"
import { DEFAULT_RAG_SETTINGS } from "@/services/rag/unified-rag"
import { coerceBooleanOrNull } from "@/services/settings/registry"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ActorSettings } from "@/types/actor"
import { maybeInjectActorMessage } from "@/utils/actor"
import { resolveApiProviderForModel } from "@/utils/resolve-api-provider"
import type { ChatModelSettings } from "@/store/model"
import type { SaveMessageData, SaveMessageErrorData } from "@/types/chat-modes"
import {
  runChatPipeline,
  type ChatModeDefinition
} from "./chatModePipeline"
import { appendSystemPromptSuffix } from "@/utils/output-formatting-guide"
import type { ChatSubmitResult } from "@/hooks/chat/chat-action-utils"

const RAG_STRING_ARRAY_KEYS = new Set([
  "sources",
  "include_note_ids",
  "expansion_strategies",
  "chunk_type_filter",
  "content_policy_types",
  "html_allowed_tags",
  "html_allowed_attrs",
  "batch_queries"
])
const RAG_NUMBER_ARRAY_KEYS = new Set(["include_media_ids"])
const RAG_NULLABLE_STRING_KEYS = new Set([
  "generation_model",
  "generation_provider",
  "generation_prompt",
  "user_id",
  "session_id"
])
const RAG_ALLOWED_KEYS = new Set([
  ...Object.keys(DEFAULT_RAG_SETTINGS).filter((key) => key !== "query"),
  "filters"
])

const normalizeStringArray = (values: unknown[]) => {
  if (values.length === 0) return null
  const normalized: string[] = []
  for (const entry of values) {
    if (typeof entry !== "string") return null
    const trimmed = entry.trim()
    if (!trimmed) return null
    normalized.push(trimmed)
  }
  return normalized
}

const normalizePositiveIntArray = (values: unknown[]) => {
  if (values.length === 0) return null
  const normalized: number[] = []
  for (const entry of values) {
    if (typeof entry !== "number" || !Number.isInteger(entry) || entry <= 0) {
      return null
    }
    normalized.push(entry)
  }
  return normalized
}

const sanitizeRagAdvancedOptions = (options?: Record<string, unknown>) => {
  if (!options) return {}
  const sanitized: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(options)) {
    if (value === undefined || value === null) continue
    if (typeof value === "string" && value.trim() === "") continue
    if (!RAG_ALLOWED_KEYS.has(key)) continue

    if (key === "filters") {
      if (typeof value !== "object" || Array.isArray(value)) continue
      sanitized[key] = value
      continue
    }

    if (RAG_STRING_ARRAY_KEYS.has(key)) {
      if (!Array.isArray(value)) continue
      if (key === "include_note_ids") {
        if (
          value.length > 0 &&
          value.every(
            (entry) => typeof entry === "number" && Number.isFinite(entry)
          )
        ) {
          sanitized[key] = value
          continue
        }

        const normalizedLegacyIds = normalizeStringArray(
          value.map((entry) =>
            typeof entry === "number" && Number.isFinite(entry)
              ? String(entry)
              : entry
          )
        )
        if (!normalizedLegacyIds) continue
        sanitized[key] = normalizedLegacyIds
        continue
      }
      const normalized = normalizeStringArray(value)
      if (!normalized) continue
      sanitized[key] = normalized
      continue
    }
    if (RAG_NUMBER_ARRAY_KEYS.has(key)) {
      if (!Array.isArray(value)) continue
      const normalized = normalizePositiveIntArray(value)
      if (!normalized) continue
      sanitized[key] = normalized
      continue
    }

    if (key === "top_k") {
      if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
        continue
      }
      sanitized[key] = value
      continue
    }

    const defaultValue = (DEFAULT_RAG_SETTINGS as Record<string, unknown>)[key]
    if (typeof defaultValue === "boolean") {
      const coerced = coerceBooleanOrNull(value)
      if (coerced === null) continue
      sanitized[key] = coerced
      continue
    }
    if (typeof defaultValue === "number") {
      if (typeof value !== "number" || !Number.isFinite(value)) continue
      sanitized[key] = value
      continue
    }
    if (typeof defaultValue === "string" || RAG_NULLABLE_STRING_KEYS.has(key)) {
      if (typeof value !== "string") continue
      const trimmed = value.trim()
      if (!trimmed) continue
      sanitized[key] = trimmed
      continue
    }
    if (
      defaultValue &&
      typeof defaultValue === "object" &&
      !Array.isArray(defaultValue)
    ) {
      if (typeof value !== "object" || Array.isArray(value)) continue
      sanitized[key] = value
    }
  }
  return sanitized
}

interface RagDocumentMetadata {
  source?: string
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

type RagSourceEntry = {
  name: string
  type: string
  mode: "rag"
  url: string
  pageContent: string
  metadata: RagDocumentMetadata
}

type RagModeParams = {
  selectedModel: string
  useOCR: boolean
  selectedKnowledge: Knowledge | null
  currentChatModelSettings: ChatModelSettings | null
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
  ragMediaIds: number[] | null
  ragSearchMode: "hybrid" | "vector" | "fts"
  ragTopK: number | null
  ragEnableGeneration: boolean
  ragEnableCitations: boolean
  ragSources: string[]
  ragAdvancedOptions?: Record<string, unknown>
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
}

const ragModeDefinition: ChatModeDefinition<RagModeParams> = {
  id: "rag",
  buildUserMessage: (ctx) => ({
    isBot: false,
    name: "You",
    message: ctx.message,
    sources: [],
    images: ctx.image ? [ctx.image] : [],
    createdAt: ctx.createdAt,
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
      const questionOllama = await pageAssistModel({
        model: ctx.selectedModel,
        toolChoice: "none",
        saveToDb: false
      })
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
      const response = await questionOllama.invoke([questionMessage])
      query = response.content.toString()
      query = removeReasoning(query)
    }

    const defaultTopK = await getNoOfRetrievedDocs()
    let context = ""
    let source: RagSourceEntry[] = []
    try {
      await tldwClient.initialize()
      const top_k =
        typeof ctx.ragTopK === "number" && ctx.ragTopK > 0
          ? ctx.ragTopK
          : defaultTopK
      const ragOptions: Record<string, unknown> = sanitizeRagAdvancedOptions(
        ctx.ragAdvancedOptions
      )
      // Precedence for top_k: (1) ctx.ragTopK if valid > 0, (2) ragOptions.top_k if valid > 0,
      // (3) defaultTopK fallback. ctx.ragSearchMode always overrides ragOptions.search_mode.
      // ctx.ragEnableGeneration/citations control presence of their flags, even if set.
      if (typeof ctx.ragTopK === "number" && ctx.ragTopK > 0) {
        ragOptions.top_k = ctx.ragTopK
      } else if (ragOptions.top_k == null) {
        ragOptions.top_k = top_k
      }
      ragOptions.search_mode = ctx.ragSearchMode
      // Delete false flags so the backend can apply its default behavior.
      if (ctx.ragEnableGeneration) {
        ragOptions.enable_generation = true
        const selectedGenerationModel = ctx.selectedModel?.trim()
        if (selectedGenerationModel) {
          ragOptions.generation_model = selectedGenerationModel
        }
        const selectedGenerationProvider = await resolveApiProviderForModel({
          modelId: selectedGenerationModel,
          explicitProvider: ctx.currentChatModelSettings?.apiProvider
        })
        if (selectedGenerationProvider) {
          ragOptions.generation_provider = selectedGenerationProvider
        }
      } else {
        delete ragOptions.enable_generation
      }
      if (ctx.ragEnableCitations) {
        ragOptions.enable_citations = true
      } else {
        delete ragOptions.enable_citations
      }
      // Precedence: ctx.ragSources overrides ragAdvancedOptions.sources; ctx.ragMediaIds
      // overrides include_media_ids and forces sources to ["media_db"].
      if (Array.isArray(ctx.ragSources) && ctx.ragSources.length > 0) {
        ragOptions.sources = ctx.ragSources
      }
      if (Array.isArray(ctx.ragMediaIds) && ctx.ragMediaIds.length > 0) {
        ragOptions.include_media_ids = ctx.ragMediaIds
        // When a specific media list is provided, ensure we query the media DB
        ragOptions.sources = ["media_db"]
      }
      const ragRes = (await tldwClient.ragSearch(
        query,
        ragOptions
      )) as RagResponse
      const docs: RagDocument[] =
        ragRes?.results || ragRes?.documents || ragRes?.docs || []
      context = formatDocs(
        docs.map((doc) => ({
          pageContent: doc.content || doc.text || doc.chunk || "",
          metadata: doc.metadata || {}
        }))
      )
      source = docs.map((doc) => ({
        name: doc.metadata?.source || doc.metadata?.title || "untitled",
        type: doc.metadata?.type || "unknown",
        mode: "rag",
        url: doc.metadata?.url || "",
        pageContent: doc.content || doc.text || doc.chunk || "",
        metadata: doc.metadata || {}
      }))
    } catch (e) {
      console.error("tldw ragSearch failed, continuing without context", e)
      context = ""
      source = []
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
  }
}

export const ragMode = async (
  message: string,
  image: string,
  isRegenerate: boolean,
  messages: Message[],
  history: ChatHistory,
  signal: AbortSignal,
  params: RagModeParams
): Promise<ChatSubmitResult> => {
  console.log("Using ragMode")
  return runChatPipeline(
    ragModeDefinition,
    message,
    image,
    isRegenerate,
    messages,
    history,
    signal,
    params
  )
}

export const __testing__ = {
  sanitizeRagAdvancedOptions
}
