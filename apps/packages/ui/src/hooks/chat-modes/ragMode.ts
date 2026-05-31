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
  type ChatModeContext,
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
  generated_answer?: string | null
  metadata?: Record<string, unknown>
  citations?: unknown[]
  academic_citations?: unknown[]
  chunk_citations?: unknown[]
  errors?: unknown[]
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

type PreparedRagRetrieval = {
  query: string
  context: string
  source: RagSourceEntry[]
  rawResponse?: RagResponse
}

const selectedSourceRetrievalCache = new WeakMap<object, PreparedRagRetrieval>()

const hasSelectedMediaSources = (ctx: Pick<RagModeParams, "ragMediaIds">) =>
  Array.isArray(ctx.ragMediaIds) && ctx.ragMediaIds.length > 0

const getRagDocuments = (ragRes: RagResponse | null | undefined) => {
  const docs = ragRes?.results || ragRes?.documents || ragRes?.docs || []
  return Array.isArray(docs) ? docs : []
}

const getMetadataString = (
  metadata: RagDocumentMetadata | undefined,
  key: string
) => {
  const value = metadata?.[key]
  return typeof value === "string" && value.trim() ? value.trim() : ""
}

const getGeneratedRagAnswer = (ragRes: RagResponse | null | undefined) =>
  typeof ragRes?.generated_answer === "string"
    ? ragRes.generated_answer.trim()
    : ""

const buildSelectedSourceNoEvidenceText = (
  ragRes: RagResponse | null | undefined
) => {
  const generatedAnswer = getGeneratedRagAnswer(ragRes)
  if (generatedAnswer) {
    return `${generatedAnswer}\n\nI did not send this as general chat because selected-source answers must be grounded in retrieved source evidence.`
  }

  return "I couldn't find supporting evidence in the selected sources for that question. I did not send this as general chat because selected-source answers must be grounded in retrieved source evidence. Try rephrasing the question, selecting more ready sources, or checking ingestion and indexing status."
}

const buildSelectedSourceRetrievalErrorText = (error: unknown) => {
  const detail =
    error instanceof Error && error.message.trim()
      ? `\n\nDetails: ${error.message.trim()}`
      : ""
  return `I couldn't retrieve evidence from the selected sources, so I did not send this as general chat. Check that the sources are ready and indexed, then try again.${detail}`
}

const buildSelectedSourceGroundingResponse = (
  fullText: string,
  reason: string,
  rawResponse?: RagResponse,
  error?: unknown
) => ({
  handled: true as const,
  fullText,
  sources: [],
  generationInfo: {
    mode: "rag",
    grounded: false,
    reason,
    retrievalMetadata: rawResponse?.metadata,
    retrievalErrors: rawResponse?.errors,
    retrievalError:
      error instanceof Error
        ? error.message
        : typeof error === "string"
          ? error
          : undefined
  },
  saveToDb: false
})

const buildSelectedSourceGeneratedAnswerResponse = (
  retrieval: PreparedRagRetrieval,
  fullText: string
) => ({
  handled: true as const,
  fullText,
  sources: retrieval.source,
  generationInfo: {
    mode: "rag",
    grounded: true,
    reason: "selected_source_generated_answer",
    retrievalMetadata: retrieval.rawResponse?.metadata,
    citations: retrieval.rawResponse?.citations,
    academicCitations: retrieval.rawResponse?.academic_citations,
    chunkCitations: retrieval.rawResponse?.chunk_citations
  },
  saveToDb: false
})

const resolveRagQuery = async (
  ctx: ChatModeContext<RagModeParams>,
  questionPrompt: string
) => {
  let query = ctx.message
  if (hasSelectedMediaSources(ctx)) {
    return query
  }

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

  if (contextMessages.length <= 2) {
    return query
  }

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
  return removeReasoning(query)
}

const buildRagOptions = async (
  ctx: ChatModeContext<RagModeParams>,
  defaultTopK: number
) => {
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
  if (hasSelectedMediaSources(ctx)) {
    ragOptions.include_media_ids = ctx.ragMediaIds
    ragOptions.sources = ["media_db"]
    // Explicit source selection is already the user's disambiguation context.
    // Backend intent routing can otherwise bypass retrieval and produce an
    // ungrounded clarification for source-bound questions.
    ragOptions.enable_intent_routing = false
    ragOptions.enable_pre_retrieval_clarification = false
  }
  return ragOptions
}

const prepareRagRetrieval = async (
  ctx: ChatModeContext<RagModeParams>,
  questionPrompt: string
): Promise<PreparedRagRetrieval> => {
  const query = await resolveRagQuery(ctx, questionPrompt)
  await tldwClient.initialize()
  const defaultTopK = await getNoOfRetrievedDocs()
  const ragOptions = await buildRagOptions(ctx, defaultTopK)
  const ragRes = (await tldwClient.ragSearch(query, ragOptions)) as RagResponse
  const docs = getRagDocuments(ragRes)
  const context = formatDocs(
    docs.map((doc) => ({
      pageContent: doc.content || doc.text || doc.chunk || "",
      metadata: doc.metadata || {}
    }))
  )
  const source = docs.map((doc) => ({
    name:
      getMetadataString(doc.metadata, "title") ||
      getMetadataString(doc.metadata, "source") ||
      "untitled",
    type:
      getMetadataString(doc.metadata, "type") ||
      getMetadataString(doc.metadata, "media_type") ||
      "unknown",
    mode: "rag" as const,
    url: getMetadataString(doc.metadata, "url"),
    pageContent: doc.content || doc.text || doc.chunk || "",
    metadata: doc.metadata || {}
  }))
  return {
    query,
    context,
    source,
    rawResponse: ragRes
  }
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
  preflight: async (ctx) => {
    if (!hasSelectedMediaSources(ctx)) {
      return null
    }

    const { ragQuestionPrompt: questionPrompt } = await promptForRag()
    try {
      const retrieval = await prepareRagRetrieval(ctx, questionPrompt)
      if (retrieval.source.length > 0) {
        const generatedAnswer = getGeneratedRagAnswer(retrieval.rawResponse)
        if (generatedAnswer) {
          return buildSelectedSourceGeneratedAnswerResponse(
            retrieval,
            generatedAnswer
          )
        }
        selectedSourceRetrievalCache.set(ctx, retrieval)
        return null
      }

      return buildSelectedSourceGroundingResponse(
        buildSelectedSourceNoEvidenceText(retrieval.rawResponse),
        "selected_source_evidence_not_found",
        retrieval.rawResponse
      )
    } catch (error) {
      return buildSelectedSourceGroundingResponse(
        buildSelectedSourceRetrievalErrorText(error),
        "selected_source_retrieval_failed",
        undefined,
        error
      )
    }
  },
  preparePrompt: async (ctx) => {
    const { ragPrompt: systemPrompt, ragQuestionPrompt: questionPrompt } =
      await promptForRag()
    const resolvedSystemPrompt = appendSystemPromptSuffix(
      systemPrompt,
      ctx.systemPromptAppendix
    )

    let context = ""
    let source: RagSourceEntry[] = []
    try {
      const cachedRetrieval = selectedSourceRetrievalCache.get(ctx)
      if (cachedRetrieval) {
        selectedSourceRetrievalCache.delete(ctx)
      }
      const retrieval =
        cachedRetrieval || (await prepareRagRetrieval(ctx, questionPrompt))
      context = retrieval.context
      source = retrieval.source
      if (hasSelectedMediaSources(ctx) && source.length === 0) {
        throw new Error(buildSelectedSourceNoEvidenceText(retrieval.rawResponse))
      }
    } catch (e) {
      if (hasSelectedMediaSources(ctx)) {
        throw e
      }
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
  sanitizeRagAdvancedOptions,
  ragModeDefinition
}
