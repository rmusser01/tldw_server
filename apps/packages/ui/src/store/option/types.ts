import type { ChatDocuments } from "@/models/ChatTypes"
import type { UploadedFile } from "@/db/dexie/types"
import type { ConversationState } from "@/services/tldw/TldwApiClient"
import type { RagPinnedResult } from "@/utils/rag-format"
import type { ToolCall, ToolCallResult } from "@/types/tool-calls"
import type { DiscoSkillComment } from "@/types/disco-skills"
import type { MessageSteeringMode } from "@/types/message-steering"
import type {
  QueuedRequest,
  QueuedRequestInput
} from "@/utils/chat-request-queue"

// Knowledge type is now server-side only; this is a placeholder for legacy compatibility
export type Knowledge = {
  id: string
  title: string
  [key: string]: any
}

export type WebSearch = {
  search_engine: string
  search_url: string
  search_query: string
  search_results: {
    title: string
    link: string
  }[]
}

export type ToolChoice = "auto" | "none" | "required"
export type CompareContinuationMode = "winner" | "compare"

export type ReplyTarget = {
  id: string
  preview: string
  name?: string
  isBot?: boolean
}

export type MessageVariant = {
  id?: string
  message: string
  sources?: any[]
  images?: string[]
  generationInfo?: any
  reasoning_time_taken?: number
  createdAt?: number
  serverMessageId?: string
  serverMessageVersion?: number
}

export type Message = {
  isBot: boolean
  name: string
  role?: "user" | "assistant" | "system"
  message: string
  sources: any[]
  images?: string[]
  search?: WebSearch
  reasoning_time_taken?: number
  createdAt?: number
  id?: string
  messageType?: string
  generationInfo?: any
  modelName?: string
  modelImage?: string
  documents?: ChatDocuments
  serverMessageId?: string
  serverMessageVersion?: number
  variants?: MessageVariant[]
  activeVariantIndex?: number
  // Compare/multi-model metadata (in-memory only)
  clusterId?: string
  modelId?: string
  parentMessageId?: string | null
  // Tool/function calls (optional)
  toolCalls?: ToolCall[]
  toolResults?: ToolCallResult[]
  // Disco skills annotation (optional)
  discoSkillComment?: DiscoSkillComment
  // Character message pin state (server metadata-backed)
  pinned?: boolean
  // Full metadata payload (if provided by server)
  metadataExtra?: Record<string, unknown>
  // Character turn metadata (multi-character + mood scaffolding)
  speakerCharacterId?: number | null
  speakerCharacterName?: string
  moodLabel?: string
  moodConfidence?: number | null
  moodTopic?: string | null
}

export type ChatHistory = {
  role: "user" | "assistant" | "system"
  content: string
  image?: string
  messageType?: string
}[]

export type State = {
  messages: Message[]
  // Accepts either direct Message[] or functional update (like React setState)
  setMessages: (
    messagesOrUpdater: Message[] | ((prev: Message[]) => Message[])
  ) => void
  history: ChatHistory
  // Accepts either direct ChatHistory or functional update (like React setState)
  setHistory: (
    historyOrUpdater:
      | ChatHistory
      | ((prev: ChatHistory) => ChatHistory)
  ) => void
  streaming: boolean
  setStreaming: (streaming: boolean) => void
  isFirstMessage: boolean
  setIsFirstMessage: (isFirstMessage: boolean) => void
  historyId: string | null
  setHistoryId: (
    history_id: string | null,
    options?: { preserveServerChatId?: boolean }
  ) => void
  isLoading: boolean
  setIsLoading: (isLoading: boolean) => void
  isProcessing: boolean
  setIsProcessing: (isProcessing: boolean) => void
  selectedModel: string | null
  setSelectedModel: (selectedModel: string | null) => void
  chatMode: "normal" | "rag" | "vision"
  setChatMode: (chatMode: "normal" | "rag" | "vision") => void
  isEmbedding: boolean
  setIsEmbedding: (isEmbedding: boolean) => void
  webSearch: boolean
  setWebSearch: (webSearch: boolean) => void
  toolChoice: ToolChoice
  setToolChoice: (toolChoice: ToolChoice) => void
  isSearchingInternet: boolean
  setIsSearchingInternet: (isSearchingInternet: boolean) => void
  selectedSystemPrompt: string | null
  setSelectedSystemPrompt: (selectedSystemPrompt: string) => void
  selectedQuickPrompt: string | null
  setSelectedQuickPrompt: (selectedQuickPrompt: string) => void
  messageSteeringMode: MessageSteeringMode
  setMessageSteeringMode: (mode: MessageSteeringMode) => void
  messageSteeringForceNarrate: boolean
  setMessageSteeringForceNarrate: (enabled: boolean) => void
  clearMessageSteering: () => void
  queuedMessages: QueuedRequest[]
  addQueuedMessage: (payload: QueuedRequestInput) => void
  setQueuedMessages: (
    messagesOrUpdater:
      | QueuedRequestInput[]
      | ((prev: QueuedRequest[]) => QueuedRequestInput[])
  ) => void
  clearQueuedMessages: () => void
  selectedKnowledge: Knowledge | null
  setSelectedKnowledge: (selectedKnowledge: Knowledge | null) => void
  setSpeechToTextLanguage: (language: string) => void
  speechToTextLanguage: string
  temporaryChat: boolean
  setTemporaryChat: (temporaryChat: boolean) => void
  useOCR: boolean
  setUseOCR: (useOCR: boolean) => void
  documentContext: ChatDocuments | null
  setDocumentContext: (documentContext: ChatDocuments) => void
  uploadedFiles: UploadedFile[]
  setUploadedFiles: (uploadedFiles: UploadedFile[]) => void
  contextFiles: UploadedFile[]
  setContextFiles: (contextFiles: UploadedFile[]) => void
  actionInfo: string | null
  setActionInfo: (actionInfo: string) => void
  fileRetrievalEnabled: boolean
  setFileRetrievalEnabled: (fileRetrievalEnabled: boolean) => void
  // RAG media-scoped filters (e.g., "chat about this media")
  ragMediaIds: number[] | null
  setRagMediaIds: (ids: number[] | null) => void
  // Unified RAG configuration (shared between Chat and Knowledge views)
  ragSearchMode: "hybrid" | "vector" | "fts"
  setRagSearchMode: (mode: "hybrid" | "vector" | "fts") => void
  ragTopK: number | null
  setRagTopK: (value: number | null) => void
  ragEnableGeneration: boolean
  setRagEnableGeneration: (value: boolean) => void
  ragEnableCitations: boolean
  setRagEnableCitations: (value: boolean) => void
  ragSources: string[]
  setRagSources: (sources: string[]) => void
  // Advanced RAG options (free-form UnifiedRAGRequest fields)
  ragAdvancedOptions: Record<string, any>
  setRagAdvancedOptions: (opts: Record<string, any>) => void
  // Session-only pinned RAG results for the Search & Context modal
  ragPinnedResults: RagPinnedResult[]
  setRagPinnedResults: (results: RagPinnedResult[]) => void
  // Server-backed character chat id
  serverChatId: string | null
  setServerChatId: (id: string | null) => void
  serverChatTitle: string | null
  setServerChatTitle: (title: string | null) => void
  serverChatCharacterId: string | number | null
  setServerChatCharacterId: (id: string | number | null) => void
  serverChatAssistantKind: "character" | "persona" | null
  setServerChatAssistantKind: (
    kind: "character" | "persona" | null
  ) => void
  serverChatAssistantId: string | null
  setServerChatAssistantId: (id: string | null) => void
  serverChatPersonaMemoryMode: "read_only" | "read_write" | null
  setServerChatPersonaMemoryMode: (
    mode: "read_only" | "read_write" | null
  ) => void
  serverChatMetaLoaded: boolean
  setServerChatMetaLoaded: (loaded: boolean) => void
  serverChatLoadState: "idle" | "loading" | "loaded" | "failed"
  setServerChatLoadState: (
    state: "idle" | "loading" | "loaded" | "failed"
  ) => void
  serverChatLoadError: string | null
  setServerChatLoadError: (error: string | null) => void
  serverChatState: ConversationState | null
  setServerChatState: (state: ConversationState | null) => void
  serverChatVersion: number | null
  setServerChatVersion: (version: number | null) => void
  serverChatTopic: string | null
  setServerChatTopic: (topic: string | null) => void
  serverChatClusterId: string | null
  setServerChatClusterId: (clusterId: string | null) => void
  serverChatSource: string | null
  setServerChatSource: (source: string | null) => void
  serverChatExternalRef: string | null
  setServerChatExternalRef: (ref: string | null) => void
  // Compare mode (multi-model) state
  compareMode: boolean
  setCompareMode: (on: boolean) => void
  compareSelectedModels: string[]
  setCompareSelectedModels: (models: string[]) => void
  compareSelectionByCluster: Record<string, string[]>
  setCompareSelectionForCluster: (clusterId: string, models: string[]) => void
  compareActiveModelsByCluster: Record<string, string[]>
  setCompareActiveModelsForCluster: (clusterId: string, models: string[]) => void
  // Compare breadcrumbs / canonical state
  compareParentByHistory: Record<
    string,
    {
      parentHistoryId: string
      clusterId?: string
    }
  >
  setCompareParentForHistory: (
    historyId: string,
    meta: { parentHistoryId: string; clusterId?: string }
  ) => void
  compareCanonicalByCluster: Record<string, string | null>
  setCompareCanonicalForCluster: (clusterId: string, messageId: string | null) => void
  compareContinuationModeByCluster: Record<string, CompareContinuationMode>
  setCompareContinuationModeForCluster: (
    clusterId: string,
    mode: CompareContinuationMode
  ) => void
  // Split-off chats per compare cluster/model
  compareSplitChats: Record<string, Record<string, string>>
  setCompareSplitChat: (clusterId: string, modelKey: string, historyId: string) => void
  replyTarget: ReplyTarget | null
  setReplyTarget: (target: ReplyTarget) => void
  clearReplyTarget: () => void
}
