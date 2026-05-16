/**
 * Types for KnowledgeQA component
 */

import type { RagSettings, RagPresetName } from "@/services/rag/unified-rag"

export type QueryStage =
  | "idle"
  | "searching"
  | "ranking"
  | "generating"
  | "verifying"
  | "complete"
  | "error"

export type ScopeSnapshot = {
  preset: RagPresetName
  sources: RagSettings["sources"]
  webFallback: boolean
  includeMediaIds: number[]
  includeNoteIds: string[]
}

export type PinnedSourceFilters = {
  mediaIds: number[]
  noteIds: string[]
}

export type ThreadHydrationResult = boolean | "terminal"

// Retrieved document with citation info
export type RagResult = {
  id?: string
  content?: string
  text?: string
  chunk?: string
  metadata?: {
    title?: string
    source?: string
    url?: string
    page_number?: number
    chunk_id?: string
    source_type?: string
    [key: string]: unknown
  }
  score?: number
  excerpt?: string
}

// Citation reference in generated answer
export type CitationRef = {
  index: number // 1-based index as shown in answer [1], [2], etc.
  documentId: string
  excerpt?: string
  startOffset?: number
  endOffset?: number
}

// RAG context to be persisted with messages
export type RagContextData = {
  search_query: string
  search_mode?: string
  settings_snapshot?: Partial<RagSettings>
  retrieved_documents: Array<{
    id?: string
    source_type?: string
    title?: string
    score?: number
    chunk_id?: string
    excerpt?: string
    url?: string
    page_number?: number
    line_range?: [number, number]
    metadata?: Record<string, unknown>
  }>
  generated_answer?: string
  citations?: Array<{
    index?: number
    text: string
    source: string
    confidence: number
    type: string
  }>
  claims_verified?: Array<{
    claim: string
    verified: boolean
    confidence: number
    source?: string
  }>
  timestamp?: string
  feedback_id?: string
}

export type SearchRuntimeDetails = {
  expandedQueries: string[]
  rerankingEnabled: boolean
  rerankingStrategy: string
  averageRelevance: number | null
  sourceStatus: Record<string, KnowledgeSourceStatus>
  webFallbackEnabled: boolean
  webFallbackTriggered: boolean
  webFallbackEngine: string | null
  tokensUsed: number | null
  estimatedCostUsd: number | null
  feedbackId: string | null
  whyTheseSources: {
    topicality: number | null
    diversity: number | null
    freshness: number | null
  } | null
  faithfulnessScore: number | null
  faithfulnessTotalClaims: number | null
  faithfulnessSupportedClaims: number | null
  faithfulnessUnsupportedClaims: number | null
  verificationRate: number | null
  verificationCoverage: number | null
  verificationTotalClaims: number | null
  verificationVerifiedClaims: number | null
  verificationReportAvailable: boolean
  retrievalLatencyMs: number | null
  documentsConsidered: number | null
  chunksConsidered: number | null
  documentsReturned: number
  candidatesConsidered: number | null
  candidatesReturned: number
  candidatesRejected: number | null
  alsoConsidered: Array<{
    id: string
    title: string
    score: number | null
    reason: string | null
  }>
}

export type KnowledgeSourceStatus = {
  status: "searched" | "empty" | "unavailable" | "skipped" | "error" | string
  count: number
  reason?: string
}

export type KnowledgeSourceIndexStatus =
  | "ready"
  | "indexing"
  | "stale"
  | "empty"
  | "unavailable"
  | "error"
  | "unknown"

export type KnowledgeSourceEmbeddingStatus =
  | "ready"
  | "indexing"
  | "missing"
  | "unavailable"
  | "not_applicable"
  | "error"
  | "unknown"

export type KnowledgeSourceHealth = {
  sourceId: RagSettings["sources"][number]
  label: string
  available: boolean
  searchable: boolean
  itemCount: number | null
  indexedCount: number | null
  lastUpdated: string | null
  lastIndexed: string | null
  indexStatus: KnowledgeSourceIndexStatus
  embeddingStatus: KnowledgeSourceEmbeddingStatus
  disabledReason: string | null
  workspaceScoped: boolean
  hiddenByDefault: boolean
  privacyNote: string | null
}

export type KnowledgeSourceHealthState = {
  bySource: Partial<Record<RagSettings["sources"][number], KnowledgeSourceHealth>>
  sources: KnowledgeSourceHealth[]
  loading: boolean
  error: string | null
  loadedAt: string | null
}

// Search history item
export type SearchHistoryItem = {
  id: string
  query: string
  timestamp: string
  conversationId?: string
  messageId?: string
  sourcesCount: number
  hasAnswer: boolean
  answerPreview?: string
  pinned?: boolean
  preset?: RagPresetName
  settingsSnapshot?: Partial<RagSettings>
  keywords?: string[]
}

// Thread/conversation for Knowledge QA
export type KnowledgeQAThread = {
  id: string
  title?: string
  topicLabel?: string
  createdAt: string
  lastModifiedAt: string
  state: "in-progress" | "resolved"
  messageCount: number
  source: "knowledge_qa"
}

// Message with RAG context
export type KnowledgeQAMessage = {
  id: string
  conversationId: string
  role: "user" | "assistant" | "system"
  content: string
  timestamp: string
  ragContext?: RagContextData
}

// State for the KnowledgeQA component
export type KnowledgeQAState = {
  // Search state
  query: string
  isSearching: boolean
  hasSearched: boolean
  results: RagResult[]
  answer: string | null
  citations: CitationRef[]
  searchDetails: SearchRuntimeDetails | null
  error: string | null
  queryWarning: string | null

  // Thread state
  currentThreadId: string | null
  isLocalOnlyThread: boolean
  messages: KnowledgeQAMessage[]
  threads: KnowledgeQAThread[]

  // Settings state
  preset: RagPresetName
  settings: RagSettings
  expertMode: boolean

  // History state
  searchHistory: SearchHistoryItem[]
  historySidebarOpen: boolean

  // UI state
  settingsPanelOpen: boolean
  focusedSourceIndex: number | null
  evidenceRailOpen: boolean
  evidenceRailTab: "sources" | "details"
  queryStage: QueryStage
  lastSearchScope: ScopeSnapshot | null
  pinnedSourceFilters: PinnedSourceFilters
  sourceHealth: KnowledgeSourceHealthState
}

// Actions for KnowledgeQA
export type KnowledgeQAActions = {
  // Search actions
  setQuery: (query: string) => void
  search: () => Promise<void>
  cancelSearch: () => void
  clearResults: () => void
  rerunWithTokenLimit: (tokenLimit: number) => Promise<void>

  // Thread actions
  createNewThread: (title?: string) => Promise<string>
  startNewTopic: () => Promise<string>
  selectThread: (threadId: string) => Promise<ThreadHydrationResult>
  selectSharedThread: (shareToken: string) => Promise<ThreadHydrationResult>
  askFollowUp: (question: string) => Promise<void>
  branchFromTurn: (messageId: string) => Promise<void>

  // Settings actions
  setPreset: (preset: RagPresetName) => void
  updateSetting: <K extends keyof RagSettings>(key: K, value: RagSettings[K]) => void
  resetSettings: () => void
  toggleExpertMode: () => void

  // History actions
  loadSearchHistory: () => Promise<void>
  restoreFromHistory: (item: SearchHistoryItem) => Promise<void>
  deleteHistoryItem: (id: string) => Promise<void>
  toggleHistoryPin: (id: string) => void

  // UI actions
  setSettingsPanelOpen: (open: boolean) => void
  setHistorySidebarOpen: (open: boolean) => void
  focusSource: (index: number | null) => void
  setEvidenceRailOpen: (open: boolean) => void
  setEvidenceRailTab: (tab: "sources" | "details") => void
  setQueryStage: (stage: QueryStage) => void
  setPinnedSourceFilters: (filters: PinnedSourceFilters) => void
  refreshSourceHealth: () => Promise<void>

  // Citation actions
  persistRagContext: (messageId: string, context: RagContextData) => Promise<boolean>
  scrollToSource: (index: number) => void
  scrollToCitation: (citationIndex: number, occurrence?: number) => void
}

// Context value combining state and actions
export type KnowledgeQAContextValue = KnowledgeQAState &
  KnowledgeQAActions & {
    historyHydrated: boolean
  }

// Export format options
export type ExportFormat = "markdown" | "pdf" | "chatbook"

export type ExportOptions = {
  format: ExportFormat
  includeSettingsSnapshot: boolean
  includeSourceExcerpts: boolean
  citationStyle: "apa" | "mla" | "chicago" | "harvard" | "ieee"
}
