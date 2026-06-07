/**
 * KnowledgeQAProvider - Context provider for Knowledge QA state management
 */

import React, {
  createContext,
  useContext,
  useReducer,
  useCallback,
  useMemo,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { useStorage } from "@plasmohq/storage/hook"
import type {
  KnowledgeQAState,
  KnowledgeQAActions,
  KnowledgeQAContextValue,
  RagResult,
  RagContextData,
  SearchHistoryItem,
  KnowledgeQAMessage,
  KnowledgeQAThread,
  CitationRef,
  KnowledgeAnswerTrustState,
  SearchRuntimeDetails,
  KnowledgeSourceStatus,
  KnowledgeSourceHealthState,
  QueryStage,
  ScopeSnapshot,
  PinnedSourceFilters,
  ThreadHydrationResult,
} from "./types"
import {
  DEFAULT_RAG_SETTINGS,
  applyRagPreset,
  buildRagSearchRequest,
  type RagSearchMode,
  type RagSettings,
  type RagPresetName,
} from "@/services/rag/unified-rag"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { KNOWLEDGE_QA_KEYWORD } from "./constants"
import { trackKnowledgeQaSearchMetric } from "@/utils/knowledge-qa-search-metrics"
import { persistKnowledgeQaHistory } from "./historyStorage"
import { mapKnowledgeQaSearchErrorMessage } from "./errorMessages"
import { truncateAnswerPreview } from "./historyUtils"
import {
  EMPTY_SOURCE_HEALTH_STATE,
  normalizeKnowledgeSourceHealth,
} from "./sourceHealth"
import {
  isKnowledgeAnswerTrustState,
  normalizeKnowledgeAnswerTrust,
} from "./trustState"
import {
  getEvidenceOrigin,
  getResultChunkId,
  getResultEvidenceText,
  getResultSourceId,
  getResultSourceStatus,
  getResultUnavailableReason,
} from "./sourceListUtils"

const LOCAL_THREAD_PREFIX = "local-"
const DEFAULT_CHARACTER_NAME = "Helpful AI Assistant"
const RAG_QUERY_MAX_LENGTH = 20000
const VALID_WEB_FALLBACK_MERGE_STRATEGIES = ["prepend", "append", "interleave"] as const
const VALID_RAG_SEARCH_MODES: ReadonlyArray<RagSearchMode> = ["fts", "vector", "hybrid"]

function isValidWebFallbackMergeStrategy(
  value: string
): value is RagSettings["web_fallback_merge_strategy"] {
  return VALID_WEB_FALLBACK_MERGE_STRATEGIES.includes(
    value as RagSettings["web_fallback_merge_strategy"]
  )
}

function isValidRagSearchMode(value: string): value is RagSearchMode {
  return VALID_RAG_SEARCH_MODES.includes(value as RagSearchMode)
}

type CreateThreadOptions = {
  parentConversationId?: string
  forkedFromMessageId?: string
  shouldActivate?: () => boolean
  cleanupRemoteIfSkipped?: boolean
}
const KNOWLEDGE_QA_SETTINGS_OVERRIDES: Partial<RagSettings> = {
  enable_web_fallback: true,
  parent_context_size: 100,
  agentic_time_budget_sec: 30,
  agentic_max_redundancy: 1,
}

// Initial state
const initialState: KnowledgeQAState = {
  query: "",
  isSearching: false,
  hasSearched: false,
  results: [],
  answer: null,
  citations: [],
  answerTrustState: "unknown_trust",
  searchDetails: null,
  error: null,
  queryWarning: null,

  currentThreadId: null,
  isLocalOnlyThread: false,
  messages: [],
  threads: [],

  preset: "balanced",
  settings: { ...DEFAULT_RAG_SETTINGS, ...KNOWLEDGE_QA_SETTINGS_OVERRIDES },
  expertMode: false,

  searchHistory: [],
  historySidebarOpen: false,

  settingsPanelOpen: false,
  focusedSourceIndex: null,
  evidenceRailOpen: false,
  evidenceRailTab: "sources",
  queryStage: "idle",
  lastSearchScope: null,
  pinnedSourceFilters: {
    mediaIds: [],
    noteIds: [],
  },
  sourceHealth: EMPTY_SOURCE_HEALTH_STATE,
}

const isLocalThreadId = (id: string | null | undefined) =>
  Boolean(id && id.startsWith(LOCAL_THREAD_PREFIX))

type ResultsPayload = {
  results: RagResult[]
  answer: string | null
  citations: CitationRef[]
  answerTrustState: KnowledgeAnswerTrustState
}

function sliceMessagesForBranch(
  messages: KnowledgeQAMessage[],
  fromUserMessageId: string
): KnowledgeQAMessage[] {
  if (!fromUserMessageId) return []
  const userIndex = messages.findIndex(
    (message) => message.id === fromUserMessageId && message.role === "user"
  )
  if (userIndex < 0) return []

  let branchEndExclusive = messages.length
  for (let index = userIndex + 1; index < messages.length; index += 1) {
    if (messages[index].role === "user") {
      branchEndExclusive = index
      break
    }
  }

  return messages.slice(0, branchEndExclusive)
}

// Action types
type Action =
  | { type: "SET_QUERY"; payload: string }
  | { type: "SET_QUERY_WARNING"; payload: string | null }
  | { type: "SET_SEARCHING"; payload: boolean }
  | { type: "SET_RESULTS"; payload: ResultsPayload }
  | { type: "SET_PARTIAL_RESULTS"; payload: ResultsPayload }
  | { type: "SET_SEARCH_DETAILS"; payload: SearchRuntimeDetails | null }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "CLEAR_RESULTS" }
  | { type: "SET_THREAD_ID"; payload: string | null }
  | { type: "SET_LOCAL_ONLY_THREAD"; payload: boolean }
  | { type: "SET_MESSAGES"; payload: KnowledgeQAMessage[] }
  | { type: "ADD_MESSAGE"; payload: KnowledgeQAMessage }
  | { type: "SET_THREADS"; payload: KnowledgeQAThread[] }
  | { type: "ADD_THREAD"; payload: KnowledgeQAThread }
  | { type: "SET_PRESET"; payload: RagPresetName }
  | {
      type: "HYDRATE_DEFAULTS"
      payload: { preset: RagPresetName; settings: RagSettings }
    }
  | { type: "SET_SETTINGS"; payload: RagSettings }
  | { type: "UPDATE_SETTING"; payload: { key: keyof RagSettings; value: unknown } }
  | { type: "TOGGLE_EXPERT_MODE" }
  | { type: "SET_SEARCH_HISTORY"; payload: SearchHistoryItem[] }
  | { type: "ADD_HISTORY_ITEM"; payload: SearchHistoryItem }
  | { type: "UPDATE_HISTORY_ITEM"; payload: { id: string; patch: Partial<SearchHistoryItem> } }
  | { type: "REMOVE_HISTORY_ITEM"; payload: string }
  | { type: "TOGGLE_HISTORY_PIN"; payload: string }
  | { type: "SET_SETTINGS_PANEL_OPEN"; payload: boolean }
  | { type: "SET_HISTORY_SIDEBAR_OPEN"; payload: boolean }
  | { type: "SET_FOCUSED_SOURCE"; payload: number | null }
  | { type: "SET_EVIDENCE_RAIL_OPEN"; payload: boolean }
  | { type: "SET_EVIDENCE_RAIL_TAB"; payload: "sources" | "details" }
  | { type: "SET_QUERY_STAGE"; payload: QueryStage }
  | { type: "SET_LAST_SEARCH_SCOPE"; payload: ScopeSnapshot | null }
  | { type: "SET_PINNED_SOURCE_FILTERS"; payload: PinnedSourceFilters }
  | { type: "SET_SOURCE_HEALTH_LOADING" }
  | { type: "SET_SOURCE_HEALTH"; payload: KnowledgeSourceHealthState }
  | { type: "SET_SOURCE_HEALTH_ERROR"; payload: string }
  | {
      type: "HYDRATE_RESTORED_SCOPE"
      payload: {
        preset?: RagPresetName
        settingsSnapshot?: Partial<RagSettings> | null
      }
    }

// Reducer
function reducer(state: KnowledgeQAState, action: Action): KnowledgeQAState {
  switch (action.type) {
    case "SET_QUERY":
      return { ...state, query: action.payload, queryWarning: null }
    case "SET_QUERY_WARNING":
      return { ...state, queryWarning: action.payload }
    case "SET_SEARCHING":
      return {
        ...state,
        isSearching: action.payload,
        error: action.payload ? null : state.error,
      }
    case "SET_RESULTS":
      return {
        ...state,
        results: action.payload.results,
        answer: action.payload.answer,
        citations: action.payload.citations,
        answerTrustState: action.payload.answerTrustState,
        hasSearched: true,
        isSearching: false,
        queryStage: "complete",
      }
    case "SET_PARTIAL_RESULTS":
      return {
        ...state,
        results: action.payload.results,
        answer: action.payload.answer,
        citations: action.payload.citations,
        answerTrustState: action.payload.answerTrustState,
        hasSearched: true,
        isSearching: true,
      }
    case "SET_SEARCH_DETAILS":
      return { ...state, searchDetails: action.payload }
    case "SET_ERROR":
      return {
        ...state,
        error: action.payload,
        answerTrustState: action.payload ? "failed_search" : state.answerTrustState,
        hasSearched: true,
        isSearching: false,
        queryStage: action.payload ? "error" : "idle",
      }
    case "CLEAR_RESULTS":
      return {
        ...state,
        results: [],
        answer: null,
        citations: [],
        answerTrustState: "unknown_trust",
        searchDetails: null,
        error: null,
        queryWarning: null,
        hasSearched: false,
        isSearching: false,
        queryStage: "idle",
        pinnedSourceFilters: {
          mediaIds: [],
          noteIds: [],
        },
      }
    case "SET_THREAD_ID":
      return { ...state, currentThreadId: action.payload }
    case "SET_LOCAL_ONLY_THREAD":
      return { ...state, isLocalOnlyThread: action.payload }
    case "SET_MESSAGES":
      return { ...state, messages: action.payload }
    case "ADD_MESSAGE":
      return { ...state, messages: [...state.messages, action.payload] }
    case "SET_THREADS":
      return { ...state, threads: action.payload }
    case "ADD_THREAD":
      return { ...state, threads: [action.payload, ...state.threads] }
    case "SET_PRESET": {
      const presetSettings =
        action.payload === "custom"
          ? state.settings
          : applyRagPreset(action.payload as Exclude<RagPresetName, "custom">)
      return {
        ...state,
        preset: action.payload,
        settings:
          action.payload === "custom"
            ? state.settings
            : {
                ...presetSettings,
                ...KNOWLEDGE_QA_SETTINGS_OVERRIDES,
                enable_web_fallback: state.settings.enable_web_fallback,
              },
      }
    }
    case "HYDRATE_DEFAULTS":
      return {
        ...state,
        preset: action.payload.preset,
        settings: action.payload.settings,
      }
    case "SET_SETTINGS":
      return { ...state, settings: action.payload, preset: "custom" }
    case "UPDATE_SETTING":
      return {
        ...state,
        settings: { ...state.settings, [action.payload.key]: action.payload.value },
        preset: "custom",
      }
    case "TOGGLE_EXPERT_MODE":
      return { ...state, expertMode: !state.expertMode }
    case "SET_SEARCH_HISTORY":
      return { ...state, searchHistory: action.payload }
    case "ADD_HISTORY_ITEM":
      return { ...state, searchHistory: [action.payload, ...state.searchHistory.slice(0, 99)] }
    case "UPDATE_HISTORY_ITEM":
      return {
        ...state,
        searchHistory: state.searchHistory.map((item) =>
          item.id === action.payload.id
            ? { ...item, ...action.payload.patch }
            : item
        ),
      }
    case "REMOVE_HISTORY_ITEM":
      return { ...state, searchHistory: state.searchHistory.filter((h) => h.id !== action.payload) }
    case "TOGGLE_HISTORY_PIN":
      return {
        ...state,
        searchHistory: state.searchHistory.map((item) =>
          item.id === action.payload
            ? { ...item, pinned: !item.pinned }
            : item
        ),
      }
    case "SET_SETTINGS_PANEL_OPEN":
      return { ...state, settingsPanelOpen: action.payload }
    case "SET_HISTORY_SIDEBAR_OPEN":
      return { ...state, historySidebarOpen: action.payload }
    case "SET_FOCUSED_SOURCE":
      return { ...state, focusedSourceIndex: action.payload }
    case "SET_EVIDENCE_RAIL_OPEN":
      return { ...state, evidenceRailOpen: action.payload }
    case "SET_EVIDENCE_RAIL_TAB":
      return { ...state, evidenceRailTab: action.payload }
    case "SET_QUERY_STAGE":
      return { ...state, queryStage: action.payload }
    case "SET_LAST_SEARCH_SCOPE":
      return { ...state, lastSearchScope: action.payload }
    case "SET_PINNED_SOURCE_FILTERS":
      return { ...state, pinnedSourceFilters: action.payload }
    case "SET_SOURCE_HEALTH_LOADING":
      return {
        ...state,
        sourceHealth: {
          ...state.sourceHealth,
          loading: true,
          error: null,
        },
      }
    case "SET_SOURCE_HEALTH":
      return { ...state, sourceHealth: action.payload }
    case "SET_SOURCE_HEALTH_ERROR":
      return {
        ...state,
        sourceHealth: {
          ...state.sourceHealth,
          loading: false,
          error: action.payload,
        },
      }
    case "HYDRATE_RESTORED_SCOPE": {
      const { nextPreset, nextSettings } = resolveHydratedScopeState(
        state.preset,
        state.settings,
        action.payload
      )
      return {
        ...state,
        preset: nextPreset,
        settings: nextSettings,
        lastSearchScope: buildScopeSnapshot(nextPreset, nextSettings),
      }
    }
    default:
      return state
  }
}

// Context
const KnowledgeQAContext = createContext<KnowledgeQAContextValue | null>(null)

// Parse citation indices from generated answer [1], [2], etc.
function parseCitations(answer: string, results: RagResult[]): CitationRef[] {
  const citationMatches = answer.match(/\[(\d+)\]/g) || []
  const indices = citationMatches
    .map((m) => parseInt(m.replace(/[\[\]]/g, ""), 10))
    .filter((i) => i >= 1 && i <= results.length)

  const uniqueIndices = [...new Set(indices)]
  return uniqueIndices.map((index) => ({
    index,
    documentId: results[index - 1]?.id || `doc_${index}`,
    excerpt: results[index - 1]?.content || results[index - 1]?.text,
  }))
}

function normalizeAnswerText(value: unknown): string | null {
  if (typeof value !== "string") return null
  return value.trim().length > 0 ? value : null
}

function normalizeNoteFilterIds(values: unknown): string[] {
  if (!Array.isArray(values)) return []
  const normalizedValues = values.map((candidate) => {
    if (typeof candidate === "string") return candidate
    if (typeof candidate === "number" && Number.isFinite(candidate)) {
      return String(Math.trunc(candidate))
    }
    return null
  })
  return mergeStringFilters(normalizedValues)
}

function normalizeMessageRole(role: unknown): KnowledgeQAMessage["role"] {
  const normalized = String(role ?? "").toLowerCase()
  if (normalized === "assistant" || normalized === "user" || normalized === "system") {
    return normalized
  }
  return "system"
}

function normalizeMessagesWithContext(
  payload: unknown,
  threadId: string
): KnowledgeQAMessage[] {
  if (!Array.isArray(payload)) return []

  return payload
    .map((entry, index) => {
      if (!entry || typeof entry !== "object") return null
      const candidate = entry as Record<string, unknown>
      const idRaw = candidate.id
      const id =
        typeof idRaw === "string" && idRaw.length > 0
          ? idRaw
          : `thread-message-${index + 1}`
      const content =
        typeof candidate.content === "string"
          ? candidate.content
          : String(candidate.content ?? "")
      const timestampCandidate = [
        candidate.timestamp,
        candidate.created_at,
        candidate.createdAt,
        candidate.updated_at,
      ].find((value) => typeof value === "string" && value.length > 0)
      const ragContextRaw = candidate.ragContext ?? candidate.rag_context

      return {
        id,
        conversationId: threadId,
        role: normalizeMessageRole(candidate.role ?? candidate.sender),
        content,
        timestamp:
          typeof timestampCandidate === "string"
            ? timestampCandidate
            : new Date().toISOString(),
        ragContext:
          ragContextRaw && typeof ragContextRaw === "object"
            ? (ragContextRaw as RagContextData)
            : undefined,
      } as KnowledgeQAMessage
    })
    .filter((message): message is KnowledgeQAMessage => message != null)
}

function mapRagContextDocumentsToResults(
  retrievedDocuments: unknown
): RagResult[] {
  if (!Array.isArray(retrievedDocuments)) return []

  return retrievedDocuments.map((document, index) => {
    const doc = document as Record<string, unknown>
    const metadata =
      doc?.metadata && typeof doc.metadata === "object" && !Array.isArray(doc.metadata)
        ? (doc.metadata as Record<string, unknown>)
        : {}
    const sourceId =
      typeof doc?.source_id === "string"
        ? doc.source_id
        : typeof metadata.source_id === "string"
          ? metadata.source_id
          : undefined
    const chunkId =
      typeof doc?.chunk_id === "string"
        ? doc.chunk_id
        : typeof metadata.chunk_id === "string"
          ? metadata.chunk_id
          : undefined
    const evidenceOrigin =
      typeof doc?.evidence_origin === "string"
        ? doc.evidence_origin
        : typeof metadata.evidence_origin === "string"
          ? metadata.evidence_origin
          : undefined
    const sourceStatus =
      typeof doc?.source_status === "string"
        ? doc.source_status
        : typeof metadata.source_status === "string"
          ? metadata.source_status
          : undefined
    const unavailableReason =
      typeof doc?.unavailable_reason === "string"
        ? doc.unavailable_reason
        : typeof metadata.unavailable_reason === "string"
          ? metadata.unavailable_reason
          : undefined
    return {
      id:
        typeof doc?.id === "string" && doc.id.length > 0
          ? doc.id
          : `doc-${index + 1}`,
      sourceId,
      chunkId,
      evidenceOrigin,
      sourceStatus,
      unavailableReason,
      score: typeof doc?.score === "number" ? doc.score : undefined,
      content:
        typeof doc?.excerpt === "string"
          ? doc.excerpt
          : typeof doc?.content === "string"
            ? doc.content
            : undefined,
      text:
        typeof doc?.excerpt === "string"
          ? doc.excerpt
          : typeof doc?.text === "string"
            ? doc.text
            : undefined,
      metadata: {
        ...metadata,
        title:
          typeof doc?.title === "string" ? doc.title : `Source ${index + 1}`,
        source:
          typeof doc?.source === "string"
            ? doc.source
            : typeof doc?.source_type === "string"
              ? doc.source_type
              : undefined,
        source_id: sourceId,
        source_type:
          typeof doc?.source_type === "string" ? doc.source_type : undefined,
        chunk_id: chunkId,
        evidence_origin: evidenceOrigin,
        source_status: sourceStatus,
        unavailable_reason: unavailableReason,
        url: typeof doc?.url === "string" ? doc.url : undefined,
        page_number:
          typeof doc?.page_number === "number" ? doc.page_number : undefined,
      },
    } as RagResult
  })
}

function deriveThreadHydrationState(messages: KnowledgeQAMessage[]): {
  query: string | null
  answer: string | null
  results: RagResult[]
  citations: CitationRef[]
  answerTrustState: KnowledgeAnswerTrustState
  answerPreview?: string
  sourcesCount: number
  settingsSnapshot?: Partial<RagSettings> | null
} | null {
  if (messages.length === 0) return null

  const latestUserMessage = [...messages]
    .reverse()
    .find((message) => message.role === "user")
  const assistantMessages = messages.filter((message) => message.role === "assistant")
  const latestAssistantWithContext = [...assistantMessages]
    .reverse()
    .find((message) => message.ragContext)
  const latestAssistantMessage =
    latestAssistantWithContext || assistantMessages[assistantMessages.length - 1]

  if (!latestAssistantMessage) {
    return {
      query: latestUserMessage?.content || null,
      answer: null,
      results: [],
      citations: [],
      answerTrustState: "unknown_trust",
      sourcesCount: 0,
      settingsSnapshot: null,
    }
  }

  const ragContext = latestAssistantMessage.ragContext
  const results = mapRagContextDocumentsToResults(ragContext?.retrieved_documents)
  const answer =
    normalizeAnswerText(ragContext?.generated_answer) ??
    normalizeAnswerText(latestAssistantMessage.content)
  const citations = answer ? parseCitations(answer, results) : []
  const storedTrustState = isKnowledgeAnswerTrustState(ragContext?.trust_state)
    ? ragContext.trust_state
    : null
  const answerTrustState =
    storedTrustState ??
    normalizeKnowledgeAnswerTrust({
      answer,
      results,
      citations,
      hasRequiredMetadata: false,
    }).state
  const queryFromContext =
    typeof ragContext?.search_query === "string" &&
    ragContext.search_query.trim().length > 0
      ? ragContext.search_query
      : null
  const query = latestUserMessage?.content || queryFromContext

  return {
    query,
    answer,
    results,
    citations,
    answerTrustState,
    answerPreview: truncateAnswerPreview(answer),
    sourcesCount: results.length,
    settingsSnapshot: normalizeRestorableSettingsSnapshot(ragContext?.settings_snapshot),
  }
}

function extractRagResponse(response: any): {
  results: RagResult[]
  answer: string | null
  expandedQueries: string[]
  metadata: Record<string, unknown>
} {
  const results: RagResult[] =
    response?.results || response?.documents || response?.docs || []
  const answer =
    normalizeAnswerText(response?.generated_answer) ??
    normalizeAnswerText(response?.answer) ??
    normalizeAnswerText(response?.response)
  const expandedQueries = Array.isArray(response?.expanded_queries)
    ? response.expanded_queries.filter(
        (value: unknown): value is string =>
          typeof value === "string" && value.trim().length > 0
      )
    : []
  const metadata =
    response?.metadata && typeof response.metadata === "object"
      ? ({ ...(response.metadata as Record<string, unknown>) } as Record<string, unknown>)
      : {}
  if (response?.faithfulness && typeof response.faithfulness === "object" && !metadata.faithfulness) {
    metadata.faithfulness = response.faithfulness
  }
  if (
    response?.verification_report &&
    typeof response.verification_report === "object" &&
    !metadata.verification_report
  ) {
    metadata.verification_report = response.verification_report
  }
  if (
    response?.retrieval_metrics &&
    typeof response.retrieval_metrics === "object" &&
    !metadata.retrieval_metrics
  ) {
    metadata.retrieval_metrics = response.retrieval_metrics
  }
  if (typeof response?.feedback_id === "string" && !metadata.feedback_id) {
    metadata.feedback_id = response.feedback_id
  }
  return { results, answer, expandedQueries, metadata }
}

function mapStreamingContextsToResults(contexts: any[]): RagResult[] {
  return contexts.map((context, index) => ({
    id:
      typeof context?.id === "string" && context.id.length > 0
        ? context.id
        : `stream-source-${index + 1}`,
    metadata: {
      title:
        typeof context?.title === "string" && context.title.length > 0
          ? context.title
          : `Source ${index + 1}`,
      source:
        typeof context?.source === "string" ? context.source : undefined,
      url: typeof context?.url === "string" ? context.url : undefined,
    },
    score: typeof context?.score === "number" ? context.score : undefined,
  }))
}

function calculateAverageRelevance(results: RagResult[]): number | null {
  const numericScores = results
    .map((result) => result.score)
    .filter((score): score is number => typeof score === "number")
  if (numericScores.length === 0) return null
  const sum = numericScores.reduce((acc, value) => acc + value, 0)
  return sum / numericScores.length
}

function hasWeakEvidence(results: RagResult[], threshold: number | undefined): boolean {
  const normalizedThreshold =
    typeof threshold === "number" && Number.isFinite(threshold) ? threshold : 0.3
  const scoredResults = results.filter(
    (result): result is RagResult & { score: number } =>
      typeof result.score === "number" && Number.isFinite(result.score)
  )
  if (scoredResults.length === 0) return false
  return scoredResults.every((result) => result.score < normalizedThreshold)
}

function normalizeMetric(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null
  return value
}

function extractTokenAndCostDetails(metadata: Record<string, unknown>): {
  tokensUsed: number | null
  estimatedCostUsd: number | null
} {
  const estimatedCostRaw = metadata?.estimated_cost
  const estimatedCostUsd =
    typeof estimatedCostRaw === "number"
      ? estimatedCostRaw
      : estimatedCostRaw &&
          typeof estimatedCostRaw === "object" &&
          typeof (estimatedCostRaw as Record<string, unknown>).total === "number"
        ? ((estimatedCostRaw as Record<string, unknown>).total as number)
        : null

  const tokenCandidates: unknown[] = [
    metadata?.tokens_used,
    metadata?.total_tokens,
    metadata?.token_count,
    metadata?.usage &&
    typeof metadata.usage === "object"
      ? (metadata.usage as Record<string, unknown>).total_tokens
      : undefined,
    estimatedCostRaw &&
    typeof estimatedCostRaw === "object"
      ? (estimatedCostRaw as Record<string, unknown>).input_tokens
      : undefined,
    estimatedCostRaw &&
    typeof estimatedCostRaw === "object"
      ? (estimatedCostRaw as Record<string, unknown>).output_tokens
      : undefined,
  ]

  const directTokenValue = tokenCandidates.find(
    (value) => typeof value === "number" && Number.isFinite(value)
  ) as number | undefined
  const estimatedCostTokens =
    estimatedCostRaw && typeof estimatedCostRaw === "object"
      ? (((estimatedCostRaw as Record<string, unknown>).input_tokens as number | undefined) ?? 0) +
        (((estimatedCostRaw as Record<string, unknown>).output_tokens as number | undefined) ?? 0)
      : null

  return {
    tokensUsed:
      typeof directTokenValue === "number"
        ? directTokenValue
        : typeof estimatedCostTokens === "number" && Number.isFinite(estimatedCostTokens) && estimatedCostTokens > 0
          ? estimatedCostTokens
          : null,
    estimatedCostUsd:
      typeof estimatedCostUsd === "number" && Number.isFinite(estimatedCostUsd)
        ? estimatedCostUsd
        : null,
  }
}

function normalizeIntegerMetric(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(0, Math.round(value))
  }
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) {
      return Math.max(0, Math.round(parsed))
    }
  }
  return null
}

function normalizeProbabilityMetric(value: unknown): number | null {
  let numericValue: number | null = null
  if (typeof value === "number" && Number.isFinite(value)) {
    numericValue = value
  } else if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) {
      numericValue = parsed
    }
  }

  if (numericValue == null) return null

  const normalized = numericValue > 1 && numericValue <= 100
    ? numericValue / 100
    : numericValue
  return Math.max(0, Math.min(1, normalized))
}

function extractFaithfulnessDetails(metadata: Record<string, unknown>): {
  faithfulnessScore: number | null
  totalClaims: number | null
  supportedClaims: number | null
  unsupportedClaims: number | null
} {
  const faithfulness =
    metadata?.faithfulness && typeof metadata.faithfulness === "object"
      ? (metadata.faithfulness as Record<string, unknown>)
      : null

  const scoreCandidate =
    typeof faithfulness?.faithfulness_score === "number" ||
    typeof faithfulness?.faithfulness_score === "string"
      ? faithfulness.faithfulness_score
      : typeof faithfulness?.score === "number" ||
          typeof faithfulness?.score === "string"
        ? faithfulness.score
        : null
  const faithfulnessScore = normalizeProbabilityMetric(scoreCandidate)

  const totalClaims = normalizeIntegerMetric(
    faithfulness?.total_claims ?? faithfulness?.claims_total
  )
  const supportedClaims = normalizeIntegerMetric(
    faithfulness?.supported_claims ?? faithfulness?.supported_count
  )
  const unsupportedClaims = normalizeIntegerMetric(
    faithfulness?.unsupported_claims ?? faithfulness?.unsupported_count
  )

  return {
    faithfulnessScore,
    totalClaims,
    supportedClaims,
    unsupportedClaims,
  }
}

function extractVerificationDetails(metadata: Record<string, unknown>): {
  verificationRate: number | null
  verificationCoverage: number | null
  totalClaims: number | null
  verifiedClaims: number | null
  reportAvailable: boolean
} {
  const verificationReport =
    metadata?.verification_report && typeof metadata.verification_report === "object"
      ? (metadata.verification_report as Record<string, unknown>)
      : null

  const verificationRateCandidate =
    typeof verificationReport?.verification_rate === "number" ||
    typeof verificationReport?.verification_rate === "string"
      ? verificationReport.verification_rate
      : typeof verificationReport?.precision === "number" ||
          typeof verificationReport?.precision === "string"
        ? verificationReport.precision
        : null
  const coverageCandidate =
    typeof verificationReport?.coverage === "number" ||
    typeof verificationReport?.coverage === "string"
      ? verificationReport.coverage
      : null
  const verificationRate = normalizeProbabilityMetric(verificationRateCandidate)
  const verificationCoverage = normalizeProbabilityMetric(coverageCandidate)

  const totalClaims = normalizeIntegerMetric(
    verificationReport?.total_claims ?? verificationReport?.claims_total
  )
  const verifiedClaims = normalizeIntegerMetric(
    verificationReport?.verified_count ?? verificationReport?.supported_count
  )

  return {
    verificationRate,
    verificationCoverage,
    totalClaims,
    verifiedClaims,
    reportAvailable:
      Boolean(verificationReport) &&
      Object.keys(verificationReport as Record<string, unknown>).length > 0,
  }
}

function normalizeAlsoConsideredCandidate(
  value: unknown,
  fallbackIndex: number
): {
  id: string
  title: string
  score: number | null
  reason: string | null
} | null {
  if (!value || typeof value !== "object") return null
  const candidate = value as Record<string, unknown>
  const idRaw = candidate.id ?? candidate.doc_id ?? candidate.document_id
  const id =
    typeof idRaw === "string" && idRaw.trim().length > 0
      ? idRaw.trim()
      : `candidate-${fallbackIndex + 1}`
  const titleRaw =
    candidate.title ?? candidate.name ?? candidate.source ?? candidate.document_title
  const title =
    typeof titleRaw === "string" && titleRaw.trim().length > 0
      ? titleRaw.trim()
      : `Candidate ${fallbackIndex + 1}`
  const score =
    normalizeProbabilityMetric(candidate.score ?? candidate.relevance)
  const reasonRaw = candidate.reason ?? candidate.exclusion_reason
  const reason =
    typeof reasonRaw === "string" && reasonRaw.trim().length > 0
      ? reasonRaw.trim()
      : null
  return { id, title, score, reason }
}

function extractCandidateTransparency(
  metadata: Record<string, unknown>,
  returnedCount: number
): {
  candidatesConsidered: number | null
  candidatesRejected: number | null
  alsoConsidered: Array<{
    id: string
    title: string
    score: number | null
    reason: string | null
  }>
} {
  const documentGrading =
    metadata?.document_grading && typeof metadata.document_grading === "object"
      ? (metadata.document_grading as Record<string, unknown>)
      : null
  const retrievalMetrics =
    metadata?.retrieval_metrics && typeof metadata.retrieval_metrics === "object"
      ? (metadata.retrieval_metrics as Record<string, unknown>)
      : null

  const candidatesConsidered = (
    [
      metadata?.candidates_considered,
      metadata?.candidate_count,
      metadata?.total_candidates,
      metadata?.documents_retrieved,
      documentGrading?.total_graded,
      retrievalMetrics?.candidates_considered,
      retrievalMetrics?.total_candidates,
      retrievalMetrics?.documents_retrieved,
      retrievalMetrics?.documents_considered,
      retrievalMetrics?.chunks_considered,
    ] as unknown[]
  )
    .map((value) => normalizeIntegerMetric(value))
    .find((value): value is number => value != null) ?? null

  const removedByGrading = normalizeIntegerMetric(documentGrading?.removed_count)
  const computedRejected =
    typeof candidatesConsidered === "number" && candidatesConsidered >= returnedCount
      ? candidatesConsidered - returnedCount
      : removedByGrading
  const candidatesRejected =
    typeof computedRejected === "number" && Number.isFinite(computedRejected)
      ? Math.max(0, computedRejected)
      : null

  const rawAlsoConsidered =
    (Array.isArray(metadata?.also_considered)
      ? metadata.also_considered
      : Array.isArray(metadata?.candidates_below_threshold)
        ? metadata.candidates_below_threshold
        : Array.isArray(metadata?.excluded_candidates)
          ? metadata.excluded_candidates
          : Array.isArray(retrievalMetrics?.also_considered)
            ? retrievalMetrics?.also_considered
            : Array.isArray(retrievalMetrics?.excluded_candidates)
              ? retrievalMetrics?.excluded_candidates
          : []) as unknown[]
  const alsoConsidered = rawAlsoConsidered
    .map((candidate, index) => normalizeAlsoConsideredCandidate(candidate, index))
    .filter(
      (
        candidate
      ): candidate is {
        id: string
        title: string
        score: number | null
        reason: string | null
      } => candidate != null
    )

  const derivedCandidatesConsidered =
    candidatesConsidered != null
      ? candidatesConsidered
      : alsoConsidered.length > 0
        ? returnedCount + alsoConsidered.length
        : null

  return {
    candidatesConsidered: derivedCandidatesConsidered,
    candidatesRejected:
      derivedCandidatesConsidered != null && derivedCandidatesConsidered >= returnedCount
        ? derivedCandidatesConsidered - returnedCount
        : candidatesRejected,
    alsoConsidered,
  }
}

function extractRetrievalCoverage(
  metadata: Record<string, unknown>,
  returnedCount: number
): {
  retrievalLatencyMs: number | null
  documentsConsidered: number | null
  chunksConsidered: number | null
  documentsReturned: number
} {
  const retrievalMetrics =
    metadata?.retrieval_metrics && typeof metadata.retrieval_metrics === "object"
      ? (metadata.retrieval_metrics as Record<string, unknown>)
      : null

  const retrievalLatencyMs = (
    [
      retrievalMetrics?.retrieval_latency_ms,
      retrievalMetrics?.latency_ms,
      retrievalMetrics?.retrieval_ms,
      metadata?.retrieval_latency_ms,
      metadata?.latency_ms,
      metadata?.retrieval_ms,
    ] as unknown[]
  )
    .map((value) => normalizeIntegerMetric(value))
    .find((value): value is number => value != null) ?? null

  const documentsConsidered = (
    [
      retrievalMetrics?.documents_considered,
      retrievalMetrics?.documents_retrieved,
      retrievalMetrics?.total_candidates,
      metadata?.documents_considered,
      metadata?.documents_retrieved,
      metadata?.candidate_count,
      metadata?.candidates_considered,
      metadata?.total_candidates,
    ] as unknown[]
  )
    .map((value) => normalizeIntegerMetric(value))
    .find((value): value is number => value != null) ?? null

  const chunksConsidered = (
    [
      retrievalMetrics?.chunks_considered,
      retrievalMetrics?.chunks_scanned,
      retrievalMetrics?.total_chunks_scanned,
      retrievalMetrics?.chunks_examined,
      metadata?.chunks_considered,
      metadata?.chunks_scanned,
      metadata?.total_chunks_scanned,
      metadata?.chunks_examined,
    ] as unknown[]
  )
    .map((value) => normalizeIntegerMetric(value))
    .find((value): value is number => value != null) ?? null

  return {
    retrievalLatencyMs,
    documentsConsidered,
    chunksConsidered,
    documentsReturned: returnedCount,
  }
}

function normalizeSourceStatus(value: unknown): Record<string, KnowledgeSourceStatus> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {}
  }

  const normalized: Record<string, KnowledgeSourceStatus> = {}
  for (const [sourceId, rawStatus] of Object.entries(value as Record<string, unknown>)) {
    if (!sourceId || !rawStatus || typeof rawStatus !== "object" || Array.isArray(rawStatus)) {
      continue
    }

    const record = rawStatus as Record<string, unknown>
    const status =
      typeof record.status === "string" && record.status.trim().length > 0
        ? record.status.trim()
        : "searched"
    const count = normalizeIntegerMetric(record.count) ?? 0
    const reason =
      typeof record.reason === "string" && record.reason.trim().length > 0
        ? record.reason.trim()
        : null

    normalized[sourceId] = reason
      ? { status, count, reason }
      : { status, count }
  }

  return normalized
}

function buildSearchDetailsFromResponse(
  response: {
    expandedQueries: string[]
    metadata: Record<string, unknown>
  },
  results: RagResult[],
  settings: RagSettings
): SearchRuntimeDetails {
  const webFallbackMetadata =
    response.metadata?.web_fallback &&
    typeof response.metadata.web_fallback === "object"
      ? (response.metadata.web_fallback as Record<string, unknown>)
      : null
  const whyTheseSourcesMetadata =
    response.metadata?.why_these_sources &&
    typeof response.metadata.why_these_sources === "object"
      ? (response.metadata.why_these_sources as Record<string, unknown>)
      : null
  const { tokensUsed, estimatedCostUsd } = extractTokenAndCostDetails(
    response.metadata
  )
  const faithfulness = extractFaithfulnessDetails(response.metadata)
  const verification = extractVerificationDetails(response.metadata)
  const candidateTransparency = extractCandidateTransparency(
    response.metadata,
    results.length
  )
  const retrievalCoverage = extractRetrievalCoverage(
    response.metadata,
    results.length
  )
  const sourceStatus = normalizeSourceStatus(response.metadata?.source_status)

  return {
    expandedQueries: response.expandedQueries,
    rerankingEnabled: Boolean(settings.enable_reranking),
    rerankingStrategy:
      typeof settings.reranking_strategy === "string"
        ? settings.reranking_strategy
        : "unknown",
    averageRelevance: calculateAverageRelevance(results),
    sourceStatus,
    webFallbackEnabled: Boolean(settings.enable_web_fallback),
    webFallbackTriggered:
      typeof webFallbackMetadata?.triggered === "boolean"
        ? webFallbackMetadata.triggered
        : false,
    webFallbackEngine:
      typeof webFallbackMetadata?.engine_used === "string"
        ? webFallbackMetadata.engine_used
        : null,
    tokensUsed,
    estimatedCostUsd,
    feedbackId:
      typeof response.metadata?.feedback_id === "string"
        ? response.metadata.feedback_id
        : null,
    whyTheseSources: whyTheseSourcesMetadata
      ? {
          topicality: normalizeMetric(whyTheseSourcesMetadata.topicality),
          diversity: normalizeMetric(whyTheseSourcesMetadata.diversity),
          freshness: normalizeMetric(whyTheseSourcesMetadata.freshness),
        }
      : null,
    faithfulnessScore: faithfulness.faithfulnessScore,
    faithfulnessTotalClaims: faithfulness.totalClaims,
    faithfulnessSupportedClaims: faithfulness.supportedClaims,
    faithfulnessUnsupportedClaims: faithfulness.unsupportedClaims,
    verificationRate: verification.verificationRate,
    verificationCoverage: verification.verificationCoverage,
    verificationTotalClaims: verification.totalClaims,
    verificationVerifiedClaims: verification.verifiedClaims,
    verificationReportAvailable: verification.reportAvailable,
    retrievalLatencyMs: retrievalCoverage.retrievalLatencyMs,
    documentsConsidered:
      retrievalCoverage.documentsConsidered ?? candidateTransparency.candidatesConsidered,
    chunksConsidered: retrievalCoverage.chunksConsidered,
    documentsReturned: retrievalCoverage.documentsReturned,
    candidatesConsidered: candidateTransparency.candidatesConsidered,
    candidatesReturned: results.length,
    candidatesRejected: candidateTransparency.candidatesRejected,
    alsoConsidered: candidateTransparency.alsoConsidered,
  }
}

function buildSearchDetailsFromStreaming(
  results: RagResult[],
  whyPayload: unknown,
  sourceStatusPayload: unknown,
  settings: RagSettings
): SearchRuntimeDetails {
  const why =
    whyPayload && typeof whyPayload === "object"
      ? (whyPayload as Record<string, unknown>)
      : null

  return {
    expandedQueries: [],
    rerankingEnabled: Boolean(settings.enable_reranking),
    rerankingStrategy:
      typeof settings.reranking_strategy === "string"
        ? settings.reranking_strategy
        : "unknown",
    averageRelevance: calculateAverageRelevance(results),
    sourceStatus: normalizeSourceStatus(sourceStatusPayload),
    webFallbackEnabled: Boolean(settings.enable_web_fallback),
    webFallbackTriggered: false,
    webFallbackEngine: null,
    tokensUsed: null,
    estimatedCostUsd: null,
    feedbackId: null,
    whyTheseSources: why
      ? {
          topicality: normalizeMetric(why.topicality),
          diversity: normalizeMetric(why.diversity),
          freshness: normalizeMetric(why.freshness),
        }
      : null,
    faithfulnessScore: null,
    faithfulnessTotalClaims: null,
    faithfulnessSupportedClaims: null,
    faithfulnessUnsupportedClaims: null,
    verificationRate: null,
    verificationCoverage: null,
    verificationTotalClaims: null,
    verificationVerifiedClaims: null,
    verificationReportAvailable: false,
    retrievalLatencyMs: null,
    documentsConsidered: null,
    chunksConsidered: null,
    documentsReturned: results.length,
    candidatesConsidered: null,
    candidatesReturned: results.length,
    candidatesRejected: null,
    alsoConsidered: [],
  }
}

function buildScopeSnapshot(
  preset: RagPresetName,
  settings: RagSettings
): ScopeSnapshot {
  return {
    preset,
    sources: [...settings.sources],
    webFallback: Boolean(settings.enable_web_fallback),
    collectionId:
      typeof settings.collection_id === "number" &&
      Number.isInteger(settings.collection_id) &&
      settings.collection_id > 0
        ? settings.collection_id
        : null,
    includeMediaIds: Array.isArray(settings.include_media_ids)
      ? settings.include_media_ids
          .filter((value): value is number => typeof value === "number" && Number.isFinite(value))
          .map((value) => Math.round(value))
      : [],
    includeNoteIds: Array.isArray(settings.include_note_ids)
      ? settings.include_note_ids
          .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
          .map((value) => value.trim())
      : [],
  }
}

function createRestorableSettingsSnapshot(settings: RagSettings): Partial<RagSettings> {
  return {
    sources: Array.isArray(settings.sources) ? [...settings.sources] : [],
    collection_id:
      typeof settings.collection_id === "number" &&
      Number.isInteger(settings.collection_id) &&
      settings.collection_id > 0
        ? settings.collection_id
        : undefined,
    include_media_ids: Array.isArray(settings.include_media_ids)
      ? settings.include_media_ids
          .filter((value): value is number => typeof value === "number" && Number.isFinite(value))
          .map((value) => Math.round(value))
      : [],
    include_note_ids: Array.isArray(settings.include_note_ids)
      ? settings.include_note_ids
          .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
          .map((value) => value.trim())
      : [],
    top_k: typeof settings.top_k === "number" && Number.isFinite(settings.top_k)
      ? settings.top_k
      : undefined,
    enable_reranking:
      typeof settings.enable_reranking === "boolean" ? settings.enable_reranking : undefined,
    enable_citations:
      typeof settings.enable_citations === "boolean" ? settings.enable_citations : undefined,
    enable_web_fallback:
      typeof settings.enable_web_fallback === "boolean" ? settings.enable_web_fallback : undefined,
    web_fallback_threshold:
      typeof settings.web_fallback_threshold === "number" &&
      Number.isFinite(settings.web_fallback_threshold)
        ? settings.web_fallback_threshold
        : undefined,
    web_search_engine:
      typeof settings.web_search_engine === "string" ? settings.web_search_engine : undefined,
    web_fallback_result_count:
      typeof settings.web_fallback_result_count === "number" &&
      Number.isFinite(settings.web_fallback_result_count)
        ? settings.web_fallback_result_count
        : undefined,
    web_fallback_merge_strategy:
      typeof settings.web_fallback_merge_strategy === "string"
        ? settings.web_fallback_merge_strategy
        : undefined,
    search_mode:
      typeof settings.search_mode === "string" ? settings.search_mode : undefined,
  }
}

function normalizeRestorableSettingsSnapshot(
  snapshot: unknown
): Partial<RagSettings> | null {
  if (!snapshot || typeof snapshot !== "object") return null

  const candidate = snapshot as Record<string, unknown>
  const normalized: Partial<RagSettings> = {}

  if (Array.isArray(candidate.sources)) {
    normalized.sources = candidate.sources.filter(
      (value): value is RagSettings["sources"][number] =>
        typeof value === "string" && value.trim().length > 0
    )
  }

  if (Array.isArray(candidate.include_media_ids)) {
    normalized.include_media_ids = candidate.include_media_ids
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value))
      .map((value) => Math.round(value))
  }

  if (
    typeof candidate.collection_id === "number" &&
    Number.isInteger(candidate.collection_id) &&
    candidate.collection_id > 0
  ) {
    normalized.collection_id = candidate.collection_id
  }

  if (Array.isArray(candidate.include_note_ids)) {
    normalized.include_note_ids = candidate.include_note_ids
      .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
      .map((value) => value.trim())
  }

  if (typeof candidate.top_k === "number" && Number.isFinite(candidate.top_k)) {
    normalized.top_k = candidate.top_k
  }
  if (typeof candidate.enable_reranking === "boolean") {
    normalized.enable_reranking = candidate.enable_reranking
  }
  if (typeof candidate.enable_citations === "boolean") {
    normalized.enable_citations = candidate.enable_citations
  }
  if (typeof candidate.enable_web_fallback === "boolean") {
    normalized.enable_web_fallback = candidate.enable_web_fallback
  }
  if (
    typeof candidate.web_fallback_threshold === "number" &&
    Number.isFinite(candidate.web_fallback_threshold)
  ) {
    normalized.web_fallback_threshold = candidate.web_fallback_threshold
  }
  if (typeof candidate.web_search_engine === "string") {
    normalized.web_search_engine = candidate.web_search_engine
  }
  if (
    typeof candidate.web_fallback_result_count === "number" &&
    Number.isFinite(candidate.web_fallback_result_count)
  ) {
    normalized.web_fallback_result_count = candidate.web_fallback_result_count
  }
  if (
    typeof candidate.web_fallback_merge_strategy === "string" &&
    isValidWebFallbackMergeStrategy(candidate.web_fallback_merge_strategy)
  ) {
    normalized.web_fallback_merge_strategy = candidate.web_fallback_merge_strategy
  }
  if (
    typeof candidate.search_mode === "string" &&
    isValidRagSearchMode(candidate.search_mode)
  ) {
    normalized.search_mode = candidate.search_mode
  }

  return Object.keys(normalized).length > 0 ? normalized : null
}

function resolveHydratedScopeState(
  currentPreset: RagPresetName,
  currentSettings: RagSettings,
  payload: {
    preset?: RagPresetName
    settingsSnapshot?: Partial<RagSettings> | null
  }
): { nextPreset: RagPresetName; nextSettings: RagSettings } {
  const nextPreset = payload.preset ?? currentPreset
  const presetSettings =
    payload.preset && payload.preset !== "custom"
      ? {
          ...applyRagPreset(payload.preset as Exclude<RagPresetName, "custom">),
          ...KNOWLEDGE_QA_SETTINGS_OVERRIDES,
          enable_web_fallback: currentSettings.enable_web_fallback,
        }
      : currentSettings
  const nextSettings = payload.settingsSnapshot
    ? {
        ...presetSettings,
        ...payload.settingsSnapshot,
      }
    : presetSettings

  return { nextPreset, nextSettings }
}

function extractHttpStatus(error: unknown): number | null {
  if (error && typeof error === "object") {
    const candidate = error as {
      status?: unknown
      response?: { status?: unknown }
      message?: unknown
    }
    if (typeof candidate.status === "number") {
      return candidate.status
    }
    if (typeof candidate.response?.status === "number") {
      return candidate.response.status
    }
    if (typeof candidate.message === "string") {
      const match = candidate.message.match(/\bHTTP\s+(\d{3})\b/i)
      return match ? Number(match[1]) : null
    }
  }

  if (error instanceof Error) {
    const match = error.message.match(/\bHTTP\s+(\d{3})\b/i)
    return match ? Number(match[1]) : null
  }

  return null
}

function classifyThreadHydrationFailure(error: unknown): ThreadHydrationResult {
  const status = extractHttpStatus(error)
  if (typeof status === "number" && status >= 400 && status < 500 && status !== 429) {
    return "terminal"
  }
  return false
}

function mergeNumberFilters(
  ...values: Array<Array<number | null | undefined> | undefined>
): number[] {
  const merged = new Set<number>()
  for (const group of values) {
    if (!Array.isArray(group)) continue
    for (const candidate of group) {
      if (typeof candidate !== "number" || !Number.isFinite(candidate)) continue
      merged.add(Math.round(candidate))
    }
  }
  return Array.from(merged)
}

function mergeStringFilters(
  ...values: Array<Array<string | null | undefined> | undefined>
): string[] {
  const merged = new Set<string>()
  for (const group of values) {
    if (!Array.isArray(group)) continue
    for (const candidate of group) {
      if (typeof candidate !== "string") continue
      const normalized = candidate.trim()
      if (!normalized) continue
      merged.add(normalized)
    }
  }
  return Array.from(merged)
}

// Provider component
export function KnowledgeQAProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  const [historyHydrated, setHistoryHydrated] = useState(false)
  const [storedPreset] = useStorage<RagPresetName>("ragSearchPreset", "balanced")
  const [storedSettings] = useStorage<RagSettings>(
    "ragSearchSettingsV2",
    DEFAULT_RAG_SETTINGS
  )
  const [streamingFeatureFlag] = useStorage<boolean>("ff_knowledgeQaStreaming", true)
  const hydratedDefaultsRef = useRef<string | null>(null)
  const activeSearchAbortRef = useRef<AbortController | null>(null)
  const activeSearchRequestIdRef = useRef(0)
  const activeThreadHydrationRequestIdRef = useRef(0)
  const activeHistoryLoadRequestIdRef = useRef(0)
  const sourceHealthRequestIdRef = useRef(0)
  const historyMutationVersionRef = useRef(0)
  const focusedSourceTimeoutRef = useRef<number | null>(null)
  const persistenceWarningShownRef = useRef(false)
  const historyQuotaWarningShownRef = useRef(false)
  const message = useAntdMessage()
  const streamingFeatureEnabled = streamingFeatureFlag !== false
  const defaultCharacterIdRef = useRef<number | null>(null)
  const defaultCharacterPromiseRef = useRef<Promise<number | null> | null>(null)

  useEffect(() => {
    if (state.currentThreadId || state.messages.length > 0) {
      hydratedDefaultsRef.current = null
      return
    }

    const normalizedSettings: RagSettings = {
      ...DEFAULT_RAG_SETTINGS,
      ...(storedSettings || {}),
      ...KNOWLEDGE_QA_SETTINGS_OVERRIDES,
      include_note_ids: normalizeNoteFilterIds(storedSettings?.include_note_ids),
      enable_web_fallback:
        typeof storedSettings?.enable_web_fallback === "boolean"
          ? storedSettings.enable_web_fallback
          : true,
    }
    const serialized = JSON.stringify({
      preset: storedPreset,
      settings: normalizedSettings,
    })
    if (serialized === hydratedDefaultsRef.current) {
      return
    }
    hydratedDefaultsRef.current = serialized

    dispatch({
      type: "HYDRATE_DEFAULTS",
      payload: {
        preset: storedPreset,
        settings: normalizedSettings,
      },
    })
  }, [
    state.currentThreadId,
    state.messages.length,
    storedPreset,
    storedSettings,
  ])

  const resolveDefaultCharacterId = useCallback(async (): Promise<number | null> => {
    if (defaultCharacterIdRef.current != null) {
      return defaultCharacterIdRef.current
    }
    if (defaultCharacterPromiseRef.current) {
      return await defaultCharacterPromiseRef.current
    }

    const resolver = (async () => {
      try {
        await tldwClient.initialize()
        const searchResults = await tldwClient
          .searchCharacters(DEFAULT_CHARACTER_NAME, { limit: 5 })
          .catch(() => [])
        const match =
          searchResults.find(
            (c: any) =>
              typeof c?.name === "string" &&
              c.name.toLowerCase() === DEFAULT_CHARACTER_NAME.toLowerCase()
          ) || searchResults[0]
        if (match && typeof match.id === "number") {
          defaultCharacterIdRef.current = match.id
          return match.id
        }

        const listResults = await tldwClient.listCharacters({ limit: 10 }).catch(() => [])
        const fallback =
          listResults.find(
            (c: any) =>
              typeof c?.name === "string" &&
              c.name.toLowerCase() === DEFAULT_CHARACTER_NAME.toLowerCase()
          ) || listResults[0]
        if (fallback && typeof fallback.id === "number") {
          defaultCharacterIdRef.current = fallback.id
          return fallback.id
        }
      } catch (error) {
        console.warn("Failed to resolve default character:", error)
      }
      return null
    })()

    defaultCharacterPromiseRef.current = resolver
    try {
      return await resolver
    } finally {
      defaultCharacterPromiseRef.current = null
    }
  }, [])

  const tagConversationKeyword = useCallback(
    async (conversationId: string, version?: number | null): Promise<void> => {
      if (!conversationId || version == null) return
      try {
        const conversationResponse = await tldwClient.fetchWithAuth(
          `/api/v1/chat/conversations/${conversationId}`
        )
        const conversationData = await conversationResponse.json()
        const currentKeywords = Array.isArray(conversationData?.keywords)
          ? conversationData.keywords
          : Array.isArray(conversationData?.conversation?.keywords)
            ? conversationData.conversation.keywords
            : []
        const mergedKeywords: string[] = []
        const seenKeywords = new Set<string>()
        for (const keyword of [...currentKeywords, KNOWLEDGE_QA_KEYWORD]) {
          const normalized = String(keyword ?? "").trim()
          if (!normalized) continue
          const normalizedKey = normalized.toLowerCase()
          if (seenKeywords.has(normalizedKey)) continue
          seenKeywords.add(normalizedKey)
          mergedKeywords.push(normalized)
        }

        await tldwClient.fetchWithAuth(`/api/v1/chat/conversations/${conversationId}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            version,
            keywords: mergedKeywords,
          }),
        })
      } catch (error) {
        console.warn("Failed to tag Knowledge QA conversation:", error)
      }
    },
    []
  )

  // Actions
  const setQuery = useCallback((query: string) => {
    dispatch({ type: "SET_QUERY", payload: query })
  }, [])

  const createNewThread = useCallback(
    async (title?: string, options?: CreateThreadOptions): Promise<string> => {
      const shouldActivate = options?.shouldActivate ?? (() => true)
      const cleanupRemoteIfSkipped = options?.cleanupRemoteIfSkipped ?? false
      try {
        const characterId = await resolveDefaultCharacterId()
        if (!characterId) {
          throw new Error("Default character unavailable")
        }

        const resolvedTitle = title?.trim()
          ? title.trim()
          : `Knowledge QA - ${new Date().toLocaleDateString()}`

        // Create a new conversation via API
        const response = await tldwClient.createChat({
          character_id: characterId,
          title: resolvedTitle,
          state: "in-progress",
          source: "knowledge_qa",
          ...(typeof options?.parentConversationId === "string" &&
          options.parentConversationId.trim().length > 0
            ? { parent_conversation_id: options.parentConversationId.trim() }
            : {}),
          ...(typeof options?.forkedFromMessageId === "string" &&
          options.forkedFromMessageId.trim().length > 0
            ? { forked_from_message_id: options.forkedFromMessageId.trim() }
            : {}),
        })

        const threadId = response?.id
        if (!threadId) {
          throw new Error("Chat creation returned no ID")
        }

        if (!shouldActivate()) {
          if (cleanupRemoteIfSkipped) {
            try {
              await tldwClient.deleteChat(String(threadId))
            } catch (cleanupError) {
              console.warn("Failed to delete skipped Knowledge QA conversation:", cleanupError)
            }
          }
          return threadId
        }

        const version =
          typeof response?.version === "number" ? response.version : null
        if (version != null) {
          await tagConversationKeyword(String(threadId), version)
        } else {
          try {
            const current = await tldwClient.getChat(String(threadId))
            const currentVersion =
              typeof current?.version === "number" ? current.version : null
            if (currentVersion != null) {
              await tagConversationKeyword(String(threadId), currentVersion)
            }
          } catch (error) {
            console.warn("Failed to fetch conversation version for tagging:", error)
          }
        }
        const normalizedState: "in-progress" | "resolved" =
          response?.state === "resolved" ? "resolved" : "in-progress"
        const newThread: KnowledgeQAThread = {
          id: threadId,
          title: response?.title || resolvedTitle || "New Knowledge QA Thread",
          createdAt: new Date().toISOString(),
          lastModifiedAt: new Date().toISOString(),
          state: normalizedState,
          messageCount: 0,
          source: "knowledge_qa",
        }

        dispatch({ type: "ADD_THREAD", payload: newThread })
        dispatch({ type: "SET_THREAD_ID", payload: threadId })
        dispatch({ type: "SET_LOCAL_ONLY_THREAD", payload: false })
        dispatch({ type: "SET_MESSAGES", payload: [] })
        dispatch({
          type: "SET_PINNED_SOURCE_FILTERS",
          payload: { mediaIds: [], noteIds: [] },
        })

        return threadId
      } catch (error) {
        console.error("Failed to create thread:", error)
        // Return a local ID as fallback
        const localId = `${LOCAL_THREAD_PREFIX}${crypto.randomUUID()}`
        if (!shouldActivate()) {
          return localId
        }
        dispatch({ type: "SET_THREAD_ID", payload: localId })
        dispatch({ type: "SET_LOCAL_ONLY_THREAD", payload: true })
        dispatch({ type: "SET_MESSAGES", payload: [] })
        dispatch({
          type: "SET_PINNED_SOURCE_FILTERS",
          payload: { mediaIds: [], noteIds: [] },
        })
        return localId
      }
    },
    [resolveDefaultCharacterId, tagConversationKeyword]
  )

  const notifyPersistenceFailure = useCallback(() => {
    if (persistenceWarningShownRef.current) {
      return
    }
    persistenceWarningShownRef.current = true
    message.open({
      type: "warning",
      content: "Unable to save conversation. Results are available but may not persist.",
      duration: 4,
    })
  }, [message])

  const beginThreadHydrationRequest = useCallback(() => {
    const nextRequestId = activeThreadHydrationRequestIdRef.current + 1
    activeThreadHydrationRequestIdRef.current = nextRequestId
    return nextRequestId
  }, [])

  const beginHistoryLoadRequest = useCallback(() => {
    const nextRequestId = activeHistoryLoadRequestIdRef.current + 1
    activeHistoryLoadRequestIdRef.current = nextRequestId
    return {
      requestId: nextRequestId,
      mutationVersion: historyMutationVersionRef.current,
    }
  }, [])

  const markHistoryMutation = useCallback(() => {
    historyMutationVersionRef.current += 1
  }, [])

  const persistChatMessage = useCallback(
    async (
      threadId: string,
      role: "user" | "assistant" | "system",
      content: string,
      parentMessageId?: string | null
    ): Promise<{ id: string; timestamp?: string } | null> => {
      if (!threadId || isLocalThreadId(threadId)) return null
      try {
        const payload: Record<string, any> = { role, content }
        if (parentMessageId) {
          payload.parent_message_id = parentMessageId
        }
        const response = await tldwClient.addChatMessage(threadId, payload)
        const id = response?.id != null ? String(response.id) : null
        if (!id) return null
        return {
          id,
          timestamp: response?.created_at ? String(response.created_at) : undefined,
        }
      } catch (error) {
        console.warn("Failed to persist chat message:", error)
        notifyPersistenceFailure()
        return null
      }
    },
    [notifyPersistenceFailure]
  )

  const persistRagContext = useCallback(
    async (messageId: string, context: RagContextData): Promise<boolean> => {
      try {
        const response = await tldwClient.fetchWithAuth(
          `/api/v1/chat/messages/${messageId}/rag-context`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message_id: messageId,
              rag_context: context,
            }),
          }
        )
        if (!response.ok) {
          let errorDetails: string | unknown = ""
          try {
            errorDetails = await response.text()
          } catch (textError) {
            errorDetails = textError
          }
          console.error("Failed to persist RAG context:", {
            status: response.status,
            error: errorDetails,
          })
          return false
        }
        const result = await response.json()
        return result?.success ?? false
      } catch (error) {
        // Metadata-only failure path: answers/sources remain usable even if context persistence fails.
        console.error("Failed to persist RAG context:", error)
        return false
      }
    },
    []
  )

  const buildRagContext = useCallback(
    (
      question: string,
      results: RagResult[],
      answer: string | null,
      settings: RagSettings,
      answerTrustState: KnowledgeAnswerTrustState
    ): RagContextData => ({
      search_query: question,
      search_mode: settings.search_mode,
      settings_snapshot: createRestorableSettingsSnapshot(settings),
      retrieved_documents: results.map((r) => ({
        id: r.id,
        source_id: getResultSourceId(r) ?? undefined,
        source_type: r.sourceType || r.metadata?.source_type,
        title: r.metadata?.title,
        score: r.score,
        chunk_id: getResultChunkId(r) ?? undefined,
        excerpt: getResultEvidenceText(r) || undefined,
        evidence_origin: getEvidenceOrigin(r),
        source_status: getResultSourceStatus(r) ?? undefined,
        unavailable_reason: getResultUnavailableReason(r) ?? undefined,
        url: r.metadata?.url,
        page_number: r.metadata?.page_number,
      })),
      generated_answer: answer || undefined,
      trust_state: answerTrustState,
      timestamp: new Date().toISOString(),
    }),
    []
  )

  const runKnowledgeQuery = useCallback(
    async (
      question: string,
      addToHistory: boolean,
      settingsOverrides?: Partial<RagSettings>
    ) => {
      const trimmedQuery = question.trim()
      if (!trimmedQuery) return
      const queryWasTruncated = trimmedQuery.length > RAG_QUERY_MAX_LENGTH
      dispatch({
        type: "SET_QUERY_WARNING",
        payload: queryWasTruncated
          ? "Query exceeded 20,000 characters and was shortened before search."
          : null,
      })

      const searchStartedAt = Date.now()
      if (activeSearchAbortRef.current) {
        activeSearchAbortRef.current.abort("superseded")
      }
      beginThreadHydrationRequest()
      const abortController = new AbortController()
      const searchRequestId = activeSearchRequestIdRef.current + 1
      activeSearchRequestIdRef.current = searchRequestId
      activeSearchAbortRef.current = abortController
      const isStaleSearchRequest = () =>
        activeSearchRequestIdRef.current !== searchRequestId ||
        activeSearchAbortRef.current !== abortController
      dispatch({ type: "SET_SEARCHING", payload: true })
      dispatch({ type: "SET_QUERY_STAGE", payload: "searching" })
      dispatch({ type: "SET_SEARCH_DETAILS", payload: null })
      const overrideSettings = settingsOverrides || {}
      const effectiveSettings: RagSettings = {
        ...state.settings,
        ...overrideSettings,
        include_media_ids: mergeNumberFilters(
          state.settings.include_media_ids,
          overrideSettings.include_media_ids,
          state.pinnedSourceFilters.mediaIds
        ),
        include_note_ids: mergeStringFilters(
          state.settings.include_note_ids,
          overrideSettings.include_note_ids,
          state.pinnedSourceFilters.noteIds
        ),
      }

      let threadId = state.currentThreadId
      if (!threadId) {
        threadId = await createNewThread(trimmedQuery, {
          shouldActivate: () => !isStaleSearchRequest(),
          cleanupRemoteIfSkipped: true,
        })
        if (isStaleSearchRequest()) {
          return
        }
      }
      const isThreadSyncFailed = () => Boolean(threadId && isLocalThreadId(threadId))

      const userTimestamp = new Date().toISOString()
      const persistedUser = threadId
        ? await persistChatMessage(threadId, "user", trimmedQuery, null)
        : null
      if (isStaleSearchRequest()) {
        return
      }
      const userMessageId = persistedUser?.id || crypto.randomUUID()

      if (threadId) {
        const userMessage: KnowledgeQAMessage = {
          id: userMessageId,
          conversationId: threadId,
          role: "user",
          content: trimmedQuery,
          timestamp: persistedUser?.timestamp || userTimestamp,
        }
        dispatch({ type: "ADD_MESSAGE", payload: userMessage })
      }

      try {
        const { options } = buildRagSearchRequest({
          ...effectiveSettings,
          query: trimmedQuery,
          enable_web_fallback: effectiveSettings.enable_web_fallback,
        })

        const canAttemptStreaming =
          streamingFeatureEnabled &&
          effectiveSettings.enable_generation &&
          typeof (tldwClient as {
            ragSearchStream?: (
              query: string,
              options?: Record<string, unknown>
            ) => AsyncGenerator<any, void, unknown>
          }).ragSearchStream === "function"

        let results: RagResult[] = []
        let answer: string | null = null
        let usedStreaming = false
        let resolvedSearchDetails: SearchRuntimeDetails | null = null

        if (
          canAttemptStreaming &&
          typeof (tldwClient as {
            ragSearchStream?: (
              query: string,
              options?: Record<string, unknown>
            ) => AsyncGenerator<any, void, unknown>
          }).ragSearchStream === "function"
        ) {
          let streamResults: RagResult[] = []
          let streamAnswer = ""
          let receivedStreamEvent = false
          let streamWhyPayload: unknown = null
          let streamSourceStatusPayload: unknown = null

          try {
            for await (const event of (tldwClient as {
              ragSearchStream: (
                query: string,
                options?: Record<string, unknown>
              ) => AsyncGenerator<any, void, unknown>
            }).ragSearchStream(trimmedQuery, {
              ...options,
              signal: abortController.signal,
            })) {
              if (isStaleSearchRequest()) {
                return
              }
              const eventType = typeof event?.type === "string" ? event.type : ""

              if (eventType === "contexts" && Array.isArray(event?.contexts)) {
                dispatch({ type: "SET_QUERY_STAGE", payload: "ranking" })
                streamResults = mapStreamingContextsToResults(event.contexts)
                streamWhyPayload = event?.why
                streamSourceStatusPayload =
                  event?.source_status ?? event?.metadata?.source_status ?? null
                resolvedSearchDetails = buildSearchDetailsFromStreaming(
                  streamResults,
                  streamWhyPayload,
                  streamSourceStatusPayload,
                  effectiveSettings
                )
                dispatch({
                  type: "SET_SEARCH_DETAILS",
                  payload: resolvedSearchDetails,
                })
                const partialAnswer = normalizeAnswerText(streamAnswer)
                const partialCitations = partialAnswer
                  ? parseCitations(partialAnswer, streamResults)
                  : []
                const partialTrustState = normalizeKnowledgeAnswerTrust({
                  answer: partialAnswer,
                  results: streamResults,
                  citations: partialCitations,
                  hasRequiredMetadata: true,
                  syncFailed: isThreadSyncFailed(),
                  weakEvidence: hasWeakEvidence(
                    streamResults,
                    effectiveSettings.strip_min_relevance
                  ),
                }).state
                dispatch({
                  type: "SET_PARTIAL_RESULTS",
                  payload: {
                    results: streamResults,
                    answer: partialAnswer,
                    citations: partialCitations,
                    answerTrustState: partialTrustState,
                  },
                })
                receivedStreamEvent = true
                continue
              }

              if (eventType === "delta") {
                dispatch({ type: "SET_QUERY_STAGE", payload: "generating" })
                const deltaText =
                  typeof event?.text === "string" ? event.text : ""
                if (!deltaText) continue
                streamAnswer += deltaText
                const partialAnswer = normalizeAnswerText(streamAnswer)
                const partialCitations = partialAnswer
                  ? parseCitations(partialAnswer, streamResults)
                  : []
                const partialTrustState = normalizeKnowledgeAnswerTrust({
                  answer: partialAnswer,
                  results: streamResults,
                  citations: partialCitations,
                  hasRequiredMetadata: true,
                  syncFailed: isThreadSyncFailed(),
                  weakEvidence: hasWeakEvidence(
                    streamResults,
                    effectiveSettings.strip_min_relevance
                  ),
                }).state
                dispatch({
                  type: "SET_PARTIAL_RESULTS",
                  payload: {
                    results: streamResults,
                    answer: partialAnswer,
                    citations: partialCitations,
                    answerTrustState: partialTrustState,
                  },
                })
                receivedStreamEvent = true
                continue
              }

              if (eventType === "error") {
                throw new Error(
                  typeof event?.message === "string" && event.message
                    ? event.message
                    : "Search failed"
                )
              }
            }

            if (receivedStreamEvent) {
              results = streamResults
              answer = normalizeAnswerText(streamAnswer)
              usedStreaming = true
              resolvedSearchDetails = buildSearchDetailsFromStreaming(
                streamResults,
                streamWhyPayload,
                streamSourceStatusPayload,
                effectiveSettings
              )
            }
          } catch (streamError) {
            const streamMessage =
              streamError instanceof Error
                ? streamError.message
                : String(streamError ?? "")
            const isStreamAbort =
              abortController.signal.aborted ||
              (streamError as { name?: string } | null)?.name === "AbortError" ||
              /abort|cancel/i.test(streamMessage)
            if (isStreamAbort) {
              throw streamError
            }

            console.warn(
              "Streaming search failed, falling back to standard search:",
              streamError
            )
          }
        }

        if (!usedStreaming) {
          dispatch({ type: "SET_QUERY_STAGE", payload: "ranking" })
          const response = await tldwClient.ragSearch(trimmedQuery, {
            ...options,
            signal: abortController.signal,
          })
          if (isStaleSearchRequest()) {
            return
          }
          const extracted = extractRagResponse(response)
          results = extracted.results
          answer = extracted.answer
          if (answer) {
            dispatch({ type: "SET_QUERY_STAGE", payload: "generating" })
          }
          resolvedSearchDetails = buildSearchDetailsFromResponse(
            extracted,
            results,
            effectiveSettings
          )
        }

        if (isStaleSearchRequest()) {
          return
        }
        // Parse citations from answer
        dispatch({ type: "SET_QUERY_STAGE", payload: "verifying" })
        const citations = answer ? parseCitations(answer, results) : []
        const answerTrustState = normalizeKnowledgeAnswerTrust({
          answer,
          results,
          citations,
          hasRequiredMetadata: true,
          syncFailed: isThreadSyncFailed(),
          weakEvidence: hasWeakEvidence(results, effectiveSettings.strip_min_relevance),
        }).state

        dispatch({
          type: "SET_RESULTS",
          payload: { results, answer, citations, answerTrustState },
        })
        dispatch({ type: "SET_SEARCH_DETAILS", payload: resolvedSearchDetails })
        dispatch({
          type: "SET_LAST_SEARCH_SCOPE",
          payload: buildScopeSnapshot(state.preset, effectiveSettings),
        })
        void trackKnowledgeQaSearchMetric({
          type: "search_complete",
          duration_ms: Date.now() - searchStartedAt,
          result_count: results.length,
          has_answer: Boolean(answer),
          used_streaming: usedStreaming,
        })

        let assistantMessageId: string | null = null
        const ragContext = buildRagContext(
          trimmedQuery,
          results,
          answer,
          effectiveSettings,
          answerTrustState
        )

        if (answer && threadId) {
          const persistedAssistant = await persistChatMessage(
            threadId,
            "assistant",
            answer,
            persistedUser?.id
          )
          if (isStaleSearchRequest()) {
            return
          }
          assistantMessageId = persistedAssistant?.id || crypto.randomUUID()
          const assistantMessage: KnowledgeQAMessage = {
            id: assistantMessageId,
            conversationId: threadId,
            role: "assistant",
            content: answer,
            timestamp: persistedAssistant?.timestamp || new Date().toISOString(),
            ragContext,
          }
          dispatch({ type: "ADD_MESSAGE", payload: assistantMessage })

          if (persistedAssistant?.id) {
            await persistRagContext(persistedAssistant.id, ragContext)
            if (isStaleSearchRequest()) {
              return
            }
          }
        }

        if (isStaleSearchRequest()) {
          return
        }
        if (addToHistory) {
          const historyItem: SearchHistoryItem = {
            id: crypto.randomUUID(),
            query: trimmedQuery,
            timestamp: new Date().toISOString(),
            sourcesCount: results.length,
            hasAnswer: !!answer,
            answerPreview: truncateAnswerPreview(answer),
            preset: state.preset,
            settingsSnapshot: createRestorableSettingsSnapshot(effectiveSettings),
            conversationId: threadId && !isLocalThreadId(threadId) ? threadId : undefined,
            messageId: assistantMessageId || undefined,
            keywords: [KNOWLEDGE_QA_KEYWORD],
            trustState: answerTrustState,
          }
          markHistoryMutation()
          dispatch({ type: "ADD_HISTORY_ITEM", payload: historyItem })
        }
      } catch (error) {
        const rawErrorMessage =
          error instanceof Error
            ? error.message
            : typeof error === "string"
              ? error
              : ""
        const isAbortError =
          abortController.signal.aborted ||
          (error as { name?: string } | null)?.name === "AbortError" ||
          /abort|cancel/i.test(rawErrorMessage)

        if (isStaleSearchRequest()) {
          return
        }
        if (isAbortError) {
          const abortReason = abortController.signal.reason
          if (abortReason === "clear" || abortReason === "superseded") {
            dispatch({ type: "SET_ERROR", payload: null })
            return
          }
          dispatch({ type: "SET_ERROR", payload: "Search cancelled" })
          dispatch({ type: "SET_QUERY_STAGE", payload: "idle" })
          return
        }
        console.error("Search failed:", error)
        const errorMessage = mapKnowledgeQaSearchErrorMessage(error, "Search failed")
        dispatch({
          type: "SET_ERROR",
          payload: errorMessage,
        })
        if (addToHistory) {
          const historyItem: SearchHistoryItem = {
            id: crypto.randomUUID(),
            query: trimmedQuery,
            timestamp: new Date().toISOString(),
            sourcesCount: 0,
            hasAnswer: false,
            preset: state.preset,
            settingsSnapshot: createRestorableSettingsSnapshot(effectiveSettings),
            conversationId: threadId && !isLocalThreadId(threadId) ? threadId : undefined,
            keywords: [KNOWLEDGE_QA_KEYWORD],
            trustState: "failed_search",
          }
          markHistoryMutation()
          dispatch({ type: "ADD_HISTORY_ITEM", payload: historyItem })
        }
      } finally {
        if (activeSearchAbortRef.current === abortController) {
          activeSearchAbortRef.current = null
        }
      }
    },
    [
      state.currentThreadId,
      state.settings,
      state.preset,
      state.pinnedSourceFilters,
      beginThreadHydrationRequest,
      markHistoryMutation,
      createNewThread,
      persistChatMessage,
      persistRagContext,
      buildRagContext,
      streamingFeatureEnabled,
    ]
  )

  const cancelSearch = useCallback(() => {
    if (!state.isSearching || !activeSearchAbortRef.current) return

    void trackKnowledgeQaSearchMetric({ type: "search_cancel" })
    activeSearchAbortRef.current.abort("cancel")
    message.open({
      type: "info",
      content: "Search cancelled.",
      duration: 2,
    })
  }, [message, state.isSearching])

  const search = useCallback(async () => {
    const queryLength = state.query.trim().length
    if (queryLength > 0) {
      void trackKnowledgeQaSearchMetric({
        type: "search_submit",
        query_length: queryLength,
      })
    }
    await runKnowledgeQuery(state.query, true)
  }, [state.query, runKnowledgeQuery])

  const rerunWithTokenLimit = useCallback(
    async (tokenLimit: number) => {
      const normalized = Math.max(64, Math.min(4000, Math.round(tokenLimit)))
      await runKnowledgeQuery(state.query, false, {
        max_generation_tokens: normalized,
        enable_generation: true,
      })
    },
    [runKnowledgeQuery, state.query]
  )

  const clearResults = useCallback(() => {
    const hasClearableState =
      state.results.length > 0 ||
      Boolean(state.answer) ||
      state.messages.length > 0 ||
      Boolean(state.currentThreadId)
    if (hasClearableState) {
      void trackKnowledgeQaSearchMetric({ type: "search_clear_full" })
    }

    if (activeSearchAbortRef.current) {
      activeSearchAbortRef.current.abort("clear")
      activeSearchAbortRef.current = null
    }
    beginThreadHydrationRequest()
    dispatch({ type: "CLEAR_RESULTS" })
    dispatch({ type: "SET_THREAD_ID", payload: null })
    dispatch({ type: "SET_LOCAL_ONLY_THREAD", payload: false })
    dispatch({ type: "SET_MESSAGES", payload: [] })
  }, [
    beginThreadHydrationRequest,
    state.answer,
    state.currentThreadId,
    state.messages.length,
    state.results.length,
  ])

  const startNewTopic = useCallback(async (): Promise<string> => {
    clearResults()
    dispatch({ type: "SET_QUERY", payload: "" })
    const newTopicRequestId = beginThreadHydrationRequest()
    return await createNewThread(undefined, {
      shouldActivate: () =>
        activeThreadHydrationRequestIdRef.current === newTopicRequestId,
      cleanupRemoteIfSkipped: true,
    })
  }, [beginThreadHydrationRequest, clearResults, createNewThread])

  const selectThread = useCallback(async (threadId: string): Promise<ThreadHydrationResult> => {
    const threadHydrationRequestId = beginThreadHydrationRequest()
    const isStaleThreadHydrationRequest = () =>
      activeThreadHydrationRequestIdRef.current !== threadHydrationRequestId

    if (activeSearchAbortRef.current) {
      activeSearchAbortRef.current.abort("superseded")
      activeSearchAbortRef.current = null
    }

    if (isLocalThreadId(threadId)) {
      dispatch({ type: "SET_THREAD_ID", payload: threadId })
      dispatch({ type: "SET_LOCAL_ONLY_THREAD", payload: true })
      dispatch({
        type: "SET_PINNED_SOURCE_FILTERS",
        payload: { mediaIds: [], noteIds: [] },
      })
      dispatch({ type: "SET_ERROR", payload: null })
      dispatch({ type: "SET_MESSAGES", payload: [] })
      dispatch({ type: "CLEAR_RESULTS" })
      return true
    }

    try {
      // Load messages with RAG context
      const response = await tldwClient.fetchWithAuth(
        `/api/v1/chat/conversations/${threadId}/messages-with-context?include_rag_context=true`
      )
      if (isStaleThreadHydrationRequest()) {
        return false
      }
      if (!response.ok) {
        throw new Error(`Failed to load thread ${threadId} (HTTP ${response.status})`)
      }
      const rawMessages = await response.json()
      if (isStaleThreadHydrationRequest()) {
        return false
      }
      const messages = normalizeMessagesWithContext(rawMessages, threadId)
      dispatch({ type: "SET_THREAD_ID", payload: threadId })
      dispatch({ type: "SET_LOCAL_ONLY_THREAD", payload: false })
      dispatch({
        type: "SET_PINNED_SOURCE_FILTERS",
        payload: { mediaIds: [], noteIds: [] },
      })
      dispatch({ type: "SET_ERROR", payload: null })
      dispatch({ type: "SET_MESSAGES", payload: messages })

      const hydration = deriveThreadHydrationState(messages)
      if (hydration?.query) {
        dispatch({ type: "SET_QUERY", payload: hydration.query })
      }
      if (hydration && (hydration.results.length > 0 || hydration.answer)) {
        dispatch({
          type: "SET_RESULTS",
          payload: {
            results: hydration.results,
            answer: hydration.answer,
            citations: hydration.citations,
            answerTrustState: hydration.answerTrustState,
          },
        })
      } else {
        dispatch({ type: "CLEAR_RESULTS" })
      }

      const matchingHistoryItem = state.searchHistory.find(
        (item) => item.conversationId === threadId
      )
      dispatch({
        type: "HYDRATE_RESTORED_SCOPE",
        payload: {
          preset: matchingHistoryItem?.preset,
          settingsSnapshot:
            hydration?.settingsSnapshot ??
            normalizeRestorableSettingsSnapshot(matchingHistoryItem?.settingsSnapshot),
        },
      })
      if (matchingHistoryItem && hydration) {
        markHistoryMutation()
        dispatch({
          type: "UPDATE_HISTORY_ITEM",
          payload: {
            id: matchingHistoryItem.id,
            patch: {
              answerPreview: hydration.answerPreview,
              hasAnswer: Boolean(hydration.answer),
              sourcesCount: hydration.sourcesCount || matchingHistoryItem.sourcesCount,
              trustState: hydration.answerTrustState,
            },
          },
        })
      }
      return true
    } catch (error) {
      if (isStaleThreadHydrationRequest()) {
        return false
      }
      console.error("Failed to load thread messages:", error)
      dispatch({ type: "SET_ERROR", payload: "Unable to load this conversation right now." })
      return classifyThreadHydrationFailure(error)
    }
  }, [beginThreadHydrationRequest, markHistoryMutation, state.searchHistory])

  const selectSharedThread = useCallback(
    async (shareToken: string): Promise<ThreadHydrationResult> => {
      const threadHydrationRequestId = beginThreadHydrationRequest()
      const isStaleThreadHydrationRequest = () =>
        activeThreadHydrationRequestIdRef.current !== threadHydrationRequestId
      const trimmedToken = shareToken.trim()
      if (!trimmedToken) {
        dispatch({ type: "SET_ERROR", payload: "Shared link is invalid." })
        return "terminal"
      }

      if (activeSearchAbortRef.current) {
        activeSearchAbortRef.current.abort("superseded")
        activeSearchAbortRef.current = null
      }

      try {
        const payload = await tldwClient.resolveConversationShareLink(trimmedToken)
        if (isStaleThreadHydrationRequest()) {
          return false
        }
        const conversationId =
          typeof payload?.conversation_id === "string" &&
          payload.conversation_id.trim().length > 0
            ? payload.conversation_id.trim()
            : `shared-${crypto.randomUUID()}`
        const rawMessages = Array.isArray(payload?.messages) ? payload.messages : []
        const messages = normalizeMessagesWithContext(rawMessages, conversationId)

        dispatch({ type: "SET_THREAD_ID", payload: null })
        dispatch({ type: "SET_LOCAL_ONLY_THREAD", payload: false })
        dispatch({ type: "SET_MESSAGES", payload: messages })
        dispatch({ type: "SET_SEARCH_DETAILS", payload: null })
        dispatch({
          type: "SET_PINNED_SOURCE_FILTERS",
          payload: { mediaIds: [], noteIds: [] },
        })
        dispatch({ type: "SET_ERROR", payload: null })

        const hydration = deriveThreadHydrationState(messages)
        if (hydration?.query) {
          dispatch({ type: "SET_QUERY", payload: hydration.query })
        }
        if (hydration && (hydration.results.length > 0 || hydration.answer)) {
          dispatch({
            type: "SET_RESULTS",
            payload: {
              results: hydration.results,
              answer: hydration.answer,
              citations: hydration.citations,
              answerTrustState: hydration.answerTrustState,
            },
          })
        } else {
          dispatch({ type: "CLEAR_RESULTS" })
        }
        dispatch({
          type: "HYDRATE_RESTORED_SCOPE",
          payload: {
            settingsSnapshot: hydration?.settingsSnapshot ?? null,
          },
        })
        return true
      } catch (error) {
        if (isStaleThreadHydrationRequest()) {
          return false
        }
        console.error("Failed to load shared thread:", error)
        dispatch({ type: "SET_THREAD_ID", payload: null })
        dispatch({ type: "SET_LOCAL_ONLY_THREAD", payload: false })
        dispatch({ type: "SET_MESSAGES", payload: [] })
        dispatch({ type: "SET_SEARCH_DETAILS", payload: null })
        dispatch({
          type: "SET_PINNED_SOURCE_FILTERS",
          payload: { mediaIds: [], noteIds: [] },
        })
        dispatch({ type: "CLEAR_RESULTS" })
        dispatch({
          type: "SET_ERROR",
          payload: "Unable to open this shared conversation link.",
        })
        return classifyThreadHydrationFailure(error)
      }
    },
    [beginThreadHydrationRequest]
  )

  const branchFromTurn = useCallback(
    async (messageId: string) => {
      const branchRequestId = beginThreadHydrationRequest()
      const isStaleBranchRequest = () =>
        activeThreadHydrationRequestIdRef.current !== branchRequestId

      if (activeSearchAbortRef.current) {
        activeSearchAbortRef.current.abort("superseded")
        activeSearchAbortRef.current = null
      }

      const branchSeedMessages = sliceMessagesForBranch(state.messages, messageId)
      if (branchSeedMessages.length === 0) {
        message.open({
          type: "warning",
          content: "Unable to branch from that turn.",
          duration: 3,
        })
        return
      }

      const branchSourceQuestion =
        branchSeedMessages.find((entry) => entry.id === messageId)?.content?.trim() || ""
      const branchTitle =
        branchSourceQuestion.length > 0
          ? `Branch: ${branchSourceQuestion.slice(0, 64)}`
          : "Knowledge QA Branch"
      const parentConversationId =
        state.currentThreadId && !isLocalThreadId(state.currentThreadId)
          ? state.currentThreadId
          : undefined
      const cleanupStaleBranchThread = async (threadId: string) => {
        if (!threadId || isLocalThreadId(threadId)) {
          return
        }
        try {
          await tldwClient.deleteChat(threadId)
        } catch (cleanupError) {
          console.warn("Failed to delete stale branch conversation:", cleanupError)
        }
      }
      const branchThreadId = await createNewThread(branchTitle, {
        parentConversationId,
        forkedFromMessageId: messageId,
        shouldActivate: () => !isStaleBranchRequest(),
        cleanupRemoteIfSkipped: true,
      })
      if (isStaleBranchRequest()) {
        return
      }

      const branchedMessages: KnowledgeQAMessage[] = []
      let parentMessageId: string | null = null

      for (const sourceMessage of branchSeedMessages) {
        const persisted = await persistChatMessage(
          branchThreadId,
          sourceMessage.role,
          sourceMessage.content,
          parentMessageId
        )
        if (isStaleBranchRequest()) {
          await cleanupStaleBranchThread(branchThreadId)
          return
        }
        const nextMessageId = persisted?.id || crypto.randomUUID()
        parentMessageId = persisted?.id || null

        const branchMessage: KnowledgeQAMessage = {
          ...sourceMessage,
          id: nextMessageId,
          conversationId: branchThreadId,
          timestamp: persisted?.timestamp || sourceMessage.timestamp,
        }
        branchedMessages.push(branchMessage)

        if (
          sourceMessage.role === "assistant" &&
          sourceMessage.ragContext &&
          persisted?.id
        ) {
          await persistRagContext(persisted.id, sourceMessage.ragContext)
          if (isStaleBranchRequest()) {
            await cleanupStaleBranchThread(branchThreadId)
            return
          }
        }
      }

      if (isStaleBranchRequest()) {
        await cleanupStaleBranchThread(branchThreadId)
        return
      }

      dispatch({ type: "SET_THREAD_ID", payload: branchThreadId })
      dispatch({ type: "SET_LOCAL_ONLY_THREAD", payload: isLocalThreadId(branchThreadId) })
      dispatch({ type: "SET_MESSAGES", payload: branchedMessages })
      dispatch({ type: "SET_SEARCH_DETAILS", payload: null })
      dispatch({
        type: "SET_PINNED_SOURCE_FILTERS",
        payload: { mediaIds: [], noteIds: [] },
      })

      const hydration = deriveThreadHydrationState(branchedMessages)
      if (hydration?.query) {
        dispatch({ type: "SET_QUERY", payload: hydration.query })
      }
      if (hydration && (hydration.results.length > 0 || hydration.answer)) {
        dispatch({
          type: "SET_RESULTS",
          payload: {
            results: hydration.results,
            answer: hydration.answer,
            citations: hydration.citations,
            answerTrustState: isLocalThreadId(branchThreadId)
              ? "unsynced_local_result"
              : hydration.answerTrustState,
          },
        })
      } else {
        dispatch({ type: "CLEAR_RESULTS" })
      }

      message.open({
        type: "success",
        content: "Branch created from selected turn.",
        duration: 3,
      })
    },
    [
      beginThreadHydrationRequest,
      createNewThread,
      message,
      persistChatMessage,
      persistRagContext,
      state.currentThreadId,
      state.messages,
    ]
  )

  const askFollowUp = useCallback(
    async (question: string) => {
      await runKnowledgeQuery(question, false)
    },
    [runKnowledgeQuery]
  )

  const setPreset = useCallback((preset: RagPresetName) => {
    dispatch({ type: "SET_PRESET", payload: preset })
  }, [])

  const updateSetting = useCallback(<K extends keyof RagSettings>(key: K, value: RagSettings[K]) => {
    dispatch({ type: "UPDATE_SETTING", payload: { key, value } })
  }, [])

  const resetSettings = useCallback(() => {
    dispatch({ type: "SET_SETTINGS", payload: { ...DEFAULT_RAG_SETTINGS, ...KNOWLEDGE_QA_SETTINGS_OVERRIDES } })
    dispatch({ type: "SET_PRESET", payload: "balanced" })
  }, [])

  const toggleExpertMode = useCallback(() => {
    dispatch({ type: "TOGGLE_EXPERT_MODE" })
  }, [])

  const loadSearchHistory = useCallback(async () => {
    const { requestId, mutationVersion } = beginHistoryLoadRequest()
    // Load from local storage for now
    let storedItems: SearchHistoryItem[] = []
    try {
      const stored = localStorage.getItem("knowledge_qa_history")
      if (stored) {
        try {
          storedItems = JSON.parse(stored) as SearchHistoryItem[]
          dispatch({ type: "SET_SEARCH_HISTORY", payload: storedItems })
        } catch (error) {
          console.error("Failed to parse Knowledge QA local history:", error)
        }
      }
      // Best-effort: merge server conversation history for cross-session continuity
      try {
        const response = await tldwClient.fetchWithAuth(
          `/api/v1/chat/conversations?order_by=recency&limit=50&keywords=${encodeURIComponent(
            KNOWLEDGE_QA_KEYWORD
          )}`
        )
        if (response.ok) {
          const data = await response.json()
          if (
            activeHistoryLoadRequestIdRef.current !== requestId ||
            historyMutationVersionRef.current !== mutationVersion
          ) {
            return
          }
          const items: any[] = Array.isArray(data)
            ? data
            : data?.items || data?.conversations || data?.chats || data?.results || []
          const serverHistory: SearchHistoryItem[] = items
            .map((conv) => {
              const id =
                conv?.id ||
                conv?.conversation_id ||
                conv?.chat_id ||
                conv?.chatId
              if (!id) return null
              const keywords = Array.isArray(conv?.keywords) ? conv.keywords : []
              const hasKeyword = keywords.some(
                (kw: string) => String(kw).toLowerCase() === KNOWLEDGE_QA_KEYWORD.toLowerCase()
              )
              if (!hasKeyword) return null
              const title = typeof conv?.title === "string" ? conv.title : "Knowledge QA"
              const lastModified =
                conv?.last_modified ||
                conv?.lastModified ||
                conv?.last_modified_at ||
                conv?.lastModifiedAt ||
                conv?.created_at ||
                conv?.createdAt ||
                new Date().toISOString()
              const messageCount =
                typeof conv?.message_count === "number"
                  ? conv.message_count
                  : typeof conv?.messageCount === "number"
                    ? conv.messageCount
                    : 0
              return {
                id: String(id),
                query: title,
                timestamp: new Date(lastModified).toISOString(),
                sourcesCount: messageCount,
                hasAnswer: messageCount >= 2,
                conversationId: String(id),
                keywords,
              } as SearchHistoryItem
            })
            .filter(Boolean) as SearchHistoryItem[]

          if (serverHistory.length > 0) {
            const mergedMap = new Map<string, SearchHistoryItem>()
            for (const item of storedItems) {
              const itemKeywords = Array.isArray(item.keywords) ? item.keywords : []
              const hasKeyword = itemKeywords.some(
                (kw) => String(kw).toLowerCase() === KNOWLEDGE_QA_KEYWORD.toLowerCase()
              )
              if (!hasKeyword) continue
              const key = item.conversationId
                ? `conv:${item.conversationId}`
                : `query:${item.query}`
              mergedMap.set(key, item)
            }
            for (const item of serverHistory) {
              const key = item.conversationId
                ? `conv:${item.conversationId}`
                : `query:${item.query}`
              if (!mergedMap.has(key)) {
                mergedMap.set(key, item)
              }
            }
            const merged = Array.from(mergedMap.values()).sort(
              (a, b) =>
                new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
            )
            dispatch({ type: "SET_SEARCH_HISTORY", payload: merged })
          }
        }
      } catch {
        // ignore server history fetch failures
      }
    } catch (error) {
      console.error("Failed to load search history:", error)
    } finally {
      setHistoryHydrated(true)
    }
  }, [beginHistoryLoadRequest])

  const restoreFromHistory = useCallback(
    async (item: SearchHistoryItem) => {
      const restoredQuery = item.query.trim()
      const restoredScopePayload = {
        preset: item.preset,
        settingsSnapshot: normalizeRestorableSettingsSnapshot(item.settingsSnapshot),
      }
      const restoredScope = resolveHydratedScopeState(
        state.preset,
        state.settings,
        restoredScopePayload
      )
      if (restoredQuery.length > 0) {
        dispatch({ type: "SET_QUERY", payload: restoredQuery })
      }
      if (item.conversationId) {
        if (isLocalThreadId(item.conversationId)) {
          dispatch({
            type: "HYDRATE_RESTORED_SCOPE",
            payload: restoredScopePayload,
          })
          if (restoredQuery.length > 0) {
            await runKnowledgeQuery(restoredQuery, false, restoredScope.nextSettings)
          }
          return
        }
        await selectThread(item.conversationId)
        return
      }

      dispatch({
        type: "HYDRATE_RESTORED_SCOPE",
        payload: restoredScopePayload,
      })
      if (restoredQuery.length > 0) {
        await runKnowledgeQuery(restoredQuery, false, restoredScope.nextSettings)
      }
    },
    [runKnowledgeQuery, selectThread, state.preset, state.settings]
  )

  const toggleHistoryPin = useCallback((id: string) => {
    markHistoryMutation()
    dispatch({ type: "TOGGLE_HISTORY_PIN", payload: id })
  }, [markHistoryMutation])

  const deleteHistoryItem = useCallback(
    async (id: string) => {
      const item = state.searchHistory.find((h) => h.id === id)
      const conversationId = item?.conversationId

      const toastKey = `knowledge-qa-delete-${id}`
      const attemptServerDelete = async () => {
        try {
          await tldwClient.deleteChat(conversationId as string)
          message.open({
            key: toastKey,
            type: "success",
            content: "Deleted from server history.",
            duration: 2.5,
          })
          return true
        } catch (error) {
          console.warn("Failed to delete Knowledge QA conversation:", error)
          message.open({
            key: toastKey,
            type: "error",
            duration: 4,
            content: (
              <span className="inline-flex items-center gap-2">
                <span>Failed to delete server history.</span>
                <button
                  className="text-primary underline"
                  onClick={() => {
                    attemptServerDelete().catch(() => undefined)
                  }}
                >
                  Retry
                </button>
              </span>
            ),
            className: "max-w-sm",
          })
          return false
        }
      }

      if (conversationId && !isLocalThreadId(conversationId)) {
        const ok = await attemptServerDelete()
        if (!ok) {
          return
        }
      } else {
        message.open({
          key: toastKey,
          type: "success",
          content: "Removed from local history.",
          duration: 2,
        })
      }

      const deletedActiveThread =
        Boolean(state.currentThreadId) &&
        (conversationId === state.currentThreadId || id === state.currentThreadId)
      if (deletedActiveThread) {
        clearResults()
        dispatch({ type: "SET_QUERY", payload: "" })
      }

      markHistoryMutation()
      dispatch({ type: "REMOVE_HISTORY_ITEM", payload: id })
    },
    [clearResults, markHistoryMutation, message, state.currentThreadId, state.searchHistory]
  )

  const setSettingsPanelOpen = useCallback((open: boolean) => {
    dispatch({ type: "SET_SETTINGS_PANEL_OPEN", payload: open })
  }, [])

  const setHistorySidebarOpen = useCallback((open: boolean) => {
    dispatch({ type: "SET_HISTORY_SIDEBAR_OPEN", payload: open })
  }, [])

  const focusSource = useCallback((index: number | null) => {
    dispatch({ type: "SET_FOCUSED_SOURCE", payload: index })
  }, [])

  const setEvidenceRailOpen = useCallback((open: boolean) => {
    dispatch({ type: "SET_EVIDENCE_RAIL_OPEN", payload: open })
  }, [])

  const setEvidenceRailTab = useCallback((tab: "sources" | "details") => {
    dispatch({ type: "SET_EVIDENCE_RAIL_TAB", payload: tab })
  }, [])

  const setQueryStage = useCallback((stage: QueryStage) => {
    dispatch({ type: "SET_QUERY_STAGE", payload: stage })
  }, [])

  const setPinnedSourceFilters = useCallback((filters: PinnedSourceFilters) => {
    const normalized: PinnedSourceFilters = {
      mediaIds: mergeNumberFilters(filters.mediaIds),
      noteIds: mergeStringFilters(filters.noteIds),
    }
    dispatch({ type: "SET_PINNED_SOURCE_FILTERS", payload: normalized })
  }, [])

  const refreshSourceHealth = useCallback(async () => {
    const requestId = sourceHealthRequestIdRef.current + 1
    sourceHealthRequestIdRef.current = requestId
    dispatch({ type: "SET_SOURCE_HEALTH_LOADING" })
    try {
      const payload = await tldwClient.ragSourceHealth()
      if (sourceHealthRequestIdRef.current !== requestId) return
      dispatch({
        type: "SET_SOURCE_HEALTH",
        payload: normalizeKnowledgeSourceHealth(payload),
      })
    } catch {
      if (sourceHealthRequestIdRef.current !== requestId) return
      dispatch({
        type: "SET_SOURCE_HEALTH_ERROR",
        payload:
          "Source health could not be loaded. You can still search selected sources.",
      })
    }
  }, [])

  // Initialize client and load safe pre-query source health once when ready.
  useEffect(() => {
    let cancelled = false

    tldwClient
      .initialize()
      .then(() => {
        if (!cancelled) {
          void refreshSourceHealth()
        }
      })
      .catch(console.error)

    return () => {
      cancelled = true
    }
  }, [refreshSourceHealth])

  const scrollToSource = useCallback((index: number) => {
    const element = document.getElementById(`source-card-${index}`)
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "center" })
      dispatch({ type: "SET_FOCUSED_SOURCE", payload: index })
    }
  }, [])

  const scrollToCitation = useCallback((citationIndex: number, occurrence = 1) => {
    if (!Number.isFinite(citationIndex) || citationIndex < 1) return
    const normalizedIndex = Math.floor(citationIndex)
    const normalizedOccurrence =
      Number.isFinite(occurrence) && occurrence >= 1 ? Math.floor(occurrence) : 1
    const selector = `[data-knowledge-citation-index="${normalizedIndex}"]`
    const occurrenceSelector = `${selector}[data-knowledge-citation-occurrence="${normalizedOccurrence}"]`
    const inlineCitationByOccurrence = document.querySelector(
      `#knowledge-answer-content ${occurrenceSelector}`
    ) as HTMLElement | null
    const inlineCitationElement = document.querySelector(
      `#knowledge-answer-content ${selector}`
    ) as HTMLElement | null
    const targetElement =
      inlineCitationByOccurrence ??
      inlineCitationElement ??
      (document.querySelector(selector) as HTMLElement | null)

    if (targetElement) {
      targetElement.scrollIntoView({ behavior: "smooth", block: "center" })
      if (typeof targetElement.focus === "function") {
        targetElement.focus()
      }
    }
  }, [])

  // Save history to local storage when it changes
  useEffect(() => {
    if (!historyHydrated) {
      return
    }
    if (state.searchHistory.length === 0) {
      try {
        localStorage.removeItem("knowledge_qa_history")
      } catch (error) {
        console.error("Failed to clear Knowledge QA history:", error)
      }
      return
    }

    try {
      const { storedHistory, wasTrimmed } = persistKnowledgeQaHistory(
        state.searchHistory,
        (serializedHistory) => {
          localStorage.setItem("knowledge_qa_history", serializedHistory)
        }
      )

      if (wasTrimmed && storedHistory.length !== state.searchHistory.length) {
        markHistoryMutation()
        dispatch({ type: "SET_SEARCH_HISTORY", payload: storedHistory })
        if (!historyQuotaWarningShownRef.current) {
          historyQuotaWarningShownRef.current = true
          message.open({
            type: "warning",
            content: "History storage is full. Oldest searches were trimmed locally.",
            duration: 4,
          })
        }
      }
    } catch (error) {
      console.error("Failed to persist Knowledge QA history:", error)
    }
  }, [historyHydrated, markHistoryMutation, message, state.searchHistory])

  // Load history on mount
  useEffect(() => {
    loadSearchHistory()
  }, [loadSearchHistory])

  // Auto-clear focused source highlight to prevent stale focus rings.
  useEffect(() => {
    if (typeof window === "undefined") return

    if (focusedSourceTimeoutRef.current != null) {
      window.clearTimeout(focusedSourceTimeoutRef.current)
      focusedSourceTimeoutRef.current = null
    }

    if (state.focusedSourceIndex == null) {
      return
    }

    focusedSourceTimeoutRef.current = window.setTimeout(() => {
      dispatch({ type: "SET_FOCUSED_SOURCE", payload: null })
      focusedSourceTimeoutRef.current = null
    }, 5000)

    return () => {
      if (focusedSourceTimeoutRef.current != null) {
        window.clearTimeout(focusedSourceTimeoutRef.current)
        focusedSourceTimeoutRef.current = null
      }
    }
  }, [state.focusedSourceIndex])

  // Memoized context value
  const contextValue = useMemo<KnowledgeQAContextValue>(
    () => ({
      ...state,
      historyHydrated,
      setQuery,
      search,
      cancelSearch,
      clearResults,
      rerunWithTokenLimit,
      createNewThread,
      startNewTopic,
      selectThread,
      selectSharedThread,
      askFollowUp,
      branchFromTurn,
      setPreset,
      updateSetting,
      resetSettings,
      toggleExpertMode,
      loadSearchHistory,
      restoreFromHistory,
      deleteHistoryItem,
      toggleHistoryPin,
      setSettingsPanelOpen,
      setHistorySidebarOpen,
      focusSource,
      setEvidenceRailOpen,
      setEvidenceRailTab,
      setQueryStage,
      setPinnedSourceFilters,
      refreshSourceHealth,
      persistRagContext,
      scrollToSource,
      scrollToCitation,
    }),
    [
      state,
      historyHydrated,
      setQuery,
      search,
      cancelSearch,
      clearResults,
      rerunWithTokenLimit,
      createNewThread,
      startNewTopic,
      selectThread,
      selectSharedThread,
      askFollowUp,
      branchFromTurn,
      setPreset,
      updateSetting,
      resetSettings,
      toggleExpertMode,
      loadSearchHistory,
      restoreFromHistory,
      deleteHistoryItem,
      toggleHistoryPin,
      setSettingsPanelOpen,
      setHistorySidebarOpen,
      focusSource,
      setEvidenceRailOpen,
      setEvidenceRailTab,
      setQueryStage,
      setPinnedSourceFilters,
      refreshSourceHealth,
      persistRagContext,
      scrollToSource,
      scrollToCitation,
    ]
  )

  return (
    <KnowledgeQAContext.Provider value={contextValue}>
      {children}
    </KnowledgeQAContext.Provider>
  )
}

// Hook to use the context
export function useKnowledgeQA(): KnowledgeQAContextValue {
  const context = useContext(KnowledgeQAContext)
  if (!context) {
    throw new Error("useKnowledgeQA must be used within a KnowledgeQAProvider")
  }
  return context
}
