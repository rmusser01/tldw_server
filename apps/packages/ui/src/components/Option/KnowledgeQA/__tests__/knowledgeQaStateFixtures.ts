import { vi } from "vitest"

import {
  DEFAULT_RAG_SETTINGS,
  type RagSettings,
} from "@/services/rag/unified-rag"
import type {
  KnowledgeQAContextValue,
  KnowledgeQAMessage,
  KnowledgeQAThread,
  RagResult,
  SearchRuntimeDetails,
  CitationRef,
} from "../types"

export type KnowledgeQaStateFixtureName =
  | "backendOffline"
  | "setupRequired"
  | "noIndexedSources"
  | "noSelectedSources"
  | "readySearch"
  | "results"
  | "noResults"
  | "settingsDrawer"
  | "exportDialog"

export type KnowledgeQaConnectionFixture = {
  online: boolean
  isChecking: boolean
  lastCheckedAt: number
  serverUrl: string | null
  configStep: "none" | "url" | "auth" | "health"
  errorKind: "none" | "auth" | "unreachable" | "partial"
  lastError: string | null
  lastStatusCode: number | null
  uxState:
    | "connected_ok"
    | "testing"
    | "configuring_url"
    | "configuring_auth"
    | "error_auth"
    | "error_unreachable"
    | "unconfigured"
  hasCompletedFirstRun: boolean
}

export type KnowledgeQaCapabilityFixture = {
  loading: boolean
  capabilities: {
    hasRag: boolean
    hasWebSearch?: boolean
  } | null
}

export type KnowledgeQaSourceInventoryFixture = {
  media: Array<{ id: number; title: string }>
  notes: Array<{ id: string; title: string }>
}

export type KnowledgeQaStateFixture = {
  knowledgeQa: KnowledgeQAContextValue
  connection: KnowledgeQaConnectionFixture
  capabilities: KnowledgeQaCapabilityFixture
  sourceInventory: KnowledgeQaSourceInventoryFixture
}

const nowIso = "2026-06-07T12:00:00.000Z"
const nowMs = Date.parse(nowIso)

function createSettings(overrides: Partial<RagSettings> = {}): RagSettings {
  return {
    ...DEFAULT_RAG_SETTINGS,
    sources: ["media_db", "notes"],
    enable_web_fallback: false,
    include_media_ids: [],
    include_note_ids: [],
    top_k: 10,
    ...overrides,
  }
}

function createResult(index: number, title = `Knowledge Source ${index}`): RagResult {
  return {
    id: `knowledge-source-${index}`,
    content: `Evidence excerpt ${index}`,
    excerpt: `Evidence excerpt ${index}`,
    score: 0.92 - index * 0.03,
    metadata: {
      title,
      source: "media_db",
      source_type: "media_db",
      url: `https://example.test/knowledge/${index}`,
      page_number: index,
    },
  }
}

function createCitation(index: number): CitationRef {
  return {
    index,
    documentId: `knowledge-source-${index}`,
    excerpt: `Evidence excerpt ${index}`,
  }
}

function createSearchDetails(): SearchRuntimeDetails {
  return {
    expandedQueries: ["What does the library say about grounded QA?"],
    rerankingEnabled: true,
    rerankingStrategy: "cross_encoder",
    averageRelevance: 0.86,
    webFallbackEnabled: false,
    webFallbackTriggered: false,
    webFallbackEngine: null,
    tokensUsed: 1240,
    estimatedCostUsd: 0.004,
    feedbackId: "feedback-knowledge-1",
    whyTheseSources: {
      topicality: 0.91,
      diversity: 0.72,
      freshness: 0.63,
    },
    faithfulnessScore: 0.9,
    faithfulnessTotalClaims: 3,
    faithfulnessSupportedClaims: 3,
    faithfulnessUnsupportedClaims: 0,
    verificationRate: 1,
    verificationCoverage: 0.88,
    verificationTotalClaims: 3,
    verificationVerifiedClaims: 3,
    verificationReportAvailable: true,
    retrievalLatencyMs: 420,
    documentsConsidered: 6,
    chunksConsidered: 24,
    documentsReturned: 3,
    candidatesConsidered: 12,
    candidatesReturned: 3,
    candidatesRejected: 9,
    alsoConsidered: [
      {
        id: "knowledge-source-4",
        title: "Related Knowledge Source",
        score: 0.52,
        reason: "Lower relevance than cited evidence",
      },
    ],
  }
}

function createMessages(): KnowledgeQAMessage[] {
  return [
    {
      id: "message-user-1",
      conversationId: "thread-knowledge-1",
      role: "user",
      content: "What does my library say about grounded QA?",
      timestamp: nowIso,
    },
    {
      id: "message-assistant-1",
      conversationId: "thread-knowledge-1",
      role: "assistant",
      content: "Your library says grounded QA should cite its evidence [1].",
      timestamp: nowIso,
    },
  ]
}

function createThreads(): KnowledgeQAThread[] {
  return [
    {
      id: "thread-knowledge-1",
      title: "Grounded QA review",
      createdAt: nowIso,
      lastModifiedAt: nowIso,
      state: "in-progress",
      messageCount: 2,
      source: "knowledge_qa",
    },
  ]
}

function createBaseKnowledgeQaState(): KnowledgeQAContextValue {
  return {
    query: "What does my library say about grounded QA?",
    isSearching: false,
    hasSearched: false,
    results: [],
    answer: null,
    citations: [],
    searchDetails: null,
    error: null,
    queryWarning: null,
    currentThreadId: null,
    isLocalOnlyThread: false,
    messages: [],
    threads: [],
    preset: "balanced",
    settings: createSettings(),
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
    historyHydrated: true,
    setQuery: vi.fn(),
    search: vi.fn(async () => undefined),
    cancelSearch: vi.fn(),
    clearResults: vi.fn(),
    rerunWithTokenLimit: vi.fn(async () => undefined),
    createNewThread: vi.fn(async () => "thread-knowledge-1"),
    startNewTopic: vi.fn(async () => "thread-knowledge-1"),
    selectThread: vi.fn(async () => true),
    selectSharedThread: vi.fn(async () => true),
    askFollowUp: vi.fn(async () => undefined),
    branchFromTurn: vi.fn(async () => undefined),
    setPreset: vi.fn(),
    updateSetting: vi.fn() as unknown as KnowledgeQAContextValue["updateSetting"],
    resetSettings: vi.fn(),
    toggleExpertMode: vi.fn(),
    loadSearchHistory: vi.fn(async () => undefined),
    restoreFromHistory: vi.fn(async () => undefined),
    deleteHistoryItem: vi.fn(async () => undefined),
    toggleHistoryPin: vi.fn(),
    setSettingsPanelOpen: vi.fn(),
    setHistorySidebarOpen: vi.fn(),
    focusSource: vi.fn(),
    setEvidenceRailOpen: vi.fn(),
    setEvidenceRailTab: vi.fn(),
    setQueryStage: vi.fn(),
    setPinnedSourceFilters: vi.fn(),
    persistRagContext: vi.fn(async () => true),
    scrollToSource: vi.fn(),
    scrollToCitation: vi.fn(),
  }
}

function createBaseFixture(): KnowledgeQaStateFixture {
  return {
    knowledgeQa: createBaseKnowledgeQaState(),
    connection: {
      online: true,
      isChecking: false,
      lastCheckedAt: nowMs,
      serverUrl: "http://127.0.0.1:8000",
      configStep: "health",
      errorKind: "none",
      lastError: null,
      lastStatusCode: null,
      uxState: "connected_ok",
      hasCompletedFirstRun: true,
    },
    capabilities: {
      loading: false,
      capabilities: {
        hasRag: true,
        hasWebSearch: true,
      },
    },
    sourceInventory: {
      media: [
        { id: 101, title: "Grounded QA Notes" },
        { id: 102, title: "Library Search Review" },
      ],
      notes: [
        { id: "note-grounded-qa", title: "Grounded QA checklist" },
      ],
    },
  }
}

function applyResultsState(fixture: KnowledgeQaStateFixture): void {
  fixture.knowledgeQa.hasSearched = true
  fixture.knowledgeQa.queryStage = "complete"
  fixture.knowledgeQa.currentThreadId = "thread-knowledge-1"
  fixture.knowledgeQa.messages = createMessages()
  fixture.knowledgeQa.threads = createThreads()
  fixture.knowledgeQa.results = [createResult(1), createResult(2), createResult(3)]
  fixture.knowledgeQa.answer =
    "Your library says grounded QA should cite answer claims with visible evidence [1]."
  fixture.knowledgeQa.citations = [createCitation(1)]
  fixture.knowledgeQa.searchDetails = createSearchDetails()
  fixture.knowledgeQa.evidenceRailOpen = true
}

export function createKnowledgeQaStateFixture(
  name: KnowledgeQaStateFixtureName
): KnowledgeQaStateFixture {
  const fixture = createBaseFixture()

  switch (name) {
    case "backendOffline":
      fixture.connection.online = false
      fixture.connection.uxState = "error_unreachable"
      fixture.connection.errorKind = "unreachable"
      fixture.connection.lastError = "Network error"
      fixture.connection.lastStatusCode = 0
      return fixture
    case "setupRequired":
      fixture.connection.online = false
      fixture.connection.uxState = "unconfigured"
      fixture.connection.serverUrl = null
      fixture.connection.configStep = "url"
      fixture.connection.hasCompletedFirstRun = false
      return fixture
    case "noIndexedSources":
      fixture.knowledgeQa.settings = createSettings({
        sources: [],
        enable_web_fallback: false,
      })
      fixture.sourceInventory = { media: [], notes: [] }
      return fixture
    case "noSelectedSources":
      fixture.knowledgeQa.settings = createSettings({
        sources: [],
        enable_web_fallback: false,
      })
      return fixture
    case "readySearch":
      return fixture
    case "results":
      applyResultsState(fixture)
      return fixture
    case "noResults":
      fixture.knowledgeQa.hasSearched = true
      fixture.knowledgeQa.queryStage = "complete"
      fixture.knowledgeQa.results = []
      fixture.knowledgeQa.answer = null
      fixture.knowledgeQa.citations = []
      return fixture
    case "settingsDrawer":
      fixture.knowledgeQa.settingsPanelOpen = true
      return fixture
    case "exportDialog":
      applyResultsState(fixture)
      return fixture
    default:
      return fixture
  }
}
