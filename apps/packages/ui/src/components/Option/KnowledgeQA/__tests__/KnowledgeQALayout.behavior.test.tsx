import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQALayout } from "../layout/KnowledgeQALayout"

const state = {
  settingsPanelOpen: false,
  results: [] as Array<{ id: string }>,
  answer: null as string | null,
  citations: [] as Array<{ id: string }>,
  hasSearched: false,
  isSearching: false,
  error: null as string | null,
  queryStage: "idle" as string,
  searchDetails: null as null | {
    alsoConsidered?: Array<{
      id: string
      title: string
      score: number | null
      reason: string | null
    }>
    sourceStatus?: Record<string, unknown>
  },
  preset: "balanced" as string,
  setPreset: vi.fn(),
  settings: {
    sources: [] as string[],
    enable_web_fallback: true,
    top_k: 10,
    include_media_ids: [] as number[],
    include_note_ids: [] as string[],
  },
  updateSetting: vi.fn(),
  setSettingsPanelOpen: vi.fn((open: boolean) => {
    state.settingsPanelOpen = open
  }),
  setQuery: vi.fn(),
  restoreFromHistory: vi.fn(),
  searchHistory: [] as Array<{
    id: string
    query: string
    timestamp: string
    sourcesCount: number
    hasAnswer: boolean
    keywords?: string[]
    conversationId?: string
  }>,
  messages: [] as Array<{ id: string; role: string; content: string }>,
  evidenceRailOpen: false,
  setEvidenceRailOpen: vi.fn((open: boolean) => {
    state.evidenceRailOpen = open
  }),
  evidenceRailTab: "sources" as "sources" | "details",
  setEvidenceRailTab: vi.fn((tab: "sources" | "details") => {
    state.evidenceRailTab = tab
  }),
  lastSearchScope: null as null | {
    preset: string
    webFallback: boolean
    sources: string[]
    includeMediaIds: number[]
    includeNoteIds: string[]
  },
  focusSource: vi.fn(),
  pinnedSourceFilters: {
    mediaIds: [] as number[],
    noteIds: [] as string[],
  },
  sourceHealth: {
    loading: false,
    error: null as string | null,
    loadedAt: "2026-05-16T00:00:00Z",
    sources: [] as unknown[],
    bySource: {},
  },
  refreshSourceHealth: vi.fn(),
}

const layoutModeState = {
  mode: "simple" as "simple" | "research" | "expert",
  isSimple: true,
  isResearch: false,
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => state,
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false,
}))

vi.mock("../hooks/useLayoutMode", () => ({
  useLayoutMode: () => ({
    mode: layoutModeState.mode,
    setLayoutMode: vi.fn(),
    isSimple: layoutModeState.isSimple,
    isResearch: layoutModeState.isResearch,
    showPromotionToast: false,
    dismissPromotion: vi.fn(),
    acceptPromotion: vi.fn(),
  }),
}))

vi.mock("../history/HistoryPane", () => ({
  HistoryPane: () => <div data-testid="knowledge-history-pane" />,
}))

vi.mock("../context/KnowledgeContextBar", () => ({
  KnowledgeContextBar: ({
    contextChangedSinceLastRun,
    sourceHealth,
    onRefreshSourceHealth,
  }: {
    contextChangedSinceLastRun: boolean
    sourceHealth?: { loadedAt: string | null }
    onRefreshSourceHealth?: () => void
  }) => (
    <div data-testid="knowledge-context-bar">
      {contextChangedSinceLastRun ? "Scope changed" : "Scope unchanged"}
      <span>{sourceHealth?.loadedAt ?? "No source health"}</span>
      <button type="button" onClick={onRefreshSourceHealth}>
        Refresh detailed source health
      </button>
    </div>
  ),
}))

vi.mock("../context/CompactToolbar", () => ({
  CompactToolbar: ({
    contextChangedSinceLastRun,
    sourceHealth,
    onRefreshSourceHealth,
  }: {
    contextChangedSinceLastRun: boolean
    sourceHealth?: { loadedAt: string | null }
    onRefreshSourceHealth?: () => void
  }) => (
    <div data-testid="knowledge-compact-toolbar">
      {contextChangedSinceLastRun ? "Scope changed" : "Scope unchanged"}
      <span>{sourceHealth?.loadedAt ?? "No source health"}</span>
      <button type="button" onClick={onRefreshSourceHealth}>
        Refresh compact source health
      </button>
    </div>
  ),
}))

vi.mock("../composer/KnowledgeComposer", () => ({
  KnowledgeComposer: () => (
    <input
      id="knowledge-search-input"
      aria-label="Search your knowledge base"
      data-testid="knowledge-composer"
    />
  ),
}))

vi.mock("../empty/KnowledgeReadyState", () => ({
  KnowledgeReadyState: ({
    sourceHealth,
  }: {
    sourceHealth?: { loadedAt: string | null }
  }) => (
    <div data-testid="knowledge-ready-state">
      {sourceHealth?.loadedAt ?? "No source health"}
    </div>
  ),
}))

vi.mock("../empty/InlineRecentSessions", () => ({
  InlineRecentSessions: () => <div data-testid="knowledge-inline-recent-sessions" />,
}))

vi.mock("../panels/AnswerWorkspace", () => ({
  AnswerWorkspace: () => <div data-testid="knowledge-answer-workspace" />,
}))

vi.mock("../panels/NoResultsRecovery", () => ({
  NoResultsRecovery: ({
    sourceHealth,
    selectedSources,
    onOpenQuickIngest,
    onShowNearestMatches,
    showNearestMatchesAvailable,
  }: {
    sourceHealth?: { loadedAt: string | null }
    selectedSources?: string[]
    onOpenQuickIngest?: () => void
    onShowNearestMatches?: () => void
    showNearestMatchesAvailable?: boolean
  }) => (
    <div data-testid="knowledge-no-results-recovery">
      <span>{sourceHealth?.loadedAt ?? "No source health"}</span>
      <span>{selectedSources?.join(",") ?? "No selected sources"}</span>
      <button type="button" onClick={onOpenQuickIngest}>
        Open Quick Ingest
      </button>
      {showNearestMatchesAvailable ? (
        <button type="button" onClick={onShowNearestMatches}>
          Show nearest matches
        </button>
      ) : null}
    </div>
  ),
}))

vi.mock("../evidence/EvidenceRail", () => ({
  EvidenceRail: ({
    open,
    onOpenChange,
  }: {
    open: boolean
    onOpenChange: (open: boolean) => void
  }) => (
    <div data-testid={open ? "knowledge-evidence-rail-open" : "knowledge-evidence-rail-closed"}>
      <button type="button" onClick={() => onOpenChange(false)}>
        Close evidence panel
      </button>
      <button type="button" onClick={() => onOpenChange(true)}>
        Open evidence panel
      </button>
    </div>
  ),
}))

describe("KnowledgeQALayout evidence-rail transitions", () => {
  const renderLayout = () => render(<KnowledgeQALayout onExportClick={vi.fn()} />)

  beforeEach(() => {
    vi.clearAllMocks()
    state.settingsPanelOpen = false
    state.results = []
    state.answer = null
    state.citations = []
    state.hasSearched = false
    state.isSearching = false
    state.error = null
    state.queryStage = "idle"
    state.searchDetails = null
    state.preset = "balanced"
    state.settings.sources = []
    state.settings.enable_web_fallback = true
    state.settings.top_k = 10
    state.settings.include_media_ids = []
    state.settings.include_note_ids = []
    state.searchHistory = []
    state.messages = []
    state.evidenceRailOpen = false
    state.evidenceRailTab = "sources"
    state.lastSearchScope = null
    state.pinnedSourceFilters.mediaIds = []
    state.pinnedSourceFilters.noteIds = []
    state.sourceHealth.loadedAt = "2026-05-16T00:00:00Z"
    state.sourceHealth.error = null
    state.refreshSourceHealth.mockClear()
    delete (window as Window & { __tldwPendingQuickIngestOpen?: unknown })
      .__tldwPendingQuickIngestOpen
    layoutModeState.mode = "simple"
    layoutModeState.isSimple = true
    layoutModeState.isResearch = false
  })

  it("uses a visible label for the persistent simple/detailed mode control", () => {
    renderLayout()

    expect(
      screen.getByRole("button", { name: "Switch to detailed view" })
    ).toHaveTextContent("Detailed")
  })

  it("keeps the evidence rail closed while the settings panel is open", async () => {
    state.results = [{ id: "r1" }]
    state.answer = "Answer"
    state.queryStage = "complete"
    state.evidenceRailOpen = true

    const { rerender } = renderLayout()
    expect(await screen.findByTestId("knowledge-evidence-rail-open")).toBeInTheDocument()

    state.settingsPanelOpen = true
    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    await act(async () => {
      await Promise.resolve()
    })

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    await act(async () => {
      await Promise.resolve()
    })

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    expect(await screen.findByTestId("knowledge-evidence-rail-closed")).toBeInTheDocument()
  })

  it("reopens the evidence rail for a new search after a manual close", async () => {
    state.results = [{ id: "r1" }, { id: "r2" }, { id: "r3" }]
    state.answer = "Answer"
    state.queryStage = "complete"
    state.evidenceRailOpen = true
    state.messages = [
      { id: "u1", role: "user", content: "First question" },
      { id: "a1", role: "assistant", content: "First answer" },
    ]

    const { rerender } = renderLayout()
    fireEvent.click(screen.getByRole("button", { name: "Close evidence panel" }))

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)
    expect(await screen.findByTestId("knowledge-evidence-rail-closed")).toBeInTheDocument()

    state.queryStage = "searching"
    state.messages = [
      ...state.messages,
      { id: "u2", role: "user", content: "Follow-up question" },
    ]
    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    await act(async () => {
      await Promise.resolve()
    })

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)
    expect(await screen.findByTestId("knowledge-evidence-rail-closed")).toBeInTheDocument()

    state.results = [{ id: "r4" }, { id: "r5" }, { id: "r6" }]
    state.answer = "Updated answer"
    state.queryStage = "complete"
    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    await act(async () => {
      await Promise.resolve()
    })

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    expect(await screen.findByTestId("knowledge-evidence-rail-open")).toBeInTheDocument()
  })

  it("does not auto-open the evidence rail when fewer than 3 results are returned", async () => {
    state.results = [{ id: "r1" }, { id: "r2" }]
    state.answer = "Short answer"
    state.queryStage = "complete"
    state.evidenceRailOpen = false

    const { rerender } = renderLayout()

    await act(async () => {
      await Promise.resolve()
    })

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    await act(async () => {
      await Promise.resolve()
    })

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    expect(await screen.findByTestId("knowledge-evidence-rail-closed")).toBeInTheDocument()
    expect(state.setEvidenceRailOpen).not.toHaveBeenCalledWith(true)
  })

  it("auto-opens the evidence rail when exactly 3 results are returned", async () => {
    state.results = [{ id: "r1" }, { id: "r2" }, { id: "r3" }]
    state.answer = "Good answer"
    state.queryStage = "complete"
    state.evidenceRailOpen = false

    const { rerender } = renderLayout()

    await act(async () => {
      await Promise.resolve()
    })

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    await act(async () => {
      await Promise.resolve()
    })

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    expect(state.setEvidenceRailOpen).toHaveBeenCalledWith(true)
  })

  it("does not auto-open the evidence rail with only 1 result even when answer exists", async () => {
    state.results = [{ id: "r1" }]
    state.answer = "Single-source answer"
    state.queryStage = "complete"
    state.evidenceRailOpen = false

    const { rerender } = renderLayout()

    await act(async () => {
      await Promise.resolve()
    })

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    await act(async () => {
      await Promise.resolve()
    })

    rerender(<KnowledgeQALayout onExportClick={vi.fn()} />)

    expect(await screen.findByTestId("knowledge-evidence-rail-closed")).toBeInTheDocument()
    expect(state.setEvidenceRailOpen).not.toHaveBeenCalledWith(true)
  })

  it("marks the scope as changed when granular source filters differ from the last search", () => {
    state.settings.sources = ["media_db"]
    state.settings.include_media_ids = [42]
    state.lastSearchScope = {
      preset: "balanced",
      webFallback: true,
      sources: ["media_db"],
      includeMediaIds: [7],
      includeNoteIds: [],
    }

    renderLayout()

    expect(screen.getByText("Scope changed")).toBeInTheDocument()
  })

  it("passes source health and refresh through the simple toolbar and ready state", async () => {
    renderLayout()

    expect(screen.getByTestId("knowledge-compact-toolbar")).toHaveTextContent(
      "2026-05-16T00:00:00Z"
    )
    expect(screen.getByTestId("knowledge-ready-state")).toHaveTextContent(
      "2026-05-16T00:00:00Z"
    )

    fireEvent.click(screen.getByRole("button", { name: "Refresh compact source health" }))

    expect(state.refreshSourceHealth).toHaveBeenCalledOnce()
  })

  it("passes source health and refresh through the detailed context bar", () => {
    layoutModeState.mode = "research"
    layoutModeState.isSimple = false
    layoutModeState.isResearch = true

    renderLayout()

    expect(screen.getByTestId("knowledge-context-bar")).toHaveTextContent(
      "2026-05-16T00:00:00Z"
    )

    fireEvent.click(screen.getByRole("button", { name: "Refresh detailed source health" }))

    expect(state.refreshSourceHealth).toHaveBeenCalledOnce()
  })

  it("passes source health and selected scope into no-results recovery", async () => {
    state.hasSearched = true
    state.settings.sources = ["media_db"]

    renderLayout()

    const recovery = await screen.findByTestId("knowledge-no-results-recovery")
    expect(recovery).toHaveTextContent("2026-05-16T00:00:00Z")
    expect(recovery).toHaveTextContent("media_db")

    fireEvent.click(screen.getByRole("button", { name: "Open Quick Ingest" }))
    expect(
      (window as Window & { __tldwPendingQuickIngestOpen?: unknown })
        .__tldwPendingQuickIngestOpen
    ).toMatchObject({
      detail: { source: "knowledge_qa" },
    })
  })

  it("shows nearest misses in no-results recovery from search metadata", async () => {
    state.hasSearched = true
    state.searchDetails = {
      alsoConsidered: [
        {
          id: "near-1",
          title: "Near miss",
          score: 0.42,
          reason: "below threshold",
        },
      ],
    }

    renderLayout()

    fireEvent.click(await screen.findByRole("button", { name: "Show nearest matches" }))
    expect(state.setEvidenceRailOpen).toHaveBeenCalledWith(true)
    expect(state.setEvidenceRailTab).toHaveBeenCalledWith("details")
    expect(state.focusSource).not.toHaveBeenCalled()
  })
})
