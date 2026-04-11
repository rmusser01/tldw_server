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
  }: {
    contextChangedSinceLastRun: boolean
  }) => (
    <div data-testid="knowledge-context-bar">
      {contextChangedSinceLastRun ? "Scope changed" : "Scope unchanged"}
    </div>
  ),
}))

vi.mock("../context/CompactToolbar", () => ({
  CompactToolbar: ({
    contextChangedSinceLastRun,
  }: {
    contextChangedSinceLastRun: boolean
  }) => (
    <div data-testid="knowledge-compact-toolbar">
      {contextChangedSinceLastRun ? "Scope changed" : "Scope unchanged"}
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
  KnowledgeReadyState: () => <div data-testid="knowledge-ready-state" />,
}))

vi.mock("../empty/InlineRecentSessions", () => ({
  InlineRecentSessions: () => <div data-testid="knowledge-inline-recent-sessions" />,
}))

vi.mock("../panels/AnswerWorkspace", () => ({
  AnswerWorkspace: () => <div data-testid="knowledge-answer-workspace" />,
}))

vi.mock("../panels/NoResultsRecovery", () => ({
  NoResultsRecovery: () => <div data-testid="knowledge-no-results-recovery" />,
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
    layoutModeState.mode = "simple"
    layoutModeState.isSimple = true
    layoutModeState.isResearch = false
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
})
