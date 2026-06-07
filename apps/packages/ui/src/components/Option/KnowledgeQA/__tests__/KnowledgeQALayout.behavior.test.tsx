import React from "react"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQALayout } from "../layout/KnowledgeQALayout"
import {
  createKnowledgeQaStateFixture,
  type KnowledgeQaStateFixtureName,
} from "./knowledgeQaStateFixtures"

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

const layoutProps = {
  knowledgeStatus: "ready" as "unknown" | "ready" | "indexing" | "offline" | "empty",
  webFallbackAvailable: true,
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
    onSourcesChange,
    onIncludeMediaIdsChange,
    onIncludeNoteIdsChange,
    onPresetChange,
    onToggleWeb,
    onOpenSettings,
  }: {
    contextChangedSinceLastRun: boolean
    onSourcesChange: (sources: string[]) => void
    onIncludeMediaIdsChange: (ids: number[]) => void
    onIncludeNoteIdsChange: (ids: string[]) => void
    onPresetChange: (preset: string) => void
    onToggleWeb: () => void
    onOpenSettings: () => void
  }) => (
    <div data-testid="knowledge-context-bar">
      <p>Source Scope</p>
      <button type="button" onClick={() => onSourcesChange(["media_db", "notes"])}>
        Select documents and notes
      </button>
      <button type="button" onClick={() => onIncludeMediaIdsChange([42])}>
        Select exact document
      </button>
      <button type="button" onClick={() => onIncludeNoteIdsChange(["note-a"])}>
        Select exact note
      </button>
      <button type="button" onClick={() => onPresetChange("thorough")}>
        Use deep preset
      </button>
      <button type="button" onClick={onToggleWeb}>
        Toggle web fallback
      </button>
      <button type="button">Profiles</button>
      <button
        type="button"
        onClick={() => {
          onSourcesChange(["media_db", "notes"])
          onIncludeMediaIdsChange([7])
          onIncludeNoteIdsChange(["note-b"])
          onPresetChange("fast")
        }}
      >
        Load saved profile
      </button>
      <button type="button" onClick={onOpenSettings}>
        Advanced settings
      </button>
      {contextChangedSinceLastRun ? "Scope changed" : "Scope unchanged"}
    </div>
  ),
}))

vi.mock("../context/CompactToolbar", () => ({
  CompactToolbar: ({
    contextChangedSinceLastRun,
    onOpenSourceSelector,
  }: {
    contextChangedSinceLastRun: boolean
    onOpenSourceSelector: () => void
  }) => (
    <div data-testid="knowledge-compact-toolbar">
      <button type="button" onClick={onOpenSourceSelector}>
        Open compact sources
      </button>
      {contextChangedSinceLastRun ? "Scope changed" : "Scope unchanged"}
    </div>
  ),
}))

vi.mock("../composer/KnowledgeComposer", () => ({
  KnowledgeComposer: ({
    searchBlockedMessage,
  }: {
    searchBlockedMessage?: string | null
  }) => (
    <div
      data-testid="knowledge-composer"
      data-search-blocked-message={searchBlockedMessage ?? ""}
    >
      <input
        id="knowledge-search-input"
        aria-label="Search your knowledge base"
      />
    </div>
  ),
}))

vi.mock("../empty/KnowledgeReadyState", () => ({
  KnowledgeReadyState: ({
    recoveryState,
    hasSources,
    webFallbackEnabled,
  }: {
    recoveryState?: { kind: string }
    hasSources: boolean
    webFallbackEnabled: boolean
  }) => (
    <div data-testid="knowledge-ready-state">
      {recoveryState
        ? `knowledge-ready-recovery:${recoveryState.kind}`
        : hasSources
        ? "Ready with selected sources"
        : webFallbackEnabled
          ? "Ready with web fallback only"
          : "No sources selected"}
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
    onShowNearestMatches,
  }: {
    onShowNearestMatches: () => void
  }) => (
    <div data-testid="knowledge-no-results-recovery">
      <button type="button" onClick={onShowNearestMatches}>
        Show nearest matches
      </button>
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
  const renderLayoutElement = () => (
    <KnowledgeQALayout
      onExportClick={vi.fn()}
      knowledgeStatus={layoutProps.knowledgeStatus}
      webFallbackAvailable={layoutProps.webFallbackAvailable}
    />
  )
  const renderLayout = () => render(renderLayoutElement())
  const applyStateFixture = (name: KnowledgeQaStateFixtureName) => {
    const fixture = createKnowledgeQaStateFixture(name).knowledgeQa
    const fullFixture = createKnowledgeQaStateFixture(name)
    layoutProps.knowledgeStatus =
      fullFixture.sourceInventory.media.length === 0 &&
      fullFixture.sourceInventory.notes.length === 0
        ? "empty"
        : "ready"
    layoutProps.webFallbackAvailable =
      fullFixture.capabilities.capabilities?.hasWebSearch !== false
    state.settingsPanelOpen = fixture.settingsPanelOpen
    state.results = fixture.results
    state.answer = fixture.answer
    state.citations = fixture.citations
    state.hasSearched = fixture.hasSearched
    state.isSearching = fixture.isSearching
    state.error = fixture.error
    state.queryStage = fixture.queryStage
    state.preset = fixture.preset
    state.settings = {
      sources: fixture.settings.sources,
      enable_web_fallback: fixture.settings.enable_web_fallback,
      top_k: fixture.settings.top_k,
      include_media_ids: fixture.settings.include_media_ids,
      include_note_ids: fixture.settings.include_note_ids,
    }
    state.searchHistory = fixture.searchHistory
    state.messages = fixture.messages
    state.evidenceRailOpen = fixture.evidenceRailOpen
    state.evidenceRailTab = fixture.evidenceRailTab
    state.lastSearchScope = fixture.lastSearchScope
    state.pinnedSourceFilters = fixture.pinnedSourceFilters
  }

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
    layoutProps.knowledgeStatus = "ready"
    layoutProps.webFallbackAvailable = true
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

  it.each([
    ["noIndexedSources", "knowledge-ready-state", "knowledge-ready-recovery:no_indexed_sources"],
    ["readySearch", "knowledge-ready-state", "knowledge-ready-recovery:ready"],
    ["noSelectedSources", "knowledge-ready-state", "knowledge-ready-recovery:no_selected_sources"],
    ["results", "knowledge-results-shell", "knowledge-answer-workspace"],
    ["noResults", "knowledge-results-shell", "knowledge-no-results-recovery"],
  ] as const)(
    "renders the %s audited state from a deterministic fixture",
    async (fixtureName, testId, expectedMarker) => {
      applyStateFixture(fixtureName)

      renderLayout()

      const element = await screen.findByTestId(testId)
      expect(element).toBeInTheDocument()
      if (expectedMarker.startsWith("knowledge-")) {
        if (expectedMarker.startsWith("knowledge-ready-recovery:")) {
          expect(screen.getByText(expectedMarker)).toBeInTheDocument()
        } else {
          expect(await screen.findByTestId(expectedMarker)).toBeInTheDocument()
        }
      } else {
        expect(screen.getByText(expectedMarker)).toBeInTheDocument()
      }
    }
  )

  it("classifies no selected sources as blocked when web fallback is unavailable", async () => {
    applyStateFixture("noSelectedSources")
    state.settings.enable_web_fallback = true
    layoutProps.webFallbackAvailable = false

    renderLayout()

    expect(await screen.findByTestId("knowledge-ready-state")).toHaveTextContent(
      "knowledge-ready-recovery:no_selected_sources"
    )
  })

  it("classifies no selected sources as web-only when web fallback is enabled and available", async () => {
    applyStateFixture("noSelectedSources")
    state.settings.enable_web_fallback = true
    layoutProps.webFallbackAvailable = true

    renderLayout()

    expect(await screen.findByTestId("knowledge-ready-state")).toHaveTextContent(
      "knowledge-ready-recovery:web_only"
    )
  })

  it("opens the details evidence view when showing nearest no-results matches", async () => {
    applyStateFixture("noResults")
    state.evidenceRailOpen = false
    state.evidenceRailTab = "sources"

    renderLayout()

    fireEvent.click(await screen.findByRole("button", { name: "Show nearest matches" }))

    expect(state.setEvidenceRailOpen).toHaveBeenCalledWith(true)
    expect(state.setEvidenceRailTab).toHaveBeenCalledWith("details")
    expect(state.focusSource).not.toHaveBeenCalled()
  })

  it("passes a visible no-indexed-source block reason to the composer", async () => {
    applyStateFixture("noIndexedSources")

    renderLayout()

    expect(await screen.findByTestId("knowledge-composer")).toHaveAttribute(
      "data-search-blocked-message",
      "Add or index library sources before asking Knowledge QA."
    )
  })

  it("opens shared source scope and profile controls from the compact source action", async () => {
    renderLayout()

    fireEvent.click(screen.getByRole("button", { name: "Open compact sources" }))

    const dialog = await screen.findByRole("dialog", {
      name: "Source scope and profiles",
    })
    expect(within(dialog).getByText("Source Scope")).toBeInTheDocument()
    expect(within(dialog).getByRole("button", { name: "Profiles" })).toBeInTheDocument()

    fireEvent.click(within(dialog).getByRole("button", { name: "Select documents and notes" }))
    expect(state.updateSetting).toHaveBeenCalledWith("sources", ["media_db", "notes"])

    fireEvent.click(within(dialog).getByRole("button", { name: "Select exact document" }))
    expect(state.updateSetting).toHaveBeenCalledWith("include_media_ids", [42])

    fireEvent.click(within(dialog).getByRole("button", { name: "Select exact note" }))
    expect(state.updateSetting).toHaveBeenCalledWith("include_note_ids", ["note-a"])

    fireEvent.click(within(dialog).getByRole("button", { name: "Use deep preset" }))
    expect(state.setPreset).toHaveBeenCalledWith("thorough")

    fireEvent.click(within(dialog).getByRole("button", { name: "Toggle web fallback" }))
    expect(state.updateSetting).toHaveBeenCalledWith("enable_web_fallback", false)

    fireEvent.click(within(dialog).getByRole("button", { name: "Advanced settings" }))
    expect(state.setSettingsPanelOpen).toHaveBeenCalledWith(true)
  })

  it("restores saved profile scope from the compact source dialog", async () => {
    renderLayout()

    fireEvent.click(screen.getByRole("button", { name: "Open compact sources" }))

    const dialog = await screen.findByRole("dialog", {
      name: "Source scope and profiles",
    })
    fireEvent.click(within(dialog).getByRole("button", { name: "Load saved profile" }))

    expect(state.updateSetting).toHaveBeenCalledWith("sources", [
      "media_db",
      "notes",
    ])
    expect(state.updateSetting).toHaveBeenCalledWith("include_media_ids", [7])
    expect(state.updateSetting).toHaveBeenCalledWith("include_note_ids", [
      "note-b",
    ])
    expect(state.setPreset).toHaveBeenCalledWith("fast")
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
