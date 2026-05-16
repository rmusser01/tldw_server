import { fireEvent, render, screen } from "@testing-library/react"
import type { ComponentProps } from "react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"
import { KnowledgeReadyState } from "../empty/KnowledgeReadyState"
import type { KnowledgeSourceHealthState } from "../types"

type KnowledgeReadyStateTestProps = ComponentProps<typeof KnowledgeReadyState>

const defaultProps: KnowledgeReadyStateTestProps = {
  suggestedPrompts: ["What changed?"],
  onPromptClick: vi.fn(),
  onContinueRecent: vi.fn(),
  onSelectSources: vi.fn(),
  onAddSources: vi.fn(),
  hasSources: true,
  hasRecentSession: false,
  webFallbackEnabled: false,
}

const sourceHealth: KnowledgeSourceHealthState = {
  loading: false,
  error: null,
  loadedAt: "2026-05-16T00:00:00Z",
  sources: [
    {
      sourceId: "media_db",
      label: "Documents & Media",
      available: true,
      searchable: true,
      itemCount: 3,
      indexedCount: 3,
      lastUpdated: null,
      lastIndexed: null,
      indexStatus: "ready",
      embeddingStatus: "not_applicable",
      disabledReason: null,
      workspaceScoped: false,
      hiddenByDefault: false,
      privacyNote: null,
    },
    {
      sourceId: "prompts",
      label: "Prompts",
      available: false,
      searchable: false,
      itemCount: null,
      indexedCount: null,
      lastUpdated: null,
      lastIndexed: null,
      indexStatus: "unavailable",
      embeddingStatus: "unavailable",
      disabledReason: "no_retriever_configured",
      workspaceScoped: false,
      hiddenByDefault: false,
      privacyNote: null,
    },
  ],
  bySource: {
    media_db: {
      sourceId: "media_db",
      label: "Documents & Media",
      available: true,
      searchable: true,
      itemCount: 3,
      indexedCount: 3,
      lastUpdated: null,
      lastIndexed: null,
      indexStatus: "ready",
      embeddingStatus: "not_applicable",
      disabledReason: null,
      workspaceScoped: false,
      hiddenByDefault: false,
      privacyNote: null,
    },
    prompts: {
      sourceId: "prompts",
      label: "Prompts",
      available: false,
      searchable: false,
      itemCount: null,
      indexedCount: null,
      lastUpdated: null,
      lastIndexed: null,
      indexStatus: "unavailable",
      embeddingStatus: "unavailable",
      disabledReason: "no_retriever_configured",
      workspaceScoped: false,
      hiddenByDefault: false,
      privacyNote: null,
    },
  },
}

function renderReadyState(overrides: Partial<KnowledgeReadyStateTestProps> = {}) {
  return render(
    <MemoryRouter>
      <KnowledgeReadyState {...defaultProps} {...overrides} />
    </MemoryRouter>
  )
}

describe("KnowledgeReadyState activation", () => {
  it("frames /knowledge as QA over existing sources and exposes Quick Ingest as the add-source path", () => {
    const onAddSources = vi.fn()
    renderReadyState({ hasSources: false, onAddSources })

    expect(screen.getByText("Ask Your Library")).toBeInTheDocument()
    expect(
      screen.getByText(/This page answers questions over searchable sources/i)
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Add sources" }))
    expect(onAddSources).toHaveBeenCalledOnce()
  })

  it("distinguishes no history from a resumable history state", () => {
    const { rerender } = renderReadyState({ hasRecentSession: false })

    expect(screen.getByText("No previous QA sessions yet.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Continue recent session/i })).toBeDisabled()

    rerender(
      <MemoryRouter>
        <KnowledgeReadyState {...defaultProps} hasRecentSession />
      </MemoryRouter>
    )

    expect(screen.getByText("Recent QA session available.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Continue recent session/i })).not.toBeDisabled()
  })

  it("explains web fallback privacy and default-provider behavior in the empty source state", () => {
    renderReadyState({ hasSources: false, webFallbackEnabled: true })

    expect(
      screen.getByText(/Web fallback uses your configured server default provider/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Queries stay on your tldw server unless web fallback is enabled/i)
    ).toBeInTheDocument()
  })

  it("warns when selected sources are unavailable", () => {
    renderReadyState({
      hasSources: true,
      selectedSources: ["prompts"],
      sourceHealth,
    })

    expect(
      screen.getByText("Selected sources are unavailable. Open source settings or choose a different scope.")
    ).toBeInTheDocument()
  })

  it("keeps search usable when source health fails to load", () => {
    renderReadyState({
      hasSources: true,
      selectedSources: ["media_db"],
      sourceHealth: {
        ...sourceHealth,
        error: "Source health could not be loaded. You can still search selected sources.",
      },
    })

    expect(
      screen.getByText("Source health could not be loaded. You can still search selected sources.")
    ).toBeInTheDocument()
  })

  it("points empty searchable sources back to owner pages instead of inline creation", () => {
    renderReadyState({
      hasSources: true,
      selectedSources: ["notes"],
      sourceHealth: {
        ...sourceHealth,
        sources: [
          {
            sourceId: "notes",
            label: "Notes",
            available: true,
            searchable: false,
            itemCount: 0,
            indexedCount: 0,
            lastUpdated: null,
            lastIndexed: null,
            indexStatus: "empty",
            embeddingStatus: "not_applicable",
            disabledReason: null,
            workspaceScoped: false,
            hiddenByDefault: false,
            privacyNote: null,
          },
        ],
        bySource: {
          notes: {
            sourceId: "notes",
            label: "Notes",
            available: true,
            searchable: false,
            itemCount: 0,
            indexedCount: 0,
            lastUpdated: null,
            lastIndexed: null,
            indexStatus: "empty",
            embeddingStatus: "not_applicable",
            disabledReason: null,
            workspaceScoped: false,
            hiddenByDefault: false,
            privacyNote: null,
          },
        },
      },
    })

    expect(
      screen.getByText("No searchable items yet. Open Quick Ingest or the source owner page to add content.")
    ).toBeInTheDocument()
  })

  it("prefers unavailable-source guidance over empty guidance when unavailable sources report zero items", () => {
    renderReadyState({
      hasSources: true,
      selectedSources: ["prompts"],
      sourceHealth: {
        ...sourceHealth,
        sources: [
          {
            sourceId: "prompts",
            label: "Prompts",
            available: false,
            searchable: false,
            itemCount: 0,
            indexedCount: 0,
            lastUpdated: null,
            lastIndexed: null,
            indexStatus: "unavailable",
            embeddingStatus: "unavailable",
            disabledReason: "no_retriever_configured",
            workspaceScoped: false,
            hiddenByDefault: false,
            privacyNote: null,
          },
        ],
        bySource: {
          prompts: {
            sourceId: "prompts",
            label: "Prompts",
            available: false,
            searchable: false,
            itemCount: 0,
            indexedCount: 0,
            lastUpdated: null,
            lastIndexed: null,
            indexStatus: "unavailable",
            embeddingStatus: "unavailable",
            disabledReason: "no_retriever_configured",
            workspaceScoped: false,
            hiddenByDefault: false,
            privacyNote: null,
          },
        },
      },
    })

    expect(
      screen.getByText("Selected sources are unavailable. Open source settings or choose a different scope.")
    ).toBeInTheDocument()
    expect(
      screen.queryByText("No searchable items yet. Open Quick Ingest or the source owner page to add content.")
    ).not.toBeInTheDocument()
  })

  it("does not classify indexing sources as unavailable", () => {
    renderReadyState({
      hasSources: true,
      selectedSources: ["notes"],
      sourceHealth: {
        ...sourceHealth,
        sources: [
          {
            sourceId: "notes",
            label: "Notes",
            available: true,
            searchable: false,
            itemCount: 4,
            indexedCount: 1,
            lastUpdated: null,
            lastIndexed: null,
            indexStatus: "indexing",
            embeddingStatus: "indexing",
            disabledReason: null,
            workspaceScoped: false,
            hiddenByDefault: false,
            privacyNote: null,
          },
        ],
        bySource: {
          notes: {
            sourceId: "notes",
            label: "Notes",
            available: true,
            searchable: false,
            itemCount: 4,
            indexedCount: 1,
            lastUpdated: null,
            lastIndexed: null,
            indexStatus: "indexing",
            embeddingStatus: "indexing",
            disabledReason: null,
            workspaceScoped: false,
            hiddenByDefault: false,
            privacyNote: null,
          },
        },
      },
    })

    expect(
      screen.queryByText("Selected sources are unavailable. Open source settings or choose a different scope.")
    ).not.toBeInTheDocument()
  })
})
