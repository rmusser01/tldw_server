import { fireEvent, render, screen } from "@testing-library/react"
import type { ComponentProps } from "react"
import { describe, expect, it, vi } from "vitest"
import { KnowledgeContextBar } from "../context/KnowledgeContextBar"
import type { KnowledgeSourceHealthState } from "../types"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    getProviders: vi.fn().mockResolvedValue({
      default_provider: "openai",
      providers: [
        { name: "openai", display_name: "OpenAI", models: ["gpt-4o-mini"] },
      ],
    }),
    listMedia: vi.fn().mockResolvedValue({ items: [] }),
    listNotes: vi.fn().mockResolvedValue({ items: [] }),
  },
}))

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
      itemCount: null,
      indexedCount: null,
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
      itemCount: null,
      indexedCount: null,
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

type KnowledgeContextBarTestProps = ComponentProps<typeof KnowledgeContextBar>

const baseProps: KnowledgeContextBarTestProps = {
  preset: "balanced" as const,
  onPresetChange: vi.fn(),
  sources: ["media_db", "prompts"],
  onSourcesChange: vi.fn(),
  includeMediaIds: [] as number[],
  onIncludeMediaIdsChange: vi.fn(),
  includeNoteIds: [] as string[],
  onIncludeNoteIdsChange: vi.fn(),
  webEnabled: false,
  onToggleWeb: vi.fn(),
  generationProvider: null as string | null,
  generationModel: null as string | null,
  onGenerationProviderChange: vi.fn(),
  onGenerationModelChange: vi.fn(),
  contextChangedSinceLastRun: false,
  onOpenSettings: vi.fn(),
}

describe("KnowledgeContextBar source health", () => {
  it("shows aggregate source health and per-source status", () => {
    render(
      <KnowledgeContextBar
        {...baseProps}
        sourceHealth={sourceHealth}
        onRefreshSourceHealth={vi.fn()}
      />
    )

    expect(screen.getByText("Sources ready: 1 of 2")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /Sources:/i }))
    expect(screen.getByText("Ready")).toBeInTheDocument()
    expect(screen.getByText("Unavailable")).toBeInTheDocument()
  })

  it("lets users retry source health from the source picker", () => {
    const onRefreshSourceHealth = vi.fn()
    render(
      <KnowledgeContextBar
        {...baseProps}
        sourceHealth={{ ...sourceHealth, error: "Source health could not be loaded." }}
        onRefreshSourceHealth={onRefreshSourceHealth}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /Sources:/i }))
    fireEvent.click(screen.getByRole("button", { name: "Refresh source health" }))

    expect(onRefreshSourceHealth).toHaveBeenCalledOnce()
  })

  it("disables source health refresh while a refresh is loading", () => {
    const onRefreshSourceHealth = vi.fn()
    render(
      <KnowledgeContextBar
        {...baseProps}
        sourceHealth={{ ...sourceHealth, loading: true }}
        onRefreshSourceHealth={onRefreshSourceHealth}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /Sources:/i }))
    const refreshButton = screen.getByRole("button", { name: "Refresh source health" })
    expect(refreshButton).toBeDisabled()
    expect(refreshButton).toHaveAttribute("aria-busy", "true")

    fireEvent.click(refreshButton)
    expect(onRefreshSourceHealth).not.toHaveBeenCalled()
  })
})
