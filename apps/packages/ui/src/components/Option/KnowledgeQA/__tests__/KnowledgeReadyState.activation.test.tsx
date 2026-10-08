import React, { useState } from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import type { ComponentProps } from "react"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"
import { KnowledgeReadyState } from "../empty/KnowledgeReadyState"
import type { KnowledgeSourceHealthState } from "../types"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) =>
      options?.defaultValue ?? key,
  }),
}))

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
    const guide = screen.getByRole("button", { name: "How it works" })
    expect(guide).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(guide)
    expect(screen.getByText(/This page answers questions over searchable sources/i)).toBeInTheDocument()

    fireEvent.click(screen.getAllByRole("button", { name: "Add sources" })[0])
    expect(onAddSources).toHaveBeenCalledOnce()
  })

  it("places Add and Ask before optional guidance and recipes", () => {
    renderReadyState({ children: <button>Ask</button> })
    const add = screen.getByRole("button", { name: "Add sources" })
    const ask = screen.getByRole("button", { name: "Ask" })
    const guide = screen.getByRole("button", { name: "How it works" })
    expect(add.compareDocumentPosition(ask) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(ask.compareDocumentPosition(guide) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(guide).toHaveAttribute("aria-expanded", "false")
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

function RecipeComposer() {
  const [query, setQuery] = useState("")
  const [answer, setAnswer] = useState("")
  const [scope, setScope] = useState("notes:note-uuid")
  return (
    <MemoryRouter>
      <KnowledgeReadyState
        {...defaultProps}
        onPromptClick={setQuery}
        onSelectSources={() => setScope("all")}
        selectedSources={["notes"]}
      />
      <textarea
        aria-label="Question"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
      />
      <button onClick={() => setAnswer(query)}>Ask</button>
      <output aria-label="Answer">{answer}</output>
      <output aria-label="Scope">{scope}</output>
    </MemoryRouter>
  )
}

describe("evidence-aware recipes", () => {
  it.each([
    "Compare these papers",
    "Extract claims with evidence",
    "Summarize this interview",
    "Save a sourced brief",
  ])(
    "populates an editable %s question without asking or changing scope",
    (label) => {
      render(<RecipeComposer />)
      fireEvent.click(screen.getByRole("button", { name: label }))
      expect(
        (screen.getByLabelText("Question") as HTMLTextAreaElement).value,
      ).toMatch(/evidence|citations/i)
      expect(screen.getByLabelText("Answer")).toHaveTextContent("")
      expect(screen.getByLabelText("Scope")).toHaveTextContent(
        "notes:note-uuid",
      )
      fireEvent.change(screen.getByLabelText("Question"), {
        target: { value: "My edited question" },
      })
      fireEvent.click(screen.getByRole("button", { name: "Ask" }))
      expect(screen.getByLabelText("Answer")).toHaveTextContent(
        "My edited question",
      )
    },
  )
})

it("offers first-add for connected services with an empty personal library", () => {
  const onAdd = vi.fn()
  render(
    <MemoryRouter>
      <KnowledgeReadyState
        suggestedPrompts={[]}
        onPromptClick={vi.fn()}
        onContinueRecent={vi.fn()}
        onSelectSources={vi.fn()}
        onAddSources={onAdd}
        hasSources
        hasRecentSession={false}
        selectedSources={["media_db", "notes"]}
        sourceHealth={
          {
            loading: false,
            error: null,
            loadedAt: null,
            bySource: {
              media_db: { ...sourceHealth.bySource.media_db, itemCount: 0 },
              notes: {
                ...sourceHealth.bySource.media_db,
                sourceId: "notes",
                itemCount: 0,
              },
            },
            sources: [],
          } satisfies KnowledgeSourceHealthState
        }
      />
    </MemoryRouter>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Add your first source" }))
  expect(onAdd).toHaveBeenCalledOnce()
})
