import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { KnowledgeContextBar } from "../context/KnowledgeContextBar"

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

describe("KnowledgeContextBar", () => {
  function createDeferred<T>() {
    let resolve!: (value: T) => void
    let reject!: (reason?: unknown) => void
    const promise = new Promise<T>((res, rej) => {
      resolve = res
      reject = rej
    })
    return { promise, resolve, reject }
  }

  it("shows inline preset descriptions for active mode", () => {
    const { rerender } = render(
      <KnowledgeContextBar
        preset="fast"
        onPresetChange={vi.fn()}
        sources={["media_db"]}
        onSourcesChange={vi.fn()}
        includeMediaIds={[]}
        onIncludeMediaIdsChange={vi.fn()}
        includeNoteIds={[]}
        onIncludeNoteIdsChange={vi.fn()}
        webEnabled={true}
        onToggleWeb={vi.fn()}
        generationProvider={null}
        generationModel={null}
        onGenerationProviderChange={vi.fn()}
        onGenerationModelChange={vi.fn()}
        contextChangedSinceLastRun={false}
        onOpenSettings={vi.fn()}
      />
    )

    expect(
      screen.getByText(
        "Quick lookup with minimal retrieval and rerank depth. Response time: Fastest. Sources checked: Fewer. Best for: Fact checks and quick lookups."
      )
    ).toBeInTheDocument()

    rerender(
      <KnowledgeContextBar
        preset="thorough"
        onPresetChange={vi.fn()}
        sources={["media_db"]}
        onSourcesChange={vi.fn()}
        includeMediaIds={[]}
        onIncludeMediaIdsChange={vi.fn()}
        includeNoteIds={[]}
        onIncludeNoteIdsChange={vi.fn()}
        webEnabled={true}
        onToggleWeb={vi.fn()}
        generationProvider={null}
        generationModel={null}
        onGenerationProviderChange={vi.fn()}
        onGenerationModelChange={vi.fn()}
        contextChangedSinceLastRun={false}
        onOpenSettings={vi.fn()}
      />
    )

    expect(
      screen.getByText(
        "Exhaustive retrieval plus extra verification steps. Response time: Slower. Sources checked: Most thorough. Best for: High-confidence synthesis."
      )
    ).toBeInTheDocument()
  })

  it("lets users update source scope from the dropdown", () => {
    const onSourcesChange = vi.fn()
    render(
      <KnowledgeContextBar
        preset="balanced"
        onPresetChange={vi.fn()}
        sources={[]}
        onSourcesChange={onSourcesChange}
        includeMediaIds={[]}
        onIncludeMediaIdsChange={vi.fn()}
        includeNoteIds={[]}
        onIncludeNoteIdsChange={vi.fn()}
        webEnabled={true}
        onToggleWeb={vi.fn()}
        generationProvider={null}
        generationModel={null}
        onGenerationProviderChange={vi.fn()}
        onGenerationModelChange={vi.fn()}
        contextChangedSinceLastRun={false}
        onOpenSettings={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /Sources:/i }))
    fireEvent.click(screen.getByRole("menuitemcheckbox", { name: /Documents & Media/ }))

    expect(onSourcesChange).toHaveBeenCalledWith(["media_db"])
  })

  it("lets users select granular media sources", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValueOnce({
      items: [{ id: 42, title: "Quarterly Planning Doc", type: "pdf" }],
    })
    vi.mocked(tldwClient.listNotes).mockResolvedValueOnce({ items: [] })

    const onIncludeMediaIdsChange = vi.fn()
    render(
      <KnowledgeContextBar
        preset="balanced"
        onPresetChange={vi.fn()}
        sources={["media_db", "notes"]}
        onSourcesChange={vi.fn()}
        includeMediaIds={[]}
        onIncludeMediaIdsChange={onIncludeMediaIdsChange}
        includeNoteIds={[]}
        onIncludeNoteIdsChange={vi.fn()}
        webEnabled={true}
        onToggleWeb={vi.fn()}
        generationProvider={null}
        generationModel={null}
        onGenerationProviderChange={vi.fn()}
        onGenerationModelChange={vi.fn()}
        contextChangedSinceLastRun={false}
        onOpenSettings={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
    expect(await screen.findByText("Quarterly Planning Doc")).toBeInTheDocument()

    fireEvent.click(screen.getByText("Quarterly Planning Doc"))

    expect(onIncludeMediaIdsChange).toHaveBeenCalledWith([42])
  })

  it("adds Documents & Media to source scope when selecting a granular media filter", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValueOnce({
      items: [{ id: 42, title: "Quarterly Planning Doc", type: "pdf" }],
    })
    vi.mocked(tldwClient.listNotes).mockResolvedValueOnce({ items: [] })

    const onSourcesChange = vi.fn()
    const onIncludeMediaIdsChange = vi.fn()
    render(
      <KnowledgeContextBar
        preset="balanced"
        onPresetChange={vi.fn()}
        sources={[]}
        onSourcesChange={onSourcesChange}
        includeMediaIds={[]}
        onIncludeMediaIdsChange={onIncludeMediaIdsChange}
        includeNoteIds={[]}
        onIncludeNoteIdsChange={vi.fn()}
        webEnabled={true}
        onToggleWeb={vi.fn()}
        generationProvider={null}
        generationModel={null}
        onGenerationProviderChange={vi.fn()}
        onGenerationModelChange={vi.fn()}
        contextChangedSinceLastRun={false}
        onOpenSettings={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
    expect(await screen.findByText("Quarterly Planning Doc")).toBeInTheDocument()

    fireEvent.click(screen.getByText("Quarterly Planning Doc"))

    expect(onSourcesChange).toHaveBeenCalledWith(["media_db"])
    expect(onIncludeMediaIdsChange).toHaveBeenCalledWith([42])
  })

  it("clears granular media filters when Documents & Media is removed from source scope", () => {
    const onSourcesChange = vi.fn()
    const onIncludeMediaIdsChange = vi.fn()

    render(
      <KnowledgeContextBar
        preset="balanced"
        onPresetChange={vi.fn()}
        sources={["media_db"]}
        onSourcesChange={onSourcesChange}
        includeMediaIds={[42]}
        onIncludeMediaIdsChange={onIncludeMediaIdsChange}
        includeNoteIds={[]}
        onIncludeNoteIdsChange={vi.fn()}
        webEnabled={true}
        onToggleWeb={vi.fn()}
        generationProvider={null}
        generationModel={null}
        onGenerationProviderChange={vi.fn()}
        onGenerationModelChange={vi.fn()}
        contextChangedSinceLastRun={false}
        onOpenSettings={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /Sources:/i }))
    fireEvent.click(screen.getByRole("menuitemcheckbox", { name: /Documents & Media/ }))

    expect(onSourcesChange).toHaveBeenCalledWith([])
    expect(onIncludeMediaIdsChange).toHaveBeenCalledWith([])
  })

  it("ignores stale granular load failures after a newer reload succeeds", async () => {
    const firstMediaLoad = createDeferred<{ items: never[] }>()
    const firstNotesLoad = createDeferred<{ items: never[] }>()

    vi.mocked(tldwClient.listMedia)
      .mockReturnValueOnce(firstMediaLoad.promise)
      .mockResolvedValueOnce({
        items: [{ id: 42, title: "Recovered Planning Doc", type: "pdf" }],
      })
    vi.mocked(tldwClient.listNotes)
      .mockReturnValueOnce(firstNotesLoad.promise)
      .mockResolvedValueOnce({ items: [] })

    render(
      <KnowledgeContextBar
        preset="balanced"
        onPresetChange={vi.fn()}
        sources={["media_db"]}
        onSourcesChange={vi.fn()}
        includeMediaIds={[]}
        onIncludeMediaIdsChange={vi.fn()}
        includeNoteIds={[]}
        onIncludeNoteIdsChange={vi.fn()}
        webEnabled={true}
        onToggleWeb={vi.fn()}
        generationProvider={null}
        generationModel={null}
        onGenerationProviderChange={vi.fn()}
        onGenerationModelChange={vi.fn()}
        contextChangedSinceLastRun={false}
        onOpenSettings={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
    expect(screen.getByText("Loading available sources...")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Reload" }))
    expect(await screen.findByText("Recovered Planning Doc")).toBeInTheDocument()

    await act(async () => {
      firstMediaLoad.reject(new Error("First load failed"))
      await Promise.resolve()
    })

    expect(screen.queryByText("First load failed")).not.toBeInTheDocument()
    expect(screen.getByText("Recovered Planning Doc")).toBeInTheDocument()
  })

  it("falls back to category counts when only non-countable sources are selected", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValueOnce({
      items: [{ id: 42, title: "Quarterly Planning Doc", type: "pdf" }],
    })
    vi.mocked(tldwClient.listNotes).mockResolvedValueOnce({
      items: [{ id: "note-1", title: "Meeting Notes" }],
    })

    render(
      <KnowledgeContextBar
        preset="balanced"
        onPresetChange={vi.fn()}
        sources={["characters"]}
        onSourcesChange={vi.fn()}
        includeMediaIds={[]}
        onIncludeMediaIdsChange={vi.fn()}
        includeNoteIds={[]}
        onIncludeNoteIdsChange={vi.fn()}
        webEnabled={true}
        onToggleWeb={vi.fn()}
        generationProvider={null}
        generationModel={null}
        onGenerationProviderChange={vi.fn()}
        onGenerationModelChange={vi.fn()}
        contextChangedSinceLastRun={false}
        onOpenSettings={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
    expect(await screen.findByText("Quarterly Planning Doc")).toBeInTheDocument()

    expect(screen.getByText(/1 of 5 categories selected/)).toBeInTheDocument()
    expect(screen.queryByText(/0 items in scope/)).not.toBeInTheDocument()
  })
})
