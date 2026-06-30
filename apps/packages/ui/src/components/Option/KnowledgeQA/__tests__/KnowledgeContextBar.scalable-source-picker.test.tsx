import { fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { RagSource } from "@/services/rag/unified-rag"
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
    listMedia: vi.fn(),
    listNotes: vi.fn(),
  },
}))

const mediaItems = [
  {
    id: 42,
    title: "Project Brief",
    type: "pdf",
    status: "ready",
    recently_imported: true,
  },
  {
    id: 43,
    title: "Indexing Transcript",
    type: "audio",
    status: "indexing",
  },
  {
    id: 44,
    title: "Generated Fixture",
    type: "markdown",
    status: "ready",
    is_generated: true,
  },
  {
    id: 45,
    title: "Workspace Scratch",
    type: "markdown",
    status: "ready",
    workspace_artifact: true,
    workspace_id: "ws-1",
    workspace_name: "Project Phoenix",
  },
]

const defaultProps = {
  preset: "balanced" as const,
  onPresetChange: vi.fn(),
  sources: ["media_db"] as RagSource[],
  onSourcesChange: vi.fn(),
  includeMediaIds: [] as number[],
  onIncludeMediaIdsChange: vi.fn(),
  includeNoteIds: [] as string[],
  onIncludeNoteIdsChange: vi.fn(),
  webEnabled: false,
  onToggleWeb: vi.fn(),
  generationProvider: null,
  generationModel: null,
  onGenerationProviderChange: vi.fn(),
  onGenerationModelChange: vi.fn(),
  contextChangedSinceLastRun: false,
  onOpenSettings: vi.fn(),
}

function renderContextBar(overrides: Partial<typeof defaultProps> = {}) {
  vi.mocked(tldwClient.listMedia).mockResolvedValueOnce({ items: mediaItems })
  vi.mocked(tldwClient.listNotes).mockResolvedValueOnce({
    items: [
      {
        id: "note-1",
        title: "Research Note",
        status: "ready",
        recently_imported: true,
      },
    ],
  })

  return render(<KnowledgeContextBar {...defaultProps} {...overrides} />)
}

async function openSpecificSources() {
  fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
  expect(await screen.findByText("Project Brief")).toBeInTheDocument()
}

describe("KnowledgeContextBar scalable source picker", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("filters specific sources by status, recent imports, and explicit workspace scope", async () => {
    renderContextBar()
    await openSpecificSources()

    expect(screen.getByPlaceholderText("Filter docs by title")).toHaveFocus()
    expect(screen.queryByText("Generated Fixture")).not.toBeInTheDocument()
    expect(screen.queryByText("Workspace Scratch")).not.toBeInTheDocument()
    expect(screen.getByText(/ID: 42/)).toBeInTheDocument()
    expect(screen.getByText(/pdf.*ready/i)).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("Source status"), {
      target: { value: "indexing" },
    })
    expect(screen.getByText("Indexing Transcript")).toBeInTheDocument()
    expect(screen.queryByText("Project Brief")).not.toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("Source status"), {
      target: { value: "all" },
    })
    fireEvent.change(screen.getByLabelText("Recent imports"), {
      target: { value: "recent" },
    })
    expect(screen.getByText("Project Brief")).toBeInTheDocument()
    expect(screen.queryByText("Indexing Transcript")).not.toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("Recent imports"), {
      target: { value: "all" },
    })
    fireEvent.change(screen.getByLabelText("Workspace scope"), {
      target: { value: "ws-1" },
    })
    expect(screen.getByText("Workspace Scratch")).toBeInTheDocument()
    expect(screen.queryByText("Generated Fixture")).not.toBeInTheDocument()
  })

  it("supports keyboard switching between document and note source groups", async () => {
    renderContextBar()
    await openSpecificSources()

    const dialog = screen.getByRole("dialog", { name: "Specific source selector" })
    fireEvent.keyDown(dialog, {
      key: "]",
    })

    expect(within(dialog).getByRole("button", { name: /Notes/ })).toHaveClass("bg-primary")
    expect(screen.getByPlaceholderText("Filter notes by title")).toHaveFocus()

    fireEvent.keyDown(dialog, {
      key: "[",
    })

    expect(within(dialog).getByRole("button", { name: /Documents & Media/ })).toHaveClass("bg-primary")
  })

  it("supports selecting visible sources, clearing visible sources, and selecting recent imports", async () => {
    const onSourcesChange = vi.fn()
    const onIncludeMediaIdsChange = vi.fn()
    renderContextBar({
      sources: [],
      onSourcesChange,
      onIncludeMediaIdsChange,
    })
    await openSpecificSources()

    fireEvent.click(screen.getByRole("button", { name: "Select visible" }))
    expect(onSourcesChange).toHaveBeenCalledWith(["media_db"])
    expect(onIncludeMediaIdsChange).toHaveBeenCalledWith([42, 43])

    fireEvent.click(screen.getByRole("button", { name: "Select recent imports" }))
    expect(onIncludeMediaIdsChange).toHaveBeenLastCalledWith([42])
  })

  it("clears only currently visible selections", async () => {
    const onIncludeMediaIdsChange = vi.fn()
    renderContextBar({
      includeMediaIds: [42, 43, 99],
      onIncludeMediaIdsChange,
    })
    await openSpecificSources()

    fireEvent.click(screen.getByRole("button", { name: "Clear visible" }))
    expect(onIncludeMediaIdsChange).toHaveBeenCalledWith([99])
  })
})
