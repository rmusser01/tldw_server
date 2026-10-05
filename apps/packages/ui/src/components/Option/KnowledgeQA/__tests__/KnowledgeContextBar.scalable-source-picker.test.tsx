import React from "react"
import {
  act,
  waitFor,
  fireEvent,
  render,
  screen,
  within,
} from "@testing-library/react"
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
    searchMedia: vi.fn(),
    searchNotes: vi.fn(),
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

    fireEvent.click(
      screen.getByRole("button", { name: "Select recent visible" }),
    )
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

function ControlledPicker() {
  const [ids, setIds] = React.useState<number[]>([])
  return (
    <>
      <KnowledgeContextBar
        {...defaultProps}
        includeMediaIds={ids}
        onIncludeMediaIdsChange={setIds}
      />
      <output aria-label="Selected media IDs">{ids.join(",")}</output>
    </>
  )
}

describe("server source selection", () => {
  it("finds page-five titles through server search while retaining a page-one selection and its title", async () => {
    vi.mocked(tldwClient.listMedia).mockImplementation(async (params) => ({
      items: params?.page === 1 ? [{ id: 3, title: "Page one paper" }] : [],
      pagination: { total_items: 250 },
    }))
    vi.mocked(tldwClient.listNotes).mockResolvedValue({ notes: [], total: 0 })
    vi.mocked(tldwClient.searchMedia).mockImplementation(
      async (payload, params) => ({
        items:
          payload.query === "page five" &&
          params?.page === 1 &&
          params?.results_per_page === 50
            ? [{ id: 207, title: "Page five paper" }]
            : [],
        pagination: { total_items: 1 },
      }),
    )
    render(<ControlledPicker />)
    fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
    fireEvent.click(
      await screen.findByRole("checkbox", { name: /Page one paper/ }),
    )
    fireEvent.change(screen.getByPlaceholderText("Filter docs by title"), {
      target: { value: "page five" },
    })
    fireEvent.click(
      await screen.findByRole("checkbox", { name: /Page five paper/ }),
    )
    expect(screen.getByLabelText("Selected media IDs")).toHaveTextContent(
      "3,207",
    )
    const selection = screen.getByRole("region", { name: "Selected sources" })
    expect(within(selection).getByText(/Page one paper/)).toBeVisible()
    expect(within(selection).getByText(/Page five paper/)).toBeVisible()
  })
  it("pages through remote sources without replacing the selection", async () => {
    vi.mocked(tldwClient.listMedia).mockImplementation(async (params) => ({
      items: [
        {
          id: params?.page === 2 ? 51 : 3,
          title: params?.page === 2 ? "Second page" : "First page",
        },
      ],
      pagination: { total_items: 51 },
    }))
    vi.mocked(tldwClient.listNotes).mockResolvedValue({ notes: [], total: 0 })
    render(<ControlledPicker />)
    fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
    fireEvent.click(await screen.findByRole("checkbox", { name: /First page/ }))
    fireEvent.click(screen.getByRole("button", { name: "Next source page" }))
    fireEvent.click(
      await screen.findByRole("checkbox", { name: /Second page/ }),
    )
    expect(screen.getByLabelText("Selected media IDs")).toHaveTextContent(
      "3,51",
    )
    expect(screen.getByText(/^Page 2/)).toBeVisible()
  })
  it("ignores an older server search after a newer search completes", async () => {
    let finish!: (value: unknown) => void
    vi.mocked(tldwClient.listMedia).mockResolvedValue({ items: [] })
    vi.mocked(tldwClient.listNotes).mockResolvedValue({ notes: [], total: 0 })
    vi.mocked(tldwClient.searchMedia).mockImplementation(async (payload) =>
      payload.query === "old"
        ? await new Promise((resolve) => {
            finish = resolve
          })
        : { items: [{ id: 7, title: "Current paper" }] },
    )
    render(<ControlledPicker />)
    fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
    fireEvent.change(screen.getByPlaceholderText("Filter docs by title"), {
      target: { value: "old" },
    })
    await waitFor(() => expect(finish).toBeDefined())
    fireEvent.change(screen.getByPlaceholderText("Filter docs by title"), {
      target: { value: "new" },
    })
    expect(await screen.findByText("Current paper")).toBeVisible()
    await act(async () => {
      finish({ items: [{ id: 3, title: "Stale paper" }] })
      await Promise.resolve()
    })
    expect(screen.queryByText("Stale paper")).not.toBeInTheDocument()
    expect(screen.getByText("Current paper")).toBeVisible()
  })
})

describe("Notes and source boundary selection", () => {
  it("searches notes beyond the first page and retains exact note selections", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [],
      pagination: { total_items: 0 },
    })
    vi.mocked(tldwClient.listNotes).mockResolvedValue({
      notes: [{ id: "note-3", title: "First note" }],
      total: 250,
    })
    vi.mocked(tldwClient.searchMedia).mockResolvedValue({ items: [] })
    vi.mocked(tldwClient.searchNotes).mockImplementation(
      async (query, params) => ({
        notes:
          query === "far note" && params?.limit === 50 && params.offset === 0
            ? [{ id: "note-207", title: "Far note" }]
            : [],
        total: 1,
      }),
    )
    function NotesPicker() {
      const [ids, setIds] = React.useState<string[]>([])
      return (
        <>
          <KnowledgeContextBar
            {...defaultProps}
            includeNoteIds={ids}
            onIncludeNoteIdsChange={setIds}
          />
          <output aria-label="Selected note IDs">{ids.join(",")}</output>
        </>
      )
    }
    render(<NotesPicker />)
    fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
    fireEvent.click(screen.getByRole("button", { name: /^Notes/ }))
    fireEvent.click(await screen.findByRole("checkbox", { name: /First note/ }))
    fireEvent.change(screen.getByPlaceholderText("Filter notes by title"), {
      target: { value: "far note" },
    })
    fireEvent.click(await screen.findByRole("checkbox", { name: /Far note/ }))
    expect(screen.getByLabelText("Selected note IDs")).toHaveTextContent(
      "note-207,note-3",
    )
    expect(
      within(
        screen.getByRole("region", { name: "Selected sources" }),
      ).getByText(/First note/),
    ).toBeVisible()
  })
  it("does not present a loaded page as the library total", async () => {
    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 7, title: "Visible source" }],
    })
    vi.mocked(tldwClient.listNotes).mockResolvedValue({ notes: [] })
    render(<ControlledPicker />)
    fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
    expect(await screen.findByText("Visible source")).toBeVisible()
    expect(screen.queryByText(/1 items in scope/)).toBeNull()
    expect(screen.getByText(/Page 1.*1 loaded/)).toBeVisible()
  })
  it("aborts source discovery and hides late sources when the owner changes", async () => {
    let finish!: (value: unknown) => void
    let signal: AbortSignal | undefined
    vi.mocked(tldwClient.listMedia).mockImplementation(
      async (_params, options) => {
        signal = options?.signal
        return await new Promise((resolve) => {
          finish = resolve
        })
      },
    )
    vi.mocked(tldwClient.listNotes).mockResolvedValue({ notes: [] })
    render(<ControlledPicker />)
    fireEvent.click(screen.getByRole("button", { name: /Specific:/i }))
    await waitFor(() => expect(finish).toBeDefined())
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: true },
        }),
      ),
    )
    expect(signal?.aborted).toBe(true)
    await act(async () => {
      finish({ items: [{ id: 7, title: "Private old source" }] })
      await Promise.resolve()
    })
    expect(screen.queryByText("Private old source")).toBeNull()
    expect(
      screen.queryByRole("dialog", { name: "Specific source selector" }),
    ).toBeNull()
  })
})
