import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SourceList } from "../SourceList"
import type { RagResult } from "../types"

const defaultResults: RagResult[] = [
  {
    id: "r1",
    content:
      "Full source content line 1.\nFull source content line 2.\nFull source content line 3.",
    metadata: {
      title: "Quarterly Financial Review",
      source_type: "notes",
      page_number: 12,
      url: "https://example.com/source/r1",
    },
    score: 0.92,
  },
]

const state = {
  results: [...defaultResults],
  citations: [{ index: 1 }] as Array<{ index: number }>,
  focusedSourceIndex: null as number | null,
  focusSource: vi.fn(),
  setQuery: vi.fn(),
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    results: state.results,
    citations: state.citations,
    focusedSourceIndex: state.focusedSourceIndex,
    focusSource: state.focusSource,
    setQuery: state.setQuery,
  }),
}))

describe("SourceList full-source viewer", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    state.results = [...defaultResults]
    state.citations = [{ index: 1 }]
    state.focusedSourceIndex = null
    vi.stubGlobal("open", vi.fn())
  })

  it.each([
    { sourceType: undefined, metadataType: undefined, label: "Document" },
    { sourceType: "notes", metadataType: undefined, label: "Note" },
    { sourceType: undefined, metadataType: "web", label: "Web" },
    { sourceType: "notes", metadataType: "web", label: "Note" }
  ] as const)(
    "keeps the source card and preview type consistent for $sourceType / $metadataType",
    async ({ sourceType, metadataType, label }) => {
      state.results = [{
        id: "source-1",
        content: "Synthetic source text.",
        score: 1,
        sourceType,
        metadata: { title: "Typed source", source_type: metadataType }
      }]
      render(<SourceList />)
      expect(screen.getByText(label, { exact: true })).toBeInTheDocument()
      fireEvent.click(screen.getByRole("button", { name: "View source 1" }))
      const dialog = await screen.findByRole("dialog")
      expect(within(dialog).getByText(label, { exact: true })).toBeInTheDocument()
    }
  )

  it("opens an uploaded media source in Media instead of resolving its filename as a route", async () => {
    state.results = [{
      id: "r1", sourceId: "1", sourceType: "media_db", content: "Project Aster source.", score: 1,
      metadata: { title: "Aster", source_type: "media_db", url: "full-single-uat-study.txt" }
    }]
    render(<SourceList />)
    fireEvent.click(screen.getByRole("button", { name: "View source 1" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Open in Media" }))
    expect(window.open).toHaveBeenCalledWith("/media?id=1", "_blank", "noopener,noreferrer")
  })

  it("opens and closes full source preview modal from source actions", async () => {
    render(<SourceList />)

    fireEvent.click(screen.getByRole("button", { name: "View source 1" }))

    const dialog = await screen.findByRole("dialog", {
      name: /Source 1: Quarterly Financial Review/i,
    })
    expect(dialog).toBeInTheDocument()
    expect(
      within(dialog).getByText(/Full source content line 2\./)
    ).toBeInTheDocument()
    expect(within(dialog).getByText(/Note • Page 12/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Close source preview" }))

    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: /Source 1: Quarterly Financial Review/i })
      ).not.toBeInTheDocument()
    )
  })

  it("shows materialized excerpt when full source content is unavailable", async () => {
    state.results = [
      {
        id: "r2",
        content: "",
        excerpt: "Matched excerpt remains inspectable in the viewer.",
        sourceId: "note-7",
        chunkId: "chunk-4",
        evidenceOrigin: "local_library",
        metadata: {
          title: "Excerpt-only note",
          source_type: "notes",
        },
        score: 0.77,
      },
    ]
    state.citations = []

    render(<SourceList />)

    fireEvent.click(screen.getByRole("button", { name: "View source 1" }))

    const dialog = await screen.findByRole("dialog", {
      name: /Source 1: Excerpt-only note/i,
    })
    expect(
      within(dialog).getByText(/Matched excerpt remains inspectable in the viewer\./)
    ).toBeInTheDocument()
    expect(within(dialog).queryByText(/Full source content is unavailable/i)).not.toBeInTheDocument()
  })

  it("shows specific unavailable reason in the source viewer", async () => {
    state.results = [
      {
        id: "r3",
        content: "",
        sourceId: "99",
        chunkId: "2",
        sourceStatus: "unavailable",
        unavailableReason: "deleted_or_unavailable",
        evidenceOrigin: "local_library",
        metadata: {
          title: "Deleted source",
          source_type: "media_db",
        },
        score: 0,
      },
    ]
    state.citations = []

    render(<SourceList />)

    fireEvent.click(screen.getByRole("button", { name: "View source 1" }))

    const dialog = await screen.findByRole("dialog", {
      name: /Source 1: Deleted source/i,
    })
    expect(within(dialog).getByText(/Source unavailable: Deleted or unavailable/i)).toBeInTheDocument()
  })
})
