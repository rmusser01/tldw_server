import React from "react"
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"

const mockUseKnowledgeSettings = vi.fn()
const mockUseKnowledgeSearch = vi.fn()
const mockUseFileSearch = vi.fn()
const mockUseQASearch = vi.fn()
const mockWithFullMediaTextIfAvailable = vi.fn()
let KnowledgePanel: typeof import("../KnowledgePanel").KnowledgePanel

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock(
  "antd",
  () => ({
    Modal: Object.assign(
      ({
        children,
        open,
        title
      }: {
        children?: React.ReactNode
        open?: boolean
        title?: React.ReactNode
      }) =>
        open ? (
          <div
            role="dialog"
            aria-label={typeof title === "string" ? title : undefined}
          >
            {title ? <h2>{title}</h2> : null}
            {children}
          </div>
        ) : null,
      { confirm: vi.fn() }
    )
  }),
  { virtual: true }
)

vi.mock("../hooks", () => {
  const toPinnedResult = (item: any) => ({
    id: `pin-${item?.metadata?.id ?? item?.metadata?.title ?? "1"}`,
    title:
      typeof item?.metadata?.title === "string"
        ? item.metadata.title
        : undefined,
    source:
      typeof item?.metadata?.source === "string"
        ? item.metadata.source
        : undefined,
    url:
      typeof item?.metadata?.url === "string"
        ? item.metadata.url
        : undefined,
    snippet: String(item?.content || item?.text || item?.chunk || ""),
    type:
      typeof item?.metadata?.type === "string"
        ? item.metadata.type
        : undefined,
    mediaId:
      typeof item?.metadata?.media_id === "number"
        ? item.metadata.media_id
        : undefined
  })

  return {
    useKnowledgeSettings: (...args: unknown[]) =>
      mockUseKnowledgeSettings(...args),
    useKnowledgeSearch: (...args: unknown[]) => mockUseKnowledgeSearch(...args),
    toPinnedResult,
    withFullMediaTextIfAvailable: (...args: unknown[]) =>
      mockWithFullMediaTextIfAvailable(...args),
    qaDocumentToRagResult: (doc: any) => ({
      content: doc.content || doc.text || doc.chunk || "",
      metadata: {
        ...(doc.metadata || {}),
        ...(doc.id !== undefined ? { id: doc.id } : {}),
        ...(doc.media_id !== undefined ? { media_id: doc.media_id } : {})
      },
      score: doc.score,
      relevance: doc.relevance
    })
  }
})

vi.mock("../hooks/useFileSearch", () => ({
  useFileSearch: (...args: unknown[]) => mockUseFileSearch(...args)
}))

vi.mock("../hooks/useQASearch", () => ({
  useQASearch: (...args: unknown[]) => mockUseQASearch(...args)
}))

describe("KnowledgePanel QA chunk preview", () => {
  const qaDocument = {
    id: "doc-1",
    content: "Chunk preview text",
    metadata: { title: "QA Doc Title", source: "qa-source" },
    score: 0.8
  }

  beforeAll(async () => {
    ;({ KnowledgePanel } = await import("../KnowledgePanel"))
  })

  beforeEach(() => {
    vi.clearAllMocks()

    mockUseKnowledgeSettings.mockReturnValue({
      preset: "balanced",
      draftSettings: {
        query: "What does the document say?",
        sources: ["media_db"],
        strategy: "standard"
      },
      storedSettings: {},
      useCurrentMessage: false,
      advancedOpen: false,
      advancedSearch: "",
      resolvedQuery: "What does the document say?",
      isDirty: false,
      updateSetting: vi.fn(),
      applyPreset: vi.fn(),
      applySettings: vi.fn(),
      resetToBalanced: vi.fn(),
      setUseCurrentMessage: vi.fn(),
      setAdvancedOpen: vi.fn(),
      setAdvancedSearch: vi.fn(),
      discardChanges: vi.fn()
    })

    mockUseKnowledgeSearch.mockReturnValue({
      pinnedResults: [],
      handlePin: vi.fn(),
      handleUnpin: vi.fn(),
      handleClearPins: vi.fn()
    })

    mockUseFileSearch.mockReturnValue({
      loading: false,
      results: [],
      sortMode: "relevance",
      timedOut: false,
      hasAttemptedSearch: false,
      queryError: null,
      mediaTypes: [],
      setMediaTypes: vi.fn(),
      attachedMediaIds: new Set<number>(),
      runSearch: vi.fn(),
      setSortMode: vi.fn(),
      sortResults: (items: unknown[]) => items,
      handleAttach: vi.fn(),
      handlePreview: vi.fn(),
      handleOpen: vi.fn(),
      handlePin: vi.fn(),
      copyResult: vi.fn()
    })

    mockUseQASearch.mockReturnValue({
      loading: false,
      response: {
        generatedAnswer: null,
        documents: [qaDocument],
        citations: [],
        academicCitations: [],
        timings: {},
        totalTime: 0,
        cacheHit: false,
        feedbackId: null,
        errors: [],
        query: "What does the document say?",
        expandedQueries: []
      },
      timedOut: false,
      hasAttemptedSearch: true,
      queryError: null,
      runQASearch: vi.fn(),
      copyAnswer: vi.fn(),
      insertAnswer: vi.fn(),
      insertChunk: vi.fn(),
      copyChunk: vi.fn(),
      pinChunk: vi.fn()
    })

    mockWithFullMediaTextIfAvailable.mockImplementation(async (pinned: unknown) => pinned)
  })

  it("opens shared preview modal from QA chunk preview action", async () => {
    render(
      <KnowledgePanel
        open
        showToggle={false}
        onInsert={vi.fn()}
        onAsk={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))

    const dialog = await screen.findByRole("dialog")
    expect(dialog).toBeInTheDocument()
    expect(within(dialog).getByText("QA Doc Title")).toBeInTheDocument()
    expect(within(dialog).getByText("Chunk preview text")).toBeInTheDocument()
  })

  it("supports Insert and Ask actions from shared preview modal for QA chunks", async () => {
    const onInsert = vi.fn()
    const onAsk = vi.fn()

    render(
      <KnowledgePanel
        open
        showToggle={false}
        onInsert={onInsert}
        onAsk={onAsk}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Insert" }))

    await waitFor(() => {
      expect(onInsert).toHaveBeenCalledWith(
        expect.stringContaining("**QA Doc Title**")
      )
    })
    expect(onInsert).toHaveBeenCalledWith(
      expect.stringContaining("Chunk preview text")
    )

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    const dialogAfterReopen = await screen.findByRole("dialog")
    fireEvent.click(within(dialogAfterReopen).getByRole("button", { name: "Ask" }))

    await waitFor(() => {
      expect(onAsk).toHaveBeenCalledWith(
        expect.stringContaining("**QA Doc Title**"),
        { ignorePinnedResults: true }
      )
    })
  })

  it("resolves shared preview modal context before Ask", async () => {
    const onAsk = vi.fn()
    mockWithFullMediaTextIfAvailable.mockResolvedValueOnce({
      id: "pin-doc-1",
      title: "QA Doc Title",
      snippet: "Resolved preview context"
    })

    render(
      <KnowledgePanel
        open
        showToggle={false}
        onInsert={vi.fn()}
        onAsk={onAsk}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Ask" }))

    await waitFor(() => {
      expect(onAsk).toHaveBeenCalledWith(
        expect.stringContaining("Resolved preview context"),
        { ignorePinnedResults: true }
      )
    })
    expect(mockWithFullMediaTextIfAvailable).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "QA Doc Title",
        snippet: "Chunk preview text"
      })
    )
  })
})
