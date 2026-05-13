import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgePanel } from "../KnowledgePanel"

const mockUseKnowledgeSettings = vi.fn()
const mockUseKnowledgeSearch = vi.fn()
const mockUseFileSearch = vi.fn()
const mockUseQASearch = vi.fn()

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key,
    i18n: { language: "en" }
  })
}))

vi.mock("../hooks", () => ({
  useKnowledgeSettings: (...args: unknown[]) =>
    mockUseKnowledgeSettings(...args),
  useKnowledgeSearch: (...args: unknown[]) => mockUseKnowledgeSearch(...args),
  toPinnedResult: (item: any) => ({
    id: `pin-${item?.metadata?.id ?? "1"}`,
    title:
      typeof item?.metadata?.title === "string" ? item.metadata.title : "",
    source:
      typeof item?.metadata?.source === "string"
        ? item.metadata.source
        : undefined,
    url:
      typeof item?.metadata?.url === "string" ? item.metadata.url : undefined,
    snippet: String(item?.content || item?.text || item?.chunk || "")
  }),
  withFullMediaTextIfAvailable: async (pinned: unknown) => pinned,
  qaDocumentToRagResult: (doc: any) => ({
    content: doc.content || doc.text || doc.chunk || "",
    metadata: doc.metadata || {},
    score: doc.score,
    relevance: doc.relevance
  })
}))

vi.mock("../hooks/useFileSearch", async () => {
  const actual = await vi.importActual<typeof import("../hooks/useFileSearch")>(
    "../hooks/useFileSearch"
  )
  return {
    ...actual,
    useFileSearch: (...args: unknown[]) => mockUseFileSearch(...args)
  }
})

vi.mock("../hooks/useQASearch", () => ({
  useQASearch: (...args: unknown[]) => mockUseQASearch(...args)
}))

const setupMocks = () => {
  mockUseKnowledgeSettings.mockReturnValue({
    preset: "balanced",
    draftSettings: {
      query: "knowledge query",
      sources: ["media_db"],
      strategy: "standard"
    },
    storedSettings: {},
    useCurrentMessage: false,
    advancedOpen: false,
    advancedSearch: "",
    resolvedQuery: "knowledge query",
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
    response: null,
    timedOut: false,
    hasAttemptedSearch: false,
    queryError: null,
    runQASearch: vi.fn(),
    copyAnswer: vi.fn(),
    insertAnswer: vi.fn(),
    insertChunk: vi.fn(),
    copyChunk: vi.fn(),
    pinChunk: vi.fn()
  })
}

describe("KnowledgePanel tab routing", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setupMocks()
  })

  it("keeps hook ordering stable when rerendering from closed to open", () => {
    const { rerender } = render(
      <KnowledgePanel
        open={false}
        showToggle={false}
        onInsert={vi.fn()}
        onAsk={vi.fn()}
      />
    )

    expect(() =>
      rerender(
        <KnowledgePanel
          open
          showToggle={false}
          onInsert={vi.fn()}
          onAsk={vi.fn()}
        />
      )
    ).not.toThrow()

    expect(screen.getByRole("tab", { name: "QA Search" })).toHaveAttribute(
      "aria-selected",
      "true"
    )
    expect(screen.getByText(/Open \/knowledge for the full QA workspace/i)).toBeInTheDocument()
  })

  it("maps backward-compat openTab=search to QA Search", () => {
    render(
      <KnowledgePanel
        open
        showToggle={false}
        openTab="search"
        onInsert={vi.fn()}
        onAsk={vi.fn()}
      />
    )

    expect(screen.getByRole("tab", { name: "QA Search" })).toHaveAttribute(
      "aria-selected",
      "true"
    )
    expect(screen.getByRole("tab", { name: "File Search" })).toHaveAttribute(
      "aria-selected",
      "false"
    )
  })

  it("re-applies openTab requests with request id and still maps search alias", () => {
    const { rerender } = render(
      <KnowledgePanel
        open
        showToggle={false}
        openTab="context"
        openTabRequestId={1}
        onInsert={vi.fn()}
        onAsk={vi.fn()}
      />
    )

    expect(screen.getByRole("tab", { name: "Context" })).toHaveAttribute(
      "aria-selected",
      "true"
    )

    fireEvent.click(screen.getByRole("tab", { name: "Settings" }))
    expect(screen.getByRole("tab", { name: "Settings" })).toHaveAttribute(
      "aria-selected",
      "true"
    )

    rerender(
      <KnowledgePanel
        open
        showToggle={false}
        openTab="search"
        openTabRequestId={2}
        onInsert={vi.fn()}
        onAsk={vi.fn()}
      />
    )

    expect(screen.getByRole("tab", { name: "QA Search" })).toHaveAttribute(
      "aria-selected",
      "true"
    )
  })
})
