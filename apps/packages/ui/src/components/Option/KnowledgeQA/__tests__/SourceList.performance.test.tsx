import { fireEvent, render, screen, waitFor, cleanup } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SourceList } from "../SourceList"

const state = {
  results: [] as Array<Record<string, any>>,
  citations: [] as Array<{ index: number }>,
  focusedSourceIndex: null as number | null,
  focusSource: vi.fn(),
  setQuery: vi.fn(),
  query: "performance benchmark query",
  currentThreadId: "thread-perf" as string | null,
  messages: [{ id: "assistant-1", role: "assistant" }] as Array<{
    id: string
    role: string
  }>,
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    storageScopeKey: "test-owner",
    isAuthorityCurrent: () => true,
    results: state.results,
    citations: state.citations,
    focusedSourceIndex: state.focusedSourceIndex,
    focusSource: state.focusSource,
    setQuery: state.setQuery,
    query: state.query,
    currentThreadId: state.currentThreadId,
    messages: state.messages,
  }),
}))

vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "session-performance-test",
  submitExplicitFeedback: vi.fn().mockResolvedValue({ ok: true }),
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: vi.fn(),
  }),
}))

vi.mock("@/utils/knowledge-qa-search-metrics", () => ({
  trackKnowledgeQaSearchMetric: vi.fn().mockResolvedValue(undefined),
}))

function makeResult(index: number) {
  return {
    id: `perf-source-${index}`,
    content: `Performance body ${index}`,
    metadata: {
      title: `Performance Source ${index}`,
      source_type: index % 2 === 0 ? "media_db" : "notes",
      created_at: `2026-02-${String((index % 28) + 1).padStart(2, "0")}T12:00:00.000Z`,
      page_number: index,
      url: `https://example.com/perf-source-${index}`,
      chunk_id: `chunk_${index}_of_${index <= 50 ? 50 : index}`,
    },
    score: Math.max(0.1, 1 - index * 0.01),
  }
}

function benchmarkRenderPath(resultCount: number): number {
  state.results = Array.from({ length: resultCount }, (_, idx) => makeResult(idx + 1))
  state.citations = [{ index: 1 }, { index: Math.min(5, resultCount) }]

  const start = performance.now()
  render(<SourceList />)
  const end = performance.now()

  expect(
    screen.getByText(`Showing ${Math.min(10, resultCount)} of ${resultCount} sources`)
  ).toBeInTheDocument()

  cleanup()
  return end - start
}

describe("SourceList performance guardrails", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    state.results = []
    state.citations = []
  })

  it("keeps bounded render-path timings for 10/25/50-result scenarios", () => {
    const duration10 = benchmarkRenderPath(10)
    const duration25 = benchmarkRenderPath(25)
    const duration50 = benchmarkRenderPath(50)

    expect(Number.isFinite(duration10)).toBe(true)
    expect(Number.isFinite(duration25)).toBe(true)
    expect(Number.isFinite(duration50)).toBe(true)

    // Wide upper bound to avoid flaky CI while still flagging severe regressions.
    expect(duration10).toBeLessThan(5000)
    expect(duration25).toBeLessThan(5000)
    expect(duration50).toBeLessThan(5000)
  })

  it("maintains threshold pagination behavior for large result sets", async () => {
    state.results = Array.from({ length: 50 }, (_, idx) => makeResult(idx + 1))

    render(<SourceList />)

    expect(screen.getByText("Showing 10 of 50 sources")).toBeInTheDocument()
    const showMoreButton = screen.getByRole("button", { name: /Show more/i })

    fireEvent.click(showMoreButton)
    await waitFor(() =>
      expect(screen.getByText("Showing 20 of 50 sources")).toBeInTheDocument()
    )

    fireEvent.click(screen.getByRole("button", { name: /Show more/i }))
    await waitFor(() =>
      expect(screen.getByText("Showing 30 of 50 sources")).toBeInTheDocument()
    )
  })
})
