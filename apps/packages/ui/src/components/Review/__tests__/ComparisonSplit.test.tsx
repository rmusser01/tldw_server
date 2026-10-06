import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ComparisonSplit } from "../ComparisonSplit"
import type { MediaDetail } from "../media-review-types"

vi.mock("@/components/Common/Markdown", () => ({
  Markdown: ({ message }: { message: string }) => (
    <div data-testid="mock-markdown">{message}</div>
  )
}))

vi.mock("@/utils/media-transcript-display", () => ({
  hasLeadingTranscriptTimings: () => false,
  stripLeadingTranscriptTimings: (s: string) => s
}))

vi.mock("@/components/Media/diff-worker-client", () => ({
  shouldRequireSampling: () => false,
  computeDiffSync: (left: string, right: string) => {
    // Simple mock: return same lines for matching, add/del for differences
    const leftLines = left.split("\n")
    const rightLines = right.split("\n")
    const result: Array<{ type: string; text: string }> = []
    const max = Math.max(leftLines.length, rightLines.length)
    for (let i = 0; i < max; i++) {
      const l = leftLines[i]
      const r = rightLines[i]
      if (l === r) {
        result.push({ type: "same", text: l })
      } else {
        if (l != null) result.push({ type: "del", text: l })
        if (r != null) result.push({ type: "add", text: r })
      }
    }
    return result
  }
}))

const t = (key: string, fallback: string, opts?: Record<string, unknown>) =>
  fallback.replace(/\{\{(\w+)\}\}/g, (_, k) => String(opts?.[k] ?? ""))

const makeItem = (id: number, title: string, content: string): MediaDetail => ({
  id,
  title,
  type: "pdf",
  created_at: "2026-01-01",
  content,
  summary: ""
})

const setScrollableMetrics = (
  element: HTMLElement,
  metrics: { clientHeight: number; scrollHeight: number; scrollTop?: number }
) => {
  Object.defineProperty(element, "clientHeight", {
    configurable: true,
    value: metrics.clientHeight
  })
  Object.defineProperty(element, "scrollHeight", {
    configurable: true,
    value: metrics.scrollHeight
  })
  Object.defineProperty(element, "scrollTop", {
    configurable: true,
    writable: true,
    value: metrics.scrollTop ?? 0
  })
}

describe("ComparisonSplit", () => {
  it("renders nothing with fewer than 2 items", () => {
    const { container } = render(
      <ComparisonSplit
        items={[makeItem(1, "A", "hello")]}
        hideTranscriptTimings={false}
        onClose={vi.fn()}
        t={t}
      />
    )
    expect(container.innerHTML).toBe("")
  })

  it("renders split panels for 2 items", () => {
    render(
      <ComparisonSplit
        items={[makeItem(1, "Alpha", "Line one"), makeItem(2, "Beta", "Line two")]}
        hideTranscriptTimings={false}
        onClose={vi.fn()}
        t={t}
      />
    )
    expect(screen.getByTestId("comparison-split")).toBeInTheDocument()
    expect(screen.getByTestId("comparison-panel-0")).toBeInTheDocument()
    expect(screen.getByTestId("comparison-panel-1")).toBeInTheDocument()
    expect(screen.getByText("Alpha")).toBeInTheDocument()
    expect(screen.getByText("Beta")).toBeInTheDocument()
  })

  it("renders 3 panels for 3 items", () => {
    render(
      <ComparisonSplit
        items={[
          makeItem(1, "A", "content a"),
          makeItem(2, "B", "content b"),
          makeItem(3, "C", "content c")
        ]}
        hideTranscriptTimings={false}
        onClose={vi.fn()}
        t={t}
      />
    )
    expect(screen.getByTestId("comparison-panel-0")).toBeInTheDocument()
    expect(screen.getByTestId("comparison-panel-1")).toBeInTheDocument()
    expect(screen.getByTestId("comparison-panel-2")).toBeInTheDocument()
  })

  it("shows diff highlights for exactly 2 items by default", () => {
    render(
      <ComparisonSplit
        items={[
          makeItem(1, "A", "same line\ndifferent left"),
          makeItem(2, "B", "same line\ndifferent right")
        ]}
        hideTranscriptTimings={false}
        onClose={vi.fn()}
        t={t}
      />
    )
    expect(screen.getByTestId("diff-content-left")).toBeInTheDocument()
    expect(screen.getByTestId("diff-content-right")).toBeInTheDocument()
  })

  it("does not show diff highlights for 3+ items", () => {
    render(
      <ComparisonSplit
        items={[
          makeItem(1, "A", "content"),
          makeItem(2, "B", "content"),
          makeItem(3, "C", "content")
        ]}
        hideTranscriptTimings={false}
        onClose={vi.fn()}
        t={t}
      />
    )
    expect(screen.queryByTestId("diff-content-left")).not.toBeInTheDocument()
    expect(screen.queryByTestId("diff-content-right")).not.toBeInTheDocument()
  })

  it("sync scroll toggle is on by default", () => {
    render(
      <ComparisonSplit
        items={[makeItem(1, "A", "a"), makeItem(2, "B", "b")]}
        hideTranscriptTimings={false}
        onClose={vi.fn()}
        t={t}
      />
    )
    const toggle = screen.getByTestId("sync-scroll-toggle")
    expect(toggle).toBeInTheDocument()
  })

  it("diff toggle can disable diff highlighting", () => {
    render(
      <ComparisonSplit
        items={[
          makeItem(1, "A", "hello"),
          makeItem(2, "B", "world")
        ]}
        hideTranscriptTimings={false}
        onClose={vi.fn()}
        t={t}
      />
    )
    // Initially diff is shown
    expect(screen.getByTestId("diff-content-left")).toBeInTheDocument()

    // Toggle diff off
    const diffToggle = screen.getByTestId("diff-toggle")
    const switchEl = diffToggle.querySelector("button") || diffToggle
    fireEvent.click(switchEl)

    // Diff should be gone, replaced by ContentRenderer
    expect(screen.queryByTestId("diff-content-left")).not.toBeInTheDocument()
  })

  it("calls onClose when close button clicked", () => {
    const onClose = vi.fn()
    render(
      <ComparisonSplit
        items={[makeItem(1, "A", "a"), makeItem(2, "B", "b")]}
        hideTranscriptTimings={false}
        onClose={onClose}
        t={t}
      />
    )
    fireEvent.click(screen.getByTestId("close-comparison"))
    expect(onClose).toHaveBeenCalled()
  })

  it("shows item count in header", () => {
    render(
      <ComparisonSplit
        items={[makeItem(1, "A", "a"), makeItem(2, "B", "b"), makeItem(3, "C", "c")]}
        hideTranscriptTimings={false}
        onClose={vi.fn()}
        t={t}
      />
    )
    expect(screen.getByText("Comparing 3 items")).toBeInTheDocument()
  })

  it("rebinds synced-scroll listeners when comparison panels remount", () => {
    const { rerender } = render(
      <ComparisonSplit
        items={[makeItem(1, "Alpha", "Left"), makeItem(2, "Beta", "Right")]}
        hideTranscriptTimings={false}
        onClose={vi.fn()}
        t={t}
      />
    )

    const firstLeftPanel = screen.getByTestId("comparison-content-0")
    const firstRightPanel = screen.getByTestId("comparison-content-1")
    setScrollableMetrics(firstLeftPanel, { clientHeight: 100, scrollHeight: 1000 })
    setScrollableMetrics(firstRightPanel, { clientHeight: 100, scrollHeight: 500 })

    rerender(
      <ComparisonSplit
        items={[makeItem(3, "Gamma", "Fresh left"), makeItem(4, "Delta", "Fresh right")]}
        hideTranscriptTimings={false}
        onClose={vi.fn()}
        t={t}
      />
    )

    const nextLeftPanel = screen.getByTestId("comparison-content-0")
    const nextRightPanel = screen.getByTestId("comparison-content-1")
    expect(nextLeftPanel).not.toBe(firstLeftPanel)
    expect(nextRightPanel).not.toBe(firstRightPanel)

    setScrollableMetrics(nextLeftPanel, {
      clientHeight: 100,
      scrollHeight: 1000,
      scrollTop: 450
    })
    setScrollableMetrics(nextRightPanel, { clientHeight: 100, scrollHeight: 500 })

    fireEvent.scroll(nextLeftPanel)

    expect(nextRightPanel.scrollTop).toBe(200)
  })
})
