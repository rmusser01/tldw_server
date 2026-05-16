import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SourceCard } from "../SourceCard"

/** Helper: open the overflow "More actions" menu, then click the named item. */
async function clickOverflowItem(label: RegExp | string) {
  const moreButton = screen.getByRole("button", { name: /More actions/i })
  fireEvent.click(moreButton)
  const item = screen.getByRole("menuitem", { name: label })
  fireEvent.click(item)
}

describe("SourceCard copy interactions", () => {
  it("ignores stale clipboard completions so the latest copy action owns the UI state", async () => {
    let resolveFirstCopy: (() => void) | null = null
    let callCount = 0
    const writeTextMock = vi.fn().mockImplementation(() => {
      callCount += 1
      if (callCount === 1) {
        return new Promise<void>((resolve) => {
          resolveFirstCopy = resolve
        })
      }
      return Promise.resolve(undefined)
    })

    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
    })

    render(
      <SourceCard
        result={{
          id: "source-1",
          content: "Important quoted source text",
          metadata: {
            title: "Source A",
            url: "https://example.com/source-a",
            source_type: "web",
          },
          score: 0.91,
        }}
        index={1}
        isCited={false}
        isFocused={false}
        onSourceHover={vi.fn()}
        onAskAbout={vi.fn()}
        onViewFull={vi.fn()}
        onSourceFeedback={vi.fn()}
        onRetrySourceFeedback={vi.fn()}
        onTogglePin={vi.fn()}
        onJumpToCitation={vi.fn()}
        feedbackThumb={null}
        feedbackSubmitting={false}
        feedbackError={null}
        isPinned={false}
        highlightTerms={[]}
        citationUsages={[]}
      />
    )

    // "Copy excerpt" is now in the overflow menu
    await clickOverflowItem(/Copy excerpt/i)
    // "Copy citation" is a primary button
    fireEvent.click(screen.getByRole("button", { name: /Copy citation/i }))

    await act(async () => {
      await Promise.resolve()
    })
    // After cite completes, primary button shows "Copied!"
    expect(screen.getByText("Copied!")).toBeInTheDocument()

    resolveFirstCopy?.()
    await act(async () => {
      await Promise.resolve()
    })

    // The cite confirmation still shows; stale copy-text does not overwrite it
    expect(screen.getByText("Copied!")).toBeInTheDocument()
    // The overflow menu item should NOT show "Copied excerpt" from the stale first copy
    const moreButton = screen.getByRole("button", { name: /More actions/i })
    fireEvent.click(moreButton)
    expect(screen.queryByRole("menuitem", { name: /Copied excerpt/i })).not.toBeInTheDocument()
  })

  it("keeps the latest copy confirmation visible until its own timeout completes", async () => {
    vi.useFakeTimers()
    const writeTextMock = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
    })

    render(
      <SourceCard
        result={{
          id: "source-1",
          content: "Important quoted source text",
          metadata: {
            title: "Source A",
            url: "https://example.com/source-a",
            source_type: "web",
          },
          score: 0.91,
        }}
        index={1}
        isCited={false}
        isFocused={false}
        onSourceHover={vi.fn()}
        onAskAbout={vi.fn()}
        onViewFull={vi.fn()}
        onSourceFeedback={vi.fn()}
        onRetrySourceFeedback={vi.fn()}
        onTogglePin={vi.fn()}
        onJumpToCitation={vi.fn()}
        feedbackThumb={null}
        feedbackSubmitting={false}
        feedbackError={null}
        isPinned={false}
        highlightTerms={[]}
        citationUsages={[]}
      />
    )

    // Copy excerpt via overflow menu
    await clickOverflowItem(/Copy excerpt/i)
    await act(async () => {
      await Promise.resolve()
    })
    // Open overflow to verify "Copied excerpt" label
    fireEvent.click(screen.getByRole("button", { name: /More actions/i }))
    expect(screen.getByRole("menuitem", { name: /Copied excerpt/i })).toBeInTheDocument()

    act(() => {
      vi.advanceTimersByTime(1000)
    })
    // Click Copy citation primary button
    fireEvent.click(screen.getByRole("button", { name: /Copy citation/i }))
    await act(async () => {
      await Promise.resolve()
    })
    expect(screen.getByText("Copied!")).toBeInTheDocument()

    act(() => {
      vi.advanceTimersByTime(1000)
    })
    // Still showing "Copied!" on the primary Copy citation button
    expect(screen.getByText("Copied!")).toBeInTheDocument()

    act(() => {
      vi.advanceTimersByTime(1000)
    })
    // Timeout expired, accessible name reverts to "Copy citation"
    expect(screen.getByRole("button", { name: /Copy citation/i })).toBeInTheDocument()
    expect(screen.queryByText("Copied!")).not.toBeInTheDocument()

    vi.useRealTimers()
  })

  it("does not schedule a copied-state reset after unmount when clipboard resolves late", async () => {
    let resolveCopy: (() => void) | null = null
    const setTimeoutSpy = vi.spyOn(window, "setTimeout")
    const writeTextMock = vi.fn().mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveCopy = resolve
        })
    )
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: writeTextMock },
      configurable: true,
    })

    const { unmount } = render(
      <SourceCard
        result={{
          id: "source-1",
          content: "Important quoted source text",
          metadata: {
            title: "Source A",
            url: "https://example.com/source-a",
            source_type: "web",
          },
          score: 0.91,
        }}
        index={1}
        isCited={false}
        isFocused={false}
        onSourceHover={vi.fn()}
        onAskAbout={vi.fn()}
        onViewFull={vi.fn()}
        onSourceFeedback={vi.fn()}
        onRetrySourceFeedback={vi.fn()}
        onTogglePin={vi.fn()}
        onJumpToCitation={vi.fn()}
        feedbackThumb={null}
        feedbackSubmitting={false}
        feedbackError={null}
        isPinned={false}
        highlightTerms={[]}
        citationUsages={[]}
      />
    )

    // Copy excerpt via overflow menu
    await clickOverflowItem(/Copy excerpt/i)
    unmount()

    resolveCopy?.()
    await act(async () => {
      await Promise.resolve()
    })

    expect(setTimeoutSpy).not.toHaveBeenCalled()
    setTimeoutSpy.mockRestore()
  })
})

describe("SourceCard action structure", () => {
  it("shows View and Copy citation as primary buttons and overflow menu with secondary actions", () => {
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true,
    })

    render(
      <SourceCard
        result={{
          id: "source-1",
          content: "Some content",
          metadata: {
            title: "Source A",
            url: "https://example.com",
            source_type: "web",
          },
          score: 0.85,
        }}
        index={1}
        isCited={false}
        isFocused={false}
        onSourceHover={vi.fn()}
        onAskAbout={vi.fn()}
        onViewFull={vi.fn()}
        onSourceFeedback={vi.fn()}
        onRetrySourceFeedback={vi.fn()}
        onTogglePin={vi.fn()}
        onJumpToCitation={vi.fn()}
        feedbackThumb={null}
        feedbackSubmitting={false}
        feedbackError={null}
        isPinned={false}
        highlightTerms={[]}
        citationUsages={[]}
      />
    )

    // Primary buttons are always visible
    expect(screen.getByRole("button", { name: /View source 1/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Copy citation/i })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /More actions/i })).toBeInTheDocument()

    // Secondary actions are hidden until overflow is opened
    expect(screen.queryByRole("menuitem", { name: /Pin/i })).not.toBeInTheDocument()

    // Open overflow
    fireEvent.click(screen.getByRole("button", { name: /More actions/i }))
    expect(screen.getByRole("menuitem", { name: /Pin/i })).toBeInTheDocument()
    expect(screen.getByRole("menuitem", { name: /Tell me more/i })).toBeInTheDocument()
    expect(screen.getByRole("menuitem", { name: /Summarize/i })).toBeInTheDocument()
    expect(screen.getByRole("menuitem", { name: /Key quotes/i })).toBeInTheDocument()
    expect(screen.getByRole("menuitem", { name: /Copy excerpt/i })).toBeInTheDocument()
    expect(screen.getByRole("menuitem", { name: /Open original/i })).toBeInTheDocument()
    expect(screen.getAllByRole("button", { name: /Copy citation/i })).toHaveLength(1)
    expect(screen.getAllByRole("menuitem", { name: /Copy excerpt/i })).toHaveLength(1)
    expect(screen.queryByRole("menuitem", { name: /Copy text/i })).not.toBeInTheDocument()
  })

  it("calls onAskAbout with the correct template from overflow menu items", async () => {
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true,
    })

    const onAskAbout = vi.fn()
    const result = {
      id: "source-1",
      content: "Some content",
      metadata: { title: "Source A", source_type: "web" as const },
      score: 0.85,
    }

    render(
      <SourceCard
        result={result}
        index={1}
        isCited={false}
        isFocused={false}
        onSourceHover={vi.fn()}
        onAskAbout={onAskAbout}
        onViewFull={vi.fn()}
        onSourceFeedback={vi.fn()}
        onRetrySourceFeedback={vi.fn()}
        onTogglePin={vi.fn()}
        onJumpToCitation={vi.fn()}
        feedbackThumb={null}
        feedbackSubmitting={false}
        feedbackError={null}
        isPinned={false}
        highlightTerms={[]}
        citationUsages={[]}
      />
    )

    await clickOverflowItem(/Tell me more/i)
    expect(onAskAbout).toHaveBeenCalledWith(result, "detail")

    await clickOverflowItem(/Summarize/i)
    expect(onAskAbout).toHaveBeenCalledWith(result, "summary")

    await clickOverflowItem(/Key quotes/i)
    expect(onAskAbout).toHaveBeenCalledWith(result, "quotes")
  })

  it("shows Unpin in overflow when source is pinned", () => {
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true,
    })

    render(
      <SourceCard
        result={{
          id: "source-1",
          content: "Some content",
          metadata: { title: "Source A", source_type: "web" as const },
          score: 0.85,
        }}
        index={1}
        isCited={false}
        isFocused={false}
        onSourceHover={vi.fn()}
        onAskAbout={vi.fn()}
        onViewFull={vi.fn()}
        onSourceFeedback={vi.fn()}
        onRetrySourceFeedback={vi.fn()}
        onTogglePin={vi.fn()}
        onJumpToCitation={vi.fn()}
        feedbackThumb={null}
        feedbackSubmitting={false}
        feedbackError={null}
        isPinned={true}
        highlightTerms={[]}
        citationUsages={[]}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /More actions/i }))
    expect(screen.getByRole("menuitem", { name: /Unpin/i })).toBeInTheDocument()
  })

  it("hides Open original from overflow when there is no URL", () => {
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true,
    })

    render(
      <SourceCard
        result={{
          id: "source-1",
          content: "Some content",
          metadata: { title: "Source A", source_type: "media_db" as const },
          score: 0.85,
        }}
        index={1}
        isCited={false}
        isFocused={false}
        onSourceHover={vi.fn()}
        onAskAbout={vi.fn()}
        onViewFull={vi.fn()}
        onSourceFeedback={vi.fn()}
        onRetrySourceFeedback={vi.fn()}
        onTogglePin={vi.fn()}
        onJumpToCitation={vi.fn()}
        feedbackThumb={null}
        feedbackSubmitting={false}
        feedbackError={null}
        isPinned={false}
        highlightTerms={[]}
        citationUsages={[]}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /More actions/i }))
    expect(screen.queryByRole("menuitem", { name: /Open original/i })).not.toBeInTheDocument()
  })

  it("drops the verbose feedback label in compact density", () => {
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true,
    })

    render(
      <SourceCard
        result={{
          id: "source-1",
          content: "Some content",
          metadata: { title: "Source A", source_type: "media_db" as const },
          score: 0.21,
        }}
        index={1}
        isCited={false}
        isFocused={false}
        onSourceHover={vi.fn()}
        onAskAbout={vi.fn()}
        onViewFull={vi.fn()}
        onSourceFeedback={vi.fn()}
        onRetrySourceFeedback={vi.fn()}
        onTogglePin={vi.fn()}
        onJumpToCitation={vi.fn()}
        feedbackThumb={null}
        feedbackSubmitting={false}
        feedbackError={null}
        isPinned={false}
        highlightTerms={[]}
        citationUsages={[]}
        density="compact"
      />
    )

    expect(screen.queryByText("Relevant?")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Yes" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "No" })).toBeInTheDocument()
  })

  it("collapses compact header metadata into a lighter hierarchy", () => {
    Object.defineProperty(globalThis.navigator, "clipboard", {
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
      configurable: true,
    })

    render(
      <SourceCard
        result={{
          id: "source-1",
          content: "Some content",
          metadata: {
            title: "Source A",
            source_type: "web",
            chunk_id: "chunk_4_of_9",
          },
          score: 0.21,
        }}
        index={1}
        isCited={true}
        isFocused={false}
        onSourceHover={vi.fn()}
        onAskAbout={vi.fn()}
        onViewFull={vi.fn()}
        onSourceFeedback={vi.fn()}
        onRetrySourceFeedback={vi.fn()}
        onTogglePin={vi.fn()}
        onJumpToCitation={vi.fn()}
        feedbackThumb={null}
        feedbackSubmitting={false}
        feedbackError={null}
        isPinned={true}
        highlightTerms={[]}
        citationUsages={[]}
        density="compact"
      />
    )

    expect(screen.getByTestId("knowledge-source-compact-relevance")).toHaveTextContent("21% match")
    expect(screen.getByTestId("knowledge-source-compact-meta")).toHaveTextContent(
      "Web • Section 4 of 9"
    )
    expect(screen.getByTestId("knowledge-source-compact-status")).toHaveTextContent("Pinned")
    expect(screen.getByRole("button", { name: /Jump to citation 1 in answer/i })).toBeInTheDocument()
    expect(screen.queryByText("Weak relevance (21%)")).not.toBeInTheDocument()
  })
})
