import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { SourceChunksList } from "../SourceChunksList"
import type { QADocument } from "../../hooks/useQASearch"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, vars?: Record<string, unknown>) =>
      (fallback ?? _key).replace(/\{\{(\w+)\}\}/g, (token, name) =>
        vars && name in vars ? String(vars[name]) : token
      )
  })
}))

describe("SourceChunksList", () => {
  const documents: QADocument[] = [
    {
      content: "Beta content",
      score: 0.4,
      metadata: { title: "Beta Doc", source: "beta.pdf", type: "pdf" }
    },
    {
      content: "Alpha content",
      score: 0.95,
      metadata: { title: "Alpha Doc", source: "alpha.pdf", type: "pdf" }
    },
    {
      content: "Gamma content",
      score: 0.95,
      metadata: { title: "Gamma Doc", source: "gamma.pdf", type: "pdf" }
    }
  ]

  it("sorts deterministically by relevance by default and supports source sorting", () => {
    render(
      <SourceChunksList
        documents={documents}
        pinnedResults={[]}
        onCopy={vi.fn()}
        onInsert={vi.fn()}
        onPin={vi.fn()}
        onPreview={vi.fn()}
      />
    )

    const initialItems = screen.getAllByRole("listitem")
    expect(initialItems[0]).toHaveTextContent("Alpha Doc")
    expect(initialItems[1]).toHaveTextContent("Gamma Doc")
    expect(initialItems[2]).toHaveTextContent("Beta Doc")

    fireEvent.click(screen.getByRole("button", { name: "Source" }))

    const sourceSortedItems = screen.getAllByRole("listitem")
    expect(sourceSortedItems[0]).toHaveTextContent("Alpha Doc")
    expect(sourceSortedItems[1]).toHaveTextContent("Beta Doc")
    expect(sourceSortedItems[2]).toHaveTextContent("Gamma Doc")
  })

  it("keeps copy/insert/pin actions mapped to the correct document after sorting", () => {
    const onCopy = vi.fn()
    const onInsert = vi.fn()
    const onPin = vi.fn()
    const onPreview = vi.fn()

    render(
      <SourceChunksList
        documents={documents}
        pinnedResults={[]}
        onCopy={onCopy}
        onInsert={onInsert}
        onPin={onPin}
        onPreview={onPreview}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Source" }))

    const firstItem = screen.getAllByRole("listitem")[0]
    const firstItemScope = within(firstItem)
    fireEvent.click(firstItemScope.getByRole("button", { name: "Copy" }))
    fireEvent.click(firstItemScope.getByRole("button", { name: "Insert" }))
    fireEvent.click(firstItemScope.getByRole("button", { name: "Preview" }))
    fireEvent.click(firstItemScope.getByRole("button", { name: "Pin" }))

    expect(onCopy).toHaveBeenCalledTimes(1)
    expect(onInsert).toHaveBeenCalledTimes(1)
    expect(onPreview).toHaveBeenCalledTimes(1)
    expect(onPin).toHaveBeenCalledTimes(1)
    expect(onCopy).toHaveBeenCalledWith(
      expect.objectContaining({
        metadata: expect.objectContaining({ title: "Alpha Doc" })
      })
    )
    expect(onInsert).toHaveBeenCalledWith(
      expect.objectContaining({
        metadata: expect.objectContaining({ title: "Alpha Doc" })
      })
    )
    expect(onPin).toHaveBeenCalledWith(
      expect.objectContaining({
        metadata: expect.objectContaining({ title: "Alpha Doc" })
      })
    )
    expect(onPreview).toHaveBeenCalledWith(
      expect.objectContaining({
        metadata: expect.objectContaining({ title: "Alpha Doc" })
      })
    )
  })

  // CC-10 (#3101), decision [2]: relevance reads as a plain label, compared
  // with the other results; the raw score stays in the tooltip.
  it("shows relevance as a plain label instead of a raw score", () => {
    render(
      <SourceChunksList
        documents={documents}
        pinnedResults={[]}
        onCopy={vi.fn()}
        onInsert={vi.fn()}
        onPin={vi.fn()}
        onPreview={vi.fn()}
      />
    )

    const items = screen.getAllByRole("listitem")
    expect(within(items[0]).getByText("Best match")).toBeInTheDocument()
    expect(within(items[2]).getByText("Weaker match")).toBeInTheDocument()
    expect(within(items[2]).getByText("Weaker match")).toHaveAttribute(
      "title",
      "Compared with the other results. Score 0.40"
    )
    expect(screen.queryByText(/^\d\.\d{3}$/)).not.toBeInTheDocument()
  })

  // CC-10 review: the comparison and score reach screen readers too, not
  // only a hover title on a non-focusable span.
  it("exposes the comparison and the score as text, not only as a title", () => {
    render(
      <SourceChunksList
        documents={documents}
        pinnedResults={[]}
        onCopy={vi.fn()}
        onInsert={vi.fn()}
        onPin={vi.fn()}
        onPreview={vi.fn()}
      />
    )

    const items = screen.getAllByRole("listitem")
    const hidden = within(items[2]).getByText(
      "Compared with the other results. Score 0.40"
    )
    expect(hidden).toHaveClass("sr-only")
  })
})
