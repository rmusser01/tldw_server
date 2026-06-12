import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import type { ExplainerNode, ExplainerSession } from "../types"
import { ExplainerDetailPanel } from "../ExplainerDetailPanel"

const baseNode: ExplainerNode = {
  id: "child",
  sessionId: "s1",
  parentId: "root",
  ordinal: 1,
  title: "Scaled dot-product attention",
  body: "Compares query and key vectors.",
  kind: "explanation",
  intent: "explain",
  status: "complete",
  evidenceState: "partially_supported",
  outsideKnowledgeUsed: true,
  citations: [],
  childNodeIds: [],
  createdAt: "2026-06-09T00:00:00Z",
  updatedAt: "2026-06-09T00:00:00Z"
}

const session: ExplainerSession = {
  id: "s1",
  ownerUserId: "7",
  title: "Learn attention",
  mode: "goal",
  status: "active",
  outputIntent: "both",
  grounding: "source_led",
  depthPreset: "standard",
  selectedSources: [],
  rootNodeIds: ["root"],
  nodes: {},
  createdAt: "2026-06-09T00:00:00Z",
  updatedAt: "2026-06-09T00:00:00Z",
  archivedAt: null
}

const renderPanel = (overrides: Partial<React.ComponentProps<typeof ExplainerDetailPanel>> = {}) =>
  render(
    <ExplainerDetailPanel
      session={session}
      node={baseNode}
      onExpand={vi.fn()}
      onDeleteNode={vi.fn()}
      {...overrides}
    />
  )

describe("ExplainerDetailPanel", () => {
  it("renders labeled session metadata instead of raw enum values", () => {
    renderPanel()

    expect(
      screen.getByText("Intent: Explain & plan · Grounding: Source-led · Depth: Standard")
    ).toBeInTheDocument()
  })

  it("hides the Complete status chip and explains the evidence state", () => {
    renderPanel()
    const detail = screen.getByRole("region", { name: "Explainer detail" })

    expect(within(detail).queryByText("Complete")).not.toBeInTheDocument()
    const evidence = within(detail).getByText("Partially supported")
    expect(evidence).toHaveAttribute("title")
    expect(evidence.getAttribute("title")).toMatch(/citation/i)
  })

  it("offers delete for non-root nodes but not for roots", () => {
    renderPanel()
    expect(screen.getByRole("button", { name: "Delete node" })).toBeInTheDocument()

    renderPanel({ node: { ...baseNode, id: "root", parentId: null } })
    expect(screen.getAllByRole("button", { name: "Delete node" })).toHaveLength(1)
  })

  it("disables break-down while the node is generating", () => {
    renderPanel({ generatingNodeId: "child" })

    expect(screen.getByRole("button", { name: "Break down" })).toBeDisabled()
  })
})

describe("ExplainerDetailPanel clarifying questions", () => {
  const questionNode = {
    ...baseNode,
    kind: "question" as const,
    questionOptions: [
      { id: "math", label: "Focus on math" },
      { id: "intuition", label: "Focus on intuition" }
    ],
    selectedOptionId: null,
    selectedCustomAnswer: null
  }

  it("lets users answer an open question by picking an option", () => {
    const onAnswerQuestion = vi.fn()
    renderPanel({ node: questionNode, onAnswerQuestion })

    fireEvent.click(screen.getByRole("button", { name: "Focus on math" }))

    expect(onAnswerQuestion).toHaveBeenCalledWith("child", { selectedOptionId: "math" })
  })

  it("lets users answer with a custom response", () => {
    const onAnswerQuestion = vi.fn()
    renderPanel({ node: questionNode, onAnswerQuestion })

    fireEvent.change(screen.getByLabelText("Custom answer"), {
      target: { value: "Both, but lead with intuition" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Submit answer" }))

    expect(onAnswerQuestion).toHaveBeenCalledWith("child", {
      selectedCustomAnswer: "Both, but lead with intuition"
    })
  })

  it("shows answered questions read-only", () => {
    const onAnswerQuestion = vi.fn()
    renderPanel({
      node: { ...questionNode, selectedOptionId: "math" },
      onAnswerQuestion
    })

    expect(screen.queryByLabelText("Custom answer")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Focus on math" })).not.toBeInTheDocument()
    expect(screen.getByText("Focus on math")).toBeInTheDocument()
  })
})

describe("ExplainerDetailPanel citation links", () => {
  const citation = {
    id: "cite-1",
    sourceId: "media-42",
    sourceType: "media",
    title: "Attention notes",
    excerpt: "Attention weights are computed from query-key similarity.",
    locationLabel: "chunk 3"
  }

  it("links external citations to their URL", () => {
    renderPanel({
      node: { ...baseNode, citations: [{ ...citation, url: "https://example.test/paper" }] }
    })

    const link = screen.getByRole("link", { name: "Open source" })
    expect(link).toHaveAttribute("href", "https://example.test/paper")
    expect(link).toHaveAttribute("target", "_blank")
  })

  it("links media citations to the media library item", () => {
    renderPanel({ node: { ...baseNode, citations: [citation] } })

    expect(screen.getByRole("link", { name: "Open source" })).toHaveAttribute(
      "href",
      "/media?id=media-42"
    )
  })

  it("links note citations to the notes manager", () => {
    renderPanel({
      node: { ...baseNode, citations: [{ ...citation, sourceId: "note-7", sourceType: "note" }] }
    })

    expect(screen.getByRole("link", { name: "Open source" })).toHaveAttribute("href", "/notes")
  })
})
