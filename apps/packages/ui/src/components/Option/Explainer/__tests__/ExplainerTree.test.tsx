import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import type { ExplainerNode, ExplainerSession } from "../types"
import { ExplainerTree } from "../ExplainerTree"

const node = (overrides: Partial<ExplainerNode> & { id: string; title: string }): ExplainerNode => ({
  sessionId: "s1",
  parentId: null,
  ordinal: 0,
  body: null,
  kind: "explanation",
  intent: "explain",
  status: "complete",
  evidenceState: "supported",
  outsideKnowledgeUsed: false,
  citations: [],
  childNodeIds: [],
  createdAt: "2026-06-09T00:00:00Z",
  updatedAt: "2026-06-09T00:00:00Z",
  ...overrides
})

const session: ExplainerSession = {
  id: "s1",
  ownerUserId: "7",
  title: "Learn attention",
  mode: "goal",
  status: "active",
  outputIntent: "explain",
  grounding: "open",
  depthPreset: "standard",
  selectedSources: [],
  rootNodeIds: ["root"],
  nodes: {
    root: node({ id: "root", title: "Explain transformer attention", childNodeIds: ["child"] }),
    child: node({
      id: "child",
      title: "Scaled dot-product attention",
      parentId: "root",
      ordinal: 1,
      status: "error",
      evidenceState: "partially_supported"
    })
  },
  createdAt: "2026-06-09T00:00:00Z",
  updatedAt: "2026-06-09T00:00:00Z",
  archivedAt: null
}

const renderTree = (overrides: Partial<React.ComponentProps<typeof ExplainerTree>> = {}) =>
  render(
    <ExplainerTree
      session={session}
      selectedNodeId="root"
      onSelectNode={vi.fn()}
      onExpandNode={vi.fn()}
      {...overrides}
    />
  )

describe("ExplainerTree", () => {
  it("hides the status chip for complete nodes but shows non-default statuses", () => {
    renderTree()
    const tree = screen.getByRole("tree", { name: "Explainer outline" })

    expect(within(tree).queryByText("Complete")).not.toBeInTheDocument()
    expect(within(tree).getByText("Error")).toBeInTheDocument()
  })

  it("shows a generating chip and disables break-down for the node being generated", () => {
    renderTree({ generatingNodeId: "root" })
    const tree = screen.getByRole("tree", { name: "Explainer outline" })

    expect(within(tree).getByText("Generating")).toBeInTheDocument()
    expect(
      within(tree).getByRole("button", { name: "Break down Explain transformer attention" })
    ).toBeDisabled()
    expect(
      within(tree).getByRole("button", { name: "Break down Scaled dot-product attention" })
    ).toBeEnabled()
  })

  it("exposes full node titles as tooltips on truncated rows", () => {
    renderTree()

    expect(screen.getByTitle("Explain transformer attention")).toBeInTheDocument()
  })

  it("supports arrow-key navigation between rows", () => {
    const onSelectNode = vi.fn()
    renderTree({ onSelectNode })

    const rootRow = screen.getByRole("button", { name: /^Explain transformer attention/ })
    const childRow = screen.getByRole("button", { name: /^Scaled dot-product attention/ })

    rootRow.focus()
    fireEvent.keyDown(rootRow, { key: "ArrowDown" })
    expect(document.activeElement).toBe(childRow)

    fireEvent.keyDown(childRow, { key: "ArrowUp" })
    expect(document.activeElement).toBe(rootRow)

    fireEvent.keyDown(rootRow, { key: "End" })
    expect(document.activeElement).toBe(childRow)

    fireEvent.keyDown(childRow, { key: "Home" })
    expect(document.activeElement).toBe(rootRow)
  })
})
