import React from "react"
import { render, screen, within } from "@testing-library/react"
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
