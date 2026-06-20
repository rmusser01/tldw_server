import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { ExplainerDetailPanel } from "../ExplainerDetailPanel"
import type { ExplainerNode, ExplainerSession } from "../types"

const rootNode: ExplainerNode = {
  id: "root",
  sessionId: "session-1",
  parentId: null,
  ordinal: 0,
  title: "Explain transformer attention",
  body: "Attention lets tokens route information to each other.",
  kind: "summary",
  intent: "both",
  status: "complete",
  evidenceState: "supported",
  outsideKnowledgeUsed: false,
  citations: [],
  questionOptions: [
    { id: "math", label: "Keep equations" },
    { id: "intuition", label: "Use intuition first" }
  ],
  selectedOptionId: "math",
  selectedCustomAnswer: null,
  generationMetadata: null,
  childNodeIds: [],
  createdAt: "2026-06-09T00:00:00Z",
  updatedAt: "2026-06-09T00:00:01Z"
}

const session: ExplainerSession = {
  id: "session-1",
  ownerUserId: "7",
  title: "Learn attention",
  mode: "goal",
  status: "active",
  outputIntent: "both",
  grounding: "source_led",
  depthPreset: "standard",
  selectedSources: [],
  rootNodeIds: ["root"],
  nodes: { root: rootNode },
  createdAt: "2026-06-09T00:00:00Z",
  updatedAt: "2026-06-09T00:00:01Z",
  archivedAt: null
}

describe("ExplainerDetailPanel", () => {
  it("renders selected answer chips without exposing a root delete action", () => {
    render(
      <ExplainerDetailPanel
        session={session}
        node={rootNode}
        onExpand={vi.fn()}
      />
    )

    expect(screen.getByText("Keep equations")).toBeInTheDocument()
    expect(screen.getByText("Use intuition first")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /delete node/i })).not.toBeInTheDocument()
  })
})
