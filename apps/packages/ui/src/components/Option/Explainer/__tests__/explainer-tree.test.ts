import { describe, expect, it } from "vitest"

import type { ExplainerNode } from "../types"
import {
  flattenExplainerTree,
  getExplainerEvidenceLabel,
  getExplainerNodeStatusLabel,
  getSelectedExplainerNode,
  pruneExplainerNodeTree
} from "../tree"

const node = (overrides: Partial<ExplainerNode> & { id: string }): ExplainerNode => ({
  id: overrides.id,
  sessionId: "session-1",
  parentId: null,
  ordinal: 0,
  title: overrides.id,
  body: null,
  kind: "explanation",
  intent: "explain",
  status: "idle",
  evidenceState: "uncited",
  outsideKnowledgeUsed: false,
  citations: [],
  childNodeIds: [],
  createdAt: "2026-06-09T00:00:00Z",
  updatedAt: "2026-06-09T00:00:00Z",
  ...overrides
})

describe("Explainer tree utilities", () => {
  it("flattens roots and children in stable ordinal order", () => {
    const nodes: Record<string, ExplainerNode> = {
      root: node({
        id: "root",
        childNodeIds: ["child-b", "child-a"]
      }),
      "child-b": node({
        id: "child-b",
        parentId: "root",
        ordinal: 2,
        createdAt: "2026-06-09T00:00:02Z"
      }),
      "child-a": node({
        id: "child-a",
        parentId: "root",
        ordinal: 1,
        createdAt: "2026-06-09T00:00:01Z"
      })
    }

    expect(flattenExplainerTree(nodes, ["root"]).map((item) => item.node.id)).toEqual([
      "root",
      "child-a",
      "child-b"
    ])
  })

  it("falls back to the first root when the selected node is absent", () => {
    const nodes: Record<string, ExplainerNode> = {
      root: node({ id: "root" })
    }

    expect(getSelectedExplainerNode(nodes, ["root"], "missing")?.id).toBe("root")
  })

  it("prunes a deleted node and its descendants", () => {
    const nodes: Record<string, ExplainerNode> = {
      root: node({ id: "root", childNodeIds: ["keep", "delete"] }),
      keep: node({ id: "keep", parentId: "root" }),
      delete: node({ id: "delete", parentId: "root", childNodeIds: ["grandchild"] }),
      grandchild: node({ id: "grandchild", parentId: "delete" })
    }

    const pruned = pruneExplainerNodeTree(nodes, ["root"], "delete")

    expect(Object.keys(pruned.nodes).sort()).toEqual(["keep", "root"])
    expect(pruned.nodes.root?.childNodeIds).toEqual(["keep"])
    expect(pruned.rootNodeIds).toEqual(["root"])
  })

  it("maps status and evidence values to readable labels", () => {
    expect(getExplainerNodeStatusLabel("queued")).toBe("Queued")
    expect(getExplainerNodeStatusLabel("generating")).toBe("Generating")
    expect(getExplainerEvidenceLabel("partially_supported")).toBe("Partially supported")
    expect(getExplainerEvidenceLabel("insufficient")).toBe("Insufficient evidence")
  })
})
