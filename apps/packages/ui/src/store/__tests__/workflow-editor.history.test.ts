// @vitest-environment jsdom
/**
 * Characterization tests for workflow-editor undo/redo history cloning.
 *
 * `createHistoryEntry` deep-clones nodes/edges with `structuredClone` (WP-21,
 * previously `JSON.parse(JSON.stringify(...))`). These tests pin the clone's
 * observable shape so the conversion cannot silently drift:
 * - history snapshots must be isolated from in-place mutations of live node
 *   objects (deep copy, not reference copy);
 * - for the JSON-origin shapes this store holds, the clone must be
 *   indistinguishable from a JSON round-trip (no undefined-dropping or
 *   Date-stringification semantics were relied upon);
 * - nodes carrying library-attached symbol-keyed internals (DOM handles) must
 *   clone without throwing, with the symbol-keyed baggage dropped — the same
 *   observable behavior JSON.stringify had. (@xyflow/react v12 keeps its
 *   internals in its own lookup rather than on user nodes, but this guards
 *   against any library attaching private fields to node objects.)
 */
import { beforeEach, describe, expect, it } from "vitest"
import { useWorkflowEditorStore } from "../workflow-editor"
import type { WorkflowEdge, WorkflowNode } from "@/types/workflow-editor"

const makeNode = (overrides: Partial<WorkflowNode> = {}): WorkflowNode =>
  ({
    id: "node-1",
    type: "prompt",
    position: { x: 10, y: 20 },
    selected: true,
    measured: { width: 120, height: 60 },
    data: {
      label: "Original label",
      stepType: "prompt",
      config: {
        prompt: "hello",
        nested: { list: [1, 2] }
      },
      isExpanded: false
    },
    ...overrides
  }) as WorkflowNode

const makeEdge = (): WorkflowEdge =>
  ({
    id: "edge-1",
    source: "node-1",
    target: "node-2",
    animated: true
  }) as WorkflowEdge

const seedCanvas = (nodes: WorkflowNode[], edges: WorkflowEdge[] = []) => {
  useWorkflowEditorStore.setState({
    nodes,
    edges,
    past: [],
    future: [],
    selectedNodeIds: [],
    isDirty: false
  })
}

describe("workflow-editor history cloning (structuredClone)", () => {
  beforeEach(() => {
    seedCanvas([])
  })

  it("history snapshots are isolated from later in-place mutations of live nodes", () => {
    const node = makeNode()
    seedCanvas([node], [makeEdge()])

    useWorkflowEditorStore.getState().pushHistory("seed snapshot")

    // Mutate the live objects in place after the snapshot was taken.
    const nested = node.data.config.nested as { list: number[] }
    nested.list.push(999)
    node.data.label = "Mutated label"

    useWorkflowEditorStore.getState().undo()

    const restored = useWorkflowEditorStore.getState().nodes
    expect(restored).toHaveLength(1)
    expect(restored[0].data.label).toBe("Original label")
    expect(restored[0].data.config.nested).toEqual({ list: [1, 2] })
  })

  it("undo/redo round-trips canvas state through history", () => {
    seedCanvas([makeNode()], [makeEdge()])

    useWorkflowEditorStore.getState().addNode({
      type: "log",
      position: { x: 1, y: 1 }
    })
    expect(useWorkflowEditorStore.getState().nodes).toHaveLength(2)

    useWorkflowEditorStore.getState().undo()
    expect(useWorkflowEditorStore.getState().nodes).toHaveLength(1)
    expect(useWorkflowEditorStore.getState().edges).toHaveLength(1)

    useWorkflowEditorStore.getState().redo()
    const afterRedo = useWorkflowEditorStore.getState().nodes
    expect(afterRedo).toHaveLength(2)
    expect(afterRedo.map((n) => n.id)).toContain("node-1")
  })

  it("history entry shape is JSON-round-trip identical for the shapes this store holds", () => {
    const node = makeNode()
    const edge = makeEdge()
    seedCanvas([node], [edge])

    useWorkflowEditorStore.getState().pushHistory("shape pin")

    const entry = useWorkflowEditorStore.getState().past[0]
    expect(entry).toBeDefined()
    // If any cloned value were structuredClone-only (an undefined-valued key,
    // a Date, a Map...), the JSON round-trip would not deep-equal the clone.
    expect(entry.nodes).toEqual(JSON.parse(JSON.stringify(entry.nodes)))
    expect(entry.edges).toEqual(JSON.parse(JSON.stringify(entry.edges)))
    expect(entry.nodes[0].data).toEqual(node.data)
    expect(entry.nodes[0].measured).toEqual({ width: 120, height: 60 })
  })

  it("clones nodes carrying symbol-keyed DOM internals without throwing and drops the internals", () => {
    const internals = Symbol.for("internals")
    const node = makeNode() as WorkflowNode & { [internals]?: unknown }
    node[internals] = {
      handleBounds: { source: [document.createElement("div")] }
    }
    seedCanvas([node], [])

    expect(() =>
      useWorkflowEditorStore.getState().pushHistory("rf internals")
    ).not.toThrow()

    const entry = useWorkflowEditorStore.getState().past[0]
    expect(entry.nodes[0].data.label).toBe("Original label")
    // Symbol-keyed properties are skipped (not cloned), matching the old
    // JSON.stringify behavior; the DOM element under them is never visited.
    expect((entry.nodes[0] as Record<symbol, unknown>)[internals]).toBeUndefined()
  })
})
