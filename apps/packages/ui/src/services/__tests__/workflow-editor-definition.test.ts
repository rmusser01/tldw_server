import { describe, expect, it } from "vitest"
import { compileWorkflowCanvas, reopenWorkflowDefinition } from "../workflow-editor-definition"
import type { ServerWorkflowDefinition } from "@/types/workflow-editor"

const canvas = (count = 2): ServerWorkflowDefinition => {
  const ids = ["start", ...Array.from({ length: count }, (_, index) => `step-${index}`), "end"]
  return { name: "Linear workflow", nodes: ids.map((id, index) => ({ id, type: id === "start" || id === "end" ? id : "step", position: { x: index * 100, y: 0 }, data: { label: id, stepType: id === "start" || id === "end" ? id : "text_clean", config: id === "start" ? { inputSchema: { type: "object", required: ["text"], properties: { text: { type: "string" } } } } : id === "end" ? {} : { text: "{{ inputs.text }}" } } })), edges: ids.slice(0, -1).map((id, index) => ({ id: `edge-${index}`, source: id, target: ids[index + 1] })) }
}

describe("workflow canvas definition boundary", () => {
  it.each(Array.from({ length: 12 }, (_, index) => index + 1))("keeps connected execution order and round-trip identity for %s steps irrespective of canvas order", (count) => {
    const original = canvas(count)
    original.nodes.reverse()
    const definition = compileWorkflowCanvas(original)
    expect(definition.steps.map((step) => step.id)).toEqual(Array.from({ length: count }, (_, index) => `step-${index}`))
    expect(reopenWorkflowDefinition({ id: 4, name: definition.name, version: 1, is_active: true, definition }).nodes).toEqual(original.nodes)
  })
  it("accepts actual server defaults without treating them as a changed canvas", () => {
    const definition = compileWorkflowCanvas(canvas())
    definition.steps = definition.steps.map((step) => ({ ...step, retry: 0, timeout_seconds: 300, on_success: null, on_failure: null, on_timeout: null }))
    expect(reopenWorkflowDefinition({ id: 4, name: definition.name, version: 1, is_active: true, definition }).id).toBe("4")
  })
  it.each(["fan-out", "disconnected", "cycle", "output-mapping", "branch", "invalid-schema"])("refuses unsupported %s authoring before Save or Run", (kind) => {
    const value = canvas()
    if (kind === "fan-out") value.edges.push({ id: "extra", source: "start", target: "step-1" })
    if (kind === "disconnected") value.edges.pop()
    if (kind === "cycle") value.edges[2].target = "step-0"
    if (kind === "output-mapping") value.nodes[3].data.config.outputMapping = "$.text"
    if (kind === "branch") value.nodes[1].data.stepType = "branch"
    if (kind === "invalid-schema") value.nodes[0].data.config.inputSchema = "{"
    expect(() => compileWorkflowCanvas(value)).toThrow()
  })
  it("refuses presentation metadata that changes the executable saved config", () => {
    const definition = compileWorkflowCanvas(canvas())
    definition.steps[0].config = { text: "a different executable instruction" }
    expect(() => reopenWorkflowDefinition({ id: 4, name: definition.name, version: 1, is_active: true, definition })).toThrow("does not match")
  })
  it.each(["plain webhook", "editor webhook", "adapter metadata", "default inputs", "tags", "future editor metadata"])("refuses %s that Open and Save cannot preserve", (kind) => {
    const definition = compileWorkflowCanvas(canvas())
    if (kind === "plain webhook") { delete definition.metadata; Object.assign(definition, { on_completion_webhook: "https://workflow.test/completed" }) }
    if (kind === "editor webhook") Object.assign(definition, { on_completion_webhook: { url: "https://workflow.test/completed", include_outputs: true } })
    if (kind === "adapter metadata") definition.metadata = { ...definition.metadata, adapter: { output_format: "research" } }
    if (kind === "default inputs") definition.inputs = { text: "saved default" }
    if (kind === "tags") Object.assign(definition, { tags: ["research"] })
    if (kind === "future editor metadata") definition.metadata = { editor: { version: 2, canvas: canvas() } }
    expect(() => reopenWorkflowDefinition({ id: 4, name: definition.name, version: 1, is_active: true, definition })).toThrow(/cannot preserve|unsupported editor metadata/)
  })
})
