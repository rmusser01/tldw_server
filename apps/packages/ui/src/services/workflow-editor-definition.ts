import type { SavedWorkflowDefinition, ServerWorkflowDefinition, WorkflowDefinitionPayload, WorkflowNode } from "@/types/workflow-editor"

const object = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) : object(value) ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])])) : value
const stepsKey = (steps: WorkflowDefinitionPayload["steps"]) => JSON.stringify(canonical(steps.map((step) => ({ id: step.id, name: step.name ?? null, type: step.type, config: step.config || {}, retry: step.retry === undefined ? 0 : step.retry, timeout_seconds: step.timeout_seconds === undefined ? 300 : step.timeout_seconds, on_success: step.on_success ?? null, on_failure: step.on_failure ?? null, on_timeout: step.on_timeout ?? null }))))

/** Copy only authored canvas fields; selection and dragging are transient. */
export const workflowCanvasSnapshot = (canvas: ServerWorkflowDefinition): ServerWorkflowDefinition => JSON.parse(JSON.stringify({
  name: canvas.name,
  description: canvas.description || "",
  nodes: canvas.nodes.map(({ id, type, position, data }) => ({ id, type, position, data })),
  edges: canvas.edges.map(({ id, source, target, sourceHandle, targetHandle, type }) => ({ id, source, target, sourceHandle, targetHandle, type }))
}))

export const workflowCanvasKey = (canvas: ServerWorkflowDefinition): string => JSON.stringify(workflowCanvasSnapshot(canvas))

/** Convert the supported linear canvas to the existing server DSL. */
export const compileWorkflowCanvas = (canvas: ServerWorkflowDefinition, version = 1): WorkflowDefinitionPayload => {
  const snapshot = workflowCanvasSnapshot(canvas)
  if (!snapshot.name?.trim()) throw new Error("Enter a workflow name before saving.")
  const nodes = new Map(snapshot.nodes.map((node) => [node.id, node]))
  if (nodes.size !== snapshot.nodes.length) throw new Error("Workflow node IDs must be unique.")
  const starts = snapshot.nodes.filter((node) => node.data.stepType === "start")
  const ends = snapshot.nodes.filter((node) => node.data.stepType === "end")
  if (starts.length !== 1 || ends.length !== 1) throw new Error("Connect exactly one Start and one End.")
  if (String(ends[0].data.config.outputMapping || "").trim()) throw new Error("End output mapping is not supported; the server returns the last step output.")
  const outgoing = new Map<string, string>()
  const incoming = new Map<string, number>()
  for (const edge of snapshot.edges) {
    if (!nodes.has(edge.source) || !nodes.has(edge.target)) throw new Error("A workflow connection refers to a missing node.")
    if (outgoing.has(edge.source)) throw new Error("Multiple outgoing connections require unsupported graph routing.")
    if (edge.source === ends[0].id || edge.target === starts[0].id) throw new Error("Start and End connections have the wrong direction.")
    outgoing.set(edge.source, edge.target)
    incoming.set(edge.target, (incoming.get(edge.target) || 0) + 1)
    if (incoming.get(edge.target)! > 1) throw new Error("Multiple incoming connections require unsupported graph routing.")
  }
  const ordered: WorkflowNode[] = []
  const visited = new Set<string>()
  let current = starts[0].id
  while (current !== ends[0].id) {
    if (visited.has(current)) throw new Error("Workflow cycles are not supported.")
    visited.add(current)
    const next = outgoing.get(current)
    if (!next) throw new Error("Connect every step from Start through End.")
    current = next
    const node = nodes.get(current)!
    if (node.data.stepType !== "end") ordered.push(node)
  }
  visited.add(current)
  if (visited.size !== nodes.size) throw new Error("Connect every node to the Start-to-End path.")
  if (!ordered.length) throw new Error("Add at least one executable workflow step.")
  for (const node of ordered) {
    if (!object(node.data.config) || !node.data.stepType || node.data.stepType === "branch" || node.data.config.true_next || node.data.config.false_next) {
      throw new Error(`Unsupported routing or configuration in ${node.data.label || node.id}.`)
    }
  }
  const inputSchema = starts[0].data.config.inputSchema
  if (inputSchema !== undefined && !object(inputSchema)) throw new Error("Start input schema must be a JSON object.")
  return {
    name: snapshot.name, description: snapshot.description, version, visibility: "private", inputs: {},
    steps: ordered.map((node) => ({ id: node.id, name: node.data.label, type: node.data.stepType, config: node.data.config, ...(node.data.retry !== undefined ? { retry: node.data.retry as number | null } : {}), ...(node.data.timeoutSeconds !== undefined ? { timeout_seconds: node.data.timeoutSeconds as number | null } : {}) })),
    metadata: { editor: { version: 1, canvas: snapshot } }
  }
}

/** The persisted DSL remains authoritative over optional canvas presentation. */
export const reopenWorkflowDefinition = (saved: SavedWorkflowDefinition): ServerWorkflowDefinition => {
  const definition = saved.definition
  const editor = definition.metadata?.editor
  const fields = definition as unknown as Record<string, unknown>
  const represented = new Set(["name", "version", "description", "tags", "inputs", "on_completion_webhook", "steps", "metadata", "visibility"])
  if (Object.keys(fields).some((key) => !represented.has(key)) || fields.on_completion_webhook != null || (fields.visibility != null && fields.visibility !== "private") || (fields.tags != null && (!Array.isArray(fields.tags) || fields.tags.length > 0)) || (fields.inputs != null && (!object(fields.inputs) || Object.keys(fields.inputs).length > 0)) || (definition.metadata && Object.keys(definition.metadata).some((key) => key !== "editor"))) {
    throw new Error("The canvas editor cannot preserve this definition's webhook, metadata, tags, inputs, or other saved settings. Use the existing Workflow API to edit it.")
  }
  if (editor !== undefined && (!object(editor) || editor.version !== 1 || !object(editor.canvas) || Object.keys(editor).some((key) => key !== "version" && key !== "canvas"))) {
    throw new Error("This definition uses unsupported editor metadata.")
  }
  let canvas: ServerWorkflowDefinition
  if (object(editor) && editor.version === 1 && object(editor.canvas)) {
    canvas = editor.canvas as unknown as ServerWorkflowDefinition
    if (!Array.isArray(canvas.nodes) || !Array.isArray(canvas.edges)) throw new Error("Saved workflow canvas is malformed.")
    const compiled = compileWorkflowCanvas(canvas, saved.version)
    if (stepsKey(compiled.steps) !== stepsKey(definition.steps) || canvas.name !== definition.name || (canvas.description || "") !== (definition.description || "")) {
      throw new Error("Saved canvas does not match the server workflow definition.")
    }
  } else {
    const steps = definition.steps
    if (!steps.length || steps.some((step, index) => step.type === "branch" || step.on_failure || step.on_timeout || (step.on_success && step.on_success !== steps[index + 1]?.id))) {
      throw new Error("This saved definition uses routing that the canvas editor does not support.")
    }
    const startId = "editor-start"
    const endId = "editor-end"
    if (steps.some((step) => step.id === startId || step.id === endId)) throw new Error("Saved step IDs conflict with editor markers.")
    const nodes: WorkflowNode[] = [
      { id: startId, type: "start", position: { x: 0, y: 100 }, data: { label: "Start", stepType: "start", config: {} } },
      ...steps.map((step, index) => ({ id: step.id, type: "step", position: { x: (index + 1) * 240, y: 100 }, data: { label: step.name || step.id, stepType: step.type, config: step.config || {}, retry: step.retry, timeoutSeconds: step.timeout_seconds } })),
      { id: endId, type: "end", position: { x: (steps.length + 1) * 240, y: 100 }, data: { label: "End", stepType: "end", config: {} } }
    ]
    canvas = { name: definition.name, description: definition.description || "", nodes, edges: nodes.slice(0, -1).map((node, index) => ({ id: `editor-edge-${index}`, source: node.id, target: nodes[index + 1].id, sourceHandle: "output", targetHandle: "input" })) }
    compileWorkflowCanvas(canvas, saved.version)
  }
  return { ...workflowCanvasSnapshot(canvas), id: String(saved.id), version: saved.version }
}

export const workflowInputSchema = (canvas: ServerWorkflowDefinition): Record<string, unknown> | null => {
  const schema = canvas.nodes.find((node) => node.data.stepType === "start")?.data.config.inputSchema
  if (schema === undefined) return null
  if (!object(schema)) throw new Error("Start input schema must be a JSON object.")
  return schema
}
