import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useWorkflowEditorStore } from "@/store/workflow-editor"
import { WorkflowEditor } from "../WorkflowEditor"
import { validateWithAjv } from "../../../../../../tldw-frontend/lib/ajv"
import { compileWorkflowCanvas, workflowCanvasKey } from "@/services/workflow-editor-definition"

const io = vi.hoisted(() => ({ request: vi.fn(), loadOwner: vi.fn(), ownerController: undefined as AbortController | undefined }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => io.request(...args) }))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: (...args: unknown[]) => io.loadOwner(...args) }))
vi.mock("@/hooks/useMediaQuery", () => ({ useDesktop: () => true }))
vi.mock("../WorkflowCanvas", () => ({ WorkflowCanvas: () => <div aria-label="Workflow canvas" /> }))
vi.mock("../NodePalette", () => ({ NodePalette: () => <div>Palette</div> }))

const initialState = useWorkflowEditorStore.getState()
const validateInputs = (data: unknown, schema: Record<string, unknown>) => validateWithAjv(data, schema, { strict: true })
const nodes = [
  { id: "start", type: "start", position: { x: 0, y: 0 }, data: { label: "Start", stepType: "start", config: { inputSchema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } } } },
  { id: "clean", type: "step", position: { x: 200, y: 0 }, data: { label: "Text Clean", stepType: "text_clean", config: { text: "{{ inputs.text }}", operations: ["strip"] } } },
  { id: "render", type: "step", position: { x: 400, y: 0 }, data: { label: "Template Render", stepType: "template_render", config: { template: "Clean: {{ prev.text }}" } } },
  { id: "end", type: "end", position: { x: 600, y: 0 }, data: { label: "End", stepType: "end", config: {} } }
]
const edges = [
  { id: "e1", source: "start", target: "clean", sourceHandle: "output", targetHandle: "input" },
  { id: "e2", source: "clean", target: "render", sourceHandle: "output", targetHandle: "input" },
  { id: "e3", source: "render", target: "end", sourceHandle: "output", targetHandle: "input" }
]

beforeEach(() => {
  io.request.mockReset()
  io.loadOwner.mockReset()
  const ownerController = new AbortController()
  io.ownerController = ownerController
  io.loadOwner.mockResolvedValue({
    scopeKey: "workflow-owner-1", requestScope: { userId: "1", config: { serverUrl: "http://workflow.test", authMode: "single-user", authSource: "cookie-session" } },
    scopeSignal: ownerController.signal, scopeInvalidatedSignal: ownerController.signal, release: vi.fn()
  })
  io.request.mockImplementation(async ({ path }: { path: string }) => {
    if (path.endsWith("/preflight")) return { valid: true, errors: [], warnings: [] }
    if (path === "/api/v1/workflows") return { id: 17, name: "C12 text workflow", version: 1, description: "", tags: [], is_active: true }
    throw new Error(`Unexpected request: ${path}`)
  })
  useWorkflowEditorStore.setState({ ...initialState, nodes: structuredClone(nodes), edges: structuredClone(edges), workflowName: "C12 text workflow", isDirty: true, stepTypesStatus: "ready", sidebarPanel: "execution" })
})
afterEach(() => {
  cleanup()
  useWorkflowEditorStore.setState(initialState, true)
  vi.restoreAllMocks()
})

describe("workflow editor server integration", () => {
  it("saves a real private definition before acknowledging Save", async () => {
    render(<WorkflowEditor validateInputs={validateInputs} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "Save" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(io.request).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/workflows", method: "POST", body: expect.objectContaining({ name: "C12 text workflow", visibility: "private", steps: [
        expect.objectContaining({ id: "clean", type: "text_clean", config: nodes[1].data.config }),
        expect.objectContaining({ id: "render", type: "template_render", config: nodes[2].data.config })
      ] })
    })))
    expect(useWorkflowEditorStore.getState().workflowId).toBe("17")
    expect(useWorkflowEditorStore.getState().isDirty).toBe(false)
  })

  it("refuses Run without required input and never invents a run ID", async () => {
    const canvas = useWorkflowEditorStore.getState().saveWorkflow()
    useWorkflowEditorStore.setState({ workflowId: "17", workflowVersion: 1, savedCanvasKey: workflowCanvasKey(canvas), isDirty: false })
    render(<WorkflowEditor validateInputs={validateInputs} />)
    const panel = await screen.findByRole("heading", { name: "Execution" })
    const execution = panel.closest("div")?.parentElement?.parentElement
    expect(execution).not.toBeNull()
    await waitFor(() => expect(within(execution as HTMLElement).getByRole("button", { name: "Run" })).toBeEnabled())
    fireEvent.click(within(execution as HTMLElement).getByRole("button", { name: "Run" }))
    await act(async () => { await Promise.resolve() })
    expect(useWorkflowEditorStore.getState().runId).toBeNull()
    expect(io.request.mock.calls.some(([request]) => request.path.includes("/run"))).toBe(false)
    expect(await screen.findByText(/must have required property 'text'/)).toBeInTheDocument()
  })

  it("keeps changes dirty after a failed Save", async () => {
    io.request.mockRejectedValue(new Error("Permission denied"))
    render(<WorkflowEditor validateInputs={validateInputs} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "Save" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    expect(await screen.findByText("Permission denied")).toBeInTheDocument()
    expect(useWorkflowEditorStore.getState().isDirty).toBe(true)
    expect(useWorkflowEditorStore.getState().workflowId).toBeNull()
  })

  it("dispatches valid inputs once and displays actual server steps and output", async () => {
    const canvas = useWorkflowEditorStore.getState().saveWorkflow()
    useWorkflowEditorStore.setState({ workflowId: "17", workflowVersion: 1, savedCanvasKey: workflowCanvasKey(canvas), isDirty: false })
    const actualRun = { run_id: "server-run-44", workflow_id: 17, user_id: "1", definition_version: 1, inputs: { text: "hello" }, status: "succeeded", outputs: { text: "Clean: hello" } }
    io.request.mockImplementation(async ({ path }: { path: string }) => {
      if (path.endsWith("/run?mode=async")) return { ...actualRun, status: "queued", outputs: null }
      if (path.endsWith("/steps")) return { run_id: actualRun.run_id, steps: [{ step_id: "clean", status: "succeeded" }, { step_id: "render", status: "succeeded" }] }
      if (path.endsWith("/server-run-44")) return actualRun
      throw new Error(`Unexpected request: ${path}`)
    })
    render(<WorkflowEditor validateInputs={validateInputs} />)
    const inputs = await screen.findByRole("textbox", { name: "Run inputs" })
    fireEvent.change(inputs, { target: { value: '{"text":"hello"}' } })
    const runButton = within(inputs.parentElement!.parentElement!).getByRole("button", { name: "Run" })
    await waitFor(() => expect(runButton).toBeEnabled())
    fireEvent.click(runButton)
    fireEvent.click(runButton)
    expect(await screen.findByText(/Clean: hello/)).toBeInTheDocument()
    expect(useWorkflowEditorStore.getState().status).toBe("completed")
    expect(useWorkflowEditorStore.getState().runId).toBe("server-run-44")
    expect(Object.values(useWorkflowEditorStore.getState().nodeStates).map((step) => step.status)).toEqual(["success", "success"])
    const dispatches = io.request.mock.calls.map(([request]) => request).filter((request) => request.path.endsWith("/run?mode=async"))
    expect(dispatches).toHaveLength(1)
    expect(dispatches[0]).toEqual(expect.objectContaining({ path: "/api/v1/workflows/17/run?mode=async", method: "POST", body: { inputs: { text: "hello" }, validation_mode: "block", idempotency_key: expect.any(String) }, headers: { "X-TLDW-Expected-User-ID": "1" }, servicePromptConfig: expect.objectContaining({ expectedUserId: "1" }) }))
  })

  it("does not mark newer graph edits clean when a Save acknowledgement arrives", async () => {
    let release: (value: unknown) => void = () => {}
    io.request.mockImplementation(async ({ path }: { path: string }) => path.endsWith("/preflight") ? { valid: true, errors: [], warnings: [] } : new Promise((resolve) => { release = resolve }))
    render(<WorkflowEditor validateInputs={validateInputs} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "Save" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(io.request).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/workflows", method: "POST" })))
    act(() => useWorkflowEditorStore.getState().updateNode("clean", { config: { text: "newer input" } }))
    await act(async () => release({ id: 17, name: "C12 text workflow", version: 1, is_active: true }))
    expect(useWorkflowEditorStore.getState().isDirty).toBe(true)
    expect(useWorkflowEditorStore.getState().nodes[1].data.config).toEqual({ text: "newer input" })
    expect(useWorkflowEditorStore.getState().workflowId).toBe("17")
  })

  it("ignores a Save acknowledgement after the canvas is replaced", async () => {
    let release: (value: unknown) => void = () => {}
    io.request.mockImplementation(async ({ path }: { path: string }) => path.endsWith("/preflight") ? { valid: true, errors: [], warnings: [] } : new Promise((resolve) => { release = resolve }))
    render(<WorkflowEditor validateInputs={validateInputs} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "Save" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(io.request).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/workflows", method: "POST" })))
    act(() => useWorkflowEditorStore.getState().newWorkflow())
    await act(async () => release({ id: 17, name: "C12 text workflow", version: 1, is_active: true }))
    expect(useWorkflowEditorStore.getState().workflowName).toBe("Untitled Workflow")
    expect(useWorkflowEditorStore.getState().workflowId).toBeNull()
  })

  it("reopens the acknowledged server version with its canvas and input schema", async () => {
    const definition = compileWorkflowCanvas(useWorkflowEditorStore.getState().saveWorkflow())
    io.request.mockImplementation(async ({ path }: { path: string }) => {
      if (path === "/api/v1/workflows") return [{ id: 17, name: "C12 text workflow", version: 1, is_active: true }]
      if (path === "/api/v1/workflows/17") return { id: 17, name: "C12 text workflow", version: 1, is_active: true, definition }
      throw new Error(`Unexpected request: ${path}`)
    })
    vi.spyOn(window, "confirm").mockReturnValue(true)
    render(<WorkflowEditor validateInputs={validateInputs} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "Open saved workflow" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "Open saved workflow" }))
    const select = await screen.findByRole("combobox", { name: "Saved workflow" })
    fireEvent.mouseDown(select)
    fireEvent.click(await screen.findByText("C12 text workflow — version 1"))
    const openButton = await screen.findByRole("button", { name: "Open selected workflow" })
    await waitFor(() => expect(openButton).toBeEnabled())
    fireEvent.click(openButton)
    await waitFor(() => expect(useWorkflowEditorStore.getState().workflowId).toBe("17"))
    expect(useWorkflowEditorStore.getState().nodes[0].data.config.inputSchema).toEqual(nodes[0].data.config.inputSchema)
    expect(useWorkflowEditorStore.getState().savedCanvasKey).toBe(workflowCanvasKey(useWorkflowEditorStore.getState().saveWorkflow()))
    expect(useWorkflowEditorStore.getState().isDirty).toBe(false)
  })

  it("sends real pause, resume and Stop controls and projects refreshed server status", async () => {
    let serverStatus = "running"
    useWorkflowEditorStore.setState({ workflowId: "17", workflowVersion: 1, ownerScopeKey: "workflow-owner-1", runId: "actual-running", runWorkflowId: 17, runWorkflowVersion: 1, status: "running" })
    io.request.mockImplementation(async ({ path, method }: { path: string; method: string }) => {
      if (method === "POST") { serverStatus = path.endsWith("/pause") ? "paused" : path.endsWith("/resume") ? "running" : "cancelled"; return { ok: true, result: "applied" } }
      if (path.endsWith("/steps")) return { run_id: "actual-running", steps: [{ step_id: "clean", status: serverStatus === "cancelled" ? "cancelled" : "running" }] }
      return { run_id: "actual-running", workflow_id: 17, user_id: "1", definition_version: 1, status: serverStatus, inputs: { text: "hello" } }
    })
    render(<WorkflowEditor validateInputs={validateInputs} />)
    const pause = await screen.findByRole("button", { name: "Pause" })
    await waitFor(() => expect(pause).toBeEnabled())
    fireEvent.click(pause)
    const resume = await screen.findByRole("button", { name: "Resume" })
    expect(useWorkflowEditorStore.getState().status).toBe("paused")
    await waitFor(() => expect(resume).toBeEnabled())
    fireEvent.click(resume)
    await waitFor(() => expect(useWorkflowEditorStore.getState().status).toBe("running"))
    const stop = screen.getByRole("button", { name: "Stop" })
    await waitFor(() => expect(stop).toBeEnabled())
    fireEvent.click(stop)
    await waitFor(() => expect(useWorkflowEditorStore.getState().status).toBe("cancelled"))
    expect(io.request.mock.calls.map(([request]) => request).filter((request) => request.method === "POST").map((request) => request.path)).toEqual(["/api/v1/workflows/runs/actual-running/pause", "/api/v1/workflows/runs/actual-running/resume", "/api/v1/workflows/runs/actual-running/cancel"])
  })

  it("retains the verified running state when server control is denied", async () => {
    useWorkflowEditorStore.setState({ workflowId: "17", workflowVersion: 1, ownerScopeKey: "workflow-owner-1", runId: "actual-running", runWorkflowId: 17, runWorkflowVersion: 1, status: "running" })
    io.request.mockImplementation(async ({ path, method }: { path: string; method: string }) => {
      if (method === "POST") throw new Error("Control permission denied")
      if (path.endsWith("/steps")) return { run_id: "actual-running", steps: [] }
      return { run_id: "actual-running", workflow_id: 17, user_id: "1", definition_version: 1, status: "running", inputs: {} }
    })
    render(<WorkflowEditor validateInputs={validateInputs} />)
    const stop = await screen.findByRole("button", { name: "Stop" })
    await waitFor(() => expect(stop).toBeEnabled())
    fireEvent.click(stop)
    expect(await screen.findByText("Control permission denied")).toBeInTheDocument()
    expect(useWorkflowEditorStore.getState().status).toBe("running")
    expect(useWorkflowEditorStore.getState().runId).toBe("actual-running")
  })

  it("refuses a held Save result after canonical owner invalidation", async () => {
    let release: (value: unknown) => void = () => {}
    io.request.mockImplementation(async ({ path }: { path: string }) => path.endsWith("/preflight") ? { valid: true, errors: [], warnings: [] } : new Promise((resolve) => { release = resolve }))
    render(<WorkflowEditor validateInputs={validateInputs} />)
    await waitFor(() => expect(screen.getByRole("button", { name: "Save" })).toBeEnabled())
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(io.request).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/workflows", method: "POST" })))
    act(() => io.ownerController?.abort())
    await act(async () => release({ id: 17, name: "C12 text workflow", version: 1, is_active: true }))
    expect(useWorkflowEditorStore.getState().workflowId).toBeNull()
    expect(useWorkflowEditorStore.getState().ownerScopeKey).toBeNull()
    expect(useWorkflowEditorStore.getState().nodes.some((node) => node.id === "clean")).toBe(false)
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
  })
})
