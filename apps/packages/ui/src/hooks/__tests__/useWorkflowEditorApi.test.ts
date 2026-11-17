import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { useWorkflowEditorApi } from "../useWorkflowEditorApi"
import { useWorkflowEditorStore } from "@/store/workflow-editor"
import { compileWorkflowCanvas, workflowCanvasKey } from "@/services/workflow-editor-definition"

const io = vi.hoisted(() => ({ request: vi.fn(), loadOwner: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => io.request(...args) }))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: (...args: unknown[]) => io.loadOwner(...args) }))
const initial = useWorkflowEditorStore.getState()
beforeEach(() => {
  io.request.mockReset()
  io.loadOwner.mockReset()
  const controller = new AbortController()
  io.loadOwner.mockResolvedValue({ scopeKey: "owner-1", requestScope: { config: { serverUrl: "http://workflow.test", authMode: "single-user", authSource: "cookie-session" }, userId: "1" }, scopeSignal: controller.signal, scopeInvalidatedSignal: controller.signal, release: vi.fn() })
  useWorkflowEditorStore.setState(initial, true)
  useWorkflowEditorStore.getState().newWorkflow()
  const store = useWorkflowEditorStore.getState()
  const start = store.nodes[0].id
  const end = store.nodes[1].id
  const step = store.addNode({ type: "text_clean", position: { x: 300, y: 200 }, data: { config: { text: "original draft" } } })
  store.addEdge({ source: start, target: step, sourceHandle: "output", targetHandle: "input" })
  store.addEdge({ source: step, target: end, sourceHandle: "output", targetHandle: "input" })
})
afterEach(() => { cleanup(); useWorkflowEditorStore.setState(initial, true) })

it("preserves an edit made while a saved workflow Open request is held", async () => {
  const definition = compileWorkflowCanvas(useWorkflowEditorStore.getState().saveWorkflow())
  let release: (value: unknown) => void = () => {}
  io.request.mockImplementation(() => new Promise((resolve) => { release = resolve }))
  const { result } = renderHook(() => useWorkflowEditorApi())
  await waitFor(() => expect(result.current.ready).toBe(true))
  let opening: Promise<unknown> | undefined
  act(() => { opening = result.current.open(9) })
  await waitFor(() => expect(io.request).toHaveBeenCalled())
  const stepId = useWorkflowEditorStore.getState().nodes.find((node) => node.data.stepType === "text_clean")!.id
  act(() => useWorkflowEditorStore.getState().updateNode(stepId, { config: { text: "new edit while waiting" } }))
  await act(async () => { release({ id: 9, name: definition.name, version: 1, is_active: true, definition }); await opening })
  expect(useWorkflowEditorStore.getState().nodes.find((node) => node.id === stepId)!.data.config.text).toBe("new edit while waiting")
  expect(useWorkflowEditorStore.getState().workflowId).toBeNull()
  expect(result.current.error).toMatch(/changed while opening/)
})

it("keeps an active server run bound to its original immutable version after Save and remount", async () => {
  const run = { run_id: "original-run", workflow_id: 7, definition_version: 1, user_id: "1", status: "running", inputs: {} }
  io.request.mockImplementation(async ({ path, body }: { path: string; body?: { name: string; version: number } }) => {
    if (path === "/api/v1/workflows/preflight") return { valid: true, errors: [] }
    if (path === "/api/v1/workflows/7/versions") return { id: 8, name: body!.name, version: body!.version, is_active: true }
    if (path.endsWith("/steps")) return { run_id: run.run_id, steps: [] }
    return run
  })
  const current = useWorkflowEditorStore.getState()
  useWorkflowEditorStore.setState({ workflowId: "7", workflowVersion: 1, savedCanvasKey: workflowCanvasKey(current.saveWorkflow()), isDirty: false })
  const first = renderHook(() => useWorkflowEditorApi())
  await waitFor(() => expect(first.result.current.ready).toBe(true))
  await act(async () => { await first.result.current.run("{}") })
  await waitFor(() => expect(useWorkflowEditorStore.getState().status).toBe("running"))
  const stepId = useWorkflowEditorStore.getState().nodes.find((node) => node.data.stepType === "text_clean")!.id
  act(() => useWorkflowEditorStore.getState().updateNode(stepId, { config: { text: "next saved version" } }))
  await act(async () => { await first.result.current.save() })
  expect(useWorkflowEditorStore.getState().workflowVersion).toBe(2)
  first.unmount()
  io.request.mockClear()
  const reopened = renderHook(() => useWorkflowEditorApi())
  await waitFor(() => expect(reopened.result.current.ready).toBe(true))
  await waitFor(() => expect(io.request).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/workflows/runs/original-run" })))
  await act(async () => { await Promise.resolve() })
  expect(reopened.result.current.error).toBeNull()
  expect(useWorkflowEditorStore.getState().runId).toBe("original-run")
  expect(useWorkflowEditorStore.getState().status).toBe("running")
})

it.each(["obsolete waiting step", "foreign investigation"])("refuses %s when projecting a failed run", async (kind) => {
  const run = { run_id: "failed-run", workflow_id: 7, definition_version: 1, user_id: "1", status: "failed", inputs: {}, error: "Actual step failed" }
  io.request.mockImplementation(async ({ path }: { path: string }) => {
    if (path.endsWith("/steps")) return { run_id: run.run_id, steps: [{ step_id: "review", status: "waiting_human" }] }
    if (path.endsWith("/investigation")) return { run_id: kind === "foreign investigation" ? "different-run" : run.run_id, status: "failed", schema_version: 1, derived_from_event_seq: 1, attempts: [], evidence: {}, recommended_actions: [] }
    return run
  })
  const current = useWorkflowEditorStore.getState()
  useWorkflowEditorStore.setState({ workflowId: "7", workflowVersion: 1, savedCanvasKey: workflowCanvasKey(current.saveWorkflow()), isDirty: false })
  const { result } = renderHook(() => useWorkflowEditorApi())
  await waitFor(() => expect(result.current.ready).toBe(true))
  await act(async () => { await result.current.run("{}") })
  await waitFor(() => expect(io.request).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/workflows/runs/failed-run/investigation" })))
  await act(async () => { await Promise.resolve() })
  if (kind === "obsolete waiting step") expect(useWorkflowEditorStore.getState().pendingApproval).toBeNull()
  else {
    expect(useWorkflowEditorStore.getState().runInvestigation).toBeNull()
    expect(useWorkflowEditorStore.getState().runInvestigationError).toMatch(/different run/)
  }
})

it("reuses the original idempotency key after a lost acknowledgement and admits only one server run", async () => {
  let submissions = 0
  const run = { run_id: "recovered-run", workflow_id: 7, definition_version: 1, user_id: "1", status: "succeeded", inputs: {} }
  io.request.mockImplementation(async ({ path }: { path: string }) => {
    if (path.endsWith("/steps")) return { run_id: run.run_id, steps: [] }
    if (path.includes("/run?")) { if (++submissions === 1) throw new Error("Acknowledgement lost"); return run }
    return run
  })
  const current = useWorkflowEditorStore.getState()
  useWorkflowEditorStore.setState({ workflowId: "7", workflowVersion: 1, savedCanvasKey: workflowCanvasKey(current.saveWorkflow()), isDirty: false })
  const { result } = renderHook(() => useWorkflowEditorApi())
  await waitFor(() => expect(result.current.ready).toBe(true))
  await act(async () => { await result.current.run("{}") })
  expect(useWorkflowEditorStore.getState().runId).toBeNull()
  await act(async () => { await result.current.run("{}"); await result.current.run("{}") })
  const requests = io.request.mock.calls.map(([request]) => request).filter((request) => request.path.includes("/run?"))
  expect(requests).toHaveLength(2)
  expect(requests[1].body.idempotency_key).toBe(requests[0].body.idempotency_key)
  expect(useWorkflowEditorStore.getState().runId).toBe("recovered-run")
})

it("submits the current server review and refreshes its actual outcome", async () => {
  const run = { run_id: "review-run", workflow_id: 7, definition_version: 1, user_id: "1", status: "waiting_human", inputs: {}, outputs: { reason: "Review these extracted claims", claims: ["a claim"] } }
  io.request.mockImplementation(async ({ path }: { path: string }) => {
    if (path.endsWith("/approve")) { run.status = "running"; return { ok: true } }
    if (path.endsWith("/steps")) return { run_id: run.run_id, steps: [{ step_id: "review", status: run.status }] }
    return { ...run }
  })
  const current = useWorkflowEditorStore.getState()
  useWorkflowEditorStore.setState({ workflowId: "7", workflowVersion: 1, savedCanvasKey: workflowCanvasKey(current.saveWorkflow()), isDirty: false })
  const { result } = renderHook(() => useWorkflowEditorApi())
  await waitFor(() => expect(result.current.ready).toBe(true))
  await act(async () => { await result.current.run("{}") })
  await waitFor(() => expect(useWorkflowEditorStore.getState().pendingApproval?.dataToReview).toEqual(run.outputs))
  await act(async () => { await result.current.review("approve") })
  expect(io.request).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/workflows/runs/review-run/steps/review/approve", method: "POST", body: {}, headers: { "X-TLDW-Expected-User-ID": "1" } }))
  expect(useWorkflowEditorStore.getState().status).toBe("running")
  expect(useWorkflowEditorStore.getState().pendingApproval).toBeNull()
})

it("ignores a held run admission after the current canvas is replaced", async () => {
  let release: (value: unknown) => void = () => {}
  io.request.mockImplementation(() => new Promise((resolve) => { release = resolve }))
  const current = useWorkflowEditorStore.getState()
  useWorkflowEditorStore.setState({ workflowId: "7", workflowVersion: 1, savedCanvasKey: workflowCanvasKey(current.saveWorkflow()), isDirty: false })
  const { result } = renderHook(() => useWorkflowEditorApi())
  await waitFor(() => expect(result.current.ready).toBe(true))
  let admission: Promise<unknown> | undefined
  act(() => { admission = result.current.run("{}") })
  await waitFor(() => expect(io.request).toHaveBeenCalled())
  act(() => useWorkflowEditorStore.getState().newWorkflow())
  await act(async () => { release({ run_id: "late-run", workflow_id: 7, definition_version: 1, status: "queued", inputs: {} }); await admission })
  expect(useWorkflowEditorStore.getState().runId).toBeNull()
  expect(useWorkflowEditorStore.getState().status).toBe("idle")
  expect(io.request).toHaveBeenCalledTimes(1)
})

it("retains an unknown server submission across remount and refuses different inputs until explicit reset", async () => {
  let submissions = 0
  io.request.mockImplementation(async ({ path }: { path: string }) => {
    if (path.endsWith("/steps")) return { run_id: "recovered-after-remount", steps: [] }
    if (path.includes("/run?")) { if (++submissions === 1) throw new Error("Acknowledgement lost"); return { run_id: "recovered-after-remount", workflow_id: 7, definition_version: 1, status: "succeeded", inputs: { text: "original accepted input" } } }
    return { run_id: "recovered-after-remount", workflow_id: 7, definition_version: 1, status: "succeeded", inputs: {} }
  })
  const current = useWorkflowEditorStore.getState()
  useWorkflowEditorStore.setState({ workflowId: "7", workflowVersion: 1, savedCanvasKey: workflowCanvasKey(current.saveWorkflow()), isDirty: false })
  const first = renderHook(() => useWorkflowEditorApi())
  await waitFor(() => expect(first.result.current.ready).toBe(true))
  await act(async () => { await first.result.current.run('{"text":"original accepted input"}') })
  const original = io.request.mock.calls.find(([request]) => request.path.includes("/run?"))![0]
  first.unmount()
  const reopened = renderHook(() => useWorkflowEditorApi())
  await waitFor(() => expect(reopened.result.current.ready).toBe(true))
  await act(async () => { await reopened.result.current.run('{"text":"a different input"}') })
  expect(submissions).toBe(1)
  expect(reopened.result.current.error).toMatch(/unknown outcome/)
  await act(async () => { await reopened.result.current.run('{"text":"original accepted input"}') })
  const requests = io.request.mock.calls.map(([request]) => request).filter((request) => request.path.includes("/run?"))
  expect(requests).toHaveLength(2)
  expect(requests[1].body.idempotency_key).toBe(original.body.idempotency_key)
  expect(useWorkflowEditorStore.getState().runId).toBe("recovered-after-remount")
})
