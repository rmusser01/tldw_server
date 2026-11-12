import { beforeEach, describe, expect, it, vi } from "vitest"
import { controlWorkflowRun, getWorkflowDefinition, getWorkflowRun, getWorkflowRunSteps, listWorkflowDefinitions, reviewWorkflowStep, runSavedWorkflow, saveWorkflowDefinition } from "../workflows"
import type { WorkflowRequestOptions } from "../workflows"

const request = vi.hoisted(() => vi.fn())
vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => request(...args) }))
const options: WorkflowRequestOptions = { requestScope: { userId: "1", config: { serverUrl: "http://workflow.test", authMode: "single-user", authSource: "cookie-session" } }, signal: new AbortController().signal }
const definition = { name: "Own", version: 2, visibility: "private" as const, steps: [{ id: "clean", type: "text_clean", config: { text: "{{ inputs.text }}" } }] }
const summary = { id: 7, name: "Own", version: 2, is_active: true }
const run = { run_id: "owned/run", workflow_id: 7, user_id: "1", definition_version: 2, status: "queued", inputs: { text: "hello" } }

beforeEach(() => { request.mockReset() })
describe("workflow editor API contracts", () => {
  it.each([
    ["list", () => listWorkflowDefinitions(options), "/api/v1/workflows", "GET", [summary]],
    ["get", () => getWorkflowDefinition(7, options), "/api/v1/workflows/7", "GET", { ...summary, definition }],
    ["create", () => saveWorkflowDefinition(definition, null, options), "/api/v1/workflows", "POST", summary],
    ["version", () => saveWorkflowDefinition(definition, 6, options), "/api/v1/workflows/6/versions", "POST", summary],
    ["run", () => runSavedWorkflow(7, { text: "hello" }, "stable-attempt", options), "/api/v1/workflows/7/run?mode=async", "POST", run],
    ["status", () => getWorkflowRun("owned/run", options), "/api/v1/workflows/runs/owned%2Frun", "GET", run],
    ["steps", () => getWorkflowRunSteps("owned/run", options), "/api/v1/workflows/runs/owned%2Frun/steps", "GET", { run_id: "owned/run", steps: [] }],
    ["control", () => controlWorkflowRun("owned/run", "cancel", options), "/api/v1/workflows/runs/owned%2Frun/cancel", "POST", { ok: true, result: "applied" }],
    ["approve", () => reviewWorkflowStep("owned/run", "review/step", "approve", options), "/api/v1/workflows/runs/owned%2Frun/steps/review%2Fstep/approve", "POST", { ok: true }]
  ] as const)("binds %s to the canonical request owner and encoded resource", async (_name, invoke, path, method, payload) => {
    request.mockResolvedValue(payload)
    await invoke()
    expect(request).toHaveBeenCalledWith(expect.objectContaining({ path, method, abortSignal: options.signal, timeoutMs: 15000, headers: { "X-TLDW-Expected-User-ID": "1" }, servicePromptConfig: expect.objectContaining({ serverUrl: "http://workflow.test", expectedUserId: "1" }) }))
  })
  it("sends the exact input and retained idempotency key", async () => {
    request.mockResolvedValue(run)
    await runSavedWorkflow(7, { text: "hello" }, "stable-attempt", options)
    expect(request.mock.calls[0][0].body).toEqual({ inputs: { text: "hello" }, idempotency_key: "stable-attempt", validation_mode: "block" })
  })
  it.each([{}, { ...run, run_id: "different" }, { ...run, user_id: "other-owner" }, { ...run, status: "fabricated-complete" }])("refuses malformed or foreign run responses", async (payload) => {
    request.mockResolvedValue(payload)
    await expect(getWorkflowRun("owned/run", options)).rejects.toThrow()
  })
  it.each([401, 403])("does not turn HTTP %s into an acknowledged save or run", async (status) => {
    request.mockRejectedValue(Object.assign(new Error(`HTTP ${status}`), { status }))
    await expect(saveWorkflowDefinition(definition, null, options)).rejects.toThrow(`HTTP ${status}`)
    await expect(runSavedWorkflow(7, {}, "stable-attempt", options)).rejects.toThrow(`HTTP ${status}`)
  })
  it("refuses a control transition rejected by the server", async () => {
    request.mockResolvedValue({ ok: true, result: "invalid_state" })
    await expect(controlWorkflowRun("owned/run", "pause", options)).rejects.toThrow("refused")
  })
})
