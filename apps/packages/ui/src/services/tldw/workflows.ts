import { bgRequest } from "@/services/background-proxy"
import { z } from "zod"
import { requestScopeFields, type ServicePromptRequestScope } from "./domains/service-prompts"
import type {
  WorkflowDefinitionPayload,
  SavedWorkflowDefinition,
  SavedWorkflowSummary,
  ServerWorkflowRun,
  WorkflowRunStepSummary,
  WorkflowPreflightRequest,
  WorkflowPreflightResult,
  WorkflowRunInvestigation,
  WorkflowStepAttemptsResponse,
  WorkflowStepTypeInfo
} from "@/types/workflow-editor"

const STEP_TYPES_TTL_MS = 5 * 60 * 1000

let stepTypesPromise: Promise<WorkflowStepTypeInfo[]> | null = null
let stepTypesFetchedAt: number | null = null

const normalizeStepTypes = (payload: unknown): WorkflowStepTypeInfo[] => {
  if (!Array.isArray(payload)) return []
  return payload
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null
      const data = entry as WorkflowStepTypeInfo
      if (!data.name) return null
      return data
    })
    .filter(Boolean) as WorkflowStepTypeInfo[]
}

export const getWorkflowStepTypes = async (
  force = false
): Promise<WorkflowStepTypeInfo[]> => {
  const now = Date.now()
  if (!force && stepTypesPromise) {
    if (!stepTypesFetchedAt) return stepTypesPromise
    if (now - stepTypesFetchedAt < STEP_TYPES_TTL_MS) {
      return stepTypesPromise
    }
  }

  stepTypesPromise = (async () => {
    try {
      const data = await bgRequest<WorkflowStepTypeInfo[]>({
        path: "/api/v1/workflows/step-types",
        method: "GET"
      })
      stepTypesFetchedAt = Date.now()
      return normalizeStepTypes(data)
    } catch (error) {
      stepTypesPromise = null
      stepTypesFetchedAt = null
      throw error
    }
  })()

  return stepTypesPromise
}

export const getWorkflowInvestigation = async (
  runId: string,
  options?: WorkflowRequestOptions
): Promise<WorkflowRunInvestigation> =>
  bgRequest<WorkflowRunInvestigation>({
    path: `/api/v1/workflows/runs/${encodeURIComponent(runId)}/investigation`,
    method: "GET",
    ...scoped(options)
  })

export const getWorkflowStepAttempts = async (
  runId: string,
  stepId: string,
  options?: WorkflowRequestOptions
): Promise<WorkflowStepAttemptsResponse> =>
  bgRequest<WorkflowStepAttemptsResponse>({
    path: `/api/v1/workflows/runs/${encodeURIComponent(runId)}/steps/${encodeURIComponent(stepId)}/attempts`,
    method: "GET",
    ...scoped(options)
  })

export const preflightWorkflowDefinition = async (
  body: WorkflowPreflightRequest,
  options?: WorkflowRequestOptions
): Promise<WorkflowPreflightResult> =>
  bgRequest<WorkflowPreflightResult>({
    path: "/api/v1/workflows/preflight",
    method: "POST",
    body,
    ...scoped(options)
  })

export type WorkflowRequestOptions = { requestScope: ServicePromptRequestScope; signal?: AbortSignal }
const scoped = (options?: WorkflowRequestOptions) => options ? { ...requestScopeFields(options.requestScope), abortSignal: options.signal, timeoutMs: 15000 } : {}
const summarySchema = z.object({ id: z.number().int().positive(), name: z.string().min(1), version: z.number().int().positive(), is_active: z.boolean(), description: z.string().nullable().optional(), tags: z.array(z.string()).optional() })
const stepSchema = z.object({ id: z.string().min(1), type: z.string().min(1), name: z.string().nullable().optional(), config: z.record(z.string(), z.unknown()).optional(), retry: z.number().nullable().optional(), timeout_seconds: z.number().nullable().optional(), on_success: z.string().nullable().optional(), on_failure: z.string().nullable().optional(), on_timeout: z.string().nullable().optional() })
const definitionSchema = z.object({ name: z.string().min(1), version: z.number().int().positive().optional(), description: z.string().nullable().optional(), steps: z.array(stepSchema).min(1), metadata: z.record(z.string(), z.unknown()).optional() }).passthrough()
const runSchema = z.object({ run_id: z.string().min(1), workflow_id: z.number().int().positive().nullable().optional(), user_id: z.string().nullable().optional(), status: z.enum(["queued", "running", "paused", "waiting_human", "waiting_approval", "succeeded", "failed", "cancelled"]), inputs: z.record(z.string(), z.unknown()), outputs: z.record(z.string(), z.unknown()).nullable().optional(), error: z.string().nullable().optional(), status_reason: z.string().nullable().optional(), definition_version: z.number().int().positive().nullable().optional() })

export const listWorkflowDefinitions = async (options: WorkflowRequestOptions): Promise<SavedWorkflowSummary[]> =>
  z.array(summarySchema).parse(await bgRequest({ path: "/api/v1/workflows", method: "GET", ...scoped(options) }))

export const getWorkflowDefinition = async (id: number, options: WorkflowRequestOptions): Promise<SavedWorkflowDefinition> => {
  const result = summarySchema.pick({ id: true, name: true, version: true, is_active: true }).extend({ definition: definitionSchema }).parse(await bgRequest({ path: `/api/v1/workflows/${id}`, method: "GET", ...scoped(options) }))
  if (result.id !== id || result.name !== result.definition.name || (result.definition.version !== undefined && result.version !== result.definition.version)) throw new Error("Saved workflow identity does not match the requested definition.")
  return result as SavedWorkflowDefinition
}

export const saveWorkflowDefinition = async (definition: WorkflowDefinitionPayload, id: number | null, options: WorkflowRequestOptions): Promise<SavedWorkflowSummary> => {
  const result = summarySchema.parse(await bgRequest({ path: id === null ? "/api/v1/workflows" : `/api/v1/workflows/${id}/versions`, method: "POST", body: definition, ...scoped(options) }))
  if (result.name !== definition.name || result.version !== definition.version || !result.is_active) throw new Error("The server did not acknowledge the requested workflow version.")
  return result
}

const readRun = (payload: unknown, options: WorkflowRequestOptions): ServerWorkflowRun => {
  const run = runSchema.parse(payload)
  if (run.user_id != null && options.requestScope.userId != null && run.user_id !== String(options.requestScope.userId)) throw new Error("Workflow run owner changed.")
  return run
}

export const runSavedWorkflow = async (id: number, inputs: Record<string, unknown>, idempotencyKey: string, options: WorkflowRequestOptions): Promise<ServerWorkflowRun> => {
  const run = readRun(await bgRequest({ path: `/api/v1/workflows/${id}/run?mode=async`, method: "POST", body: { inputs, idempotency_key: idempotencyKey, validation_mode: "block" }, ...scoped(options) }), options)
  if (run.workflow_id !== id) throw new Error("Workflow run does not belong to the saved definition.")
  return run
}

export const getWorkflowRun = async (runId: string, options: WorkflowRequestOptions): Promise<ServerWorkflowRun> => {
  const run = readRun(await bgRequest({ path: `/api/v1/workflows/runs/${encodeURIComponent(runId)}`, method: "GET", ...scoped(options) }), options)
  if (run.run_id !== runId) throw new Error("Workflow status belongs to a different run.")
  return run
}

export const getWorkflowRunSteps = async (runId: string, options: WorkflowRequestOptions): Promise<WorkflowRunStepSummary[]> => {
  const result = z.object({ run_id: z.string(), steps: z.array(z.object({ step_id: z.string().min(1), status: runSchema.shape.status, started_at: z.string().nullable().optional(), ended_at: z.string().nullable().optional(), error: z.string().nullable().optional() }).passthrough()) }).parse(await bgRequest({ path: `/api/v1/workflows/runs/${encodeURIComponent(runId)}/steps`, method: "GET", ...scoped(options) }))
  if (result.run_id !== runId) throw new Error("Workflow steps belong to a different run.")
  return result.steps
}

export const controlWorkflowRun = async (runId: string, action: "pause" | "resume" | "cancel", options: WorkflowRequestOptions): Promise<void> => {
  const result = z.object({ ok: z.literal(true), result: z.enum(["applied", "already_applied", "invalid_state", "conflict"]) }).parse(await bgRequest({ path: `/api/v1/workflows/runs/${encodeURIComponent(runId)}/${action}`, method: "POST", ...scoped(options) }))
  if (result.result !== "applied" && result.result !== "already_applied") throw new Error(`Workflow ${action} was refused: ${result.result}.`)
}

export const reviewWorkflowStep = async (runId: string, stepId: string, action: "approve" | "reject", options: WorkflowRequestOptions): Promise<void> => {
  const result = z.object({ ok: z.literal(true) }).passthrough().parse(await bgRequest({ path: `/api/v1/workflows/runs/${encodeURIComponent(runId)}/steps/${encodeURIComponent(stepId)}/${action}`, method: "POST", body: {}, ...scoped(options) }))
  if (!result.ok) throw new Error("Workflow review was not acknowledged.")
}
