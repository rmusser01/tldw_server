import { useEffect, useRef, useState } from "react"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { compileWorkflowCanvas, reopenWorkflowDefinition, workflowCanvasKey, workflowInputSchema } from "@/services/workflow-editor-definition"
import { controlWorkflowRun, getWorkflowDefinition, getWorkflowInvestigation, getWorkflowRun, getWorkflowRunSteps, listWorkflowDefinitions, preflightWorkflowDefinition, reviewWorkflowStep, runSavedWorkflow, saveWorkflowDefinition, type WorkflowRequestOptions } from "@/services/tldw/workflows"
import { useWorkflowEditorStore } from "@/store/workflow-editor"
import type { SavedWorkflowSummary, ServerWorkflowRun, WorkflowInputValidator, WorkflowRunStepSummary, StepExecutionState } from "@/types/workflow-editor"

const terminal = (status: ServerWorkflowRun["status"]) => ["succeeded", "failed", "cancelled"].includes(status)
const timestamp = (value: unknown): number | undefined => typeof value === "string" && Number.isFinite(Date.parse(value)) ? Date.parse(value) : undefined
const message = (error: unknown) => error instanceof Error ? error.message : "Workflow request failed."

/** Bind editor requests and async projections to the verified owner and view. */
export const useWorkflowEditorApi = (validateInputs?: WorkflowInputValidator) => {
  const owner = useRef<ServicePromptSnapshot | null>(null)
  const mounted = useRef(false)
  const busy = useRef(false)
  const pollTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const polling = useRef<Promise<void> | null>(null)
  const operationController = useRef<AbortController | null>(null)
  const runBinding = useRef<{ id: string; view: number; workflowId: number; version: number; owner: ServicePromptSnapshot } | null>(null)
  const refreshRef = useRef<(binding: NonNullable<typeof runBinding.current>) => Promise<void>>(async () => {})
  const [ready, setReady] = useState(false)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [output, setOutput] = useState<Record<string, unknown> | null>(null)
  const [definitions, setDefinitions] = useState<SavedWorkflowSummary[]>([])

  const currentOwner = (snapshot: ServicePromptSnapshot) => mounted.current && owner.current === snapshot && !snapshot.scopeSignal.aborted && !snapshot.scopeInvalidatedSignal.aborted && useWorkflowEditorStore.getState().ownerScopeKey === snapshot.scopeKey
  const currentRun = (binding: NonNullable<typeof runBinding.current>) => currentOwner(binding.owner) && runBinding.current === binding && useWorkflowEditorStore.getState().viewRevision === binding.view && useWorkflowEditorStore.getState().runId === binding.id
  const requestOptions = (snapshot: ServicePromptSnapshot, signal = snapshot.scopeSignal): WorkflowRequestOptions => ({ requestScope: snapshot.requestScope, signal })
  const stopPolling = () => { clearTimeout(pollTimer.current); pollTimer.current = undefined }

  useEffect(() => {
    mounted.current = true
    const controller = new AbortController()
    let snapshot: ServicePromptSnapshot | null = null
    const revoke = () => {
      if (!mounted.current || owner.current !== snapshot) return
      owner.current = null
      operationController.current?.abort()
      runBinding.current = null
      stopPolling()
      useWorkflowEditorStore.getState().newWorkflow()
      useWorkflowEditorStore.setState({ ownerScopeKey: null })
      setReady(false)
      setOutput(null)
      setDefinitions([])
      setError("The authenticated workflow owner changed. Reopen the editor to reconnect.")
    }
    void loadServicePromptSnapshot([], { signal: controller.signal }).then((resolved) => {
      if (!mounted.current || controller.signal.aborted || resolved.scopeSignal.aborted || resolved.scopeInvalidatedSignal.aborted) { resolved.release(); return }
      snapshot = resolved
      const state = useWorkflowEditorStore.getState()
      if (state.ownerScopeKey && state.ownerScopeKey !== resolved.scopeKey) state.newWorkflow()
      useWorkflowEditorStore.setState({ ownerScopeKey: resolved.scopeKey })
      owner.current = resolved
      resolved.scopeInvalidatedSignal.addEventListener("abort", revoke, { once: true })
      setReady(true)
      if (state.ownerScopeKey === resolved.scopeKey && state.runId && state.runWorkflowId && state.runWorkflowVersion) {
        const binding = { id: state.runId, workflowId: state.runWorkflowId, version: state.runWorkflowVersion, view: state.viewRevision, owner: resolved }
        runBinding.current = binding
        void refreshRef.current(binding)
      }
    }).catch((failure) => { if (mounted.current && !controller.signal.aborted) setError(message(failure)) })
    const unsubscribe = useWorkflowEditorStore.subscribe((state, previous) => {
      if (state.viewRevision !== previous.viewRevision || (previous.runId && !state.runId)) {
        operationController.current?.abort()
        runBinding.current = null
        stopPolling()
        setOutput(null)
        setError(null)
      }
    })
    return () => {
      mounted.current = false
      controller.abort()
      operationController.current?.abort()
      stopPolling()
      runBinding.current = null
      unsubscribe()
      snapshot?.scopeInvalidatedSignal.removeEventListener("abort", revoke)
      snapshot?.release()
      owner.current = null
    }
  }, [])

  const perform = async <T,>(action: (snapshot: ServicePromptSnapshot, view: number, options: WorkflowRequestOptions, isCurrent: () => boolean) => Promise<T>): Promise<T | undefined> => {
    if (busy.current) return undefined
    const snapshot = owner.current
    if (!snapshot || !currentOwner(snapshot)) { setError("Connect to the server before using workflows."); return undefined }
    const view = useWorkflowEditorStore.getState().viewRevision
    const controller = new AbortController()
    const abort = () => controller.abort()
    snapshot.scopeSignal.addEventListener("abort", abort, { once: true })
    operationController.current = controller
    const isCurrent = () => currentOwner(snapshot) && !controller.signal.aborted && useWorkflowEditorStore.getState().viewRevision === view
    busy.current = true
    setPending(true)
    setError(null)
    try { return await action(snapshot, view, requestOptions(snapshot, controller.signal), isCurrent) }
    catch (failure) { if (isCurrent()) setError(message(failure)); return undefined }
    finally {
      snapshot.scopeSignal.removeEventListener("abort", abort)
      if (operationController.current === controller) operationController.current = null
      busy.current = false
      if (mounted.current) setPending(false)
    }
  }

  const refresh = async (binding = runBinding.current): Promise<void> => {
    if (!binding || !currentRun(binding)) return
    while (polling.current) { await polling.current; if (!currentRun(binding)) return }
    stopPolling()
    const task = (async () => {
      try {
        const options = requestOptions(binding.owner)
        const [run, steps] = await Promise.all([getWorkflowRun(binding.id, options), getWorkflowRunSteps(binding.id, options)])
        if (!currentRun(binding)) return
        if (run.workflow_id !== binding.workflowId || run.definition_version !== binding.version) throw new Error("Workflow status belongs to a different saved version.")
        project(run, steps)
        if (run.status === "failed") await inspect(binding)
        if (!terminal(run.status) && currentRun(binding)) pollTimer.current = setTimeout(() => void refresh(binding), 1500)
      } catch (failure) {
        if (!currentRun(binding)) return
        setError(message(failure))
        // A failed refresh cannot change the last verified run state.
        pollTimer.current = setTimeout(() => void refresh(binding), 5000)
      }
    })()
    polling.current = task
    try { await task } finally { if (polling.current === task) polling.current = null }
  }
  refreshRef.current = refresh

  const project = (run: ServerWorkflowRun, steps: WorkflowRunStepSummary[] = []) => {
    const previous = useWorkflowEditorStore.getState()
    const nodeStates: Record<string, StepExecutionState> = {}
    for (const step of steps) {
      const startedAt = timestamp(step.started_at)
      const completedAt = timestamp(step.ended_at)
      nodeStates[step.step_id] = { nodeId: step.step_id, status: step.status === "succeeded" ? "success" : step.status as StepExecutionState["status"], startedAt, completedAt, durationMs: startedAt !== undefined && completedAt !== undefined ? completedAt - startedAt : undefined, error: typeof step.error === "string" ? step.error : undefined }
    }
    const waiting = (run.status === "waiting_human" || run.status === "waiting_approval") ? steps.find((step) => step.status === run.status) : undefined
    const outputs = run.outputs || null
    useWorkflowEditorStore.setState({
      runId: run.run_id, status: run.status === "succeeded" ? "completed" : run.status,
      runWorkflowId: run.workflow_id ?? null, runWorkflowVersion: run.definition_version ?? null,
      nodeStates: steps.length ? nodeStates : previous.nodeStates,
      error: run.error || null,
      startedAt: steps.map((step) => timestamp(step.started_at)).filter((value): value is number => value !== undefined).sort((a, b) => a - b)[0] ?? previous.startedAt,
      completedAt: terminal(run.status) ? steps.map((step) => timestamp(step.ended_at)).filter((value): value is number => value !== undefined).sort((a, b) => b - a)[0] : undefined,
      pendingApproval: waiting ? { id: waiting.step_id, nodeId: waiting.step_id, nodeName: waiting.name || waiting.step_id, promptMessage: typeof outputs?.reason === "string" ? outputs.reason : "Review the server-provided step output before responding.", dataToReview: outputs, allowEdit: false, editableFields: [], createdAt: timestamp(waiting.started_at) || 0 } : null
    })
    setOutput(outputs)
  }

  const inspect = async (binding = runBinding.current) => {
    if (!binding || !currentRun(binding)) return
    useWorkflowEditorStore.setState({ runInvestigationLoading: true, runInvestigationError: null })
    try {
      const result = await getWorkflowInvestigation(binding.id, requestOptions(binding.owner))
      if (result?.run_id !== binding.id) throw new Error("Workflow investigation belongs to a different run.")
      if (!Array.isArray(result.attempts) || !Array.isArray(result.recommended_actions) || !result.recommended_actions.every((action) => typeof action === "string") || !result.evidence || typeof result.evidence !== "object" || Array.isArray(result.evidence)) throw new Error("Workflow investigation is malformed.")
      if (currentRun(binding)) useWorkflowEditorStore.setState({ runInvestigation: result, runInvestigationLoading: false })
    } catch (failure) { if (currentRun(binding)) useWorkflowEditorStore.setState({ runInvestigationLoading: false, runInvestigationError: message(failure) }) }
  }

  const save = () => perform(async (_snapshot, _view, options, isCurrent) => {
    const state = useWorkflowEditorStore.getState()
    const canvas = state.saveWorkflow()
    const key = workflowCanvasKey(canvas)
    const definition = compileWorkflowCanvas(canvas, (state.workflowVersion || 0) + 1)
    const schema = workflowInputSchema(canvas)
    if (schema) {
      if (!validateInputs) throw new Error("Workflow schema validation is unavailable in this client.")
      // Compilation validates the schema even when required run inputs are absent.
      await validateInputs({}, schema)
    }
    if (!isCurrent()) return undefined
    const preflight = await preflightWorkflowDefinition({ definition, validation_mode: "block" }, options)
    if (!isCurrent()) return undefined
    if (preflight.valid !== true || !Array.isArray(preflight.errors) || preflight.errors.length) throw new Error(preflight.errors?.map((issue) => issue.message).join("; ") || "Workflow preflight failed.")
    const saved = await saveWorkflowDefinition(definition, state.workflowId ? Number(state.workflowId) : null, options)
    if (!isCurrent()) return undefined
    const clean = workflowCanvasKey(useWorkflowEditorStore.getState().saveWorkflow()) === key
    useWorkflowEditorStore.setState({ workflowId: String(saved.id), workflowVersion: saved.version, savedCanvasKey: key, isDirty: !clean })
    return { ...saved, clean }
  })

  const list = () => perform(async (_snapshot, _view, options, isCurrent) => {
    const results = await listWorkflowDefinitions(options)
    if (isCurrent()) setDefinitions(results)
    return results
  })

  const open = (id: number) => perform(async (_snapshot, _view, options, isCurrent) => {
    const before = workflowCanvasKey(useWorkflowEditorStore.getState().saveWorkflow())
    const saved = await getWorkflowDefinition(id, options)
    const canvas = reopenWorkflowDefinition(saved)
    if (!isCurrent()) return
    if (workflowCanvasKey(useWorkflowEditorStore.getState().saveWorkflow()) !== before) throw new Error("Workflow changed while opening the saved workflow. Retry Open after reviewing your draft.")
    useWorkflowEditorStore.getState().loadWorkflow(canvas)
    useWorkflowEditorStore.setState({ savedCanvasKey: workflowCanvasKey(canvas) })
    return saved
  })

  const run = (inputJson: string) => perform(async (snapshot, view, options, isCurrent) => {
    const inputs: unknown = JSON.parse(inputJson)
    if (!inputs || typeof inputs !== "object" || Array.isArray(inputs)) throw new Error("Workflow inputs must be a JSON object.")
    const state = useWorkflowEditorStore.getState()
    const canvas = state.saveWorkflow()
    const key = workflowCanvasKey(canvas)
    compileWorkflowCanvas(canvas, state.workflowVersion || 1)
    const schema = workflowInputSchema(canvas)
    if (schema) {
      if (!validateInputs) throw new Error("Workflow schema validation is unavailable in this client.")
      const errors = await validateInputs(inputs, schema)
      if (errors.length) throw new Error(`Workflow inputs: ${errors.join("; ")}`)
    }
    if (!isCurrent()) return
    if (!state.workflowId || !state.workflowVersion || state.isDirty || state.savedCanvasKey !== key) throw new Error("Save the current workflow before running it.")
    if (state.runId) throw new Error("Reset the previous run before starting another.")
    if (workflowCanvasKey(useWorkflowEditorStore.getState().saveWorkflow()) !== key) throw new Error("Workflow changed while inputs were being validated.")
    const dispatchKey = JSON.stringify([state.workflowId, state.workflowVersion, inputs])
    const existing = useWorkflowEditorStore.getState().pendingRunSubmission
    if (existing && existing.key !== dispatchKey) throw new Error("A previous run submission has an unknown outcome. Retry its original inputs or reset before starting a different run.")
    const submission = existing ?? { key: dispatchKey, idempotencyKey: crypto.randomUUID(), inputs: inputs as Record<string, unknown> }
    useWorkflowEditorStore.setState({ pendingRunSubmission: submission })
    const result = await runSavedWorkflow(Number(state.workflowId), submission.inputs, submission.idempotencyKey, options)
    if (!isCurrent()) return
    if (result.definition_version !== state.workflowVersion) throw new Error("The server ran a different workflow version.")
    const binding = { id: result.run_id, workflowId: Number(state.workflowId), version: state.workflowVersion, owner: snapshot, view }
    runBinding.current = binding
    useWorkflowEditorStore.setState({ pendingRunSubmission: null })
    project(result)
    void refresh(binding)
    return result
  })

  const control = (action: "pause" | "resume" | "cancel") => perform(async (_snapshot, _view, options, isCurrent) => {
    const binding = runBinding.current
    if (!binding || !currentRun(binding)) throw new Error("No current server run is connected.")
    stopPolling()
    try { await controlWorkflowRun(binding.id, action, options) }
    finally { if (isCurrent() && currentRun(binding)) await refresh(binding) }
  })

  const review = (action: "approve" | "reject") => perform(async (_snapshot, _view, options, isCurrent) => {
    const binding = runBinding.current
    const step = useWorkflowEditorStore.getState().pendingApproval?.nodeId
    if (!binding || !step || !currentRun(binding)) throw new Error("No current server review is connected.")
    await reviewWorkflowStep(binding.id, step, action, options)
    if (isCurrent() && currentRun(binding)) await refresh(binding)
  })

  return { ready, pending, error, output, definitions, save, list, open, run, control, review, inspect }
}

export type WorkflowEditorApi = ReturnType<typeof useWorkflowEditorApi>
