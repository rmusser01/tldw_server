import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { RecipeRunReport } from "@/services/evaluations"
import { RecipesTab } from "../RecipesTab"

const mocks = vi.hoisted(() => ({ request: vi.fn(), success: vi.fn(), error: vi.fn() }))
// External transport only: real services, query hooks, store and rendered UI remain mounted.
vi.mock("@/services/api-send", () => ({ apiSend: mocks.request }))
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ success: mocks.success, error: mocks.error })
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: string | Record<string, unknown>) => {
      let value = typeof options === "string" ? options : String(options?.defaultValue ?? key)
      if (options && typeof options === "object") {
        for (const [name, replacement] of Object.entries(options)) {
          value = value.replaceAll(`{{${name}}}`, String(replacement))
        }
      }
      return value
    }
  })
}))

const RECIPE = "persona_dialogue_tree_robustness"
const RUN = "recipe-run-persona-visible"
const SCENARIO = [{ case_id: "benign_basic", prompt: "Can you summarize this topic calmly and clearly?", candidates: [{ action_type: "assistant", text: "A concise summary.", metadata: { grounded: true } }] }]
const CONFIG = { targets: [{ target_id: "persona-1:character-1", persona: { id: "persona-1", policy_snapshot: { mode: "strict" } }, character: { id: "character-1" } }], include_trace_artifacts: true }
const SUMMARY = { target_count: 1, total_cases: 2, hard_prune_count: 0, soft_prune_count: 3, selected_trajectory_count: 1, skipped_scorer_count: 4, trace_artifact_count: 2 }
const TRACE = { target_id: "persona-1:character-1", artifact_id: "persona-1:character-1:benign_basic:trace", case_id: "benign_basic" }
const REPORT: RecipeRunReport = {
  run: {
    run_id: RUN, recipe_id: RECIPE, recipe_version: "1", status: "completed", review_state: "pending_review",
    dataset_snapshot_ref: "inline:snapshot-persona", dataset_content_hash: "sha256-persona-content", created_at: "2026-10-09T03:00:00Z",
    metadata: {
      dataset_mode: "unlabeled", jobs: { job_id: "job-persona-982", worker_state: "completed" },
      run_config: { api_key: "SECRET_CONFIG_MUST_NOT_RENDER" },
      trace_artifacts: [{ raw_response: "SECRET_TRACE_MUST_NOT_RENDER" }],
      recipe_report: {
        summary: SUMMARY,
        target_results: [{
          target_id: "persona-1:character-1", persona_id: "persona-1", character_id: "character-1",
          summary: { total_cases: 2, hard_prune_count: 0, soft_prune_count: 3, selected_trajectory_count: 1, skipped_scorer_count: 4, trace_artifact_count: 2 },
          cases: [
            { case_id: "benign_basic", selected_node_id: "node-selected-8", hard_prune_count: 0, soft_prune_count: 0, selected_trajectory_count: 1, skipped_scorer_count: 2 },
            { case_id: "fallback_case", selected_node_id: null, hard_prune_count: 1, soft_prune_count: 0, selected_trajectory_count: 0, skipped_scorer_count: 0 }
          ], trace_artifact_refs: [TRACE]
        }],
        trace_artifact_refs: [TRACE, { ...TRACE, artifact_id: "persona-1:character-1:fallback_case:trace", case_id: "fallback_case" }]
      }
    }
  },
  confidence_summary: null,
  recommendation_slots: {
    best_overall: { candidate_run_id: null }, best_cheap: { candidate_run_id: null },
    best_local: { candidate_run_id: null }, best_performance: { candidate_run_id: null }
  }
}
const MANIFESTS = [
  { recipe_id: RECIPE, recipe_version: "1", name: "Persona Dialogue-Tree Robustness", description: "Offline robustness scenarios.", supported_modes: ["labeled", "unlabeled"], tags: ["persona"], launchable: true },
  { recipe_id: "summarization_quality", recipe_version: "1", name: "Summarization Quality", description: "Compare summaries.", supported_modes: ["labeled", "unlabeled"], tags: ["summarization"], launchable: true }
]
const clients: QueryClient[] = []
let report: RecipeRunReport
const reportRequest = vi.fn<() => Promise<unknown>>()
const ok = (data: unknown, status = 200) => ({ ok: true, status, data })
const mount = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  return render(<QueryClientProvider client={client}><RecipesTab /></QueryClientProvider>)
}
const launch = async (waitForReport = true) => {
  await screen.findByRole("button", { name: "Use Persona Dialogue-Tree Robustness" })
  fireEvent.click(screen.getByText("Advanced JSON"))
  const editors = await screen.findAllByRole("textbox", { name: "" })
  fireEvent.change(editors[0], { target: { value: JSON.stringify(SCENARIO) } })
  fireEvent.change(editors[1], { target: { value: JSON.stringify(CONFIG) } })
  fireEvent.click(screen.getByRole("button", { name: "Run recipe" }))
  await waitFor(() => expect(reportRequest).toHaveBeenCalledTimes(1))
  expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({ method: "POST", body: { dataset_id: undefined, dataset: SCENARIO, run_config: CONFIG, force_rerun: false } }))
  if (waitForReport) await screen.findByText(report.run.status, { selector: ".ant-tag" })
}
const section = (name: string) => screen.getByRole("region", { name })
const count = (region: HTMLElement, label: string, value: number) => {
  const term = within(region).getByText(label, { selector: "dt" })
  expect(term.nextElementSibling).toHaveTextContent(String(value))
}

beforeEach(() => {
  vi.clearAllMocks()
  report = structuredClone(REPORT)
  reportRequest.mockImplementation(async () => ok(report))
  mocks.request.mockImplementation(async ({ path, method }: { path: string, method?: string }) => {
    if (path === "/api/v1/evaluations/recipes") return ok(MANIFESTS)
    if (path.startsWith("/api/v1/evaluations/datasets?")) return ok({ data: [] })
    if (path.endsWith("/launch-readiness")) return ok({ ready: true, can_enqueue_runs: true, can_reuse_completed_runs: true, message: null })
    if (path === `/api/v1/evaluations/recipes/${RECIPE}/runs` && method === "POST") return ok({ ...report.run, status: "pending" }, 202)
    if (path === `/api/v1/evaluations/recipe-runs/${RUN}/report`) return reportRequest()
    throw new Error(`Unexpected external request: ${method} ${path}`)
  })
})
afterEach(() => {
  cleanup()
  clients.splice(0).forEach((client) => client.clear())
  vi.restoreAllMocks()
})

describe("Persona Recipe report", () => {
  it("shows actual summary counts including zero instead of empty model recommendations", async () => {
    mount()
    await launch()
    const summary = section("Persona summary")
    for (const [label, value] of [["Targets", 1], ["Cases", 2], ["Hard prunes", 0], ["Soft prunes", 3], ["Selected trajectories", 1], ["Skipped scorers", 4], ["Trace artifacts", 2]] as const) count(summary, label, value)
    expect(screen.queryByText("No winner yet")).not.toBeInTheDocument()
    expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({ method: "POST", body: { dataset_id: undefined, dataset: SCENARIO, run_config: CONFIG, force_rerun: false } }))
  }, 30000)

  it("shows target identities and selected or fallback cases with their actual counts", async () => {
    mount()
    await launch()
    const targets = section("Target results")
    expect(within(targets).getByText("persona-1:character-1")).toBeVisible()
    expect(within(targets).getByText("persona-1")).toBeVisible()
    expect(within(targets).getByText("character-1")).toBeVisible()
    expect(within(targets).getByText("node-selected-8")).toBeVisible()
    expect(within(targets).getByText("fallback_case")).toBeVisible()
    expect(within(targets).getByText("No trajectory selected")).toBeVisible()
    const selectedCase = within(targets).getByText("benign_basic").closest("section")
    if (!selectedCase) throw new Error("Missing selected case section")
    count(selectedCase, "Selected trajectories", 1)
    count(selectedCase, "Skipped scorers", 2)
  }, 30000)

  it("shows exact trace references as escaped text without exposing raw metadata", async () => {
    const opaque = '<img src=x onerror="steal()">'
    report.run.metadata!.recipe_report.trace_artifact_refs[0].artifact_id = opaque
    mount()
    await launch()
    const traces = section("Trace references")
    expect(within(traces).getByText(opaque)).toBeVisible()
    expect(within(traces).getAllByText("persona-1:character-1")).toHaveLength(2)
    expect(within(traces).getByText("benign_basic")).toBeVisible()
    expect(within(traces).getByText("persona-1:character-1:fallback_case:trace")).toBeVisible()
    expect(traces.querySelector("img")).toBeNull()
    expect(screen.queryByText(/SECRET_CONFIG_MUST_NOT_RENDER|SECRET_TRACE_MUST_NOT_RENDER/)).not.toBeInTheDocument()
  }, 30000)

  it("shows authoritative run, recipe, worker and saved dataset lineage", async () => {
    report.run.status = "failed"
    report.run.metadata!.jobs = { job_id: "job-failed-36", worker_state: "failed", error_message: "RAW_JOB_ERROR_MUST_NOT_RENDER" }
    report.run.metadata!.dataset_id = "dataset-owned-47"
    mount()
    await launch()
    const lineage = section("Run details")
    for (const value of [RUN, `${RECIPE} v1`, "job-failed-36", "failed", "dataset-owned-47", "inline:snapshot-persona", "sha256-persona-content", "unlabeled", "pending_review"]) expect(within(lineage).getByText(value)).toBeVisible()
    expect(screen.queryByText("RAW_JOB_ERROR_MUST_NOT_RENDER")).not.toBeInTheDocument()
  }, 30000)

  it("keeps absent or malformed result fields unavailable without fabricated counts or lineage", async () => {
    report.run.status = "pending"
    report.run.dataset_snapshot_ref = null
    report.run.dataset_content_hash = null
    report.run.metadata = { recipe_report: { summary: "not a count object", target_results: {}, trace_artifact_refs: null } }
    mount()
    await launch()
    expect(screen.getByText("Persona results are not available yet.")).toBeVisible()
    expect(screen.queryByRole("region", { name: "Persona summary" })).not.toBeInTheDocument()
    expect(screen.queryByRole("region", { name: "Target results" })).not.toBeInTheDocument()
    expect(screen.queryByRole("region", { name: "Trace references" })).not.toBeInTheDocument()
    const lineage = section("Run details")
    expect(within(lineage).queryByText("Job ID")).not.toBeInTheDocument()
    expect(within(lineage).queryByText("Dataset snapshot")).not.toBeInTheDocument()
    expect(within(lineage).queryByText("Dataset content hash")).not.toBeInTheDocument()
    expect(screen.queryByText("No winner yet")).not.toBeInTheDocument()
  }, 30000)

  it("retains loading before a report arrives and then renders the completed results", async () => {
    let resolve!: (value: unknown) => void
    reportRequest.mockImplementationOnce(() => new Promise((accept) => { resolve = accept }))
    mount()
    await launch(false)
    expect(screen.getByText("Current run")).toBeVisible()
    expect(screen.queryByRole("region", { name: "Persona summary" })).not.toBeInTheDocument()
    await act(async () => { resolve(ok(report)) })
    await screen.findByRole("region", { name: "Persona summary" })
    expect(reportRequest).toHaveBeenCalledTimes(1)
  }, 30000)

  it("keeps report failure recovery read-only and shows results after Try again", async () => {
    reportRequest.mockResolvedValueOnce({ ok: false, status: 503, error: "Report temporarily unavailable" })
    mount()
    await launch(false)
    await screen.findByText("Unable to load the recipe report")
    expect(screen.queryByRole("region", { name: "Persona summary" })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    await screen.findByRole("region", { name: "Persona summary" })
    expect(reportRequest).toHaveBeenCalledTimes(2)
    expect(mocks.request.mock.calls.filter(([request]) => request.method === "POST")).toHaveLength(1)
  }, 30000)

  it("preserves ordinary recommendation, candidate and failure results by returned recipe ID", async () => {
    report.run.recipe_id = "summarization_quality"
    report.run.metadata = { recipe_report: {
      candidates: [{ candidate_id: "candidate-normal", model: "model-normal", provider: "ollama", metrics: { grounding: 0.9 } }],
      failure_examples: [{ sample_id: "failed-normal", query: "An ordinary query", answer: "Candidate answer text", reference_answer: "Reference answer text" }]
    } }
    report.recommendation_slots = { best_overall: { candidate_run_id: "candidate-normal", explanation: "Best grounded candidate." } }
    mount()
    await launch()
    for (const text of ["Best Overall", "candidate-normal", "Best grounded candidate.", "model-normal", "grounding: 0.9", "failed-normal", "Candidate answer text", "Reference answer text"]) expect(screen.getByText(text)).toBeVisible()
    expect(screen.queryByRole("region", { name: "Persona summary" })).not.toBeInTheDocument()
  }, 30000)

  it("clears the old Persona report when another recipe is selected", async () => {
    mount()
    await launch()
    await screen.findByRole("region", { name: "Persona summary" })
    fireEvent.click(screen.getByRole("button", { name: "Use Summarization Quality" }))
    await waitFor(() => expect(screen.queryByText("Current run")).not.toBeInTheDocument())
    expect(screen.queryByText(RUN)).not.toBeInTheDocument()
    expect(reportRequest).toHaveBeenCalledTimes(1)
  }, 30000)
})


const setupPersona = async (mode: string) => {
  const originalTransport = mocks.request.getMockImplementation()!
  mocks.request.mockImplementation(async (request) => {
    if (request.path.startsWith("/api/v1/evaluations/datasets?")) {
      return ok({ data: [{ id: "persona-scenarios", name: "Persona scenarios", sample_count: 1 }] })
    }
    return originalTransport(request)
  })
  mount()
  fireEvent.click(await screen.findByText("Use Persona Dialogue-Tree Robustness"))
  if (mode === "saved") fireEvent.click(screen.getByText("Saved dataset"))
}

it.each(["inline", "saved"])("Persona setup uses scenario JSON in %s dataset mode", async (mode) => {
  await setupPersona(mode)
  expect(screen.queryByLabelText("Candidate models")).not.toBeInTheDocument()
  expect(screen.queryByLabelText("Judge provider")).not.toBeInTheDocument()
  expect(screen.queryByLabelText("Source text 1")).not.toBeInTheDocument()
  expect(screen.getByText("Persona scenario setup")).toBeVisible()
  fireEvent.click(screen.getByText("Advanced JSON"))
  const editors = await screen.findAllByRole("textbox", { name: "" })
  expect(editors).toHaveLength(mode === "inline" ? 2 : 1)
  const target = { targets: [{ target_id: "persona-only", persona: { id: "persona-only" } }], include_trace_artifacts: true }
  fireEvent.change(editors[mode === "inline" ? 1 : 0], { target: { value: JSON.stringify(target) } })
  expect(JSON.parse((editors[mode === "inline" ? 1 : 0] as HTMLTextAreaElement).value)).toEqual(target)
  expect(mocks.request.mock.calls.some(([request]) => request.method === "POST")).toBe(false)
})

it.each(["inline", "saved"])("Persona setup retains Summarization fields after leaving %s dataset mode", async (mode) => {
  await setupPersona(mode)
  fireEvent.click(screen.getByText("Use Summarization Quality"))
  expect(await screen.findByLabelText("Candidate models")).toBeVisible()
  expect(screen.queryByText("Persona scenario setup")).not.toBeInTheDocument()
})
