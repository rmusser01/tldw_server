import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ConfigProvider } from "antd"
import { MemoryRouter, useLocation, useNavigate } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { EvaluationsPage } from "../EvaluationsPage"
import { useEvaluationsStore } from "@/store/evaluations"
import type { RecipeRunReport } from "@/services/evaluations"

const mocks = vi.hoisted(() => ({ request: vi.fn() }))
// Only external transport, connection readiness and unrelated tab panels are mocked.
// The host, router, RecipesTab, services, query hooks, store and report renderer are real.
vi.mock("@/services/api-send", () => ({ apiSend: mocks.request }))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({ uxState: "connected", hasCompletedFirstRun: true })
}))
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ success: vi.fn(), error: vi.fn() })
}))
vi.mock("./../tabs/SyntheticReviewTab", () => ({ SyntheticReviewTab: () => null }))
vi.mock("./../tabs/EvaluationsTab", () => ({ EvaluationsTab: () => null }))
vi.mock("./../tabs/RunsTab", () => ({ RunsTab: () => null }))
vi.mock("./../tabs/DatasetsTab", () => ({ DatasetsTab: () => null }))
vi.mock("./../tabs/WebhooksTab", () => ({ WebhooksTab: () => null }))
vi.mock("./../tabs/HistoryTab", () => ({ HistoryTab: () => null }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: string | Record<string, unknown>) => {
      let text = typeof options === "string" ? options : String(options?.defaultValue ?? key)
      if (options && typeof options === "object") {
        for (const [name, value] of Object.entries(options)) text = text.replaceAll(`{{${name}}}`, String(value))
      }
      return text
    }
  })
}))

const RECIPE = "persona_dialogue_tree_robustness"
const RUN = "recipe_run_existing_12"
const NEXT_RUN = "recipe_run_existing_34"
const CREATED_RUN = "recipe_run_created_56"
const MANIFESTS = [
  { recipe_id: "summarization_quality", recipe_version: "1", name: "Summarization Quality", description: "Summary comparison.", supported_modes: ["labeled"], tags: [], launchable: true },
  { recipe_id: RECIPE, recipe_version: "1", name: "Persona Dialogue-Tree Robustness", description: "Offline scenarios.", supported_modes: ["unlabeled"], tags: [], launchable: true }
]
const makeReport = (id: string): RecipeRunReport => ({
  run: {
    run_id: id, recipe_id: RECIPE, recipe_version: "1", status: "completed",
    created_at: "2026-10-09T03:00:00Z",
    metadata: { recipe_report: { summary: { target_count: 1, total_cases: 2, hard_prune_count: 0 }, target_results: [], trace_artifact_refs: [] } }
  },
  recommendation_slots: {}
})
const PREVIEW = { run_id: RUN, recipe_id: "embeddings_model_selection", slot_name: "best_overall", candidate_run_id: "candidate-1", apply_available: true, proposed: { provider: "ollama", model: "model-owned" }, warnings: [] }
const setEmbeddingReport = () => {
  const report = makeReport(RUN)
  report.run.recipe_id = "embeddings_model_selection"
  report.recommendation_slots = { best_overall: { candidate_run_id: "candidate-1", metadata: { apply_eligible: true, provider: "ollama", model: "model-owned" } } }
  reports.set(RUN, async () => ok(report))
}
const ok = (data: unknown, status = 200) => ({ ok: true, status, data })
const reports = new Map<string, () => Promise<unknown>>()
const clients: QueryClient[] = []
let createResponse: () => Promise<unknown>
let manifestsResponse: () => Promise<unknown>
let previewResponse: () => Promise<unknown>
let applyResponse: () => Promise<unknown>

const LocationObserver = () => {
  const location = useLocation()
  const navigate = useNavigate()
  return <>
    <output aria-label="Current route">{location.pathname}{location.search}</output>
    <button onClick={() => navigate(`/evaluations?tab=recipes&recipeRunId=${NEXT_RUN}&tour=1`)}>Navigate to another recipe run</button>
    <button onClick={() => navigate(`${location.pathname}${location.search}&custom=updated&added=later`)}>Update route parameters</button>
  </>
}
const mount = (url = "/evaluations?tab=recipes") => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  return render(<QueryClientProvider client={client}>
    <MemoryRouter initialEntries={[url]}><EvaluationsPage /><LocationObserver /></MemoryRouter>
  </QueryClientProvider>)
}
const route = () => screen.getByLabelText("Current route").textContent || ""
const reportCalls = () => mocks.request.mock.calls.filter(([request]) => request.method === "GET" && request.path.endsWith("/report"))
const postCalls = () => mocks.request.mock.calls.filter(([request]) => request.method === "POST")
const waitForManifests = () => screen.findByRole("button", { name: "Use Persona Dialogue-Tree Robustness" })
const openRun = async (id: string) => {
  fireEvent.change(await screen.findByRole("textbox", { name: "Existing recipe run ID" }), { target: { value: id } })
  fireEvent.click(screen.getByRole("button", { name: "Open existing run" }))
}
const launchRun = async () => {
  await waitForManifests()
  fireEvent.click(screen.getByText("Advanced JSON"))
  const editors = await screen.findAllByRole("textbox", { name: "" })
  fireEvent.change(editors[0], { target: { value: JSON.stringify([{ input: "A grounded source.", expected: "A concise summary." }]) } })
  fireEvent.change(editors[1], { target: { value: JSON.stringify({ candidate_model_ids: ["ollama:qwen2.5:0.5b"], judge_config: { provider: "ollama", model: "qwen2.5:0.5b" } }) } })
  fireEvent.click(screen.getByRole("button", { name: "Run recipe" }))
}
const expectReport = async (id: string) => {
  const details = await screen.findByRole("region", { name: "Run details" })
  await waitFor(() => expect(within(details).getByText(id)).toBeVisible())
}
const deferred = () => {
  let resolve!: (value: unknown) => void
  const promise = new Promise<unknown>((accept) => { resolve = accept })
  return { promise, resolve }
}

beforeEach(() => {
  vi.clearAllMocks()
  useEvaluationsStore.getState().resetStore()
  reports.clear()
  for (const id of [RUN, NEXT_RUN, CREATED_RUN]) reports.set(id, async () => ok(makeReport(id)))
  createResponse = async () => ok({ run_id: CREATED_RUN, status: "pending" }, 202)
  manifestsResponse = async () => ok(MANIFESTS)
  previewResponse = async () => ok(PREVIEW)
  applyResponse = async () => ok({ ...PREVIEW, applied: true })
  mocks.request.mockImplementation(async ({ path, method = "GET" }: { path: string, method?: string }) => {
    if (path === "/api/v1/evaluations/recipes" && method === "GET") return manifestsResponse()
    if (path.startsWith("/api/v1/evaluations/datasets?") && method === "GET") return ok({ data: [] })
    if (path.endsWith("/launch-readiness") && method === "GET") return ok({ ready: true, can_enqueue_runs: true, can_reuse_completed_runs: true })
    if (path.endsWith("/runs") && method === "POST") return createResponse()
    if (path.endsWith("/apply-preview") && method === "POST") return previewResponse()
    if (path.endsWith("/apply") && method === "POST") return applyResponse()
    const match = path.match(/^\/api\/v1\/evaluations\/recipe-runs\/([^/]+)\/report$/)
    if (match && method === "GET") {
      const response = reports.get(decodeURIComponent(match[1]))
      return response ? response() : { ok: false, status: 404, error: "Recipe run not found" }
    }
    throw new Error(`Unexpected transport: ${method} ${path}`)
  })
})
afterEach(() => {
  cleanup()
  clients.splice(0).forEach((client) => client.clear())
  useEvaluationsStore.getState().resetStore()
})

describe("Existing recipe run recovery", () => {
  it("loads a completed deep-linked report through GET and retains it after initial manifest hydration", async () => {
    mount(`/evaluations?tab=recipes&recipeRunId=${RUN}`)
    await waitForManifests()
    await waitFor(() => expect(reportCalls()).toHaveLength(1))
    await expectReport(RUN)
    expect(screen.getByRole("region", { name: "Persona summary" })).toBeVisible()
    expect(postCalls()).toHaveLength(0)
  })

  it("opens an existing ID with GET, preserves unrelated parameters and recovers after a fresh mount", async () => {
    const mounted = mount("/evaluations?tab=recipes&tour=1&runId=standard_run_99&custom=kept")
    await waitForManifests()
    await openRun(`  ${RUN}  `)
    await expectReport(RUN)
    const savedUrl = route()
    const params = new URLSearchParams(savedUrl.split("?")[1])
    expect(params.get("recipeRunId")).toBe(RUN)
    expect(params.get("runId")).toBe("standard_run_99")
    expect(params.get("tour")).toBe("1")
    expect(params.get("custom")).toBe("kept")
    mounted.unmount()
    mount(savedUrl)
    await expectReport(RUN)
    expect(reportCalls()).toHaveLength(2)
    expect(postCalls()).toHaveLength(0)
  })

  it("surfaces an owner-scoped not-found response without creating or applying a run", async () => {
    mount()
    await openRun("recipe_run_other_owner")
    await screen.findByText("Unable to load the recipe report")
    fireEvent.click(screen.getByText("Diagnostics", { selector: "summary" }))
    expect(screen.getByText("HTTP 404: Recipe run not found")).toBeVisible()
    expect(reportCalls()).toHaveLength(1)
    expect(postCalls()).toHaveLength(0)
    expect(screen.queryByRole("region", { name: "Persona summary" })).not.toBeInTheDocument()
  })

  it("encodes an unsupported opaque ID and shows the server refusal truthfully", async () => {
    mount()
    await openRun("not/a/run?<tag>")
    await screen.findByText("HTTP 404: Recipe run not found")
    expect(reportCalls()[0][0]).toEqual(expect.objectContaining({
      method: "GET", path: "/api/v1/evaluations/recipe-runs/not%2Fa%2Frun%3F%3Ctag%3E/report"
    }))
    expect(route()).toContain("recipeRunId=not%2Fa%2Frun%3F%3Ctag%3E")
    expect(postCalls()).toHaveLength(0)
  })

  it("keeps blank input disabled without a report or mutation request", async () => {
    mount()
    await waitForManifests()
    fireEvent.change(screen.getByRole("textbox", { name: "Existing recipe run ID" }), { target: { value: "   " } })
    expect(screen.getByRole("button", { name: "Open existing run" })).toBeDisabled()
    expect(reportCalls()).toHaveLength(0)
    expect(postCalls()).toHaveLength(0)
  })

  it("writes a new run URL only after the creation acknowledgement and keeps the latest parameters", async () => {
    const pending = deferred()
    createResponse = () => pending.promise
    mount("/evaluations?tab=recipes&custom=kept")
    await launchRun()
    await waitFor(() => expect(postCalls()).toHaveLength(1))
    expect(route()).not.toContain("recipeRunId=")
    fireEvent.click(screen.getByRole("button", { name: "Update route parameters" }))
    await act(async () => { pending.resolve(ok({ run_id: CREATED_RUN, status: "pending" }, 202)) })
    await expectReport(CREATED_RUN)
    expect(new URLSearchParams(route().split("?")[1]).get("recipeRunId")).toBe(CREATED_RUN)
    expect(route()).toContain("added=later")
    expect(postCalls()).toHaveLength(1)
  })

  it("clears run routing on an intentional different-recipe selection while retaining other parameters", async () => {
    mount(`/evaluations?tab=recipes&recipeRunId=${RUN}&custom=kept`)
    await expectReport(RUN)
    fireEvent.click(await waitForManifests())
    await waitFor(() => expect(screen.queryByText("Current run")).not.toBeInTheDocument())
    expect(route()).not.toContain("recipeRunId=")
    expect(route()).toContain("custom=kept")
    expect(postCalls()).toHaveLength(0)
  })

  it("ignores an old report arriving after another ID is opened", async () => {
    const old = deferred()
    reports.set(RUN, () => old.promise)
    mount(`/evaluations?tab=recipes&recipeRunId=${RUN}`)
    await waitFor(() => expect(reportCalls()).toHaveLength(1))
    await openRun(NEXT_RUN)
    await expectReport(NEXT_RUN)
    await act(async () => { old.resolve(ok(makeReport(RUN))) })
    expect(screen.queryByText(RUN, { selector: "dd" })).not.toBeInTheDocument()
    expect(route()).toContain(NEXT_RUN)
    expect(postCalls()).toHaveLength(0)
  })

  it("keeps a later explicit Open authoritative over a held creation acknowledgement", async () => {
    const pending = deferred()
    createResponse = () => pending.promise
    mount()
    await launchRun()
    await waitFor(() => expect(postCalls()).toHaveLength(1))
    await openRun(NEXT_RUN)
    await expectReport(NEXT_RUN)
    await act(async () => { pending.resolve(ok({ run_id: CREATED_RUN, status: "pending" }, 202)) })
    expect(route()).toContain(NEXT_RUN)
    expect(screen.queryByText(CREATED_RUN)).not.toBeInTheDocument()
    expect(reportCalls()).toHaveLength(1)
    expect(postCalls()).toHaveLength(1)
  })

  it("responds to normal URL navigation using the newly selected report identity", async () => {
    mount(`/evaluations?tab=recipes&recipeRunId=${RUN}`)
    await expectReport(RUN)
    fireEvent.click(screen.getByRole("button", { name: "Navigate to another recipe run" }))
    await expectReport(NEXT_RUN)
    expect(screen.queryByText(RUN, { selector: "dd" })).not.toBeInTheDocument()
    expect(postCalls()).toHaveLength(0)
  })

  it("opens a report even when the manifest request fails", async () => {
    manifestsResponse = async () => ({ ok: false, status: 503, error: "Recipe catalog unavailable" })
    mount(`/evaluations?tab=recipes&recipeRunId=${RUN}`)
    await screen.findByText("Unable to load recipes")
    await expectReport(RUN)
    expect(postCalls()).toHaveLength(0)
  })

  it("retries a failed report GET without resubmitting the recipe", async () => {
    let calls = 0
    reports.set(RUN, async () => ++calls === 1
      ? { ok: false, status: 503, error: "Report temporarily unavailable" }
      : ok(makeReport(RUN)))
    mount(`/evaluations?tab=recipes&recipeRunId=${RUN}`)
    await screen.findByText("Unable to load the recipe report")
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    await expectReport(RUN)
    expect(reportCalls()).toHaveLength(2)
    expect(postCalls()).toHaveLength(0)
  })
})


describe("Existing recipe run view settlement", () => {
  it("does not restore a held old preview after another report is opened", async () => {
    const pending = deferred()
    previewResponse = () => pending.promise
    setEmbeddingReport()
    mount(`/evaluations?tab=recipes&recipeRunId=${RUN}`)
    fireEvent.click(await screen.findByRole("button", { name: "Preview RAG config change" }))
    await waitFor(() => expect(postCalls()).toHaveLength(1))
    expect(postCalls()[0][0]).toEqual(expect.objectContaining({ path: `/api/v1/evaluations/recipe-runs/${RUN}/apply-preview`, body: { slot_name: "best_overall", candidate_run_id: "candidate-1" } }))
    await openRun(NEXT_RUN)
    await expectReport(NEXT_RUN)
    await act(async () => { pending.resolve(ok(PREVIEW)) })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Apply to RAG config" })).not.toBeInTheDocument()
    expect(postCalls()).toHaveLength(1)
  })

  it("keeps a committed old Apply response from restoring a closed modal on a newer run", async () => {
    const pending = deferred()
    applyResponse = () => pending.promise
    setEmbeddingReport()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    clients.push(client)
    render(<ConfigProvider theme={{ token: { motion: false } }}>
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[`/evaluations?tab=recipes&recipeRunId=${RUN}`]}>
          <EvaluationsPage /><LocationObserver />
        </MemoryRouter>
      </QueryClientProvider>
    </ConfigProvider>)
    fireEvent.click(await screen.findByRole("button", { name: "Preview RAG config change" }))
    const dialog = await screen.findByRole("dialog")
    fireEvent.click(within(dialog).getByRole("button", { name: "Apply to RAG config" }))
    await waitFor(() => expect(postCalls()).toHaveLength(2))
    expect(postCalls()[1][0]).toEqual(expect.objectContaining({ path: `/api/v1/evaluations/recipe-runs/${RUN}/apply`, body: { slot_name: "best_overall", candidate_run_id: "candidate-1", confirmed_provider: "ollama", confirmed_model: "model-owned" } }))
    fireEvent.click(within(dialog).getAllByRole("button", { name: "Close" })[0])
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument(), { timeout: 5000 })
    await openRun(NEXT_RUN)
    await expectReport(NEXT_RUN)
    await act(async () => { pending.resolve(ok({ ...PREVIEW, applied: true })) })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.queryByText("RAG config updated")).not.toBeInTheDocument()
    expect(postCalls()).toHaveLength(2)
  })
})
