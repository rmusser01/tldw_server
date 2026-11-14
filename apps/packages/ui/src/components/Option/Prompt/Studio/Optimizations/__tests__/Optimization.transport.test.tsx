import React from "react"
import { App, ConfigProvider } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { apiSend } from "@/services/api-send"
import { cancelOptimization, listOptimizations } from "@/services/prompt-studio"
import { usePromptStudioStore } from "@/store/prompt-studio"
import { OptimizationsTab } from "../OptimizationsTab"
import { OptimizationProgressPanel } from "../OptimizationProgressPanel"

vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, options?: { defaultValue?: string; [key: string]: unknown }) =>
    Object.entries(options ?? {}).reduce((text, [name, value]) => text.replaceAll(`{{${name}}}`, String(value)), options?.defaultValue ?? key) })
}))

const saved = {
  id: 81, project_id: 7, name: "Saved Rowan optimization", initial_prompt_id: 21,
  optimized_prompt_id: 22, optimizer_type: "iterative", status: "completed",
  optimization_config: { max_iterations: 3, model_config: { provider: "ollama", model: "qwen2.5:0.5b" } },
  initial_metrics: { score: 0.25 }, final_metrics: { score: 0.75 },
  iterations_completed: 2, max_iterations: 3, error_message: null,
  started_at: "2026-10-09T04:00:00Z", completed_at: "2026-10-09T04:01:00Z"
}
let rows: unknown[]
let detail: unknown
let iterations: unknown[]
let refused: string | null
let envelopeRefusal: boolean
let held: Promise<unknown> | null
let client: QueryClient
const envelope = (data: unknown) => ({ ok: true, status: 200, data: { success: true, data } })
const calls = () => vi.mocked(apiSend).mock.calls.map(([request]) => request)
function mount(node: React.ReactNode = <OptimizationsTab />) {
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const wrap = (content: React.ReactNode) => <ConfigProvider theme={{ token: { motion: false } }}><App><QueryClientProvider client={client}>{content}</QueryClientProvider></App></ConfigProvider>
  const view = render(wrap(node))
  return { ...view, show: (content: React.ReactNode) => view.rerender(wrap(content)) }
}
const panel = (id = 81) => <OptimizationProgressPanel open optimizationId={id} onClose={vi.fn()} />

beforeEach(() => {
  vi.clearAllMocks()
  usePromptStudioStore.getState().resetStore()
  usePromptStudioStore.getState().setSelectedProjectId(7)
  rows = [saved]; detail = saved; iterations = []; refused = null; envelopeRefusal = false; held = null
  vi.mocked(apiSend).mockImplementation(async (request) => {
    const path = String(request.path)
    if (refused && path.includes(refused)) return (envelopeRefusal
      ? { ok: true, status: 200, data: { success: false, error: "Owner refused" } }
      : { ok: false, status: 404, error: "Not found" }) as never
    if (path.startsWith("/api/v1/prompt-studio/optimizations/list/7")) return {
      ...envelope(rows), data: { success: true, data: rows, metadata: { page: 1, per_page: 100, total: rows.length, total_pages: 1 } }
    } as never
    if (path === "/api/v1/prompt-studio/optimizations/get/81") return (held ? await held : envelope(detail)) as never
    if (path === "/api/v1/prompt-studio/optimizations/get/82") return envelope({ ...saved, id: 82, name: "New selection" }) as never
    if (path.includes("/optimizations/iterations/")) return envelope({ iterations }) as never
    if (path.includes("/optimizations/cancel/")) return envelope({ message: "Optimization cancelled" }) as never
    return { ok: false, status: 404, error: "No canonical route" } as never
  })
})
afterEach(() => { cleanup(); client?.clear() })

describe("canonical saved optimization transport", () => {
  it.each(["list", "panel"])("C02 completed early shows completion while retaining the used iteration count in %s", async target => {
    const early = { ...saved, optimized_prompt_id: saved.initial_prompt_id, iterations_completed: 0, max_iterations: 1, initial_metrics: { score: 1 }, final_metrics: { score: 1 } }
    rows = [early]
    detail = early
    mount(target === "list" ? <OptimizationsTab /> : panel())
    await screen.findByText("Saved Rowan optimization")
    await waitFor(() => expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100"))
    expect(screen.getByText(target === "list" ? "0 / 1" : "0 / 1 iterations")).toBeTruthy()
    expect(calls().every(request => request.method === "GET")).toBe(true)
  })

  it("reopens the saved optimization through project list, detail and stored iterations", async () => {
    mount()
    const title = await screen.findByText("Saved Rowan optimization")
    expect(calls()[0].path).toBe("/api/v1/prompt-studio/optimizations/list/7?page=1&per_page=100")
    fireEvent.doubleClick(title.closest("tr")!)
    await screen.findByText("Optimization Progress")
    await screen.findByText(/optimized prompt has been saved as Prompt #22/)
    expect(screen.getByText("21")).toBeTruthy()
    expect(screen.getAllByText(/2 \/ 3/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/75\.0/).length).toBeGreaterThan(0)
    expect(calls().some(request => request.path === "/api/v1/prompt-studio/optimizations/get/81")).toBe(true)
    expect(calls().some(request => String(request.path).startsWith("/api/v1/prompt-studio/optimizations/iterations/81?"))).toBe(true)
    expect(calls().every(request => request.method === "GET")).toBe(true)
  })

  it("uses the canonical page and status filter without unrelated catalog requests", async () => {
    await listOptimizations({ project_id: 7, page: 2, per_page: 2, status: "failed" })
    expect(calls()).toHaveLength(1)
    expect(calls()[0]).toMatchObject({ method: "GET", path: "/api/v1/prompt-studio/optimizations/list/7?page=2&per_page=2&status=failed" })
  })

  it("shows an empty saved iteration history truthfully", async () => {
    mount(panel())
    expect(await screen.findByText("No stored iteration records")).toBeTruthy()
    expect(screen.queryByText(/Iteration 1/)).toBeNull()
  })

  it("renders canonical persisted iteration number, score, note and timestamp", async () => {
    iterations = [{ id: 901, optimization_id: 81, iteration_number: 2, prompt_variant: { system_prompt: "Keep source facts." }, metrics: { score: 0 }, note: "Retained source support.", created_at: "2026-10-09T04:00:30Z" }]
    mount(panel())
    expect(await screen.findByText("Iteration 2")).toBeTruthy()
    expect(screen.getByText("0.0%")).toBeTruthy()
    expect(screen.getByText("Retained source support.")).toBeTruthy()
    expect(screen.queryByText("No stored iteration records")).toBeNull()
  })

  it("preserves a real zero final score", async () => {
    detail = { ...saved, final_metrics: { score: 0 } }
    mount(panel())
    await screen.findByText("Best Score")
    expect(document.querySelector(".ant-statistic-content")?.textContent).toBe("0.0%")
  })

  it("does not invent a zero score when no final score was stored", async () => {
    detail = { ...saved, final_metrics: {} }
    mount(panel())
    expect(await screen.findByText("Unavailable")).toBeTruthy()
    expect(document.querySelector(".ant-statistic-content")?.textContent).not.toBe("0.0%")
  })

  it("does not infer a score from unrelated iteration metrics", async () => {
    iterations = [{ id: 902, iteration_number: 1, metrics: { latency_ms: 40 }, prompt_variant: {}, note: null, created_at: "2026-10-09T04:00:30Z" }]
    mount(panel())
    await screen.findByText("Iteration 1")
    expect(screen.getByText("Score unavailable")).toBeTruthy()
  })

  it("shows an owner refusal rather than an empty optimization library", async () => {
    refused = "/list/"
    mount()
    expect(await screen.findByText("Unable to load optimizations")).toBeTruthy()
    expect(screen.queryByText("No optimizations yet")).toBeNull()
  })

  it("shows missing detail truthfully without report success", async () => {
    refused = "/get/"
    mount(panel())
    expect(await screen.findByText("Unable to load optimization")).toBeTruthy()
    expect(screen.queryByText("Best prompt found")).toBeNull()
  })

  it("distinguishes iteration retrieval failure from empty persisted history", async () => {
    refused = "/iterations/"
    mount(panel())
    expect(await screen.findByText("Unable to load stored iterations")).toBeTruthy()
    expect(screen.queryByText("No stored iteration records")).toBeNull()
  })

  it("uses the optional JSON string reason on the canonical cancel route", async () => {
    await cancelOptimization(81, "Reviewed stop")
    expect(calls()[0]).toMatchObject({ path: "/api/v1/prompt-studio/optimizations/cancel/81", method: "POST", body: "Reviewed stop" })
  })

  it.each([false, true])("rejects cancellation refusal, including envelope refusal=%s", async (isEnvelope) => {
    refused = "/cancel/"; envelopeRefusal = isEnvelope
    await expect(cancelOptimization(81)).rejects.toThrow(isEnvelope ? "Owner refused" : "Not found")
    expect(calls()).toHaveLength(1)
  })

  it("offers view progress and omits unsupported Delete", async () => {
    mount()
    const title = await screen.findByText("Saved Rowan optimization")
    fireEvent.click(within(title.closest("tr")!).getByRole("button"))
    await screen.findByText("View Progress")
    expect(screen.queryByText("Delete")).toBeNull()
    expect(calls().every(request => request.method === "GET")).toBe(true)
  })

  it("keeps a late old detail response out of the newly selected report", async () => {
    let release!: (value: unknown) => void
    held = new Promise(resolve => { release = resolve })
    const view = mount(panel())
    await waitFor(() => expect(calls().some(request => String(request.path).endsWith("/get/81"))).toBe(true))
    view.show(panel(82))
    await screen.findByText("New selection")
    release(envelope({ ...saved, name: "Late old report" }))
    await waitFor(() => expect(client.getQueryState(["prompt-studio", "optimization", 81])?.fetchStatus).toBe("idle"))
    expect(screen.queryByText("Late old report")).toBeNull()
    expect(screen.getByText("New selection")).toBeTruthy()
  })
})
