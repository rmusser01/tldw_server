import React from "react"
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ConfigProvider } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { apiSend } from "@/services/api-send"
import { usePromptStudioStore } from "@/store/prompt-studio"
import { EvaluationsTab } from "../EvaluationsTab"

vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (key: string, options?: { defaultValue?: string; [key: string]: unknown }) =>
    Object.entries(options ?? {}).reduce((text, [name, value]) => text.replaceAll(`{{${name}}}`, String(value)), options?.defaultValue ?? key)
}) }))
vi.mock("@/components/Common/confirm-danger", () => ({ useConfirmDanger: () => vi.fn() }))
vi.mock("../CreateEvaluationWizard", () => ({ CreateEvaluationWizard: () => null }))

let client: QueryClient
let saved: Record<string, unknown>
const configs = [
  { provider: "ollama", model: "qwen2.5:0.5b", temperature: 0, max_tokens: 32 },
  { api_name: "openai", model_name: "recorded-second-model", temperature: 0.2, top_p: 0.8 }
]
const getRequests = () => vi.mocked(apiSend).mock.calls.map(([request]) => request)

function mount() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(<ConfigProvider theme={{ token: { motion: false } }}>
    <QueryClientProvider client={client}><EvaluationsTab /></QueryClientProvider>
  </ConfigProvider>)
}
async function openDetail() {
  fireEvent.doubleClick((await screen.findByText("Rowan completed")).closest("tr")!)
  const dialog = await screen.findByRole("dialog")
  expect(within(dialog).getByText("Evaluation Details")).toBeVisible()
  return dialog
}

beforeEach(() => {
  vi.clearAllMocks()
  usePromptStudioStore.getState().resetStore()
  usePromptStudioStore.getState().setSelectedProjectId(2)
  saved = { id: 51, project_id: 2, prompt_id: 22, name: "Rowan completed", description: "Saved full-sentence comparison",
    status: "completed", test_case_ids: [31], test_run_ids: [41], model_configs: configs, config: configs[0],
    aggregate_metrics: { accuracy: 0, pass_rate: 0, f1: 0 }, metrics: { accuracy: 0, pass_rate: 0, f1: 0 },
    created_at: "2026-10-08T18:00:00Z", completed_at: "2026-10-08T18:01:00Z", error_message: null }
  vi.mocked(apiSend).mockImplementation(async request => {
    const path = String(request.path)
    if (request.method !== "GET") throw new Error("Opening saved provenance must use GET")
    if (path.startsWith("/api/v1/prompt-studio/evaluations?")) return { ok: true, status: 200,
      data: { evaluations: [saved], total: 1, limit: 100, offset: 0 } }
    if (path.endsWith("/evaluations/51")) return { ok: true, status: 200, data: saved }
    throw new Error(`Unexpected evaluation read: ${path}`)
  })
})
afterEach(() => { cleanup(); client?.clear() })

describe("saved evaluation provenance through actual list and detail consumers", () => {
  it("shows saved identity, one recorded case and finite zero metrics in the table", async () => {
    mount()
    const row = (await screen.findByText("Rowan completed")).closest("tr")!
    expect(within(row).getByText("Saved full-sentence comparison")).toBeVisible()
    expect(within(row).getByText("1")).toBeVisible()
    expect(within(row).getAllByText("0.0%")).toHaveLength(2)
    expect(within(row).getByText("0.000")).toBeVisible()
    expect(getRequests().every(request => request.method === "GET")).toBe(true)
  })

  it("opens every canonical recorded model configuration and preserves model aliases", async () => {
    mount()
    const dialog = await openDetail()
    expect(await within(dialog).findByText("qwen2.5:0.5b")).toBeVisible()
    expect(within(dialog).getByText("recorded-second-model")).toBeVisible()
    expect(within(dialog).getByText("ollama")).toBeVisible()
    expect(within(dialog).getByText("openai")).toBeVisible()
    expect(within(dialog).getByText("32")).toBeVisible()
    expect(within(dialog).getByText("0.8")).toBeVisible()
    expect(getRequests().map(request => request.method)).toEqual(["GET", "GET"])
  })

  it("retains an older server's single legacy configuration when canonical configs are absent", async () => {
    delete saved.model_configs
    saved.config = { provider: "ollama", model_name: "legacy-saved-model", temperature: 0.3, max_tokens: 12 }
    mount()
    const dialog = await openDetail()
    expect(await within(dialog).findByText("legacy-saved-model")).toBeVisible()
    expect(within(dialog).getByText("0.3")).toBeVisible()
    expect(within(dialog).getByText("12")).toBeVisible()
  })

  it("shows missing counts, metrics and configs as unavailable rather than inventing zero or defaults", async () => {
    saved.test_case_ids = null
    saved.model_configs = null
    saved.config = {}
    saved.aggregate_metrics = null
    saved.metrics = {}
    mount()
    const row = (await screen.findByText("Rowan completed")).closest("tr")!
    expect(within(row).getAllByText("Unavailable")).toHaveLength(2)
    expect(within(row).queryByText("0")).not.toBeInTheDocument()
    const dialog = await openDetail()
    expect(await within(dialog).findAllByText("Unavailable")).toHaveLength(3)
    expect(within(dialog).queryByText("qwen2.5:0.5b")).not.toBeInTheDocument()
  })

  it("distinguishes an actual empty case list from missing metadata", async () => {
    saved.test_case_ids = []
    mount()
    const row = (await screen.findByText("Rowan completed")).closest("tr")!
    expect(within(row).getByText("0")).toBeVisible()
    const dialog = await openDetail()
    expect(await within(dialog).findByText("Test Cases")).toBeVisible()
    const item = within(dialog).getByText("Test Cases").closest("tr")!
    expect(within(item).getByText("0")).toBeVisible()
  })

  it("uses existing legacy metrics when the canonical aggregate field is absent", async () => {
    delete saved.aggregate_metrics
    saved.metrics = { accuracy: 0.25 }
    mount()
    expect(await screen.findByText("25.0%")).toBeVisible()
    const dialog = await openDetail()
    expect((await within(dialog).findByText("Accuracy")).closest(".ant-statistic")).toHaveTextContent("25.0%")
  })

  it("does not render nonnumeric or absent metrics as a scored zero", async () => {
    saved.aggregate_metrics = { accuracy: null, pass_rate: "invalid", f1: null }
    mount()
    const row = (await screen.findByText("Rowan completed")).closest("tr")!
    expect(within(row).getByText("Unavailable")).toBeVisible()
    expect(within(row).queryByText("0.0%")).not.toBeInTheDocument()
    const dialog = await openDetail()
    expect(await within(dialog).findByText("Unavailable")).toBeVisible()
    expect(within(dialog).queryByText("NaN")).not.toBeInTheDocument()
  })

  it("preserves the saved failed status and error while showing recorded model provenance", async () => {
    saved.status = "failed"
    saved.error_message = "Recorded provider refusal"
    mount()
    const dialog = await openDetail()
    expect(await within(dialog).findByText("Recorded provider refusal")).toBeVisible()
    expect(within(dialog).getByText("Failed")).toBeVisible()
    expect(within(dialog).getByText("qwen2.5:0.5b")).toBeVisible()
    expect(getRequests().every(request => request.method === "GET")).toBe(true)
  })
})
