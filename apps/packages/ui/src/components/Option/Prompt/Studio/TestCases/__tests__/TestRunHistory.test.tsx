import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ConfigProvider } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { apiSend } from "@/services/api-send"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import { usePromptStudioStore } from "@/store/prompt-studio"
import { TestCasesTab } from "../TestCasesTab"

vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({ config, loading: false, authorityLoading: false })
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key })
}))
vi.mock("@/components/Common/confirm-danger", () => ({ useConfirmDanger: () => vi.fn() }))
vi.mock("../TestCaseFormModal", () => ({ TestCaseFormModal: () => null }))
vi.mock("../TestCaseBulkPanel", () => ({ TestCaseBulkPanel: () => null }))
vi.mock("../TestCaseGenerateModal", () => ({ TestCaseGenerateModal: () => null }))
vi.mock("../TestCaseRunModal", () => ({ TestCaseRunModal: () => null }))

const token = (id: number) => `header.${btoa(JSON.stringify({ sub: String(id) }))}.signature`
let config: TldwConfig
let client: QueryClient
let root: ReturnType<typeof render>
let requests: Array<{ path: string; method: string; headers: Headers }>
let rows: Array<Record<string, unknown>>
let detailStatus: number
let detailOverride: Record<string, unknown> | undefined
let heldDetail: Promise<Response> | undefined
let heldPrompt: Promise<Response> | undefined
let emptyCases: boolean
let cookieOwnerId: number
const original = {
  id: 41, project_id: 2, prompt_id: 21, test_case_id: 31, prompt_name: "Rowan",
  prompt_version_number: 1, model_name: "qwen2.5:0.5b",
  prompt_system_prompt: "Give the full name.", prompt_user_prompt: "{{source}}", prompt_deleted: false,
  model_params: { provider: "ollama", temperature: 0.2, max_tokens: 32 },
  inputs: { source: "Dr. Mira Vale directs Rowan." }, expected_outputs: { response: "Dr. Mira Vale" },
  outputs: { response: "Mira Vale" }, created_at: "2026-10-08T18:00:00Z", execution_time_ms: 250,
  tokens_used: 7, scores: null, cost_estimate: null, error_message: null
}
const json = (data: unknown, status = 200) => new Response(JSON.stringify(data), {
  status, headers: { "Content-Type": "application/json" }
})
const component = () => <ConfigProvider theme={{ token: { motion: false } }}>
  <QueryClientProvider client={client}><TestCasesTab /></QueryClientProvider>
</ConfigProvider>

function mount() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  root = render(component())
}
async function openHistory() {
  fireEvent.click(await screen.findByRole("button", { name: "History" }))
  const dialog = await screen.findByRole("dialog")
  expect(within(dialog).getByText("Test run history")).toBeVisible()
  return dialog
}
async function openRun(id = 41) {
  fireEvent.click(await screen.findByRole("button", { name: `Open run #${id}` }))
}
const mutations = () => requests.filter(request => request.method !== "GET")

beforeEach(() => {
  vi.clearAllMocks()
  config = { serverUrl: "http://127.0.0.1:8000", authMode: "multi-user", accessToken: token(7) }
  requests = []
  rows = [original]
  detailStatus = 200
  detailOverride = undefined
  heldDetail = heldPrompt = undefined
  emptyCases = false
  cookieOwnerId = 7
  usePromptStudioStore.getState().resetStore()
  usePromptStudioStore.getState().setSelectedProjectId(2)
  vi.spyOn(tldwClient, "initialize").mockResolvedValue(undefined)
  vi.spyOn(tldwClient, "ensureConfigForRequest").mockImplementation(async () => ({ ...config }))
  vi.mocked(apiSend).mockImplementation(async () => ({ ok: true, status: 200, data: {
    success: true, data: emptyCases ? [] : [{ id: 31, project_id: 2, name: "Name case", inputs: { source: "Current edited source" }, is_golden: false }]
  } }))
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), window.location.origin)
    const request = { path: url.pathname + url.search, method: init?.method ?? "GET", headers: new Headers(init?.headers) }
    requests.push(request)
    if (url.pathname === "/api/v1/users/me/capabilities") return json({ user_id: config.authSource === "cookie-session" ? cookieOwnerId : Number(config.accessToken === token(8) ? 8 : 7),
      can_read_scheduled_tasks: false, can_read_notifications: false, can_read_monitoring_alerts: false })
    if (url.pathname === "/api/v1/prompt-studio/test-cases/runs") {
      const page = Number(url.searchParams.get("page"))
      const project = Number(url.searchParams.get("project_id"))
      return json({ success: true, data: project === 2 ? rows.slice((page - 1) * 20, page * 20) : [],
        metadata: { page, per_page: 20, total: project === 2 ? rows.length : 0 },
        pagination: { page, per_page: 20, total: project === 2 ? rows.length : 0 } })
    }
    if (url.pathname === "/api/v1/prompt-studio/test-cases/runs/41") {
      if (config.authSource === "cookie-session" && request.headers.get("X-TLDW-Expected-User-ID") !== String(cookieOwnerId))
        return json({ detail: { code: "request_config_scope_changed" } }, 412)
      if (heldPrompt && requests.filter(row => row.path.endsWith("runs/41")).length > 1) return heldPrompt
      if (heldDetail) return heldDetail
      return detailStatus === 200 ? json({ success: true, data: detailOverride ?? original })
        : json({ detail: "Refused" }, detailStatus)
    }
    throw new Error(`Unexpected history transport: ${request.path}`)
  }))
})
afterEach(() => { cleanup(); client?.clear(); vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.unstubAllEnvs() })

describe("actual Test Cases durable history", () => {
  it("opens saved snapshots and their exact original prompt version using only GET", async () => {
    mount()
    await openHistory()
    await openRun()
    expect(await screen.findByText('"Mira Vale"', { exact: false })).toBeVisible()
    expect(screen.getByText('"Dr. Mira Vale"', { exact: false })).toBeVisible()
    expect(screen.getByText("qwen2.5:0.5b")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Open prompt version #21" }))
    expect(await screen.findByText('"Give the full name."')).toBeVisible()
    expect(requests.find(request => request.path.includes("runs/41"))?.headers.get("X-TLDW-Expected-User-ID")).toBe("7")
    expect(requests.filter(request => request.path.endsWith("runs/41"))).toHaveLength(2)
    expect(requests.some(request => request.path.includes("/prompts/"))).toBe(false)
    expect(mutations()).toEqual([])
  })

  it("supports existing history even when there are no current test cases", async () => {
    emptyCases = true
    mount()
    await openHistory()
    expect(await screen.findByRole("button", { name: "Open run #41" })).toBeVisible()
    expect(mutations()).toEqual([])
  })

  it("refreshes and reopens server history without rerunning a case", async () => {
    mount()
    const drawer = await openHistory()
    await screen.findByRole("button", { name: "Open run #41" })
    rows = []
    fireEvent.click(within(drawer).getByRole("button", { name: "Refresh" }))
    await waitFor(() => expect(screen.queryByRole("button", { name: "Open run #41" })).not.toBeInTheDocument())
    fireEvent.click(within(drawer).getByRole("button", { name: "Close" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    rows = [original]
    await openHistory()
    expect(await screen.findByRole("button", { name: "Open run #41" })).toBeVisible()
    expect(mutations()).toEqual([])
  })

  it.each([403, 404, 412])("shows a truthful saved-run refusal (%s) without fabricated output", async status => {
    detailStatus = status
    mount()
    await openHistory()
    await openRun()
    expect(await screen.findByText("Unable to load saved test run.")).toBeVisible()
    expect(screen.queryByText('"Mira Vale"', { exact: false })).not.toBeInTheDocument()
    expect(mutations()).toEqual([])
  })

  it("renders a failed run and absent snapshots honestly", async () => {
    detailOverride = { ...original, outputs: null, expected_outputs: null, model_params: null,
      created_at: null, error_message: "Provider unavailable" }
    mount()
    await openHistory()
    await openRun()
    expect(await screen.findByText("Provider unavailable")).toBeVisible()
    expect(screen.getAllByText("Not recorded").length).toBeGreaterThan(1)
    expect(screen.queryByText('"Mira Vale"', { exact: false })).not.toBeInTheDocument()
  })

  it.each(["project", "principal"])("retires a held detail on %s change", async kind => {
    let resolve!: (value: Response) => void
    heldDetail = new Promise<Response>(done => { resolve = done })
    mount()
    await openHistory()
    await openRun()
    await waitFor(() => expect(requests.some(request => request.path.endsWith("runs/41"))).toBe(true))
    await act(async () => {
      if (kind === "project") usePromptStudioStore.getState().setSelectedProjectId(3)
      else { config = { ...config, accessToken: token(8) }; root.rerender(component()) }
    })
    await act(async () => { resolve(json({ success: true, data: original })) })
    expect(screen.queryByText('"Mira Vale"', { exact: false })).not.toBeInTheDocument()
    expect(mutations()).toEqual([])
  })

  it("refuses a mismatched stored project response instead of adopting it", async () => {
    detailOverride = { ...original, project_id: 3 }
    mount()
    await openHistory()
    await openRun()
    expect(await screen.findByText("Unable to load saved test run.")).toBeVisible()
    expect(screen.queryByText('"Mira Vale"', { exact: false })).not.toBeInTheDocument()
  })

  it("uses bounded server pages rather than silently truncating history", async () => {
    rows = Array.from({ length: 21 }, (_, index) => ({ ...original, id: 41 + index }))
    mount()
    await openHistory()
    await screen.findByRole("button", { name: "Open run #41" })
    fireEvent.click(screen.getByRole("button", { name: "Next page" }))
    expect(await screen.findByRole("button", { name: "Open run #61" })).toBeVisible()
    expect(screen.queryByRole("button", { name: "Open run #41" })).not.toBeInTheDocument()
    expect(requests.some(request => request.path.includes("page=2") && request.path.includes("per_page=20"))).toBe(true)
  })

  it("retires a held original-version preview when its run selection closes", async () => {
    let resolve!: (value: Response) => void
    heldPrompt = new Promise<Response>(done => { resolve = done })
    mount()
    const drawer = await openHistory()
    await openRun()
    await screen.findByText('"Mira Vale"', { exact: false })
    fireEvent.click(screen.getByRole("button", { name: "Open prompt version #21" }))
    await waitFor(() => expect(requests.filter(request => request.path.endsWith("runs/41"))).toHaveLength(2))
    fireEvent.click(within(drawer).getByRole("button", { name: "Close" }))
    await act(async () => { resolve(json({ success: true, data: { ...original, prompt_system_prompt: "Late private instructions" } })) })
    expect(screen.queryByText('"Late private instructions"')).not.toBeInTheDocument()
    expect(mutations()).toEqual([])
  })

  it("refuses an original-version preview after a cookie owner switch with colliding numeric IDs", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    config = { serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" }
    mount()
    await openHistory()
    await openRun()
    await screen.findByRole("button", { name: "Open prompt version #21" })
    cookieOwnerId = 8
    detailOverride = { ...original, prompt_system_prompt: "Other owner's colliding template" }
    fireEvent.click(screen.getByRole("button", { name: "Open prompt version #21" }))
    expect(await screen.findByText("Unable to load original prompt version.")).toBeVisible()
    expect(screen.queryByText('"Other owner\'s colliding template"')).not.toBeInTheDocument()
    expect(requests.filter(request => request.path.endsWith("runs/41"))).toHaveLength(2)
    expect(mutations()).toEqual([])
  })

  it.each([false, true])("shows missing/deleted original-version metadata honestly (deleted: %s)", async deleted => {
    detailOverride = { ...original, prompt_deleted: deleted, prompt_version_number: deleted ? 1 : null,
      prompt_system_prompt: deleted ? "Saved deleted template" : null, prompt_user_prompt: null }
    mount()
    await openHistory()
    await openRun()
    fireEvent.click(await screen.findByRole("button", { name: "Open prompt version #21" }))
    expect(await screen.findByText(deleted ? "Original prompt version is deleted; showing its saved template."
      : "Original prompt version is unavailable.")).toBeVisible()
    if (deleted) expect(screen.getByText('"Saved deleted template"')).toBeVisible()
    else expect(screen.queryByText('"Give the full name."')).not.toBeInTheDocument()
  })
})
