import React from "react"
import { ConfigProvider } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import commonEn from "@/assets/locale/en/common.json"
import settingsEn from "@/assets/locale/en/settings.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import { apiSend } from "@/services/api-send"
import { PromptPageErrorBoundary } from "../../../PromptPageErrorBoundary"
import { OptimizationProgressPanel } from "../OptimizationProgressPanel"

vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))

const pending = {
  id: 1, project_id: 2, initial_prompt_id: 3, status: "pending",
  optimizer_type: "iterative", max_iterations: 1, iterations_completed: 0,
  optimized_prompt_id: null, initial_metrics: null, final_metrics: null,
  error_message: null,
  optimization_config: { max_iterations: 1, model_config: {}, optimizer_type: "iterative" }
}
let detail: Record<string, unknown>
let iterations: unknown[]
let refused: "detail" | "iterations" | null
let client: QueryClient | undefined
const envelope = (data: unknown) => ({ ok: true, status: 200, data: { success: true, data } })
const calls = () => vi.mocked(apiSend).mock.calls.map(([request]) => request)

async function mount() {
  const i18n = createInstance()
  await i18n.use(ICUWithInterpolation).init({
    lng: "en", fallbackLng: false, ns: ["settings", "common"], defaultNS: "settings",
    resources: { en: { common: commonEn, settings: settingsEn } },
    interpolation: { escapeValue: false }
  })
  expect(i18n.t("managePrompts.studio.optimizations.progress.title")).toBe("Optimization Progress")
  client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  return render(
    <I18nextProvider i18n={i18n}>
      <ConfigProvider theme={{ token: { motion: false } }}>
        <QueryClientProvider client={client}>
          <PromptPageErrorBoundary>
            <OptimizationProgressPanel open optimizationId={1} onClose={vi.fn()} />
          </PromptPageErrorBoundary>
        </QueryClientProvider>
      </ConfigProvider>
    </I18nextProvider>
  )
}

beforeEach(() => {
  vi.mocked(apiSend).mockReset()
  detail = { ...pending }
  iterations = []
  refused = null
  vi.mocked(apiSend).mockImplementation(async (request) => {
    if (request.method !== "GET") throw new Error("Progress must not submit work")
    if (request.path === "/api/v1/prompt-studio/optimizations/get/1") {
      return (refused === "detail" ? { ok: false, status: 404, error: "Not found" } : envelope(detail)) as never
    }
    if (request.path === "/api/v1/prompt-studio/optimizations/iterations/1?page=1&per_page=50") {
      return (refused === "iterations" ? { ok: true, status: 200, data: { success: false, error: "Iteration access refused" } } : envelope({ iterations })) as never
    }
    throw new Error(`Unexpected request: ${request.path}`)
  })
})
afterEach(() => {
  cleanup()
  client?.clear()
  client = undefined
  vi.restoreAllMocks()
})

describe("optimization progress with real English resources and ICU", () => {
  it("renders the original pending optimization without crashing the Prompts boundary", async () => {
    await mount()
    expect(await screen.findByRole("heading", { name: "Optimization #1" })).toBeInTheDocument()
    expect(screen.getByText("Pending")).toBeInTheDocument()
    expect(screen.getAllByText("Optimization Progress")).toHaveLength(2)
    expect(screen.getByText("0 / 1 iterations")).toBeInTheDocument()
    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(await screen.findByText("No stored iteration records")).toBeInTheDocument()
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0")
    expect(screen.queryByTestId("prompts-error-boundary")).not.toBeInTheDocument()
    expect(calls()).toEqual([
      { path: "/api/v1/prompt-studio/optimizations/get/1", method: "GET" },
      { path: "/api/v1/prompt-studio/optimizations/iterations/1?page=1&per_page=50", method: "GET" }
    ])
  })

  it("renders running progress without inventing a final score", async () => {
    detail = { ...pending, status: "running", max_iterations: 2, iterations_completed: 1 }
    await mount()
    expect(await screen.findByText("Running")).toBeInTheDocument()
    expect(screen.getByText("1 / 2 iterations")).toBeInTheDocument()
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50")
    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.queryByTestId("prompts-error-boundary")).not.toBeInTheDocument()
  })

  it("renders completed stored scores, iteration evidence and interpolated saved prompt", async () => {
    detail = { ...pending, status: "completed", iterations_completed: 1, optimized_prompt_id: 22, final_metrics: { score: 0.75 } }
    iterations = [{ id: 101, iteration_number: 1, metrics: { score: 0 }, note: "Source support retained.", created_at: "2026-10-09T04:00:30Z" }]
    await mount()
    expect(await screen.findByText("Completed", { selector: '[data-ds-component="Badge"]' })).toBeInTheDocument()
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100")
    expect(document.querySelector(".ant-statistic-content")?.textContent).toBe("75.0%")
    expect(screen.getByText(/optimized prompt has been saved as Prompt #22/)).toBeInTheDocument()
    expect(await screen.findByText("Iteration 1")).toBeInTheDocument()
    expect(screen.getByText("0.0%")).toBeInTheDocument()
    expect(screen.getByText("Source support retained.")).toBeInTheDocument()
    expect(screen.queryByText("No stored iteration records")).not.toBeInTheDocument()
  })

  it("renders an actual failed status and stored error without a page crash", async () => {
    detail = { ...pending, status: "failed", error_message: "Provider unavailable" }
    await mount()
    expect(await screen.findByText("Failed")).toBeInTheDocument()
    expect(screen.getByText("Provider unavailable")).toBeInTheDocument()
    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.queryByTestId("prompts-error-boundary")).not.toBeInTheDocument()
  })

  it("preserves detail refusal as an error instead of a successful report", async () => {
    refused = "detail"
    await mount()
    expect(await screen.findByText("Unable to load optimization")).toBeInTheDocument()
    expect(screen.queryByRole("heading", { name: "Optimization #1" })).not.toBeInTheDocument()
    expect(screen.queryByText("Best prompt found")).not.toBeInTheDocument()
    expect(screen.queryByTestId("prompts-error-boundary")).not.toBeInTheDocument()
  })

  it("distinguishes refused stored iterations from an empty history", async () => {
    refused = "iterations"
    await mount()
    expect(await screen.findByText("Unable to load stored iterations")).toBeInTheDocument()
    expect(screen.getByText("Pending")).toBeInTheDocument()
    expect(screen.queryByText("No stored iteration records")).not.toBeInTheDocument()
    await waitFor(() => expect(calls().every((request) => request.method === "GET")).toBe(true))
    expect(screen.queryByTestId("prompts-error-boundary")).not.toBeInTheDocument()
  })
})
