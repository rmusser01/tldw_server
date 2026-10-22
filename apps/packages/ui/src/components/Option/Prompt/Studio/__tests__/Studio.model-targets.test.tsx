import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ConfigProvider, notification } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { apiSend } from "@/services/api-send"
import { usePromptStudioStore } from "@/store/prompt-studio"
import { TestCaseRunModal } from "../TestCases/TestCaseRunModal"
import { CreateEvaluationWizard } from "../Evaluations/CreateEvaluationWizard"
import { CreateOptimizationWizard } from "../Optimizations/CreateOptimizationWizard"

vi.mock("@/services/api-send", () => ({ apiSend: vi.fn() }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, options?: { defaultValue?: string; [key: string]: unknown }) =>
    Object.entries(options ?? {}).reduce((s, [k, v]) => s.replaceAll(`{{${k}}}`, String(v)), options?.defaultValue ?? key) })
}))

const prompts = [
  { id: 21, project_id: 2, name: "Mira question", version_number: 1, content: "Who directs Rowan Observatory?" },
  { id: 22, project_id: 2, name: "Mira question", version_number: 2, content: "Answer only the director's name." }
]
const cases = [{ id: 31, project_id: 2, name: "F-SOURCE director", inputs: { source: "Rowan Observatory is directed by Dr. Mira Vale." }, expected_outputs: { response: "Dr. Mira Vale" }, is_golden: true }]
const providers = {
  default_provider: "ollama",
  providers: [
    { name: "ollama", display_name: "Ollama", default_model: "qwen2.5:0.5b", models: ["qwen2.5:0.5b", "gemma3:1b"] },
    { name: "anthropic", display_name: "Anthropic", default_model: "claude-sonnet", models: ["claude-sonnet"] }
  ]
}
const run = { id: 41, test_case_id: 31, prompt_id: 22, inputs: cases[0].inputs, expected: { response: "Dr. Mira Vale" }, actual: { response: "Mira Vale directs Rowan." }, model: "qwen2.5:0.5b", execution_time_ms: 250, tokens_used: 12 }
let catalog: unknown
let runData: unknown
let refusal = false
let client: QueryClient
const posts = () => vi.mocked(apiSend).mock.calls.map(([p]) => p).filter(p => p.method === "POST")

function mount(kind: "run" | "evaluation" | "optimization") {
  const onClose = vi.fn()
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const component = kind === "run" ? <TestCaseRunModal open projectId={2} testCaseIds={[31]} onClose={onClose} />
    : kind === "evaluation" ? <CreateEvaluationWizard open projectId={2} onClose={onClose} />
      : <CreateOptimizationWizard open projectId={2} onClose={onClose} />
  render(<ConfigProvider theme={{ token: { motion: false } }}><QueryClientProvider client={client}>{component}</QueryClientProvider></ConfigProvider>)
  return onClose
}

async function selectOption(label: string, option: string) {
  const combo = screen.getByRole("combobox", { name: label })
  fireEvent.mouseDown(combo)
  const item = await screen.findByText(option, { selector: ".ant-select-item-option-content" })
  fireEvent.click(item)
}
async function selectRunPrompt() {
  const combo = screen.getAllByRole("combobox")[0]
  fireEvent.mouseDown(combo)
  fireEvent.click(await screen.findByText("Mira question (v2)", { selector: ".ant-select-item-option-content" }))
}
async function configureWizard(kind: "evaluation" | "optimization") {
  await screen.findByText("v2")
  const row = screen.getByText("v2").closest("tr")!
  fireEvent.click(within(row).getByRole("radio"))
  fireEvent.click(screen.getByRole("button", { name: "Next" }))
  if (kind === "evaluation") fireEvent.click(await screen.findByRole("checkbox", { name: /Use all test cases/ }))
  fireEvent.click(screen.getByRole("button", { name: "Next" }))
  await screen.findByRole("combobox", { name: /Model/ })
}
async function submitWizard(kind: "evaluation" | "optimization") {
  fireEvent.click(screen.getByRole("button", { name: "Next" }))
  fireEvent.click(screen.getByRole("button", { name: kind === "evaluation" ? "Run Evaluation" : "Start Optimization" }))
  await waitFor(() => expect(posts()).toHaveLength(1))
}

beforeEach(() => {
  vi.clearAllMocks()
  usePromptStudioStore.getState().resetStore()
  catalog = providers
  runData = { results: [run] }
  refusal = false
  vi.spyOn(notification, "success").mockImplementation(() => {})
  vi.spyOn(notification, "error").mockImplementation(() => {})
  vi.mocked(apiSend).mockImplementation(async payload => {
    const path = String(payload.path)
    if (payload.method === "POST") return refusal ? { ok: false, status: 422, error: "Provider refused this target" } : { ok: true, status: 200, data: path.endsWith("test-cases/run") ? runData : { id: 51, status: "queued" } }
    if (path.startsWith("/api/v1/llm/providers")) return { ok: true, status: 200, data: catalog }
    if (path.includes("prompts/list/2")) return { ok: true, status: 200, data: { success: true, data: prompts } }
    if (path.includes("test-cases/list/2")) return { ok: true, status: 200, data: { success: true, data: cases } }
    if (path.endsWith("optimizations/strategies")) return { ok: true, status: 200, data: { success: true, data: [{ name: "iterative", display_name: "Iterative", description: "Controlled iteration", default_params: { max_iterations: 10, num_examples: 3 } }] } }
    throw new Error(`Unexpected transport ${path}`)
  })
})
afterEach(() => { cleanup(); client?.clear(); notification.destroy() })

describe("Prompt Studio canonical targets and run contracts", () => {
  it("sends the catalog default target with the exact saved prompt version for a test run", async () => {
    mount("run")
    await selectRunPrompt()
    fireEvent.click(screen.getByRole("button", { name: "Run Tests" }))
    await waitFor(() => expect(posts()).toHaveLength(1))
    expect(posts()[0]).toMatchObject({ path: "/api/v1/prompt-studio/test-cases/run", body: { prompt_id: 22, test_case_ids: [31], provider: "ollama", model: "qwen2.5:0.5b" } })
    expect(posts()[0].body).not.toHaveProperty("config")
  })
  it("displays canonical actual and expected responses, milliseconds, and an ungraded result", async () => {
    mount("run"); await selectRunPrompt()
    fireEvent.click(screen.getByRole("button", { name: "Run Tests" }))
    expect(await screen.findByText("Mira Vale directs Rowan.")).toBeVisible()
    expect(screen.getByText("Dr. Mira Vale")).toBeVisible()
    expect(screen.getByText("0.25s")).toBeVisible()
    expect(screen.getByText("1 ungraded")).toBeVisible()
    expect(screen.queryByText("Pass")).not.toBeInTheDocument()
    expect(screen.queryByText("Fail")).not.toBeInTheDocument()
  })
  it("preserves explicit evaluation provider and model through the review step", async () => {
    mount("evaluation"); await configureWizard("evaluation")
    await selectOption("Provider (optional)", "Anthropic")
    await selectOption("Model (optional)", "claude-sonnet")
    await submitWizard("evaluation")
    expect(posts()[0].body).toMatchObject({ project_id: 2, prompt_id: 22, test_case_ids: [31], config: { provider: "anthropic", model_name: "claude-sonnet" }, run_async: true })
  })
  it("preserves the chosen optimization cap over strategy defaults and sends its explicit target", async () => {
    mount("optimization"); await configureWizard("optimization")
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max Iterations" }), { target: { value: "1" } })
    await selectOption("Provider (optional)", "Ollama")
    await selectOption("Model (optional)", "qwen2.5:0.5b")
    await submitWizard("optimization")
    expect(posts()[0].body).toMatchObject({ project_id: 2, prompt_id: 22, config: { strategy: "iterative", max_iterations: 1, num_examples: 3 }, model_config: { provider: "ollama", model_name: "qwen2.5:0.5b" } })
  })
  it("keeps the reviewed iteration budget when model defaults are used", async () => {
    mount("optimization"); await configureWizard("optimization")
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max Iterations" }), { target: { value: "1" } })
    await submitWizard("optimization")
    expect(posts()[0].body).toMatchObject({ config: { strategy: "iterative", max_iterations: 1, num_examples: 3 } })
    expect(posts()[0].body).toHaveProperty("model_config", { provider: "ollama", model_name: "qwen2.5:0.5b", temperature: undefined })
  })
  it("resolves a blank evaluation target to the displayed catalog defaults", async () => {
    mount("evaluation"); await configureWizard("evaluation")
    await submitWizard("evaluation")
    expect(posts()[0].body).toMatchObject({ config: { provider: "ollama", model_name: "qwen2.5:0.5b" } })
  })
  it("clears an incompatible model when the provider changes and sends the new provider default", async () => {
    mount("run"); await selectRunPrompt()
    await selectOption("Provider (optional)", "Ollama")
    await selectOption("Model (optional)", "gemma3:1b")
    await selectOption("Provider (optional)", "Anthropic")
    await waitFor(() => expect(screen.queryByText("gemma3:1b", { selector: ".ant-select-selection-item" })).not.toBeInTheDocument())
    fireEvent.click(screen.getByRole("button", { name: "Run Tests" }))
    await waitFor(() => expect(posts()).toHaveLength(1))
    expect(posts()[0].body).toMatchObject({ provider: "anthropic", model: "claude-sonnet" })
  })
  it("refuses a selected provider removed from the current catalog before dispatch", async () => {
    mount("run"); await selectRunPrompt()
    await selectOption("Provider (optional)", "Ollama")
    await act(async () => { client.setQueryData(["prompt-studio", "llm-providers"], { ok: true, status: 200, data: { default_provider: "anthropic", providers: [providers.providers[1]] } }) })
    fireEvent.click(screen.getByRole("button", { name: "Run Tests" }))
    await waitFor(() => expect(notification.error).toHaveBeenCalledWith(expect.objectContaining({ description: "Provider is not available on this server" })))
    expect(posts()).toHaveLength(0)
  })
  it("preserves server defaults when the catalog is empty", async () => {
    catalog = { providers: [], default_provider: null }
    mount("run"); await selectRunPrompt()
    fireEvent.click(screen.getByRole("button", { name: "Run Tests" }))
    await waitFor(() => expect(posts()).toHaveLength(1))
    expect(posts()[0].body).toMatchObject({ provider: undefined, model: undefined })
    expect(await screen.findByText("Mira Vale directs Rowan.")).toBeVisible()
  })
  it("keeps an exact canonical answer ungraded and displays zero milliseconds", async () => {
    runData = { results: [{ ...run, actual: { response: "Dr. Mira Vale" }, execution_time_ms: 0 }] }
    mount("run"); await selectRunPrompt()
    fireEvent.click(screen.getByRole("button", { name: "Run Tests" }))
    await waitFor(() => expect(screen.getByText("1 ungraded")).toBeVisible())
    expect(screen.getAllByText("Dr. Mira Vale")).toHaveLength(2)
    expect(screen.getByText("0.00s")).toBeVisible()
    expect(screen.queryByText("Pass")).not.toBeInTheDocument()
  })
  it("preserves legacy explicit grades, output, error, and seconds", async () => {
    runData = { success: true, data: [
      { test_case_id: 31, output: "Reviewed answer", passed: true, execution_time: 0.42 },
      { test_case_id: 32, output: "Wrong answer", passed: false, execution_time: 0.67 },
      { test_case_id: 33, output: "", error: "Provider timed out", execution_time: 1.25 }
    ] }
    mount("run"); await selectRunPrompt()
    fireEvent.click(screen.getByRole("button", { name: "Run Tests" }))
    expect(await screen.findByText("Reviewed answer")).toBeVisible()
    expect(screen.getByText("Pass")).toBeVisible()
    expect(screen.getByText("Fail")).toBeVisible()
    expect(screen.getByText("Provider timed out")).toBeVisible()
    expect(screen.getByText("0.42s")).toBeVisible()
    expect(screen.queryByText(/ungraded/)).not.toBeInTheDocument()
  })
  it("refuses a fractional optimization budget before POST", async () => {
    mount("optimization"); await configureWizard("optimization")
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max Iterations" }), { target: { value: "1.5" } })
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    fireEvent.click(screen.getByRole("button", { name: "Start Optimization" }))
    await waitFor(() => expect(notification.error).toHaveBeenCalledWith(expect.objectContaining({ description: "Max Iterations must be an integer from 1 to 100" })))
    expect(posts()).toHaveLength(0)
  })
  it.each(["run", "evaluation", "optimization"] as const)("keeps an API refusal a failure in %s", async kind => {
    refusal = true
    const onClose = mount(kind)
    if (kind === "run") { await selectRunPrompt(); fireEvent.click(screen.getByRole("button", { name: "Run Tests" })); await waitFor(() => expect(posts()).toHaveLength(1)) }
    else { await configureWizard(kind); await submitWizard(kind) }
    await waitFor(() => expect(notification.error).toHaveBeenCalledWith(expect.objectContaining({ description: "Provider refused this target" })))
    expect(notification.success).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
    expect(screen.queryByText("Results")).not.toBeInTheDocument()
  })
})
