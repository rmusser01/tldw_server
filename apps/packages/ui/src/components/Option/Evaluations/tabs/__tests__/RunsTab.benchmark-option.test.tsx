// @vitest-environment jsdom

import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { RunsTab } from "../RunsTab"
import type { EvaluationRunDetail } from "@/services/evaluations"
import { downloadBlob } from "@/utils/download-blob"

const createRun = vi.fn().mockResolvedValue({})

const storeState = {
  selectedEvalId: null as string | null,
  setSelectedEvalId: vi.fn(),
  selectedRunId: null as string | null,
  setSelectedRunId: vi.fn(),
  runConfigText: "",
  setRunConfigText: vi.fn(),
  datasetOverrideText: "",
  setDatasetOverrideText: vi.fn(),
  runIdempotencyKey: "run-idem-1",
  regenerateRunIdempotencyKey: vi.fn(),
  quotaSnapshot: null as any,
  setQuotaSnapshot: vi.fn(),
  isPolling: false,
  setIsPolling: vi.fn(),
  adhocEndpoint: "benchmark-run",
  setAdhocEndpoint: vi.fn(),
  selectedBenchmark: "bullshit_benchmark" as string | null,
  setSelectedBenchmark: vi.fn(),
  adhocPayloadText: "{}",
  setAdhocPayloadText: vi.fn(),
  adhocResult: null as any
}

const runDetailState = {
  data: null as { ok: boolean; data: EvaluationRunDetail } | null,
  isLoading: false,
  isError: false,
  error: null as Error | null
}

vi.mock("@/utils/download-blob", () => ({ downloadBlob: vi.fn() }))

const runsListState = {
  data: { data: { data: [] as any[] } },
  isLoading: false,
  isError: false,
  error: null as Error | null
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            id?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: vi.fn()
  })
}))

vi.mock("../../hooks/useRuns", () => {
  return {
    useRateLimits: () => ({ data: null, isLoading: false, isError: false }),
    useRunsList: () => runsListState,
    useRunDetail: () => runDetailState,
    useCreateRun: () => ({ mutateAsync: createRun, isPending: false }),
    useCancelRun: () => ({ mutateAsync: vi.fn(), isPending: false }),
    useAdhocEvaluation: () => ({ mutateAsync: vi.fn(), isPending: false }),
    useRunBenchmark: () => ({ mutateAsync: vi.fn(), isPending: false }),
    extractMetricsSummary: () => [],
    useBenchmarksCatalog: () => ({
      data: { data: { data: [{ name: "bullshit_benchmark", description: "Bullshit" }] } },
      isLoading: false,
      isError: false
    }),
    adhocEndpointOptions: [
      { value: "response-quality", label: "response-quality" },
      { value: "benchmark-run", label: "benchmark-run" }
    ]
  }
})

vi.mock("../../hooks/useEvaluations", () => ({
  useEvaluationsList: () => ({ data: { data: { data: [] } } })
}))

vi.mock("@/store/evaluations", () => ({
  useEvaluationsStore: Object.assign(
    (selector: (state: typeof storeState) => unknown) => selector(storeState),
    { getState: () => storeState }
  )
}))

vi.mock("../../components", () => ({
  CopyButton: () => null,
  JsonEditor: ({
    value,
    onChange
  }: {
    value?: string
    onChange?: (next: string) => void
  }) => (
    <textarea
      data-testid="json-editor"
      value={value || ""}
      onChange={(e) => onChange?.(e.target.value)}
    />
  ),
  EvaluationsBreadcrumb: () => null,
  MetricsChart: () => null,
  PollingIndicator: () => null,
  RateLimitsWidget: () => null,
  RunComparisonView: () => null,
  StatusBadge: () => null
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("RunsTab benchmark run mode", () => {
  const originalMatchMedia = window.matchMedia

  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
  })

  beforeEach(() => {
    vi.clearAllMocks()
    storeState.selectedEvalId = null
    storeState.selectedRunId = null
    storeState.runConfigText = ""
    storeState.runIdempotencyKey = "run-idem-1"
    storeState.regenerateRunIdempotencyKey.mockReset()
    storeState.datasetOverrideText = ""
    storeState.adhocEndpoint = "benchmark-run"
    storeState.selectedBenchmark = "bullshit_benchmark"
    storeState.adhocPayloadText = "{}"
    runsListState.data = { data: { data: [] } }
    runsListState.isLoading = false
    runsListState.isError = false
    runsListState.error = null
    runDetailState.data = null
    runDetailState.isLoading = false
    runDetailState.isError = false
    runDetailState.error = null
  })

  it("binds the visible run key through regeneration and manual submission", async () => {
    storeState.selectedEvalId = "eval-key"
    let nextKey = 1
    storeState.regenerateRunIdempotencyKey.mockImplementation(() => {
      storeState.runIdempotencyKey = `run-idem-${++nextKey}`
    })
    const { container } = render(<RunsTab />)
    fireEvent.click(screen.getByRole("button", { name: /Advanced options/ }))
    const input = container.querySelector(".ant-space-compact input") as HTMLInputElement
    expect(input).toHaveValue("run-idem-1")
    expect(screen.getByLabelText("Idempotency key")).toBe(input)
    fireEvent.click(screen.getByRole("button", { name: "Regenerate", exact: true }))
    await waitFor(() => expect(input).toHaveValue("run-idem-2"))
    fireEvent.change(input, { target: { value: "manual-run-key" } })
    fireEvent.click(within(input.closest("form")!).getByRole("button", { name: "Start run", exact: true }))
    await waitFor(() => expect(createRun).toHaveBeenCalledWith(expect.objectContaining({
      evalId: "eval-key", idempotencyKey: "manual-run-key",
    })))
    await waitFor(() => expect(input).toHaveValue("run-idem-3"))
    fireEvent.click(within(input.closest("form")!).getByRole("button", { name: "Start run", exact: true }))
    await waitFor(() => expect(createRun).toHaveBeenCalledTimes(2))
    expect(createRun.mock.calls.map(([request]) => request.idempotencyKey)).toEqual(["manual-run-key", "run-idem-3"])
  })

  it.each(["manual edit", "Regenerate"])("preserves a newer key after admission: %s", async (change) => {
    storeState.selectedEvalId = "eval-key"
    storeState.regenerateRunIdempotencyKey.mockImplementation(() => {
      storeState.runIdempotencyKey = `generated-${storeState.runIdempotencyKey}`
    })
    let resolveAdmission!: (value: object) => void
    createRun.mockImplementationOnce(() => new Promise((resolve) => { resolveAdmission = resolve }))
    render(<RunsTab />)
    fireEvent.click(screen.getByRole("button", { name: /Advanced options/ }))
    const input = screen.getByLabelText("Idempotency key")
    fireEvent.click(within(input.closest("form")!).getByRole("button", { name: "Start run", exact: true }))
    await waitFor(() => expect(createRun).toHaveBeenCalledTimes(1))
    try {
      if (change === "manual edit") {
        fireEvent.change(input, { target: { value: "newer-manual-key" } })
      } else {
        fireEvent.click(screen.getByRole("button", { name: "Regenerate", exact: true }))
      }
      const newerKey = (input as HTMLInputElement).value
      expect(newerKey).not.toBe("run-idem-1")
      resolveAdmission({})
      await waitFor(() => expect(storeState.regenerateRunIdempotencyKey).toHaveBeenCalledTimes(change === "manual edit" ? 1 : 2))
      expect(input).toHaveValue(newerKey)
      expect(createRun.mock.calls[0][0].idempotencyKey).toBe("run-idem-1")
    } finally {
      resolveAdmission({})
    }
  })

  it("shows bullshit_benchmark in benchmark selector when benchmark-run mode is selected", async () => {
    render(<RunsTab />)
    expect(await screen.findByText("bullshit_benchmark")).toBeInTheDocument()
  })

  it("shows run-list recovery diagnostics without clearing run form input", () => {
    storeState.selectedEvalId = "eval-1"
    storeState.runConfigText = '{"temperature":0.2}'
    runsListState.isError = true
    runsListState.error = new Error("HTTP 503")

    render(<RunsTab />)

    expect(screen.getByText("Error")).toBeInTheDocument()
    expect(screen.getByText("Unable to load runs")).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent(
      "/api/v1/evaluations/eval-1/runs"
    )
    expect(
      screen
        .getAllByTestId("json-editor")
        .some((editor) => (editor as HTMLTextAreaElement).value === '{"temperature":0.2}')
    ).toBe(true)
  })
})

const savedRun: EvaluationRunDetail = {
  id: "run-owned",
  eval_id: "eval-owned",
  status: "completed",
  target_model: "precomputed",
  created: 1700000000,
  results: {
    provenance: {
      evaluation_id: "eval-owned",
      evaluation_type: "exact_match",
      evaluation_content_hash: "executed-recipe-hash",
      dataset_id: "dataset-owned",
      dataset_source: "dataset",
      dataset_content_hash: "executed-data-hash"
    },
    sample_results: [
      { sample_id: "sample_000000", input: { output: "Mira" }, expected: "Mira", passed: true },
      { sample_id: "sample_000001", input: { output: "Wrong" }, expected: "Mira", passed: false }
    ]
  }
}

function selectSavedRun(run: EvaluationRunDetail = savedRun) {
  storeState.selectedEvalId = run.eval_id
  storeState.selectedRunId = run.id
  runDetailState.data = { ok: true, data: run }
}

async function readExportedJson() {
  const [blob] = vi.mocked(downloadBlob).mock.calls.at(-1)!
  const text = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsText(blob)
  })
  return JSON.parse(text)
}

describe("saved run provenance and export", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    selectSavedRun()
    runDetailState.isError = false
    runDetailState.isLoading = false
    runDetailState.error = null
  })

  it("exports the loaded owned run with original expected/actual samples and dataset version", async () => {
    render(<RunsTab />)
    expect(screen.getByText("dataset-owned")).toBeInTheDocument()
    expect(screen.getByText("sha256:executed-data-hash")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Export JSON" }))
    expect(downloadBlob).toHaveBeenCalledTimes(1)
    expect(vi.mocked(downloadBlob).mock.calls[0][1]).toBe("run-owned.json")
    expect(await readExportedJson()).toEqual(savedRun)
  })

  it("reports unavailable historical provenance and exports legacy scores without inventing examples", async () => {
    const legacy = { ...savedRun, results: { aggregate: { mean_score: 0.5 } } }
    selectSavedRun(legacy)
    render(<RunsTab />)
    expect(screen.getByText("Dataset provenance was not captured for this run.")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Export JSON" }))
    expect(await readExportedJson()).toEqual(legacy)
  })

  it("blocks export of stale detail data when the owner-filtered read fails", () => {
    runDetailState.isError = true
    runDetailState.error = new Error("HTTP 404")
    render(<RunsTab />)
    expect(screen.getByRole("button", { name: "Export JSON" })).toBeDisabled()
    expect(screen.getByText("Unable to load run details")).toBeInTheDocument()
    expect(downloadBlob).not.toHaveBeenCalled()
  })
})
