// @vitest-environment jsdom

import { render, screen } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { RunsTab } from "../RunsTab"

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
    useRunDetail: () => ({ data: null, isLoading: false, isError: false }),
    useCreateRun: () => ({ mutateAsync: vi.fn(), isPending: false }),
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
  useEvaluationsStore: (selector: (state: typeof storeState) => unknown) =>
    selector(storeState)
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
    storeState.datasetOverrideText = ""
    storeState.adhocEndpoint = "benchmark-run"
    storeState.selectedBenchmark = "bullshit_benchmark"
    storeState.adhocPayloadText = "{}"
    runsListState.data = { data: { data: [] } }
    runsListState.isLoading = false
    runsListState.isError = false
    runsListState.error = null
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

    expect(screen.getByText("Unavailable")).toBeInTheDocument()
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
