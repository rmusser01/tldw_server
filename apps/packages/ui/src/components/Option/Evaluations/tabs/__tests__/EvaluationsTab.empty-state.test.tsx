import { fireEvent, render, screen } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { EvaluationsTab } from "../EvaluationsTab"

const storeState = {
  selectedEvalId: null as string | null,
  setActiveTab: vi.fn(),
  setSelectedEvalId: vi.fn(),
  setSelectedRunId: vi.fn(),
  editingEvalId: null as string | null,
  setEditingEvalId: vi.fn(),
  createEvalOpen: false,
  openCreateEval: vi.fn(),
  closeCreateEval: vi.fn(),
  evalSpecText: "",
  setEvalSpecText: vi.fn(),
  evalSpecError: null as string | null,
  setEvalSpecError: vi.fn(),
  inlineDatasetEnabled: false,
  setInlineDatasetEnabled: vi.fn(),
  inlineDatasetText: "",
  setInlineDatasetText: vi.fn(),
  evalIdempotencyKey: "idem-1",
  regenerateEvalIdempotencyKey: vi.fn()
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
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

vi.mock("../../hooks/useEvaluations", () => ({
  useEvaluationsList: () => ({
    data: { data: { data: [] } },
    isLoading: false,
    isError: false
  }),
  useEvaluationDetail: () => ({
    data: null,
    isLoading: false,
    isError: false
  }),
  useCreateEvaluation: () => ({
    mutateAsync: vi.fn(),
    isPending: false
  }),
  useUpdateEvaluation: () => ({
    mutateAsync: vi.fn(),
    isPending: false
  }),
  useDeleteEvaluation: () => ({
    mutateAsync: vi.fn(),
    isPending: false
  }),
  useEvaluationDefaults: () => ({
    data: undefined
  }),
  getDefaultEvalSpecForType: () => ({})
}))

vi.mock("../../hooks/useDatasets", () => ({
  useDatasetsList: () => ({
    data: { data: { data: [] } },
    isLoading: false
  })
}))

vi.mock("@/store/evaluations", () => ({
  useEvaluationsStore: (selector: (state: typeof storeState) => unknown) =>
    selector(storeState)
}))

vi.mock("../../components", () => ({
  CopyButton: () => null,
  CreateEvaluationWizard: () => <div data-testid="create-eval-wizard" />
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("EvaluationsTab empty-state guardrails", () => {
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
    storeState.createEvalOpen = false
  })

  it("keeps the clean empty state and disabled edit/delete affordances", () => {
    render(<EvaluationsTab />)

    expect(
      screen.getByText("No evaluations yet. Start with Recipes for guided setup, or create a custom evaluation.")
    ).toBeInTheDocument()
    expect(
      screen.getByText("Select an evaluation to inspect its spec.")
    ).toBeInTheDocument()

    expect(screen.getByRole("button", { name: "New evaluation" })).toBeEnabled()
    expect(screen.getByRole("button", { name: "Edit" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Delete" })).toBeDisabled()
  })

  it("offers a direct recovery path back to the recipe-first workflow", () => {
    render(<EvaluationsTab />)

    fireEvent.click(screen.getByRole("button", { name: "Open Recipes" }))

    expect(storeState.setActiveTab).toHaveBeenCalledWith("recipes")
  })
})
