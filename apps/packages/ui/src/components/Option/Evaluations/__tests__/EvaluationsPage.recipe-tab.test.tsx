// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import { EvaluationsPage } from "../EvaluationsPage"

const { mockSearchParams, mockSetSearchParams } = vi.hoisted(() => ({
  mockSearchParams: {
    value: new URLSearchParams()
  },
  mockSetSearchParams: vi.fn()
}))

const storeState = {
  activeTab: "recipes" as any,
  setActiveTab: vi.fn(),
  setSelectedEvalId: vi.fn(),
  setSelectedRunId: vi.fn(),
  resetStore: vi.fn()
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

vi.mock("react-router-dom", () => ({
  useNavigate: () => vi.fn(),
  useSearchParams: () => [mockSearchParams.value, mockSetSearchParams]
}))

vi.mock("@/store/evaluations", () => ({
  useEvaluationsStore: (selector: (state: typeof storeState) => unknown) =>
    selector(storeState)
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))

vi.mock("@/components/Common/WorkspaceConnectionGate", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))

vi.mock("@/components/Common/DismissibleBetaAlert", () => ({
  DismissibleBetaAlert: () => null
}))

vi.mock("antd", () => ({
  Tabs: ({ items = [], activeKey }: any) => (
    <div data-testid="mock-tabs">
      <div>
        {items.map((item: any) => (
          <div key={item.key}>{item.label}</div>
        ))}
      </div>
      <div data-testid="active-tab-content">
        {items.find((item: any) => item.key === activeKey)?.children}
      </div>
    </div>
  )
}))

vi.mock("../tabs/RecipesTab", () => ({
  RecipesTab: () => <div data-testid="recipes-tab-panel">Recipes panel</div>
}))

vi.mock("../tabs/SyntheticReviewTab", () => ({
  SyntheticReviewTab: () => <div>Synthetic review panel</div>
}))

vi.mock("../tabs/EvaluationsTab", () => ({
  EvaluationsTab: () => <div>Evaluations panel</div>
}))

vi.mock("../tabs/RunsTab", () => ({
  RunsTab: () => <div>Runs panel</div>
}))

vi.mock("../tabs/DatasetsTab", () => ({
  DatasetsTab: () => <div>Datasets panel</div>
}))

vi.mock("../tabs/WebhooksTab", () => ({
  WebhooksTab: () => <div>Webhooks panel</div>
}))

vi.mock("../tabs/HistoryTab", () => ({
  HistoryTab: () => <div>History panel</div>
}))

describe("EvaluationsPage recipe-first entry", () => {
  beforeEach(() => {
    mockSearchParams.value = new URLSearchParams()
    vi.clearAllMocks()
  })

  it("exposes a Recipes tab as the primary entry point", () => {
    render(<EvaluationsPage />)

    expect(screen.getByText("Recipes")).toBeInTheDocument()
    expect(screen.getByText("Review")).toBeInTheDocument()
    expect(screen.getByTestId("recipes-tab-panel")).toBeInTheDocument()
  })

  it("renders the tour notice with the design-system Alert primitive", () => {
    mockSearchParams.value = new URLSearchParams("tour=1")

    const { container } = render(<EvaluationsPage />)

    expect(screen.getByText("Evaluations tour")).toBeInTheDocument()
    expect(
      screen.getByText("Evaluations tour").closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
  })
})
