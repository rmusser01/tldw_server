import { fireEvent, render, screen } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"
import { DataTablesPage } from "../DataTablesPage"

const state = {
  activeTab: "tables" as "tables" | "create",
  setActiveTab: vi.fn(),
  addSource: vi.fn(),
  setWizardStep: vi.fn(),
  setGeneratedTable: vi.fn(),
  resetStore: vi.fn(),
  resetWizard: vi.fn()
}

const connectionMocks = vi.hoisted(() => ({
  useConnectionUxState: vi.fn()
}))

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

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => connectionMocks.useConnectionUxState()
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="page-shell">{children}</div>
  )
}))

vi.mock("@/components/Common/DismissibleBetaAlert", () => ({
  DismissibleBetaAlert: () => <div data-testid="dt-beta-alert" />
}))

vi.mock("@/store/data-tables", () => ({
  useDataTablesStore: (selector: (store: typeof state) => unknown) =>
    selector(state)
}))

vi.mock("@/utils/data-tables-prefill", () => ({
  consumeDataTablesPrefill: vi.fn().mockResolvedValue(null)
}))

vi.mock("../DataTablesList", () => ({
  DataTablesList: () => <div data-testid="dt-list-pane">Tables pane</div>
}))

vi.mock("../CreateTableWizard", () => ({
  CreateTableWizard: () => <div data-testid="dt-create-pane">Create pane</div>
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("DataTablesPage golden path guardrails", () => {
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
    state.activeTab = "tables"
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    })
  })

  it("renders the studio shell with expected tab affordances", () => {
    render(
      <MemoryRouter initialEntries={["/data-tables"]}>
        <DataTablesPage />
      </MemoryRouter>
    )

    expect(screen.getByText("Data Tables Studio")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Generate structured tables from your chats, documents, and knowledge base using natural language prompts."
      )
    ).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "My Tables" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Create Table" })).toBeInTheDocument()
    expect(screen.getByTestId("dt-list-pane")).toBeInTheDocument()
  })

  it("preserves tab-switch interaction wiring to create flow", () => {
    render(
      <MemoryRouter initialEntries={["/data-tables"]}>
        <DataTablesPage />
      </MemoryRouter>
    )

    fireEvent.click(screen.getByRole("tab", { name: "Create Table" }))
    expect(state.setActiveTab).toHaveBeenCalledWith("create")
  })
})
