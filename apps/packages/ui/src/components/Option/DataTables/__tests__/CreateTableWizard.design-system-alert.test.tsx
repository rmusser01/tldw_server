import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { expectInsideDesignSystemAlert } from "@/test-utils/designSystemAlert"
import { CreateTableWizard } from "../CreateTableWizard"
import type { DataTable } from "@/types/data-tables"

const dataTablesState = {
  wizardStep: "sources" as "sources" | "prompt" | "preview" | "save",
  selectedSources: [] as Array<unknown>,
  tableName: "",
  prompt: "",
  generatedTable: null as DataTable | null,
  isGenerating: false,
  setWizardStep: vi.fn(),
  resetWizard: vi.fn()
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

vi.mock("@/store/data-tables", () => ({
  useDataTablesStore: (selector: (state: typeof dataTablesState) => unknown) =>
    selector(dataTablesState)
}))

vi.mock("../SourceSelector", () => ({
  SourceSelector: () => <div data-testid="source-selector" />
}))

vi.mock("../GenerationPanel", () => ({
  GenerationPanel: () => <div data-testid="generation-panel" />
}))

vi.mock("../TablePreview", () => ({
  TablePreview: () => <div data-testid="table-preview" />
}))

vi.mock("../SaveTablePanel", () => ({
  SaveTablePanel: () => <div data-testid="save-table-panel" />
}))

describe("CreateTableWizard design-system alerts", () => {
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

  beforeEach(() => {
    vi.clearAllMocks()
    dataTablesState.wizardStep = "sources"
    dataTablesState.selectedSources = []
    dataTablesState.tableName = ""
    dataTablesState.prompt = ""
    dataTablesState.generatedTable = null
    dataTablesState.isGenerating = false
  })

  it("renders the source-selection tip through the design-system Alert primitive", () => {
    render(<CreateTableWizard />)

    const alert = expectInsideDesignSystemAlert("Tip")
    expect(alert).toHaveAttribute("role", "status")
    expect(
      screen.getByText(
        "Select chats, documents, or search your knowledge base to extract structured data. The more specific your sources, the better the results."
      )
    ).toBeInTheDocument()
  })
})
