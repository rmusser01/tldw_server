import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { expectInsideDesignSystemAlert } from "@/test-utils/designSystemAlert"
import { SaveTablePanel } from "../SaveTablePanel"
import type { DataTable } from "@/types/data-tables"

const dataTablesState = {
  generatedTable: null as DataTable | null,
  addTable: vi.fn(),
  setActiveTab: vi.fn(),
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

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    updateDataTable: vi.fn()
  }
}))

vi.mock("@/utils/data-table-export", () => ({
  exportAndDownload: vi.fn()
}))

const renderPanel = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <SaveTablePanel />
    </QueryClientProvider>
  )
}

describe("SaveTablePanel design-system alerts", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    dataTablesState.generatedTable = null
  })

  it("renders the missing-table warning through the design-system Alert primitive", () => {
    renderPanel()

    const alert = expectInsideDesignSystemAlert("No table to save")
    expect(alert).toHaveAttribute("role", "alert")
    expect(
      screen.getByText("Go back to the previous step to generate a table first.")
    ).toBeInTheDocument()
  })
})
