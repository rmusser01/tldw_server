import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { expectInsideDesignSystemAlertAsync } from "@/test-utils/designSystemAlert"
import { TableDetailModal } from "../TableDetailModal"
import type { DataTable } from "@/types/data-tables"

const { getDataTableMock } = vi.hoisted(() => ({
  getDataTableMock: vi.fn()
}))

const dataTablesState = {
  currentTable: null as DataTable | null,
  currentTableLoading: false,
  editingState: {
    editingCellKey: null as string | null,
    isDirty: false,
    pendingChanges: []
  },
  setCurrentTable: vi.fn(),
  setCurrentTableLoading: vi.fn(),
  stopEditing: vi.fn(),
  updateTableInList: vi.fn()
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
    getDataTable: getDataTableMock,
    getDataTableJob: vi.fn(),
    regenerateDataTable: vi.fn()
  }
}))

vi.mock("@/utils/data-table-export", () => ({
  exportAndDownload: vi.fn()
}))

vi.mock("@/utils/data-tables-jobs", () => ({
  pollDataTableJob: vi.fn()
}))

vi.mock("../EditableDataTable", () => ({
  EditableDataTable: () => <div data-testid="editable-data-table" />
}))

const renderModal = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <TableDetailModal open tableId="table-1" onClose={vi.fn()} />
    </QueryClientProvider>
  )
}

describe("TableDetailModal design-system alerts", () => {
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

    if (!(globalThis as any).ResizeObserver) {
      ;(globalThis as any).ResizeObserver = class ResizeObserver {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
    }
  })

  beforeEach(() => {
    vi.clearAllMocks()
    dataTablesState.currentTable = null
    dataTablesState.currentTableLoading = false
    dataTablesState.editingState.editingCellKey = null
    dataTablesState.editingState.isDirty = false
    dataTablesState.editingState.pendingChanges = []
    getDataTableMock.mockRejectedValue(new Error("Failed to load table details"))
  })

  it("renders load failures through the design-system Alert primitive", async () => {
    renderModal()

    const alert = await expectInsideDesignSystemAlertAsync("Error")
    expect(alert).toHaveAttribute("role", "alert")
    expect(screen.getByText("Failed to load table details")).toBeInTheDocument()
  })
})
