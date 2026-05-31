import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { TablePreview } from "../TablePreview"
import type { DataTable } from "@/types/data-tables"

const { dataTablesState, generateDataTableMock } = vi.hoisted(() => ({
  generateDataTableMock: vi.fn(),
  dataTablesState: {
    tableName: "Research notes",
    prompt: "Extract claims",
    selectedSources: [
      {
        type: "document",
        id: "doc-1",
        title: "Notes",
        snippet: "A short source preview"
      }
    ],
    columnHints: [],
    selectedModel: null,
    maxRows: 100,
    generatedTable: null as DataTable | null,
    generationError: null as string | null,
    generationWarnings: [] as string[],
    editingTable: null as DataTable | null,
    editingRows: [] as Array<Record<string, unknown>>,
    editingState: {
      editingCellKey: null as string | null,
      isDirty: false,
      pendingChanges: []
    },
    setIsGenerating: vi.fn(),
    setGeneratedTable: vi.fn(),
    setGenerationError: vi.fn(),
    setGenerationWarnings: vi.fn(),
    startEditing: vi.fn(),
    stopEditing: vi.fn(),
    updateCell: vi.fn(),
    addRow: vi.fn(),
    deleteRow: vi.fn(),
    addColumn: vi.fn(),
    deleteColumn: vi.fn(),
    reorderColumns: vi.fn(),
    setEditingCellKey: vi.fn(),
    discardChanges: vi.fn()
  }
}))

const sampleTable: DataTable = {
  id: "table-1",
  name: "Research notes",
  prompt: "Extract claims",
  columns: [
    {
      id: "claim",
      name: "Claim",
      type: "text"
    }
  ],
  rows: [
    {
      Claim: "The dataset includes one useful row"
    }
  ],
  sources: [
    {
      type: "document",
      id: "doc-1",
      title: "Notes"
    }
  ],
  created_at: "2026-05-16T00:00:00.000Z",
  updated_at: "2026-05-16T00:00:00.000Z",
  row_count: 1
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

vi.mock("@dnd-kit/react", () => ({
  DragDropProvider: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="drag-drop-provider">{children}</div>
  )
}))

vi.mock("@dnd-kit/react/sortable", () => ({
  useSortable: () => ({
    ref: vi.fn(),
    handleRef: vi.fn(),
    isDragging: false
  })
}))

vi.mock("@dnd-kit/collision", () => ({
  closestCenter: vi.fn()
}))

vi.mock("@/store/data-tables", () => ({
  useDataTablesStore: (selector: (state: typeof dataTablesState) => unknown) =>
    selector(dataTablesState)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    generateDataTable: generateDataTableMock,
    getDataTableJob: vi.fn(),
    getDataTable: vi.fn()
  }
}))

vi.mock("../EditableCell", () => ({
  EditableCell: ({ value }: { value?: React.ReactNode }) => <span>{value}</span>
}))

vi.mock("../AddColumnModal", () => ({
  AddColumnModal: () => null
}))

const renderPreview = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <TablePreview />
    </QueryClientProvider>
  )
}

const expectInsideDesignSystemAlert = (text: string | RegExp) => {
  const node = screen.getByText(text)
  const alert = node.closest('[data-ds-component="Alert"]')
  expect(alert).toHaveAttribute("data-ds-component", "Alert")
  return alert
}

describe("TablePreview product states", () => {
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

    if (!(globalThis as any).ResizeObserver) {
      ;(globalThis as any).ResizeObserver = class ResizeObserver {
        observe() {}
        unobserve() {}
        disconnect() {}
      }
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
    dataTablesState.generatedTable = null
    dataTablesState.generationError = null
    dataTablesState.generationWarnings = []
    dataTablesState.editingTable = null
    dataTablesState.editingRows = []
    dataTablesState.editingState.editingCellKey = null
    dataTablesState.editingState.isDirty = false
    dataTablesState.editingState.pendingChanges = []
  })

  it("renders generation failures through the design-system Alert primitive", () => {
    dataTablesState.generationError = "The table generation job failed"

    renderPreview()

    const alert = expectInsideDesignSystemAlert("Generation Failed")
    expect(alert).toHaveAttribute("role", "alert")
    expect(screen.getByText("The table generation job failed")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Try Again" })).toBeInTheDocument()
  })

  it("renders generation warnings through the design-system Alert primitive", () => {
    dataTablesState.generatedTable = sampleTable
    dataTablesState.editingTable = sampleTable
    dataTablesState.editingRows = sampleTable.rows.map((row, index) => ({
      ...row,
      _id: `row-${index}`
    }))
    dataTablesState.generationWarnings = [
      "Some source rows were skipped",
      "The preview was limited to the first 100 rows"
    ]

    renderPreview()

    const alert = expectInsideDesignSystemAlert("Warnings")
    expect(alert).toHaveAttribute("role", "alert")
    expect(screen.getByText("Some source rows were skipped")).toBeInTheDocument()
    expect(
      screen.getByText("The preview was limited to the first 100 rows")
    ).toBeInTheDocument()
  })
})
