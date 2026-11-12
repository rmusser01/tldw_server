/**
 * Data Tables Zustand Store
 * Manages state for the Data Tables Studio feature
 */

import { createWithEqualityFn } from "zustand/traditional"
import type {
  CreateTableStep,
  DataTable,
  DataTableColumn,
  DataTableRow,
  DataTableSource,
  DataTableSourceType,
  DataTableSummary,
  DataTablesTab,
  ExportFormat,
  TableChange,
  TableEditingState
} from "@/types/data-tables"

// ─────────────────────────────────────────────────────────────────────────────
// State Types
// ─────────────────────────────────────────────────────────────────────────────

interface TablesState {
  tables: DataTableSummary[]
  tablesLoading: boolean
  tablesError: string | null
  tablesTotal: number
  tablesPage: number
  tablesPageSize: number
  tablesSearch: string
  selectedTableId: string | null
  currentTable: DataTable | null
  currentTableLoading: boolean
}

interface WizardState {
  wizardStep: CreateTableStep
  // Source selection
  selectedSources: DataTableSource[]
  sourceSearchQuery: string
  activeSourceType: DataTableSourceType
  // Generation
  tableName: string
  prompt: string
  columnHints: Partial<DataTableColumn>[]
  selectedModel: string | null
  maxRows: number
  // Generation result
  isGenerating: boolean
  generatedTable: DataTable | null
  generationError: string | null
  generationWarnings: string[]
}

interface ExportState {
  isExporting: boolean
  exportFormat: ExportFormat | null
  exportError: string | null
}

interface UIState {
  activeTab: DataTablesTab
  tableDetailOpen: boolean
  deleteConfirmOpen: boolean
  deleteTargetId: string | null
}

interface EditingState {
  editingTable: DataTable | null // Working copy for editing
  originalTable: DataTable | null // Snapshot for discard/diff
  editingRows: DataTableRow[] // Rows with stable IDs
  editingState: TableEditingState
}

// ─────────────────────────────────────────────────────────────────────────────
// Actions Types
// ─────────────────────────────────────────────────────────────────────────────

interface TablesActions {
  setTables: (tables: DataTableSummary[], total?: number) => void
  setTablesLoading: (loading: boolean) => void
  setTablesError: (error: string | null) => void
  setTablesPage: (page: number) => void
  setTablesPageSize: (size: number) => void
  setTablesSearch: (search: string) => void
  setSelectedTableId: (id: string | null) => void
  setCurrentTable: (table: DataTable | null) => void
  setCurrentTableLoading: (loading: boolean) => void
  addTable: (table: DataTableSummary) => void
  updateTableInList: (id: string, updates: Partial<DataTableSummary>) => void
  removeTable: (id: string) => void
}

interface WizardActions {
  setWizardStep: (step: CreateTableStep) => void
  // Source selection
  addSource: (source: DataTableSource) => void
  removeSource: (id: string) => void
  clearSources: () => void
  setSourceSearchQuery: (query: string) => void
  setActiveSourceType: (type: DataTableSourceType) => void
  // Generation
  setTableName: (name: string) => void
  setPrompt: (prompt: string) => void
  addColumnHint: (hint: Partial<DataTableColumn>) => void
  updateColumnHint: (index: number, hint: Partial<DataTableColumn>) => void
  removeColumnHint: (index: number) => void
  clearColumnHints: () => void
  setSelectedModel: (model: string | null) => void
  setMaxRows: (maxRows: number) => void
  // Generation result
  setIsGenerating: (isGenerating: boolean) => void
  setGeneratedTable: (table: DataTable | null) => void
  setGenerationError: (error: string | null) => void
  setGenerationWarnings: (warnings: string[]) => void
  // Reset wizard
  resetWizard: () => void
}

interface ExportActions {
  setIsExporting: (isExporting: boolean) => void
  setExportFormat: (format: ExportFormat | null) => void
  setExportError: (error: string | null) => void
}

interface UIActions {
  setActiveTab: (tab: DataTablesTab) => void
  openTableDetail: (id: string) => void
  closeTableDetail: () => void
  openDeleteConfirm: (id: string) => void
  closeDeleteConfirm: () => void
  resetStore: () => void
}

interface EditingActions {
  startEditing: (table: DataTable) => void
  stopEditing: () => void
  updateCell: (rowIndex: number, columnId: string, value: any) => void
  addRow: (afterIndex?: number) => void
  deleteRow: (rowIndex: number) => void
  addColumn: (column: DataTableColumn, afterColumnId?: string) => void
  deleteColumn: (columnId: string) => void
  reorderColumns: (fromIndex: number, toIndex: number) => void
  setEditingCellKey: (key: string | null) => void
  discardChanges: () => void
  getEditedTableData: () => { columns: DataTableColumn[]; rows: Record<string, any>[] } | null
}

// ─────────────────────────────────────────────────────────────────────────────
// Combined State & Actions
// ─────────────────────────────────────────────────────────────────────────────

export type DataTablesState = TablesState &
  WizardState &
  ExportState &
  UIState &
  EditingState &
  TablesActions &
  WizardActions &
  ExportActions &
  UIActions &
  EditingActions

// ─────────────────────────────────────────────────────────────────────────────
// Initial State
// ─────────────────────────────────────────────────────────────────────────────

const initialTablesState: TablesState = {
  tables: [],
  tablesLoading: false,
  tablesError: null,
  tablesTotal: 0,
  tablesPage: 1,
  tablesPageSize: 20,
  tablesSearch: "",
  selectedTableId: null,
  currentTable: null,
  currentTableLoading: false
}

const initialWizardState: WizardState = {
  wizardStep: "sources",
  selectedSources: [],
  sourceSearchQuery: "",
  activeSourceType: "chat",
  tableName: "",
  prompt: "",
  columnHints: [],
  selectedModel: null,
  maxRows: 100,
  isGenerating: false,
  generatedTable: null,
  generationError: null,
  generationWarnings: []
}

const initialExportState: ExportState = {
  isExporting: false,
  exportFormat: null,
  exportError: null
}

const initialUIState: UIState = {
  activeTab: "tables",
  tableDetailOpen: false,
  deleteConfirmOpen: false,
  deleteTargetId: null
}

const initialEditingState: EditingState = {
  editingTable: null,
  originalTable: null,
  editingRows: [],
  editingState: {
    editingCellKey: null,
    isDirty: false,
    pendingChanges: []
  }
}

const initialState = {
  ...initialTablesState,
  ...initialWizardState,
  ...initialExportState,
  ...initialUIState,
  ...initialEditingState
}

// ─────────────────────────────────────────────────────────────────────────────
// Store
// ─────────────────────────────────────────────────────────────────────────────

export const useDataTablesStore = createWithEqualityFn<DataTablesState>()((set) => ({
  ...initialState,

  // ─────────────────────────────────────────────────────────────────────────
  // Tables Actions
  // ─────────────────────────────────────────────────────────────────────────

  setTables: (tables, total) =>
    set({ tables, tablesTotal: total ?? tables.length }),
  setTablesLoading: (tablesLoading) => set({ tablesLoading }),
  setTablesError: (tablesError) => set({ tablesError }),
  setTablesPage: (tablesPage) => set({ tablesPage }),
  setTablesPageSize: (tablesPageSize) => set({ tablesPageSize, tablesPage: 1 }),
  setTablesSearch: (tablesSearch) => set({ tablesSearch, tablesPage: 1 }),
  setSelectedTableId: (selectedTableId) => set({ selectedTableId }),
  setCurrentTable: (currentTable) => set({ currentTable }),
  setCurrentTableLoading: (currentTableLoading) => set({ currentTableLoading }),

  addTable: (table) =>
    set((state) => ({
      tables: [table, ...state.tables],
      tablesTotal: state.tablesTotal + 1
    })),

  updateTableInList: (id, updates) =>
    set((state) => ({
      tables: state.tables.map((t) =>
        t.id === id ? { ...t, ...updates } : t
      )
    })),

  removeTable: (id) =>
    set((state) => ({
      tables: state.tables.filter((t) => t.id !== id),
      tablesTotal: Math.max(0, state.tablesTotal - 1),
      selectedTableId:
        state.selectedTableId === id ? null : state.selectedTableId,
      currentTable: state.currentTable?.id === id ? null : state.currentTable
    })),

  // ─────────────────────────────────────────────────────────────────────────
  // Wizard Actions
  // ─────────────────────────────────────────────────────────────────────────

  setWizardStep: (wizardStep) => set({ wizardStep }),

  // Source selection
  addSource: (source) =>
    set((state) => {
      // Prevent duplicates
      if (state.selectedSources.some((s) => s.id === source.id)) {
        return state
      }
      return { selectedSources: [...state.selectedSources, source] }
    }),

  removeSource: (id) =>
    set((state) => ({
      selectedSources: state.selectedSources.filter((s) => s.id !== id)
    })),

  clearSources: () => set({ selectedSources: [] }),

  setSourceSearchQuery: (sourceSearchQuery) => set({ sourceSearchQuery }),

  setActiveSourceType: (activeSourceType) => set({ activeSourceType }),

  // Generation
  setTableName: (tableName) => set({ tableName }),
  setPrompt: (prompt) => set({ prompt }),

  addColumnHint: (hint) =>
    set((state) => ({
      columnHints: [...state.columnHints, hint]
    })),

  updateColumnHint: (index, hint) =>
    set((state) => ({
      columnHints: state.columnHints.map((h, i) =>
        i === index ? { ...h, ...hint } : h
      )
    })),

  removeColumnHint: (index) =>
    set((state) => ({
      columnHints: state.columnHints.filter((_, i) => i !== index)
    })),

  clearColumnHints: () => set({ columnHints: [] }),

  setSelectedModel: (selectedModel) => set({ selectedModel }),

  setMaxRows: (maxRows) => set({ maxRows }),

  // Generation result
  setIsGenerating: (isGenerating) => set({ isGenerating }),

  setGeneratedTable: (generatedTable) => set({ generatedTable }),

  setGenerationError: (generationError) => set({ generationError }),

  setGenerationWarnings: (generationWarnings) => set({ generationWarnings }),

  // Reset wizard
  resetWizard: () =>
    set({
      ...initialWizardState,
      activeTab: "create"
    }),

  // ─────────────────────────────────────────────────────────────────────────
  // Export Actions
  // ─────────────────────────────────────────────────────────────────────────

  setIsExporting: (isExporting) => set({ isExporting }),
  setExportFormat: (exportFormat) => set({ exportFormat }),
  setExportError: (exportError) => set({ exportError }),

  // ─────────────────────────────────────────────────────────────────────────
  // UI Actions
  // ─────────────────────────────────────────────────────────────────────────

  setActiveTab: (activeTab) => set({ activeTab }),

  openTableDetail: (id) =>
    set({ tableDetailOpen: true, selectedTableId: id }),

  closeTableDetail: () =>
    set({ tableDetailOpen: false, selectedTableId: null, currentTable: null }),

  openDeleteConfirm: (id) =>
    set({ deleteConfirmOpen: true, deleteTargetId: id }),

  closeDeleteConfirm: () =>
    set({ deleteConfirmOpen: false, deleteTargetId: null }),

  resetStore: () => set(initialState),

  // ─────────────────────────────────────────────────────────────────────────
  // Editing Actions
  // ─────────────────────────────────────────────────────────────────────────

  startEditing: (table) => {
    // Convert rows to have stable _id for editing
    const editingRows: DataTableRow[] = (table.rows || []).map((row, index) => ({
      ...row,
      _id: `row-${index}-${Date.now()}`
    }))
    set({
      // structuredClone is safe here: DataTable shapes are JSON-origin (API
      // responses; timestamps are ISO strings, never Date objects).
      editingTable: structuredClone(table), // Deep clone
      originalTable: structuredClone(table),
      editingRows,
      editingState: {
        editingCellKey: null,
        isDirty: false,
        pendingChanges: []
      }
    })
  },

  stopEditing: () => {
    set({
      ...initialEditingState
    })
  },

  updateCell: (rowIndex, columnId, value) =>
    set((state) => {
      if (!state.editingTable) return state
      const newRows = [...state.editingRows]
      const oldValue = newRows[rowIndex]?.[columnId]
      if (newRows[rowIndex]) {
        newRows[rowIndex] = { ...newRows[rowIndex], [columnId]: value }
      }
      const change: TableChange = {
        type: "cell",
        rowIndex,
        columnId,
        oldValue,
        newValue: value,
        timestamp: Date.now()
      }
      return {
        editingRows: newRows,
        editingState: {
          ...state.editingState,
          isDirty: true,
          pendingChanges: [...state.editingState.pendingChanges, change]
        }
      }
    }),

  addRow: (afterIndex) =>
    set((state) => {
      if (!state.editingTable) return state
      const newRow: DataTableRow = {
        _id: `row-new-${Date.now()}`
      }
      // Initialize with empty values for each column
      state.editingTable.columns.forEach((col) => {
        newRow[col.name] = null
      })
      const newRows = [...state.editingRows]
      const insertIndex = afterIndex !== undefined ? afterIndex + 1 : newRows.length
      newRows.splice(insertIndex, 0, newRow)
      const change: TableChange = {
        type: "row_add",
        rowIndex: insertIndex,
        timestamp: Date.now()
      }
      return {
        editingRows: newRows,
        editingState: {
          ...state.editingState,
          isDirty: true,
          pendingChanges: [...state.editingState.pendingChanges, change]
        }
      }
    }),

  deleteRow: (rowIndex) =>
    set((state) => {
      if (!state.editingTable) return state
      const newRows = [...state.editingRows]
      const deletedRow = newRows[rowIndex]
      newRows.splice(rowIndex, 1)
      const change: TableChange = {
        type: "row_delete",
        rowIndex,
        oldValue: deletedRow,
        timestamp: Date.now()
      }
      return {
        editingRows: newRows,
        editingState: {
          ...state.editingState,
          isDirty: true,
          pendingChanges: [...state.editingState.pendingChanges, change]
        }
      }
    }),

  addColumn: (column, afterColumnId) =>
    set((state) => {
      if (!state.editingTable) return state
      const newColumns = [...state.editingTable.columns]
      let insertIndex = newColumns.length
      if (afterColumnId) {
        const afterIndex = newColumns.findIndex((c) => c.id === afterColumnId)
        if (afterIndex >= 0) insertIndex = afterIndex + 1
      }
      newColumns.splice(insertIndex, 0, column)
      // Add empty value for new column to all rows
      const newRows = state.editingRows.map((row) => ({
        ...row,
        [column.name]: null
      }))
      const change: TableChange = {
        type: "column_add",
        columnId: column.id,
        newValue: column,
        timestamp: Date.now()
      }
      return {
        editingTable: { ...state.editingTable, columns: newColumns },
        editingRows: newRows,
        editingState: {
          ...state.editingState,
          isDirty: true,
          pendingChanges: [...state.editingState.pendingChanges, change]
        }
      }
    }),

  deleteColumn: (columnId) =>
    set((state) => {
      if (!state.editingTable) return state
      const columnToDelete = state.editingTable.columns.find((c) => c.id === columnId)
      if (!columnToDelete) return state
      const newColumns = state.editingTable.columns.filter((c) => c.id !== columnId)
      // Remove column data from all rows
      const newRows = state.editingRows.map((row) => {
        const { [columnToDelete.name]: _, ...rest } = row
        return rest as DataTableRow
      })
      const change: TableChange = {
        type: "column_delete",
        columnId,
        oldValue: columnToDelete,
        timestamp: Date.now()
      }
      return {
        editingTable: { ...state.editingTable, columns: newColumns },
        editingRows: newRows,
        editingState: {
          ...state.editingState,
          isDirty: true,
          pendingChanges: [...state.editingState.pendingChanges, change]
        }
      }
    }),

  reorderColumns: (fromIndex, toIndex) =>
    set((state) => {
      if (!state.editingTable) return state
      const newColumns = [...state.editingTable.columns]
      const [movedColumn] = newColumns.splice(fromIndex, 1)
      newColumns.splice(toIndex, 0, movedColumn)
      const change: TableChange = {
        type: "column_reorder",
        oldValue: fromIndex,
        newValue: toIndex,
        timestamp: Date.now()
      }
      return {
        editingTable: { ...state.editingTable, columns: newColumns },
        editingState: {
          ...state.editingState,
          isDirty: true,
          pendingChanges: [...state.editingState.pendingChanges, change]
        }
      }
    }),

  setEditingCellKey: (key) =>
    set((state) => ({
      editingState: {
        ...state.editingState,
        editingCellKey: key
      }
    })),

  discardChanges: () =>
    set((state) => {
      if (!state.originalTable) return state
      const editingRows: DataTableRow[] = (state.originalTable.rows || []).map(
        (row, index) => ({
          ...row,
          _id: `row-${index}-${Date.now()}`
        })
      )
      return {
        // structuredClone is safe: originalTable was itself cloned from a
        // JSON-origin DataTable (see startEditing).
        editingTable: structuredClone(state.originalTable),
        editingRows,
        editingState: {
          editingCellKey: null,
          isDirty: false,
          pendingChanges: []
        }
      }
    }),

  getEditedTableData: () => {
    const state = useDataTablesStore.getState()
    if (!state.editingTable) return null
    // Remove _id from rows before returning
    const rows = state.editingRows.map((row) => {
      const { _id, ...rest } = row
      return rest
    })
    return {
      columns: state.editingTable.columns,
      rows
    }
  }
}))

// Expose for debugging
if (typeof window !== "undefined") {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ;(window as any).__tldw_useDataTablesStore = useDataTablesStore
}
