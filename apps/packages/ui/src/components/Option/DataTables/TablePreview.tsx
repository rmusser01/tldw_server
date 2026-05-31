import React, { Suspense, useCallback, useEffect, useMemo, useState } from "react"
import { DragDropProvider, type DragDropEvents } from "@dnd-kit/react"
import { useSortable } from "@dnd-kit/react/sortable"
import { closestCenter } from "@dnd-kit/collision"
import {
  Button,
  Empty,
  Popconfirm,
  Space,
  Spin,
  Table,
  Tooltip,
  message
} from "antd"
import type { ColumnsType } from "antd/es/table"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  GripVertical,
  Plus,
  PlusCircle,
  RefreshCw,
  RotateCcw,
  Sparkles,
  Trash2
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useDataTablesStore } from "@/store/data-tables"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { pollDataTableJob } from "@/utils/data-tables-jobs"
import type { DataTableColumn, DataTableRow } from "@/types/data-tables"
import { Alert } from "@/components/ui/primitives"
import { EditableCell } from "./EditableCell"
const AddColumnModal = React.lazy(() =>
  import("./AddColumnModal").then((module) => ({
    default: module.AddColumnModal
  }))
)

type DragEndEvent = Parameters<DragDropEvents["dragend"]>[0]

// Sortable column header for preview
const SortablePreviewHeader: React.FC<{
  column: DataTableColumn
  index: number
  onDelete: () => void
}> = ({ column, index, onDelete }) => {
  const { t } = useTranslation(["dataTables", "common"])
  const {
    ref,
    handleRef,
    isDragging
  } = useSortable({ id: column.id, index, collisionDetector: closestCenter })

  const style = {
    opacity: isDragging ? 0.5 : 1
  }

  return (
    <div
      ref={ref}
      style={style}
      className="flex items-center gap-1 group"
    >
      <span
        ref={handleRef}
        className="cursor-grab hover:text-primary opacity-0 group-hover:opacity-100 transition-opacity"
      >
        <GripVertical className="h-4 w-4" />
      </span>
      <div className="flex-1">
        <div className="font-medium">{column.name}</div>
        <div className="text-xs text-text-subtle">{column.type}</div>
      </div>
      <Popconfirm
        title={t("dataTables:deleteColumn", "Delete column?")}
        onConfirm={onDelete}
        okText={t("common:delete", "Delete")}
        cancelText={t("common:cancel", "Cancel")}
        okButtonProps={{ danger: true }}
      >
        <button className="opacity-0 group-hover:opacity-100 hover:text-danger transition-opacity p-1">
          <Trash2 className="h-3 w-3" />
        </button>
      </Popconfirm>
    </div>
  )
}

/**
 * TablePreview
 *
 * Component for previewing the generated table with editing capability.
 * Allows inline editing, adding/removing rows and columns before saving.
 */
export const TablePreview: React.FC = () => {
  const { t } = useTranslation(["dataTables", "common"])
  const queryClient = useQueryClient()

  // Store state
  const tableName = useDataTablesStore((s) => s.tableName)
  const prompt = useDataTablesStore((s) => s.prompt)
  const selectedSources = useDataTablesStore((s) => s.selectedSources)
  const columnHints = useDataTablesStore((s) => s.columnHints)
  const selectedModel = useDataTablesStore((s) => s.selectedModel)
  const maxRows = useDataTablesStore((s) => s.maxRows)
  const generatedTable = useDataTablesStore((s) => s.generatedTable)
  const generationError = useDataTablesStore((s) => s.generationError)
  const generationWarnings = useDataTablesStore((s) => s.generationWarnings)

  // Editing state from store
  const editingTable = useDataTablesStore((s) => s.editingTable)
  const editingRows = useDataTablesStore((s) => s.editingRows)
  const editingState = useDataTablesStore((s) => s.editingState)

  // Store actions
  const setIsGenerating = useDataTablesStore((s) => s.setIsGenerating)
  const setGeneratedTable = useDataTablesStore((s) => s.setGeneratedTable)
  const setGenerationError = useDataTablesStore((s) => s.setGenerationError)
  const setGenerationWarnings = useDataTablesStore((s) => s.setGenerationWarnings)

  // Editing actions
  const startEditing = useDataTablesStore((s) => s.startEditing)
  const stopEditing = useDataTablesStore((s) => s.stopEditing)
  const updateCell = useDataTablesStore((s) => s.updateCell)
  const addRow = useDataTablesStore((s) => s.addRow)
  const deleteRow = useDataTablesStore((s) => s.deleteRow)
  const addColumn = useDataTablesStore((s) => s.addColumn)
  const deleteColumn = useDataTablesStore((s) => s.deleteColumn)
  const reorderColumns = useDataTablesStore((s) => s.reorderColumns)
  const setEditingCellKey = useDataTablesStore((s) => s.setEditingCellKey)
  const discardChanges = useDataTablesStore((s) => s.discardChanges)

  // Local state
  const [addColumnModalOpen, setAddColumnModalOpen] = useState(false)
  const [generationJob, setGenerationJob] = useState<{
    jobId: number
    tableUuid: string
    rowsLimit: number
  } | null>(null)

  useEffect(() => {
    return () => {
      queryClient.cancelQueries({ queryKey: ["dataTableGeneration"] })
    }
  }, [queryClient])

  // Generate table
  const generateMutation = useMutation({
    mutationFn: async (payload: {
      name: string
      prompt: string
      sources: {
        type: string
        id: string
        title: string
        snippet?: string
      }[]
      columnHints?: Partial<DataTableColumn>[]
      model?: string
      maxRows: number
      rowsLimit: number
    }) => {
      const response = await tldwClient.generateDataTable({
        name: payload.name,
        prompt: payload.prompt,
        sources: payload.sources,
        column_hints: payload.columnHints,
        model: payload.model,
        max_rows: payload.maxRows
      })

      const jobId = response?.job_id
      const tableUuid = response?.table?.uuid
      if (!jobId || !tableUuid) {
        throw new Error("Generation job was not created")
      }

      return {
        jobId,
        tableUuid,
        rowsLimit: payload.rowsLimit
      }
    },
    onSuccess: (job) => {
      setGenerationJob(job)
    },
    onError: (error) => {
      console.error("Failed to start table generation:", error)
      const errorMessage =
        error instanceof Error ? error.message : "Generation failed"
      setGenerationError(errorMessage)
      message.error(errorMessage)
    }
  })

  const pollQuery = useQuery({
    queryKey: ["dataTableGeneration", generationJob?.jobId],
    enabled: Boolean(generationJob?.jobId),
    queryFn: async ({ signal }) => {
      if (!generationJob) {
        throw new Error("No generation job")
      }
      const jobStatus = await pollDataTableJob({
        jobId: generationJob.jobId,
        fetchJob: (id) => tldwClient.getDataTableJob(id),
        signal
      })
      const status = String(jobStatus?.status || "").toLowerCase()
      if (status !== "completed") {
        throw new Error(jobStatus?.error_message || "Generation failed")
      }

      const table = await tldwClient.getDataTable(
        jobStatus.table_uuid || generationJob.tableUuid,
        { rows_limit: generationJob.rowsLimit }
      )
      if (!table) {
        throw new Error("No table data in response")
      }

      return table
    },
    retry: false
  })

  const isGenerating = generateMutation.isPending || pollQuery.isFetching

  useEffect(() => {
    setIsGenerating(isGenerating)
  }, [isGenerating, setIsGenerating])

  useEffect(() => {
    if (!pollQuery.data) return
    const tableWithId = {
      ...pollQuery.data,
      id: pollQuery.data.id || `preview-${Date.now()}`
    }
    setGenerationError(null)
    setGeneratedTable(tableWithId)
    startEditing(tableWithId)
    setGenerationJob(null)
    message.success(t("dataTables:generateSuccess", "Table generated successfully!"))
  }, [
    pollQuery.data,
    setGeneratedTable,
    setGenerationError,
    setGenerationJob,
    startEditing,
    t
  ])

  useEffect(() => {
    if (!pollQuery.error) return
    if (pollQuery.error instanceof Error && pollQuery.error.message === "Polling cancelled") {
      return
    }
    console.error("Table generation failed:", pollQuery.error)
    const errorMessage =
      pollQuery.error instanceof Error ? pollQuery.error.message : "Generation failed"
    setGenerationError(errorMessage)
    message.error(errorMessage)
    setGenerationJob(null)
  }, [pollQuery.error, setGenerationError, setGenerationJob])

  const generateTable = useCallback(() => {
    const rowsLimit = Math.min(Math.max(maxRows, 1), 2000)
    void queryClient.cancelQueries({ queryKey: ["dataTableGeneration"] })
    setGenerationError(null)
    setGenerationWarnings([])
    setGeneratedTable(null)
    setGenerationJob(null)
    stopEditing() // Clear any previous editing state

    generateMutation.mutate({
      name: tableName.trim() || "Untitled Table",
      prompt,
      sources: selectedSources.map((s) => ({
        type: s.type,
        id: s.id,
        title: s.title,
        snippet: s.snippet
      })),
      columnHints: columnHints.length > 0 ? columnHints : undefined,
      model: selectedModel || undefined,
      maxRows,
      rowsLimit
    })
  }, [
    columnHints,
    generateMutation,
    maxRows,
    prompt,
    queryClient,
    selectedModel,
    selectedSources,
    setGeneratedTable,
    setGenerationError,
    setGenerationWarnings,
    setGenerationJob,
    stopEditing,
    tableName
  ])

  // Generate on mount if not already generated
  useEffect(() => {
    if (!generatedTable && !isGenerating && !generationError) {
      generateTable()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Sync editing state when generatedTable changes
  useEffect(() => {
    if (generatedTable && !editingTable) {
      const tableWithId = {
        ...generatedTable,
        id: generatedTable.id || `preview-${Date.now()}`
      }
      startEditing(tableWithId)
    }
  }, [generatedTable]) // eslint-disable-line react-hooks/exhaustive-deps

  // Update generatedTable when editing changes (keep in sync for save step)
  useEffect(() => {
    if (editingTable) {
      // Update the generated table with edited data
      const updatedTable = {
        ...editingTable,
        rows: editingRows.map(({ _id, ...rest }) => rest)
      }
      setGeneratedTable(updatedTable)
    }
  }, [editingRows, editingTable]) // eslint-disable-line react-hooks/exhaustive-deps

  // Handle drag end for column reordering
  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      if (event.canceled) return
      const sourceId = event.operation.source?.id
      const targetId = event.operation.target?.id
      if (!sourceId || !targetId || sourceId === targetId || !editingTable) return

      const oldIndex = editingTable.columns.findIndex((c) => c.id === sourceId)
      const newIndex = editingTable.columns.findIndex((c) => c.id === targetId)

      if (oldIndex !== -1 && newIndex !== -1) {
        reorderColumns(oldIndex, newIndex)
      }
    },
    [editingTable, reorderColumns]
  )

  // Handle add column
  const handleAddColumn = (column: DataTableColumn) => {
    addColumn(column)
    setAddColumnModalOpen(false)
    message.success(
      t("dataTables:columnAdded", "Column '{{name}}' added", { name: column.name })
    )
  }

  // Handle discard (reset to original generated data)
  const handleDiscard = () => {
    discardChanges()
    message.info(t("dataTables:changesDiscarded", "Changes discarded"))
  }

  // Build columns
  const columns = useMemo((): ColumnsType<DataTableRow> => {
    const tableColumns = editingTable?.columns || generatedTable?.columns || []
    const dataColumns: ColumnsType<DataTableRow> = tableColumns.map((col, index) => ({
      title: (
        <SortablePreviewHeader
          column={col}
          index={index}
          onDelete={() => deleteColumn(col.id)}
        />
      ),
      dataIndex: col.name,
      key: col.id,
      width: 150,
      ellipsis: true,
      render: (value: any, _: DataTableRow, rowIndex: number) => {
        const cellKey = `${rowIndex}-${col.id}`
        const isEditing = editingState.editingCellKey === cellKey

        return (
          <EditableCell
            value={value}
            columnType={col.type}
            columnName={col.name}
            rowIndex={rowIndex}
            isEditing={isEditing}
            onStartEdit={() => setEditingCellKey(cellKey)}
            onFinishEdit={(newValue) => {
              updateCell(rowIndex, col.name, newValue)
              setEditingCellKey(null)
            }}
            onCancelEdit={() => setEditingCellKey(null)}
          />
        )
      }
    }))

    // Add actions column
    dataColumns.push({
      title: "",
      key: "_actions",
      width: 50,
      fixed: "right",
      render: (_: any, __: DataTableRow, rowIndex: number) => (
        <Popconfirm
          title={t("dataTables:deleteRow", "Delete row?")}
          onConfirm={() => deleteRow(rowIndex)}
          okText={t("common:delete", "Delete")}
          cancelText={t("common:cancel", "Cancel")}
          okButtonProps={{ danger: true }}
        >
          <Tooltip title={t("dataTables:deleteRow", "Delete row")}>
            <button className="text-text-subtle hover:text-danger p-1">
              <Trash2 className="h-4 w-4" />
            </button>
          </Tooltip>
        </Popconfirm>
      )
    })

    return dataColumns
  }, [
    editingTable,
    generatedTable,
    editingState.editingCellKey,
    deleteColumn,
    deleteRow,
    updateCell,
    setEditingCellKey,
    t
  ])

  // Loading state
  if (isGenerating) {
    return (
      <div className="flex flex-col items-center justify-center py-16">
        <Spin size="large" />
        <p className="mt-4 text-text-muted">
          <span className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 animate-pulse" />
            {t("dataTables:generating", "Generating table from your sources...")}
          </span>
        </p>
        <p className="text-xs text-text-subtle mt-2">
          {t("dataTables:generatingNote", "This may take a moment depending on the amount of data.")}
        </p>
      </div>
    )
  }

  // Error state
  if (generationError) {
    return (
      <div className="space-y-4">
        <Alert
          variant="error"
          title={t("dataTables:generationFailed", "Generation Failed")}
        >
          {generationError}
        </Alert>
        <div className="flex justify-center">
          <Button
            type="primary"
            icon={<RefreshCw className="h-4 w-4" />}
            onClick={generateTable}
          >
            {t("dataTables:tryAgain", "Try Again")}
          </Button>
        </div>
      </div>
    )
  }

  // No table yet
  if (!generatedTable && !editingTable) {
    return (
      <Empty
        description={t("dataTables:noPreview", "No preview available")}
      />
    )
  }

  // Show editable table preview
  return (
    <div className="space-y-4">
      {/* Warnings */}
      {generationWarnings.length > 0 && (
        <Alert
          variant="warning"
          title={t("dataTables:warnings", "Warnings")}
        >
          <ul className="list-disc list-inside">
            {generationWarnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </Alert>
      )}

      {/* Toolbar */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <Space>
          <Button
            icon={<Plus className="h-4 w-4" />}
            onClick={() => addRow()}
            size="small"
          >
            {t("dataTables:addRow", "Add Row")}
          </Button>
          <Button
            icon={<PlusCircle className="h-4 w-4" />}
            onClick={() => setAddColumnModalOpen(true)}
            size="small"
          >
            {t("dataTables:addColumn", "Add Column")}
          </Button>
          {editingState.isDirty && (
            <Button
              icon={<RotateCcw className="h-4 w-4" />}
              onClick={handleDiscard}
              size="small"
            >
              {t("dataTables:discard", "Discard")}
            </Button>
          )}
        </Space>
        <Button
          icon={<RefreshCw className="h-4 w-4" />}
          onClick={generateTable}
          loading={isGenerating}
        >
          {t("dataTables:regenerate", "Regenerate")}
        </Button>
      </div>

      {/* Dirty indicator */}
      {editingState.isDirty && (
        <div className="text-xs text-warn flex items-center gap-1">
          <span className="w-2 h-2 bg-warn rounded-full animate-pulse" />
          {t("dataTables:previewModified", "You've modified the preview. Changes will be saved with the table.")}
        </div>
      )}

      {/* Table info */}
      <div className="text-sm text-text-muted">
        <span className="font-medium">{editingRows.length || generatedTable?.rows?.length || 0}</span>{" "}
        {t("dataTables:rows", "rows")} &bull;{" "}
        <span className="font-medium">{(editingTable?.columns || generatedTable?.columns)?.length || 0}</span>{" "}
        {t("dataTables:columnsLabel", "columns")}
      </div>

      {/* Editable table with DnD */}
      <DragDropProvider onDragEnd={handleDragEnd}>
        <Table
          dataSource={editingRows.length > 0 ? editingRows : generatedTable?.rows?.map((row, index) => ({
            ...row,
            _id: `row-${index}`
          }))}
          columns={columns}
          rowKey="_id"
          pagination={{
            pageSize: 10,
            showSizeChanger: false,
            showTotal: (total) => t("dataTables:showingRows", "Showing {{total}} rows", { total })
          }}
          scroll={{ x: true }}
          size="small"
        />
      </DragDropProvider>

      {/* Prompt used */}
      <div className="mt-4 p-3 bg-surface rounded-lg">
        <p className="text-xs font-medium text-text-muted mb-1">
          {t("dataTables:promptUsed", "Prompt used:")}
        </p>
        <p className="text-sm text-text italic">
          "{prompt}"
        </p>
      </div>

      {/* Add Column Modal */}
      <Suspense fallback={null}>
        <AddColumnModal
          open={addColumnModalOpen}
          onClose={() => setAddColumnModalOpen(false)}
          onAdd={handleAddColumn}
          existingColumns={editingTable?.columns || generatedTable?.columns || []}
        />
      </Suspense>
    </div>
  )
}
