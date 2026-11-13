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
import { useDataTableGeneration } from "@/hooks/useDataTableGeneration"
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
  const generation = useDataTableGeneration()
  const isGenerating = generation.isGenerating
  const runGeneration = generation.run

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
  useEffect(() => {
    setIsGenerating(isGenerating)
    return () => setIsGenerating(false)
  }, [isGenerating, setIsGenerating])

  const generateTable = useCallback(() => {
    setGenerationError(null)
    setGenerationWarnings([])
    void runGeneration((options) => tldwClient.generateDataTable({
      name: tableName.trim() || "Untitled Table",
      prompt,
      sources: selectedSources.map(({ type, id, title, snippet }) => ({ type, id, title, snippet })),
      column_hints: columnHints.length > 0 ? columnHints : undefined,
      model: selectedModel || undefined,
      max_rows: maxRows
    }, options), Math.min(Math.max(maxRows, 1), 2000), {
      onComplete: (table) => {
        setGenerationError(null)
        setGeneratedTable(table)
        startEditing(table)
        message.success(t("dataTables:generateSuccess", "Table generated successfully!"))
      },
      onError: (error) => {
        const detail = error instanceof Error ? error.message : "Generation failed"
        setGenerationError(detail)
        message.error(detail)
      }
    })
  }, [columnHints, runGeneration, maxRows, prompt, selectedModel, selectedSources, setGeneratedTable, setGenerationError, setGenerationWarnings, startEditing, t, tableName])

  const cancelGeneration = async () => {
    try {
      await generation.cancel(() => setGenerationError(t("dataTables:generationCancelled", "Generation cancelled. Your saved table is unchanged.")))
    } catch (error) {
      message.error(error instanceof Error ? error.message : "Cancellation failed")
    }
  }

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
        <Button onClick={cancelGeneration} loading={generation.isCancelling} className="mt-4">
          {t("dataTables:cancelGeneration", "Cancel generation")}
        </Button>
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
