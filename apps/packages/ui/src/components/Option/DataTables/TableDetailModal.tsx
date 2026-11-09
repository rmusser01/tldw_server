import React, { useEffect, useState } from "react"
import {
  Button,
  Descriptions,
  Drawer,
  Empty,
  Spin,
  Switch,
  Tag,
  message
} from "antd"
import { Download, RefreshCw, X } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useQueryClient } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useDataTablesStore } from "@/store/data-tables"
import { exportAndDownload } from "@/utils/data-table-export"
import { useDataTableGeneration } from "@/hooks/useDataTableGeneration"
import { Alert } from "@/components/ui/primitives"
import type { DataTable, ExportFormat } from "@/types/data-tables"
import { EditableDataTable } from "./EditableDataTable"

interface TableDetailModalProps {
  open: boolean
  tableId: string | null
  onClose: () => void
}

/**
 * TableDetailModal
 *
 * Drawer modal for viewing and editing table details, data, and exporting.
 */
export const TableDetailModal: React.FC<TableDetailModalProps> = ({
  open,
  tableId,
  onClose
}) => {
  const { t } = useTranslation(["dataTables", "common"])
  const queryClient = useQueryClient()

  // Store state
  const currentTable = useDataTablesStore((s) => s.currentTable)
  const currentTableLoading = useDataTablesStore((s) => s.currentTableLoading)
  const editingState = useDataTablesStore((s) => s.editingState)

  // Store actions
  const setCurrentTable = useDataTablesStore((s) => s.setCurrentTable)
  const setCurrentTableLoading = useDataTablesStore((s) => s.setCurrentTableLoading)
  const stopEditing = useDataTablesStore((s) => s.stopEditing)
  const updateTableInList = useDataTablesStore((s) => s.updateTableInList)

  // Local state
  const [error, setError] = useState<string | null>(null)
  const [isExporting, setIsExporting] = useState(false)
  const generation = useDataTableGeneration(open && Boolean(tableId), tableId)
  const isRegenerating = generation.isGenerating
  const [editMode, setEditMode] = useState(false)

  // A closed drawer or a different table must not receive a late detail response.
  useEffect(() => {
    const controller = new AbortController()
    if (open && tableId) {
      setCurrentTableLoading(true)
      setError(null)
      setEditMode(false)
      void tldwClient.getDataTable(tableId, { rows_limit: 2000 }, { signal: controller.signal })
        .then((table) => {
          if (!controller.signal.aborted) setCurrentTable(table)
        })
        .catch((err) => {
          if (!controller.signal.aborted) setError(err instanceof Error ? err.message : "Failed to load table")
        })
        .finally(() => {
          if (!controller.signal.aborted) setCurrentTableLoading(false)
        })
    }
    return () => {
      controller.abort()
      setCurrentTable(null)
      setCurrentTableLoading(false)
      setError(null)
      stopEditing()
    }
  }, [open, tableId, setCurrentTable, setCurrentTableLoading, stopEditing])

  // Handle close with unsaved changes check
  const handleClose = () => {
    if (editMode && editingState.isDirty) {
      // Show confirmation
      if (!window.confirm(t("dataTables:discardUnsavedChanges", "You have unsaved changes. Are you sure you want to close?"))) {
        return
      }
    }
    stopEditing()
    onClose()
  }

  // Handle export
  const handleExport = async (format: ExportFormat) => {
    if (!currentTable) return

    setIsExporting(true)
    try {
      await exportAndDownload(currentTable, format)
      message.success(
        t("dataTables:exportSuccess", "Table exported as {{format}}", {
          format: format.toUpperCase()
        })
      )
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Export failed"
      message.error(errorMessage)
    } finally {
      setIsExporting(false)
    }
  }

  // Replace the visible table only after the captured job has completed.
  const handleRegenerate = () => {
    if (!tableId) return
    void generation.run((options) => tldwClient.regenerateDataTable(tableId, {}, options), 2000, {
      onComplete: (table) => {
        setCurrentTable(table)
        updateTableInList(tableId, {
          row_count: table.row_count,
          column_count: table.columns.length,
          updated_at: table.updated_at
        })
        void queryClient.invalidateQueries({ queryKey: ["dataTables"] })
        message.success(t("dataTables:regenerateSuccess", "Table regenerated successfully!"))
      },
      onError: (error) => message.error(error instanceof Error ? error.message : "Regeneration failed")
    })
  }

  const cancelGeneration = async () => {
    try {
      await generation.cancel(() => message.info(t("dataTables:generationCancelled", "Generation cancelled. Your saved table is unchanged.")))
    } catch (error) {
      message.error(error instanceof Error ? error.message : "Cancellation failed")
    }
  }

  // Handle save success
  const handleSaveSuccess = (updatedTable: DataTable) => {
    setCurrentTable(updatedTable)
    // Update in list
    if (updatedTable && tableId) {
      updateTableInList(tableId, {
        row_count: updatedTable.row_count || updatedTable.rows?.length || 0,
        column_count: updatedTable.columns?.length || 0,
        updated_at: updatedTable.updated_at
      })
    }
  }

  // Format date
  const formatDate = (dateStr?: string) => {
    if (!dateStr) return "-"
    try {
      return new Date(dateStr).toLocaleString()
    } catch {
      return dateStr
    }
  }

  return (
    <Drawer
      title={currentTable?.name || t("dataTables:tableDetails", "Table Details")}
      placement="right"
      size={900}
      open={open}
      onClose={handleClose}
      extra={
        <Button
          type="text"
          icon={<X className="h-4 w-4" />}
          onClick={handleClose}
          aria-label={t("common:close", "Close")}
        />
      }
    >
      {/* Loading */}
      {currentTableLoading && (
        <div className="flex justify-center py-12">
          <Spin size="large" />
        </div>
      )}

      {/* Error */}
      {error && (
        <Alert
          variant="error"
          title={t("common:error", "Error")}
        >
          {error}
        </Alert>
      )}

      {/* Content */}
      {!currentTableLoading && !error && currentTable && (
        <div className="space-y-6">
          {/* Actions */}
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex gap-2">
              <Button
                icon={<Download className="h-4 w-4" />}
                onClick={() => handleExport("csv")}
                loading={isExporting}
                size="small"
              >
                CSV
              </Button>
              <Button
                icon={<Download className="h-4 w-4" />}
                onClick={() => handleExport("xlsx")}
                loading={isExporting}
                size="small"
              >
                Excel
              </Button>
              <Button
                icon={<Download className="h-4 w-4" />}
                onClick={() => handleExport("json")}
                loading={isExporting}
                size="small"
              >
                JSON
              </Button>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-sm text-text-muted">
                  {t("dataTables:editMode", "Edit Mode")}
                </span>
                <Switch
                  checked={editMode}
                  onChange={setEditMode}
                  size="small"
                />
              </div>
              {isRegenerating && (
                <Button onClick={cancelGeneration} loading={generation.isCancelling} size="small">
                  {t("dataTables:cancelGeneration", "Cancel generation")}
                </Button>
              )}
              <Button
                icon={<RefreshCw className="h-4 w-4" />}
                onClick={handleRegenerate}
                loading={isRegenerating}
                size="small"
              >
                {t("dataTables:regenerate", "Regenerate")}
              </Button>
            </div>
          </div>

          {/* Metadata */}
          <Descriptions bordered column={2} size="small">
            <Descriptions.Item label={t("dataTables:rows", "Rows")}>
              {currentTable.row_count || currentTable.rows?.length || 0}
            </Descriptions.Item>
            <Descriptions.Item label={t("dataTables:columnsLabel", "Columns")}>
              {currentTable.columns?.length || 0}
            </Descriptions.Item>
            <Descriptions.Item label={t("dataTables:created", "Created")}>
              {formatDate(currentTable.created_at)}
            </Descriptions.Item>
            <Descriptions.Item label={t("dataTables:updated", "Updated")}>
              {formatDate(currentTable.updated_at)}
            </Descriptions.Item>
            {currentTable.generation_model && (
              <Descriptions.Item
                label={t("dataTables:model", "Model")}
                span={2}
              >
                {currentTable.generation_model}
              </Descriptions.Item>
            )}
          </Descriptions>

          {/* Description */}
          {currentTable.description && (
            <div>
              <h4 className="text-sm font-medium text-text mb-1">
                {t("dataTables:descriptionLabel", "Description")}
              </h4>
              <p className="text-sm text-text-muted">
                {currentTable.description}
              </p>
            </div>
          )}

          {/* Prompt */}
          {currentTable.prompt && (
            <div>
              <h4 className="text-sm font-medium text-text mb-1">
                {t("dataTables:prompt", "Generation Prompt")}
              </h4>
              <p className="text-sm text-text-muted italic bg-surface p-2 rounded">
                "{currentTable.prompt}"
              </p>
            </div>
          )}

          {/* Sources */}
          {currentTable.sources && currentTable.sources.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-text mb-2">
                {t("dataTables:sources", "Sources")} ({currentTable.sources.length})
              </h4>
              <div className="flex flex-wrap gap-2">
                {currentTable.sources.map((source, index) => (
                  <Tag key={index}>
                    {source.type}: {source.title}
                  </Tag>
                ))}
              </div>
            </div>
          )}

          {/* Data table */}
          <div>
            <h4 className="text-sm font-medium text-text mb-2">
              {t("dataTables:data", "Data")}
              {editMode && (
                <span className="ml-2 text-xs font-normal text-primary">
                  ({t("dataTables:editModeActive", "Click cells to edit")})
                </span>
              )}
            </h4>
            {currentTable.rows && currentTable.rows.length > 0 ? (
              <EditableDataTable
                table={currentTable}
                onSaveSuccess={handleSaveSuccess}
                readOnly={!editMode}
              />
            ) : (
              <Empty description={t("dataTables:noData", "No data")} />
            )}
          </div>
        </div>
      )}

      {/* No table */}
      {!currentTableLoading && !error && !currentTable && (
        <Empty description={t("dataTables:tableNotFound", "Table not found")} />
      )}
    </Drawer>
  )
}
