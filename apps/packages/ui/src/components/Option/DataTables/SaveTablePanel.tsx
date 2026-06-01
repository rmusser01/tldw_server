import React, { useState } from "react"
import { Button, Card, Input, message, Result } from "antd"
import { CheckCircle, Download, Save, Table2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useQueryClient } from "@tanstack/react-query"
import { useDataTablesStore } from "@/store/data-tables"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { exportAndDownload } from "@/utils/data-table-export"
import { Alert } from "@/components/ui/primitives"
import type { ExportFormat } from "@/types/data-tables"

const { TextArea } = Input

/**
 * SaveTablePanel
 *
 * Component for saving the generated table to the library and exporting.
 */
export const SaveTablePanel: React.FC = () => {
  const { t } = useTranslation(["dataTables", "common"])
  const queryClient = useQueryClient()

  // Store state
  const generatedTable = useDataTablesStore((s) => s.generatedTable)

  // Store actions
  const addTable = useDataTablesStore((s) => s.addTable)
  const setActiveTab = useDataTablesStore((s) => s.setActiveTab)
  const resetWizard = useDataTablesStore((s) => s.resetWizard)

  // Local state
  const [tableName, setTableName] = useState(generatedTable?.name || "")
  const [tableDescription, setTableDescription] = useState(
    generatedTable?.description || ""
  )
  const [isSaving, setIsSaving] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [saved, setSaved] = useState(false)

  // Handle save to library
  const handleSave = async () => {
    if (!generatedTable) return

    const name = tableName.trim() || "Untitled Table"
    const description = tableDescription.trim() || undefined
    const fallbackId = `local-${Date.now()}`
    const tableId = generatedTable.id ? String(generatedTable.id) : fallbackId
    const serverId =
      typeof generatedTable.id === "string" &&
      generatedTable.id &&
      !generatedTable.id.startsWith("preview-") &&
      !generatedTable.id.startsWith("artifact-") &&
      !generatedTable.id.startsWith("local-")
        ? generatedTable.id
        : undefined
    setIsSaving(true)

    try {
      // If table has an ID, update it; otherwise the server should have assigned one
      if (serverId) {
        await tldwClient.updateDataTable(serverId, {
          name,
          description
        })
      }

      // Add to local store
      addTable({
        id: tableId,
        name,
        description,
        row_count: generatedTable.row_count || generatedTable.rows?.length || 0,
        column_count: generatedTable.columns?.length || 0,
        created_at: generatedTable.created_at || new Date().toISOString(),
        updated_at: new Date().toISOString(),
        source_count: generatedTable.sources?.length || 0
      })

      await queryClient.invalidateQueries({ queryKey: ["dataTables"] })
      setSaved(true)
      message.success(t("dataTables:saveSuccess", "Table saved to library!"))
    } catch (error) {
      console.error("[SaveTablePanel] Save table failed", {
        error,
        tableId,
        serverId
      })
      const errorMessage = error instanceof Error ? error.message : "Failed to save table"
      message.error(errorMessage)
    } finally {
      setIsSaving(false)
    }
  }

  // Handle export
  const handleExport = async (format: ExportFormat) => {
    if (!generatedTable) return

    const tableId = generatedTable.id ? String(generatedTable.id) : "unknown"
    setIsExporting(true)
    try {
      // Update name before export
      const tableToExport = {
        ...generatedTable,
        name: tableName.trim() || generatedTable.name || "data-table",
        description: tableDescription.trim() || generatedTable.description
      }
      await exportAndDownload(tableToExport, format)
      message.success(
        t("dataTables:exportSuccess", "Table exported as {{format}}", { format: format.toUpperCase() })
      )
    } catch (error) {
      console.error("[SaveTablePanel] Export failed", { error, format, tableId })
      const errorMessage = error instanceof Error ? error.message : "Export failed"
      message.error(errorMessage)
    } finally {
      setIsExporting(false)
    }
  }

  // If no table, show error
  if (!generatedTable) {
    return (
      <Alert
        variant="warning"
        title={t("dataTables:noTableToSave", "No table to save")}
      >
        {t(
          "dataTables:goBackToGenerate",
          "Go back to the previous step to generate a table first."
        )}
      </Alert>
    )
  }

  // Success view
  if (saved) {
    return (
      <Result
        icon={<CheckCircle className="h-16 w-16 text-success mx-auto" />}
        title={t("dataTables:tableSaved", "Table Saved!")}
        subTitle={t(
          "dataTables:tableSavedDesc",
          "Your table has been saved to the library. You can view and export it anytime."
        )}
        extra={[
          <Button
            key="view"
            type="primary"
            icon={<Table2 className="h-4 w-4" />}
            onClick={() => setActiveTab("tables")}
          >
            {t("dataTables:viewTables", "View My Tables")}
          </Button>,
          <Button
            key="create"
            onClick={resetWizard}
          >
            {t("dataTables:createAnother", "Create Another")}
          </Button>
        ]}
      />
    )
  }

  return (
    <div className="space-y-6">
      {/* Table summary */}
      <Card className="bg-surface">
        <div className="flex items-start gap-4">
          <Table2 className="h-10 w-10 text-primary flex-shrink-0" />
          <div className="flex-1">
            <h4 className="font-medium text-text">
              {t("dataTables:tableSummary", "Table Summary")}
            </h4>
            <p className="text-sm text-text-muted mt-1">
              {generatedTable.row_count || generatedTable.rows?.length || 0}{" "}
              {t("dataTables:rows", "rows")} &bull;{" "}
              {generatedTable.columns?.length || 0}{" "}
              {t("dataTables:columnsLabel", "columns")} &bull;{" "}
              {generatedTable.sources?.length || 0}{" "}
              {t("dataTables:sources", "sources")}
            </p>
          </div>
        </div>
      </Card>

      {/* Name and description */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-text mb-2">
            {t("dataTables:tableName", "Table Name")}
          </label>
          <Input
            value={tableName}
            onChange={(e) => setTableName(e.target.value)}
            placeholder={t("dataTables:tableNamePlaceholder", "Enter a name for your table")}
            maxLength={100}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-text mb-2">
            {t("dataTables:tableDescription", "Description (optional)")}
          </label>
          <TextArea
            value={tableDescription}
            onChange={(e) => setTableDescription(e.target.value)}
            placeholder={t("dataTables:tableDescPlaceholder", "Add a description to help you remember what this table contains")}
            rows={2}
            maxLength={500}
          />
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-col gap-4">
        {/* Save button */}
        <Button
          type="primary"
          size="large"
          icon={<Save className="h-4 w-4" />}
          onClick={handleSave}
          loading={isSaving}
          block
        >
          {t("dataTables:saveToLibrary", "Save to Library")}
        </Button>

        {/* Export options */}
        <div>
          <p className="text-sm text-text-muted mb-2">
            {t("dataTables:orExport", "Or export directly:")}
          </p>
          <div className="flex gap-2">
            <Button
              icon={<Download className="h-4 w-4" />}
              onClick={() => handleExport("csv")}
              loading={isExporting}
            >
              CSV
            </Button>
            <Button
              icon={<Download className="h-4 w-4" />}
              onClick={() => handleExport("xlsx")}
              loading={isExporting}
            >
              Excel
            </Button>
            <Button
              icon={<Download className="h-4 w-4" />}
              onClick={() => handleExport("json")}
              loading={isExporting}
            >
              JSON
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
