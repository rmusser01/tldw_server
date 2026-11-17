import React, { useState } from "react"
import { Button, Dropdown, message, Tooltip } from "antd"
import type { MenuProps } from "antd"
import { Download, FileJson, FileSpreadsheet, FileText } from "lucide-react"
import { useTranslation } from "react-i18next"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { downloadBlob } from "@/utils/download-blob"
import type { ExportFormat } from "@/types/data-tables"

interface ExportMenuProps {
  tableId: string
  tableName: string
}

/**
 * ExportMenu
 *
 * Dropdown menu for exporting a data table in various formats.
 */
export const ExportMenu: React.FC<ExportMenuProps> = ({ tableId, tableName }) => {
  const { t } = useTranslation(["dataTables", "common"])
  const [isExporting, setIsExporting] = useState(false)

  // Handle export
  const handleExport = async (format: ExportFormat) => {
    setIsExporting(true)
    try {
      const { blob, filename } = await tldwClient.exportDataTable(tableId, format, async (requireAuth: boolean) => { await tldwClient.ensureConfigForRequest(requireAuth) })
      downloadBlob(blob, filename)
      message.success(
        t("dataTables:exportSuccess", "Table exported as {{format}}", {
          format: format.toUpperCase()
        })
      )
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Export failed"
      message.error(errorMessage)
    } finally {
      setIsExporting(false)
    }
  }

  // Menu items
  const menuItems: MenuProps["items"] = [
    {
      key: "csv",
      icon: <FileText className="h-4 w-4" />,
      label: t("dataTables:exportCSV", "Export as CSV"),
      onClick: () => handleExport("csv")
    },
    {
      key: "xlsx",
      icon: <FileSpreadsheet className="h-4 w-4" />,
      label: t("dataTables:exportExcel", "Export as Excel"),
      onClick: () => handleExport("xlsx")
    },
    {
      key: "json",
      icon: <FileJson className="h-4 w-4" />,
      label: t("dataTables:exportJSON", "Export as JSON"),
      onClick: () => handleExport("json")
    }
  ]

  return (
    <Dropdown menu={{ items: menuItems }} trigger={["click"]} disabled={isExporting}>
      <Tooltip title={t("dataTables:export", "Export")}>
        <Button
          type="text"
          size="small"
          aria-label={t("dataTables:export", "Export")}
          className="min-h-[44px] min-w-[44px] md:min-h-[32px] md:min-w-[32px]"
          icon={<Download className="h-4 w-4" />}
          loading={isExporting}
        />
      </Tooltip>
    </Dropdown>
  )
}
