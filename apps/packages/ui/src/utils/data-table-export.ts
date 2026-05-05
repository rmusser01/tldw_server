/**
 * Data Table Export Utilities
 * Client-side export functions for CSV, JSON, and Excel formats
 */

import type { DataTable, DataTableColumn, ExportFormat } from "@/types/data-tables"
import { downloadBlob } from "@/utils/download-blob"

/**
 * Export table data to CSV format
 */
export function exportToCSV(table: DataTable): Blob {
  const columns = table.columns
  const rows = table.rows

  // Build header row
  const headerRow = columns.map((col) => escapeCSVField(col.name)).join(",")

  // Build data rows
  const dataRows = rows.map((row) => {
    return columns
      .map((col) => {
        const value = row[col.name] ?? row[col.id] ?? ""
        return escapeCSVField(formatCellValue(value, col))
      })
      .join(",")
  })

  const csvContent = [headerRow, ...dataRows].join("\n")
  return new Blob([csvContent], { type: "text/csv;charset=utf-8" })
}

/**
 * Export table data to JSON format
 */
export function exportDataTableToJSON(table: DataTable): Blob {
  const exportData = {
    name: table.name,
    description: table.description,
    columns: table.columns.map((col) => ({
      name: col.name,
      type: col.type,
      description: col.description,
      format: col.format
    })),
    rows: table.rows,
    metadata: {
      row_count: table.row_count,
      generation_model: table.generation_model,
      created_at: table.created_at,
      updated_at: table.updated_at,
      exported_at: new Date().toISOString()
    },
    sources: table.sources.map((source) => ({
      type: source.type,
      id: source.id,
      title: source.title
    }))
  }

  const jsonContent = JSON.stringify(exportData, null, 2)
  return new Blob([jsonContent], { type: "application/json;charset=utf-8" })
}

/**
 * Export table data to Excel format using ExcelJS
 */
export async function exportToExcel(table: DataTable): Promise<Blob> {
  // Dynamic import to avoid bundling exceljs if not used
  const ExcelJS = await import("exceljs")

  const columns = table.columns
  const rows = table.rows

  const workbook = new ExcelJS.Workbook()
  const sheetName = table.name.slice(0, 31).replace(/[\\/*?[\]]/g, "_")
  const worksheet = workbook.addWorksheet(sheetName)

  // Set up columns with headers and widths
  worksheet.columns = columns.map((col) => {
    let maxWidth = col.name.length
    rows.forEach((row) => {
      const value = row[col.name] ?? row[col.id] ?? ""
      const cellWidth = String(value).length
      if (cellWidth > maxWidth) maxWidth = cellWidth
    })
    return { header: col.name, key: col.name, width: Math.min(maxWidth + 2, 50) }
  })

  // Add data rows
  rows.forEach((row) => {
    const rowData: Record<string, any> = {}
    columns.forEach((col) => {
      rowData[col.name] = formatCellValueForExcel(row[col.name] ?? row[col.id] ?? "", col)
    })
    worksheet.addRow(rowData)
  })

  // Write to buffer
  const buffer = await workbook.xlsx.writeBuffer()
  return new Blob([new Uint8Array(buffer)], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
  })
}

/**
 * Helper to escape CSV fields
 */
function escapeCSVField(value: string): string {
  const stringValue = String(value)
  // If contains comma, newline, or quote, wrap in quotes and escape quotes
  if (
    stringValue.includes(",") ||
    stringValue.includes("\n") ||
    stringValue.includes('"')
  ) {
    return `"${stringValue.replace(/"/g, '""')}"`
  }
  return stringValue
}

/**
 * Format cell value based on column type for display
 */
function formatCellValue(value: any, column: DataTableColumn): string {
  if (value === null || value === undefined) return ""

  switch (column.type) {
    case "date":
      if (value instanceof Date) {
        return value.toISOString().split("T")[0]
      }
      if (typeof value === "string" && !isNaN(Date.parse(value))) {
        return new Date(value).toISOString().split("T")[0]
      }
      return String(value)

    case "number":
      if (typeof value === "number") {
        return value.toString()
      }
      return String(value)

    case "currency":
      if (typeof value === "number") {
        const format = column.format || "USD"
        try {
          return new Intl.NumberFormat("en-US", {
            style: "currency",
            currency: format
          }).format(value)
        } catch {
          return `${format} ${value.toFixed(2)}`
        }
      }
      return String(value)

    case "boolean":
      if (typeof value === "boolean") {
        return value ? "Yes" : "No"
      }
      return String(value)

    case "url":
    case "text":
    default:
      return String(value)
  }
}

/**
 * Format cell value for Excel (preserves types)
 */
function formatCellValueForExcel(value: any, column: DataTableColumn): any {
  if (value === null || value === undefined) return ""

  switch (column.type) {
    case "date":
      if (value instanceof Date) return value
      if (typeof value === "string" && !isNaN(Date.parse(value))) {
        return new Date(value)
      }
      return value

    case "number":
    case "currency":
      if (typeof value === "number") return value
      if (typeof value === "string") {
        const num = parseFloat(value)
        if (!isNaN(num)) return num
      }
      return value

    case "boolean":
      if (typeof value === "boolean") return value
      return value

    default:
      return value
  }
}

/**
 * Get file extension for export format
 */
export function getExportExtension(format: ExportFormat): string {
  switch (format) {
    case "csv":
      return ".csv"
    case "xlsx":
      return ".xlsx"
    case "json":
      return ".json"
    default:
      return ""
  }
}

/**
 * Get MIME type for export format
 */
export function getExportMimeType(format: ExportFormat): string {
  switch (format) {
    case "csv":
      return "text/csv"
    case "xlsx":
      return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    case "json":
      return "application/json"
    default:
      return "application/octet-stream"
  }
}

/**
 * Export table to specified format and trigger download
 */
export async function exportAndDownload(
  table: DataTable,
  format: ExportFormat
): Promise<void> {
  let blob: Blob

  switch (format) {
    case "csv":
      blob = exportToCSV(table)
      break
    case "xlsx":
      blob = await exportToExcel(table)
      break
    case "json":
      blob = exportDataTableToJSON(table)
      break
    default:
      throw new Error(`Unsupported export format: ${format}`)
  }

  const sanitizedName = table.name.replace(/[^a-zA-Z0-9_-]/g, "_")
  const filename = `${sanitizedName}${getExportExtension(format)}`
  downloadBlob(blob, filename)
}
