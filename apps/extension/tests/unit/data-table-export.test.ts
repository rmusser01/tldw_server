import { describe, expect, test } from "bun:test"
import type { DataTable } from "@tldw/ui/types/data-tables"
import {
  exportDataTableToJSON,
  exportToCSV,
  exportToExcel
} from "@tldw/ui/utils/data-table-export"

const makeSampleTable = (): DataTable => ({
  id: "table-1",
  name: "Sample Table",
  description: "Test table",
  prompt: "Generate a sample table",
  columns: [
    { id: "col-1", name: "Name", type: "text" },
    { id: "col-2", name: "Price", type: "number", format: "USD" }
  ],
  rows: [
    { Name: "Widget", Price: 10 },
    { Name: "Gadget", Price: 20 }
  ],
  sources: [],
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  row_count: 2
})

describe("data table export", () => {
  test("exports CSV with headers and rows", async () => {
    const table = makeSampleTable()
    const blob = exportToCSV(table)
    const text = await blob.text()
    expect(text).toBe("Name,Price\nWidget,10\nGadget,20")
  })

  test("exports JSON with metadata", async () => {
    const table = makeSampleTable()
    const blob = exportDataTableToJSON(table)
    const data = JSON.parse(await blob.text())
    expect(data.name).toBe("Sample Table")
    expect(data.columns).toHaveLength(2)
    expect(data.rows[0]).toEqual({ Name: "Widget", Price: 10 })
    expect(data.metadata.row_count).toBe(2)
  })

  test("exports Excel workbook with values", async () => {
    const table = makeSampleTable()
    const blob = await exportToExcel(table)
    expect(blob.size).toBeGreaterThan(0)

    const ExcelJS = await import("exceljs")
    const workbook = new ExcelJS.Workbook()
    const buffer = await blob.arrayBuffer()
    await workbook.xlsx.load(buffer)
    const worksheet = workbook.worksheets[0]

    expect(worksheet.getCell("A1").value).toBe("Name")
    expect(worksheet.getCell("A2").value).toBe("Widget")
    expect(worksheet.getCell("B2").value).toBe(10)
  })
})
