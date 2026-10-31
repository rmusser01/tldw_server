import { Tooltip, ConfigProvider } from "antd"
import {
  CopyCheckIcon,
  CopyIcon,
  DownloadIcon,
  TableIcon,
  ExpandIcon
} from "lucide-react"
import { FC, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { IconButton } from "./IconButton"
import { useArtifactsStore, type ArtifactTableData } from "@/store/artifacts"
import { useUiModeStore } from "@/store/ui-mode"
import { useHorizontalOverflow } from "./useHorizontalOverflow"

interface TableProps {
  children: React.ReactNode
}

interface TableData {
  headers: string[]
  rows: string[][]
}

/** Header cells named in a table region's label; more are left out. */
const LABEL_COLUMN_LIMIT = 4

/** The table's first column headers, joined for its region label. */
function columnSummary(table: HTMLTableElement | null): string {
  if (!table) return ""
  const headers = Array.from(table.querySelectorAll("thead th"))
    .map((cell) => (cell.textContent || "").trim())
    .filter(Boolean)
  const shown = headers.slice(0, LABEL_COLUMN_LIMIT).join(", ")
  return headers.length > LABEL_COLUMN_LIMIT ? `${shown}, …` : shown
}

/** The table's caption when it has one, otherwise its columns. */
function tableRegionName(table: HTMLTableElement | null): string {
  const caption = table?.querySelector("caption")?.textContent?.replace(/\s+/g, " ").trim()
  return caption || columnSummary(table)
}

export const TableBlock: FC<TableProps> = ({ children }) => {
  const [copyStatus, setCopyStatus] = useState<string>("")
  const [artifactOriginId, setArtifactOriginId] = useState<string | null>(null)
  const { t } = useTranslation("common")
  const ref = useRef<HTMLTableElement>(null)
  const scrollRegionRef = useRef<HTMLDivElement>(null)
  const { hasMoreRight } = useHorizontalOverflow(scrollRegionRef)
  // Several tables in one answer need distinct region names (AX-10, XS-14),
  // so the label uses the caption, or else the columns once they render.
  const [tableLabel, setTableLabel] = useState("")
  useEffect(() => {
    const next = tableRegionName(ref.current)
    setTableLabel((current) => (current === next ? current : next))
  })
  const { openArtifact, isPinned } = useArtifactsStore()
  const uiMode = useUiModeStore((state) => state.mode)
  const isProMode = uiMode === "pro"

  const autoOpenMapRef = useRef<Map<string, boolean> | null>(null)
  if (!autoOpenMapRef.current) {
    if (typeof window !== "undefined") {
      const win = window as any
      if (!win.__tableArtifactAutoOpenState) {
        win.__tableArtifactAutoOpenState = new Map<string, boolean>()
      }
      autoOpenMapRef.current =
        win.__tableArtifactAutoOpenState as Map<string, boolean>
    } else {
      autoOpenMapRef.current = new Map()
    }
  }
  const autoOpenStateMap = autoOpenMapRef.current!

  const parseData = () => {
    // get table from ref
    const table = ref.current
    if (!table) return

    const headers: string[] = []
    const rows: string[][] = []

    const headerCells = table.querySelectorAll("thead th")
    headerCells.forEach((cell) => {
      headers.push(cell.textContent || "")
    })

    const bodyRows = table.querySelectorAll("tbody tr")
    bodyRows.forEach((row) => {
      const rowData: string[] = []
      const cells = row.querySelectorAll("td")
      cells.forEach((cell) => {
        rowData.push(cell.textContent || "")
      })
      rows.push(rowData)
    })

    return { headers, rows }
  }

  const convertToCSV = (tableData?: TableData) => {
    const data = tableData ?? parseData()
    if (!data) return

    const escapeCSV = (value: string): string => {
      if (value.includes(",") || value.includes('"') || value.includes("\n")) {
        return `"${value.replace(/"/g, '""')}"`
      }
      return value
    }

    const csvRows = []

    // Add headers
    if (data.headers.length > 0) {
      csvRows.push(data.headers.map(escapeCSV).join(","))
    }

    // Add data rows
    data.rows.forEach((row) => {
      csvRows.push(row.map(escapeCSV).join(","))
    })

    return csvRows.join("\n")
  }

  const handleCopyCSV = () => {
    const csvContent = convertToCSV()
    navigator.clipboard.writeText(csvContent)
    setCopyStatus("csv")
    setTimeout(() => setCopyStatus(""), 3000)
  }

  const handleDownloadCSV = () => {
    const csvContent = convertToCSV()
    // eslint-disable-next-line react-hooks/purity -- TASK-12116: this click handler timestamps the download when requested, never during render.
    downloadFile(csvContent, `table-${Date.now()}.csv`, "text/csv")
  }

  const buildArtifactId = (tableData: ArtifactTableData) => {
    const previewRow = tableData.rows[0]?.join("|") || ""
    const base = `${tableData.headers.join("|")}::${tableData.rows.length}::${previewRow}`
    let hash = 0
    for (let i = 0; i < base.length; i++) {
      hash = (hash * 31 + base.charCodeAt(i)) >>> 0
    }
    return `table-${hash.toString(36)}`
  }

  useEffect(() => {
    const tableData = parseData()
    if (!tableData) return
    const nextOriginId = `artifact-origin-${buildArtifactId(tableData)}`
    setArtifactOriginId((prev) => (prev === nextOriginId ? prev : nextOriginId))
  }, [children])

  const handleOpenArtifact = () => {
    const tableData = parseData()
    if (!tableData) return
    const csvContent = convertToCSV(tableData) || ""
    openArtifact({
      id: buildArtifactId(tableData),
      title: tableData.headers[0] || t("artifactsTitle", "Artifact"),
      content: csvContent,
      language: "csv",
      kind: "table",
      table: tableData
    })
  }

  useEffect(() => {
    if (!isProMode || isPinned) {
      return
    }
    const tableData = parseData()
    if (!tableData) {
      return
    }
    const artifactId = buildArtifactId(tableData)
    if (autoOpenStateMap.get(artifactId)) {
      return
    }
    const csvContent = convertToCSV(tableData) || ""
    openArtifact(
      {
        id: artifactId,
        title: tableData.headers[0] || t("artifactsTitle", "Artifact"),
        content: csvContent,
        language: "csv",
        kind: "table",
        table: tableData
      },
      { auto: true }
    )
    autoOpenStateMap.set(artifactId, true)
  }, [autoOpenStateMap, isPinned, isProMode, openArtifact, t])

  const downloadFile = (
    content: string,
    filename: string,
    mimeType: string
  ) => {
    const blob = new Blob([content], { type: mimeType })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }

  return (
    <div
      id={artifactOriginId || undefined}
      data-artifact-origin={
        artifactOriginId ? artifactOriginId.replace("artifact-origin-", "") : undefined
      }
      className="not-prose"
    >
      <div className="my-4 bg-surface rounded-xl border border-border overflow-hidden">
        <div className="flex flex-row px-4 py-2 rounded-t-xl bg-surface2 border-b border-border">
          <div className="flex min-w-0 items-center gap-2 flex-1">
            <TableIcon className="size-4 text-text-muted" />
            <span className="font-mono text-xs text-text-muted">
              Table
            </span>
            {hasMoreRight && (
              <span className="truncate text-[11px] text-text-muted">
                {t("table.scrollForMore", "Scroll for more columns →")}
              </span>
            )}
          </div>

          <div className="flex items-center gap-1">
            <Tooltip title={t("view", "View")}>
              <button
                type="button"
                onClick={handleOpenArtifact}
                className="inline-flex items-center gap-1 rounded-full border border-border bg-surface px-2 py-1 text-[11px] font-medium text-text-muted hover:text-text"
                aria-label={t("view", "View")}>
                <ExpandIcon className="size-3" />
                <span>{t("view", "View")}</span>
              </button>
            </Tooltip>
            <Tooltip title={t('table.copyCsv', 'Copy as CSV')}>
              <IconButton
                ariaLabel={t('table.copyCsv', 'Copy as CSV') as string}
                onClick={handleCopyCSV}
                className="flex gap-1.5 items-center rounded bg-none p-1 text-xs text-text-muted hover:bg-surface2 hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus transition-colors h-11 w-11 sm:h-7 sm:w-7 sm:min-w-0 sm:min-h-0">
                {copyStatus === "csv" ? (
                  <CopyCheckIcon className="size-4 text-success" />
                ) : (
                  <CopyIcon className="size-4" />
                )}
              </IconButton>
            </Tooltip>

            <ConfigProvider
              theme={{
                components: {
                  Dropdown: {
                    colorBgElevated: "var(--color-elevated)",
                    colorText: "var(--color-text)",
                    colorBgTextHover: "var(--color-surface-2)",
                    borderRadiusOuter: 8,
                    boxShadowSecondary:
                      "0 6px 16px 0 rgba(0, 0, 0, 0.08), 0 3px 6px -4px rgba(0, 0, 0, 0.12), 0 9px 28px 8px rgba(0, 0, 0, 0.05)"
                  }
                }
              }}>
              <Tooltip title={t('table.downloadCsv', 'Download CSV')}>
                <IconButton
                  ariaLabel={t('table.downloadCsv', 'Download CSV') as string}
                  onClick={handleDownloadCSV}
                  className="flex gap-1.5 items-center rounded bg-none p-1 text-xs text-text-muted hover:bg-surface2 hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus transition-colors h-11 w-11 sm:h-7 sm:w-7 sm:min-w-0 sm:min-h-0">
                  <DownloadIcon className="size-4" />
                </IconButton>
              </Tooltip>
            </ConfigProvider>
          </div>
        </div>

        <div className="relative">
          {/* A real <table> keeps row/column semantics for screen readers (AX-10);
              the focusable region lets keyboard users scroll wide tables. */}
          <div
            ref={scrollRegionRef}
            data-testid="table-scroll-region"
            role="region"
            aria-label={
              tableLabel
                ? t("table.labelWithColumns", {
                    defaultValue: "Table: {{columns}}",
                    columns: tableLabel
                  })
                : t("table.label", "Table")
            }
            tabIndex={0}
            className="overflow-x-auto rounded-b-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus">
            <div className="prose dark:prose-invert max-w-none">
              <table
                ref={ref}
                className="my-0 table-fixed text-sm w-full border-collapse [&_thead]:bg-surface2 [&_th]:px-6 [&_th]:py-4 [&_th]:text-left [&_th]:font-semibold [&_th]:text-text [&_th]:text-xs [&_th]:uppercase [&_th]:tracking-wider [&_th]:whitespace-nowrap [&_th:nth-child(1)]:w-1/2 [&_th:nth-child(2)]:w-1/2 [&_th:nth-child(3)]:w-1/3 [&_th]:border-b [&_th]:border-border [&_td]:px-6 [&_td]:py-4 [&_td]:text-text-muted [&_td]:text-sm [&_td]:text-left [&_td]:whitespace-nowrap [&_td]:border-b [&_td]:border-border [&_tr:last-child_td]:border-b-0">
                {children}
              </table>
            </div>
          </div>
          {hasMoreRight && (
            <div
              aria-hidden="true"
              data-testid="table-scroll-fade"
              className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-surface to-transparent"
            />
          )}
        </div>
      </div>
    </div>
  )
}
