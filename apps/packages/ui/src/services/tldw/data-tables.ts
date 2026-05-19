import type {
  ColumnType,
  DataTable,
  DataTableColumn,
  DataTableSource,
  DataTableSourceType,
  DataTableSummary
} from "@/types/data-tables"
import type { OffsetPaginationMeta } from "@/services/response-envelope"

export type ApiDataTableSource = {
  source_type: string
  source_id: string
  title?: string | null
  snapshot?: any
  retrieval_params?: any
}

export type ApiDataTableColumn = {
  column_id: string
  name: string
  type: string
  description?: string | null
  format?: string | null
  position?: number
}

export type ApiDataTableRow = {
  row_id: string
  row_index: number
  data: Record<string, any>
  row_hash?: string | null
}

export type ApiDataTableSummary = {
  uuid: string
  name: string
  description?: string | null
  prompt: string
  column_hints?: any
  status: string
  row_count: number
  generation_model?: string | null
  last_error?: string | null
  created_at?: string | null
  updated_at?: string | null
  last_modified?: string | null
  version?: number | null
  column_count?: number | null
  source_count?: number | null
}

export type ApiDataTableDetailResponse = {
  table: ApiDataTableSummary
  columns: ApiDataTableColumn[]
  rows: ApiDataTableRow[]
  sources: ApiDataTableSource[]
  rows_limit?: number
  rows_offset?: number
}

export type ApiDataTablesListResponse = {
  tables?: ApiDataTableSummary[]
  items?: ApiDataTableSummary[]
  results?: ApiDataTableSummary[]
  count?: number
  total?: number
  limit?: number
  offset?: number
  pagination?: Partial<OffsetPaginationMeta>
}

export type ApiDataTableGenerateResponse = {
  job_id: number
  job_uuid?: string | null
  status: string
  table: ApiDataTableSummary
}

export type ApiDataTableJobStatus = {
  id: number
  uuid?: string | null
  status: string
  job_type?: string | null
  owner_user_id?: string | null
  created_at?: string | null
  started_at?: string | null
  completed_at?: string | null
  cancelled_at?: string | null
  cancellation_reason?: string | null
  progress_percent?: number | null
  progress_message?: string | null
  result?: Record<string, any> | null
  error_message?: string | null
  table_uuid?: string | null
}

const toSafeSourceType = (value: string): DataTableSourceType => {
  if (value === "chat" || value === "document" || value === "rag_query") {
    return value
  }
  return "document"
}

export const mapApiSummaryToUi = (summary: ApiDataTableSummary): DataTableSummary => ({
  id: summary.uuid,
  name: summary.name,
  description: summary.description ?? undefined,
  row_count: summary.row_count ?? 0,
  column_count: summary.column_count ?? 0,
  created_at: summary.created_at || summary.updated_at || new Date().toISOString(),
  updated_at: summary.updated_at || summary.created_at || new Date().toISOString(),
  source_count: summary.source_count ?? 0
})

export const mapApiListToUi = (
  response: ApiDataTablesListResponse | ApiDataTableSummary[] | null | undefined
): {
  tables: DataTableSummary[]
  total: number
} => {
  if (!response) {
    return { tables: [], total: 0 }
  }
  const tablesList = Array.isArray(response)
    ? (response as ApiDataTableSummary[])
    : response.tables || response.items || response.results || []
  const tables = tablesList.map(mapApiSummaryToUi)
  const total = Array.isArray(response)
    ? tables.length
    : response.total ?? response.pagination?.total ?? response.count ?? tables.length
  return { tables, total }
}

export const mapApiDetailToUi = (detail: ApiDataTableDetailResponse): DataTable => {
  const table = detail.table
  const columns: DataTableColumn[] = (detail.columns || []).map((col) => ({
    id: col.column_id,
    name: col.name,
    type: col.type as ColumnType,
    description: col.description ?? undefined,
    format: col.format ?? undefined
  }))
  const columnIdToName = new Map(columns.map((col) => [col.id, col.name]))
  const rows = (detail.rows || []).map((row) => {
    const mapped: Record<string, any> = {}
    const data = row.data || {}
    for (const [key, value] of Object.entries(data)) {
      const name = columnIdToName.get(key) || key
      mapped[name] = value
    }
    return mapped
  })
  const sources: DataTableSource[] = (detail.sources || []).map((source) => ({
    type: toSafeSourceType(String(source.source_type || "")),
    id: String(source.source_id || ""),
    title: source.title || String(source.source_id || ""),
    snippet: typeof source.snapshot === "string" ? source.snapshot : undefined
  }))
  return {
    id: table.uuid,
    name: table.name,
    description: table.description ?? undefined,
    prompt: table.prompt,
    columns,
    rows,
    sources,
    created_at: table.created_at || table.updated_at || new Date().toISOString(),
    updated_at: table.updated_at || table.created_at || new Date().toISOString(),
    row_count: table.row_count ?? rows.length,
    generation_model: table.generation_model ?? undefined
  }
}

export const mapUiSourceToApi = (source: DataTableSource): ApiDataTableSource => ({
  source_type: source.type,
  source_id: source.id,
  title: source.title
})

export const buildContentPayload = (
  columns: DataTableColumn[],
  rows: Record<string, any>[]
): {
  columns: Array<{
    column_id: string
    name: string
    type: string
    description?: string
    format?: string
    position: number
  }>
  rows: Array<Record<string, any>>
} => {
  const mappedColumns = columns.map((col, index) => ({
    column_id: col.id,
    name: col.name,
    type: col.type,
    description: col.description,
    format: col.format,
    position: index
  }))
  const rowsPayload = rows.map((row) => {
    const rowJson: Record<string, any> = {}
    for (const column of mappedColumns) {
      rowJson[column.column_id] = row[column.name] ?? null
    }
    return rowJson
  })
  return { columns: mappedColumns, rows: rowsPayload }
}
