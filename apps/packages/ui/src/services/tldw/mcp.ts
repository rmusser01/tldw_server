import { bgRequestClient } from "@/services/background-proxy"
import type { ClientPathOrUrlWithQuery } from "@/services/tldw/openapi-guard"

export type McpToolTier = "read" | "write" | "exec" | string

export type McpToolDefinition = {
  name?: string
  description?: string | null
  parameters?: Record<string, unknown>
  input_schema?: Record<string, unknown>
  json_schema?: Record<string, unknown>
  tier?: McpToolTier
  canExecute?: boolean
  [key: string]: unknown
}

export type McpToolCatalog = {
  id: number
  name: string
  description?: string | null
  org_id?: number | null
  team_id?: number | null
  is_active?: boolean
  created_at?: string | null
  updated_at?: string | null
}

export type McpToolListParams = {
  catalog?: string
  catalogId?: number | null
  module?: string | string[]
  catalogStrict?: boolean
}

export const fetchMcpTools = async (
  params: McpToolListParams = {}
): Promise<McpToolDefinition[]> => {
  try {
    const query = new URLSearchParams()
    const catalog = typeof params.catalog === "string" ? params.catalog.trim() : ""
    if (catalog) query.set("catalog", catalog)
    const catalogId = typeof params.catalogId === "number" ? params.catalogId : null
    if (catalogId !== null && Number.isFinite(catalogId)) {
      query.set("catalog_id", String(catalogId))
    }
    const moduleValues = Array.isArray(params.module)
      ? params.module
      : typeof params.module === "string"
        ? [params.module]
        : []
    const moduleIds = moduleValues
      .map((value) => (typeof value === "string" ? value.trim() : ""))
      .filter((value) => value.length > 0)
    if (moduleIds.length > 0) {
      for (const moduleId of moduleIds) {
        query.append("module", moduleId)
      }
    }
    if (params.catalogStrict) query.set("catalog_strict", "1")
    const queryString = query.toString()
    const path = (queryString
      ? `/api/v1/mcp/tools?${queryString}`
      : "/api/v1/mcp/tools") as ClientPathOrUrlWithQuery
    const res = await bgRequestClient<any>({
      path,
      method: "GET"
    })
    if (!res) return []
    if (Array.isArray(res)) return res
    if (Array.isArray(res.tools)) return res.tools
    if (Array.isArray(res.data)) return res.data
    return []
  } catch {
    return []
  }
}

export const fetchMcpToolCatalogs = async (): Promise<McpToolCatalog[]> => {
  try {
    const res = await bgRequestClient<any>({
      path: "/api/v1/mcp/tool_catalogs",
      method: "GET"
    })
    if (!res) return []
    if (Array.isArray(res)) return res as McpToolCatalog[]
    if (Array.isArray(res.catalogs)) return res.catalogs as McpToolCatalog[]
    if (Array.isArray(res.data)) return res.data as McpToolCatalog[]
    return []
  } catch {
    return []
  }
}

export const executeMcpTool = async (
  payload: Record<string, unknown>
): Promise<Record<string, unknown>> => {
  return await bgRequestClient<Record<string, unknown>>({
    path: "/api/v1/mcp/tools/execute",
    method: "POST",
    body: payload
  })
}

const unwrapToolResult = (payload: Record<string, unknown> | null | undefined): unknown => {
  if (!payload || typeof payload !== "object") return null
  const value = "result" in payload ? payload.result : payload
  if (!value || typeof value !== "object") return value
  const result = value as Record<string, unknown>
  const content = result.content
  if (Array.isArray(content)) {
    const jsonContent = content.find(
      (entry) =>
        entry &&
        typeof entry === "object" &&
        "json" in entry &&
        (entry as Record<string, unknown>).json != null
    ) as Record<string, unknown> | undefined
    if (jsonContent) return jsonContent.json
  }
  return result
}

const normalizeCatalogList = (payload: unknown): McpToolCatalog[] => {
  if (!payload) return []
  if (Array.isArray(payload)) return payload as McpToolCatalog[]
  if (typeof payload !== "object") return []
  const data = payload as Record<string, unknown>
  if (Array.isArray(data.catalogs)) return data.catalogs as McpToolCatalog[]
  const catalogs = data.catalogs as Record<string, unknown> | undefined
  if (!catalogs || typeof catalogs !== "object") return []
  const merged: McpToolCatalog[] = []
  for (const entry of Object.values(catalogs)) {
    if (Array.isArray(entry)) {
      merged.push(...(entry as McpToolCatalog[]))
    }
  }
  return merged
}

const normalizeToolList = (payload: unknown): McpToolDefinition[] => {
  if (!payload) return []
  if (Array.isArray(payload)) return payload as McpToolDefinition[]
  if (typeof payload !== "object") return []
  const data = payload as Record<string, unknown>
  if (Array.isArray(data.tools)) return data.tools as McpToolDefinition[]
  if (Array.isArray(data.data)) return data.data as McpToolDefinition[]
  return []
}

export const fetchMcpToolCatalogsViaDiscovery = async (
  scope: "all" | "global" | "org" | "team" = "all"
): Promise<McpToolCatalog[]> => {
  const res = await executeMcpTool({
    tool_name: "mcp.catalogs.list",
    arguments: { scope }
  })
  const payload = unwrapToolResult(res)
  return normalizeCatalogList(payload)
}

export const fetchMcpModulesViaDiscovery = async (): Promise<string[]> => {
  const res = await executeMcpTool({
    tool_name: "mcp.modules.list",
    arguments: {}
  })
  const payload = unwrapToolResult(res)
  if (!payload || typeof payload !== "object") return []
  const modules = (payload as Record<string, unknown>).modules
  if (!Array.isArray(modules)) return []
  return modules
    .map((entry) =>
      typeof (entry as Record<string, unknown>)?.module_id === "string"
        ? String((entry as Record<string, unknown>).module_id)
        : null
    )
    .filter((value): value is string => Boolean(value && value.trim().length > 0))
}

export const fetchMcpToolsViaDiscovery = async (
  params: McpToolListParams = {}
): Promise<McpToolDefinition[]> => {
  const args: Record<string, unknown> = {}
  if (typeof params.catalog === "string" && params.catalog.trim()) {
    args.catalog = params.catalog.trim()
  }
  if (typeof params.catalogId === "number" && Number.isFinite(params.catalogId)) {
    args.catalog_id = params.catalogId
  }
  if (params.catalogStrict) {
    args.catalog_strict = true
  }
  const moduleValues = Array.isArray(params.module)
    ? params.module
    : typeof params.module === "string"
      ? [params.module]
      : []
  const moduleIds = moduleValues
    .map((value) => (typeof value === "string" ? value.trim() : ""))
    .filter((value) => value.length > 0)
  if (moduleIds.length === 1) {
    args.module = moduleIds[0]
  } else if (moduleIds.length > 1) {
    args.modules = moduleIds
  }
  const res = await executeMcpTool({
    tool_name: "mcp.tools.list",
    arguments: args
  })
  const payload = unwrapToolResult(res)
  return normalizeToolList(payload)
}
