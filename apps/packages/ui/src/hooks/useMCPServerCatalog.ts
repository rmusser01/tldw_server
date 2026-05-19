import React from "react"

import { apiSend } from "@/services/api-send"
import type { MCPCatalogEntry } from "@/types/archetype"

type UseMCPServerCatalogResult = {
  entries: MCPCatalogEntry[]
  loading: boolean
  error: string | null
}

let cachedEntries: MCPCatalogEntry[] | null = null

/**
 * Fetches the MCP server catalog from `GET /api/v1/mcp/catalog`.
 * Results are cached in module scope so the request is made only once
 * across all consumers.
 */
export function useMCPServerCatalog(): UseMCPServerCatalogResult {
  const [entries, setEntries] = React.useState<MCPCatalogEntry[]>(
    cachedEntries ?? []
  )
  const [loading, setLoading] = React.useState(cachedEntries === null)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (cachedEntries !== null) {
      setEntries(cachedEntries)
      setLoading(false)
      return
    }

    let cancelled = false

    const fetchCatalog = async () => {
      setLoading(true)
      setError(null)
      try {
        const path = "/api/v1/mcp/catalog" as const
        const res = await apiSend<MCPCatalogEntry[]>({
          path,
          method: "GET"
        })
        if (cancelled) return
        if (res.ok && Array.isArray(res.data)) {
          cachedEntries = res.data
          setEntries(res.data)
        } else {
          setError(res.error ?? "Failed to load MCP server catalog")
        }
      } catch {
        if (!cancelled) {
          setError("Failed to load MCP server catalog")
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void fetchCatalog()

    return () => {
      cancelled = true
    }
  }, [])

  return { entries, loading, error }
}
