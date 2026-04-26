import React from "react"

import { apiSend } from "@/services/api-send"
import type { MCPAuthType, MCPConnectionTestResult } from "@/types/archetype"

type UseMCPConnectionTestResult = {
  test: (url: string, authType?: MCPAuthType, secret?: string) => Promise<void>
  result: MCPConnectionTestResult | null
  loading: boolean
  error: string | null
}

/**
 * Provides a `test` function that calls
 * `POST /api/v1/mcp/catalog/test-connection` to verify connectivity
 * to an MCP server endpoint.
 */
export function useMCPConnectionTest(): UseMCPConnectionTestResult {
  const [result, setResult] = React.useState<MCPConnectionTestResult | null>(
    null
  )
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const test = React.useCallback(
    async (url: string, authType?: MCPAuthType, secret?: string) => {
      setLoading(true)
      setError(null)
      setResult(null)
      try {
        const body: Record<string, unknown> = { url }
        if (authType) body.auth_type = authType
        if (secret) body.secret = secret

        const path = "/api/v1/mcp/catalog/test-connection" as const
        const res = await apiSend<MCPConnectionTestResult>({
          path,
          method: "POST",
          body
        })
        if (res.ok && res.data && typeof res.data === "object") {
          setResult(res.data)
        } else {
          setError(res.error ?? "Connection test failed")
        }
      } catch {
        setError("Connection test failed")
      } finally {
        setLoading(false)
      }
    },
    []
  )

  return { test, result, loading, error }
}
