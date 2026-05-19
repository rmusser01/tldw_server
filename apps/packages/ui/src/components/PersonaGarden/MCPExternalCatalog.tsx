import React from "react"

import { useMCPConnectionTest } from "@/hooks/useMCPConnectionTest"
import { useMCPServerCatalog } from "@/hooks/useMCPServerCatalog"
import type { MCPAuthType, MCPConnectionDraft } from "@/types/archetype"

import { MCPServerCatalogCard } from "./MCPServerCatalogCard"

type MCPExternalCatalogProps = {
  suggestedServers: string[]
  connectedServers: string[]
  onConnect: (draft: MCPConnectionDraft) => void
}

export const MCPExternalCatalog: React.FC<MCPExternalCatalogProps> = ({
  suggestedServers,
  connectedServers,
  onConnect
}) => {
  const { entries, loading, error } = useMCPServerCatalog()
  const {
    test: runTest,
    result: testResult,
    loading: testLoading
  } = useMCPConnectionTest()

  const [showCustom, setShowCustom] = React.useState(false)
  const [customName, setCustomName] = React.useState("")
  const [customUrl, setCustomUrl] = React.useState("")
  const [customAuth, setCustomAuth] = React.useState<MCPAuthType>("none")
  const [customSecret, setCustomSecret] = React.useState("")

  // Track which card is actively being tested so we show the result on the right card
  const [testingKey, setTestingKey] = React.useState<string | null>(null)

  const sorted = React.useMemo(() => {
    const suggested = entries.filter((e) => suggestedServers.includes(e.key))
    const rest = entries.filter((e) => !suggestedServers.includes(e.key))
    return [...suggested, ...rest]
  }, [entries, suggestedServers])

  const handleTestForCard = React.useCallback(
    (key: string) => (url: string, authType?: MCPAuthType, secret?: string) => {
      setTestingKey(key)
      void runTest(url, authType, secret)
    },
    [runTest]
  )

  const handleCustomConnect = React.useCallback(() => {
    const name = customName.trim()
    const url = customUrl.trim()
    if (!name || !url) return
    onConnect({
      serverKey: null,
      name,
      baseUrl: url,
      authType: customAuth,
      secret: customSecret.trim()
    })
    setShowCustom(false)
    setCustomName("")
    setCustomUrl("")
    setCustomAuth("none")
    setCustomSecret("")
  }, [customAuth, customName, customSecret, customUrl, onConnect])

  if (loading) {
    return (
      <div
        data-testid="mcp-catalog-loading"
        className="space-y-2"
      >
        {Array.from({ length: 3 }, (_, i) => (
          <div
            key={i}
            className="h-16 animate-pulse rounded-lg border border-border bg-surface2"
          />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
        {error}
      </div>
    )
  }

  return (
    <div data-testid="mcp-external-catalog" className="space-y-2">
      {sorted.length === 0 && !showCustom ? (
        <div className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text-muted">
          No MCP servers in catalog. Add a custom server below.
        </div>
      ) : (
        sorted.map((entry) => (
          <MCPServerCatalogCard
            key={entry.key}
            entry={entry}
            isRecommended={suggestedServers.includes(entry.key)}
            isConnected={connectedServers.includes(entry.key)}
            onConnect={onConnect}
            onTestConnection={handleTestForCard(entry.key)}
            testResult={testingKey === entry.key ? testResult : null}
            testLoading={testingKey === entry.key ? testLoading : false}
          />
        ))
      )}

      {showCustom ? (
        <div className="space-y-2 rounded-lg border border-border bg-surface2 p-3">
          <div className="text-xs font-semibold text-text">
            Custom MCP server
          </div>
          <input
            type="text"
            value={customName}
            aria-label="Server name"
            placeholder="Server name"
            className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
            onChange={(e) => setCustomName(e.target.value)}
          />
          <input
            type="text"
            value={customUrl}
            aria-label="Server URL"
            placeholder="Server URL (https://...)"
            className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
            onChange={(e) => setCustomUrl(e.target.value)}
          />
          <label className="block text-xs text-text-muted">
            Authentication
            <select
              aria-label="Authentication type"
              value={customAuth}
              className="mt-1 w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
              onChange={(e) => setCustomAuth(e.target.value)}
            >
              <option value="none">None</option>
              <option value="bearer">Bearer token</option>
              <option value="api_key">API key</option>
            </select>
          </label>
          {customAuth !== "none" ? (
            <input
              type="password"
              value={customSecret}
              aria-label="Secret"
              placeholder="Secret / token"
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
              onChange={(e) => setCustomSecret(e.target.value)}
            />
          ) : null}
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={!customName.trim() || !customUrl.trim()}
              className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleCustomConnect}
            >
              Save connection
            </button>
            <button
              type="button"
              className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text-muted"
              onClick={() => setShowCustom(false)}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button
          type="button"
          className="w-full rounded-lg border border-dashed border-border bg-surface2 px-3 py-2.5 text-sm text-text-muted hover:border-text-subtle hover:text-text"
          onClick={() => setShowCustom(true)}
        >
          + Add custom MCP server
        </button>
      )}
    </div>
  )
}
