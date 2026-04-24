import React from "react"

import type {
  MCPAuthType,
  MCPCatalogEntry,
  MCPConnectionDraft,
  MCPConnectionTestResult
} from "@/types/archetype"

export type MCPServerCatalogCardProps = {
  entry: MCPCatalogEntry
  isRecommended: boolean
  isConnected: boolean
  onConnect: (draft: MCPConnectionDraft) => void
  onTestConnection: (url: string, authType?: MCPAuthType, secret?: string) => void
  testResult?: MCPConnectionTestResult | null
  testLoading?: boolean
}

const validateUrl = (raw: string): string | null => {
  const trimmed = raw.trim()
  if (!trimmed) return null
  try {
    const parsed = new URL(trimmed)
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return "Enter a valid http or https URL."
    }
    return null
  } catch {
    return "Enter a valid http or https URL."
  }
}

export const MCPServerCatalogCard: React.FC<MCPServerCatalogCardProps> = ({
  entry,
  isRecommended,
  isConnected,
  onConnect,
  onTestConnection,
  testResult = null,
  testLoading = false
}) => {
  const [expanded, setExpanded] = React.useState(false)
  const [baseUrl, setBaseUrl] = React.useState(entry.url_template || "")
  const [authType, setAuthType] = React.useState<MCPAuthType>(entry.auth_type || "none")
  const [secret, setSecret] = React.useState("")

  const urlError = React.useMemo(() => validateUrl(baseUrl), [baseUrl])
  const trimmedSecret = secret.trim()
  const secretRequired = authType === "bearer" || authType === "api_key"
  const hasRequiredSecret = !secretRequired || trimmedSecret.length > 0

  const canConnect =
    baseUrl.trim().length > 0 &&
    urlError === null &&
    !isConnected &&
    hasRequiredSecret

  const handleConnect = React.useCallback(() => {
    if (!canConnect) return
    onConnect({
      serverKey: entry.key,
      name: entry.name,
      baseUrl: baseUrl.trim(),
      authType,
      secret: trimmedSecret
    })
  }, [authType, baseUrl, canConnect, entry.key, entry.name, onConnect, trimmedSecret])

  const handleTest = React.useCallback(() => {
    if (baseUrl.trim() && !urlError && hasRequiredSecret) {
      onTestConnection(baseUrl.trim(), authType, trimmedSecret)
    }
  }, [authType, baseUrl, hasRequiredSecret, onTestConnection, trimmedSecret, urlError])

  const handleAuthTypeChange = React.useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      const nextAuthType = event.target.value as MCPAuthType
      setAuthType(nextAuthType)
      if (nextAuthType === "none") {
        setSecret("")
      }
    },
    []
  )

  return (
    <div
      data-testid={`mcp-catalog-card-${entry.key}`}
      className="rounded-lg border border-border bg-surface2 p-3"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-text">{entry.name}</span>
            {isRecommended ? (
              <span className="inline-block rounded-full bg-blue-500/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-blue-300">
                Recommended
              </span>
            ) : null}
            {isConnected ? (
              <span className="inline-block rounded-full bg-green-500/20 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-green-300">
                Connected
              </span>
            ) : null}
          </div>
          <div className="mt-0.5 text-xs text-text-muted">
            {entry.description}
          </div>
          {entry.category ? (
            <div className="mt-1 text-[10px] uppercase tracking-wide text-text-subtle">
              {entry.category}
            </div>
          ) : null}
        </div>
        {!isConnected ? (
          <button
            type="button"
            className="shrink-0 rounded-md border border-border px-2 py-1 text-xs font-medium text-text hover:bg-surface"
            onClick={() => setExpanded((prev) => !prev)}
          >
            {expanded ? "Cancel" : "Connect"}
          </button>
        ) : null}
      </div>

      {expanded && !isConnected ? (
        <div className="mt-3 space-y-2 border-t border-border pt-3">
          <input
            type="text"
            value={baseUrl}
            aria-label="Server URL"
            placeholder="Server URL"
            className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
            onChange={(e) => setBaseUrl(e.target.value)}
          />
          {urlError ? (
            <div className="text-xs text-red-200">{urlError}</div>
          ) : null}

          <label className="block text-xs text-text-muted">
            Authentication
            <select
              aria-label="Authentication type"
              value={authType}
              className="mt-1 w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
              onChange={handleAuthTypeChange}
            >
              <option value="none">None</option>
              <option value="bearer">Bearer token</option>
              <option value="api_key">API key</option>
            </select>
          </label>

          {authType !== "none" ? (
            <input
              type="password"
              value={secret}
              aria-label="Secret"
              placeholder="Secret / token"
              className="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
              onChange={(e) => setSecret(e.target.value)}
            />
          ) : null}

          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={!canConnect}
              className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleConnect}
            >
              Save connection
            </button>
            <button
              type="button"
              disabled={!baseUrl.trim() || !!urlError || testLoading || !hasRequiredSecret}
              className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text disabled:cursor-not-allowed disabled:opacity-60"
              onClick={handleTest}
            >
              {testLoading ? "Testing..." : "Test connection"}
            </button>
          </div>

          {testResult ? (
            <div
              className={
                "rounded-md border px-3 py-2 text-xs " +
                (testResult.reachable
                  ? "border-green-500/40 bg-green-500/10 text-green-200"
                  : "border-red-500/40 bg-red-500/10 text-red-200")
              }
            >
              {testResult.reachable ? (
                <div>
                  Reachable.{" "}
                  {testResult.tools_discovered.length > 0
                    ? `Discovered ${testResult.tools_discovered.length} tool(s): ${testResult.tools_discovered.join(", ")}`
                    : "Tool discovery is not available for this connection test."}
                </div>
              ) : (
                <div>
                  Unreachable.{" "}
                  {testResult.error ? testResult.error : "Check the URL and try again."}
                </div>
              )}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
