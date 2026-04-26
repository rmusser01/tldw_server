import React from "react"

import type { ArchetypeTemplate, MCPConnectionDraft } from "@/types/archetype"

import { MCPAccessControlTier } from "./MCPAccessControlTier"
import { MCPExternalCatalog } from "./MCPExternalCatalog"
import { MCPModuleToggleGrid } from "./MCPModuleToggleGrid"

type ConfirmationMode = "always" | "destructive_only" | "never"

type ToolsConnectionsStepPayload = {
  enabledModules: string[]
  confirmationMode: string
  connections: MCPConnectionDraft[]
}

type ToolsConnectionsStepProps = {
  archetypeDefaults: ArchetypeTemplate | null
  onContinue: (payload: ToolsConnectionsStepPayload) => void
  saving: boolean
  error?: string | null
}

export const ToolsConnectionsStep: React.FC<ToolsConnectionsStepProps> = ({
  archetypeDefaults,
  onContinue,
  saving,
  error = null
}) => {
  const defaultEnabledModules = archetypeDefaults?.mcp_modules?.enabled ?? []
  const defaultConfirmationMode =
    archetypeDefaults?.policy?.confirmation_mode ?? "destructive_only"

  // Section A -- built-in module toggles
  const [enabledModules, setEnabledModules] = React.useState<string[]>(
    () => defaultEnabledModules
  )

  // Section B -- external connections
  const [connections, setConnections] = React.useState<MCPConnectionDraft[]>([])
  const connectedServerKeys = React.useMemo(
    () =>
      connections.flatMap((connection) =>
        connection.serverKey ? [connection.serverKey] : []
      ),
    [connections]
  )

  // Section C -- access control
  const [confirmationMode, setConfirmationMode] =
    React.useState<ConfirmationMode>(
      () => defaultConfirmationMode
    )

  React.useEffect(() => {
    setEnabledModules((prev) => {
      if (
        prev.length === defaultEnabledModules.length &&
        prev.every((moduleId, index) => moduleId === defaultEnabledModules[index])
      ) {
        return prev
      }
      return defaultEnabledModules
    })
    setConfirmationMode((prev) =>
      prev === defaultConfirmationMode ? prev : defaultConfirmationMode
    )
  }, [defaultConfirmationMode, defaultEnabledModules])

  const handleModuleToggle = React.useCallback(
    (moduleId: string, enabled: boolean) => {
      setEnabledModules((prev) =>
        enabled
          ? prev.includes(moduleId)
            ? prev
            : [...prev, moduleId]
          : prev.filter((id) => id !== moduleId)
      )
    },
    []
  )

  const handleAddConnection = React.useCallback((draft: MCPConnectionDraft) => {
    setConnections((prev) => [...prev, draft])
  }, [])

  const handleContinue = React.useCallback(() => {
    onContinue({
      enabledModules,
      confirmationMode,
      connections
    })
  }, [confirmationMode, connections, enabledModules, onContinue])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="text-sm font-semibold text-text">
          Tools and connections
        </div>
        <div className="text-xs text-text-muted">
          Enable built-in modules, connect external MCP servers, and set
          approval behavior for tool actions.
        </div>
      </div>

      {error ? (
        <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      ) : null}

      {/* Section A -- Built-in module toggles */}
      <div className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
          Built-in modules
        </div>
        <div className="text-xs text-text-muted">
          Toggle which internal MCP modules are available for this persona.
        </div>
        <MCPModuleToggleGrid
          enabledModules={enabledModules}
          onToggle={handleModuleToggle}
        />
      </div>

      {/* Section B -- External MCP catalog */}
      <div className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
          External MCP servers
        </div>
        <div className="text-xs text-text-muted">
          Connect to external MCP servers for additional capabilities. Servers
          recommended by your archetype are pinned to the top.
        </div>
        <MCPExternalCatalog
          suggestedServers={archetypeDefaults?.suggested_external_servers ?? []}
          connectedServers={connectedServerKeys}
          onConnect={handleAddConnection}
        />
        {connections.length > 0 ? (
          <div className="text-xs text-text-muted">
            {connections.length} connection(s) added:{" "}
            {connections.map((c) => c.name).join(", ")}
          </div>
        ) : null}
      </div>

      {/* Section C -- Access controls */}
      <div className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
          Action approval
        </div>
        <div className="text-xs text-text-muted">
          Choose how often the assistant should stop for approval before
          executing tool actions.
        </div>
        <MCPAccessControlTier
          serverId="global"
          mode={confirmationMode}
          onChange={setConfirmationMode}
        />
      </div>

      {/* Continue */}
      <button
        type="button"
        className="rounded-md border border-border px-3 py-2 text-sm font-medium text-text disabled:cursor-not-allowed disabled:opacity-60"
        disabled={saving}
        onClick={handleContinue}
      >
        {saving ? "Saving..." : "Continue"}
      </button>
    </div>
  )
}
