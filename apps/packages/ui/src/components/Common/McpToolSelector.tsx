import React from "react"
import { Button, Input, Switch } from "antd"

import { getDesignSystemState } from "@/design-system"
import type { ChatToolFilterCounts, ResolvedMcpTool } from "@/utils/chat-tools"

type TranslateFn = (
  key: string,
  fallback?: string,
  options?: Record<string, unknown>
) => string

export type McpToolSelectorProps = {
  discoveredTools: ResolvedMcpTool[]
  toolCounts: ChatToolFilterCounts
  toolsLoading?: boolean
  hasMcp?: boolean
  healthState?: string
  onToolEnabledChange: (toolName: string, enabled: boolean) => void
  onReset?: () => void
  t?: TranslateFn
  compact?: boolean
}

const defaultT: TranslateFn = (_key, fallback, options) => {
  const template = fallback ?? _key
  if (!options) return template
  return template.replace(/\{\{(\w+)\}\}/g, (_match, token) =>
    options[token] == null ? "" : String(options[token])
  )
}

const unavailableState = getDesignSystemState("unavailable")

const getToolStatus = (
  tool: ResolvedMcpTool,
  t: TranslateFn
): { label: string; muted?: boolean } => {
  if (!tool.canExecute) {
    return {
      label: t("mcpToolSelector.statusUnavailable", unavailableState.label),
      muted: true
    }
  }
  if (tool.colliding) {
    return {
      label: t("mcpToolSelector.statusConflict", "Name conflict"),
      muted: true
    }
  }
  if (tool.disabled) {
    return {
      label: t("mcpToolSelector.statusOff", "Off"),
      muted: true
    }
  }
  return {
    label: t("mcpToolSelector.statusOn", "On")
  }
}

export const McpToolSelector: React.FC<McpToolSelectorProps> = ({
  discoveredTools,
  toolCounts,
  toolsLoading = false,
  hasMcp = true,
  healthState = "healthy",
  onToolEnabledChange,
  onReset,
  t = defaultT,
  compact = false
}) => {
  const [query, setQuery] = React.useState("")
  const unavailableCount = Math.max(
    0,
    toolCounts.discovered - toolCounts.executable
  )
  const normalizedQuery = query.trim().toLowerCase()
  const filteredTools = React.useMemo(() => {
    if (!normalizedQuery) return discoveredTools
    return discoveredTools.filter((tool) => {
      const haystack = [
        tool.rawName,
        tool.chatName,
        tool.description,
        tool.groupLabel
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
      return haystack.includes(normalizedQuery)
    })
  }, [discoveredTools, normalizedQuery])

  const groupedTools = React.useMemo(() => {
    const groups = new Map<string, ResolvedMcpTool[]>()
    for (const tool of filteredTools) {
      const label = tool.groupLabel || "MCP"
      groups.set(label, [...(groups.get(label) ?? []), tool])
    }
    return [...groups.entries()].sort(([left], [right]) =>
      left.localeCompare(right)
    )
  }, [filteredTools])

  if (!hasMcp) {
    return (
      <div className="text-xs text-text-muted">
        {t("mcpToolSelector.unavailable", "MCP tools unavailable")}
      </div>
    )
  }

  if (healthState === "unhealthy") {
    return (
      <div className="text-xs text-text-muted">
        {t("mcpToolSelector.unhealthy", "MCP tools are offline")}
      </div>
    )
  }

  if (toolsLoading) {
    return (
      <div className="text-xs text-text-muted">
        {t("mcpToolSelector.loading", "Loading tools...")}
      </div>
    )
  }

  if (discoveredTools.length === 0) {
    return (
      <div className="text-xs text-text-muted">
        {t("mcpToolSelector.empty", "No MCP tools discovered")}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2" data-testid="mcp-tool-selector">
      <div className="flex flex-wrap items-center gap-1 text-[11px] text-text-muted">
        <span className="rounded border border-border px-1.5 py-0.5">
          {t("mcpToolSelector.countEnabled", "{{count}} enabled", {
            count: toolCounts.chatEnabled
          })}
        </span>
        <span className="rounded border border-border px-1.5 py-0.5">
          {t("mcpToolSelector.countDisabled", "{{count}} disabled", {
            count: toolCounts.disabled
          })}
        </span>
        <span className="rounded border border-border px-1.5 py-0.5">
          {t("mcpToolSelector.countUnavailable", "{{count}} unavailable", {
            count: unavailableCount
          })}
        </span>
        {toolCounts.colliding > 0 && (
          <span className="rounded border border-border px-1.5 py-0.5">
            {t("mcpToolSelector.countConflicts", "{{count}} conflicts", {
              count: toolCounts.colliding
            })}
          </span>
        )}
        {onReset && toolCounts.disabled > 0 && (
          <Button
            size="small"
            type="link"
            className="h-auto p-0 text-[11px]"
            onClick={onReset}
          >
            {t("mcpToolSelector.reset", "Reset")}
          </Button>
        )}
      </div>
      <Input
        size="small"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder={t("mcpToolSelector.searchPlaceholder", "Search tools")}
        aria-label={t("mcpToolSelector.searchAriaLabel", "Search MCP tools")}
      />
      <div className={compact ? "flex max-h-48 flex-col gap-2 overflow-auto" : "flex max-h-72 flex-col gap-3 overflow-auto"}>
        {groupedTools.map(([groupLabel, tools]) => (
          <div key={groupLabel} className="flex flex-col gap-1">
            <div className="text-[11px] font-medium uppercase text-text-muted">
              {groupLabel}
            </div>
            {tools.map((tool) => {
              const status = getToolStatus(tool, t)
              const switchDisabled = !tool.canExecute || tool.colliding
              return (
                <div
                  key={`${tool.chatName}-${tool.rawName}`}
                  className="flex items-start justify-between gap-3 rounded border border-border px-2 py-1.5"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm text-text">
                      {tool.displayName}
                    </div>
                    {tool.description && !compact && (
                      <div className="line-clamp-2 text-[11px] text-text-muted">
                        {tool.description}
                      </div>
                    )}
                    <div
                      className={
                        status.muted
                          ? "text-[11px] text-text-muted"
                          : "text-[11px] text-accent"
                      }
                    >
                      {status.label}
                    </div>
                  </div>
                  <Switch
                    size="small"
                    checked={!tool.disabled && tool.canExecute && !tool.colliding}
                    disabled={switchDisabled}
                    aria-label={t(
                      "mcpToolSelector.toggleAriaLabel",
                      "{{name}} MCP tool",
                      { name: tool.displayName }
                    )}
                    onChange={(checked) =>
                      onToolEnabledChange(tool.chatName, checked)
                    }
                  />
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}
