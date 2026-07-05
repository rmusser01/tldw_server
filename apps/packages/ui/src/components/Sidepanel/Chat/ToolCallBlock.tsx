/**
 * ToolCallBlock - Display tool/function calls in chat messages
 * Simplified version of ToolCallLog for inline display in chat bubbles
 */

import React from "react"
import { useTranslation } from "react-i18next"
import {
  FileText,
  Search,
  GitBranch,
  Terminal,
  Pencil,
  Trash2,
  FolderOpen,
  Globe,
  Wrench,
  ChevronDown,
  ChevronRight,
  ChefHat
} from "lucide-react"
import { classNames } from "@/libs/class-name"
import { RecipeCard } from "@/components/Common/RecipeCard/RecipeCard"
import type { ToolCall, ToolCallResult } from "@/types/tool-calls"
import { parseRecipeCardToolResult } from "@/utils/recipe-card-ui"

interface ToolCallBlockProps {
  toolCalls: ToolCall[]
  results?: ToolCallResult[]
  className?: string
}

// Map tool names to icons
const TOOL_ICONS: Record<string, React.FC<{ className?: string }>> = {
  fs_list: FolderOpen,
  fs_read: FileText,
  fs_write: Pencil,
  fs_apply_patch: Pencil,
  fs_mkdir: FolderOpen,
  fs_delete: Trash2,
  search_grep: Search,
  search_glob: Search,
  git_status: GitBranch,
  git_diff: GitBranch,
  git_log: GitBranch,
  git_branch: GitBranch,
  git_add: GitBranch,
  git_commit: GitBranch,
  exec_run: Terminal,
  web_search: Globe,
  web_fetch: Globe,
  "cooking.recipe_card.render": ChefHat
}

// Fallback display names
const TOOL_LABELS: Record<string, string> = {
  fs_list: "List Directory",
  fs_read: "Read File",
  fs_write: "Write File",
  fs_apply_patch: "Apply Patch",
  fs_mkdir: "Create Directory",
  fs_delete: "Delete",
  search_grep: "Search Content",
  search_glob: "Find Files",
  git_status: "Git Status",
  git_diff: "Git Diff",
  git_log: "Git Log",
  git_branch: "Git Branch",
  git_add: "Stage Files",
  git_commit: "Commit",
  exec_run: "Run Command",
  web_search: "Web Search",
  web_fetch: "Fetch URL",
  "cooking.recipe_card.render": "Recipe Card"
}

// Format tool arguments for compact display
const formatArgsPreview = (
  toolName: string,
  argsStr?: string | null
): string => {
  try {
    const args = JSON.parse(argsStr || "{}")

    switch (toolName) {
      case "fs_read":
      case "fs_write":
      case "fs_delete":
      case "fs_mkdir":
        return args.path || ""
      case "fs_list":
        return args.path || "."
      case "search_grep":
        return `"${args.pattern}"${args.glob ? ` (${args.glob})` : ""}`
      case "search_glob":
        return args.pattern || ""
      case "git_commit":
        return args.message
          ? `"${args.message.substring(0, 40)}${args.message.length > 40 ? "..." : ""}"`
          : ""
      case "exec_run":
        return args.command
          ? `${args.command.substring(0, 50)}${args.command.length > 50 ? "..." : ""}`
          : ""
      case "web_search":
        return args.query || ""
      case "web_fetch":
        return args.url || ""
      default:
        // Show first key-value for unknown tools
        const keys = Object.keys(args)
        if (keys.length > 0) {
          const val = args[keys[0]]
          if (typeof val === "string" && val.length < 50) {
            return val
          }
        }
        return ""
    }
  } catch {
    return ""
  }
}

// Format full arguments for expanded view
const formatFullArgs = (argsStr?: string | null): string => {
  try {
    const parsed = JSON.parse(argsStr)
    return JSON.stringify(parsed, null, 2)
  } catch {
    return argsStr || "{}"
  }
}

// Format result for display
const formatResult = (content: string): string => {
  try {
    const parsed = JSON.parse(content)
    if (typeof parsed === "object" && parsed !== null) {
      // Check for common result patterns
      if (parsed.ok === false && parsed.error) {
        return `Error: ${parsed.error}`
      }
      if (parsed.data) {
        if (Array.isArray(parsed.data.entries)) {
          return `${parsed.data.entries.length} entries`
        }
        if (Array.isArray(parsed.data.matches)) {
          return `${parsed.data.matches.length} matches`
        }
        if (typeof parsed.data.bytes === "number") {
          return `${parsed.data.bytes} bytes`
        }
      }
      return JSON.stringify(parsed, null, 2)
    }
    return content
  } catch {
    // Not JSON, return as-is (truncated if too long)
    return content.length > 200 ? content.substring(0, 200) + "..." : content
  }
}

export const ToolCallBlock: React.FC<ToolCallBlockProps> = ({
  toolCalls,
  results = [],
  className
}) => {
  const { t } = useTranslation(["common"])
  const [expandedIds, setExpandedIds] = React.useState<Set<string>>(new Set())

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const getToolDisplayName = (name: string): string => {
    return t(`common:tools.${name}`, {
      defaultValue: TOOL_LABELS[name] || name.replace(/_/g, " ")
    })
  }

  // Create a map of results by tool_call_id
  const resultsMap = React.useMemo(() => {
    const map = new Map<string, ToolCallResult>()
    results.forEach((r) => map.set(r.tool_call_id, r))
    return map
  }, [results])

  if (toolCalls.length === 0) {
    return null
  }

  return (
    <div
      className={classNames(
        "mt-2 space-y-1 border-t border-border/50 pt-2",
        className
      )}
      data-testid="tool-call-block"
    >
      <div className="text-xs text-text-muted mb-1">
        {t("common:toolCalls", "Tool Calls")} ({toolCalls.length})
      </div>

      {toolCalls.map((call) => {
        const Icon = TOOL_ICONS[call.function.name] || Wrench
        const isExpanded = expandedIds.has(call.id)
        const argsPreview = formatArgsPreview(call.function.name, call.function.arguments)
        const result = resultsMap.get(call.id)
        const recipePayload =
          call.function.name === "cooking.recipe_card.render"
            ? parseRecipeCardToolResult(result)
            : null

        return (
          <div
            key={call.id}
            className="rounded-md border border-border/60 bg-surface2/50"
          >
            {/* Header row */}
            <button
              type="button"
              onClick={() => toggleExpand(call.id)}
              className="w-full flex items-center gap-2 px-2 py-1.5 text-left hover:bg-surface2 rounded-md transition-colors"
              aria-expanded={isExpanded}
              title={getToolDisplayName(call.function.name)}
            >
              {isExpanded ? (
                <ChevronDown className="size-3 text-text-subtle flex-shrink-0" />
              ) : (
                <ChevronRight className="size-3 text-text-subtle flex-shrink-0" />
              )}

              <Icon className="size-3.5 text-primary flex-shrink-0" />

              <span className="text-xs font-medium text-text truncate">
                {getToolDisplayName(call.function.name)}
              </span>

              {argsPreview && (
                <span className="text-xs text-text-subtle truncate max-w-[150px]">
                  {argsPreview}
                </span>
              )}

              {/* Result indicator */}
              {result && (
                <span
                  className={classNames(
                    "ml-auto text-xs px-1.5 py-0.5 rounded",
                    result.error
                      ? "bg-danger/10 text-danger"
                      : "bg-success/10 text-success"
                  )}
                >
                  {result.error ? t("common:error", "Error") : t("common:done", "Done")}
                </span>
              )}
            </button>

            {/* Expanded content */}
            {isExpanded && (
              <div className="border-t border-border/40 px-2 py-2 space-y-2">
                {/* Arguments */}
                <div>
                  <div className="text-[10px] uppercase tracking-wide text-text-muted mb-1">
                    {t("common:arguments", "Arguments")}
                  </div>
                  <pre className="text-xs bg-surface rounded p-2 overflow-x-auto max-h-32">
                    {formatFullArgs(call.function.arguments)}
                  </pre>
                </div>

                {/* Result */}
                {result && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wide text-text-muted mb-1">
                      {t("common:result", "Result")}
                    </div>
                    {recipePayload ? (
                      <div className="max-h-96 overflow-y-auto overflow-x-hidden">
                        <RecipeCard payload={recipePayload} />
                      </div>
                    ) : (
                      <pre
                        className={classNames(
                          "text-xs rounded p-2 overflow-x-auto max-h-48",
                          result.error
                            ? "bg-danger/10 text-danger"
                            : "bg-surface"
                        )}
                      >
                        {formatResult(result.content)}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export default ToolCallBlock
