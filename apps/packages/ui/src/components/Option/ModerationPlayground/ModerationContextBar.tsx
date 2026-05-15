import React from "react"
import { Tooltip } from "antd"
import { RefreshCw, Zap } from "lucide-react"
import { PolicyStatusBadges } from "./components/PolicyStatusBadges"
import { QuickTestInline } from "./components/QuickTestInline"
import type { ModerationScope } from "./moderation-utils"
import type { ModerationTestResponse } from "@/services/moderation"

interface ModerationContextBarProps {
  scope: ModerationScope
  onScopeChange: (scope: ModerationScope) => void
  userIdDraft: string
  onUserIdDraftChange: (value: string) => void
  onLoadUser: () => void
  activeUserId: string | null
  onClearUser: () => void
  userLoading: boolean
  policy: Record<string, any>
  hasUnsavedChanges: boolean
  onReload: () => void
  onRunQuickTest: (text: string, phase: "input" | "output") => Promise<ModerationTestResponse | undefined>
  onOpenTestTab: () => void
}

export const ModerationContextBar: React.FC<ModerationContextBarProps> = ({
  scope,
  onScopeChange,
  userIdDraft,
  onUserIdDraftChange,
  onLoadUser,
  activeUserId,
  onClearUser,
  userLoading,
  policy,
  hasUnsavedChanges,
  onReload,
  onRunQuickTest,
  onOpenTestTab
}) => {
  const [quickTestOpen, setQuickTestOpen] = React.useState(false)
  const scopeId = React.useId()
  const userIdInputId = React.useId()
  const quickTestButtonRef = React.useRef<HTMLButtonElement>(null)

  React.useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "t") {
        e.preventDefault()
        setQuickTestOpen((prev) => !prev)
      }
    }
    window.addEventListener("keydown", handleKey)
    return () => window.removeEventListener("keydown", handleKey)
  }, [])

  const closeQuickTest = React.useCallback(() => {
    setQuickTestOpen(false)
    window.setTimeout(() => quickTestButtonRef.current?.focus(), 0)
  }, [])

  return (
    <>
      <div className="sticky top-0 z-10 border-b border-border bg-bg/95 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3 py-2.5 flex-wrap">
            {/* Scope selector */}
            <label htmlFor={scopeId} className="sr-only">
              Moderation scope
            </label>
            <select
              id={scopeId}
              value={scope}
              onChange={(e) => onScopeChange(e.target.value as ModerationScope)}
              className="px-2 py-1 text-sm border border-border rounded bg-bg text-text"
            >
              <option value="server">Server (Global)</option>
              <option value="user">User (Individual)</option>
            </select>

            {/* User ID input */}
            {scope === "user" && !activeUserId && (
              <>
                <label htmlFor={userIdInputId} className="sr-only">
                  User ID
                </label>
                <input
                  id={userIdInputId}
                  type="text"
                  placeholder="Enter User ID"
                  value={userIdDraft}
                  onChange={(e) => onUserIdDraftChange(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && onLoadUser()}
                  className="px-2 py-1 text-sm border border-border rounded bg-bg text-text placeholder:text-text-muted w-40 sm:w-52"
                />
                <button
                  type="button"
                  onClick={onLoadUser}
                  disabled={userLoading}
                  className="px-2 py-1 text-sm border border-border rounded hover:bg-surface disabled:opacity-50"
                >
                  Load
                </button>
              </>
            )}

            {/* Active user badge */}
            {activeUserId && (
              <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-sm font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300">
                Configuring: {activeUserId}
                <button
                  type="button"
                  onClick={onClearUser}
                  className="hover:text-blue-600 ml-0.5"
                  aria-label={`Clear active user ${activeUserId}`}
                >
                  &times;
                </button>
              </span>
            )}

            {/* Spacer */}
            <div className="flex-1" />

            {/* Status badges — hidden on mobile */}
            <div className="hidden sm:block">
              <PolicyStatusBadges
                enabled={policy.enabled}
                inputAction={policy.input_action}
                outputAction={policy.output_action}
                ruleCount={policy.blocklist_count ?? 0}
                compact
              />
            </div>

            {/* Unsaved indicator */}
            {hasUnsavedChanges && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300">
                <span className="w-1.5 h-1.5 rounded-full bg-orange-500" />
                Unsaved
              </span>
            )}

            {/* Quick test toggle */}
            <Tooltip title="Quick Test (Ctrl+T)">
              <button
                ref={quickTestButtonRef}
                type="button"
                onClick={() => setQuickTestOpen((prev) => !prev)}
                aria-label="Toggle quick test"
                aria-expanded={quickTestOpen}
                className={`p-1.5 rounded hover:bg-surface ${quickTestOpen ? "bg-surface" : ""}`}
              >
                <Zap className="h-4 w-4 text-text-muted" />
              </button>
            </Tooltip>

            {/* Reload */}
            <Tooltip title="Reload config from disk">
              <button
                type="button"
                onClick={onReload}
                className="p-1.5 rounded hover:bg-surface"
                aria-label="Reload moderation config"
              >
                <RefreshCw className="h-4 w-4 text-text-muted" />
              </button>
            </Tooltip>
          </div>
        </div>

        {/* Mobile status badges */}
        <div className="sm:hidden border-t border-border px-4 py-1.5">
          <PolicyStatusBadges
            enabled={policy.enabled}
            inputAction={policy.input_action}
            outputAction={policy.output_action}
            ruleCount={policy.blocklist_count ?? 0}
            compact
          />
        </div>
      </div>

      {/* Quick test slide-down */}
      <QuickTestInline
        open={quickTestOpen}
        onClose={closeQuickTest}
        onRunTest={onRunQuickTest}
        onOpenFull={onOpenTestTab}
        userId={activeUserId || undefined}
      />
    </>
  )
}
