import React, { useCallback, useMemo, useState } from "react"
import { Input, Modal } from "antd"
import {
  CalendarClock,
  FileOutput,
  FileText,
  Newspaper,
  Play,
  Plus,
  RefreshCw,
  Rss,
  Search,
  Settings
} from "lucide-react"
import { useTranslation } from "react-i18next"

export interface CommandPaletteCommand {
  id: string
  label: string
  icon: React.ReactNode
  category: "navigate" | "create" | "action"
  keywords?: string[]
  onExecute: () => void
}

interface WatchlistsCommandPaletteProps {
  open: boolean
  onClose: () => void
  commands: CommandPaletteCommand[]
}

const categoryOrder: Record<string, number> = {
  navigate: 0,
  create: 1,
  action: 2
}

const categoryLabels: Record<string, string> = {
  navigate: "Navigate",
  create: "Create",
  action: "Actions"
}

export const WatchlistsCommandPalette: React.FC<WatchlistsCommandPaletteProps> = ({
  open,
  onClose,
  commands
}) => {
  const { t } = useTranslation(["watchlists"])
  const [query, setQuery] = useState("")

  const filtered = useMemo(() => {
    if (!query.trim()) return commands
    const q = query.toLowerCase()
    return commands.filter(
      (cmd) =>
        cmd.label.toLowerCase().includes(q) ||
        cmd.id.toLowerCase().includes(q) ||
        cmd.keywords?.some((kw) => kw.toLowerCase().includes(q))
    )
  }, [commands, query])

  const grouped = useMemo(() => {
    const groups = new Map<string, CommandPaletteCommand[]>()
    for (const cmd of filtered) {
      const list = groups.get(cmd.category) || []
      list.push(cmd)
      groups.set(cmd.category, list)
    }
    return [...groups.entries()].sort(
      ([a], [b]) => (categoryOrder[a] ?? 99) - (categoryOrder[b] ?? 99)
    )
  }, [filtered])

  const handleSelect = useCallback(
    (cmd: CommandPaletteCommand) => {
      onClose()
      setQuery("")
      cmd.onExecute()
    },
    [onClose]
  )

  const handleAfterClose = useCallback(() => {
    setQuery("")
  }, [])

  return (
    <Modal
      open={open}
      onCancel={onClose}
      afterClose={handleAfterClose}
      footer={null}
      closable={false}
      width={480}
      styles={{ body: { padding: 0 } }}
      data-testid="watchlists-command-palette"
    >
      <div className="p-3 pb-0">
        <Input
          prefix={<Search className="h-4 w-4 text-text-muted" />}
          placeholder={t(
            "watchlists:commandPalette.placeholder",
            "Type a command or search..."
          )}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          autoFocus
          allowClear
          data-testid="watchlists-command-palette-input"
        />
      </div>
      <div className="max-h-[300px] overflow-y-auto p-3">
        {grouped.length === 0 ? (
          <div className="py-4 text-center text-sm text-text-muted">
            {t("watchlists:commandPalette.noResults", "No matching commands")}
          </div>
        ) : (
          grouped.map(([category, cmds]) => (
            <div key={category} className="mb-2">
              <div className="mb-1 text-xs font-medium uppercase text-text-muted">
                {categoryLabels[category] || category}
              </div>
              {cmds.map((cmd) => (
                <div
                  key={cmd.id}
                  className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-surface-hover"
                  onClick={() => handleSelect(cmd)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleSelect(cmd)
                  }}
                  data-testid={`watchlists-command-${cmd.id}`}
                >
                  <span className="text-text-muted">{cmd.icon}</span>
                  <span className="flex-1">{cmd.label}</span>
                </div>
              ))}
            </div>
          ))
        )}
      </div>
      <div className="border-t border-border px-3 py-2 text-xs text-text-muted">
        {t("watchlists:commandPalette.hint", "Tip: Use Cmd+K to open this palette anytime")}
      </div>
    </Modal>
  )
}

/** Build the default command list for the watchlists module */
export const useWatchlistsCommands = (actions: {
  setActiveTab: (tab: string) => void
  openSourceForm: () => void
  openJobForm: () => void
  openSettings: () => void
  refreshCurrentView: () => void
  startGuidedTour: () => void
}): CommandPaletteCommand[] => {
  const { t } = useTranslation(["watchlists"])

  return useMemo(
    () => [
      {
        id: "nav-feeds",
        label: t("watchlists:commandPalette.commands.openFeeds", "Open Feeds"),
        icon: <Rss className="h-4 w-4" />,
        category: "navigate" as const,
        keywords: ["sources", "rss", "feeds"],
        onExecute: () => actions.setActiveTab("sources")
      },
      {
        id: "nav-articles",
        label: t("watchlists:commandPalette.commands.openArticles", "Open Updates"),
        icon: <Newspaper className="h-4 w-4" />,
        category: "navigate" as const,
        keywords: ["items", "updates", "content"],
        onExecute: () => actions.setActiveTab("items")
      },
      {
        id: "nav-reports",
        label: t("watchlists:commandPalette.commands.openReports", "Open Reports"),
        icon: <FileOutput className="h-4 w-4" />,
        category: "navigate" as const,
        keywords: ["outputs", "briefings", "reports"],
        onExecute: () => actions.setActiveTab("outputs")
      },
      {
        id: "nav-monitors",
        label: t("watchlists:commandPalette.commands.openMonitors", "Open Monitors"),
        icon: <CalendarClock className="h-4 w-4" />,
        category: "navigate" as const,
        keywords: ["jobs", "monitors", "schedule"],
        onExecute: () => actions.setActiveTab("jobs")
      },
      {
        id: "nav-activity",
        label: t("watchlists:commandPalette.commands.openActivity", "Open Activity"),
        icon: <Play className="h-4 w-4" />,
        category: "navigate" as const,
        keywords: ["runs", "activity", "history"],
        onExecute: () => actions.setActiveTab("runs")
      },
      {
        id: "nav-templates",
        label: t("watchlists:commandPalette.commands.openTemplates", "Open Templates"),
        icon: <FileText className="h-4 w-4" />,
        category: "navigate" as const,
        keywords: ["templates", "format"],
        onExecute: () => actions.setActiveTab("templates")
      },
      {
        id: "create-feed",
        label: t("watchlists:commandPalette.commands.createFeed", "Create feed"),
        icon: <Plus className="h-4 w-4" />,
        category: "create" as const,
        keywords: ["add", "new", "feed", "source"],
        onExecute: () => {
          actions.setActiveTab("sources")
          actions.openSourceForm()
        }
      },
      {
        id: "create-monitor",
        label: t("watchlists:commandPalette.commands.createMonitor", "Create monitor"),
        icon: <Plus className="h-4 w-4" />,
        category: "create" as const,
        keywords: ["add", "new", "monitor", "job"],
        onExecute: () => {
          actions.setActiveTab("jobs")
          actions.openJobForm()
        }
      },
      {
        id: "action-refresh",
        label: t("watchlists:commandPalette.commands.refresh", "Refresh current view"),
        icon: <RefreshCw className="h-4 w-4" />,
        category: "action" as const,
        keywords: ["reload", "update"],
        onExecute: actions.refreshCurrentView
      },
      {
        id: "action-settings",
        label: t("watchlists:commandPalette.commands.openSettings", "Open settings"),
        icon: <Settings className="h-4 w-4" />,
        category: "action" as const,
        keywords: ["settings", "preferences", "config"],
        onExecute: actions.openSettings
      },
      {
        id: "action-tour",
        label: t("watchlists:commandPalette.commands.startTour", "Start guided tour"),
        icon: <Play className="h-4 w-4" />,
        category: "action" as const,
        keywords: ["tour", "guide", "help", "onboarding"],
        onExecute: actions.startGuidedTour
      }
    ],
    [actions, t]
  )
}
