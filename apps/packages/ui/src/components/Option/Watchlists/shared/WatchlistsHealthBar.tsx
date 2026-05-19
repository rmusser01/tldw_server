import React, { useCallback, useEffect, useRef, useState } from "react"
import { Button, Spin, Tag, Tooltip } from "antd"
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Clock,
  Newspaper,
  RefreshCw,
  Rss,
  Settings,
  Workflow
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useWatchlistsStore } from "@/store/watchlists"
import {
  fetchWatchlistsOverviewData,
  type WatchlistsOverviewData,
  type WatchlistsOverviewHealthModel
} from "@/services/watchlists-overview"
import { formatRelativeTime } from "@/utils/dateFormatters"

const HEALTH_BAR_STORAGE_KEY = "watchlists:health-bar-expanded:v1"
const HEALTH_BAR_REFRESH_MS = 30_000

const readHealthBarExpanded = (): boolean => {
  if (typeof window === "undefined") return false
  try {
    return localStorage.getItem(HEALTH_BAR_STORAGE_KEY) === "true"
  } catch {
    return false
  }
}

const writeHealthBarExpanded = (expanded: boolean): void => {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(HEALTH_BAR_STORAGE_KEY, String(expanded))
  } catch {
    // localStorage may be unavailable
  }
}

interface HealthBarProps {
  onOpenSettings?: () => void
  onNavigate?: (tab: string) => void
}

export const WatchlistsHealthBar: React.FC<HealthBarProps> = ({ onOpenSettings, onNavigate }) => {
  const { t } = useTranslation(["watchlists", "common"])
  const [expanded, setExpanded] = useState(readHealthBarExpanded)
  const [data, setData] = useState<WatchlistsOverviewData | null>(null)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const setOverviewHealth = useWatchlistsStore((s) => s.setOverviewHealth)
  const storeSetActiveTab = useWatchlistsStore((s) => s.setActiveTab)
  const overviewHealth = useWatchlistsStore((s) => s.overviewHealth)
  const goToTab = onNavigate ?? storeSetActiveTab

  const loadData = useCallback(
    async (showLoading: boolean) => {
      if (showLoading) setLoading(true)
      else setRefreshing(true)
      try {
        const result = await fetchWatchlistsOverviewData()
        setData(result)
        if (typeof setOverviewHealth === "function") {
          setOverviewHealth(result.health, result.fetchedAt)
        }
      } catch (err) {
        console.warn("[WatchlistsHealthBar] Failed to fetch overview data:", err)
      } finally {
        setLoading(false)
        setRefreshing(false)
      }
    },
    [setOverviewHealth]
  )

  useEffect(() => {
    void loadData(true)
    intervalRef.current = setInterval(() => {
      void loadData(false)
    }, HEALTH_BAR_REFRESH_MS)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [loadData])

  const toggleExpanded = useCallback(() => {
    setExpanded((prev) => {
      const next = !prev
      writeHealthBarExpanded(next)
      return next
    })
  }, [])

  const handleRefresh = useCallback(() => {
    void loadData(false)
  }, [loadData])

  const feedsCount = data?.sources.total ?? 0
  const monitorsTotal = data?.jobs.total ?? 0
  const monitorsActive = data?.jobs.active ?? 0
  const runningRuns = data?.runs.running ?? 0
  const pendingRuns = data?.runs.pending ?? 0
  const failedRuns = data?.runs.failed ?? 0
  const outputsTotal = data?.outputs.total ?? 0
  const lastCheckedAt = data?.runs.running > 0
    ? t("watchlists:healthBar.runningNow", "running now")
    : data?.jobs.nextRunAt
      ? formatRelativeTime(data.jobs.nextRunAt, t)
      : data?.fetchedAt
        ? formatRelativeTime(data.fetchedAt, t)
        : null
  const unreadArticles = data?.items.unread ?? 0
  const attentionTotal = overviewHealth?.attention?.total ?? 0
  const hasAttention = attentionTotal > 0
  const hasOperationalData =
    feedsCount +
      monitorsTotal +
      unreadArticles +
      runningRuns +
      pendingRuns +
      failedRuns +
      outputsTotal >
    0
  const hasNoWatchlistData = Boolean(data) && !hasOperationalData

  const summaryParts: string[] = []
  if (feedsCount > 0) {
    summaryParts.push(
      t("watchlists:healthBar.feedsSummary", "{{count}} feeds", { count: feedsCount })
    )
  }
  if (monitorsActive > 0) {
    summaryParts.push(
      t("watchlists:healthBar.monitorsActive", "{{count}} monitors active", {
        count: monitorsActive
      })
    )
  }
  if (failedRuns > 0) {
    summaryParts.push(
      t("watchlists:healthBar.runsFailed", "{{count}} failed", {
        count: failedRuns
      })
    )
  }
  if (lastCheckedAt && hasOperationalData) {
    summaryParts.push(
      data?.jobs.nextRunAt
        ? t("watchlists:healthBar.nextRun", "Next run {{time}}", { time: lastCheckedAt })
        : t("watchlists:healthBar.lastChecked", "Checked {{time}}", { time: lastCheckedAt })
    )
  }
  if (unreadArticles > 0) {
    summaryParts.push(
      t("watchlists:healthBar.articlesPending", "{{count}} articles pending", {
        count: unreadArticles
      })
    )
  }

  if (loading && !data) {
    return (
      <div
        className="mb-4 rounded-lg border border-border bg-surface px-4 py-2"
        data-testid="watchlists-health-bar"
      >
        <Spin size="small" />
      </div>
    )
  }

  return (
    <div
      className="mb-4 rounded-lg border border-border bg-surface"
      data-testid="watchlists-health-bar"
    >
      {/* Collapsed bar */}
      <div
        className="flex cursor-pointer items-center gap-3 px-4 py-2"
        onClick={toggleExpanded}
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        aria-label={t("watchlists:healthBar.toggleLabel", "Toggle health bar")}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            toggleExpanded()
          }
        }}
      >
        {hasAttention && (
          <Tooltip title={t("watchlists:healthBar.attentionTooltip", "{{count}} items need attention", { count: attentionTotal })}>
            <AlertTriangle className="h-4 w-4 text-amber-500" />
          </Tooltip>
        )}
        <span className="flex-1 text-sm text-text-muted" data-testid="watchlists-health-bar-summary">
          {summaryParts.length > 0 ? summaryParts.join(" \u00B7 ") : t("watchlists:healthBar.noData", "No watchlist data yet")}
        </span>
        <div className="flex items-center gap-2">
          {hasNoWatchlistData && (
            <>
              <Button
                size="small"
                type="default"
                onClick={(e) => {
                  e.stopPropagation()
                  goToTab("sources")
                }}
                data-testid="watchlists-health-setup-feeds"
              >
                {t("watchlists:quickActions.sources", "Set up feeds")}
              </Button>
              <Button
                size="small"
                type="default"
                onClick={(e) => {
                  e.stopPropagation()
                  goToTab("jobs")
                }}
                data-testid="watchlists-health-setup-monitors"
              >
                {t("watchlists:quickActions.jobs", "Configure monitors")}
              </Button>
            </>
          )}
          {refreshing && <Spin size="small" />}
          <Tooltip title={t("watchlists:healthBar.refresh", "Refresh")}>
            <Button
              type="text"
              size="small"
              icon={<RefreshCw className="h-3.5 w-3.5" />}
              onClick={(e) => {
                e.stopPropagation()
                handleRefresh()
              }}
              data-testid="watchlists-health-bar-refresh"
            />
          </Tooltip>
          {onOpenSettings && (
            <Tooltip title={t("watchlists:tabs.settings", "Settings")}>
              <Button
                type="text"
                size="small"
                icon={<Settings className="h-3.5 w-3.5" />}
                onClick={(e) => {
                  e.stopPropagation()
                  onOpenSettings()
                }}
                data-testid="watchlists-health-bar-settings"
              />
            </Tooltip>
          )}
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-text-muted" />
          ) : (
            <ChevronDown className="h-4 w-4 text-text-muted" />
          )}
        </div>
      </div>

      {/* Expanded section */}
      {expanded && data && (
        <div
          className="border-t border-border px-4 py-3"
          data-testid="watchlists-health-bar-expanded"
        >
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {/* Feeds health */}
            <HealthCard
              icon={<Rss className="h-4 w-4" />}
              label={t("watchlists:terminology.canonical.feeds", "Feeds")}
              value={String(data.sources.total)}
              status={overviewHealth?.statuses.sources}
              detail={
                data.sources.degraded > 0
                  ? t("watchlists:healthBar.feedsDegraded", "{{count}} need review", {
                      count: data.sources.degraded
                    })
                  : undefined
              }
              onClick={() => goToTab("sources")}
            />
            {/* Monitors health */}
            <HealthCard
              icon={<Workflow className="h-4 w-4" />}
              label={t("watchlists:terminology.canonical.monitors", "Monitors")}
              value={`${data.jobs.active}/${data.jobs.total}`}
              status={overviewHealth?.statuses.jobs}
              detail={
                data.jobs.attention > 0
                  ? t("watchlists:healthBar.monitorsAttention", "{{count}} need fixes", {
                      count: data.jobs.attention
                    })
                  : undefined
              }
              onClick={() => goToTab("jobs")}
            />
            {/* Activity health */}
            <HealthCard
              icon={<Clock className="h-4 w-4" />}
              label={t("watchlists:terminology.canonical.activity", "Activity")}
              value={
                data.runs.running > 0
                  ? t("watchlists:healthBar.runsRunning", "{{count}} running", {
                      count: data.runs.running
                    })
                  : data.runs.failed > 0
                    ? t("watchlists:healthBar.runsFailed", "{{count}} failed", {
                        count: data.runs.failed
                      })
                    : t("watchlists:healthBar.runsOk", "OK")
              }
              status={overviewHealth?.statuses.runs}
              detail={
                data.runs.failed > 0
                  ? t("watchlists:healthBar.runsFailedDetail", "{{count}} recent failures", {
                      count: data.runs.failed
                    })
                  : undefined
              }
              onClick={() => goToTab("runs")}
            />
            {/* Articles */}
            <HealthCard
              icon={<Newspaper className="h-4 w-4" />}
              label={t("watchlists:terminology.canonical.articles", "Articles")}
              value={String(data.items.unread)}
              detail={t("watchlists:healthBar.articlesUnread", "unread")}
              onClick={() => goToTab("items")}
            />
          </div>

          {/* Attention items */}
          {hasAttention && (
            <div className="mt-3 flex flex-wrap gap-2" data-testid="watchlists-health-bar-attention">
              {(overviewHealth?.attention?.sources ?? 0) > 0 && (
                <Tag
                  color="warning"
                  className="cursor-pointer"
                  onClick={() => goToTab("sources")}
                >
                  {t("watchlists:overview.attention.sources", "Feeds need review ({{count}})", {
                    count: overviewHealth?.attention?.sources ?? 0
                  })}
                </Tag>
              )}
              {(overviewHealth?.attention?.runs ?? 0) > 0 && (
                <Button
                  size="small"
                  danger
                  onClick={() => goToTab("runs")}
                  data-testid="watchlists-health-open-activity"
                >
                  {t("watchlists:healthBar.openActivityFailures", "Open Activity ({{count}} failed)", {
                    count: overviewHealth?.attention?.runs ?? 0
                  })}
                </Button>
              )}
              {(overviewHealth?.attention?.outputs ?? 0) > 0 && (
                <Tag
                  color="warning"
                  className="cursor-pointer"
                  onClick={() => goToTab("outputs")}
                >
                  {t("watchlists:overview.attention.outputs", "Reports with delivery issues ({{count}})", {
                    count: overviewHealth?.attention?.outputs ?? 0
                  })}
                </Tag>
              )}
              {(overviewHealth?.attention?.jobs ?? 0) > 0 && (
                <Tag
                  color="warning"
                  className="cursor-pointer"
                  onClick={() => goToTab("jobs")}
                >
                  {t("watchlists:overview.attention.jobs", "Monitors need schedule fixes ({{count}})", {
                    count: overviewHealth?.attention?.jobs ?? 0
                  })}
                </Tag>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

interface HealthCardProps {
  icon: React.ReactNode
  label: string
  value: string
  status?: string
  detail?: string
  onClick?: () => void
}

const statusColor = (status?: string): string => {
  switch (status) {
    case "healthy":
      return "text-green-600"
    case "attention":
      return "text-amber-500"
    case "inactive":
      return "text-text-muted"
    default:
      return "text-text-muted"
  }
}

const HealthCard: React.FC<HealthCardProps> = ({
  icon,
  label,
  value,
  status,
  detail,
  onClick
}) => (
  <div
    className="flex cursor-pointer items-start gap-2 rounded-md border border-border p-2 transition-colors hover:bg-surface-hover"
    onClick={onClick}
    role="button"
    tabIndex={0}
    onKeyDown={(e) => {
      if ((e.key === "Enter" || e.key === " ") && onClick) {
        e.preventDefault()
        onClick()
      }
    }}
  >
    <span className={statusColor(status)}>{icon}</span>
    <div className="min-w-0 flex-1">
      <div className="text-xs font-medium text-text-muted">{label}</div>
      <div className="text-sm font-semibold">{value}</div>
      {detail && <div className="text-xs text-text-muted">{detail}</div>}
    </div>
  </div>
)
