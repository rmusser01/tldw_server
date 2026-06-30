import React, { useCallback, useEffect, useState } from "react"
import {
  Button,
  Descriptions,
  Drawer,
  InputNumber,
  Popconfirm,
  Space,
  Spin,
  Tag,
  message
} from "antd"
import { RefreshCw } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Alert } from "@/components/ui/primitives"
import { clearSourceSeen, getSourceSeenStats } from "@/services/watchlists"
import type { SourceSeenStats } from "@/types/watchlists"
import { formatRelativeTime } from "@/utils/dateFormatters"

interface SourceSeenDrawerProps {
  open: boolean
  onClose: () => void
  sourceId: number | null
  sourceName?: string
  isAdmin?: boolean
}

export const SourceSeenDrawer: React.FC<SourceSeenDrawerProps> = ({
  open,
  onClose,
  sourceId,
  sourceName,
  isAdmin
}) => {
  const { t } = useTranslation(["watchlists", "common"])
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState<SourceSeenStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [resetting, setResetting] = useState(false)
  const [targetUserId, setTargetUserId] = useState<number | null>(null)

  const loadStats = useCallback(async () => {
    if (!sourceId) return
    setLoading(true)
    setError(null)
    try {
      const params: { target_user_id?: number; keys_limit?: number } = {
        keys_limit: 50
      }
      if (targetUserId) params.target_user_id = targetUserId
      const result = await getSourceSeenStats(sourceId, params)
      setStats(result)
    } catch (err: any) {
      const msg = err?.message || "Failed to load seen stats"
      setError(msg)
      console.error("Failed to load seen stats:", err)
    } finally {
      setLoading(false)
    }
  }, [sourceId, targetUserId])

  useEffect(() => {
    if (open && sourceId) {
      loadStats()
    } else {
      setStats(null)
      setError(null)
      setTargetUserId(null)
    }
  }, [open, sourceId, loadStats])

  const handleReset = async (clearBackoff: boolean) => {
    if (!sourceId) return
    setResetting(true)
    try {
      const params: { target_user_id?: number; clear_backoff?: boolean } = {
        clear_backoff: clearBackoff
      }
      if (targetUserId) params.target_user_id = targetUserId
      const result = await clearSourceSeen(sourceId, params)
      message.success(
        t("watchlists:sources.seen.resetSuccess", "Cleared {{count}} seen items", {
          count: result.cleared
        })
      )
      await loadStats()
    } catch (err: any) {
      console.error("Failed to reset seen:", err)
      message.error(
        t("watchlists:sources.seen.resetError", "Failed to reset seen data")
      )
    } finally {
      setResetting(false)
    }
  }

  const backoffStatus = (): { color: string; label: string } => {
    if (!stats) return { color: "default", label: "-" }
    const consec = stats.consec_not_modified ?? 0
    if (!stats.defer_until && consec === 0) {
      return { color: "green", label: t("watchlists:sources.seen.noBackoff", "None") }
    }
    if (consec >= 5) {
      return { color: "red", label: t("watchlists:sources.seen.highBackoff", "High ({{n}})", { n: consec }) }
    }
    return { color: "orange", label: t("watchlists:sources.seen.activeBackoff", "Active ({{n}})", { n: consec }) }
  }

  const title = sourceName
    ? t("watchlists:sources.seen.titleNamed", "Seen / Dedup — {{name}}", { name: sourceName })
    : t("watchlists:sources.seen.title", "Seen / Dedup")

  return (
    <Drawer
      title={title}
      placement="right"
      onClose={onClose}
      open={open}
      styles={{ wrapper: { width: 520 } }}
    >
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Spin size="large" />
        </div>
      ) : error ? (
        <Alert
          variant="error"
          title={error}
          className="mb-4"
          data-testid="alert-error"
        />
      ) : stats ? (
        <div className="space-y-6">
          {/* Stats section */}
          <Descriptions column={1} size="small" bordered>
            <Descriptions.Item label={t("watchlists:sources.seen.seenCount", "Seen Count")}>
              {stats.seen_count}
            </Descriptions.Item>
            <Descriptions.Item label={t("watchlists:sources.seen.latestSeen", "Latest Seen")}>
              {stats.latest_seen_at
                ? formatRelativeTime(stats.latest_seen_at, t)
                : t("watchlists:sources.seen.never", "Never")}
            </Descriptions.Item>
            <Descriptions.Item label={t("watchlists:sources.seen.backoffStatus", "Backoff Status")}>
              <Tag color={backoffStatus().color}>{backoffStatus().label}</Tag>
            </Descriptions.Item>
          </Descriptions>

          {/* Backoff details */}
          {(stats.defer_until || (stats.consec_not_modified ?? 0) > 0) && (
            <Descriptions
              column={1}
              size="small"
              bordered
              title={t("watchlists:sources.seen.backoffDetails", "Backoff Details")}
            >
              <Descriptions.Item label={t("watchlists:sources.seen.deferUntil", "Defer Until")}>
                {stats.defer_until
                  ? formatRelativeTime(stats.defer_until, t)
                  : "-"}
              </Descriptions.Item>
              <Descriptions.Item label={t("watchlists:sources.seen.consecNotModified", "Consecutive Not Modified")}>
                {stats.consec_not_modified ?? 0}
              </Descriptions.Item>
            </Descriptions>
          )}

          {/* Recent keys */}
          {stats.recent_keys.length > 0 && (
            <div>
              <div className="text-sm font-medium mb-2">
                {t("watchlists:sources.seen.recentKeys", "Recent Keys ({{count}})", {
                  count: stats.recent_keys.length
                })}
              </div>
              <div className="max-h-48 overflow-auto rounded border border-border p-2">
                {stats.recent_keys.map((key, idx) => (
                  <div
                    key={idx}
                    className="text-xs font-mono py-0.5 border-b border-border last:border-b-0 truncate"
                    title={key}
                  >
                    {key}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Reset controls */}
          <div className="space-y-3">
            <div className="text-sm font-medium">
              {t("watchlists:sources.seen.resetControls", "Reset Controls")}
            </div>
            <Space>
              <Popconfirm
                title={t("watchlists:sources.seen.clearConfirm", "Clear all seen items for this source?")}
                onConfirm={() => handleReset(false)}
                okText={t("common:yes", "Yes")}
                cancelText={t("common:no", "No")}
              >
                <Button loading={resetting} data-testid="clear-seen-btn">
                  {t("watchlists:sources.seen.clearSeen", "Clear Seen Items")}
                </Button>
              </Popconfirm>
              <Popconfirm
                title={t("watchlists:sources.seen.clearAllConfirm", "Clear seen items AND reset backoff state?")}
                onConfirm={() => handleReset(true)}
                okText={t("common:yes", "Yes")}
                cancelText={t("common:no", "No")}
              >
                <Button danger loading={resetting} data-testid="clear-all-btn">
                  {t("watchlists:sources.seen.clearAll", "Clear All + Reset Backoff")}
                </Button>
              </Popconfirm>
              <Button
                icon={<RefreshCw className="h-4 w-4" />}
                onClick={loadStats}
                loading={loading}
              />
            </Space>
          </div>

          {/* Admin section */}
          {isAdmin && (
            <div className="border-t border-border pt-4 space-y-3">
              <div className="text-sm font-medium">
                {t("watchlists:sources.seen.adminSection", "Admin: Inspect Other User")}
              </div>
              <Space>
                <InputNumber
                  placeholder={t("watchlists:sources.seen.targetUserId", "User ID")}
                  value={targetUserId}
                  onChange={(val) => setTargetUserId(val)}
                  min={1}
                  data-testid="target-user-input"
                />
                <Button onClick={loadStats} data-testid="load-target-btn">
                  {t("watchlists:sources.seen.loadUser", "Load")}
                </Button>
              </Space>
            </div>
          )}
        </div>
      ) : null}
    </Drawer>
  )
}
