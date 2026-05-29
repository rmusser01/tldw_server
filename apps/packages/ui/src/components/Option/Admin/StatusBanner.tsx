import React from "react"
import { Tag, Button, Space } from "antd"
import { RefreshCw, Square } from "lucide-react"
import { LoadingState } from "@/components/ui/feedback/LoadingState"
import { Alert } from "@/components/ui/primitives/Alert"
import { sanitizeAdminErrorMessage } from "./admin-error-utils"

type StatusState = "running" | "online" | "stopped" | "offline" | "loading" | "active" | "inactive" | "unknown"

interface StatusItem {
  label: string
  value: string | number | undefined | null
  code?: boolean
}

interface StatusBannerProps {
  /** Current state of the service/model */
  state: StatusState | string
  /** Whether status is currently loading */
  loading?: boolean
  /** Error message to display */
  error?: string | null
  /** Additional status items to display (e.g., model name, port) */
  items?: StatusItem[]
  /** Callback for refresh button */
  onRefresh?: () => void
  /** Quick action button (e.g., Stop, Unload) */
  quickAction?: {
    label: string
    onClick: () => void
    loading?: boolean
    danger?: boolean
    disabled?: boolean
  }
  /** Translation function for state labels */
  stateLabel?: (state: string) => string
  className?: string
}

const getStateColor = (state: string): string => {
  const normalized = state.toLowerCase()
  if (normalized === "running" || normalized === "online" || normalized === "active") {
    return "green"
  }
  if (normalized === "stopped" || normalized === "offline" || normalized === "inactive") {
    return "red"
  }
  if (normalized === "loading") {
    return "orange"
  }
  return "default"
}

/**
 * Reusable status banner for admin pages.
 * Displays current state, status items, and quick actions in a condensed format.
 */
export const StatusBanner: React.FC<StatusBannerProps> = ({
  state,
  loading = false,
  error,
  items = [],
  onRefresh,
  quickAction,
  stateLabel,
  className
}) => {
  const displayState = stateLabel ? stateLabel(state) : state
  const stateColor = getStateColor(state)

  if (loading) {
    return (
      <LoadingState
        mode="inline"
        size="sm"
        label="Loading status..."
        className={`rounded-lg border border-border bg-bg p-3 ${className || ""}`}
      />
    )
  }

  if (error) {
    const safeError = sanitizeAdminErrorMessage(
      error,
      "Unable to load status details."
    )
    return (
      <Alert
        variant="error"
        title="Status Error"
        className={className}
        action={onRefresh ? { label: "Retry", onClick: onRefresh } : undefined}
      >
        {safeError}
      </Alert>
    )
  }

  return (
    <div className={`flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-bg p-3 ${className || ""}`}>
      <Space wrap size="middle">
        <Tag color={stateColor} className="m-0">
          {displayState}
        </Tag>
        {items.map((item, index) =>
          item.value != null ? (
            <Space key={index} size="small" className="text-sm">
              <span className="text-text-muted">{item.label}:</span>
              {item.code ? (
                <code className="rounded bg-surface2 px-1.5 py-0.5 text-xs">
                  {String(item.value)}
                </code>
              ) : (
                <span>{String(item.value)}</span>
              )}
            </Space>
          ) : null
        )}
      </Space>

      <Space size="small">
        {onRefresh && (
          <Button
            size="small"
            icon={<RefreshCw size={14} />}
            onClick={onRefresh}
            title="Refresh status"
          />
        )}
        {quickAction && (
          <Button
            size="small"
            danger={quickAction.danger}
            onClick={quickAction.onClick}
            loading={quickAction.loading}
            disabled={quickAction.disabled}
            icon={quickAction.danger ? <Square size={14} /> : undefined}
          >
            {quickAction.label}
          </Button>
        )}
      </Space>
    </div>
  )
}

export default StatusBanner
