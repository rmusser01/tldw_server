import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { useConnectionState } from "@/hooks/useConnectionState"
import { ConnectionPhase } from "@/types/connection"
import {
  getDesignSystemState,
  type DesignSystemSeverity,
  type DesignSystemStateKey
} from "@/design-system"
import {
  Badge,
  getBadgeVariantForDesignSystemSeverity
} from "@/components/ui/primitives"

type StatusKind = "unknown" | "ok" | "fail"

const SEVERITY_STYLES = {
  success: {
    bg: "border-success/30 bg-success/10",
    text: "text-success"
  },
  error: {
    bg: "border-danger/30 bg-danger/10",
    text: "text-danger"
  },
  warning: {
    bg: "border-warn/30 bg-warn/10",
    text: "text-warn"
  },
  info: {
    bg: "border-primary/30 bg-primary/10",
    text: "text-primary"
  },
  neutral: {
    bg: "border-border bg-surface2",
    text: "text-text-muted"
  }
} satisfies Record<DesignSystemSeverity, { bg: string; text: string }>

interface ConnectionStatusProps {
  /** Custom click handler (defaults to navigating to /settings/health) */
  onClick?: () => void
  /** Whether to show the label text (default: true) */
  showLabel?: boolean
  /** Additional CSS classes */
  className?: string
}

/**
 * Connection status indicator with clickable health diagnostics link.
 * Extracted from Header.tsx for reuse.
 */
export function ConnectionStatus({
  onClick,
  showLabel = true,
  className,
}: ConnectionStatusProps) {
  const { t } = useTranslation(["settings", "common"])
  const navigate = useNavigate()
  const { phase, isConnected } = useConnectionState()

  const coreStatus: StatusKind =
    phase === ConnectionPhase.SEARCHING
      ? "unknown"
      : isConnected && phase === ConnectionPhase.CONNECTED
        ? "ok"
        : phase === ConnectionPhase.ERROR
          ? "fail"
          : "unknown"

  const stateKeyForCoreStatus = (status: StatusKind): DesignSystemStateKey => {
    if (phase === ConnectionPhase.UNCONFIGURED) {
      return "setup_required"
    }
    if (status === "ok") {
      return "ready"
    }
    if (status === "fail") {
      return "unavailable"
    }
    return "retrying"
  }

  const coreStateKey = stateKeyForCoreStatus(coreStatus)
  const coreState = getDesignSystemState(coreStateKey)

  const statusLabelForCore = (status: StatusKind): string => {
    if (phase === ConnectionPhase.UNCONFIGURED) {
      return t(
        "settings:healthSummary.coreUnconfigured",
        "Server: Not configured"
      )
    }
    if (status === "ok") {
      return t("settings:healthSummary.coreOnline", "Server: Online")
    }
    if (status === "fail") {
      return t("settings:healthSummary.coreOffline", "Server: Offline")
    }
    return t("settings:healthSummary.coreChecking", "Server: Checking...")
  }

  const handleClick = () => {
    if (onClick) {
      onClick()
    } else {
      navigate("/settings/health")
    }
  }

  const severityStyles = SEVERITY_STYLES[coreState.severity]
  const statusBgClass = severityStyles.bg
  const statusTextClass = severityStyles.text

  return (
    <button
      type="button"
      data-testid="connection-status"
      onClick={handleClick}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition hover:opacity-80 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus ${statusBgClass} ${className || ""}`}
      title={
        t(
          "settings:healthSummary.coreAria",
          "Server status - click for diagnostics"
        ) as string
      }
      aria-label={
        t(
          "settings:healthSummary.fullDiagnosticsAria",
          "{{label}}. {{status}}. {{help}}",
          {
            label: t(
              "settings:healthSummary.diagnostics",
              "Health & diagnostics"
            ),
            status: statusLabelForCore(coreStatus),
            help: t(
              "settings:healthSummary.diagnosticsTooltip",
              "Open detailed diagnostics to troubleshoot or inspect health checks."
            ),
          }
        )
      }
    >
      <StatusDot status={coreStatus} stateKey={coreStateKey} />
      {showLabel && (
        <span className={statusTextClass}>
          {statusLabelForCore(coreStatus)}
        </span>
      )}
    </button>
  )
}

/**
 * Simple status dot indicator with animation for unknown state
 */
export function StatusDot({
  status,
  stateKey
}: {
  status: StatusKind
  stateKey: DesignSystemStateKey
}) {
  const state = getDesignSystemState(stateKey)

  return (
    <Badge
      data-testid="connection-status-dot-badge"
      variant={getBadgeVariantForDesignSystemSeverity(state.severity)}
      size="sm"
      outline
      className="gap-0 px-1 py-1 leading-none"
    >
      <span
        data-testid="connection-status-dot"
        aria-hidden
        className={`inline-block h-2 w-2 rounded-full bg-current ${
          status === "unknown" ? "animate-pulse" : ""
        }`}
      />
    </Badge>
  )
}

export default ConnectionStatus
