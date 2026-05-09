import { Tooltip } from "antd"
import { useTranslation } from "react-i18next"
import { Check, Loader2, AlertCircle } from "lucide-react"
import {
  useConnectionActions,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import { getDesignSystemState, type DesignSystemStateKey } from "@/design-system"
import {
  Badge,
  getBadgeVariantForDesignSystemSeverity,
  type BadgeVariant
} from "@/components/ui/primitives"

/**
 * Compact connection status indicator with icon and color for accessibility.
 *
 * States:
 * - Connected: Green checkmark
 * - Checking: Yellow spinner
 * - Disconnected/Error: Amber warning icon
 *
 * Uses both color AND shape for color-blind accessibility.
 */
export const StatusDot = () => {
  const { t } = useTranslation(["sidepanel"])
  const { mode, isConnectedUx, isChecking, isConfigOrError } =
    useConnectionUxState()
  const { checkOnce } = useConnectionActions()

  const tooltip = (() => {
    if (isChecking) {
      return t(
        "sidepanel:header.connection.checking",
        "Checking connection to your tldw server…"
      )
    }
    if (isConnectedUx && mode === "demo") {
      return t(
        "sidepanel:header.connection.demo",
        "Demo mode: explore with a sample workspace."
      )
    }
    if (isConnectedUx) {
      return t(
        "sidepanel:header.connection.ok",
        "Connected to your tldw server"
      )
    }
    if (isConfigOrError) {
      return t(
        "sidepanel:header.connection.unconfigured",
        "Not connected. Open Settings to configure."
      )
    }
    return t(
      "sidepanel:header.connection.failed",
      "Connection failed. Click to retry."
    )
  })()

  const handleClick = () => {
    if (isChecking) return
    if (!isConnectedUx && !isConfigOrError) {
      // Retry connection
      void checkOnce()
    }
  }

  // Render icon based on state - uses shape AND color for accessibility
  const renderStatusIcon = () => {
    if (isChecking) {
      return (
        <Loader2 className="h-3.5 w-3.5 animate-spin text-current" />
      )
    }
    if (isConnectedUx) {
      return (
        <Check className="h-3.5 w-3.5 text-current" />
      )
    }
    return (
      <AlertCircle className="h-3.5 w-3.5 text-current" />
    )
  }

  const connectionStateKey: DesignSystemStateKey = (() => {
    if (isChecking) return "retrying"
    if (isConnectedUx) return "ready"
    if (isConfigOrError) return "setup_required"
    return "unavailable"
  })()
  const connectionState = getDesignSystemState(connectionStateKey)
  const badgeVariant: BadgeVariant =
    isConnectedUx && mode === "demo"
      ? "demo"
      : getBadgeVariantForDesignSystemSeverity(connectionState.severity)

  return (
    <Tooltip title={tooltip}>
      <button
        type="button"
        data-testid="status-dot"
        onClick={handleClick}
        disabled={isChecking}
        className="rounded-full p-1 hover:bg-surface2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-default"
        aria-label={tooltip}
        title={tooltip}
      >
        <Badge
          data-testid="status-dot-badge"
          variant={badgeVariant}
          size="sm"
          outline
          className="gap-0 leading-none"
        >
          {renderStatusIcon()}
        </Badge>
      </button>
    </Tooltip>
  )
}
