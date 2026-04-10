import React, { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Tooltip } from "antd"
import { Cloud, CloudOff, Loader2, AlertCircle, RefreshCw } from "lucide-react"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import type { AnnotationSyncStatus } from "./types"

interface SyncStatusIndicatorProps {
  onRetry?: () => void
  className?: string
}

/**
 * SyncStatusIndicator - Shows the current sync status for annotations
 *
 * States:
 * - synced: Green cloud icon, all changes saved
 * - pending: Yellow spinner, changes being synced
 * - error: Red icon with retry button
 */
export const SyncStatusIndicator: React.FC<SyncStatusIndicatorProps> = ({
  onRetry,
  className = ""
}) => {
  const { t } = useTranslation(["option", "common"])
  const syncStatus = useDocumentWorkspaceStore((s) => s.annotationSyncStatus)
  const pendingCount = useDocumentWorkspaceStore((s) => s.pendingAnnotations.length)

  // Track previous status to detect recovery from error
  const prevStatusRef = useRef<AnnotationSyncStatus>(syncStatus)
  const [justRecovered, setJustRecovered] = useState(false)

  useEffect(() => {
    if (prevStatusRef.current === "error" && syncStatus === "synced") {
      setJustRecovered(true)
      const timer = setTimeout(() => setJustRecovered(false), 2000)
      return () => clearTimeout(timer)
    }
    prevStatusRef.current = syncStatus
  }, [syncStatus])

  const getStatusConfig = (status: AnnotationSyncStatus) => {
    switch (status) {
      case "synced":
        return {
          icon: <Cloud className="h-4 w-4" />,
          color: "text-success",
          bgColor: "bg-success/10",
          tooltip: t("option:documentWorkspace.syncStatus.synced", "All changes saved"),
          ariaLabel: t("option:documentWorkspace.syncStatus.synced", "All changes saved")
        }
      case "pending":
        return {
          icon: <Loader2 className="h-4 w-4 animate-spin" />,
          color: "text-warning",
          bgColor: "bg-warning/10",
          tooltip: t(
            "option:documentWorkspace.syncStatus.pending",
            "Saving... ({{count}} pending)",
            { count: pendingCount }
          ),
          ariaLabel: t(
            "option:documentWorkspace.syncStatus.pending",
            "Saving"
          )
        }
      case "error":
        return {
          icon: <CloudOff className="h-4 w-4" />,
          color: "text-error",
          bgColor: "bg-error/10",
          tooltip: t(
            "option:documentWorkspace.syncStatus.error",
            "Failed to save highlights. Click to retry."
          ),
          ariaLabel: t("option:documentWorkspace.syncStatus.error", "Save failed")
        }
    }
  }

  const config = getStatusConfig(syncStatus)

  // For error state, make it a button to retry
  if (syncStatus === "error" && onRetry) {
    return (
      <Tooltip title={config.tooltip}>
        <button
          onClick={onRetry}
          className={`
            flex items-center gap-1.5 px-2 py-1 rounded-md
            ${config.bgColor} ${config.color}
            hover:opacity-80 transition-opacity
            focus:outline-none focus:ring-2 focus:ring-error/50
            ${className}
          `}
          aria-label={config.ariaLabel}
        >
          <AlertCircle className="h-4 w-4" />
          <span className="text-xs font-medium">
            {t("option:documentWorkspace.syncStatus.retryLabel", "Save failed")}
          </span>
          <RefreshCw className="h-3 w-3 ml-1" />
        </button>
      </Tooltip>
    )
  }

  // For synced state, show minimal indicator (pulse briefly after recovery)
  if (syncStatus === "synced") {
    return (
      <Tooltip title={config.tooltip}>
        <div
          className={`
            flex items-center gap-1 px-2 py-1 rounded-md
            ${config.color} ${justRecovered ? "opacity-100 animate-pulse" : "opacity-60"} hover:opacity-100
            transition-opacity cursor-default
            ${className}
          `}
          role="status"
          aria-label={config.ariaLabel}
        >
          {config.icon}
        </div>
      </Tooltip>
    )
  }

  // For pending state, show spinner with count
  return (
    <Tooltip title={config.tooltip}>
      <div
        className={`
          flex items-center gap-1.5 px-2 py-1 rounded-md
          ${config.bgColor} ${config.color}
          ${className}
        `}
        role="status"
        aria-label={config.ariaLabel}
        aria-live="polite"
      >
        {config.icon}
        {pendingCount > 0 && (
          <span className="text-xs font-medium">{pendingCount}</span>
        )}
      </div>
    </Tooltip>
  )
}

export default SyncStatusIndicator
