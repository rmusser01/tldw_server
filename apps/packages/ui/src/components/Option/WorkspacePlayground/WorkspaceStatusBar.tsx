import React from "react"
import { useTranslation } from "react-i18next"
import { Tooltip, Modal, Button } from "antd"
import { Loader2, Trash2 } from "lucide-react"
import { useConnectionStore } from "@/store/connection"
import { deriveConnectionUxState, type ConnectionUxState } from "@/types/connection"
import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"

type ConnectionTone = {
  label: string
  detail: string
  description: string
  toneClass: string
  dotClass: string
}

interface WorkspaceStatusBarProps {
  /** Approximate persisted workspace payload bytes in local storage */
  storageUsedBytes?: number
  /** Estimated available local storage budget for workspace payload data */
  storageQuotaBytes?: number
  /** Active operation labels shown in the status bar */
  activeOperations?: string[]
  /** Rollout gate for status/guardrails surfaces */
  statusGuardrailsEnabled?: boolean
}

const deriveConnectionTone = (
  connectionState: ReturnType<typeof useConnectionStore.getState>["state"]
): ConnectionTone => {
  const ux = deriveConnectionUxState(connectionState)

  if (ux === "connected_ok") {
    return {
      label: "Connected",
      detail: "Connection healthy",
      description: "",
      toneClass: "text-success",
      dotClass: "bg-success"
    }
  }
  if (
    ux === "testing" ||
    ux === "connected_degraded" ||
    ux === "demo_mode"
  ) {
    return {
      label: "Degraded",
      detail: "Connection degraded",
      description: "",
      toneClass: "text-warning",
      dotClass: "bg-warning"
    }
  }
  return {
    label: "Disconnected",
    detail: "Cannot reach backend",
    description: "",
    toneClass: "text-error",
    dotClass: "bg-error"
  }
}

export const WorkspaceStatusBar: React.FC<WorkspaceStatusBarProps> = ({
  storageUsedBytes,
  storageQuotaBytes,
  activeOperations = [],
  statusGuardrailsEnabled = true
}) => {
  const { t } = useTranslation(["playground", "common"])
  const connectionState = useConnectionStore((s) => s.state)
  const connection = deriveConnectionTone(connectionState)
  const connectionUxState: ConnectionUxState = deriveConnectionUxState(connectionState)
  const [storageModalOpen, setStorageModalOpen] = React.useState(false)

  const storageItems = React.useMemo(() => {
    if (!storageModalOpen) return []
    const items: Array<{ key: string; label: string; bytes: number }> = []
    try {
      const prefix = `${WORKSPACE_STORAGE_KEY}:workspace:`
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i)
        if (!key?.startsWith(prefix)) continue
        const value = localStorage.getItem(key)
        if (!value) continue
        const bytes = new Blob([value]).size
        let label = key.slice(prefix.length)
        try {
          const parsed = JSON.parse(value)
          if (parsed?.state?.workspaceName) label = parsed.state.workspaceName
        } catch (err) { console.warn("WorkspaceStatusBar: localStorage error", err) }
        items.push({ key, label, bytes })
      }
    } catch (err) { console.warn("WorkspaceStatusBar: localStorage error", err) }
    return items.sort((a, b) => b.bytes - a.bytes)
  }, [storageModalOpen])

  const handleDeleteStorageItem = React.useCallback((key: string) => {
    try {
      localStorage.removeItem(key)
      setStorageModalOpen(false)
      setTimeout(() => setStorageModalOpen(true), 50)
    } catch (err) { console.warn("WorkspaceStatusBar: localStorage error", err) }
  }, [])

  const storageBar = React.useMemo(() => {
    if (
      typeof storageUsedBytes !== "number" ||
      !Number.isFinite(storageUsedBytes) ||
      storageUsedBytes < 0 ||
      typeof storageQuotaBytes !== "number" ||
      !Number.isFinite(storageQuotaBytes) ||
      storageQuotaBytes <= 0
    ) {
      return null
    }
    const ratio = Math.max(0, Math.min(1, storageUsedBytes / storageQuotaBytes))
    const usedMb = (storageUsedBytes / (1024 * 1024)).toFixed(1)
    const quotaMb = Math.round(storageQuotaBytes / (1024 * 1024))
    const toneClass =
      ratio >= 0.95
        ? "bg-error"
        : ratio >= 0.8
          ? "bg-warning"
          : "bg-primary/60"
    return { ratio, usedMb, quotaMb, toneClass }
  }, [storageUsedBytes, storageQuotaBytes])

  if (!statusGuardrailsEnabled) return null

  return (
    <footer
      data-testid="workspace-status-bar"
      className="flex h-7 shrink-0 items-center justify-between border-t border-border/60 bg-surface px-3 text-[11px] text-text-muted"
    >
      <div className="flex items-center gap-3">
        {/* Connection indicator */}
        <Tooltip title={connection.detail}>
          <span
            data-testid="workspace-statusbar-connection"
            className={`inline-flex items-center gap-1.5 ${connection.toneClass}`}
          >
            <span className={`inline-block h-2 w-2 rounded-full ${connection.dotClass}`} />
            {connection.label}
          </span>
        </Tooltip>
        {(connectionUxState === "error_unreachable" || connectionUxState === "error_auth") && (
          <button
            type="button"
            data-testid="workspace-statusbar-retry"
            className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] text-error hover:bg-error/10 hover:text-error"
            onClick={() => {
              void useConnectionStore.getState().checkOnce()
            }}
          >
            {t("playground:statusBar.retry", "Retry")}
          </button>
        )}

        {/* Storage mini progress — click to manage */}
        {storageBar && (
          <Tooltip title={t("playground:statusBar.manageStorage", "Click to manage storage")}>
            <button
              type="button"
              data-testid="workspace-statusbar-storage"
              className="inline-flex items-center gap-1.5 hover:text-text"
              onClick={() => setStorageModalOpen(true)}
            >
              <span className="relative inline-block h-1.5 w-16 overflow-hidden rounded-full bg-border/60">
                <span
                  className={`absolute inset-y-0 left-0 rounded-full transition-all ${storageBar.toneClass}`}
                  style={{ width: `${Math.round(storageBar.ratio * 100)}%` }}
                />
              </span>
              <span>{storageBar.usedMb}/{storageBar.quotaMb} MB</span>
            </button>
          </Tooltip>
        )}
      </div>

      {/* Active operations */}
      <div className="flex items-center gap-3">
        {activeOperations.length > 0 && (
          <div
            data-testid="workspace-statusbar-activity"
            role="status"
            aria-live="polite"
            className="flex items-center gap-3"
          >
            <Loader2 className="h-3 w-3 animate-spin text-primary" />
            {activeOperations.map((op, i) => (
              <span key={i} className="inline-flex items-center gap-1">
                <span className="truncate">{op}</span>
                {i < activeOperations.length - 1 && (
                  <span className="text-text-subtle">&bull;</span>
                )}
              </span>
            ))}
          </div>
        )}
        <a
          href="https://github.com/rmusser01/tldw_server2#readme"
          target="_blank"
          rel="noreferrer"
          className="text-[10px] text-text-muted hover:text-primary transition-colors"
        >
          Help
        </a>
      </div>
      <Modal
        title={t("playground:statusBar.storageTitle", "Workspace Storage")}
        open={storageModalOpen}
        onCancel={() => setStorageModalOpen(false)}
        footer={null}
        width={480}
        destroyOnHidden
      >
        {storageItems.length === 0 ? (
          <p className="py-4 text-center text-sm text-text-muted">
            {t("playground:statusBar.noStorageItems", "No workspace data found in local storage.")}
          </p>
        ) : (
          <div className="space-y-1">
            {storageItems.map((item) => (
              <div
                key={item.key}
                className="flex items-center justify-between rounded-md border border-border px-3 py-2"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium text-text">
                    {item.label}
                  </div>
                  <div className="text-xs text-text-muted">
                    {(item.bytes / 1024).toFixed(1)} KB
                  </div>
                </div>
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<Trash2 className="h-3.5 w-3.5" />}
                  onClick={() => handleDeleteStorageItem(item.key)}
                  aria-label={t("common:delete", "Delete")}
                />
              </div>
            ))}
          </div>
        )}
      </Modal>
    </footer>
  )
}
