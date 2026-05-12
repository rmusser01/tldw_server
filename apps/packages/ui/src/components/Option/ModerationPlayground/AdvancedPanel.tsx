import React from "react"
import { Modal, Tooltip } from "antd"
import { Download, Upload, RefreshCw } from "lucide-react"

import { setUserOverride } from "@/services/moderation"
import type { RawReplacePreview } from "./hooks/useBlocklist"
import type { ModerationSettingsState } from "./hooks/useModerationSettings"
import type { BlocklistState } from "./hooks/useBlocklist"
import type { UserOverridesState } from "./hooks/useUserOverrides"

interface AdvancedPanelProps {
  settings: ModerationSettingsState
  blocklist: BlocklistState
  overrides: UserOverridesState
  messageApi: { success: (msg: string) => void; error: (msg: string) => void; warning: (msg: string) => void }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function triggerDownload(filename: string, content: string, mime = "text/plain") {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

interface PerfFieldProps {
  label: string
  value: number | string
  description: string
}

function PerfField({ label, value, description }: PerfFieldProps) {
  const inputId = React.useId()
  const descriptionId = React.useId()
  return (
    <div className="mb-3">
      <label htmlFor={inputId} className="block text-sm font-medium text-text-secondary mb-0.5">{label}</label>
      <input
        id={inputId}
        type="text"
        readOnly
        value={String(value)}
        className="w-full rounded border border-border bg-bg-secondary px-2 py-1 text-sm text-text-primary"
        aria-describedby={descriptionId}
      />
      <p id={descriptionId} className="mt-0.5 text-xs text-text-muted">{description}</p>
    </div>
  )
}

const pluralize = (count: number, singular: string, plural = `${singular}s`) =>
  `${count} ${count === 1 ? singular : plural}`

const BlocklistUploadPreview: React.FC<{ preview: RawReplacePreview }> = ({ preview }) => (
  <div className="space-y-3 text-sm">
    <p>Uploading this file will replace the current blocklist.</p>
    <div className="grid grid-cols-2 gap-2">
      <span>{pluralize(preview.lint.valid_count, "valid row")}</span>
      <span>{pluralize(preview.lint.invalid_count, "invalid row")}</span>
      <span>{pluralize(preview.addedCount, "added line")}</span>
      <span>{pluralize(preview.removedCount, "removed line")}</span>
    </div>
    {preview.lint.invalid_count > 0 && (
      <p className="text-red-600 dark:text-red-400">
        Fix invalid rows before replacing the blocklist.
      </p>
    )}
  </div>
)

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const AdvancedPanel: React.FC<AdvancedPanelProps> = ({ settings, blocklist, overrides, messageApi }) => {
  const policy = settings.policyQuery.data
  const blocklistFileRef = React.useRef<HTMLInputElement>(null)
  const overridesFileRef = React.useRef<HTMLInputElement>(null)
  const blocklistUploadButtonRef = React.useRef<HTMLButtonElement>(null)

  // ---- Performance tuning values (read-only from policy) ----
  const maxScanChars = policy?.max_scan_chars ?? 200000
  const maxReplacementsPerPattern = policy?.max_replacements_per_pattern ?? 1000
  const matchWindowChars = policy?.match_window_chars ?? 4096
  const blocklistWriteDebounceMs = policy?.blocklist_write_debounce_ms ?? 0

  // ---- Export handlers ----
  const [pendingBlocklistDownload, setPendingBlocklistDownload] = React.useState(false)

  const onDownloadBlocklist = React.useCallback(async () => {
    try {
      await blocklist.loadRaw()
      setPendingBlocklistDownload(true)
    } catch {
      messageApi.error("Failed to load blocklist for export")
    }
  }, [blocklist, messageApi])

  React.useEffect(() => {
    if (pendingBlocklistDownload) {
      triggerDownload("blocklist.txt", blocklist.rawText)
      messageApi.success("Blocklist downloaded")
      setPendingBlocklistDownload(false)
    }
  }, [pendingBlocklistDownload, blocklist.rawText, messageApi])

  const onDownloadOverrides = React.useCallback(() => {
    try {
      const data = overrides.overridesQuery.data
      const overridesList = (data as any)?.overrides ?? data ?? []
      const json = JSON.stringify(overridesList, null, 2)
      triggerDownload("overrides.json", json, "application/json")
      messageApi.success("Overrides downloaded")
    } catch {
      messageApi.error("Failed to export overrides")
    }
  }, [overrides.overridesQuery.data, messageApi])

  // ---- Import handlers ----
  const onUploadBlocklist = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (!file) return
      const reader = new FileReader()
      reader.onload = async () => {
        try {
          const content = reader.result as string
          const preview = await blocklist.previewRawReplace(content)
          const returnFocus = () => window.setTimeout(() => blocklistUploadButtonRef.current?.focus(), 0)
          Modal.confirm({
            title: "Confirm blocklist upload",
            content: <BlocklistUploadPreview preview={preview} />,
            okText: "Replace blocklist",
            cancelText: "Cancel",
            okButtonProps: { danger: true, disabled: preview.lint.invalid_count > 0 },
            onCancel: () => {
              blocklist.cancelRawReplace()
              returnFocus()
            },
            onOk: async () => {
              await blocklist.confirmRawReplace()
              messageApi.success("Blocklist replaced successfully")
            }
          })
        } catch (err: any) {
          messageApi.error(err?.message || "Failed to import blocklist")
        }
      }
      reader.readAsText(file)
      // Reset input so same file can be re-uploaded
      e.target.value = ""
    },
    [blocklist, messageApi]
  )

  const onUploadOverrides = React.useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (!file) return
      const reader = new FileReader()
      reader.onload = async () => {
        try {
          const content = reader.result as string
          const parsed = JSON.parse(content)
          const items = Array.isArray(parsed) ? parsed : parsed.overrides ?? []
          for (const item of items) {
            if (item.user_id && typeof item.user_id === "string" && item.user_id.trim()) {
              const { user_id, ...payload } = item
              await setUserOverride(user_id, payload)
            }
          }
          await overrides.overridesQuery.refetch()
          messageApi.success("Overrides imported successfully")
        } catch {
          messageApi.error("Failed to import overrides")
        }
      }
      reader.readAsText(file)
      e.target.value = ""
    },
    [overrides.overridesQuery, messageApi]
  )

  // ---- Reload handler ----
  const onReload = React.useCallback(async () => {
    try {
      await settings.reload()
      messageApi.success("Configuration reloaded from disk")
    } catch {
      messageApi.error("Failed to reload configuration")
    }
  }, [settings, messageApi])

  // ---- Per-user overrides enabled ----
  const perUserOverridesEnabled = policy?.per_user_overrides_enabled ?? policy?.user_overrides_enabled ?? false

  return (
    <div className="space-y-6">
      {/* Section 1: Performance Tuning */}
      <section>
        <h3 className="mb-3 text-base font-semibold text-text-primary">Performance Tuning</h3>
        <div className="grid gap-2 sm:grid-cols-2">
          <PerfField
            label="max_scan_chars"
            value={maxScanChars}
            description="Maximum text length to scan per request"
          />
          <PerfField
            label="max_replacements_per_pattern"
            value={maxReplacementsPerPattern}
            description="Cap on replacements per pattern"
          />
          <PerfField
            label="match_window_chars"
            value={matchWindowChars}
            description="Overlap window for chunked scanning"
          />
          <PerfField
            label="blocklist_write_debounce_ms"
            value={blocklistWriteDebounceMs}
            description="Delay before writing blocklist changes to disk"
          />
        </div>
        <p className="mt-2 text-xs text-text-muted italic">
          These values are read from server config. Changes require a config file edit + reload.
        </p>
      </section>

      {/* Section 2: Export / Import */}
      <section>
        <h3 className="mb-3 text-base font-semibold text-text-primary">Export / Import</h3>

        {/* Blocklist */}
        <div className="mb-4">
          <h4 className="mb-1 text-sm font-medium text-text-secondary">Blocklist</h4>
          <div className="flex flex-wrap gap-2">
            <Tooltip title="Download the current blocklist as a text file">
              <button
                type="button"
                onClick={onDownloadBlocklist}
                className="inline-flex items-center gap-1.5 rounded border border-border bg-bg-secondary px-3 py-1.5 text-sm hover:bg-bg-tertiary"
              >
                <Download size={14} />
                Download blocklist.txt
              </button>
            </Tooltip>
            <Tooltip title="Upload a text file to replace the current blocklist">
              <button
                ref={blocklistUploadButtonRef}
                type="button"
                onClick={() => blocklistFileRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded border border-border bg-bg-secondary px-3 py-1.5 text-sm hover:bg-bg-tertiary"
                aria-label="Replace blocklist from uploaded file"
              >
                <Upload size={14} />
                Upload &amp; replace
              </button>
            </Tooltip>
            <input
              ref={blocklistFileRef}
              type="file"
              accept=".txt"
              className="hidden"
              onChange={onUploadBlocklist}
              data-testid="blocklist-file-input"
              aria-label="Upload blocklist file"
            />
          </div>
        </div>

        {/* User Overrides */}
        <div className="mb-4">
          <h4 className="mb-1 text-sm font-medium text-text-secondary">User Overrides</h4>
          <div className="flex flex-wrap gap-2">
            <Tooltip title="Download all user overrides as JSON">
              <button
                type="button"
                onClick={onDownloadOverrides}
                className="inline-flex items-center gap-1.5 rounded border border-border bg-bg-secondary px-3 py-1.5 text-sm hover:bg-bg-tertiary"
              >
                <Download size={14} />
                Download overrides.json
              </button>
            </Tooltip>
            <Tooltip title="Upload a JSON file to replace all user overrides">
              <button
                type="button"
                onClick={() => overridesFileRef.current?.click()}
                className="inline-flex items-center gap-1.5 rounded border border-border bg-bg-secondary px-3 py-1.5 text-sm hover:bg-bg-tertiary"
                aria-label="Replace user overrides from uploaded file"
              >
                <Upload size={14} />
                Upload &amp; replace
              </button>
            </Tooltip>
            <input
              ref={overridesFileRef}
              type="file"
              accept=".json"
              className="hidden"
              onChange={onUploadOverrides}
              data-testid="overrides-file-input"
              aria-label="Upload user overrides file"
            />
          </div>
        </div>

        <p className="text-xs text-amber-600 dark:text-amber-400">
          Warning: Upload replaces the entire file. A backup is recommended before importing.
        </p>
      </section>

      {/* Section 3: System Operations */}
      <section>
        <h3 className="mb-3 text-base font-semibold text-text-primary">System Operations</h3>
        <div className="space-y-3">
          <div>
            <button
              type="button"
              onClick={onReload}
              className="inline-flex items-center gap-1.5 rounded border border-border bg-bg-secondary px-3 py-1.5 text-sm hover:bg-bg-tertiary"
            >
              <RefreshCw size={14} />
              Reload from disk
            </button>
            <p className="mt-1 text-xs text-text-muted">
              Re-reads config files, blocklist, and overrides from disk.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-text-secondary">Per-user overrides:</span>
            <span
              className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${
                perUserOverridesEnabled
                  ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
                  : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
              }`}
            >
              {perUserOverridesEnabled ? "Enabled" : "Disabled"}
            </span>
          </div>
        </div>
      </section>

      {/* Section 4: Server Configuration (collapsible) */}
      <section>
        <h3 className="mb-3 text-base font-semibold text-text-primary">Server Configuration</h3>
        <details className="rounded border border-border">
          <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-text-secondary hover:bg-bg-secondary">
            View current configuration
          </summary>
          <div className="p-3">
            <textarea
              readOnly
              value={policy ? JSON.stringify(policy, null, 2) : "No policy data available"}
              rows={16}
              className="w-full rounded border border-border bg-bg-secondary p-2 font-mono text-xs text-text-primary"
              aria-label="Effective policy JSON"
            />
          </div>
        </details>
      </section>
    </div>
  )
}

export default AdvancedPanel
