import React from "react"
import { Modal, Tooltip } from "antd"
import { CategoryPicker } from "./components/CategoryPicker"
import { ACTION_OPTIONS } from "./moderation-utils"
import type { ModerationSettingsState } from "./hooks/useModerationSettings"

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface PolicySettingsPanelProps {
  settings: ModerationSettingsState
  messageApi: {
    success: (msg: string) => void
    error: (msg: string) => void
    warning: (msg: string) => void
  }
}

// ---------------------------------------------------------------------------
// Toggle component (Tailwind-only, no antd Switch)
// ---------------------------------------------------------------------------

const Toggle: React.FC<{
  checked: boolean
  onChange?: (next: boolean) => void
  disabled?: boolean
  readOnly?: boolean
  label?: string
}> = ({ checked, onChange, disabled, readOnly, label }) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    aria-label={label}
    disabled={disabled || readOnly}
    onClick={() => !readOnly && onChange?.(!checked)}
    className={`
      relative inline-flex h-6 w-11 shrink-0 items-center rounded-full
      transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
      ${checked ? "bg-blue-600" : "bg-gray-300 dark:bg-gray-600"}
      ${disabled || readOnly ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
    `}
  >
    <span
      className={`
        inline-block h-4 w-4 rounded-full bg-white shadow transition-transform duration-200
        ${checked ? "translate-x-6" : "translate-x-1"}
      `}
    />
  </button>
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const actionLabel = (value: string | undefined) =>
  ACTION_OPTIONS.find((opt) => opt.value === value)?.label ?? value ?? "—"

const actionDescription = (value: string | undefined) =>
  ACTION_OPTIONS.find((opt) => opt.value === value)?.description ?? ""

// ---------------------------------------------------------------------------
// PolicySettingsPanel
// ---------------------------------------------------------------------------

const PolicySettingsPanel: React.FC<PolicySettingsPanelProps> = ({ settings, messageApi }) => {
  const { draft, setDraft, isDirty, save, reset, policyQuery } = settings
  const policy = policyQuery.data || ({} as Record<string, any>)
  const [saving, setSaving] = React.useState(false)

  // ---- Handlers ----

  const handlePiiToggle = (next: boolean) => {
    setDraft((prev) => ({ ...prev, piiEnabled: next }))
  }

  const handleCategoriesChange = (categories: string[]) => {
    setDraft((prev) => ({ ...prev, categoriesEnabled: categories }))
  }

  const handlePersistToggle = (next: boolean) => {
    if (next) {
      Modal.confirm({
        title: "Persist settings to disk?",
        content:
          "This will write the current runtime settings to the server configuration file. " +
          "Changes will survive server restarts. Are you sure?",
        okText: "Yes, persist",
        cancelText: "Cancel",
        onOk: () => setDraft((prev) => ({ ...prev, persist: true }))
      })
    } else {
      setDraft((prev) => ({ ...prev, persist: false }))
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await save()
      messageApi.success("Settings saved")
    } catch (err: any) {
      messageApi.error(err?.message || "Failed to save settings")
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    reset()
  }

  // ---- Render ----

  return (
    <div className="space-y-6 max-w-3xl">
      {/* 1. Master Toggle (read-only) */}
      <section className="rounded-lg border border-border bg-surface/50 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <h3 className="font-semibold text-text">Moderation Enabled</h3>
            <p className="text-sm text-text-muted mt-0.5">
              Whether the moderation engine is active server-wide.
            </p>
          </div>
          <Tooltip title="Controlled by server policy — change via config file or admin API">
            <div>
              <Toggle checked={Boolean(policy.enabled)} readOnly label="Moderation enabled" />
            </div>
          </Tooltip>
        </div>
        <p className="text-xs text-text-muted mt-2 italic">
          Read-only global policy — this value comes from server configuration and is not changed by runtime overrides below.
        </p>
      </section>

      {/* 2. Input / Output Split Controls (read-only) */}
      <section className="rounded-lg border border-border bg-surface/50 p-4">
        <h3 className="font-semibold text-text mb-3">Filtering Actions</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Input column */}
          <div className="space-y-1">
            <h4 className="text-sm font-medium text-text">Filter user messages</h4>
            <Tooltip title={actionDescription(policy.input_action)}>
              <span className="inline-block px-2.5 py-1 text-sm bg-gray-100 dark:bg-gray-800 rounded border border-border">
                {actionLabel(policy.input_action)}
              </span>
            </Tooltip>
          </div>
          {/* Output column */}
          <div className="space-y-1">
            <h4 className="text-sm font-medium text-text">Filter AI responses</h4>
            <Tooltip title={actionDescription(policy.output_action)}>
              <span className="inline-block px-2.5 py-1 text-sm bg-gray-100 dark:bg-gray-800 rounded border border-border">
                {actionLabel(policy.output_action)}
              </span>
            </Tooltip>
            {policy.redact_replacement && (
              <p className="text-xs text-text-muted mt-1">
                Replacement text:{" "}
                <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded text-xs">
                  {policy.redact_replacement}
                </code>
              </p>
            )}
          </div>
        </div>
        <p className="text-xs text-text-muted mt-3 italic">
          Read-only global defaults — user overrides can change the effective action for a selected user in the User Overrides tab.
        </p>
      </section>

      {/* 3. PII Detection Toggle (editable) */}
      <section className="rounded-lg border border-border bg-surface/50 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <h3 className="font-semibold text-text">Personal Data Protection</h3>
            <p className="text-sm text-text-muted mt-0.5">
              Runtime override for built-in personal-data detection. Saving applies immediately; persistence is controlled separately.
            </p>
          </div>
          <Toggle checked={draft.piiEnabled} onChange={handlePiiToggle} label="PII detection" />
        </div>
      </section>

      {/* 4. Content Categories (editable) */}
      <section className="rounded-lg border border-border bg-surface/50 p-4">
        <h3 className="font-semibold text-text mb-1">Content Categories</h3>
        <p className="text-sm text-text-muted mb-3">
          Runtime category override for this server session. Leave empty to use the server default category behavior.
        </p>
        <CategoryPicker value={draft.categoriesEnabled} onChange={handleCategoriesChange} />
      </section>

      {/* 5. Persist Toggle (warning-styled) */}
      <section className="rounded-lg border border-yellow-300 dark:border-yellow-700 bg-yellow-50 dark:bg-yellow-900/20 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="min-w-0">
            <h3 className="font-semibold text-yellow-800 dark:text-yellow-300">
              Persist to Disk
            </h3>
            <p className="text-sm text-yellow-700 dark:text-yellow-400 mt-0.5">
              Write runtime overrides to the server config file so these changes survive restarts.
            </p>
          </div>
          <Toggle checked={draft.persist} onChange={handlePersistToggle} label="Persist to disk" />
        </div>
      </section>

      {/* 6. Save / Reset Buttons */}
      <div className="flex flex-wrap gap-3">
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {saving ? "Saving..." : "Save runtime settings"}
        </button>
        <button
          type="button"
          onClick={handleReset}
          disabled={!isDirty}
          className="px-4 py-2 text-sm font-medium border border-border rounded-lg hover:bg-surface disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Reset changes
        </button>
      </div>

      {/* 7. Active Policy Summary */}
      <section className="rounded-lg border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-4">
        <h3 className="font-semibold text-blue-800 dark:text-blue-300 mb-2">
          Active Policy Summary
        </h3>
        <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1 list-disc list-inside">
          <li>Engine: {policy.enabled ? "Enabled" : "Disabled"}</li>
          <li>Input action: {actionLabel(policy.input_action)}</li>
          <li>Output action: {actionLabel(policy.output_action)}</li>
          {typeof policy.blocklist_count === "number" && (
            <li>Blocklist entries: {policy.blocklist_count}</li>
          )}
          <li>PII detection: {draft.piiEnabled ? "On" : "Off"}</li>
          <li>
            Categories:{" "}
            {draft.categoriesEnabled.length > 0
              ? draft.categoriesEnabled.join(", ")
              : "All (monitoring mode)"}
          </li>
        </ul>
      </section>
    </div>
  )
}

export default PolicySettingsPanel
