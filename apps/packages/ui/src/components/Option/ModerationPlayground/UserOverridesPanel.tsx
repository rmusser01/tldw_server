import React from "react"
import { Modal, Tooltip, Select } from "antd"
import { Trash2, Search } from "lucide-react"
import { CategoryPicker } from "./components/CategoryPicker"
import { PRESET_PROFILES, ACTION_OPTIONS } from "./moderation-utils"
import type { ModerationContextState } from "./hooks/useModerationContext"
import type { UserOverridesState } from "./hooks/useUserOverrides"
import type { ModerationOverrideRule } from "@/services/moderation"

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface UserOverridesPanelProps {
  ctx: ModerationContextState
  overrides: UserOverridesState
  messageApi: {
    success: (msg: string) => void
    error: (msg: string) => void
    warning: (msg: string) => void
  }
}

// ---------------------------------------------------------------------------
// Toggle component (matches PolicySettingsPanel pattern)
// ---------------------------------------------------------------------------

const Toggle: React.FC<{
  checked: boolean
  onChange?: (next: boolean) => void
  disabled?: boolean
  label?: string
}> = ({ checked, onChange, disabled, label }) => (
  <button
    type="button"
    role="switch"
    aria-checked={checked}
    aria-label={label}
    disabled={disabled}
    onClick={() => onChange?.(!checked)}
    className={`
      relative inline-flex h-6 w-11 shrink-0 items-center rounded-full
      transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
      ${checked ? "bg-blue-600" : "bg-gray-300 dark:bg-gray-600"}
      ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
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
// RuleItem — single phrase rule display
// ---------------------------------------------------------------------------

const RuleItem: React.FC<{
  rule: ModerationOverrideRule
  onRemove: (id: string) => void
}> = ({ rule, onRemove }) => (
  <div className="flex items-center justify-between gap-2 px-3 py-2 rounded-md border border-border bg-surface/50">
    <div className="flex items-center gap-2 min-w-0">
      <code className="text-sm truncate" title={rule.pattern}>
        {rule.pattern}
      </code>
      <span
        className={`text-[10px] uppercase font-bold px-1.5 py-0.5 rounded ${
          rule.is_regex
            ? "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300"
            : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
        }`}
      >
        {rule.is_regex ? "regex" : "literal"}
      </span>
      <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
        {rule.phase}
      </span>
    </div>
    <button
      type="button"
      onClick={() => onRemove(rule.id)}
      className="shrink-0 p-1 text-red-500 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
      aria-label={`Remove rule ${rule.pattern}`}
    >
      <Trash2 size={14} />
    </button>
  </div>
)

// ---------------------------------------------------------------------------
// UserOverridesPanel
// ---------------------------------------------------------------------------

const UserOverridesPanel: React.FC<UserOverridesPanelProps> = ({ ctx, overrides, messageApi }) => {
  const { draft, updateDraft, isDirty, bannedRules, notifyRules } = overrides
  const userPickerId = React.useId()
  const userIdErrorId = React.useId()
  const inputActionId = React.useId()
  const outputActionId = React.useId()
  const redactReplacementId = React.useId()
  const phrasePatternId = React.useId()
  const phraseActionLabelId = React.useId()
  const phrasePhaseId = React.useId()
  const tableFilterId = React.useId()
  const phraseActionBlockRef = React.useRef<HTMLButtonElement>(null)
  const phraseActionWarnRef = React.useRef<HTMLButtonElement>(null)

  // Local state for add-phrase form
  const [phrasePattern, setPhrasePattern] = React.useState("")
  const [phraseAction, setPhraseAction] = React.useState<"block" | "warn">("block")
  const [phrasePhase, setPhrasePhase] = React.useState<"input" | "output" | "both">("both")
  const [phraseIsRegex, setPhraseIsRegex] = React.useState(false)

  // Saving state
  const [saving, setSaving] = React.useState(false)

  // Table search filter
  const [tableFilter, setTableFilter] = React.useState("")

  // Row selection for bulk delete
  const [selectedRows, setSelectedRows] = React.useState<Set<string>>(new Set())

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  const handleAddRule = () => {
    const trimmed = phrasePattern.trim()
    if (!trimmed) return
    const added = overrides.addRule({
      pattern: trimmed,
      action: phraseAction,
      is_regex: phraseIsRegex,
      phase: phrasePhase
    })
    if (added) {
      setPhrasePattern("")
      setPhraseIsRegex(false)
    } else {
      messageApi.warning("Duplicate rule — this pattern already exists with the same settings")
    }
  }

  const handlePhraseActionKeyDown = (
    event: React.KeyboardEvent<HTMLButtonElement>,
    currentAction: "block" | "warn"
  ) => {
    let nextAction: "block" | "warn" | null = null
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextAction = currentAction === "block" ? "warn" : "block"
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextAction = currentAction === "warn" ? "block" : "warn"
    } else if (event.key === "Home") {
      nextAction = "block"
    } else if (event.key === "End") {
      nextAction = "warn"
    }
    if (!nextAction) return
    event.preventDefault()
    setPhraseAction(nextAction)
    const nextRef = nextAction === "block" ? phraseActionBlockRef : phraseActionWarnRef
    window.setTimeout(() => nextRef.current?.focus(), 0)
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await overrides.save()
      messageApi.success("Override saved")
    } catch (err: any) {
      messageApi.error(err?.message || "Failed to save override")
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = () => {
    Modal.confirm({
      title: "Delete user override?",
      content: `This will permanently remove the override for "${ctx.activeUserId}" and return that user to the global moderation policy. This cannot be undone from the UI.`,
      okText: "Delete",
      okButtonProps: { danger: true },
      cancelText: "Cancel",
      onOk: async () => {
        try {
          await overrides.remove()
          messageApi.success("Override deleted")
          ctx.clearUser()
        } catch (err: any) {
          messageApi.error(err?.message || "Failed to delete override")
        }
      }
    })
  }

  const handleApplyPreset = async (key: string) => {
    try {
      await overrides.applyPreset(key)
      messageApi.success(`Applied "${PRESET_PROFILES[key]?.label}" preset`)
    } catch (err: any) {
      messageApi.error(err?.message || "Failed to apply preset")
    }
  }

  const handleBulkDelete = async () => {
    const ids = [...selectedRows]
    if (!ids.length) return
    Modal.confirm({
      title: `Delete ${ids.length} override(s)?`,
      content: `This will permanently remove overrides for: ${ids.join(", ")}. Affected users will fall back to the global moderation policy.`,
      okText: "Delete",
      okButtonProps: { danger: true },
      cancelText: "Cancel",
      onOk: async () => {
        try {
          const failed = await overrides.bulkDelete(ids)
          setSelectedRows(new Set())
          if (failed.length > 0) {
            messageApi.warning(`Deleted ${ids.length - failed.length} overrides, ${failed.length} failed`)
          } else {
            messageApi.success(`Deleted ${ids.length} override(s)`)
          }
        } catch (err: any) {
          messageApi.error(err?.message || "Bulk delete failed")
        }
      }
    })
  }

  const handleEditUser = (userId: string) => {
    ctx.setUserIdDraft(userId)
    ctx.setActiveUserId(userId)
  }

  const handleDeleteUser = (userId: string) => {
    Modal.confirm({
      title: "Delete user override?",
      content: `Remove override for "${userId}" and return this user to the global moderation policy? This cannot be undone from the UI.`,
      okText: "Delete",
      okButtonProps: { danger: true },
      cancelText: "Cancel",
      onOk: async () => {
        try {
          await overrides.remove(userId)
          messageApi.success(`Deleted override for "${userId}"`)
        } catch (err: any) {
          messageApi.error(err?.message || "Delete failed")
        }
      }
    })
  }

  // ---------------------------------------------------------------------------
  // Derived: overrides table data
  // ---------------------------------------------------------------------------

  const allOverrides = overrides.overridesQuery.data as
    | { overrides: Record<string, Record<string, any>> }
    | undefined
  const overrideEntries = React.useMemo(() => {
    const raw = allOverrides?.overrides ?? {}
    return Object.entries(raw)
      .filter(([userId]) =>
        !tableFilter || userId.toLowerCase().includes(tableFilter.toLowerCase())
      )
      .sort(([a], [b]) => a.localeCompare(b))
  }, [allOverrides, tableFilter])

  const toggleRow = (userId: string) => {
    setSelectedRows((prev) => {
      const next = new Set(prev)
      if (next.has(userId)) next.delete(userId)
      else next.add(userId)
      return next
    })
  }

  const toggleAllRows = () => {
    if (selectedRows.size === overrideEntries.length) {
      setSelectedRows(new Set())
    } else {
      setSelectedRows(new Set(overrideEntries.map(([id]) => id)))
    }
  }

  const summarizeOverride = (data: Record<string, any>): string => {
    const parts: string[] = []
    parts.push(data.enabled === false ? "disabled" : "active")
    if (data.input_action) parts.push(`in:${data.input_action}`)
    if (data.output_action) parts.push(`out:${data.output_action}`)
    const ruleCount = Array.isArray(data.rules) ? data.rules.length : 0
    if (ruleCount > 0) parts.push(`${ruleCount} rule(s)`)
    return parts.join(" | ")
  }

  const activeDraftSummary = summarizeOverride(draft as Record<string, any>)

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="space-y-6">
      {/* ================================================================== */}
      {/* User Picker Section */}
      {/* ================================================================== */}
      <section className="rounded-lg border border-border bg-surface/50 p-4">
        <h3 className="font-semibold text-text mb-3">User Override Editor</h3>

        {!ctx.activeUserId ? (
          <div className="flex flex-col gap-2 sm:flex-row">
            <label htmlFor={userPickerId} className="sr-only">
              Search or enter user ID
            </label>
            <input
              id={userPickerId}
              type="text"
              placeholder="Search or enter user ID"
              value={ctx.userIdDraft}
              onChange={(e) => ctx.setUserIdDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ctx.loadUser()}
              aria-describedby={overrides.userIdError ? userIdErrorId : undefined}
              className="flex-1 px-3 py-2 text-sm border border-border rounded-md bg-bg text-text placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
            <button
              type="button"
              onClick={ctx.loadUser}
              disabled={!ctx.userIdDraft.trim()}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Load / Create
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 rounded-full">
              Configuring: {ctx.activeUserId}
            </span>
            <button
              type="button"
              onClick={ctx.clearUser}
              className="text-sm text-text-muted hover:text-text underline"
              aria-label={`Clear active user ${ctx.activeUserId}`}
            >
              Clear
            </button>
          </div>
        )}

        {overrides.userIdError && (
          <p id={userIdErrorId} className="mt-2 text-sm text-blue-600 dark:text-blue-400" aria-live="polite">
            {overrides.userIdError}
          </p>
        )}
      </section>

      {/* ================================================================== */}
      {/* Two-Column Editor (visible when activeUserId is set) */}
      {/* ================================================================== */}
      {ctx.activeUserId && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ---- Left column: Settings ---- */}
          <div className="space-y-5">
            {/* Quick Presets */}
            <section className="rounded-lg border border-border bg-surface/50 p-4">
              <h4 className="text-sm font-semibold text-text mb-2">Quick Presets</h4>
              <div className="flex flex-wrap gap-2">
                {Object.entries(PRESET_PROFILES).map(([key, preset]) => (
                  <Tooltip key={key} title={preset.description}>
                    <button
                      type="button"
                      onClick={() => handleApplyPreset(key)}
                      className="px-3 py-1.5 text-sm font-medium border border-border rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/20 hover:border-blue-400 transition-colors"
                    >
                      {preset.label}
                    </button>
                  </Tooltip>
                ))}
              </div>
            </section>

            {/* Toggles */}
            <section className="rounded-lg border border-border bg-surface/50 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-text">Moderation Enabled</span>
                <Toggle
                  checked={Boolean(draft.enabled)}
                  onChange={(v) => updateDraft({ enabled: v })}
                  label="Moderation enabled"
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-text">Filter user messages</span>
                <Toggle
                  checked={Boolean(draft.input_enabled)}
                  onChange={(v) => updateDraft({ input_enabled: v })}
                  label="Filter user messages"
                />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-text">Filter AI responses</span>
                <Toggle
                  checked={Boolean(draft.output_enabled)}
                  onChange={(v) => updateDraft({ output_enabled: v })}
                  label="Filter AI responses"
                />
              </div>
            </section>

            {/* Action Selectors */}
            <section className="rounded-lg border border-border bg-surface/50 p-4 space-y-3">
              <h4 className="text-sm font-semibold text-text">Actions</h4>
              <div>
                <label htmlFor={inputActionId} className="block text-xs text-text-muted mb-1">Input action</label>
                <select
                  id={inputActionId}
                  value={draft.input_action ?? "block"}
                  onChange={(e) => updateDraft({ input_action: e.target.value as any })}
                  className="w-full px-3 py-2 text-sm border border-border rounded-md bg-bg text-text focus:outline-none focus:ring-1 focus:ring-blue-500"
                  aria-label="Input action"
                >
                  {ACTION_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor={outputActionId} className="block text-xs text-text-muted mb-1">Output action</label>
                <select
                  id={outputActionId}
                  value={draft.output_action ?? "redact"}
                  onChange={(e) => updateDraft({ output_action: e.target.value as any })}
                  className="w-full px-3 py-2 text-sm border border-border rounded-md bg-bg text-text focus:outline-none focus:ring-1 focus:ring-blue-500"
                  aria-label="Output action"
                >
                  {ACTION_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor={redactReplacementId} className="block text-xs text-text-muted mb-1">Redact replacement text</label>
                <input
                  id={redactReplacementId}
                  type="text"
                  value={draft.redact_replacement ?? ""}
                  onChange={(e) => updateDraft({ redact_replacement: e.target.value })}
                  placeholder="[REDACTED]"
                  className="w-full px-3 py-2 text-sm border border-border rounded-md bg-bg text-text placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
            </section>

            {/* Categories */}
            <section className="rounded-lg border border-border bg-surface/50 p-4">
              <h4 className="text-sm font-semibold text-text mb-2">Categories</h4>
              <CategoryPicker
                value={Array.isArray(draft.categories_enabled) ? draft.categories_enabled : []}
                onChange={(cats) => updateDraft({ categories_enabled: cats })}
              />
            </section>

            <section className="rounded-lg border border-blue-200 bg-blue-50 p-4 text-blue-800 dark:border-blue-800 dark:bg-blue-900/20 dark:text-blue-300">
              <h4 className="text-sm font-semibold mb-1">Effective After Save</h4>
              <p className="text-sm">{activeDraftSummary}</p>
              <p className="mt-1 text-xs opacity-80">
                This summary applies only to {ctx.activeUserId}; other users keep the global policy unless they have their own override.
              </p>
            </section>
          </div>

          {/* ---- Right column: Phrase Lists ---- */}
          <div className="space-y-5">
            {/* Add Phrase Form */}
            <section className="rounded-lg border border-border bg-surface/50 p-4">
              <h4 className="text-sm font-semibold text-text mb-3">Add Phrase Rule</h4>
              <div className="space-y-3">
                <label htmlFor={phrasePatternId} className="sr-only">
                  Phrase pattern
                </label>
                <input
                  id={phrasePatternId}
                  type="text"
                  placeholder="Pattern (e.g. badword or ^regex$)"
                  value={phrasePattern}
                  onChange={(e) => setPhrasePattern(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAddRule()}
                  className="w-full px-3 py-2 text-sm border border-border rounded-md bg-bg text-text placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-blue-500"
                  data-testid="phrase-pattern-input"
                />
                <div className="flex flex-wrap gap-3 items-center">
                  <span id={phraseActionLabelId} className="sr-only">
                    Phrase action
                  </span>
                  <div
                    className="flex rounded-md border border-border overflow-hidden"
                    role="radiogroup"
                    aria-labelledby={phraseActionLabelId}
                  >
                    <button
                      ref={phraseActionBlockRef}
                      type="button"
                      role="radio"
                      aria-checked={phraseAction === "block"}
                      tabIndex={phraseAction === "block" ? 0 : -1}
                      onClick={() => setPhraseAction("block")}
                      onKeyDown={(event) => handlePhraseActionKeyDown(event, "block")}
                      className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                        phraseAction === "block"
                          ? "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                          : "bg-surface text-text-muted hover:bg-gray-50 dark:hover:bg-gray-800"
                      }`}
                    >
                      Ban
                    </button>
                    <button
                      ref={phraseActionWarnRef}
                      type="button"
                      role="radio"
                      aria-checked={phraseAction === "warn"}
                      tabIndex={phraseAction === "warn" ? 0 : -1}
                      onClick={() => setPhraseAction("warn")}
                      onKeyDown={(event) => handlePhraseActionKeyDown(event, "warn")}
                      className={`px-3 py-1.5 text-xs font-medium border-l border-border transition-colors ${
                        phraseAction === "warn"
                          ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300"
                          : "bg-surface text-text-muted hover:bg-gray-50 dark:hover:bg-gray-800"
                      }`}
                    >
                      Notify
                    </button>
                  </div>
                  <label htmlFor={phrasePhaseId} className="sr-only">
                    Phrase phase
                  </label>
                  <select
                    id={phrasePhaseId}
                    value={phrasePhase}
                    onChange={(e) => setPhrasePhase(e.target.value as any)}
                    className="px-2 py-1.5 text-xs border border-border rounded-md bg-bg text-text focus:outline-none focus:ring-1 focus:ring-blue-500"
                    aria-label="Phrase phase"
                  >
                    <option value="input">Input</option>
                    <option value="output">Output</option>
                    <option value="both">Both</option>
                  </select>
                  <label className="flex items-center gap-1.5 text-xs text-text">
                    <input
                      type="checkbox"
                      checked={phraseIsRegex}
                      onChange={(e) => setPhraseIsRegex(e.target.checked)}
                      className="rounded border-gray-300"
                    />
                    Regex
                  </label>
                </div>
                <button
                  type="button"
                  onClick={handleAddRule}
                  disabled={!phrasePattern.trim()}
                  className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Add rule
                </button>
              </div>
            </section>

            {/* Banned Phrases */}
            <section className="rounded-lg border border-border bg-surface/50 p-4">
              <h4 className="text-sm font-semibold text-text mb-2">
                Banned Phrases
                <span className="ml-2 text-xs text-text-muted font-normal">({bannedRules.length})</span>
              </h4>
              {bannedRules.length === 0 ? (
                <p className="text-sm text-text-muted italic">No banned phrases configured.</p>
              ) : (
                <div className="space-y-2">
                  {bannedRules.map((rule) => (
                    <RuleItem key={rule.id} rule={rule} onRemove={overrides.removeRule} />
                  ))}
                </div>
              )}
            </section>

            {/* Notify Phrases */}
            <section className="rounded-lg border border-border bg-surface/50 p-4">
              <h4 className="text-sm font-semibold text-text mb-2">
                Notify Phrases
                <span className="ml-2 text-xs text-text-muted font-normal">({notifyRules.length})</span>
              </h4>
              {notifyRules.length === 0 ? (
                <p className="text-sm text-text-muted italic">No notify phrases configured.</p>
              ) : (
                <div className="space-y-2">
                  {notifyRules.map((rule) => (
                    <RuleItem key={rule.id} rule={rule} onRemove={overrides.removeRule} />
                  ))}
                </div>
              )}
            </section>
          </div>
        </div>
      )}

      {/* ================================================================== */}
      {/* Action Buttons (visible when activeUserId is set) */}
      {/* ================================================================== */}
      {ctx.activeUserId && (
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={handleSave}
            disabled={saving}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {saving ? "Saving..." : "Save override"}
          </button>
          <button
            type="button"
            onClick={overrides.reset}
            disabled={!isDirty}
            className="px-4 py-2 text-sm font-medium border border-border rounded-lg hover:bg-surface disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Reset changes
          </button>
          <button
            type="button"
            onClick={handleDelete}
            className="px-4 py-2 text-sm font-medium text-red-600 border border-red-300 dark:border-red-700 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
          >
            Delete override
          </button>
        </div>
      )}

      {/* ================================================================== */}
      {/* All User Overrides Table (always visible) */}
      {/* ================================================================== */}
      <section className="rounded-lg border border-border bg-surface/50 p-4">
        <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="font-semibold text-text">All User Overrides</h3>
          <div className="flex flex-wrap items-center gap-2">
            {selectedRows.size > 0 && (
              <button
                type="button"
                onClick={handleBulkDelete}
                className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium text-red-600 border border-red-300 dark:border-red-700 rounded-md hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
              >
                <Trash2 size={12} />
                Delete {selectedRows.size} selected
              </button>
            )}
            <div className="relative">
              <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
              <label htmlFor={tableFilterId} className="sr-only">
                Filter user overrides
              </label>
              <input
                id={tableFilterId}
                type="text"
                placeholder="Filter users..."
                value={tableFilter}
                onChange={(e) => setTableFilter(e.target.value)}
                className="pl-8 pr-3 py-1.5 text-sm border border-border rounded-md bg-bg text-text placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-blue-500"
                data-testid="overrides-table-filter"
              />
            </div>
          </div>
        </div>

        {overrideEntries.length === 0 ? (
          <p className="text-sm text-text-muted italic py-4 text-center">
            {tableFilter ? "No overrides match your filter." : "No user overrides configured yet."}
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-[680px] w-full text-sm" data-testid="overrides-table">
              <thead>
                <tr className="border-b border-border text-left text-text-muted">
                  <th className="py-2 pr-2 w-8">
                    <input
                      type="checkbox"
                      checked={selectedRows.size === overrideEntries.length && overrideEntries.length > 0}
                      onChange={toggleAllRows}
                      className="rounded border-gray-300"
                      aria-label="Select all"
                    />
                  </th>
                  <th className="py-2 px-2">User ID</th>
                  <th className="py-2 px-2">Summary</th>
                  <th className="py-2 px-2 w-24">Actions</th>
                </tr>
              </thead>
              <tbody>
                {overrideEntries.map(([userId, data]) => (
                  <tr key={userId} className="border-b border-border last:border-0 hover:bg-surface/80">
                    <td className="py-2 pr-2">
                      <input
                        type="checkbox"
                        checked={selectedRows.has(userId)}
                        onChange={() => toggleRow(userId)}
                        className="rounded border-gray-300"
                        aria-label={`Select ${userId}`}
                      />
                    </td>
                    <td className="py-2 px-2 font-mono text-xs">{userId}</td>
                    <td className="py-2 px-2 text-text-muted text-xs">{summarizeOverride(data)}</td>
                    <td className="py-2 px-2">
                      <div className="flex gap-1">
                        <button
                          type="button"
                          onClick={() => handleEditUser(userId)}
                          className="px-2 py-1 text-xs text-blue-600 hover:underline"
                        >
                          Edit
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeleteUser(userId)}
                          className="px-2 py-1 text-xs text-red-500 hover:underline"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}

export default UserOverridesPanel
