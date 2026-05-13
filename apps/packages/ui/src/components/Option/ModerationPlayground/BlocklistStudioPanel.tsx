import React from "react"
import { Select, Modal, Tooltip } from "antd"
import { Trash2 } from "lucide-react"
import { BlocklistSyntaxRef } from "./components/BlocklistSyntaxRef"
import {
  CATEGORY_SUGGESTIONS,
  ACTION_OPTIONS,
  normalizeManagedBlocklistRows,
  type NormalizedManagedBlocklistRow
} from "./moderation-utils"
import type { RawReplacePreview, useBlocklist } from "./hooks/useBlocklist"
import type { BlocklistLintItem } from "@/services/moderation"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BlocklistStudioPanelProps {
  blocklist: ReturnType<typeof useBlocklist>
  messageApi: { success: (msg: string) => void; error: (msg: string) => void; warning: (msg: string) => void }
}

type SubTab = "managed" | "raw"

const SUB_TABS: { key: SubTab; label: string }[] = [
  { key: "managed", label: "Managed Rules" },
  { key: "raw", label: "Raw Editor" }
]

const PAGE_SIZE = 10

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Compose a blocklist grammar line from structured fields */
function composeLine(
  pattern: string,
  action: string,
  replacement: string,
  categories: string[]
): string {
  let line = pattern.trim()
  if (!line) return ""

  // action suffix
  if (action === "redact" && replacement) {
    line += ` -> redact:${replacement}`
  } else if (action && action !== "block") {
    line += ` -> ${action}`
  } else if (action === "block") {
    line += " -> block"
  }

  // categories
  if (categories.length > 0) {
    line += ` #${categories.join(",")}`
  }

  return line
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const LintResultsTable: React.FC<{ items: BlocklistLintItem[] }> = ({ items }) => {
  if (items.length === 0) return null
  return (
    <div className="border border-border rounded-lg overflow-x-auto">
      <table className="min-w-[640px] w-full text-sm" data-testid="lint-results-table">
        <thead>
          <tr className="text-left text-text-muted bg-surface/50">
            <th className="px-3 py-2 font-medium">#</th>
            <th className="px-3 py-2 font-medium">Line</th>
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="px-3 py-2 font-medium">Details</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, idx) => (
            <tr key={idx} className="border-t border-border">
              <td className="px-3 py-2 text-text-muted">{item.index}</td>
              <td className="px-3 py-2 font-mono text-xs max-w-[200px] truncate">{item.line}</td>
              <td className="px-3 py-2">
                <span
                  className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                    item.ok
                      ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                      : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                  }`}
                >
                  {item.ok ? "Valid" : "Error"}
                </span>
              </td>
              <td className="px-3 py-2 text-xs text-text-muted">
                {item.error || item.warning || (item.ok ? `${item.pattern_type ?? "literal"} / ${item.action ?? "block"}` : "")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const pluralize = (count: number, singular: string, plural = `${singular}s`) =>
  `${count} ${count === 1 ? singular : plural}`

const RawReplacePreviewSummary: React.FC<{ preview: RawReplacePreview }> = ({ preview }) => (
  <div className="space-y-3 text-sm">
    <p>
      This will replace the current blocklist after linting. Review the summary before continuing.
    </p>
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

const TYPE_LABELS: Record<NormalizedManagedBlocklistRow["rowKind"], string> = {
  literal: "literal",
  regex: "regex",
  comment: "comment",
  empty: "blank"
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const BlocklistStudioPanel: React.FC<BlocklistStudioPanelProps> = ({ blocklist, messageApi }) => {
  const [subTab, setSubTab] = React.useState<SubTab>("managed")
  const [page, setPage] = React.useState(1)
  const [searchQuery, setSearchQuery] = React.useState("")
  const [showInactiveRows, setShowInactiveRows] = React.useState(false)
  const [patternTypeFilter, setPatternTypeFilter] = React.useState<"all" | NormalizedManagedBlocklistRow["rowKind"]>("all")
  const [actionFilter, setActionFilter] = React.useState<"all" | "block" | "redact" | "warn">("all")
  const [categoryFilter, setCategoryFilter] = React.useState("all")
  const [sortKey, setSortKey] = React.useState<"line" | "action" | "category" | "pattern_type">("line")
  const [managedUndo, setManagedUndo] = React.useState<{ line: string } | null>(null)

  // Add-rule form state
  const [pattern, setPattern] = React.useState("")
  const [action, setAction] = React.useState("block")
  const [replacement, setReplacement] = React.useState("")
  const [categories, setCategories] = React.useState<string[]>([])
  const [inlineLint, setInlineLint] = React.useState<BlocklistLintItem[] | null>(null)
  const tabPanelBaseId = React.useId()
  const patternInputId = React.useId()
  const actionSelectId = React.useId()
  const replacementInputId = React.useId()
  const categoriesSelectId = React.useId()
  const showInactiveRowsId = React.useId()
  const searchInputId = React.useId()
  const patternTypeFilterId = React.useId()
  const actionFilterId = React.useId()
  const categoryFilterId = React.useId()
  const sortSelectId = React.useId()
  const rawEditorId = React.useId()
  const subTabRefs = React.useRef<Record<SubTab, HTMLButtonElement | null>>({
    managed: null,
    raw: null
  })
  const rawReplaceButtonRef = React.useRef<HTMLButtonElement>(null)
  const undoManagedButtonRef = React.useRef<HTMLButtonElement>(null)
  const undoRawButtonRef = React.useRef<HTMLButtonElement>(null)

  // Auto-load managed rules on mount
  React.useEffect(() => {
    void blocklist.loadManaged().catch((err) => {
      console.error("[ModerationPlayground] Failed to load managed blocklist:", err)
      messageApi.error("Failed to load managed blocklist")
    })
  }, [blocklist.loadManaged]) // stable callback from useCallback([], [])

  const composed = composeLine(pattern, action, replacement, categories)

  const getSubTabId = (tab: SubTab) => `${tabPanelBaseId}-${tab}-tab`
  const getSubTabPanelId = (tab: SubTab) => `${tabPanelBaseId}-${tab}-panel`

  const activateSubTab = (tab: SubTab) => {
    setSubTab(tab)
    subTabRefs.current[tab]?.focus()
  }

  const handleSubTabKeyDown = (
    event: React.KeyboardEvent<HTMLButtonElement>,
    currentTab: SubTab
  ) => {
    const currentIndex = SUB_TABS.findIndex((tab) => tab.key === currentTab)
    let nextIndex: number | null = null
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = (currentIndex + 1) % SUB_TABS.length
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex = (currentIndex - 1 + SUB_TABS.length) % SUB_TABS.length
    } else if (event.key === "Home") {
      nextIndex = 0
    } else if (event.key === "End") {
      nextIndex = SUB_TABS.length - 1
    }
    if (nextIndex == null) return
    event.preventDefault()
    activateSubTab(SUB_TABS[nextIndex].key)
  }

  React.useEffect(() => {
    if (managedUndo) {
      window.setTimeout(() => undoManagedButtonRef.current?.focus(), 0)
    }
  }, [managedUndo])

  React.useEffect(() => {
    if (blocklist.rawReplaceUndo) {
      window.setTimeout(() => undoRawButtonRef.current?.focus(), 0)
    }
  }, [blocklist.rawReplaceUndo])

  const handleValidate = async () => {
    if (!composed) {
      messageApi.warning("Enter a pattern first")
      return
    }
    try {
      const lint = await blocklist.lintLine(composed)
      setInlineLint(lint.items)
    } catch (err: any) {
      messageApi.error(err?.message || "Validation failed")
    }
  }

  // Keep inline lint in sync with managedLint
  React.useEffect(() => {
    if (blocklist.managedLint) {
      setInlineLint(blocklist.managedLint.items)
    }
  }, [blocklist.managedLint])

  const handleAddRule = async () => {
    if (!composed) {
      messageApi.warning("Enter a pattern first")
      return
    }
    try {
      await blocklist.appendLine(composed)
      messageApi.success("Rule added")
      // Reset form
      setPattern("")
      setAction("block")
      setReplacement("")
      setCategories([])
      setInlineLint(null)
    } catch (err: any) {
      messageApi.error(err?.message || "Failed to add rule")
    }
  }

  const handleDelete = (itemId: number, trigger?: HTMLButtonElement | null) => {
    const rowToDelete = normalizedRows.find((row) => row.id === itemId)
    Modal.confirm({
      title: "Delete rule?",
      content: "This removes the rule from the managed blocklist. You can undo this during the current session.",
      okText: "Delete",
      okButtonProps: { danger: true },
      cancelText: "Cancel",
      onCancel: () => window.setTimeout(() => trigger?.focus(), 0),
      onOk: async () => {
        try {
          await blocklist.deleteManaged(itemId)
          setManagedUndo(rowToDelete ? { line: rowToDelete.line } : null)
          messageApi.success("Rule deleted")
        } catch (err: any) {
          messageApi.error(err?.message || "Delete failed")
        }
      }
    })
  }

  const handleUndoManagedDelete = async () => {
    if (!managedUndo) return
    try {
      await blocklist.appendLine(managedUndo.line)
      setManagedUndo(null)
      messageApi.success("Deleted rule restored")
    } catch (err: any) {
      messageApi.error(err?.message || "Undo failed")
    }
  }

  const openRawReplaceConfirm = (preview: RawReplacePreview, title: string, successMessage: string) => {
    const returnFocus = () => window.setTimeout(() => rawReplaceButtonRef.current?.focus(), 0)
    Modal.confirm({
      title,
      content: <RawReplacePreviewSummary preview={preview} />,
      okText: "Replace blocklist",
      cancelText: "Cancel",
      okButtonProps: { danger: true, disabled: preview.lint.invalid_count > 0 },
      onCancel: () => {
        blocklist.cancelRawReplace()
        returnFocus()
      },
      onOk: async () => {
        await blocklist.confirmRawReplace()
        messageApi.success(successMessage)
      }
    })
  }

  const handlePreviewRawReplace = async () => {
    try {
      const preview = await blocklist.previewRawReplace(blocklist.rawText)
      openRawReplaceConfirm(preview, "Confirm blocklist replacement", "Blocklist saved")
    } catch (err: any) {
      messageApi.error(err?.message || "Preview failed")
    }
  }

  const handleUndoRawReplace = async () => {
    try {
      await blocklist.undoRawReplace()
      messageApi.success("Previous blocklist restored")
    } catch (err: any) {
      messageApi.error(err?.message || "Undo failed")
    }
  }

  const normalizedRows = React.useMemo(
    () => normalizeManagedBlocklistRows(blocklist.managedItems),
    [blocklist.managedItems]
  )
  const activeCount = normalizedRows.filter((row) => row.isActive).length
  const hiddenInactiveCount = normalizedRows.length - activeCount
  const categoryOptions = React.useMemo(() => {
    const categories = new Set<string>()
    normalizedRows.forEach((row) => row.categories.forEach((category) => categories.add(category)))
    return [...categories].sort((left, right) => left.localeCompare(right))
  }, [normalizedRows])
  const filteredRows = React.useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    const rows = normalizedRows.filter((row) => {
      if (!showInactiveRows && !row.isActive) return false
      if (patternTypeFilter !== "all" && row.rowKind !== patternTypeFilter) return false
      if (actionFilter !== "all" && row.action !== actionFilter) return false
      if (categoryFilter !== "all" && !row.categories.includes(categoryFilter)) return false
      if (!query) return true
      return [
        row.line,
        row.pattern,
        row.action,
        row.rowKind,
        row.statusLabel,
        ...row.categories
      ].some((value) => String(value).toLowerCase().includes(query))
    })

    return [...rows].sort((left, right) => {
      if (sortKey === "action") return left.action.localeCompare(right.action) || left.line.localeCompare(right.line)
      if (sortKey === "category") {
        const leftCategory = left.categories[0] ?? ""
        const rightCategory = right.categories[0] ?? ""
        return leftCategory.localeCompare(rightCategory) || left.line.localeCompare(right.line)
      }
      if (sortKey === "pattern_type") return left.rowKind.localeCompare(right.rowKind) || left.line.localeCompare(right.line)
      return left.line.localeCompare(right.line)
    })
  }, [normalizedRows, showInactiveRows, patternTypeFilter, actionFilter, categoryFilter, searchQuery, sortKey])

  React.useEffect(() => {
    setPage(1)
  }, [showInactiveRows, patternTypeFilter, actionFilter, categoryFilter, searchQuery, sortKey])

  // Pagination
  const totalItems = filteredRows.length
  const totalPages = Math.max(1, Math.ceil(totalItems / PAGE_SIZE))
  React.useEffect(() => {
    setPage((current) => Math.max(1, Math.min(current, totalPages)))
  }, [totalPages])
  const pagedItems = filteredRows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  // ---------------------------------------------------------------------------
  // Managed Rules view
  // ---------------------------------------------------------------------------

  const renderManaged = () => (
    <div className="space-y-6">
      {/* Add rule form */}
      <div className="border border-border rounded-lg p-4 space-y-4">
        <h4 className="text-sm font-semibold">Add Rule</h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* Pattern */}
          <div className="sm:col-span-2">
            <label htmlFor={patternInputId} className="block text-xs text-text-muted mb-1">Pattern</label>
            <input
              id={patternInputId}
              type="text"
              value={pattern}
              onChange={(e) => setPattern(e.target.value)}
              placeholder='Enter pattern or /regex/'
              className="w-full px-3 py-2 border border-border rounded-lg bg-bg text-text text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              data-testid="pattern-input"
            />
          </div>

          {/* Action */}
          <div>
            <label htmlFor={actionSelectId} className="block text-xs text-text-muted mb-1">Action</label>
            <select
              id={actionSelectId}
              value={action}
              onChange={(e) => setAction(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded-lg bg-bg text-text text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              data-testid="action-select"
            >
              {ACTION_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Replacement (only for redact) */}
          {action === "redact" && (
            <div>
              <label htmlFor={replacementInputId} className="block text-xs text-text-muted mb-1">Replacement</label>
              <input
                id={replacementInputId}
                type="text"
                value={replacement}
                onChange={(e) => setReplacement(e.target.value)}
                placeholder="[REDACTED]"
                className="w-full px-3 py-2 border border-border rounded-lg bg-bg text-text text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}

          {/* Categories */}
          <div className={action === "redact" ? "sm:col-span-2" : ""}>
            <label htmlFor={categoriesSelectId} className="block text-xs text-text-muted mb-1">Categories</label>
            <Select
              id={categoriesSelectId}
              aria-label="Rule categories"
              mode="tags"
              value={categories}
              onChange={setCategories}
              placeholder="Select or type categories"
              className="w-full"
              options={CATEGORY_SUGGESTIONS.map((c) => ({ value: c.value, label: c.label }))}
              data-testid="categories-select"
            />
          </div>

        </div>

        {/* Composed preview */}
        {composed && (
          <div className="text-xs text-text-muted font-mono bg-surface/30 rounded px-3 py-2">
            Preview: <code>{composed}</code>
          </div>
        )}

        {/* Buttons */}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleValidate}
            disabled={blocklist.loading || !composed}
            className="px-4 py-2 text-sm font-medium rounded-lg border border-border text-text hover:bg-surface/50 transition-colors disabled:opacity-50"
          >
            Validate
          </button>
          <button
            type="button"
            onClick={handleAddRule}
            disabled={blocklist.loading || !composed}
            className="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            Add rule
          </button>
        </div>

        {/* Inline lint results */}
        {inlineLint && inlineLint.length > 0 && (
          <div className="mt-2" aria-live="polite">
            <LintResultsTable items={inlineLint} />
          </div>
        )}
      </div>

      {/* Managed lint results */}
      {blocklist.managedLint && blocklist.managedLint.items.length > 0 && (
        <div aria-live="polite">
          <h4 className="text-sm font-semibold mb-2">Lint Results</h4>
          <LintResultsTable items={blocklist.managedLint.items} />
        </div>
      )}

      {/* Rules table */}
      <div>
        <div className="mb-3 flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h4 className="text-sm font-semibold">Current Rules</h4>
              <p className="text-xs text-text-muted">
                {pluralize(activeCount, "active rule")} total
                {hiddenInactiveCount > 0 && !showInactiveRows
                  ? `, ${pluralize(hiddenInactiveCount, "non-active row")} hidden`
                  : ""}
              </p>
            </div>
            <label htmlFor={showInactiveRowsId} className="inline-flex items-center gap-2 text-xs text-text-muted">
              <input
                id={showInactiveRowsId}
                type="checkbox"
                checked={showInactiveRows}
                onChange={(event) => setShowInactiveRows(event.target.checked)}
              />
              Show comments and blanks
            </label>
          </div>
          {managedUndo && (
            <div className="flex items-center justify-between gap-3 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-800 dark:border-blue-800 dark:bg-blue-900/20 dark:text-blue-300">
              <span>Deleted rule available to restore during this session.</span>
              <button
                ref={undoManagedButtonRef}
                type="button"
                onClick={handleUndoManagedDelete}
                className="rounded border border-blue-300 px-2 py-1 text-xs font-medium hover:bg-blue-100 dark:border-blue-700 dark:hover:bg-blue-900/40"
                aria-label="Undo deleted rule"
              >
                Undo delete
              </button>
            </div>
          )}
          <div className="grid grid-cols-1 gap-2 md:grid-cols-5">
            <label htmlFor={searchInputId} className="sr-only">
              Rule search
            </label>
            <input
              id={searchInputId}
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search rules"
              className="rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text focus:outline-none focus:ring-2 focus:ring-blue-500 md:col-span-2"
            />
            <label htmlFor={patternTypeFilterId} className="sr-only">
              Filter by pattern type
            </label>
            <select
              id={patternTypeFilterId}
              value={patternTypeFilter}
              onChange={(event) => setPatternTypeFilter(event.target.value as typeof patternTypeFilter)}
              className="rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text"
              aria-label="Filter by pattern type"
            >
              <option value="all">All types</option>
              <option value="literal">Literal</option>
              <option value="regex">Regex</option>
              <option value="comment">Comment</option>
              <option value="empty">Blank</option>
            </select>
            <label htmlFor={actionFilterId} className="sr-only">
              Filter by action
            </label>
            <select
              id={actionFilterId}
              value={actionFilter}
              onChange={(event) => setActionFilter(event.target.value as typeof actionFilter)}
              className="rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text"
              aria-label="Filter by action"
            >
              <option value="all">All actions</option>
              <option value="block">Block</option>
              <option value="redact">Redact</option>
              <option value="warn">Warn</option>
            </select>
            <label htmlFor={categoryFilterId} className="sr-only">
              Filter by category
            </label>
            <select
              id={categoryFilterId}
              value={categoryFilter}
              onChange={(event) => setCategoryFilter(event.target.value)}
              className="rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text"
              aria-label="Filter by category"
            >
              <option value="all">All categories</option>
              {categoryOptions.map((category) => (
                <option key={category} value={category}>{category}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-text-muted" htmlFor={sortSelectId}>Sort</label>
            <select
              id={sortSelectId}
              value={sortKey}
              onChange={(event) => setSortKey(event.target.value as typeof sortKey)}
              className="rounded border border-border bg-bg px-2 py-1 text-xs text-text"
            >
              <option value="line">Line</option>
              <option value="action">Action</option>
              <option value="category">Category</option>
              <option value="pattern_type">Pattern type</option>
            </select>
          </div>
        </div>
        {normalizedRows.length === 0 ? (
          <div className="border border-border rounded-lg p-8 text-center text-text-muted text-sm" data-testid="empty-rules">
            No rules loaded. Rules will appear here after loading the managed blocklist.
          </div>
        ) : totalItems === 0 ? (
          <div className="border border-border rounded-lg p-8 text-center text-text-muted text-sm" data-testid="empty-rules">
            No rules match the current filters.
          </div>
        ) : (
          <>
            <div className="border border-border rounded-lg overflow-x-auto">
              <table className="min-w-[760px] w-full text-sm" data-testid="rules-table">
                <thead>
                  <tr className="text-left text-text-muted bg-surface/50">
                    <th className="px-3 py-2 font-medium w-10">#</th>
                    <th className="px-3 py-2 font-medium">Pattern</th>
                    <th className="px-3 py-2 font-medium">Type</th>
                    <th className="px-3 py-2 font-medium">Action</th>
                    <th className="px-3 py-2 font-medium">Categories</th>
                    <th className="px-3 py-2 font-medium w-10"></th>
                  </tr>
                </thead>
                <tbody>
                  {pagedItems.map((item) => {
                    const actionColors: Record<string, string> = {
                      block: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
                      redact: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
                      warn: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300"
                    }
                    const typeColors: Record<NormalizedManagedBlocklistRow["rowKind"], string> = {
                      literal: "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300",
                      regex: "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300",
                      comment: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
                      empty: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
                    }
                    const patternLabel = item.rowKind === "empty" ? "(blank line)" : item.pattern || item.line

                    return (
                      <tr key={item.id} className={`border-t border-border ${item.isActive ? "" : "bg-surface/30"}`}>
                        <td className="px-3 py-2 text-text-muted">{item.id}</td>
                        <td className="px-3 py-2 font-mono text-xs max-w-[200px] truncate">
                          <Tooltip title={item.line}>{patternLabel}</Tooltip>
                          {!item.isValid && item.error && (
                            <div className="mt-1 text-[11px] text-red-600 dark:text-red-400">{item.error}</div>
                          )}
                          {item.warning && item.isValid && (
                            <div className="mt-1 text-[11px] text-text-muted">{item.warning}</div>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${typeColors[item.rowKind]}`}
                          >
                            {TYPE_LABELS[item.rowKind]}
                          </span>
                        </td>
                        <td className="px-3 py-2">
                          {item.isActive ? (
                            <span
                              className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${actionColors[item.action] ?? ""}`}
                            >
                              {item.action}
                            </span>
                          ) : (
                            <span className="text-xs text-text-muted">{item.statusLabel}</span>
                          )}
                        </td>
                        <td className="px-3 py-2">
                          <div className="flex flex-wrap gap-1">
                            {item.categories.map((cat) => (
                              <span
                                key={cat}
                                className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-surface text-text-muted"
                              >
                                {cat}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td className="px-3 py-2">
                          {item.isActive && (
                            <button
                              type="button"
                              onClick={(event) => handleDelete(item.id, event.currentTarget)}
                              className="p-1 text-text-muted hover:text-red-500 transition-colors"
                              aria-label={`Delete rule ${item.id}`}
                            >
                              <Trash2 size={14} />
                            </button>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between mt-3">
                <span className="text-xs text-text-muted">
                  Page {page} of {totalPages} ({totalItems} rules)
                </span>
                <div className="flex gap-1">
                  <button
                    type="button"
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page <= 1}
                    className="px-3 py-1 text-xs rounded border border-border hover:bg-surface/50 disabled:opacity-50"
                  >
                    Prev
                  </button>
                  <button
                    type="button"
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    disabled={page >= totalPages}
                    className="px-3 py-1 text-xs rounded border border-border hover:bg-surface/50 disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}

            {/* Version footer */}
            {blocklist.managedVersion && (
              <div className="mt-2 text-xs text-text-muted">
                Version: <code>{blocklist.managedVersion.slice(0, 12)}</code>
              </div>
            )}
          </>
        )}
      </div>

      {/* Syntax reference */}
      <BlocklistSyntaxRef />
    </div>
  )

  // ---------------------------------------------------------------------------
  // Raw Editor view
  // ---------------------------------------------------------------------------

  const renderRaw = () => (
    <div className="space-y-4">
      {/* Warning banner */}
      <div className="p-3 border border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg text-sm text-yellow-800 dark:text-yellow-300">
        Raw file editing replaces all existing rules. Use with caution.
      </div>

      {/* Buttons */}
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={async () => {
            try {
              await blocklist.loadRaw()
              messageApi.success("Blocklist loaded")
            } catch (err: any) {
              messageApi.error(err?.message || "Load failed")
            }
          }}
          disabled={blocklist.loading}
          className="px-4 py-2 text-sm font-medium rounded-lg border border-border text-text hover:bg-surface/50 transition-colors disabled:opacity-50"
        >
          Load blocklist
        </button>
        <button
          type="button"
          onClick={async () => {
            try {
              await blocklist.lintRaw()
              messageApi.success("Validation complete")
            } catch (err: any) {
              messageApi.error(err?.message || "Validation failed")
            }
          }}
          disabled={blocklist.loading}
          className="px-4 py-2 text-sm font-medium rounded-lg border border-border text-text hover:bg-surface/50 transition-colors disabled:opacity-50"
        >
          Validate all
        </button>
        <button
          ref={rawReplaceButtonRef}
          type="button"
          onClick={handlePreviewRawReplace}
          disabled={blocklist.loading}
          className="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors disabled:opacity-50"
        >
          Save / Replace
        </button>
      </div>

      {blocklist.rawReplaceUndo && (
        <div className="flex items-center justify-between gap-3 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-800 dark:border-blue-800 dark:bg-blue-900/20 dark:text-blue-300">
          <span>Previous blocklist available to restore during this session.</span>
          <button
            ref={undoRawButtonRef}
            type="button"
            onClick={handleUndoRawReplace}
            className="rounded border border-blue-300 px-2 py-1 text-xs font-medium hover:bg-blue-100 dark:border-blue-700 dark:hover:bg-blue-900/40"
            aria-label="Undo raw replace"
          >
            Undo replace
          </button>
        </div>
      )}

      {/* TextArea */}
      <label htmlFor={rawEditorId} className="sr-only">
        Raw blocklist editor
      </label>
      <textarea
        id={rawEditorId}
        value={blocklist.rawText}
        onChange={(e) => blocklist.setRawText(e.target.value)}
        rows={12}
        className="w-full px-3 py-2 border border-border rounded-lg bg-bg text-text font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
        placeholder="# Enter blocklist rules, one per line..."
        data-testid="raw-editor"
      />

      {/* Lint results */}
      {blocklist.rawLint && blocklist.rawLint.items.length > 0 && (
        <div aria-live="polite">
          <h4 className="text-sm font-semibold mb-2">
            Lint Results ({blocklist.rawLint.valid_count} valid, {blocklist.rawLint.invalid_count} invalid)
          </h4>
          <LintResultsTable items={blocklist.rawLint.items} />
        </div>
      )}

      {/* Syntax reference */}
      <BlocklistSyntaxRef />
    </div>
  )

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="space-y-6">
      {/* Sub-tab bar */}
      <div className="border-b border-border">
        <div className="flex overflow-x-auto -mb-px" role="tablist" aria-label="Blocklist editor views">
          {SUB_TABS.map((tab) => (
            <button
              key={tab.key}
              id={getSubTabId(tab.key)}
              ref={(node) => {
                subTabRefs.current[tab.key] = node
              }}
              role="tab"
              aria-selected={subTab === tab.key}
              aria-controls={getSubTabPanelId(tab.key)}
              tabIndex={subTab === tab.key ? 0 : -1}
              onClick={() => setSubTab(tab.key)}
              onKeyDown={(event) => handleSubTabKeyDown(event, tab.key)}
              className={`
                px-4 py-2.5 text-sm font-medium whitespace-nowrap border-b-2 transition-colors
                ${
                  subTab === tab.key
                    ? "border-blue-500 text-blue-600 dark:text-blue-400"
                    : "border-transparent text-text-muted hover:text-text hover:border-gray-300"
                }
              `}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div
        id={getSubTabPanelId(subTab)}
        role="tabpanel"
        aria-labelledby={getSubTabId(subTab)}
      >
        {subTab === "managed" ? renderManaged() : renderRaw()}
      </div>
    </div>
  )
}

export default BlocklistStudioPanel
