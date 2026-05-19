import React from "react"
import { Button, Input, InputNumber, Select, Switch, Tag, Tooltip, message } from "antd"
import { ArrowDown, ArrowUp, Code2, Plus, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { FilterAction, FilterType, WatchlistFilter } from "@/types/watchlists"
import type { FilterPreviewOutcome } from "./filter-preview"

interface FilterBuilderProps {
  value: WatchlistFilter[]
  onChange: (filters: WatchlistFilter[]) => void
  preview?: {
    loading?: boolean
    unavailableReason?: string | null
    outcome?: FilterPreviewOutcome | null
  }
}

interface FilterPreset {
  id: string
  label: string
  filters: WatchlistFilter[]
}

const FILTER_TYPES: { value: FilterType; label: string }[] = [
  { value: "keyword", label: "Keyword" },
  { value: "author", label: "Author" },
  { value: "regex", label: "Regex" },
  { value: "date_range", label: "Date Range" }
]

const FILTER_ACTIONS: { value: FilterAction; label: string; color: string }[] = [
  { value: "include", label: "Include", color: "green" },
  { value: "exclude", label: "Exclude", color: "red" },
  { value: "flag", label: "Flag", color: "orange" }
]

const FILTER_PRESETS: FilterPreset[] = [
  {
    id: "ai-signals",
    label: "AI Signals",
    filters: [
      {
        type: "keyword",
        action: "include",
        value: { keywords: ["ai", "llm", "model"], match: "any" },
        is_active: true,
        priority: 100
      }
    ]
  },
  {
    id: "exclude-sponsored",
    label: "Exclude Sponsored",
    filters: [
      {
        type: "regex",
        action: "exclude",
        value: { pattern: "(?i)sponsored|advertisement|promo", field: "title", flags: "i" },
        is_active: true,
        priority: 110
      }
    ]
  },
  {
    id: "flag-breaking",
    label: "Flag Breaking",
    filters: [
      {
        type: "regex",
        action: "flag",
        value: { pattern: "(?i)breaking|urgent|alert", field: "title", flags: "i" },
        is_active: true,
        priority: 95
      }
    ]
  }
]

const createEmptyFilter = (): WatchlistFilter => ({
  type: "keyword",
  action: "include",
  value: { keywords: [], match: "any" },
  is_active: true
})

const isValidFilterType = (value: string): value is FilterType =>
  FILTER_TYPES.some((opt) => opt.value === value)

const isValidFilterAction = (value: string): value is FilterAction =>
  FILTER_ACTIONS.some((opt) => opt.value === value)

const cloneFilters = (filters: WatchlistFilter[]): WatchlistFilter[] =>
  filters.map((filter) => ({
    ...filter,
    value: { ...(filter.value as Record<string, unknown>) }
  }))

const FILTER_PREVIEW_PANEL_ID = "watchlists-filter-preview-panel"

const toStringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => String(entry || "").trim())
    .filter((entry) => entry.length > 0)
}

const toStringOrNull = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const formatList = (values: string[]): string => {
  if (values.length === 0) return "—"
  return values.join(", ")
}

const getRegexValidationError = (pattern: unknown, flags: unknown): string | null => {
  const normalizedPattern = typeof pattern === "string" ? pattern.trim() : ""
  const normalizedFlags = typeof flags === "string" ? flags.trim() : ""
  if (!normalizedPattern) return null
  try {
    // Validate user-provided regex while preserving advanced use-cases.
    void new RegExp(normalizedPattern, normalizedFlags)
    return null
  } catch (error) {
    return error instanceof Error ? error.message : String(error)
  }
}

export const FilterBuilder: React.FC<FilterBuilderProps> = ({
  value,
  onChange,
  preview
}) => {
  const { t } = useTranslation(["watchlists"])
  const [selectedPresetId, setSelectedPresetId] = React.useState<string | null>(null)
  const [advancedMode, setAdvancedMode] = React.useState(false)
  const [advancedJson, setAdvancedJson] = React.useState("")
  const [advancedError, setAdvancedError] = React.useState<string | null>(null)

  const getFilterSummary = React.useCallback(
    (filter: WatchlistFilter): string => {
      const filterValue = filter.value as Record<string, unknown>
      const actionLabel = t(`watchlists:jobs.filters.action.${filter.action}`, filter.action)

      if (filter.type === "keyword") {
        const keywords = toStringArray(filterValue.keywords)
        const matchMode = String(filterValue.match || filterValue.mode || "any")
        const matchLabel =
          matchMode === "all"
            ? t("watchlists:filters.summary.matchAll", "all terms")
            : t("watchlists:filters.summary.matchAny", "any term")
        if (keywords.length === 0) {
          return t("watchlists:filters.summary.keywordEmpty", "{{action}} items with no keyword terms yet.", {
            action: actionLabel
          })
        }
        return t(
          "watchlists:filters.summary.keyword",
          "{{action}} items containing {{keywords}} ({{matchLabel}}).",
          {
            action: actionLabel,
            keywords: formatList(keywords),
            matchLabel
          }
        )
      }

      if (filter.type === "author") {
        const names = toStringArray(filterValue.names || filterValue.authors)
        if (names.length === 0) {
          return t("watchlists:filters.summary.authorEmpty", "{{action}} items with no author list yet.", {
            action: actionLabel
          })
        }
        return t("watchlists:filters.summary.author", "{{action}} items by {{authors}}.", {
          action: actionLabel,
          authors: formatList(names)
        })
      }

      if (filter.type === "regex") {
        const fieldKey = String(filterValue.field || "title")
        const fieldLabel = t(`watchlists:filters.field${fieldKey.charAt(0).toUpperCase()}${fieldKey.slice(1)}`, fieldKey)
        const pattern = toStringOrNull(filterValue.pattern) || t("watchlists:filters.summary.regexMissingPattern", "(missing pattern)")
        const flags = toStringOrNull(filterValue.flags)
        const renderedPattern = `/${pattern}/${flags || ""}`
        return t("watchlists:filters.summary.regex", "{{action}} when {{field}} matches {{pattern}}.", {
          action: actionLabel,
          field: fieldLabel.toLowerCase(),
          pattern: renderedPattern
        })
      }

      if (filter.type === "date_range") {
        const since = toStringOrNull(filterValue.since || filterValue.start) || t("watchlists:filters.summary.anyDate", "any date")
        const until = toStringOrNull(filterValue.until || filterValue.end) || t("watchlists:filters.summary.anyDate", "any date")
        return t("watchlists:filters.summary.dateRange", "{{action}} items published from {{since}} to {{until}}.", {
          action: actionLabel,
          since,
          until
        })
      }

      return t("watchlists:filters.summary.fallback", "{{action}} using {{type}} filter.", {
        action: actionLabel,
        type: filter.type
      })
    },
    [t]
  )

  const handleAddFilter = () => {
    onChange([...value, createEmptyFilter()])
  }

  const handleRemoveFilter = (index: number) => {
    const newFilters = [...value]
    newFilters.splice(index, 1)
    onChange(newFilters)
  }

  const handleUpdateFilter = (index: number, updates: Partial<WatchlistFilter>) => {
    const newFilters = [...value]
    newFilters[index] = { ...newFilters[index], ...updates }
    onChange(newFilters)
  }

  const handleMoveFilter = (index: number, direction: -1 | 1) => {
    const targetIndex = index + direction
    if (targetIndex < 0 || targetIndex >= value.length) return
    const newFilters = [...value]
    const [moved] = newFilters.splice(index, 1)
    newFilters.splice(targetIndex, 0, moved)
    onChange(newFilters)
  }

  const handleTypeChange = (index: number, type: FilterType) => {
    // Reset value when type changes
    let newValue: Record<string, unknown> = {}
    switch (type) {
      case "keyword":
        newValue = { keywords: [], match: "any" }
        break
      case "author":
        newValue = { names: [], match: "any" }
        break
      case "regex":
        newValue = { pattern: "", field: "title", flags: "i" }
        break
      case "date_range":
        newValue = { since: null, until: null }
        break
    }
    handleUpdateFilter(index, { type, value: newValue })
  }

  const parseAdvancedFilters = (raw: string): WatchlistFilter[] => {
    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch {
      throw new Error(t("watchlists:filters.invalidJson", "Invalid JSON"))
    }
    if (!Array.isArray(parsed)) {
      throw new Error(t("watchlists:filters.arrayRequired", "Filters JSON must be an array"))
    }

    return parsed.map((entry, idx) => {
      if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
        throw new Error(
          t("watchlists:filters.invalidEntry", "Filter at index {{index}} must be an object", { index: idx })
        )
      }
      const filter = entry as Record<string, unknown>
      const typeRaw = String(filter.type || "")
      const actionRaw = String(filter.action || "")

      if (!isValidFilterType(typeRaw)) {
        throw new Error(
          t("watchlists:filters.invalidType", "Filter type is invalid at index {{index}}", { index: idx })
        )
      }
      if (!isValidFilterAction(actionRaw)) {
        throw new Error(
          t("watchlists:filters.invalidAction", "Filter action is invalid at index {{index}}", { index: idx })
        )
      }

      const valueRaw = filter.value
      if (!valueRaw || typeof valueRaw !== "object" || Array.isArray(valueRaw)) {
        throw new Error(
          t("watchlists:filters.invalidValue", "Filter value must be an object at index {{index}}", { index: idx })
        )
      }

      const priorityRaw = filter.priority
      let priority: number | undefined = undefined
      if (priorityRaw !== null && priorityRaw !== undefined && priorityRaw !== "") {
        const numeric = Number(priorityRaw)
        if (!Number.isFinite(numeric) || numeric < 0) {
          throw new Error(
            t("watchlists:filters.invalidPriority", "Filter priority must be a non-negative number")
          )
        }
        priority = Math.floor(numeric)
      }

      return {
        type: typeRaw,
        action: actionRaw,
        value: valueRaw as Record<string, unknown>,
        is_active: filter.is_active !== false,
        ...(priority !== undefined ? { priority } : {})
      } satisfies WatchlistFilter
    })
  }

  const handleApplyAdvancedJson = () => {
    try {
      const nextFilters = parseAdvancedFilters(advancedJson)
      onChange(nextFilters)
      setAdvancedError(null)
      message.success(t("watchlists:filters.advancedApplied", "Advanced JSON applied"))
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err)
      setAdvancedError(detail)
    }
  }

  const handleAddPreset = () => {
    if (!selectedPresetId) return
    const preset = FILTER_PRESETS.find((item) => item.id === selectedPresetId)
    if (!preset) return
    onChange([...value, ...cloneFilters(preset.filters)])
    setSelectedPresetId(null)
  }

  React.useEffect(() => {
    if (!advancedMode) {
      setAdvancedJson(JSON.stringify(value, null, 2))
      setAdvancedError(null)
    }
  }, [advancedMode, value])

  const renderFilterValue = (filter: WatchlistFilter, index: number) => {
    const filterValue = filter.value as Record<string, unknown>

    switch (filter.type) {
      case "keyword":
        return (
          <div className="flex-1 space-y-2">
            <Select
              mode="tags"
              placeholder={t("watchlists:filters.keywordsPlaceholder", "Enter keywords")}
              value={(filterValue.keywords as string[]) || []}
              onChange={(keywords) =>
                handleUpdateFilter(index, {
                  value: { ...filterValue, keywords }
                })
              }
              className="w-full"
              tokenSeparators={[","]}
            />
            <Select
              value={(filterValue.match as string) || (filterValue.mode as string) || "any"}
              onChange={(match) =>
                handleUpdateFilter(index, {
                  value: { ...filterValue, match }
                })
              }
              className="w-24"
              size="small"
              options={[
                { value: "any", label: t("watchlists:filters.matchAny", "Match any") },
                { value: "all", label: t("watchlists:filters.matchAll", "Match all") }
              ]}
            />
          </div>
        )

      case "author":
        return (
          <Select
            mode="tags"
            placeholder={t("watchlists:filters.authorsPlaceholder", "Enter author names")}
            value={(filterValue.names as string[]) || (filterValue.authors as string[]) || []}
            onChange={(names) =>
              handleUpdateFilter(index, {
                value: { ...filterValue, names }
              })
            }
            className="flex-1"
            tokenSeparators={[","]}
          />
        )

      case "regex": {
        const regexError = getRegexValidationError(filterValue.pattern, filterValue.flags)
        return (
          <div className="flex-1 space-y-2">
            <div className="flex gap-2">
              <Input
                placeholder={t("watchlists:filters.regexPlaceholder", "Regular expression pattern")}
                value={(filterValue.pattern as string) || ""}
                onChange={(e) =>
                  handleUpdateFilter(index, {
                    value: { ...filterValue, pattern: e.target.value }
                  })
                }
                className="flex-1"
              />
              <Select
                value={(filterValue.field as string) || "title"}
                onChange={(field) =>
                  handleUpdateFilter(index, {
                    value: { ...filterValue, field }
                  })
                }
                className="w-28"
                options={[
                  { value: "title", label: t("watchlists:filters.fieldTitle", "Title") },
                  { value: "summary", label: t("watchlists:filters.fieldSummary", "Summary") },
                  { value: "content", label: t("watchlists:filters.fieldContent", "Content") },
                  { value: "author", label: t("watchlists:filters.fieldAuthor", "Author") }
                ]}
              />
              <Input
                placeholder={t("watchlists:filters.regexFlags", "Flags")}
                value={(filterValue.flags as string) || ""}
                onChange={(e) =>
                  handleUpdateFilter(index, {
                    value: { ...filterValue, flags: e.target.value }
                  })
                }
                className="w-20"
              />
            </div>
            <div className="text-xs text-text-muted">
              {t(
                "watchlists:filters.regexExamples",
                "Examples: sponsored|promo, (?i)breaking, \\bAI\\b"
              )}
            </div>
            {regexError ? (
              <div className="text-xs text-danger" data-testid={`filter-regex-error-${index}`}>
                {t("watchlists:filters.regexInvalid", "Regex pattern is invalid: {{error}}", {
                  error: regexError
                })}
              </div>
            ) : null}
          </div>
        )
      }

      case "date_range":
        return (
          <div className="flex-1 flex gap-2 items-center">
            <Input
              type="date"
              placeholder={t("watchlists:filters.startDate", "Start date")}
              value={(filterValue.since as string) || (filterValue.start as string) || ""}
              onChange={(e) =>
                handleUpdateFilter(index, {
                  value: { ...filterValue, since: e.target.value || null }
                })
              }
              className="w-36"
            />
            <span className="text-text-subtle">to</span>
            <Input
              type="date"
              placeholder={t("watchlists:filters.endDate", "End date")}
              value={(filterValue.until as string) || (filterValue.end as string) || ""}
              onChange={(e) =>
                handleUpdateFilter(index, {
                  value: { ...filterValue, until: e.target.value || null }
                })
              }
              className="w-36"
            />
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-border bg-surface p-3">
        <div className="mb-2 flex items-center justify-between gap-2">
          <div className="text-xs font-medium text-text-muted">
            {t("watchlists:filters.presets", "Presets")}
          </div>
          <div className="flex items-center gap-2">
            <div className="text-xs text-text-muted">
              {t("watchlists:filters.advancedMode", "Advanced JSON")}
            </div>
            <Switch
              size="small"
              checked={advancedMode}
              onChange={(checked) => {
                setAdvancedMode(checked)
                if (checked) {
                  setAdvancedJson(JSON.stringify(value, null, 2))
                  setAdvancedError(null)
                }
              }}
            />
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select
            className="min-w-56 flex-1"
            value={selectedPresetId}
            onChange={setSelectedPresetId}
            allowClear
            placeholder={t("watchlists:filters.selectPreset", "Select a preset")}
            options={FILTER_PRESETS.map((preset) => ({
              label: preset.label,
              value: preset.id
            }))}
          />
          <Button
            icon={<Plus className="h-4 w-4" />}
            disabled={!selectedPresetId}
            onClick={handleAddPreset}
          >
            {t("watchlists:filters.addPreset", "Add preset")}
          </Button>
          {preview ? (
            <Button
              size="small"
              data-testid="filter-preview-impact-button"
              onClick={() => {
                const node = document.getElementById(FILTER_PREVIEW_PANEL_ID)
                node?.scrollIntoView?.({ behavior: "smooth", block: "start" })
              }}
            >
              {t("watchlists:filters.previewImpact", "Preview impact")}
            </Button>
          ) : null}
        </div>
      </div>

      {advancedMode && (
        <div className="space-y-2 rounded-lg border border-border bg-surface p-3">
          <div className="text-xs font-medium text-text-muted">
            {t("watchlists:filters.advancedHelp", "Edit filters as raw JSON")}
          </div>
          <Input.TextArea
            value={advancedJson}
            onChange={(e) => {
              setAdvancedJson(e.target.value)
              setAdvancedError(null)
            }}
            rows={12}
            className="font-mono text-xs"
          />
          {advancedError && (
            <div className="text-xs text-danger">{advancedError}</div>
          )}
          <div className="flex items-center justify-end gap-2">
            <Button
              type="default"
              icon={<Code2 className="h-4 w-4" />}
              onClick={() => {
                try {
                  const formatted = JSON.stringify(parseAdvancedFilters(advancedJson), null, 2)
                  setAdvancedJson(formatted)
                  setAdvancedError(null)
                } catch (err) {
                  const detail = err instanceof Error ? err.message : String(err)
                  setAdvancedError(detail)
                }
              }}
            >
              {t("watchlists:filters.formatJson", "Format JSON")}
            </Button>
            <Button type="primary" onClick={handleApplyAdvancedJson}>
              {t("watchlists:filters.applyJson", "Apply JSON")}
            </Button>
          </div>
        </div>
      )}

      {value.length === 0 ? (
        <div className="text-center py-4 text-text-muted border border-dashed border-border rounded-lg">
          {t("watchlists:filters.noFilters", "No filters configured. All items will be included.")}
        </div>
      ) : (
        value.map((filter, index) => (
          <div
            key={index}
            className="space-y-2 border border-border rounded-lg bg-surface p-3"
          >
            <div className="flex items-start gap-3">
              {/* Filter type */}
              <Select
                value={filter.type}
                onChange={(type) => handleTypeChange(index, type)}
                className="w-28"
                options={FILTER_TYPES}
              />

              {/* Filter action */}
              <Select
                value={filter.action}
                onChange={(action) => handleUpdateFilter(index, { action })}
                className="w-24"
                options={FILTER_ACTIONS.map((a) => ({
                  ...a,
                  label: (
                    <Tag color={a.color} className="m-0">
                      {a.label}
                    </Tag>
                  )
                }))}
              />

              <InputNumber
                min={0}
                value={typeof filter.priority === "number" ? filter.priority : null}
                onChange={(priority) =>
                  handleUpdateFilter(index, {
                    priority: typeof priority === "number" ? priority : undefined
                  })
                }
                className="w-24"
                size="small"
                placeholder={t("watchlists:filters.priority", "Priority")}
              />

              {/* Filter value (type-specific) */}
              {renderFilterValue(filter, index)}

              {/* Active toggle and delete */}
              <div className="flex items-center gap-2">
                <Tooltip title={t("watchlists:filters.moveUp", "Move up")}>
                  <Button
                    type="text"
                    size="small"
                    icon={<ArrowUp className="h-4 w-4" />}
                    onClick={() => handleMoveFilter(index, -1)}
                    disabled={index === 0}
                  />
                </Tooltip>
                <Tooltip title={t("watchlists:filters.moveDown", "Move down")}>
                  <Button
                    type="text"
                    size="small"
                    icon={<ArrowDown className="h-4 w-4" />}
                    onClick={() => handleMoveFilter(index, 1)}
                    disabled={index === value.length - 1}
                  />
                </Tooltip>
                <Switch
                  checked={filter.is_active !== false}
                  onChange={(checked) =>
                    handleUpdateFilter(index, { is_active: checked })
                  }
                  size="small"
                />
                <Button
                  type="text"
                  size="small"
                  danger
                  icon={<Trash2 className="h-4 w-4" />}
                  onClick={() => handleRemoveFilter(index)}
                />
              </div>
            </div>
            <div
              className="rounded-md border border-border/70 bg-background px-2 py-1 text-xs text-text-muted"
              data-testid={`filter-rule-summary-${index}`}
            >
              {getFilterSummary(filter)}
            </div>
          </div>
        ))
      )}

      <Button
        type="dashed"
        icon={<Plus className="h-4 w-4" />}
        onClick={handleAddFilter}
        className="w-full"
      >
        {t("watchlists:filters.addFilter", "Add Filter")}
      </Button>

      {preview && (
        <div
          id={FILTER_PREVIEW_PANEL_ID}
          className="rounded-lg border border-border bg-surface p-3"
          data-testid="filter-preview-panel"
        >
          <div className="text-xs font-medium text-text-muted">
            {t("watchlists:filters.preview.title", "Sample filter preview")}
          </div>
          {preview.loading ? (
            <div className="mt-2 text-xs text-text-muted">
              {t("watchlists:filters.preview.loading", "Loading sample candidates...")}
            </div>
          ) : preview.unavailableReason ? (
            <div className="mt-2 text-xs text-text-muted">
              {preview.unavailableReason}
            </div>
          ) : preview.outcome ? (
            <div className="mt-2 space-y-2">
              <div className="text-xs text-text-muted">
                {t(
                  "watchlists:filters.preview.summary",
                  "Found {{ingestable}} articles ({{filtered}} skipped) out of {{total}} sample items.",
                  {
                    ingestable: preview.outcome.ingestable,
                    filtered: preview.outcome.filtered,
                    total: preview.outcome.total
                  }
                )}
              </div>
              {preview.outcome.total > 0 ? (
                <div className="space-y-1">
                  {preview.outcome.items.slice(0, 4).map((item) => {
                    const reason = item.preview_filter_key || item.preview_filter_type || "-"
                    return (
                      <div
                        key={`${item.source_id}-${item.url || item.title || ""}`}
                        className="flex items-start justify-between gap-2 text-xs"
                      >
                        <div className="min-w-0 flex-1 truncate text-text-muted">
                          {item.title || item.url || t("watchlists:runs.detail.itemsUntitled", "Untitled")}
                        </div>
                        <div className="shrink-0 flex items-center gap-1">
                          <Tag color={item.preview_decision === "ingest" ? "green" : "red"} className="m-0">
                            {item.preview_decision}
                          </Tag>
                          <span className="text-text-subtle">{reason}</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="text-xs text-text-muted">
                  {t("watchlists:filters.preview.empty", "No sample candidates available.")}
                </div>
              )}
            </div>
          ) : (
            <div className="mt-2 text-xs text-text-muted">
              {t("watchlists:filters.preview.unavailable", "Sample preview unavailable.")}
            </div>
          )}
        </div>
      )}

      <div className="text-xs text-text-muted">
        {t(
          "watchlists:filters.help",
          "Filters determine which items are included, excluded, or flagged during monitor runs. Include filters require at least one match."
        )}
      </div>
    </div>
  )
}
