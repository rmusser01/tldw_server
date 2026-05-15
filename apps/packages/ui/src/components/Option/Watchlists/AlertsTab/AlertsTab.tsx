import React, { useCallback, useEffect, useMemo, useState } from "react"
import { Alert, Button, Empty, Input, Select, Switch, Tag, Tooltip } from "antd"
import {
  BellRing,
  CheckCircle2,
  Edit2,
  ExternalLink,
  Plus,
  RefreshCw,
  Trash2,
  XCircle
} from "lucide-react"
import { useTranslation } from "react-i18next"
import {
  createWatchlistContentAlertRule,
  deleteWatchlistContentAlertRule,
  fetchWatchlistContentAlertRules,
  fetchWatchlistContentAlerts,
  updateWatchlistContentAlert,
  updateWatchlistContentAlertRule
} from "@/services/watchlists"
import { useWatchlistsStore } from "@/store/watchlists"
import type {
  WatchlistContentAlert,
  WatchlistContentAlertMatchMode,
  WatchlistContentAlertRule,
  WatchlistContentAlertRuleKind,
  WatchlistContentAlertSeverity,
  WatchlistContentAlertStatus
} from "@/types/watchlists"

type AlertStatusFilter = WatchlistContentAlertStatus | "all"
type SeverityFilter = WatchlistContentAlertSeverity | "all"

interface RuleFormState {
  id: number | null
  name: string
  rule_kind: WatchlistContentAlertRuleKind
  match_mode: WatchlistContentAlertMatchMode
  pattern: string
  severity: WatchlistContentAlertSeverity
  sourceTagsText: string
}

const DEFAULT_RULE_FORM: RuleFormState = {
  id: null,
  name: "",
  rule_kind: "keyword",
  match_mode: "contains",
  pattern: "",
  severity: "medium",
  sourceTagsText: ""
}

const RULE_KIND_OPTIONS: Array<{ value: WatchlistContentAlertRuleKind; label: string }> = [
  { value: "keyword", label: "Keyword" },
  { value: "descriptor", label: "Descriptor" },
  { value: "classification", label: "Classification" },
  { value: "entity", label: "Entity" },
  { value: "ioc", label: "IOC" },
  { value: "cve", label: "CVE" },
  { value: "regex", label: "Regex" }
]

const MATCH_MODE_OPTIONS: Array<{ value: WatchlistContentAlertMatchMode; label: string }> = [
  { value: "contains", label: "Contains" },
  { value: "exact", label: "Exact" },
  { value: "regex", label: "Regex" }
]

const SEVERITY_OPTIONS: Array<{ value: WatchlistContentAlertSeverity; label: string }> = [
  { value: "info", label: "Info" },
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
  { value: "critical", label: "Critical" }
]

const STATUS_OPTIONS: Array<{ value: AlertStatusFilter; label: string }> = [
  { value: "unread", label: "Unread" },
  { value: "read", label: "Read" },
  { value: "dismissed", label: "Dismissed" },
  { value: "all", label: "All" }
]

const toCsv = (values: unknown): string => {
  if (!Array.isArray(values)) return ""
  return values.map((value) => String(value).trim()).filter(Boolean).join(", ")
}

const toSourceTags = (value: string): string[] =>
  value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)

const toNumberFilter = (value: string): number | undefined => {
  if (!value.trim()) return undefined
  const parsed = Number(value.trim())
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
}

const formatTimestamp = (value: string | null | undefined): string => {
  if (!value) return ""
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(parsed)
}

const severityColor = (severity: WatchlistContentAlertSeverity): string => {
  if (severity === "critical") return "red"
  if (severity === "high") return "volcano"
  if (severity === "medium") return "orange"
  if (severity === "low") return "blue"
  return "default"
}

const matchesSearch = (alert: WatchlistContentAlert, query: string): boolean => {
  const normalized = query.trim().toLowerCase()
  if (!normalized) return true
  const evidence = alert.evidence || {}
  return [
    alert.title,
    alert.snippet,
    alert.matched_text,
    evidence.source_name,
    evidence.url,
    evidence.source_url
  ]
    .filter((value): value is string => typeof value === "string")
    .some((value) => value.toLowerCase().includes(normalized))
}

export const AlertsTab: React.FC = () => {
  const { t } = useTranslation(["watchlists", "common"])
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)

  const [rules, setRules] = useState<WatchlistContentAlertRule[]>([])
  const [alerts, setAlerts] = useState<WatchlistContentAlert[]>([])
  const [rulesLoading, setRulesLoading] = useState(false)
  const [alertsLoading, setAlertsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [ruleFormOpen, setRuleFormOpen] = useState(false)
  const [ruleSaving, setRuleSaving] = useState(false)
  const [ruleForm, setRuleForm] = useState<RuleFormState>(DEFAULT_RULE_FORM)
  const [ruleFormError, setRuleFormError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<AlertStatusFilter>("unread")
  const [severityFilter, setSeverityFilter] = useState<SeverityFilter>("all")
  const [ruleFilter, setRuleFilter] = useState<string>("all")
  const [sourceFilterText, setSourceFilterText] = useState("")
  const [searchText, setSearchText] = useState("")
  const [updatingAlertId, setUpdatingAlertId] = useState<number | null>(null)

  const loadRules = useCallback(async () => {
    if (selectedWatchlistId == null) return
    setRulesLoading(true)
    setError(null)
    try {
      const response = await fetchWatchlistContentAlertRules(selectedWatchlistId, { page: 1, size: 100 })
      setRules(Array.isArray(response.items) ? response.items : [])
    } catch {
      setError(t("watchlists:alerts.loadRulesError", "Failed to load content alert rules"))
    } finally {
      setRulesLoading(false)
    }
  }, [selectedWatchlistId, t])

  const loadAlerts = useCallback(async () => {
    if (selectedWatchlistId == null) return
    setAlertsLoading(true)
    setError(null)
    try {
      const response = await fetchWatchlistContentAlerts(selectedWatchlistId, {
        status: statusFilter === "all" ? undefined : statusFilter,
        severity: severityFilter === "all" ? undefined : severityFilter,
        rule_id: ruleFilter === "all" ? undefined : Number(ruleFilter),
        source_id: toNumberFilter(sourceFilterText),
        page: 1,
        size: 50
      })
      setAlerts(Array.isArray(response.items) ? response.items : [])
    } catch {
      setError(t("watchlists:alerts.loadAlertsError", "Failed to load content alerts"))
    } finally {
      setAlertsLoading(false)
    }
  }, [ruleFilter, selectedWatchlistId, severityFilter, sourceFilterText, statusFilter, t])

  useEffect(() => {
    void loadRules()
  }, [loadRules])

  useEffect(() => {
    void loadAlerts()
  }, [loadAlerts])

  const filteredAlerts = useMemo(
    () => alerts.filter((alert) => matchesSearch(alert, searchText)),
    [alerts, searchText]
  )

  const ruleOptions = useMemo(
    () => [
      { value: "all", label: t("watchlists:alerts.filters.allRules", "All rules") },
      ...rules.map((rule) => ({ value: String(rule.id), label: rule.name }))
    ],
    [rules, t]
  )

  const openCreateRule = useCallback(() => {
    setRuleForm(DEFAULT_RULE_FORM)
    setRuleFormError(null)
    setRuleFormOpen(true)
  }, [])

  const openEditRule = useCallback((rule: WatchlistContentAlertRule) => {
    setRuleForm({
      id: rule.id,
      name: rule.name,
      rule_kind: rule.rule_kind,
      match_mode: rule.match_mode,
      pattern: rule.pattern,
      severity: rule.severity,
      sourceTagsText: toCsv(rule.source_constraints?.source_tags)
    })
    setRuleFormError(null)
    setRuleFormOpen(true)
  }, [])

  const saveRule = useCallback(async () => {
    if (selectedWatchlistId == null) return
    const name = ruleForm.name.trim()
    const pattern = ruleForm.pattern.trim()
    if (!name || !pattern) {
      setRuleFormError(t("watchlists:alerts.validation.required", "Name and pattern are required"))
      return
    }

    const sourceTags = toSourceTags(ruleForm.sourceTagsText)
    const payload = {
      name,
      rule_kind: ruleForm.rule_kind,
      match_mode: ruleForm.match_mode,
      pattern,
      severity: ruleForm.severity,
      source_constraints: sourceTags.length ? { source_tags: sourceTags } : null
    }

    setRuleSaving(true)
    setRuleFormError(null)
    try {
      if (ruleForm.id != null) {
        await updateWatchlistContentAlertRule(selectedWatchlistId, ruleForm.id, payload)
      } else {
        await createWatchlistContentAlertRule(selectedWatchlistId, payload)
      }
      setRuleFormOpen(false)
      await loadRules()
      await loadAlerts()
    } catch {
      setRuleFormError(t("watchlists:alerts.validation.saveFailed", "Failed to save content alert rule"))
    } finally {
      setRuleSaving(false)
    }
  }, [loadAlerts, loadRules, ruleForm, selectedWatchlistId, t])

  const toggleRule = useCallback(async (rule: WatchlistContentAlertRule, enabled: boolean) => {
    if (selectedWatchlistId == null) return
    await updateWatchlistContentAlertRule(selectedWatchlistId, rule.id, { enabled })
    await loadRules()
  }, [loadRules, selectedWatchlistId])

  const deleteRule = useCallback(async (rule: WatchlistContentAlertRule) => {
    if (selectedWatchlistId == null) return
    await deleteWatchlistContentAlertRule(selectedWatchlistId, rule.id)
    await loadRules()
  }, [loadRules, selectedWatchlistId])

  const updateAlertStatus = useCallback(async (alert: WatchlistContentAlert, status: WatchlistContentAlertStatus) => {
    if (selectedWatchlistId == null) return
    setUpdatingAlertId(alert.id)
    try {
      await updateWatchlistContentAlert(selectedWatchlistId, alert.id, { status })
      await loadAlerts()
    } finally {
      setUpdatingAlertId(null)
    }
  }, [loadAlerts, selectedWatchlistId])

  if (selectedWatchlistId == null) {
    return (
      <Empty
        description={t(
          "watchlists:alerts.noWatchlist",
          "Select a Watchlist to manage content alert rules and review matching items."
        )}
      />
    )
  }

  return (
    <div className="space-y-4" data-testid="watchlists-alerts-tab">
      <Alert
        type="info"
        showIcon
        message={t("watchlists:alerts.healthBoundary", "Run failures and source problems are health issues, not content alerts.")}
        description={t(
          "watchlists:alerts.boundaryDescription",
          "Use content alert rules for newly collected items that match a descriptor, keyword, classification, entity, IOC, CVE, or source constraint."
        )}
      />

      {error && (
        <Alert
          type="error"
          showIcon
          message={error}
          action={(
            <Button size="small" onClick={() => { void loadRules(); void loadAlerts() }}>
              {t("common:refresh", "Refresh")}
            </Button>
          )}
        />
      )}

      <section className="rounded-lg border border-border bg-surface p-4" aria-label={t("watchlists:alerts.rulesTitle", "Content alert rules")}>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h3 className="m-0 text-base font-semibold text-text">
              {t("watchlists:alerts.rulesTitle", "Content alert rules")}
            </h3>
            <p className="mt-1 max-w-3xl text-sm text-text-muted">
              {t(
                "watchlists:alerts.rulesDescription",
                "Create a rule to be notified when new Watchlist items match a descriptor, keyword, classification, entity, or source constraint."
              )}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button icon={<RefreshCw className="h-4 w-4" />} onClick={() => void loadRules()} loading={rulesLoading}>
              {t("common:refresh", "Refresh")}
            </Button>
            <Button type="primary" icon={<Plus className="h-4 w-4" />} onClick={openCreateRule}>
              {t("watchlists:alerts.createRule", "Create rule")}
            </Button>
          </div>
        </div>

        {ruleFormOpen && (
          <div className="mt-4 rounded-md border border-border bg-background p-3" data-testid="watchlists-alert-rule-form">
            <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_180px_160px_150px]">
              <label className="text-sm font-medium text-text">
                {t("watchlists:alerts.form.name", "Rule name")}
                <Input
                  aria-label={t("watchlists:alerts.form.name", "Rule name")}
                  className="mt-1"
                  value={ruleForm.name}
                  onChange={(event) => setRuleForm((prev) => ({ ...prev, name: event.currentTarget.value }))}
                />
              </label>
              <label className="text-sm font-medium text-text">
                {t("watchlists:alerts.form.kind", "Rule kind")}
                <Select
                  aria-label={t("watchlists:alerts.form.kind", "Rule kind")}
                  className="mt-1 w-full"
                  value={ruleForm.rule_kind}
                  options={RULE_KIND_OPTIONS}
                  onChange={(value) => setRuleForm((prev) => ({
                    ...prev,
                    rule_kind: value as WatchlistContentAlertRuleKind
                  }))}
                />
              </label>
              <label className="text-sm font-medium text-text">
                {t("watchlists:alerts.form.matchMode", "Match mode")}
                <Select
                  aria-label={t("watchlists:alerts.form.matchMode", "Match mode")}
                  className="mt-1 w-full"
                  value={ruleForm.match_mode}
                  options={MATCH_MODE_OPTIONS}
                  onChange={(value) => setRuleForm((prev) => ({
                    ...prev,
                    match_mode: value as WatchlistContentAlertMatchMode
                  }))}
                />
              </label>
              <label className="text-sm font-medium text-text">
                {t("watchlists:alerts.form.severity", "Severity")}
                <Select
                  aria-label={t("watchlists:alerts.form.severity", "Severity")}
                  className="mt-1 w-full"
                  value={ruleForm.severity}
                  options={SEVERITY_OPTIONS}
                  onChange={(value) => setRuleForm((prev) => ({
                    ...prev,
                    severity: value as WatchlistContentAlertSeverity
                  }))}
                />
              </label>
            </div>
            <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(220px,320px)]">
              <label className="text-sm font-medium text-text">
                {t("watchlists:alerts.form.pattern", "Pattern")}
                <Input
                  aria-label={t("watchlists:alerts.form.pattern", "Pattern")}
                  className="mt-1"
                  value={ruleForm.pattern}
                  onChange={(event) => setRuleForm((prev) => ({ ...prev, pattern: event.currentTarget.value }))}
                />
              </label>
              <label className="text-sm font-medium text-text">
                {t("watchlists:alerts.form.sourceTags", "Source tags")}
                <Input
                  aria-label={t("watchlists:alerts.form.sourceTags", "Source tags")}
                  className="mt-1"
                  value={ruleForm.sourceTagsText}
                  onChange={(event) => setRuleForm((prev) => ({ ...prev, sourceTagsText: event.currentTarget.value }))}
                  placeholder={t("watchlists:alerts.form.sourceTagsPlaceholder", "advisory, cti")}
                />
              </label>
            </div>
            {ruleFormError && (
              <div className="mt-2 text-sm text-red-600">{ruleFormError}</div>
            )}
            <div className="mt-3 flex flex-wrap gap-2">
              <Button type="primary" onClick={() => void saveRule()} loading={ruleSaving}>
                {t("watchlists:alerts.saveRule", "Save rule")}
              </Button>
              <Button onClick={() => setRuleFormOpen(false)}>
                {t("common:cancel", "Cancel")}
              </Button>
            </div>
          </div>
        )}

        <div className="mt-4 grid gap-2">
          {rules.length === 0 && !rulesLoading ? (
            <Empty description={t("watchlists:alerts.emptyRules", "No content alert rules yet.")} />
          ) : (
            rules.map((rule) => (
              <div
                key={rule.id}
                className="flex flex-col gap-3 rounded-md border border-border bg-background p-3 md:flex-row md:items-start md:justify-between"
              >
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium text-text">{rule.name}</span>
                    <Tag color={severityColor(rule.severity)}>{rule.severity}</Tag>
                    <Tag>{rule.rule_kind}</Tag>
                    <Tag>{rule.match_mode}</Tag>
                    {!rule.enabled && <Tag>{t("watchlists:alerts.disabled", "Disabled")}</Tag>}
                  </div>
                  <div className="mt-1 break-words text-sm text-text-muted">{rule.pattern}</div>
                  {Array.isArray(rule.source_constraints?.source_tags) && rule.source_constraints.source_tags.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1 text-xs text-text-muted">
                      {rule.source_constraints.source_tags.map((tag) => (
                        <Tag key={tag}>{tag}</Tag>
                      ))}
                    </div>
                  )}
                </div>
                <div className="flex shrink-0 flex-wrap items-center gap-2">
                  <Tooltip title={t("watchlists:alerts.enabledTooltip", "Enable or pause this content alert rule")}>
                    <Switch
                      aria-label={t("watchlists:alerts.enabledTooltip", "Enable or pause this content alert rule")}
                      checked={rule.enabled}
                      onChange={(checked) => void toggleRule(rule, checked)}
                    />
                  </Tooltip>
                  <Button icon={<Edit2 className="h-4 w-4" />} onClick={() => openEditRule(rule)}>
                    {t("common:edit", "Edit")}
                  </Button>
                  <Button icon={<Trash2 className="h-4 w-4" />} danger onClick={() => void deleteRule(rule)}>
                    {t("common:delete", "Delete")}
                  </Button>
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="rounded-lg border border-border bg-surface p-4" aria-label={t("watchlists:alerts.inboxTitle", "Alert inbox")}>
        <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
          <div>
            <h3 className="m-0 inline-flex items-center gap-2 text-base font-semibold text-text">
              <BellRing className="h-4 w-4" />
              {t("watchlists:alerts.inboxTitle", "Alert inbox")}
            </h3>
            <p className="mt-1 max-w-3xl text-sm text-text-muted">
              {t(
                "watchlists:alerts.inboxDescription",
                "Review why a new item matched, then mark it read or dismiss it from the active queue."
              )}
            </p>
          </div>
          <Button icon={<RefreshCw className="h-4 w-4" />} onClick={() => void loadAlerts()} loading={alertsLoading}>
            {t("common:refresh", "Refresh")}
          </Button>
        </div>

        <div className="mt-4 grid gap-3 lg:grid-cols-[160px_160px_minmax(180px,1fr)_160px_minmax(200px,1fr)]">
          <Select
            aria-label={t("watchlists:alerts.filters.status", "Alert status")}
            value={statusFilter}
            options={STATUS_OPTIONS}
            onChange={(value) => setStatusFilter(value as AlertStatusFilter)}
          />
          <Select
            aria-label={t("watchlists:alerts.filters.severity", "Severity")}
            value={severityFilter}
            options={[{ value: "all", label: t("watchlists:alerts.filters.allSeverity", "All severity") }, ...SEVERITY_OPTIONS]}
            onChange={(value) => setSeverityFilter(value as SeverityFilter)}
          />
          <Select
            aria-label={t("watchlists:alerts.filters.rule", "Rule")}
            value={ruleFilter}
            options={ruleOptions}
            onChange={(value) => setRuleFilter(String(value))}
          />
          <Input
            aria-label={t("watchlists:alerts.filters.sourceId", "Source ID")}
            value={sourceFilterText}
            onChange={(event) => setSourceFilterText(event.currentTarget.value)}
            placeholder={t("watchlists:alerts.filters.sourceId", "Source ID")}
          />
          <Input
            aria-label={t("watchlists:alerts.filters.search", "Search loaded alerts")}
            value={searchText}
            onChange={(event) => setSearchText(event.currentTarget.value)}
            placeholder={t("watchlists:alerts.filters.search", "Search loaded alerts")}
          />
        </div>

        <div className="mt-4 grid gap-3">
          {filteredAlerts.length === 0 && !alertsLoading ? (
            <Empty description={t("watchlists:alerts.emptyInbox", "No content alerts match these filters.")} />
          ) : (
            filteredAlerts.map((alert) => {
              const sourceName = alert.evidence?.source_name || `Source ${alert.source_id}`
              const sourceUrl = alert.evidence?.source_url || null
              const itemUrl = alert.evidence?.url || null
              const rule = rules.find((candidate) => candidate.id === alert.rule_id)
              const nextReadStatus: WatchlistContentAlertStatus = alert.status === "read" ? "unread" : "read"
              return (
                <article
                  key={alert.id}
                  className="rounded-md border border-border bg-background p-3"
                  data-testid={`watchlists-alert-${alert.id}`}
                >
                  <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                    <div className="min-w-0 space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <Tag color={severityColor(alert.severity)}>{alert.severity}</Tag>
                        <Tag>{alert.status}</Tag>
                        {rule && <Tag>{rule.name}</Tag>}
                        <span className="text-xs text-text-muted">{formatTimestamp(alert.created_at)}</span>
                      </div>
                      <h4 className="m-0 break-words text-base font-semibold text-text">
                        {alert.title || t("watchlists:alerts.untitled", "Untitled alert")}
                      </h4>
                      {alert.snippet && (
                        <p className="m-0 max-w-4xl break-words text-sm text-text-muted">{alert.snippet}</p>
                      )}
                      <div className="flex flex-wrap items-center gap-2 text-sm text-text-muted">
                        <span>{sourceName}</span>
                        {sourceUrl && (
                          <a href={sourceUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-primary">
                            {t("watchlists:alerts.sourceLink", "Source")}
                            <ExternalLink className="h-3.5 w-3.5" />
                          </a>
                        )}
                        {itemUrl && (
                          <a href={itemUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-primary">
                            {t("watchlists:alerts.itemLink", "Item")}
                            <ExternalLink className="h-3.5 w-3.5" />
                          </a>
                        )}
                      </div>
                      <div className="flex flex-wrap gap-2 text-xs text-text-muted">
                        <span>{t("watchlists:alerts.ids.item", "Item #{{id}}", { id: alert.item_id })}</span>
                        <span>{t("watchlists:alerts.ids.run", "Run #{{id}}", { id: alert.run_id })}</span>
                        <span>{t("watchlists:alerts.ids.job", "Job #{{id}}", { id: alert.job_id })}</span>
                      </div>
                    </div>
                    <div className="flex shrink-0 flex-wrap gap-2">
                      <Button
                        icon={alert.status === "read" ? <XCircle className="h-4 w-4" /> : <CheckCircle2 className="h-4 w-4" />}
                        loading={updatingAlertId === alert.id}
                        onClick={() => void updateAlertStatus(alert, nextReadStatus)}
                      >
                        {alert.status === "read"
                          ? t("watchlists:alerts.actions.markUnread", "Mark unread")
                          : t("watchlists:alerts.actions.markRead", "Mark read")}
                      </Button>
                      {alert.status !== "dismissed" && (
                        <Button
                          loading={updatingAlertId === alert.id}
                          onClick={() => void updateAlertStatus(alert, "dismissed")}
                        >
                          {t("watchlists:alerts.actions.dismiss", "Dismiss")}
                        </Button>
                      )}
                    </div>
                  </div>
                </article>
              )
            })
          )}
        </div>
      </section>
    </div>
  )
}
