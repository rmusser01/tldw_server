import { Link } from "react-router-dom"
import { useTranslation } from "react-i18next"

import type {
  ScheduledTaskAutomationHomeItem,
  ScheduledTaskResultSeverity
} from "../../ScheduledTasks/scheduled-task-results"

type AutomationInboxCardProps = {
  items: ScheduledTaskAutomationHomeItem[]
  loading: boolean
  partial: boolean
  error: string | null
  maxItems?: number
}

const formatUpdatedAt = (value: string | null | undefined): string => {
  if (!value) return "Updated recently"
  const timestamp = Date.parse(value)
  if (!Number.isFinite(timestamp)) return "Updated recently"
  return `Updated ${new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric"
  }).format(new Date(timestamp))}`
}

const severityClasses = (severity: ScheduledTaskResultSeverity): string => {
  if (severity === "error") {
    return "border-danger/30 bg-danger/10 text-danger"
  }
  if (severity === "warning") {
    return "border-warn/30 bg-warn/10 text-warn"
  }
  if (severity === "success") {
    return "border-success/30 bg-success/10 text-success"
  }
  return "border-primary/30 bg-primary/10 text-primary"
}

export function AutomationInboxCard({
  items,
  loading,
  partial,
  error,
  maxItems = 4
}: AutomationInboxCardProps) {
  const { t } = useTranslation()
  const visibleItems = items.slice(0, maxItems)
  const hasItems = visibleItems.length > 0
  const signalCountLabel = `${items.length} signal${items.length === 1 ? "" : "s"}`
  const subtitle = loading && !hasItems
    ? t("companionHome.automationInbox.checking", { defaultValue: "Checking now" })
    : items.length > 0
      ? t("companionHome.automationInbox.signalCount", {
          count: items.length,
          defaultValue: signalCountLabel
        })
      : error
        ? t("companionHome.automationInbox.zeroSignals", { defaultValue: "0 signals" })
        : t(
            "companionHome.automationInbox.nothingNew",
            { defaultValue: "Nothing new from automations." }
          )
  const emptyLabel = loading
    ? t(
        "companionHome.automationInbox.loadingLabel",
        { defaultValue: "Loading automation signals" }
      )
    : error && !hasItems
      ? error
      : t(
          "companionHome.automationInbox.emptyLabel",
          { defaultValue: "No automation results yet" }
        )
  const emptyDescription = loading
    ? t(
        "companionHome.automationInbox.loadingDescription",
        { defaultValue: "Checking recent scheduled-task results and notifications." }
      )
    : error && !hasItems
      ? t(
          "companionHome.automationInbox.errorDescription",
          {
            defaultValue:
              "Scheduled-task results are temporarily unavailable. Other Home cards remain available."
          }
        )
      : t(
          "companionHome.automationInbox.emptyDescription",
          {
            defaultValue:
              "Results and failures from scheduled tasks appear here after a run. Future scheduled questions and agent outputs appear here only when routed by task visibility policy."
          }
        )

  return (
    <section className="rounded-3xl border border-border/80 bg-surface/90 p-5 shadow-sm backdrop-blur-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-text">
            {t("companionHome.automationInbox.title", {
              defaultValue: "Automation Inbox"
            })}
          </h2>
          <p className="mt-1 text-sm text-text-muted">{subtitle}</p>
        </div>
        <span className="rounded-full border border-border/70 bg-bg/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-text-muted">
          {items.length}
        </span>
      </div>

      {error && partial && hasItems ? (
        <div className="mt-4 rounded-2xl border border-warn/30 bg-warn/10 px-4 py-3">
          <div className="text-sm font-semibold text-text">
            {t(
              "companionHome.automationInbox.partialTitle",
              { defaultValue: "Partial automation data" }
            )}
          </div>
          <p className="mt-1 text-sm leading-6 text-text-muted">{error}</p>
        </div>
      ) : null}

      {hasItems ? (
        <ul className="mt-4 space-y-3">
          {visibleItems.map((item) => (
            <li
              key={item.id}
              className="rounded-2xl border border-border/70 bg-bg/60 p-3"
            >
              <Link
                className="block rounded-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                to={item.href}
              >
                <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                  <div className="min-w-0">
                    <div className="text-sm font-semibold text-text">{item.title}</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <span
                        className={`rounded-full border px-2 py-1 text-xs font-semibold ${severityClasses(item.severity)}`}
                      >
                        {item.statusLabel}
                      </span>
                      <span className="rounded-full border border-border/70 bg-surface/75 px-2 py-1 text-xs font-semibold text-text-muted">
                        {item.ownerLabel}
                      </span>
                    </div>
                  </div>
                  <span className="shrink-0 text-xs text-text-muted">
                    {formatUpdatedAt(item.updatedAt)}
                  </span>
                </div>
                <p className="mt-2 text-sm leading-6 text-text-muted">{item.summary}</p>
              </Link>
            </li>
          ))}
        </ul>
      ) : (
        <div className="mt-4 rounded-2xl border border-dashed border-border/70 bg-bg/60 p-4">
          <div className="text-sm font-semibold text-text">{emptyLabel}</div>
          <p className="mt-2 text-sm leading-6 text-text-muted">{emptyDescription}</p>
        </div>
      )}
    </section>
  )
}
