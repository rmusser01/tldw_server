import React from "react"
import {
  CalendarClock,
  FileOutput,
  FileText,
  Newspaper,
  Play,
  Plus,
  Rss,
  type LucideIcon
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { EmptyState } from "@/components/ui/feedback/EmptyState"

type EntityType = "feeds" | "monitors" | "activity" | "articles" | "reports" | "templates"

interface WatchlistsEmptyStateProps {
  entity: EntityType
  onPrimaryAction?: () => void
  onSecondaryAction?: () => void
  /** Override the primary CTA label */
  primaryLabel?: string
  /** Override the secondary CTA label */
  secondaryLabel?: string
  /** Contextual hint for cross-entity guidance */
  contextHint?: string
}

const entityConfig: Record<
  EntityType,
  {
    icon: LucideIcon
    titleKey: string
    titleDefault: string
    descriptionKey: string
    descriptionDefault: string
    primaryCtaKey: string
    primaryCtaDefault: string
    secondaryCtaKey?: string
    secondaryCtaDefault?: string
  }
> = {
  feeds: {
    icon: Rss,
    titleKey: "watchlists:emptyState.feeds.title",
    titleDefault: "No feeds yet",
    descriptionKey: "watchlists:emptyState.feeds.description",
    descriptionDefault: "Feeds are the sources your monitors check for new content.",
    primaryCtaKey: "watchlists:emptyState.feeds.primaryCta",
    primaryCtaDefault: "Add your first feed",
    secondaryCtaKey: "watchlists:emptyState.feeds.secondaryCta",
    secondaryCtaDefault: "Import from OPML"
  },
  monitors: {
    icon: CalendarClock,
    titleKey: "watchlists:emptyState.monitors.title",
    titleDefault: "No monitors yet",
    descriptionKey: "watchlists:emptyState.monitors.description",
    descriptionDefault:
      "Monitors run on a schedule to fetch and process content from your feeds.",
    primaryCtaKey: "watchlists:emptyState.monitors.primaryCta",
    primaryCtaDefault: "Create your first monitor"
  },
  activity: {
    icon: Play,
    titleKey: "watchlists:emptyState.activity.title",
    titleDefault: "No activity yet",
    descriptionKey: "watchlists:emptyState.activity.description",
    descriptionDefault:
      "Activity shows the history of monitor runs. Set up a monitor to start seeing activity here.",
    primaryCtaKey: "watchlists:emptyState.activity.primaryCta",
    primaryCtaDefault: "Set up a monitor"
  },
  articles: {
    icon: Newspaper,
    titleKey: "watchlists:emptyState.articles.title",
    titleDefault: "No updates yet",
    descriptionKey: "watchlists:emptyState.articles.description",
    descriptionDefault:
      "Updates are captured content from successful monitor runs, ready for review.",
    primaryCtaKey: "watchlists:emptyState.articles.primaryCta",
    primaryCtaDefault: "Set up a monitor to start capturing updates"
  },
  reports: {
    icon: FileOutput,
    titleKey: "watchlists:emptyState.reports.title",
    titleDefault: "No reports yet",
    descriptionKey: "watchlists:emptyState.reports.description",
    descriptionDefault:
      "Reports are generated briefings from monitor runs using your templates.",
    primaryCtaKey: "watchlists:emptyState.reports.primaryCta",
    primaryCtaDefault: "Run a monitor to generate your first report"
  },
  templates: {
    icon: FileText,
    titleKey: "watchlists:emptyState.templates.title",
    titleDefault: "No templates yet",
    descriptionKey: "watchlists:emptyState.templates.description",
    descriptionDefault:
      "Templates define the format and structure of your generated reports.",
    primaryCtaKey: "watchlists:emptyState.templates.primaryCta",
    primaryCtaDefault: "Create a template"
  }
}

export const WatchlistsEmptyState: React.FC<WatchlistsEmptyStateProps> = ({
  entity,
  onPrimaryAction,
  onSecondaryAction,
  primaryLabel,
  secondaryLabel,
  contextHint
}) => {
  const { t } = useTranslation(["watchlists"])
  const config = entityConfig[entity]
  const primaryActionLabel =
    primaryLabel || t(config.primaryCtaKey, config.primaryCtaDefault)
  const secondaryActionLabel =
    secondaryLabel ||
    (config.secondaryCtaKey
      ? t(config.secondaryCtaKey, config.secondaryCtaDefault || "")
      : undefined)

  return (
    <EmptyState
      title={t(config.titleKey, config.titleDefault)}
      icon={config.icon}
      iconClassName="text-text-muted"
      size="md"
      variant="card"
      description={
        <div className="space-y-2">
          <p className="text-sm text-text-muted">
            {t(config.descriptionKey, config.descriptionDefault)}
          </p>
          {contextHint && (
            <p className="text-xs text-text-muted italic">{contextHint}</p>
          )}
        </div>
      }
      data-testid={`watchlists-empty-state-${entity}`}
      primaryAction={
        onPrimaryAction
          ? {
              label: primaryActionLabel,
              icon: <Plus className="h-4 w-4" />,
              onClick: onPrimaryAction,
              "data-testid": `watchlists-empty-state-${entity}-primary`
            }
          : undefined
      }
      secondaryAction={
        onSecondaryAction && config.secondaryCtaKey && secondaryActionLabel
          ? {
              label: secondaryActionLabel,
              onClick: onSecondaryAction,
              "data-testid": `watchlists-empty-state-${entity}-secondary`
            }
          : undefined
      }
    />
  )
}
