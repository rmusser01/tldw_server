import type { TFunction } from "i18next"

interface FormatRelativeTimeOptions {
  compact?: boolean
  /** BCP 47 language for dates older than a week; the browser's when unset. */
  locale?: string
}

export const formatRelativeTime = (
  isoString: string,
  t: TFunction,
  options: FormatRelativeTimeOptions = {}
): string => {
  const date = new Date(isoString)
  if (Number.isNaN(date.getTime())) {
    return t("relativeTime.unknown", "Unknown")
  }
  const { compact = false, locale } = options
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMins = Math.floor(diffMs / (1000 * 60))
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffMins < 1) {
    return t("relativeTime.justNow", "Just now")
  }

  if (compact) {
    const agoShort = t("relativeTime.agoShort", "ago")

    if (diffMins < 60) {
      return t("relativeTime.minutesAgoShort", {
        count: diffMins,
        defaultValue: `${diffMins}m ${agoShort}`
      })
    }

    if (diffHours < 24) {
      return t("relativeTime.hoursAgoShort", {
        count: diffHours,
        defaultValue: `${diffHours}h ${agoShort}`
      })
    }

    if (diffDays < 7) {
      return t("relativeTime.daysAgoShort", {
        count: diffDays,
        defaultValue: `${diffDays}d ${agoShort}`
      })
    }

    return date.toLocaleDateString(locale)
  }

  if (diffMins < 60) {
    const defaultLabel = `${diffMins} minute${diffMins !== 1 ? "s" : ""} ago`
    return t("relativeTime.minutesAgo", {
      count: diffMins,
      defaultValue: defaultLabel
    })
  }

  if (diffHours < 24) {
    const defaultLabel = `${diffHours} hour${diffHours !== 1 ? "s" : ""} ago`
    return t("relativeTime.hoursAgo", {
      count: diffHours,
      defaultValue: defaultLabel
    })
  }

  if (diffDays < 7) {
    const defaultLabel = `${diffDays} day${diffDays !== 1 ? "s" : ""} ago`
    return t("relativeTime.daysAgo", {
      count: diffDays,
      defaultValue: defaultLabel
    })
  }

  return date.toLocaleDateString(locale)
}
