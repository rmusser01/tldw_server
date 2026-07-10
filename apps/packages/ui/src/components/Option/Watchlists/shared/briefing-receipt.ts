import type { BriefingPipelineContractV1 } from "@/types/watchlists"

export type BriefingReceiptArtifact = "text_report" | "show_notes" | "audio"
export type BriefingReceiptDestination = "reports" | "email" | "chatbook"

export interface BriefingReceiptInput {
  contract: BriefingPipelineContractV1
  sourceCount: number
  nextRunAt?: string
  followingRunAt?: string
  scheduled?: boolean
  timezone: string
  locale?: string
}

export interface BriefingReceiptModel {
  scheduleMode: "manual" | "scheduled"
  outcomeNoun: "briefing" | "episode"
  programFormat: BriefingPipelineContractV1["editorial"]["program_format"]
  speakerCount: number
  targetMinutes?: number
  sourceCount: number
  nextRunAt?: string
  followingRunAt?: string
  timezone: string
  timezoneAbbreviation: string
  followingTimezoneAbbreviation?: string
  nextRunLabel?: string
  followingRunLabel?: string
  hasDstChange: boolean
  artifacts: BriefingReceiptArtifact[]
  destinations: BriefingReceiptDestination[]
  showName?: string
  emailRecipients: string[]
  chatbookTitle?: string
}

const occurrenceFormatter = (
  timezone: string,
  locale: string
): Intl.DateTimeFormat => new Intl.DateTimeFormat(locale, {
  timeZone: timezone,
  weekday: "long",
  month: "long",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  timeZoneName: "short"
})

const timezoneName = (
  formatter: Intl.DateTimeFormat,
  date: Date,
  fallback: string
): string => formatter
  .formatToParts(date)
  .find((part) => part.type === "timeZoneName")
  ?.value || fallback

const timezoneOffset = (date: Date, timezone: string, locale: string): string =>
  new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    timeZoneName: "shortOffset"
  }).formatToParts(date).find((entry) => entry.type === "timeZoneName")?.value || ""

const validDate = (value: string | undefined): Date | null => {
  if (!value) return null
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? null : date
}

export const buildBriefingReceiptModel = (
  input: BriefingReceiptInput
): BriefingReceiptModel => {
  const locale = input.locale || "en-US"
  const nextRun = validDate(input.nextRunAt)
  const followingRun = validDate(input.followingRunAt)
  const formatter = occurrenceFormatter(input.timezone, locale)
  const speakerCount = input.contract.audio.cast?.speaker_count || (
    input.contract.audio.enabled ? 1 : 0
  )
  const sourceCount = Math.max(0, Math.floor(Number(input.sourceCount) || 0))
  const emailRecipients = input.contract.delivery.email.enabled
    ? Array.from(new Set(input.contract.delivery.email.recipients)).sort()
    : []
  const chatbookTitle = input.contract.delivery.chatbook.enabled
    ? input.contract.delivery.chatbook.title?.trim() || undefined
    : undefined
  const artifacts: BriefingReceiptArtifact[] = [
    input.contract.text.show_notes ? "show_notes" : "text_report",
    ...(input.contract.audio.enabled ? ["audio" as const] : [])
  ]
  const destinations: BriefingReceiptDestination[] = [
    "reports",
    ...(emailRecipients.length > 0 ? ["email" as const] : []),
    ...(input.contract.delivery.chatbook.enabled ? ["chatbook" as const] : [])
  ]
  const timezoneAbbreviation = nextRun
    ? timezoneName(formatter, nextRun, input.timezone)
    : input.timezone
  const followingTimezoneAbbreviation = followingRun
    ? timezoneName(formatter, followingRun, input.timezone)
    : undefined
  const hasDstChange = Boolean(
    nextRun &&
    followingRun &&
    timezoneOffset(nextRun, input.timezone, locale) !==
      timezoneOffset(followingRun, input.timezone, locale)
  )

  return {
    scheduleMode: input.scheduled || nextRun ? "scheduled" : "manual",
    outcomeNoun: input.contract.editorial.outcome_noun,
    programFormat: input.contract.editorial.program_format,
    speakerCount,
    ...(input.contract.audio.enabled && input.contract.audio.target_minutes !== undefined
      ? { targetMinutes: input.contract.audio.target_minutes }
      : {}),
    sourceCount,
    ...(input.nextRunAt && nextRun ? { nextRunAt: input.nextRunAt } : {}),
    ...(input.followingRunAt && followingRun ? { followingRunAt: input.followingRunAt } : {}),
    timezone: input.timezone,
    timezoneAbbreviation,
    ...(followingTimezoneAbbreviation ? { followingTimezoneAbbreviation } : {}),
    ...(nextRun ? { nextRunLabel: formatter.format(nextRun) } : {}),
    ...(followingRun ? { followingRunLabel: formatter.format(followingRun) } : {}),
    hasDstChange,
    artifacts,
    destinations,
    ...(input.contract.editorial.show_name?.trim()
      ? { showName: input.contract.editorial.show_name.trim() }
      : {}),
    emailRecipients,
    ...(chatbookTitle ? { chatbookTitle } : {})
  }
}
