import type { BriefingPipelineContractV1 } from "@/types/watchlists"

export interface BriefingReceiptInput {
  contract: BriefingPipelineContractV1
  sourceCount: number
  nextRunAt: string
  followingRunAt?: string
  timezone: string
  locale?: string
}

export interface BriefingReceiptModel {
  outcomeNoun: "briefing" | "episode"
  programFormat: BriefingPipelineContractV1["editorial"]["program_format"]
  speakerCount: number
  targetMinutes?: number
  sourceCount: number
  nextRunAt: string
  timezone: string
  timezoneAbbreviation: string
  nextRunLabel: string
  sentence: string
  dstNote?: string
}

const numberWords: Record<number, string> = {
  1: "one",
  2: "two",
  3: "three",
  4: "four"
}

const formatParts = (date: Date, timezone: string, locale: string) => {
  const formatter = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    weekday: "long",
    month: "long",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short"
  })
  return Object.fromEntries(
    formatter
      .formatToParts(date)
      .filter((part) => part.type !== "literal")
      .map((part) => [part.type, part.value])
  )
}

const timezoneOffset = (date: Date, timezone: string, locale: string): string => {
  const part = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    timeZoneName: "shortOffset"
  }).formatToParts(date).find((entry) => entry.type === "timeZoneName")
  return part?.value || ""
}

const formatOccurrence = (date: Date, timezone: string, locale: string) => {
  const parts = formatParts(date, timezone, locale)
  const nextRunLabel = `${parts.weekday}, ${parts.month} ${parts.day} at ${parts.hour}:${parts.minute} ${parts.dayPeriod} ${parts.timeZoneName}`
  return {
    nextRunLabel,
    timezoneAbbreviation: parts.timeZoneName || timezone
  }
}

const programLabel = (
  contract: BriefingPipelineContractV1,
  speakerCount: number
): string => {
  const format = contract.editorial.program_format.replaceAll("_", " ")
  if (!contract.audio.enabled) return "text report"
  if (speakerCount === 1) return `solo ${format}`
  if (speakerCount > 1) return `${numberWords[speakerCount] || speakerCount}-host ${format}`
  return `audio ${format}`
}

export const buildBriefingReceiptModel = (
  input: BriefingReceiptInput
): BriefingReceiptModel => {
  const locale = input.locale || "en-US"
  const nextRun = new Date(input.nextRunAt)
  const { nextRunLabel, timezoneAbbreviation } = formatOccurrence(
    nextRun,
    input.timezone,
    locale
  )
  const speakerCount = input.contract.audio.cast?.speaker_count || (
    input.contract.audio.enabled ? 1 : 0
  )
  const targetMinutes = input.contract.audio.enabled
    ? input.contract.audio.target_minutes
    : undefined
  const sourceCount = Math.max(0, Math.floor(Number(input.sourceCount) || 0))
  const sources = `${sourceCount} source${sourceCount === 1 ? "" : "s"}`
  const showName = input.contract.editorial.show_name
    ? ` for “${input.contract.editorial.show_name}”`
    : ""
  const target = targetMinutes === undefined ? "" : ` targeting ${targetMinutes} minutes`
  const generated = input.contract.audio.enabled
    ? `${input.contract.text.show_notes ? "show notes" : "a text report"} and a ${programLabel(input.contract, speakerCount)}${target}${showName}`
    : "a text report"
  const saved = input.contract.audio.enabled ? "save both in Reports" : "save it in Reports"
  const deliveries = [
    input.contract.delivery.email.enabled ? "email the outcome" : null,
    input.contract.delivery.chatbook.enabled ? "save it to Chatbook" : null
  ].filter((entry): entry is string => Boolean(entry))
  const delivery = deliveries.length > 0 ? `, and ${deliveries.join(" and ")}` : ""
  const sentence = `${nextRunLabel} (${input.timezone}), collect new items from ${sources}, generate ${generated}, ${saved}${delivery}.`

  let dstNote: string | undefined
  if (input.followingRunAt) {
    const followingRun = new Date(input.followingRunAt)
    if (
      timezoneOffset(nextRun, input.timezone, locale) !==
      timezoneOffset(followingRun, input.timezone, locale)
    ) {
      const following = formatOccurrence(followingRun, input.timezone, locale)
      dstNote = `The following run observes ${following.timezoneAbbreviation} in ${input.timezone} after the daylight-saving offset change.`
    }
  }

  return {
    outcomeNoun: input.contract.editorial.outcome_noun,
    programFormat: input.contract.editorial.program_format,
    speakerCount,
    targetMinutes,
    sourceCount,
    nextRunAt: input.nextRunAt,
    timezone: input.timezone,
    timezoneAbbreviation,
    nextRunLabel,
    sentence,
    ...(dstNote ? { dstNote } : {})
  }
}
