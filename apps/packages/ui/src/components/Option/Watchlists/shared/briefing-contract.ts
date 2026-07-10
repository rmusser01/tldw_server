import type {
  BriefingPipelineContractV1,
  JobOutputPrefs,
  JobScope,
  WatchlistAudioCast,
  WatchlistFiltersPayload,
  WatchlistJobCreate,
  WatchlistProgramFormat
} from "@/types/watchlists"
import { normalizeWatchlistTemplateName } from "./templateNames"

export type { BriefingPipelineContractV1, WatchlistProgramFormat }

export const BRIEFING_PIPELINE_VERSION = 1 as const
export const DEFAULT_BRIEFING_SELECTION_MAX_ITEMS = 100
export const DEFAULT_BRIEFING_AUDIO_TARGET_MINUTES = 8

export const WATCHLIST_PROGRAM_FORMATS: readonly WatchlistProgramFormat[] = [
  "concise_briefing",
  "solo_update",
  "host_discussion",
  "sportscast",
  "culture_roundtable",
  "custom"
]

const LEGACY_BRIEFING_KEYS = new Set([
  "auto_output",
  "template",
  "template_name",
  "deliveries",
  "delivery_config",
  "generate_audio",
  "target_audio_minutes",
  "audio_language",
  "audio_provider",
  "audio_model",
  "audio_voice",
  "tts_provider",
  "tts_model",
  "tts_voice",
  "audio_speed",
  "audio_cast",
  "voice_map",
  "llm_provider",
  "llm_model",
  "persona_summarize",
  "persona_id",
  "persona_provider",
  "persona_model",
  "background_audio_uri",
  "background_volume",
  "background_delay_ms",
  "background_fade_seconds"
])

type UnknownRecord = Record<string, unknown>

export interface BriefingSetupDraft {
  monitorName: string
  scope: JobScope
  active: boolean
  scheduleExpr?: string | null
  timezone?: string | null
  description?: string
  jobFilters?: WatchlistFiltersPayload
  watchlistId?: number | null
  selectionMode?: "automatic" | "manual_override"
  maxItems?: number
  programFormat?: WatchlistProgramFormat
  outcomeNoun?: "briefing" | "episode"
  showName?: string
  premise?: string
  audience?: string
  tone?: string
  episodeTitlePattern?: string
  customInstructions?: string
  templateName: string
  templateFormat?: "md" | "html"
  templateVersion?: number
  showNotes?: boolean
  audioEnabled: boolean
  targetAudioMinutes?: number
  audioLanguage?: string
  audioProvider?: string
  audioModel?: string
  audioVoice?: string
  audioCast?: WatchlistAudioCast
  voiceMap?: Record<string, string>
  emailEnabled?: boolean
  emailRecipients?: string[]
  chatbookEnabled?: boolean
  chatbookTitle?: string
  preservedOutputPrefs?: JobOutputPrefs | null
}

export interface NormalizedLegacyBriefingContract {
  outputPrefs: JobOutputPrefs
  contract: BriefingPipelineContractV1
  warnings: string[]
}

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const cloneValue = <T>(value: T): T => {
  if (Array.isArray(value)) return value.map((entry) => cloneValue(entry)) as T
  if (!isRecord(value)) return value
  return Object.fromEntries(
    Object.entries(value).map(([key, entry]) => [key, cloneValue(entry)])
  ) as T
}

const record = (value: unknown): UnknownRecord =>
  isRecord(value) ? cloneValue(value) : {}

const deepMerge = (base: UnknownRecord, overlay: UnknownRecord): UnknownRecord => {
  const merged = cloneValue(base)
  for (const [key, value] of Object.entries(overlay)) {
    merged[key] = isRecord(value) && isRecord(merged[key])
      ? deepMerge(merged[key] as UnknownRecord, value)
      : cloneValue(value)
  }
  return merged
}

const clampInteger = (value: unknown, fallback: number, min: number, max: number): number => {
  if (
    value === null ||
    value === undefined ||
    (typeof value === "string" && value.trim().length === 0)
  ) {
    return fallback
  }
  const parsed = Number(value)
  const normalized = Number.isFinite(parsed) ? Math.floor(parsed) : fallback
  return Math.min(max, Math.max(min, normalized))
}

const coerceBoolean = (value: unknown, fallback = false): boolean => {
  if (typeof value === "boolean") return value
  const normalized = typeof value === "string" ? value.trim().toLowerCase() : value
  if (normalized === 1 || normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on") {
    return true
  }
  if (normalized === 0 || normalized === "0" || normalized === "false" || normalized === "no" || normalized === "off" || normalized === "") {
    return false
  }
  return fallback
}

const trimmed = (value: unknown): string =>
  typeof value === "string" ? value.trim() : ""

const normalizeRecipients = (value: unknown): string[] =>
  Array.isArray(value)
    ? Array.from(new Set(value.map((entry) => trimmed(entry)).filter(Boolean)))
    : []

const normalizeVoiceMap = (value: unknown): Record<string, string> | undefined => {
  if (!isRecord(value)) return undefined
  const entries = Object.entries(value)
    .map(([marker, voice]) => [marker.trim(), trimmed(voice)] as const)
    .filter(([marker, voice]) => marker.length > 0 && voice.length > 0)
  return entries.length > 0 ? Object.fromEntries(entries) : undefined
}

const normalizeAudioCast = (value: unknown): WatchlistAudioCast | undefined => {
  if (!isRecord(value) || !Array.isArray(value.speakers)) return undefined
  const speakers = value.speakers
    .filter(isRecord)
    .map((speaker, index) => ({
      id: trimmed(speaker.id) || `speaker_${index + 1}`,
      label: trimmed(speaker.label) || `Speaker ${index + 1}`,
      ...(trimmed(speaker.role) ? { role: trimmed(speaker.role) } : {}),
      voice: trimmed(speaker.voice),
      ...(trimmed(speaker.persona) ? { persona: trimmed(speaker.persona) } : {})
    }))
    .filter((speaker) => speaker.voice.length > 0)
    .slice(0, 4)
  if (speakers.length === 0) return undefined
  return {
    speaker_count: speakers.length as 1 | 2 | 3 | 4,
    speakers
  }
}

const baseContract = (): UnknownRecord => ({
  version: BRIEFING_PIPELINE_VERSION,
  selection: {
    mode: "automatic",
    max_items: DEFAULT_BRIEFING_SELECTION_MAX_ITEMS
  },
  editorial: {
    program_format: "concise_briefing",
    outcome_noun: "briefing"
  },
  text: {
    enabled: true,
    type: "briefing_markdown",
    format: "md",
    template_name: "",
    show_notes: false
  },
  audio: {
    enabled: false,
    language: "en"
  },
  delivery: {
    reports: { enabled: true },
    email: { enabled: false, recipients: [] },
    chatbook: { enabled: false }
  },
  test: {
    external_delivery: false,
    audio_sample_seconds: 60
  }
})

const legacyContract = (source: UnknownRecord): UnknownRecord => {
  const legacy: UnknownRecord = {}
  const autoOutput = record(source.auto_output)
  const template = record(source.template)
  const text: UnknownRecord = { ...autoOutput }
  if (Object.keys(template).length > 0) {
    for (const [key, value] of Object.entries(template)) {
      if (!["default_name", "default_version", "default_format"].includes(key)) {
        text[key] ??= cloneValue(value)
      }
    }
    text.template_name ??= template.default_name
    text.template_version ??= template.default_version
    text.format ??= template.default_format
  }
  if (source.template_name !== undefined) text.template_name = source.template_name
  if (Object.keys(text).length > 0) legacy.text = text

  const audio: UnknownRecord = {}
  if ("generate_audio" in source) audio.enabled = coerceBoolean(source.generate_audio)
  const audioKeys: Record<string, string> = {
    target_audio_minutes: "target_minutes",
    audio_language: "language",
    audio_provider: "provider",
    audio_model: "model",
    audio_voice: "voice",
    audio_speed: "speed",
    audio_cast: "cast",
    voice_map: "voice_map",
    llm_provider: "llm_provider",
    llm_model: "llm_model",
    persona_summarize: "persona_summarize",
    persona_id: "persona_id",
    persona_provider: "persona_provider",
    persona_model: "persona_model",
    background_audio_uri: "background_audio_uri",
    background_volume: "background_volume",
    background_delay_ms: "background_delay_ms",
    background_fade_seconds: "background_fade_seconds"
  }
  for (const [legacyKey, canonicalKey] of Object.entries(audioKeys)) {
    if (source[legacyKey] !== undefined) audio[canonicalKey] = cloneValue(source[legacyKey])
  }
  for (const [canonicalKey, aliases] of Object.entries({
    provider: ["tts_provider"],
    model: ["tts_model"],
    voice: ["tts_voice"]
  })) {
    if (audio[canonicalKey] !== undefined) continue
    const alias = aliases.find((key) => source[key] !== undefined)
    if (alias) audio[canonicalKey] = cloneValue(source[alias])
  }
  if (Object.keys(audio).length > 0) legacy.audio = audio

  const delivery = record(source.deliveries)
  const deliveryConfig = record(source.delivery_config)
  if (Object.keys(deliveryConfig).length > 0) {
    const email = record(delivery.email)
    if (deliveryConfig.email_recipients !== undefined) {
      email.recipients ??= cloneValue(deliveryConfig.email_recipients)
    }
    if (deliveryConfig.email_enabled !== undefined) {
      email.enabled ??= coerceBoolean(deliveryConfig.email_enabled)
    }
    if (Object.keys(email).length > 0) delivery.email = email
    const chatbook = record(delivery.chatbook)
    if (deliveryConfig.create_chatbook !== undefined) {
      chatbook.enabled ??= coerceBoolean(deliveryConfig.create_chatbook)
    }
    if (Object.keys(chatbook).length > 0) delivery.chatbook = chatbook
    for (const [key, value] of Object.entries(deliveryConfig)) {
      if (!["email_recipients", "email_enabled", "create_chatbook"].includes(key)) {
        delivery[key] ??= cloneValue(value)
      }
    }
  }
  if (Object.keys(delivery).length > 0) legacy.delivery = delivery
  return legacy
}

const finalizeContract = (raw: UnknownRecord): BriefingPipelineContractV1 => {
  const selection = record(raw.selection)
  const editorial = record(raw.editorial)
  const text = record(raw.text)
  const audio = record(raw.audio)
  const delivery = record(raw.delivery)
  const email = record(delivery.email)
  const chatbook = record(delivery.chatbook)
  const reports = record(delivery.reports)
  const test = record(raw.test)

  const programFormat = WATCHLIST_PROGRAM_FORMATS.includes(
    editorial.program_format as WatchlistProgramFormat
  )
    ? editorial.program_format as WatchlistProgramFormat
    : "concise_briefing"
  const audioEnabled = coerceBoolean(audio.enabled)
  const cast = normalizeAudioCast(audio.cast)
  const voiceMap = normalizeVoiceMap(audio.voice_map)
  const targetMinutes = audio.target_minutes === undefined && !audioEnabled
    ? undefined
    : clampInteger(
      audio.target_minutes,
      DEFAULT_BRIEFING_AUDIO_TARGET_MINUTES,
      1,
      60
    )
  const finalizedText: UnknownRecord = {
    ...text,
    enabled: true,
    type: "briefing_markdown",
    format: text.format === "html" ? "html" : "md",
    template_name: trimmed(text.template_name),
    show_notes: coerceBoolean(text.show_notes)
  }
  if (Number.isFinite(Number(text.template_version)) && Number(text.template_version) > 0) {
    finalizedText.template_version = Math.floor(Number(text.template_version))
  } else {
    delete finalizedText.template_version
  }
  const finalizedAudio: UnknownRecord = {
    ...audio,
    enabled: audioEnabled,
    language: trimmed(audio.language) || "en"
  }
  if (targetMinutes === undefined) delete finalizedAudio.target_minutes
  else finalizedAudio.target_minutes = targetMinutes
  if (trimmed(audio.voice)) finalizedAudio.voice = trimmed(audio.voice)
  else delete finalizedAudio.voice
  if (cast) finalizedAudio.cast = cast
  else delete finalizedAudio.cast
  if (voiceMap) finalizedAudio.voice_map = voiceMap
  else delete finalizedAudio.voice_map
  const finalizedChatbook: UnknownRecord = {
    ...chatbook,
    enabled: coerceBoolean(chatbook.enabled)
  }
  if (trimmed(chatbook.title)) finalizedChatbook.title = trimmed(chatbook.title)
  else delete finalizedChatbook.title

  return {
    ...raw,
    version: BRIEFING_PIPELINE_VERSION,
    selection: {
      ...selection,
      mode: selection.mode === "manual_override" ? "manual_override" : "automatic",
      max_items: clampInteger(
        selection.max_items,
        DEFAULT_BRIEFING_SELECTION_MAX_ITEMS,
        1,
        1000
      )
    },
    editorial: {
      ...editorial,
      program_format: programFormat,
      outcome_noun: editorial.outcome_noun === "episode" ? "episode" : "briefing"
    },
    text: finalizedText,
    audio: finalizedAudio,
    delivery: {
      ...delivery,
      reports: { ...reports, enabled: true },
      email: {
        ...email,
        enabled: coerceBoolean(email.enabled),
        recipients: normalizeRecipients(email.recipients)
      },
      chatbook: finalizedChatbook
    },
    test: {
      ...test,
      external_delivery: false,
      audio_sample_seconds: 60
    }
  } as BriefingPipelineContractV1
}

export const normalizeLegacyBriefingContract = (
  raw: JobOutputPrefs | null | undefined,
  _options: { scheduled: boolean }
): NormalizedLegacyBriefingContract => {
  const source = record(raw)
  const contract = finalizeContract(
    deepMerge(
      deepMerge(baseContract(), legacyContract(source)),
      record(source.briefing_pipeline)
    )
  )
  const legacyConsumed = Object.keys(source).some((key) => LEGACY_BRIEFING_KEYS.has(key))
  const outputPrefs = cloneValue(source)
  for (const key of LEGACY_BRIEFING_KEYS) delete outputPrefs[key]
  outputPrefs.briefing_pipeline = contract
  return {
    outputPrefs: outputPrefs as JobOutputPrefs,
    contract,
    warnings: legacyConsumed ? ["legacy_briefing_preferences_normalized"] : []
  }
}

const setOptionalString = (
  target: UnknownRecord,
  key: string,
  value: string | undefined
) => {
  if (value === undefined) return
  const normalized = value.trim()
  if (normalized) target[key] = normalized
  else delete target[key]
}

export const buildBriefingPipelineContract = (
  draft: BriefingSetupDraft
): BriefingPipelineContractV1 => {
  const normalized = normalizeLegacyBriefingContract(draft.preservedOutputPrefs, {
    scheduled: Boolean(trimmed(draft.scheduleExpr))
  })
  const existing = normalized.contract
  const preservedContract = record(draft.preservedOutputPrefs?.briefing_pipeline)
  const preservedEditorial = record(preservedContract.editorial)
  const preservedText = record(preservedContract.text)
  const format = draft.programFormat || existing.editorial.program_format
  const preservedOutcomeNoun = preservedEditorial.outcome_noun === "briefing" ||
    preservedEditorial.outcome_noun === "episode"
    ? existing.editorial.outcome_noun
    : undefined
  const outcomeNoun = draft.outcomeNoun || preservedOutcomeNoun || (
      format === "concise_briefing" ? "briefing" : "episode"
    )
  const editorial = record(existing.editorial)
  editorial.program_format = format
  editorial.outcome_noun = outcomeNoun
  setOptionalString(editorial, "show_name", draft.showName)
  setOptionalString(editorial, "premise", draft.premise)
  setOptionalString(editorial, "audience", draft.audience)
  setOptionalString(editorial, "tone", draft.tone)
  setOptionalString(editorial, "episode_title_pattern", draft.episodeTitlePattern)
  setOptionalString(editorial, "custom_instructions", draft.customInstructions)

  const text = record(existing.text)
  const normalizedTemplateName = normalizeWatchlistTemplateName(draft.templateName)
  text.enabled = true
  text.type = "briefing_markdown"
  text.format = draft.templateFormat || existing.text.format || "md"
  text.template_name = normalizedTemplateName
  if (draft.templateVersion !== undefined) {
    if (Number.isFinite(Number(draft.templateVersion)) && Number(draft.templateVersion) > 0) {
      text.template_version = Math.floor(Number(draft.templateVersion))
    } else {
      delete text.template_version
    }
  }
  const preservedShowNotes = Object.prototype.hasOwnProperty.call(
    preservedText,
    "show_notes"
  )
    ? existing.text.show_notes
    : undefined
  text.show_notes = draft.showNotes ?? preservedShowNotes ?? outcomeNoun === "episode"

  const audio = record(existing.audio)
  audio.enabled = Boolean(draft.audioEnabled)
  audio.language = trimmed(draft.audioLanguage) || trimmed(existing.audio.language) || "en"
  setOptionalString(audio, "provider", draft.audioProvider)
  setOptionalString(audio, "model", draft.audioModel)
  if (draft.audioEnabled) {
    audio.target_minutes = clampInteger(
      draft.targetAudioMinutes ?? existing.audio.target_minutes,
      DEFAULT_BRIEFING_AUDIO_TARGET_MINUTES,
      1,
      60
    )
  } else {
    delete audio.target_minutes
  }
  if (draft.audioVoice !== undefined) {
    if (trimmed(draft.audioVoice)) audio.voice = trimmed(draft.audioVoice)
    else delete audio.voice
  }
  if (draft.audioEnabled && !trimmed(audio.voice)) audio.voice = "alloy"
  let cast = draft.audioCast !== undefined
    ? normalizeAudioCast(draft.audioCast)
    : normalizeAudioCast(audio.cast)
  if (draft.audioEnabled && !cast) {
    cast = {
      speaker_count: 1,
      speakers: [
        {
          id: "speaker_1",
          label: "Speaker 1",
          role: "host",
          voice: trimmed(audio.voice) || "alloy"
        }
      ]
    }
  }
  if (cast) audio.cast = cast
  else delete audio.cast
  let voiceMap = draft.voiceMap !== undefined
    ? normalizeVoiceMap(draft.voiceMap)
    : normalizeVoiceMap(audio.voice_map)
  if (draft.audioEnabled && !voiceMap && cast) {
    voiceMap = Object.fromEntries(
      cast.speakers.map((speaker) => [speaker.id, speaker.voice])
    )
  }
  if (voiceMap) audio.voice_map = voiceMap
  else delete audio.voice_map

  const delivery = record(existing.delivery)
  delivery.reports = { ...record(delivery.reports), enabled: true }
  const email = record(delivery.email)
  const emailEnabled = draft.emailEnabled ?? coerceBoolean(email.enabled)
  email.enabled = emailEnabled
  email.recipients = emailEnabled
    ? normalizeRecipients(draft.emailRecipients ?? email.recipients)
    : []
  delivery.email = email
  const chatbook = record(delivery.chatbook)
  const chatbookEnabled = draft.chatbookEnabled ?? coerceBoolean(chatbook.enabled)
  chatbook.enabled = chatbookEnabled
  if (draft.chatbookTitle !== undefined) {
    if (trimmed(draft.chatbookTitle)) chatbook.title = trimmed(draft.chatbookTitle)
    else delete chatbook.title
  }
  delivery.chatbook = chatbook

  return finalizeContract({
    ...record(existing),
    version: BRIEFING_PIPELINE_VERSION,
    selection: {
      ...record(existing.selection),
      mode: draft.selectionMode || existing.selection.mode,
      max_items: clampInteger(
        draft.maxItems ?? existing.selection.max_items,
        DEFAULT_BRIEFING_SELECTION_MAX_ITEMS,
        1,
        1000
      )
    },
    editorial,
    text,
    audio,
    delivery,
    test: {
      ...record(existing.test),
      external_delivery: false,
      audio_sample_seconds: 60
    }
  })
}

export const toCanonicalWatchlistJobPayload = (
  draft: BriefingSetupDraft
): WatchlistJobCreate => {
  const normalized = normalizeLegacyBriefingContract(draft.preservedOutputPrefs, {
    scheduled: Boolean(trimmed(draft.scheduleExpr))
  })
  const contract = buildBriefingPipelineContract(draft)
  const scheduleExpr = trimmed(draft.scheduleExpr) || undefined
  const timezone = trimmed(draft.timezone) || undefined
  const description = trimmed(draft.description) || undefined
  const watchlistId = Number(draft.watchlistId)
  return {
    name: trimmed(draft.monitorName),
    ...(description ? { description } : {}),
    scope: cloneValue(draft.scope),
    active: draft.active,
    schedule_expr: scheduleExpr,
    timezone,
    output_prefs: {
      ...normalized.outputPrefs,
      briefing_pipeline: contract
    },
    ...(draft.jobFilters ? { job_filters: cloneValue(draft.jobFilters) } : {}),
    ...(Number.isFinite(watchlistId) && watchlistId > 0
      ? { watchlist_id: Math.floor(watchlistId) }
      : {})
  }
}
