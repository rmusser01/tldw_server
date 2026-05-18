import type {
  WatchlistOutput,
  WatchlistOutputCreate,
  WatchlistReportPreset,
  WatchlistReportReadiness,
  WatchlistReportReadinessState,
  WatchlistReportReadinessWarning,
  WatchlistReportReadinessWarningSeverity
} from "@/types/watchlists"

export interface DeliveryStatusSummary {
  channel: string
  status: string
  detail?: string
}

export interface DeliveryDisclosureSummary {
  visible: DeliveryStatusSummary[]
  hidden: DeliveryStatusSummary[]
}

const AUDIO_OUTPUT_FORMATS = new Set(["mp3", "wav", "ogg", "m4a", "aac", "flac", "opus"])
const REPORT_PRESETS = new Set<WatchlistReportPreset>([
  "auto",
  "cti_osint",
  "news_briefing",
  "general_research"
])
const READINESS_STATES = new Set<WatchlistReportReadinessState>([
  "ready",
  "warning",
  "blocked",
  "legacy_live_only"
])
const READINESS_WARNING_SEVERITIES = new Set<WatchlistReportReadinessWarningSeverity>([
  "info",
  "warning",
  "blocking"
])
const OUTPUT_MIME_TYPES: Record<string, string> = {
  md: "text/markdown",
  html: "text/html",
  mp3: "audio/mpeg",
  wav: "audio/wav",
  ogg: "audio/ogg",
  m4a: "audio/mp4",
  aac: "audio/aac",
  flac: "audio/flac",
  opus: "audio/ogg"
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const asNonEmptyString = (value: unknown): string | undefined => {
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

const asPositiveInteger = (value: unknown): number | undefined => {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) return undefined
  return value
}

const asNonNegativeInteger = (value: unknown): number | undefined => {
  if (typeof value === "number") {
    return Number.isInteger(value) && value >= 0 ? value : undefined
  }
  const fromString = asNonEmptyString(value)
  if (!fromString || !/^\d+$/.test(fromString)) return undefined
  const parsed = Number.parseInt(fromString, 10)
  return Number.isSafeInteger(parsed) ? parsed : undefined
}

const getMetadataRecord = (metadata: unknown): Record<string, unknown> | null =>
  isRecord(metadata) ? metadata : null

const normalizeOutputFormat = (format: unknown): string =>
  asNonEmptyString(format)?.toLowerCase() || ""

const isAudioTypeHint = (typeValue: unknown): boolean => {
  const normalized = asNonEmptyString(typeValue)?.toLowerCase() || ""
  return normalized.includes("audio") || normalized.includes("tts")
}

export const isAudioOutput = (
  output: Pick<WatchlistOutput, "format" | "type"> | null | undefined
): boolean => {
  if (!output) return false
  const normalizedFormat = normalizeOutputFormat(output.format)
  return AUDIO_OUTPUT_FORMATS.has(normalizedFormat) || isAudioTypeHint(output.type)
}

export const getOutputMimeType = (format: unknown): string => {
  const normalized = normalizeOutputFormat(format)
  return OUTPUT_MIME_TYPES[normalized] || "application/octet-stream"
}

export const getOutputFileExtension = (
  output: Pick<WatchlistOutput, "format" | "type"> | null | undefined
): string => {
  if (!output) return "txt"
  const normalizedFormat = normalizeOutputFormat(output.format)
  if (normalizedFormat.length > 0) return normalizedFormat
  if (isAudioTypeHint(output.type)) return "mp3"
  return "txt"
}

export const getOutputArtifactLabel = (
  output: Pick<WatchlistOutput, "format" | "type"> | null | undefined
): string => {
  if (!output) return "Output"
  if (isAudioOutput(output)) return "Audio briefing"
  const normalized = normalizeOutputFormat(output.format)
  if (normalized === "html") return "HTML"
  if (normalized === "md" || normalized === "markdown") return "Markdown"
  if (normalized.length > 0) return normalized.toUpperCase()
  return "Output"
}

export const getOutputArtifactTagColor = (
  output: Pick<WatchlistOutput, "format" | "type"> | null | undefined
): string => {
  if (!output) return "default"
  if (isAudioOutput(output)) return "purple"
  const normalized = normalizeOutputFormat(output.format)
  if (normalized === "html") return "blue"
  if (normalized === "md" || normalized === "markdown") return "green"
  return "default"
}

export const getOutputTemplateName = (metadata: unknown): string | undefined => {
  const record = getMetadataRecord(metadata)
  if (!record) return undefined
  return asNonEmptyString(record.template_name)
}

export const getOutputTemplateVersion = (metadata: unknown): number | undefined => {
  const record = getMetadataRecord(metadata)
  if (!record) return undefined
  const numeric = asPositiveInteger(record.template_version)
  if (numeric != null) return numeric
  const fromString = asNonEmptyString(record.template_version)
  if (!fromString) return undefined
  const parsed = Number.parseInt(fromString, 10)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined
}

const asReportPreset = (value: unknown): WatchlistReportPreset | undefined => {
  const preset = asNonEmptyString(value)?.toLowerCase()
  if (!preset) return undefined
  return REPORT_PRESETS.has(preset as WatchlistReportPreset)
    ? (preset as WatchlistReportPreset)
    : undefined
}

const asReadinessState = (value: unknown): WatchlistReportReadinessState | undefined => {
  const state = asNonEmptyString(value)?.toLowerCase()
  if (!state) return undefined
  return READINESS_STATES.has(state as WatchlistReportReadinessState)
    ? (state as WatchlistReportReadinessState)
    : undefined
}

const asReadinessWarningSeverity = (
  value: unknown
): WatchlistReportReadinessWarningSeverity => {
  const severity = asNonEmptyString(value)?.toLowerCase()
  if (severity && READINESS_WARNING_SEVERITIES.has(severity as WatchlistReportReadinessWarningSeverity)) {
    return severity as WatchlistReportReadinessWarningSeverity
  }
  return "warning"
}

const normalizeAffectedItemIds = (value: unknown): number[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => asNonNegativeInteger(item))
    .filter((item): item is number => item != null)
}

const normalizeReadinessWarnings = (value: unknown): WatchlistReportReadinessWarning[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => {
      const record = getMetadataRecord(entry)
      if (!record) return null
      const code = asNonEmptyString(record.code)
      const message = asNonEmptyString(record.message)
      if (!code || !message) return null
      return {
        code,
        severity: asReadinessWarningSeverity(record.severity),
        message,
        affected_item_ids: normalizeAffectedItemIds(record.affected_item_ids)
      }
    })
    .filter((entry): entry is WatchlistReportReadinessWarning => entry !== null)
}

export const getOutputReportPreset = (metadata: unknown): WatchlistReportPreset => {
  const record = getMetadataRecord(metadata)
  return asReportPreset(record?.report_preset) || "general_research"
}

export const getOutputReportSnapshotAvailable = (metadata: unknown): boolean => {
  const record = getMetadataRecord(metadata)
  return Boolean(asNonEmptyString(record?.report_snapshot_path))
}

export const getOutputReportReadiness = (metadata: unknown): WatchlistReportReadiness => {
  const record = getMetadataRecord(metadata)
  const readiness = getMetadataRecord(record?.report_readiness)
  const state = asReadinessState(readiness?.state)
  if (!readiness || !state) {
    return {
      state: "legacy_live_only",
      score: 0,
      warnings: []
    }
  }
  const score = asNonNegativeInteger(readiness.score)
  return {
    state,
    score: Math.min(score ?? 0, 100),
    warnings: normalizeReadinessWarnings(readiness.warnings)
  }
}

const getOutputReportCount = (metadata: unknown, key: string): number => {
  const record = getMetadataRecord(metadata)
  return asNonNegativeInteger(record?.[key]) ?? 0
}

export const getIncludedItemCount = (metadata: unknown): number =>
  getOutputReportCount(metadata, "included_item_count")

export const getExcludedItemCount = (metadata: unknown): number =>
  getOutputReportCount(metadata, "excluded_item_count")

export const getSourceCount = (metadata: unknown): number =>
  getOutputReportCount(metadata, "source_count")

export const getAlertCount = (metadata: unknown): number =>
  getOutputReportCount(metadata, "alert_count")

export const getWeakEvidenceWarningCount = (metadata: unknown): number =>
  getOutputReportCount(metadata, "weak_evidence_warning_count")

export const getReadinessTagColor = (state: WatchlistReportReadinessState): string => {
  if (state === "ready") return "green"
  if (state === "warning") return "gold"
  if (state === "blocked") return "red"
  return "default"
}

export const getReadinessLabel = (state: WatchlistReportReadinessState): string => {
  if (state === "ready") return "Ready"
  if (state === "warning") return "Needs review"
  if (state === "blocked") return "Blocked"
  return "Live provenance only"
}

const normalizeDelivery = (
  value: unknown,
  fallbackChannel?: string
): DeliveryStatusSummary | null => {
  if (typeof value === "string") {
    const status = asNonEmptyString(value)
    if (!status) return null
    return {
      channel: fallbackChannel || "delivery",
      status
    }
  }

  if (!isRecord(value)) return null

  const channel = asNonEmptyString(value.channel) || fallbackChannel || "delivery"
  const status = asNonEmptyString(value.status) || "unknown"
  const detail =
    asNonEmptyString(value.error) ||
    asNonEmptyString(value.message) ||
    asNonEmptyString(value.detail) ||
    asNonEmptyString(value.reason)

  return {
    channel,
    status,
    detail
  }
}

export const getOutputDeliveryStatuses = (metadata: unknown): DeliveryStatusSummary[] => {
  const record = getMetadataRecord(metadata)
  if (!record) return []

  const rawDeliveries = record.deliveries
  if (Array.isArray(rawDeliveries)) {
    return rawDeliveries
      .map((entry) => normalizeDelivery(entry))
      .filter((entry): entry is DeliveryStatusSummary => entry !== null)
  }

  if (isRecord(rawDeliveries)) {
    return Object.entries(rawDeliveries)
      .map(([channel, value]) => normalizeDelivery(value, channel))
      .filter((entry): entry is DeliveryStatusSummary => entry !== null)
  }

  return []
}

export const buildDeliveryDisclosureSummary = (
  deliveries: DeliveryStatusSummary[],
  options?: { maxVisible?: number }
): DeliveryDisclosureSummary => {
  const maxVisible = Math.max(1, Number(options?.maxVisible ?? 1))
  if (!Array.isArray(deliveries) || deliveries.length <= maxVisible) {
    return {
      visible: Array.isArray(deliveries) ? [...deliveries] : [],
      hidden: []
    }
  }
  return {
    visible: deliveries.slice(0, maxVisible),
    hidden: deliveries.slice(maxVisible)
  }
}

export const getDeliveryStatusColor = (status: string): string => {
  const normalized = status.trim().toLowerCase()
  if (normalized === "sent" || normalized === "stored" || normalized === "success") return "green"
  if (normalized === "partial" || normalized === "warning") return "gold"
  if (normalized === "queued" || normalized === "pending" || normalized === "in_progress") return "blue"
  if (normalized === "failed" || normalized === "error") return "red"
  return "default"
}

export const getDeliveryStatusLabel = (status: string): string => {
  const normalized = status.trim().toLowerCase()
  if (normalized === "sent") return "Sent"
  if (normalized === "stored") return "Stored"
  if (normalized === "success") return "Success"
  if (normalized === "partial") return "Partial"
  if (normalized === "warning") return "Warning"
  if (normalized === "queued") return "Queued"
  if (normalized === "pending") return "Pending"
  if (normalized === "in_progress") return "In progress"
  if (normalized === "skipped") return "Skipped"
  if (normalized === "failed") return "Failed"
  if (normalized === "error") return "Error"
  return status
}

interface BuildRegenerateOptions {
  title?: string | null
  templateName?: string | null
  templateVersion?: number | null
  allowTemplateOverrides?: boolean
}

export const buildRegenerateOutputRequest = (
  output: Pick<WatchlistOutput, "run_id" | "type">,
  options: BuildRegenerateOptions
): WatchlistOutputCreate => {
  const request: WatchlistOutputCreate = {
    run_id: output.run_id,
    type: output.type || undefined
  }

  const title = asNonEmptyString(options.title)
  if (title) {
    request.title = title
  }

  const allowTemplateOverrides =
    options.allowTemplateOverrides !== false && !isAudioTypeHint(output.type)
  if (allowTemplateOverrides) {
    const templateName = asNonEmptyString(options.templateName)
    if (templateName) {
      request.template_name = templateName
      const version = asPositiveInteger(options.templateVersion)
      if (version != null) {
        request.template_version = version
      }
    }
  }

  return request
}
