import type { AudioErrorCategory } from "@/components/Option/Audio/audio-error-classification"

export type TextPreviewMetadata = {
  inputTextLength: number
  inputTextPreview: string
  inputTextPreviewTruncated: boolean
  inputTextHash: string
}

export type SttComparisonConfig = {
  model: string
  language?: string
  task?: string
  responseFormat?: string
  timestampGranularities?: string[]
  segmentationEnabled?: boolean
  diarizationRequested?: boolean
}

export type SttComparisonMetadata = {
  createdAt: string
  audioSourceLabel: string
  audioSizeBytes?: number
  clientLatencyMs?: number
  language?: string
  durationSeconds?: number
  segmentCount?: number
  wordCount?: number
  errorCategory?: AudioErrorCategory
}

export type NormalizedSttResponse = {
  text: string
  metadata: Partial<
    Pick<
      SttComparisonMetadata,
      "language" | "durationSeconds" | "segmentCount" | "wordCount"
    >
  >
}

export type TtsResultMetadata = TextPreviewMetadata & {
  createdAt: string
  clientLatencyMs?: number
  audioSizeBytes?: number
}

export type BuildSttProvenanceTagsArgs = {
  metadata?: Partial<SttComparisonMetadata>
  config?: SttComparisonConfig
  latencyMs?: number
  wordCount?: number
  disabled?: boolean
  wordsLabel?: string
}

const BYTES_PER_KB = 1024
const BYTES_PER_MB = BYTES_PER_KB * 1024

export function formatByteSize(bytes?: number): string | undefined {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) {
    return undefined
  }
  if (bytes < BYTES_PER_KB) return `${bytes} B`
  if (bytes < BYTES_PER_MB) return `${(bytes / BYTES_PER_KB).toFixed(1)} KB`
  return `${(bytes / BYTES_PER_MB).toFixed(1)} MB`
}

export function formatClientLatency(ms?: number): string | undefined {
  if (typeof ms !== "number" || !Number.isFinite(ms) || ms < 0) {
    return undefined
  }
  if (ms < 1000) return `Client measured ${Math.round(ms)}ms`
  return `Client measured ${(ms / 1000).toFixed(1)}s`
}

export function formatCreatedAt(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  const normalized = date.toISOString()
  return `${normalized.slice(0, 10)} ${normalized.slice(11, 19)} UTC`
}

export function hashTextForLocalComparison(text: string): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return `local-${(hash >>> 0).toString(16).padStart(8, "0")}`
}

export function buildTextPreview(
  text: string,
  maxLength = 80
): TextPreviewMetadata {
  const normalized = text.replace(/\s+/g, " ").trim()
  const truncated = normalized.length > maxLength
  const previewBody = normalized.slice(0, maxLength)
  const wordBoundary = previewBody.lastIndexOf(" ")
  const truncatedPreview =
    wordBoundary > 0 ? previewBody.slice(0, wordBoundary) : previewBody
  return {
    inputTextLength: text.length,
    inputTextPreview: truncated ? `${truncatedPreview.trimEnd()}...` : normalized,
    inputTextPreviewTruncated: truncated,
    inputTextHash: hashTextForLocalComparison(text)
  }
}

const readStringOption = (
  options: Record<string, unknown>,
  key: string
): string | undefined => {
  const value = options[key]
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed ? trimmed : undefined
}

const readStringArrayOption = (
  options: Record<string, unknown>,
  key: string
): string[] | undefined => {
  const value = options[key]
  if (!Array.isArray(value)) return undefined
  const strings = value
    .map((item) => (typeof item === "string" ? item.trim() : ""))
    .filter((item) => item.length > 0)
  return strings.length > 0 ? strings : undefined
}

export function buildSttComparisonConfig(
  model: string,
  sttOptions: Record<string, unknown>
): SttComparisonConfig {
  const timestampGranularities =
    readStringArrayOption(sttOptions, "timestamp_granularities") ||
    readStringArrayOption(sttOptions, "timestampGranularities")
  const segmentationEnabled =
    sttOptions.segment === true || sttOptions.useSegmentation === true
  const diarizationRequested =
    sttOptions.diarize === true || sttOptions.diarization === true

  return {
    model: model.trim() || model,
    language: readStringOption(sttOptions, "language"),
    task: readStringOption(sttOptions, "task"),
    responseFormat:
      readStringOption(sttOptions, "response_format") ||
      readStringOption(sttOptions, "responseFormat"),
    timestampGranularities,
    segmentationEnabled: segmentationEnabled || undefined,
    diarizationRequested: diarizationRequested || undefined
  }
}

export function buildSttProvenanceTags({
  metadata,
  config,
  latencyMs,
  wordCount,
  disabled,
  wordsLabel = "words"
}: BuildSttProvenanceTagsArgs): string[] {
  const resolvedWordCount = metadata?.wordCount ?? wordCount
  return [
    metadata?.createdAt ? formatCreatedAt(metadata.createdAt) : undefined,
    metadata?.audioSourceLabel,
    formatByteSize(metadata?.audioSizeBytes),
    formatClientLatency(metadata?.clientLatencyMs ?? latencyMs),
    metadata?.language || config?.language
      ? `Language ${metadata?.language || config?.language}`
      : undefined,
    config?.task ? `Task ${config.task}` : undefined,
    config?.responseFormat ? `Format ${config.responseFormat}` : undefined,
    config?.timestampGranularities?.length
      ? `Timestamps ${config.timestampGranularities.join(", ")}`
      : undefined,
    config?.segmentationEnabled ? "Segmentation on" : undefined,
    config?.diarizationRequested ? "Diarization requested" : undefined,
    metadata?.durationSeconds != null
      ? `Duration ${metadata.durationSeconds.toFixed(1)}s`
      : undefined,
    metadata?.segmentCount != null
      ? `${metadata.segmentCount} ${
          metadata.segmentCount === 1 ? "segment" : "segments"
        }`
      : undefined,
    resolvedWordCount != null ? `${resolvedWordCount} ${wordsLabel}` : undefined,
    disabled ? "Disabled for Run All" : undefined
  ].filter((tag): tag is string => Boolean(tag))
}

function extractTextFromSegments(segments: unknown): string | undefined {
  if (!Array.isArray(segments)) return undefined
  const text = segments
    .map((segment) => {
      if (segment && typeof segment === "object") {
        const record = segment as Record<string, unknown>
        return typeof record.text === "string" ? record.text : ""
      }
      return ""
    })
    .filter(Boolean)
    .join(" ")
  return text || undefined
}

function readNumber(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key]
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

export function normalizeSttResponse(response: unknown): NormalizedSttResponse {
  if (typeof response === "string") {
    return { text: response, metadata: {} }
  }
  if (!response || typeof response !== "object") {
    return { text: "", metadata: {} }
  }

  const record = response as Record<string, unknown>
  const text =
    (typeof record.text === "string" ? record.text : undefined) ||
    (typeof record.transcript === "string" ? record.transcript : undefined) ||
    extractTextFromSegments(record.segments) ||
    ""

  const metadata: NormalizedSttResponse["metadata"] = {}
  if (typeof record.language === "string" && record.language.trim()) {
    metadata.language = record.language
  }

  const duration =
    readNumber(record, "duration") ?? readNumber(record, "duration_seconds")
  if (duration != null) metadata.durationSeconds = duration

  if (Array.isArray(record.segments)) {
    metadata.segmentCount = record.segments.length
  }
  if (Array.isArray(record.words)) {
    metadata.wordCount = record.words.length
  }

  return { text, metadata }
}

export function buildTtsResultMetadata(
  text: string,
  createdAt: string,
  extras: Pick<TtsResultMetadata, "clientLatencyMs" | "audioSizeBytes"> = {}
): TtsResultMetadata {
  return {
    ...buildTextPreview(text),
    createdAt,
    ...extras
  }
}
