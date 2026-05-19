import type { FlashcardsImportError, StructuredQaImportPreviewDraft } from "@/services/flashcards"
import type { FlashcardsGenerateIntent } from "@/services/tldw/flashcards-generate-handoff"

import { getUtf8ByteLength } from "../../utils/field-byte-limit"

export interface ImportResultSummary {
  imported: number
  skipped: number
  errors: FlashcardsImportError[]
}

export interface ImportedCardReference {
  uuid: string
}

export interface GeneratedCardDraft {
  id: string
  front: string
  back: string
  tags: string[]
  model_type: "basic" | "basic_reverse" | "cloze"
  notes?: string | null
  extra?: string | null
}

export interface StructuredImportDraft extends StructuredQaImportPreviewDraft {
  id: string
  selected: boolean
  tags: string[]
}

export type SupportedDelimiter = "\t" | "," | ";" | "|"
export type ImportMode = "delimited" | "json" | "apkg" | "structured"
export type GenerateSourceType = Exclude<
  NonNullable<FlashcardsGenerateIntent["sourceType"]>,
  "manual"
>

export interface GenerateSourceContext {
  sourceType: GenerateSourceType
  sourceId: string | null
  sourceTitle: string | null
}

export interface GeneratePanelProps {
  initialIntent?: FlashcardsGenerateIntent | null
}

export type TransferActionStatus = "success" | "warning" | "error"

export interface TransferActionSummaryInput {
  area: "import" | "export" | "generate" | "occlusion"
  status: TransferActionStatus
  message: string
}

export interface TransferActionSummary extends TransferActionSummaryInput {
  at: string
}

export interface TransferActionReporterProps {
  onTransferAction?: (summary: TransferActionSummaryInput) => void
}

export const IMPORT_UNDO_SECONDS = 30
export const IMPORT_UNDO_CHUNK_SIZE = 50
export const LARGE_IMPORT_CONFIRM_THRESHOLD_ROWS = 300
export const LARGE_IMPORT_CONFIRM_THRESHOLD_APKG_BYTES = 5 * 1024 * 1024
export const APKG_ESTIMATED_BYTES_PER_CARD = 4096
export const SUPPORTED_DELIMITERS: SupportedDelimiter[] = ["\t", ",", ";", "|"]
export const NEW_DECK_OPTION_VALUE = "__new__" as const
export type DeckSelectionValue = number | typeof NEW_DECK_OPTION_VALUE | null | undefined
export const IMPORT_HELP_ANCHORS = {
  columns: "flashcards-import-help-columns",
  delimiter: "flashcards-import-help-delimiter",
  cloze: "flashcards-import-help-cloze",
  json: "flashcards-import-help-json"
} as const

export const detectJsonImportFormat = (rawContent: string): "json" | "jsonl" | "unknown" => {
  const trimmed = rawContent.trim()
  if (!trimmed) return "unknown"

  try {
    const parsed = JSON.parse(trimmed)
    if (Array.isArray(parsed)) return "json"
    if (parsed && typeof parsed === "object") return "json"
  } catch {
    // Continue to JSONL detection.
  }

  const lines = trimmed
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
  if (lines.length === 0) return "unknown"
  const allLinesJson = lines.every((line) => {
    try {
      const parsed = JSON.parse(line)
      return parsed && typeof parsed === "object" && !Array.isArray(parsed)
    } catch {
      return false
    }
  })
  return allLinesJson ? "jsonl" : "unknown"
}

export const estimateJsonItemCount = (rawContent: string): number => {
  const trimmed = rawContent.trim()
  if (!trimmed) return 0
  try {
    const parsed = JSON.parse(trimmed)
    if (Array.isArray(parsed)) return parsed.length
    if (
      parsed &&
      typeof parsed === "object" &&
      Array.isArray((parsed as Record<string, unknown>).items)
    ) {
      return ((parsed as Record<string, unknown>).items as unknown[]).length
    }
  } catch {
    // Fallback to JSONL line estimate.
  }
  return trimmed
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0).length
}

export const normalizeGeneratedCards = (value: unknown): GeneratedCardDraft[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry, index) => {
      if (!entry || typeof entry !== "object") return null
      const item = entry as Record<string, unknown>
      const front = String(item.front || "").trim()
      const back = String(item.back || "").trim()
      if (!front || !back) return null
      const modelTypeRaw = String(item.model_type || "basic").toLowerCase()
      const model_type: GeneratedCardDraft["model_type"] =
        modelTypeRaw === "cloze"
          ? "cloze"
          : modelTypeRaw === "basic_reverse"
            ? "basic_reverse"
            : "basic"
      const tagsRaw = item.tags
      const tags =
        Array.isArray(tagsRaw)
          ? tagsRaw.map((tag) => String(tag || "").trim()).filter((tag) => tag.length > 0)
          : typeof tagsRaw === "string"
            ? tagsRaw
                .replace(/,/g, " ")
                .split(/\s+/)
                .map((tag) => tag.trim())
                .filter((tag) => tag.length > 0)
            : []
      const draft: GeneratedCardDraft = {
        id: `generated-${index}-${front.slice(0, 16)}`,
        front,
        back,
        tags,
        model_type,
        notes: typeof item.notes === "string" ? item.notes : null,
        extra: typeof item.extra === "string" ? item.extra : null
      }
      return draft
    })
    .filter((item): item is GeneratedCardDraft => item !== null)
}

export const normalizeStructuredDrafts = (
  drafts: StructuredQaImportPreviewDraft[]
): StructuredImportDraft[] =>
  drafts.map((draft, index) => ({
    ...draft,
    id: `structured-${index}-${draft.line_start}`,
    selected: true,
    tags: Array.isArray(draft.tags)
      ? draft.tags.map((tag) => String(tag || "").trim()).filter((tag) => tag.length > 0)
      : []
  }))

export const normalizeImportErrors = (value: unknown): FlashcardsImportError[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null
      const row = entry as Record<string, unknown>
      const rawError = row.error
      if (typeof rawError !== "string" || rawError.trim().length === 0) {
        return null
      }
      const line = typeof row.line === "number" ? row.line : null
      const index = typeof row.index === "number" ? row.index : null
      const importError: FlashcardsImportError = {
        error: rawError,
        line,
        index
      }
      return importError
    })
    .filter((item): item is FlashcardsImportError => item !== null)
}

export const normalizeImportedItems = (value: unknown): ImportedCardReference[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null
      const row = entry as Record<string, unknown>
      const uuid = row.uuid
      if (typeof uuid !== "string" || uuid.trim().length === 0) return null
      return {
        uuid
      }
    })
    .filter((item): item is ImportedCardReference => item !== null)
}

export const buildStructuredDraftSaveError = (
  draft: StructuredImportDraft,
  maxFieldLength: number | null
): FlashcardsImportError | null => {
  const front = draft.front.trim()
  const back = draft.back.trim()

  if (!front) {
    return {
      line: draft.line_start,
      error: "Missing required field: Front"
    }
  }
  if (!back) {
    return {
      line: draft.line_start,
      error: "Missing required field: Back"
    }
  }
  if (maxFieldLength != null) {
    const fieldLengths = [
      ["Front", front],
      ["Back", back],
      ["Notes", draft.notes || ""],
      ["Extra", draft.extra || ""]
    ] as const
    const tooLongField = fieldLengths.find(
      ([, value]) => getUtf8ByteLength(value) > maxFieldLength
    )
    if (tooLongField) {
      return {
        line: draft.line_start,
        error: `Field too long: ${tooLongField[0]} (> ${maxFieldLength} bytes)`
      }
    }
  }
  return null
}

export const countDelimiterOccurrences = (line: string, delimiter: string): number =>
  Math.max(0, line.split(delimiter).length - 1)

export const normalizeHeaderToken = (value: string): string =>
  value.trim().toLowerCase().replace(/\s+/g, "").replace(/_/g, "")

export interface ImportErrorGuidance {
  copy: string
  helpAnchorId?: (typeof IMPORT_HELP_ANCHORS)[keyof typeof IMPORT_HELP_ANCHORS]
}

export const getImportErrorGuidance = (
  error: string,
  t: (key: string, options?: Record<string, unknown>) => string
): ImportErrorGuidance | null => {
  const normalized = error.toLowerCase()
  if (normalized.includes("missing required field: front")) {
    return {
      copy: t("option:flashcards.importGuidanceMissingFront", {
        defaultValue:
          "Add a non-empty Front value on that row, or map your header to the Front column."
      }),
      helpAnchorId: IMPORT_HELP_ANCHORS.columns
    }
  }
  if (normalized.includes("missing required field: deck")) {
    return {
      copy: t("option:flashcards.importGuidanceMissingDeck", {
        defaultValue:
          "Add a Deck value, or remove/rename the Deck header if your file uses a different column."
      }),
      helpAnchorId: IMPORT_HELP_ANCHORS.columns
    }
  }
  if (normalized.includes("invalid cloze")) {
    return {
      copy: t("option:flashcards.importGuidanceInvalidCloze", {
        defaultValue:
          "For cloze rows, include at least one deletion in Front like {{c1::answer}}."
      }),
      helpAnchorId: IMPORT_HELP_ANCHORS.cloze
    }
  }
  if (normalized.includes("field too long")) {
    return {
      copy: t("option:flashcards.importGuidanceFieldTooLong", {
        defaultValue:
          "Shorten the referenced field so its UTF-8 size fits your configured field byte limit."
      }),
      helpAnchorId: IMPORT_HELP_ANCHORS.columns
    }
  }
  if (normalized.includes("line too long")) {
    return {
      copy: t("option:flashcards.importGuidanceLineTooLong", {
        defaultValue:
          "Check delimiter choice and line breaks; malformed rows can produce oversized lines."
      }),
      helpAnchorId: IMPORT_HELP_ANCHORS.delimiter
    }
  }
  if (normalized.includes("maximum import")) {
    return {
      copy: t("option:flashcards.importGuidanceMaxLimit", {
        defaultValue:
          "Split this file into smaller batches, then import each batch separately."
      })
    }
  }
  return null
}
