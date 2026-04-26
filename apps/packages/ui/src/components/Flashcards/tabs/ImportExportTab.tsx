import React from "react"
import { Link } from "react-router-dom"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Alert,
  Button,
  Card,
  Collapse,
  Form,
  Input,
  Modal,
  Select,
  Space,
  Switch,
  Tooltip,
  Typography
} from "antd"
import { useTranslation } from "react-i18next"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { useUndoNotification } from "@/hooks/useUndoNotification"
import type { DeckReviewPromptSide } from "@/services/flashcards"
import { processInChunks } from "@/utils/chunk-processing"
import { ImageOcclusionTransferPanel } from "./ImageOcclusionTransferPanel"
import { NewDeckConfigurationFields } from "../components/NewDeckConfigurationFields"
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useCreateFlashcardsBulkMutation,
  useDecksQuery,
  useGenerateFlashcardsMutation,
  useImportFlashcardsMutation,
  useImportFlashcardsApkgMutation,
  useImportFlashcardsJsonMutation,
  useImportLimitsQuery,
  usePreviewStructuredQaImportMutation
} from "../hooks"
import { getUtf8ByteLength } from "../utils/field-byte-limit"
import { useDeckSchedulerDraft } from "../hooks/useDeckSchedulerDraft"
import { FileDropZone } from "../components"
import { FLASHCARDS_HELP_LINKS } from "../constants"
import { formatSchedulerSummary } from "../utils/scheduler-settings"
import {
  deleteFlashcard,
  exportFlashcards,
  exportFlashcardsFile,
  getFlashcard,
  listFlashcards,
  type FlashcardsImportError,
  type StructuredQaImportPreviewDraft
} from "@/services/flashcards"
import type { FlashcardsGenerateIntent } from "@/services/tldw/flashcards-generate-handoff"
import { getLlmProviders } from "@/services/prompt-studio"
import type { StudyPackIntent } from "@/services/tldw/study-pack-handoff"
import { StudyPackCreateDrawer } from "../components/StudyPackCreateDrawer"

const { Text } = Typography

interface ImportResultSummary {
  imported: number
  skipped: number
  errors: FlashcardsImportError[]
}

interface ImportedCardReference {
  uuid: string
}

interface GeneratedCardDraft {
  id: string
  front: string
  back: string
  tags: string[]
  model_type: "basic" | "basic_reverse" | "cloze"
  notes?: string | null
  extra?: string | null
}

interface StructuredImportDraft extends StructuredQaImportPreviewDraft {
  id: string
  selected: boolean
  tags: string[]
}

type SupportedDelimiter = "\t" | "," | ";" | "|"
type ImportMode = "delimited" | "json" | "apkg" | "structured"
type GenerateSourceType = Exclude<
  NonNullable<FlashcardsGenerateIntent["sourceType"]>,
  "manual"
>

interface GenerateSourceContext {
  sourceType: GenerateSourceType
  sourceId: string | null
  sourceTitle: string | null
}

interface GeneratePanelProps {
  initialIntent?: FlashcardsGenerateIntent | null
}

type TransferActionStatus = "success" | "warning" | "error"

interface TransferActionSummaryInput {
  area: "import" | "export" | "generate" | "occlusion"
  status: TransferActionStatus
  message: string
}

interface TransferActionSummary extends TransferActionSummaryInput {
  at: string
}

interface TransferActionReporterProps {
  onTransferAction?: (summary: TransferActionSummaryInput) => void
}

const IMPORT_UNDO_SECONDS = 30
const IMPORT_UNDO_CHUNK_SIZE = 50
const LARGE_IMPORT_CONFIRM_THRESHOLD_ROWS = 300
const LARGE_IMPORT_CONFIRM_THRESHOLD_APKG_BYTES = 5 * 1024 * 1024
const APKG_ESTIMATED_BYTES_PER_CARD = 4096
const SUPPORTED_DELIMITERS: SupportedDelimiter[] = ["\t", ",", ";", "|"]
const NEW_DECK_OPTION_VALUE = "__new__" as const
type DeckSelectionValue = number | typeof NEW_DECK_OPTION_VALUE | null | undefined
const IMPORT_HELP_ANCHORS = {
  columns: "flashcards-import-help-columns",
  delimiter: "flashcards-import-help-delimiter",
  cloze: "flashcards-import-help-cloze",
  json: "flashcards-import-help-json"
} as const

const detectJsonImportFormat = (rawContent: string): "json" | "jsonl" | "unknown" => {
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

const estimateJsonItemCount = (rawContent: string): number => {
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

const normalizeGeneratedCards = (value: unknown): GeneratedCardDraft[] => {
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

const normalizeStructuredDrafts = (
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

const normalizeImportErrors = (value: unknown): FlashcardsImportError[] => {
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

const normalizeImportedItems = (value: unknown): ImportedCardReference[] => {
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

const buildStructuredDraftSaveError = (
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

const countDelimiterOccurrences = (line: string, delimiter: string): number =>
  Math.max(0, line.split(delimiter).length - 1)

const normalizeHeaderToken = (value: string): string =>
  value.trim().toLowerCase().replace(/\s+/g, "").replace(/_/g, "")

interface ImportErrorGuidance {
  copy: string
  helpAnchorId?: (typeof IMPORT_HELP_ANCHORS)[keyof typeof IMPORT_HELP_ANCHORS]
}

const getImportErrorGuidance = (
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

/**
 * Import panel for CSV/TSV flashcard import.
 */
const ImportPanel: React.FC<TransferActionReporterProps> = ({ onTransferAction }) => {
  const qc = useQueryClient()
  const message = useAntdMessage()
  const { showUndoNotification } = useUndoNotification()
  const { t } = useTranslation(["option", "common"])
  const limitsQuery = useImportLimitsQuery()
  const decksQuery = useDecksQuery()
  const createDeckMutation = useCreateDeckMutation()
  const createBulkMutation = useCreateFlashcardsBulkMutation()
  const importMutation = useImportFlashcardsMutation()
  const importJsonMutation = useImportFlashcardsJsonMutation()
  const importApkgMutation = useImportFlashcardsApkgMutation()
  const previewStructuredMutation = usePreviewStructuredQaImportMutation()
  const decks = decksQuery.data || []

  const [content, setContent] = React.useState("")
  const [importMode, setImportMode] = React.useState<ImportMode>("delimited")
  const [apkgFile, setApkgFile] = React.useState<File | null>(null)
  const [delimiter, setDelimiter] = React.useState<string>("\t")
  const [hasHeader, setHasHeader] = React.useState<boolean>(true)
  const [lastResult, setLastResult] = React.useState<ImportResultSummary | null>(null)
  const [structuredDrafts, setStructuredDrafts] = React.useState<StructuredImportDraft[]>([])
  const [structuredPreviewErrors, setStructuredPreviewErrors] = React.useState<
    FlashcardsImportError[]
  >([])
  const [structuredTargetDeckId, setStructuredTargetDeckId] =
    React.useState<DeckSelectionValue>(undefined)
  const [structuredNewDeckName, setStructuredNewDeckName] = React.useState(() =>
    t("option:flashcards.structuredImportDeckName", {
      defaultValue: "Structured Import"
    })
  )
  const [structuredReviewPromptSide, setStructuredReviewPromptSide] =
    React.useState<DeckReviewPromptSide>("front")
  const [confirmLargeImportOpen, setConfirmLargeImportOpen] = React.useState(false)
  const [importHelpActiveKeys, setImportHelpActiveKeys] = React.useState<string[]>([
    "columns"
  ])
  const structuredSchedulerDraft = useDeckSchedulerDraft()
  const structuredSelectedDeck = React.useMemo(
    () =>
      typeof structuredTargetDeckId === "number"
        ? decks.find((deck) => deck.id === structuredTargetDeckId) ?? null
        : null,
    [decks, structuredTargetDeckId]
  )
  const structuredDeckOptions = React.useMemo(
    () => [
      ...decks.map((deck) => ({
        label: deck.name,
        value: deck.id
      })),
      {
        label: t("option:flashcards.createNewDeck", {
          defaultValue: "Create new deck"
        }),
        value: NEW_DECK_OPTION_VALUE
      }
    ],
    [decks, t]
  )

  const selectedDelimiterLabel = React.useMemo(() => {
    if (delimiter === "\t") {
      return t("option:flashcards.tab", { defaultValue: "Tab" })
    }
    if (delimiter === ",") {
      return t("option:flashcards.commaShort", { defaultValue: "Comma" })
    }
    if (delimiter === ";") {
      return t("option:flashcards.semicolonShort", { defaultValue: "Semicolon" })
    }
    return t("option:flashcards.pipeShort", { defaultValue: "Pipe" })
  }, [delimiter, t])

  React.useEffect(() => {
    if (structuredTargetDeckId !== undefined) return
    if (decks.length > 0) {
      setStructuredTargetDeckId(decks[0].id)
      return
    }
    setStructuredTargetDeckId(NEW_DECK_OPTION_VALUE)
  }, [decks, structuredTargetDeckId])

  const structuredMaxFieldLength = React.useMemo(() => {
    const rawValue = limitsQuery.data?.max_field_length
    return typeof rawValue === "number" && rawValue > 0 ? rawValue : null
  }, [limitsQuery.data])

  const updateStructuredDraft = React.useCallback(
    (id: string, patch: Partial<StructuredImportDraft>) => {
      setStructuredDrafts((prev) =>
        prev.map((draft) => (draft.id === id ? { ...draft, ...patch } : draft))
      )
    },
    []
  )

  const removeStructuredDraft = React.useCallback((id: string) => {
    setStructuredDrafts((prev) => prev.filter((draft) => draft.id !== id))
  }, [])

  const resolveStructuredTargetDeckId = React.useCallback(async (): Promise<number> => {
    if (typeof structuredTargetDeckId === "number") return structuredTargetDeckId
    if (structuredTargetDeckId === undefined && decks.length > 0) {
      return decks[0].id
    }
    if (
      structuredTargetDeckId === NEW_DECK_OPTION_VALUE ||
      (structuredTargetDeckId == null && decks.length === 0)
    ) {
      const name = structuredNewDeckName.trim()
      if (!name) {
        throw new Error(
          t("option:flashcards.newDeckNameRequired", {
            defaultValue: "Enter a deck name."
          })
        )
      }
      const schedulerSettings = structuredSchedulerDraft.getValidatedSettings()
      if (!schedulerSettings) {
        throw new Error(
          t("option:flashcards.schedulerDraftInvalid", {
            defaultValue: "Draft has validation errors."
          })
        )
      }
      const createdDeck = await createDeckMutation.mutateAsync({
        name,
        review_prompt_side: structuredReviewPromptSide,
        scheduler_type: schedulerSettings.scheduler_type,
        scheduler_settings: schedulerSettings.scheduler_settings
      })
      setStructuredTargetDeckId(createdDeck.id)
      return createdDeck.id
    }
    if (structuredTargetDeckId == null && decks.length > 0) {
      return decks[0].id
    }
    throw new Error(
      t("option:flashcards.newDeckNameRequired", {
        defaultValue: "Enter a deck name."
      })
    )
  }, [
    createDeckMutation,
    decks,
    structuredNewDeckName,
    structuredReviewPromptSide,
    structuredSchedulerDraft,
    structuredTargetDeckId,
    t
  ])

  const scrollToImportHelp = React.useCallback((anchorId?: string) => {
    if (!anchorId || typeof document === "undefined") return
    const panelKey =
      anchorId === IMPORT_HELP_ANCHORS.delimiter
        ? "delimiter"
        : anchorId === IMPORT_HELP_ANCHORS.json
          ? "json"
          : "columns"
    setImportHelpActiveKeys((prev) =>
      prev.includes(panelKey) ? prev : [...prev, panelKey]
    )
    const schedule =
      typeof window !== "undefined" && typeof window.requestAnimationFrame === "function"
        ? window.requestAnimationFrame
        : (cb: FrameRequestCallback) => window.setTimeout(cb, 0)
    schedule(() => {
      const target = document.getElementById(anchorId)
      target?.scrollIntoView({ behavior: "smooth", block: "center" })
    })
  }, [])

  const importPreflightWarning = React.useMemo(() => {
    if (importMode !== "delimited") return null
    const sampleLine = content
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) => line.length > 0)
    if (!sampleLine) return null

    const selectedCount = countDelimiterOccurrences(sampleLine, delimiter)
    const bestAlternative = SUPPORTED_DELIMITERS
      .filter((candidate) => candidate !== delimiter)
      .map((candidate) => ({
        delimiter: candidate,
        count: countDelimiterOccurrences(sampleLine, candidate)
      }))
      .sort((a, b) => b.count - a.count)[0]

    if (selectedCount === 0 && bestAlternative && bestAlternative.count > 0) {
      const suggested =
        bestAlternative.delimiter === "\t"
          ? t("option:flashcards.tab", { defaultValue: "Tab" })
          : bestAlternative.delimiter === ","
            ? t("option:flashcards.commaShort", { defaultValue: "Comma" })
            : bestAlternative.delimiter === ";"
              ? t("option:flashcards.semicolonShort", { defaultValue: "Semicolon" })
              : t("option:flashcards.pipeShort", { defaultValue: "Pipe" })
      return t("option:flashcards.importPreflightDelimiterMismatch", {
        defaultValue:
          "Selected delimiter ({{selected}}) may be incorrect. This sample looks {{suggested}}-delimited.",
        selected: selectedDelimiterLabel,
        suggested
      })
    }

    if (hasHeader && selectedCount > 0) {
      const tokens = sampleLine.split(delimiter).map(normalizeHeaderToken)
      const hasFront = tokens.some((token) => token === "front" || token === "question")
      const hasBack = tokens.some((token) => token === "back" || token === "answer")
      if (!hasFront || !hasBack) {
        return t("option:flashcards.importPreflightHeaderColumns", {
          defaultValue:
            "Header is missing Front/Back columns. Accepted names include Deck, Front, Back, Tags, Notes, Extra, Model_Type, Reverse, Is_Cloze.",
        })
      }
    }

    return null
  }, [content, delimiter, hasHeader, importMode, selectedDelimiterLabel, t])

  const detectedJsonImportFormat = React.useMemo(
    () => detectJsonImportFormat(content),
    [content]
  )

  const nonEmptyLineCount = React.useMemo(
    () =>
      content
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0).length,
    [content]
  )
  const estimatedImportRows = Math.max(0, nonEmptyLineCount - (hasHeader ? 1 : 0))
  const apkgFileSizeBytes = apkgFile?.size ?? 0
  const estimatedApkgItems = apkgFile
    ? Math.max(1, Math.round(apkgFileSizeBytes / APKG_ESTIMATED_BYTES_PER_CARD))
    : 0
  const estimatedImportItems =
    importMode === "delimited"
      ? estimatedImportRows
      : importMode === "json"
        ? estimateJsonItemCount(content)
        : importMode === "structured"
          ? nonEmptyLineCount
          : estimatedApkgItems
  const importPayloadBytes = getUtf8ByteLength(content)
  const requiresLargeImportConfirmation =
    importMode === "apkg"
      ? apkgFileSizeBytes >= LARGE_IMPORT_CONFIRM_THRESHOLD_APKG_BYTES ||
        estimatedImportItems >= LARGE_IMPORT_CONFIRM_THRESHOLD_ROWS
      : importMode === "structured"
        ? false
        : estimatedImportItems >= LARGE_IMPORT_CONFIRM_THRESHOLD_ROWS

  const invalidateFlashcardQueries = React.useCallback(async () => {
    await qc.invalidateQueries({
      predicate: (query) =>
        Array.isArray(query.queryKey) &&
        typeof query.queryKey[0] === "string" &&
        query.queryKey[0].startsWith("flashcards:")
    })
  }, [qc])

  const handleStructuredPreview = React.useCallback(async () => {
    try {
      setLastResult(null)
      const preview = await previewStructuredMutation.mutateAsync({
        content
      })
      const drafts = normalizeStructuredDrafts(preview.drafts)
      setStructuredDrafts(drafts)
      setStructuredPreviewErrors(normalizeImportErrors(preview.errors))

      if (drafts.length === 0) {
        const warningCopy = t("option:flashcards.structuredPreviewEmpty", {
          defaultValue:
            "No labeled Q&A pairs were detected. Use Q:/A: or Question:/Answer: labels."
        })
        message.warning(warningCopy)
        onTransferAction?.({
          area: "import",
          status: "warning",
          message: warningCopy
        })
        return
      }

      const successCopy = t("option:flashcards.structuredPreviewReady", {
        defaultValue: "Prepared {{count}} structured drafts for review.",
        count: drafts.length
      })
      message.success(successCopy)
      onTransferAction?.({
        area: "import",
        status: preview.errors.length > 0 ? "warning" : "success",
        message: successCopy
      })
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : "Structured preview failed"
      message.error(errorMessage)
      onTransferAction?.({
        area: "import",
        status: "error",
        message: errorMessage
      })
    }
  }, [content, message, onTransferAction, previewStructuredMutation, t])

  const handleSaveStructuredDrafts = React.useCallback(async () => {
    const selectedDrafts = structuredDrafts.filter((draft) => draft.selected)
    const draftValidationResults = selectedDrafts.map((draft) => ({
      draft,
      error: buildStructuredDraftSaveError(draft, structuredMaxFieldLength)
    }))
    const savableDrafts = draftValidationResults
      .filter((entry) => entry.error === null)
      .map((entry) => entry.draft)
    const skippedSaveErrors = draftValidationResults.flatMap((entry) =>
      entry.error ? [entry.error] : []
    )

    if (savableDrafts.length === 0) {
      message.warning(
        t("option:flashcards.structuredSaveNoneSelected", {
          defaultValue: "Select at least one valid draft to save."
        })
      )
      return
    }

    try {
      const deckId = await resolveStructuredTargetDeckId()
      const payload = savableDrafts.map((draft) => ({
        deck_id: deckId,
        front: draft.front.trim(),
        back: draft.back.trim(),
        notes: draft.notes || undefined,
        extra: draft.extra || undefined,
        tags: draft.tags,
        model_type: "basic" as const,
        reverse: false,
        is_cloze: false,
        source_ref_type: "manual" as const
      }))
      const created = await createBulkMutation.mutateAsync(payload)
      const createdItems = normalizeImportedItems(created.items)
      const submittedDraftIds = new Set(savableDrafts.map((draft) => draft.id))
      const resultErrors = [...structuredPreviewErrors, ...skippedSaveErrors]

      setStructuredDrafts((prev) =>
        prev.filter((draft) => !submittedDraftIds.has(draft.id))
      )
      setLastResult({
        imported: createdItems.length,
        skipped: resultErrors.length,
        errors: resultErrors
      })

      const saveFeedbackCopy =
        resultErrors.length > 0
          ? t("option:flashcards.structuredSavePartial", {
              defaultValue:
                "Saved {{count}} structured cards, skipped {{skipped}} drafts.",
              count: createdItems.length,
              skipped: resultErrors.length
            })
          : t("option:flashcards.structuredSaveSuccess", {
              defaultValue: "Saved {{count}} structured cards.",
              count: createdItems.length
            })
      if (resultErrors.length > 0) {
        message.warning(saveFeedbackCopy)
      } else {
        message.success(saveFeedbackCopy)
      }
      onTransferAction?.({
        area: "import",
        status: resultErrors.length > 0 ? "warning" : "success",
        message: saveFeedbackCopy
      })

      if (createdItems.length > 0) {
        showUndoNotification({
          title: t("option:flashcards.structuredUndoTitle", {
            defaultValue: "Structured import saved"
          }),
          description: t("option:flashcards.importUndoHint", {
            defaultValue:
              "Undo within {{seconds}}s to remove {{count}} imported cards.",
            seconds: IMPORT_UNDO_SECONDS,
            count: createdItems.length
          }),
          duration: IMPORT_UNDO_SECONDS,
          onUndo: async () => {
            let failedRollbacks = 0
            await processInChunks(createdItems, IMPORT_UNDO_CHUNK_SIZE, async (chunk) => {
              const results = await Promise.allSettled(
                chunk.map(async (item) => {
                  const latest = await getFlashcard(item.uuid)
                  await deleteFlashcard(item.uuid, latest.version)
                })
              )
              failedRollbacks += results.filter((result) => result.status === "rejected").length
            })
            await invalidateFlashcardQueries()
            if (failedRollbacks > 0) {
              throw new Error(
                t("option:flashcards.importUndoPartialFailure", {
                  defaultValue: "Some imported cards could not be rolled back."
                })
              )
            }
          }
        })
      }
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : "Structured save failed"
      message.error(errorMessage)
      onTransferAction?.({
        area: "import",
        status: "error",
        message: errorMessage
      })
    }
  }, [
    createBulkMutation,
    invalidateFlashcardQueries,
    message,
    onTransferAction,
    resolveStructuredTargetDeckId,
    showUndoNotification,
    structuredDrafts,
    structuredMaxFieldLength,
    structuredPreviewErrors,
    t
  ])

  const performImport = React.useCallback(async () => {
    try {
      const result =
        importMode === "delimited"
          ? await importMutation.mutateAsync({
              content,
              delimiter,
              hasHeader
            })
          : importMode === "json"
            ? await importJsonMutation.mutateAsync({
                content,
                filename:
                  detectedJsonImportFormat === "jsonl"
                    ? "flashcards.jsonl"
                    : "flashcards.json"
              })
            : await (async () => {
                if (!apkgFile) {
                  throw new Error(
                    t("option:flashcards.importApkgMissingFile", {
                      defaultValue: "Select an APKG file before importing."
                    })
                  )
                }
                const fileBytes = new Uint8Array(await apkgFile.arrayBuffer())
                return importApkgMutation.mutateAsync({
                  bytes: fileBytes,
                  filename: apkgFile.name
                })
              })()
      const importedItems = normalizeImportedItems(result.items)
      const imported =
        typeof result.imported === "number"
          ? result.imported
          : importedItems.length
      const errors = normalizeImportErrors(result.errors)
      const skipped = errors.length

      setLastResult({
        imported,
        skipped,
        errors
      })

      if (errors.length > 0) {
        const warningCopy = t("option:flashcards.importResultWithErrors", {
          defaultValue: "Imported {{imported}} cards, skipped {{skipped}} rows ({{errorCount}} errors).",
          imported,
          skipped,
          errorCount: errors.length
        })
        message.warning(warningCopy)
        onTransferAction?.({
          area: "import",
          status: "warning",
          message: warningCopy
        })
      } else {
        const successCopy = t("option:flashcards.importResultSuccess", {
          defaultValue: "Imported {{count}} cards.",
          count: imported
        })
        message.success(successCopy)
        onTransferAction?.({
          area: "import",
          status: "success",
          message: successCopy
        })
        setContent("")
        if (importMode === "apkg") {
          setApkgFile(null)
        }
      }

      if (importedItems.length > 0) {
        showUndoNotification({
          title:
            errors.length > 0
              ? t("option:flashcards.importUndoTitlePartial", {
                  defaultValue: "Partial import completed"
                })
              : t("option:flashcards.importUndoTitle", {
                  defaultValue: "Import completed"
                }),
          description: t("option:flashcards.importUndoHint", {
            defaultValue:
              "Undo within {{seconds}}s to remove {{count}} imported cards.",
            seconds: IMPORT_UNDO_SECONDS,
            count: importedItems.length
          }),
          duration: IMPORT_UNDO_SECONDS,
          onUndo: async () => {
            let failedRollbacks = 0
            await processInChunks(importedItems, IMPORT_UNDO_CHUNK_SIZE, async (chunk) => {
              const results = await Promise.allSettled(
                chunk.map(async (item) => {
                  const latest = await getFlashcard(item.uuid)
                  await deleteFlashcard(item.uuid, latest.version)
                })
              )
              failedRollbacks += results.filter((result) => result.status === "rejected").length
            })
            await invalidateFlashcardQueries()
            if (failedRollbacks > 0) {
              throw new Error(
                t("option:flashcards.importUndoPartialFailure", {
                  defaultValue: "Some imported cards could not be rolled back."
                })
              )
            }
          }
        })
      }
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : "Import failed"
      message.error(errorMessage)
      onTransferAction?.({
        area: "import",
        status: "error",
        message: errorMessage
      })
    }
  }, [
    content,
    delimiter,
    hasHeader,
    importMode,
    importMutation,
    importJsonMutation,
    importApkgMutation,
    apkgFile,
    detectedJsonImportFormat,
    invalidateFlashcardQueries,
    message,
    onTransferAction,
    showUndoNotification,
    t
  ])

  const handleImport = React.useCallback(() => {
    if (requiresLargeImportConfirmation) {
      setConfirmLargeImportOpen(true)
      return
    }
    void performImport()
  }, [performImport, requiresLargeImportConfirmation])

  const handleConfirmLargeImport = React.useCallback(() => {
    setConfirmLargeImportOpen(false)
    void performImport()
  }, [performImport])

  const activeImportPending =
    importMode === "delimited"
      ? importMutation.isPending
      : importMode === "json"
        ? importJsonMutation.isPending
        : importMode === "structured"
          ? previewStructuredMutation.isPending || createBulkMutation.isPending
          : importApkgMutation.isPending

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-2">
        <Form.Item
          label={t("option:flashcards.importFormat", {
            defaultValue: "Import format"
          })}
          className="!mb-1"
        >
          <Select<ImportMode>
            value={importMode}
            onChange={(value) => {
              setImportMode(value)
              if (value !== "apkg") {
                setApkgFile(null)
              }
              if (value !== "structured") {
                setStructuredDrafts([])
                setStructuredPreviewErrors([])
              }
            }}
            data-testid="flashcards-import-format"
            options={[
              {
                value: "delimited",
                label: t("option:flashcards.importFormatDelimited", {
                  defaultValue: "Delimited (CSV/TSV)"
                })
              },
              {
                value: "json",
                label: t("option:flashcards.importFormatJson", {
                  defaultValue: "JSON / JSONL"
                })
              },
              {
                value: "structured",
                label: t("option:flashcards.importFormatStructured", {
                  defaultValue: "Structured Q&A"
                })
              },
              {
                value: "apkg",
                label: t("option:flashcards.importFormatApkg", {
                  defaultValue: "APKG (Anki)"
                })
              }
            ]}
          />
        </Form.Item>
        <Text type="secondary">
          {importMode === "delimited"
            ? t("option:flashcards.importHelp", {
                defaultValue: "Paste TSV/CSV lines: Deck, Front, Back, Tags, Notes"
              })
            : importMode === "json"
              ? t("option:flashcards.importHelpJson", {
                  defaultValue:
                    "Paste JSON array, {\"items\": [...]}, or JSONL (one JSON object per line)."
                })
              : importMode === "structured"
                ? t("option:flashcards.importHelpStructured", {
                    defaultValue:
                      "Paste labeled notes with Q:/A: or Question:/Answer: pairs, then preview and approve drafts before saving."
                  })
              : t("option:flashcards.importHelpApkg", {
                  defaultValue:
                    "Upload an APKG file exported from Anki. Decks, card templates, tags, and scheduling state will be imported."
                })}
        </Text>
        <pre className="mt-1 rounded bg-surface2 p-2 text-xs text-text">
          {importMode === "delimited"
            ? "Deck\tFront\tBack\tTags\tNotes\nMy deck\tWhat is a closure?\tA function with preserved outer scope.\tjavascript; fundamentals\tLecture 3"
            : importMode === "json"
              ? '[{"deck":"My deck","front":"What is a closure?","back":"A function with preserved outer scope.","tags":["javascript","fundamentals"]}]'
              : importMode === "structured"
                ? "Q: What is ATP?\nA: Primary cellular energy currency.\n\nQuestion: What is glycolysis?\nAnswer: Cytosolic glucose breakdown."
              : "my_deck.apkg"}
        </pre>
        <Collapse
          size="small"
          className="mt-2"
          activeKey={importHelpActiveKeys}
          onChange={(nextKeys) =>
            setImportHelpActiveKeys(
              Array.isArray(nextKeys) ? nextKeys.map(String) : [String(nextKeys)]
            )
          }
          items={
            importMode === "delimited"
              ? [
                  {
                    key: "columns",
                    label: t("option:flashcards.importHelpColumnsTitle", {
                      defaultValue: "Accepted columns and field rules"
                    }),
                    children: (
                      <div
                        id={IMPORT_HELP_ANCHORS.columns}
                        className="space-y-1"
                        data-testid="flashcards-import-help-columns"
                      >
                        <Text type="secondary" className="block text-xs">
                          {t("option:flashcards.importColumnsHelp", {
                            defaultValue:
                              "Accepted headers: Deck, Front, Back, Tags, Notes, Extra, Model_Type, Reverse, Is_Cloze, Deck_Description."
                          })}
                        </Text>
                        <Text type="secondary" className="block text-xs">
                          {t("option:flashcards.importTagsHelp", {
                            defaultValue:
                              "Tags can be comma- or space-delimited. Without headers, default order is Deck, Front, Back, Tags, Notes."
                          })}
                        </Text>
                        <Text
                          type="secondary"
                          className="block text-xs"
                          id={IMPORT_HELP_ANCHORS.cloze}
                        >
                          {t("option:flashcards.importClozeHelp", {
                            defaultValue:
                              "Cloze rows need Front text with at least one deletion like {{c1::answer}}."
                          })}
                        </Text>
                        <a
                          href={FLASHCARDS_HELP_LINKS.cloze}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-primary hover:underline"
                          data-testid="flashcards-import-cloze-doc-link"
                        >
                          {t("option:flashcards.importClozeDocLink", {
                            defaultValue: "Cloze syntax reference"
                          })}
                        </a>
                      </div>
                    )
                  },
                  {
                    key: "delimiter",
                    label: t("option:flashcards.importHelpDelimiterTitle", {
                      defaultValue: "Delimiter troubleshooting"
                    }),
                    children: (
                      <div
                        id={IMPORT_HELP_ANCHORS.delimiter}
                        data-testid="flashcards-import-help-delimiter"
                      >
                        <Text type="secondary" className="block text-xs">
                          {t("option:flashcards.importDelimiterHelp", {
                            defaultValue:
                              "Choose Tab for Anki exports, Comma for CSV, Semicolon for locale CSV variants, and Pipe when commas appear in card text."
                          })}
                        </Text>
                      </div>
                    )
                  }
                ]
              : importMode === "json"
                ? [
                    {
                      key: "json",
                      label: t("option:flashcards.importHelpJsonTitle", {
                        defaultValue: "JSON field mapping"
                      }),
                      children: (
                        <div
                          id={IMPORT_HELP_ANCHORS.json}
                          data-testid="flashcards-import-help-json"
                        >
                          <Text type="secondary" className="block text-xs">
                            {t("option:flashcards.importJsonFieldsHelp", {
                              defaultValue:
                                "JSON fields: deck/deck_name, front/question, back/answer, tags (array or string), notes, extra, model_type, reverse, is_cloze."
                            })}
                          </Text>
                        </div>
                      )
                    }
                  ]
              : importMode === "structured"
                ? [
                    {
                      key: "structured",
                      label: t("option:flashcards.importHelpStructuredTitle", {
                        defaultValue: "Structured preview rules"
                      }),
                      children: (
                        <div data-testid="flashcards-import-help-structured">
                          <Text type="secondary" className="block text-xs">
                            {t("option:flashcards.importStructuredFieldsHelp", {
                              defaultValue:
                                "Accepted labels: Q:/A: and Question:/Answer:. Continuation lines stay with the current question or answer until the next labeled block."
                            })}
                          </Text>
                        </div>
                      )
                    }
                  ]
              : [
                  {
                    key: "apkg",
                    label: t("option:flashcards.importHelpApkgTitle", {
                      defaultValue: "APKG import notes"
                    }),
                    children: (
                      <div
                        data-testid="flashcards-import-help-apkg"
                      >
                        <Text type="secondary" className="block text-xs">
                          {t("option:flashcards.importApkgHelp", {
                            defaultValue:
                              "APKG imports preserve deck names, tags, model types (basic/basic reverse/cloze), and scheduling metadata where available."
                          })}
                        </Text>
                      </div>
                    )
                  }
                ]
          }
          data-testid="flashcards-import-help-accordion"
        />
        <a
          href={
            importMode === "structured"
              ? FLASHCARDS_HELP_LINKS.structuredImport
              : FLASHCARDS_HELP_LINKS.importFormats
          }
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-primary hover:underline"
          data-testid="flashcards-import-doc-link"
        >
          {t("option:flashcards.importDocLink", {
            defaultValue: "Open full import/export guide"
          })}
        </a>
      </div>

      {importMode !== "apkg" ? (
        <>
          <FileDropZone
            onFileContent={setContent}
            onError={(error) => message.error(error)}
            accept={
              importMode === "delimited"
                ? ".csv,.tsv,.txt"
                : importMode === "structured"
                  ? ".txt,.md"
                : ".json,.jsonl,.ndjson,.txt"
            }
          />

          <Text type="secondary" className="text-center text-xs">
            {t("option:flashcards.orPasteBelow", {
              defaultValue: "or paste content below"
            })}
          </Text>

          <Input.TextArea
            rows={8}
            placeholder={t("option:flashcards.pasteContent", {
              defaultValue: "Paste content here..."
            })}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            data-testid="flashcards-import-textarea"
          />
          {importMode === "delimited" ? (
            <Space>
              <Select
                value={delimiter}
                onChange={setDelimiter}
                data-testid="flashcards-import-delimiter"
                options={[
                  {
                    label: t("option:flashcards.tab", { defaultValue: "Tab" }),
                    value: "\t"
                  },
                  {
                    label: t("option:flashcards.comma", { defaultValue: ", (Comma)" }),
                    value: ","
                  },
                  {
                    label: t("option:flashcards.semicolon", {
                      defaultValue: "; (Semicolon)"
                    }),
                    value: ";"
                  },
                  {
                    label: t("option:flashcards.pipe", { defaultValue: "| (Pipe)" }),
                    value: "|"
                  }
                ]}
              />
              <Space>
                <Text>
                  {t("option:flashcards.hasHeader", { defaultValue: "Has header" })}
                </Text>
                <Switch
                  checked={hasHeader}
                  onChange={setHasHeader}
                  data-testid="flashcards-import-has-header"
                />
              </Space>
            </Space>
          ) : importMode === "structured" ? (
            <div className="space-y-3">
              <Form.Item
                label={t("option:flashcards.structuredTargetDeck", {
                  defaultValue: "Target deck"
                })}
                className="!mb-1"
              >
                <Select
                  allowClear
                  value={structuredTargetDeckId ?? undefined}
                  onChange={(value) => {
                    setStructuredTargetDeckId((value as DeckSelectionValue) ?? null)
                  }}
                  options={structuredDeckOptions}
                  data-testid="flashcards-structured-target-deck"
                />
              </Form.Item>
              {structuredTargetDeckId === NEW_DECK_OPTION_VALUE ? (
                <NewDeckConfigurationFields
                  deckName={structuredNewDeckName}
                  onDeckNameChange={setStructuredNewDeckName}
                  reviewPromptSide={structuredReviewPromptSide}
                  onReviewPromptSideChange={setStructuredReviewPromptSide}
                  schedulerDraft={structuredSchedulerDraft}
                  nameTestId="flashcards-structured-new-deck-name"
                />
              ) : structuredSelectedDeck?.scheduler_settings ? (
                <Text
                  type="secondary"
                  className="block text-xs"
                  data-testid="flashcards-structured-selected-deck-summary"
                >
                  {formatSchedulerSummary(
                    structuredSelectedDeck.scheduler_type,
                    structuredSelectedDeck.scheduler_settings
                  )}
                </Text>
              ) : null}
              <Button
                type="primary"
                onClick={() => void handleStructuredPreview()}
                loading={previewStructuredMutation.isPending}
                disabled={!content.trim()}
                data-testid="flashcards-structured-preview-button"
              >
                {t("option:flashcards.structuredPreviewButton", {
                  defaultValue: "Preview structured drafts"
                })}
              </Button>
            </div>
          ) : (
            <Text
              type="secondary"
              className="text-xs"
              data-testid="flashcards-import-json-detected"
            >
              {detectedJsonImportFormat === "json"
                ? t("option:flashcards.importJsonDetectedJson", {
                    defaultValue: "Detected format: JSON array/object"
                  })
                : detectedJsonImportFormat === "jsonl"
                  ? t("option:flashcards.importJsonDetectedJsonl", {
                      defaultValue: "Detected format: JSONL"
                    })
                  : t("option:flashcards.importJsonDetectedUnknown", {
                      defaultValue:
                        "Detected format: unknown (import will still attempt JSON/JSONL parsing)"
                    })}
            </Text>
          )}
        </>
      ) : (
        <div className="space-y-2 rounded border border-dashed border-border p-4">
          <input
            type="file"
            accept=".apkg"
            onChange={(event) => {
              const selected = event.target.files?.[0] ?? null
              setApkgFile(selected)
            }}
            data-testid="flashcards-import-apkg-input"
            className="block w-full text-sm"
          />
          <Text type="secondary" className="text-xs block">
            {apkgFile
              ? t("option:flashcards.importApkgSelected", {
                  defaultValue: "Selected file: {{name}}",
                  name: apkgFile.name
                })
              : t("option:flashcards.importApkgPrompt", {
                  defaultValue: "Select an APKG file to import."
                })}
          </Text>
        </div>
      )}
      {limitsQuery.data && (
        <Text type="secondary" className="text-xs">
          {t("option:flashcards.importLimits", {
            defaultValue:
              "Limits: max {{maxCards}} cards, {{maxSize}} bytes per import",
            maxCards: limitsQuery.data.max_cards_per_import,
            maxSize: limitsQuery.data.max_content_size_bytes
          })}
        </Text>
      )}
      {importPreflightWarning && (
        <Alert
          type="warning"
          showIcon
          data-testid="flashcards-import-preflight-warning"
          title={t("option:flashcards.importPreflightTitle", {
            defaultValue: "Check import format before continuing"
          })}
          description={importPreflightWarning}
        />
      )}
      {importMode !== "structured" && (
        <>
          <Button
            type="primary"
            onClick={handleImport}
            loading={activeImportPending}
            disabled={importMode === "apkg" ? !apkgFile : !content.trim()}
            data-testid="flashcards-import-button"
          >
            {t("option:flashcards.importButton", { defaultValue: "Import" })}
          </Button>
          <Modal
            open={confirmLargeImportOpen}
            onCancel={() => setConfirmLargeImportOpen(false)}
            title={t("option:flashcards.largeImportConfirmTitle", {
              defaultValue: "Confirm large import"
            })}
            footer={[
              <Button key="cancel" onClick={() => setConfirmLargeImportOpen(false)}>
                {t("common:cancel", { defaultValue: "Cancel" })}
              </Button>,
              <Button
                key="confirm"
                type="primary"
                onClick={handleConfirmLargeImport}
                data-testid="flashcards-import-confirm-large"
              >
                {t("option:flashcards.largeImportConfirmAction", {
                  defaultValue: "Import now"
                })}
              </Button>
            ]}
          >
            <div className="space-y-1 text-sm">
              <Text>
                {t("option:flashcards.largeImportConfirmRows", {
                  defaultValue: "You are about to import approximately {{count}} items.",
                  count: estimatedImportItems
                })}
              </Text>
              <Text type="secondary" className="block">
                {t("option:flashcards.largeImportConfirmImpact", {
                  defaultValue:
                    importMode === "delimited"
                      ? "This may create many cards at once. Review delimiter/header settings before confirming."
                      : importMode === "json"
                        ? "This may create many cards at once. Review JSON structure before confirming."
                        : "This APKG may expand into many cards. Review selected file details before confirming."
                })}
              </Text>
              <Text type="secondary" className="block">
                {t("option:flashcards.largeImportUndoHint", {
                  seconds: IMPORT_UNDO_SECONDS,
                  defaultValue:
                    "Large imports may take a moment to process. You'll have {{seconds}} seconds to undo after import completes."
                })}
              </Text>
              {importMode === "delimited" ? (
                <Text type="secondary" className="block">
                  {t("option:flashcards.largeImportConfirmSummary", {
                    defaultValue:
                      "Summary: {{rows}} non-empty lines, delimiter {{delimiter}}, header {{header}}, payload {{bytes}} bytes.",
                    rows: nonEmptyLineCount,
                    delimiter: selectedDelimiterLabel,
                    header: hasHeader
                      ? t("common:yes", { defaultValue: "Yes" })
                      : t("common:no", { defaultValue: "No" }),
                    bytes: importPayloadBytes
                  })}
                </Text>
              ) : importMode === "json" ? (
                <Text type="secondary" className="block">
                  {t("option:flashcards.largeImportConfirmSummaryJson", {
                    defaultValue:
                      "Summary: detected {{format}} format, {{rows}} non-empty lines, payload {{bytes}} bytes.",
                    format: detectedJsonImportFormat === "jsonl" ? "JSONL" : "JSON",
                    rows: nonEmptyLineCount,
                    bytes: importPayloadBytes
                  })}
                </Text>
              ) : (
                <Text type="secondary" className="block">
                  {t("option:flashcards.largeImportConfirmSummaryApkg", {
                    defaultValue:
                      "Summary: file {{fileName}}, size {{bytes}} bytes, estimated {{count}} cards.",
                    fileName:
                      apkgFile?.name ||
                      t("option:flashcards.importApkgUnknownFile", {
                        defaultValue: "unknown.apkg"
                      }),
                    bytes: apkgFileSizeBytes,
                    count: estimatedImportItems
                  })}
                </Text>
              )}
            </div>
          </Modal>
        </>
      )}

      {importMode === "structured" && structuredPreviewErrors.length > 0 && (
        <Alert
          type="warning"
          showIcon
          data-testid="flashcards-structured-preview-errors"
          title={t("option:flashcards.structuredPreviewErrorsTitle", {
            defaultValue: "Preview warnings"
          })}
          description={
            <div className="space-y-1 text-xs">
              {structuredPreviewErrors.map((error, index) => (
                <div key={`${error.line ?? "line"}-${index}`}>
                  <Text code>
                    {typeof error.line === "number"
                      ? t("option:flashcards.importErrorLine", {
                          defaultValue: "Line {{line}}",
                          line: error.line
                        })
                      : t("option:flashcards.importErrorRowUnknown", {
                          defaultValue: "Unknown row"
                        })}
                  </Text>
                  <Text className="ml-2">{error.error}</Text>
                </div>
              ))}
            </div>
          }
        />
      )}

      {importMode === "structured" && structuredDrafts.length > 0 && (
        <div className="space-y-2">
          <Text strong>
            {t("option:flashcards.structuredPreviewTitle", {
              defaultValue: "Structured drafts (editable before save)"
            })}
          </Text>
          {structuredDrafts.map((draft, index) => (
            <Card
              key={draft.id}
              size="small"
              title={t("option:flashcards.generatedCardTitle", {
                defaultValue: "Card {{index}}",
                index: index + 1
              })}
              extra={
                <Button
                  type="text"
                  danger
                  size="small"
                  onClick={() => removeStructuredDraft(draft.id)}
                >
                  {t("common:remove", { defaultValue: "Remove" })}
                </Button>
              }
            >
              <Space orientation="vertical" className="w-full">
                <label className="inline-flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={draft.selected}
                    onChange={(event) =>
                      updateStructuredDraft(draft.id, {
                        selected: event.target.checked
                      })
                    }
                    data-testid={`flashcards-structured-draft-selected-${index}`}
                  />
                  <span>
                    {t("option:flashcards.structuredDraftMeta", {
                      defaultValue: "Use lines {{start}}-{{end}}",
                      start: draft.line_start,
                      end: draft.line_end
                    })}
                  </span>
                </label>
                <Input.TextArea
                  value={draft.front}
                  rows={2}
                  onChange={(event) =>
                    updateStructuredDraft(draft.id, { front: event.target.value })
                  }
                />
                <Input.TextArea
                  value={draft.back}
                  rows={3}
                  onChange={(event) =>
                    updateStructuredDraft(draft.id, { back: event.target.value })
                  }
                />
              </Space>
            </Card>
          ))}
          <Button
            type="primary"
            onClick={() => void handleSaveStructuredDrafts()}
            loading={createBulkMutation.isPending || createDeckMutation.isPending}
            disabled={!structuredDrafts.some((draft) => draft.selected)}
            data-testid="flashcards-structured-save-button"
          >
            {t("option:flashcards.structuredSaveButton", {
              defaultValue: "Save selected drafts"
            })}
          </Button>
        </div>
      )}

      {lastResult && (
        <Alert
          data-testid="flashcards-import-last-result"
          showIcon
          type={lastResult.errors.length > 0 ? "warning" : "success"}
          title={
            lastResult.errors.length > 0
              ? t("option:flashcards.lastImportPartial", {
                  defaultValue: "Last import: {{imported}} imported, {{skipped}} skipped",
                  imported: lastResult.imported,
                  skipped: lastResult.skipped
                })
              : t("option:flashcards.lastImportSuccess", {
                  defaultValue: "Last import: {{imported}} cards imported",
                  imported: lastResult.imported
                })
          }
          description={
            lastResult.errors.length > 0 && (
              <div className="mt-1 space-y-1 text-xs">
                {lastResult.errors.slice(0, 6).map((err, idx) => {
                  const location =
                    typeof err.line === "number"
                      ? t("option:flashcards.importErrorLine", {
                          defaultValue: "Line {{line}}",
                          line: err.line
                        })
                      : typeof err.index === "number"
                        ? t("option:flashcards.importErrorItem", {
                            defaultValue: "Item {{index}}",
                            index: err.index
                          })
                        : t("option:flashcards.importErrorRowUnknown", {
                            defaultValue: "Unknown row"
                          })
                  const guidance = getImportErrorGuidance(err.error, t)
                  return (
                    <div key={`${location}-${idx}`} className="space-y-1">
                      <div>
                        <Text code>{location}</Text>
                        <Text className="ml-2">{err.error}</Text>
                      </div>
                      {guidance && (
                        <div className="flex flex-wrap items-center gap-2 pl-1">
                          <Text type="secondary" className="block">
                            {guidance.copy}
                          </Text>
                          {guidance.helpAnchorId && (
                            <Button
                              type="link"
                              size="small"
                              className="!h-auto !p-0 text-xs"
                              onClick={() => scrollToImportHelp(guidance.helpAnchorId)}
                              data-testid={`flashcards-import-error-help-${idx}`}
                            >
                              {t("option:flashcards.importErrorHelpLink", {
                                defaultValue: "View format help"
                              })}
                            </Button>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
                {lastResult.errors.length > 6 && (
                  <Text type="secondary">
                    {t("option:flashcards.importErrorsMore", {
                      defaultValue: "+{{count}} more errors",
                      count: lastResult.errors.length - 6
                    })}
                  </Text>
                )}
              </div>
            )
          }
        />
      )}
    </div>
  )
}

/**
 * Export panel for CSV/JSON/APKG export.
 */
const ExportPanel: React.FC<TransferActionReporterProps> = ({ onTransferAction }) => {
  const { t } = useTranslation(["option", "common"])
  const message = useAntdMessage()
  const decksQuery = useDecksQuery()
  const [exportDeckId, setExportDeckId] = React.useState<number | null>(null)
  const [exportFormat, setExportFormat] = React.useState<"csv" | "apkg" | "json">("csv")
  const [exportTag, setExportTag] = React.useState("")
  const [exportQueryText, setExportQueryText] = React.useState("")
  const [exportIncludeReverse, setExportIncludeReverse] = React.useState(false)
  const [exportDelimiter, setExportDelimiter] = React.useState<string>("\t")
  const [exportIncludeHeader, setExportIncludeHeader] = React.useState(false)
  const [exportExtendedHeader, setExportExtendedHeader] = React.useState(false)
  const [isExporting, setIsExporting] = React.useState(false)

  const normalizedExportTag = exportTag.trim()
  const normalizedExportQuery = exportQueryText.trim()

  const exportPreviewCountQuery = useQuery({
    queryKey: [
      "flashcards:export-preview-count",
      exportDeckId ?? null,
      normalizedExportTag.toLowerCase(),
      normalizedExportQuery
    ],
    queryFn: async () => {
      const response = await listFlashcards({
        deck_id: exportDeckId ?? undefined,
        tag: normalizedExportTag || undefined,
        q: normalizedExportQuery || undefined,
        due_status: "all",
        limit: 1,
        offset: 0
      })
      return response.total ?? response.count ?? 0
    }
  })

  const selectedDeckLabel = React.useMemo(() => {
    if (exportDeckId == null) {
      return t("option:flashcards.allDecks", {
        defaultValue: "All decks"
      })
    }
    return (
      (decksQuery.data || []).find((deck) => deck.id === exportDeckId)?.name ||
      `${t("option:flashcards.deck", { defaultValue: "Deck" })} ${exportDeckId}`
    )
  }, [decksQuery.data, exportDeckId, t])

  const handleExport = async () => {
    setIsExporting(true)
    try {
      const exportParams = {
        deck_id: exportDeckId ?? undefined,
        tag: normalizedExportTag || undefined,
        q: normalizedExportQuery || undefined,
        include_reverse: exportIncludeReverse || undefined
      }

      let blob: Blob
      if (exportFormat === "apkg") {
        blob = await exportFlashcardsFile({
          ...exportParams,
          format: "apkg"
        })
      } else if (exportFormat === "json") {
        const text = await exportFlashcards({
          ...exportParams,
          format: "json"
        })
        blob = new Blob([text], {
          type: "application/json;charset=utf-8"
        })
      } else {
        const text = await exportFlashcards({
          ...exportParams,
          format: "csv",
          delimiter: exportDelimiter,
          include_header: exportIncludeHeader,
          extended_header: exportExtendedHeader
        })
        blob = new Blob([text], {
          type:
            exportDelimiter === "\t"
              ? "text/tab-separated-values;charset=utf-8"
              : "text/csv;charset=utf-8"
        })
      }
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      if (exportFormat === "apkg") {
        a.download = "flashcards.apkg"
      } else if (exportFormat === "json") {
        a.download = "flashcards.json"
      } else {
        a.download = exportDelimiter === "\t" ? "flashcards.tsv" : "flashcards.csv"
      }
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
      const successCopy = t("option:flashcards.exportSuccess", {
        defaultValue: "Export ready: {{fileName}}",
        fileName:
          exportFormat === "apkg"
            ? "flashcards.apkg"
            : exportFormat === "json"
              ? "flashcards.json"
              : exportDelimiter === "\t"
                ? "flashcards.tsv"
                : "flashcards.csv"
      })
      message.success(successCopy)
      onTransferAction?.({
        area: "export",
        status: "success",
        message: successCopy
      })
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : "Export failed"
      message.error(errorMessage)
      onTransferAction?.({
        area: "export",
        status: "error",
        message: errorMessage
      })
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div>
        <Text type="secondary">
          {t("option:flashcards.exportHelp", {
            defaultValue:
              "Export filtered flashcards to delimited text (CSV/TSV), JSON, or Anki-compatible APKG format."
          })}
        </Text>
      </div>
      <Form.Item
        label={t("option:flashcards.deck", { defaultValue: "Deck" })}
        className="!mb-2"
      >
        <Select
          placeholder={t("option:flashcards.allDecks", {
            defaultValue: "All decks"
          })}
          allowClear
          loading={decksQuery.isLoading}
          value={exportDeckId ?? undefined}
          onChange={setExportDeckId}
          data-testid="flashcards-export-deck"
          options={(decksQuery.data || []).map((d) => ({
            label: d.name,
            value: d.id
          }))}
        />
      </Form.Item>
      <Form.Item
        label={t("option:flashcards.exportFormat", { defaultValue: "Format" })}
        className="!mb-2"
      >
        <Select
          value={exportFormat}
          onChange={setExportFormat}
          data-testid="flashcards-export-format"
          options={[
            {
              label: t("option:flashcards.exportFormatDelimited", {
                defaultValue: "Delimited (CSV/TSV)"
              }),
              value: "csv"
            },
            {
              label: t("option:flashcards.exportFormatJson", {
                defaultValue: "JSON"
              }),
              value: "json"
            },
            { label: "APKG (Anki)", value: "apkg" }
          ]}
        />
      </Form.Item>
      <Form.Item
        label={t("option:flashcards.exportTagFilter", { defaultValue: "Tag filter" })}
        className="!mb-2"
      >
        <Input
          value={exportTag}
          onChange={(event) => setExportTag(event.target.value)}
          placeholder={t("option:flashcards.exportTagFilterPlaceholder", {
            defaultValue: "Optional single tag"
          })}
          data-testid="flashcards-export-tag"
        />
      </Form.Item>
      <Form.Item
        label={t("option:flashcards.exportQueryFilter", { defaultValue: "Text filter" })}
        className="!mb-2"
      >
        <Input
          value={exportQueryText}
          onChange={(event) => setExportQueryText(event.target.value)}
          placeholder={t("option:flashcards.exportQueryFilterPlaceholder", {
            defaultValue: "Optional search query"
          })}
          data-testid="flashcards-export-query"
        />
      </Form.Item>
      <div className="flex items-center justify-between gap-3">
        <Text>{t("option:flashcards.exportIncludeReverse", { defaultValue: "Include reverse cards" })}</Text>
        <Switch
          checked={exportIncludeReverse}
          onChange={setExportIncludeReverse}
          data-testid="flashcards-export-include-reverse"
        />
      </div>
      {exportFormat === "csv" && (
        <>
          <Form.Item
            label={t("option:flashcards.exportDelimiter", { defaultValue: "Delimiter" })}
            className="!mb-2"
          >
            <Select
              value={exportDelimiter}
              onChange={setExportDelimiter}
              data-testid="flashcards-export-delimiter"
              options={[
                {
                  label: t("option:flashcards.tab", { defaultValue: "Tab" }),
                  value: "\t"
                },
                {
                  label: t("option:flashcards.comma", { defaultValue: ", (Comma)" }),
                  value: ","
                },
                {
                  label: t("option:flashcards.semicolon", {
                    defaultValue: "; (Semicolon)"
                  }),
                  value: ";"
                },
                {
                  label: t("option:flashcards.pipe", { defaultValue: "| (Pipe)" }),
                  value: "|"
                }
              ]}
            />
          </Form.Item>
          <div className="flex items-center justify-between gap-3">
            <Text>{t("option:flashcards.exportIncludeHeader", { defaultValue: "Include header row" })}</Text>
            <Switch
              checked={exportIncludeHeader}
              onChange={setExportIncludeHeader}
              data-testid="flashcards-export-include-header"
            />
          </div>
          <div className="flex items-center justify-between gap-3">
            <Text>{t("option:flashcards.exportExtendedHeader", { defaultValue: "Use extended header columns" })}</Text>
            <Switch
              checked={exportExtendedHeader}
              onChange={setExportExtendedHeader}
              data-testid="flashcards-export-extended-header"
            />
          </div>
        </>
      )}
      <Alert
        type="info"
        showIcon
        data-testid="flashcards-export-preview"
        title={t("option:flashcards.exportPreviewTitle", {
          defaultValue: "Export preview"
        })}
        description={t("option:flashcards.exportPreviewDescription", {
          defaultValue:
            "{{count}} cards from {{deck}}. Tag filter: {{tag}}. Query filter: {{query}}.",
          count: exportPreviewCountQuery.data ?? 0,
          deck: selectedDeckLabel,
          tag:
            normalizedExportTag ||
            t("option:flashcards.noneLabel", { defaultValue: "none" }),
          query:
            normalizedExportQuery ||
            t("option:flashcards.noneLabel", { defaultValue: "none" })
        })}
      />
      <Button
        type="primary"
        onClick={handleExport}
        loading={isExporting}
        disabled={exportPreviewCountQuery.isLoading}
        data-testid="flashcards-export-button"
      >
        {t("option:flashcards.exportButton", { defaultValue: "Export" })}
      </Button>
    </div>
  )
}

/**
 * Generate panel for LLM-assisted card generation from free text.
 */
const GeneratePanel: React.FC<GeneratePanelProps & TransferActionReporterProps> = ({
  initialIntent,
  onTransferAction
}) => {
  const { t } = useTranslation(["option", "common"])
  const message = useAntdMessage()
  const qc = useQueryClient()
  const decksQuery = useDecksQuery()
  const generateMutation = useGenerateFlashcardsMutation()
  const createMutation = useCreateFlashcardMutation()
  const createDeckMutation = useCreateDeckMutation()
  const decks = decksQuery.data || []

  const llmProvidersQuery = useQuery({
    queryKey: ["flashcards", "llm-providers"],
    queryFn: () => getLlmProviders(),
    staleTime: 60_000,
    retry: 1
  })

  const hasLlmProviders = React.useMemo(() => {
    if (llmProvidersQuery.isLoading || llmProvidersQuery.isError) return true // optimistic while loading or on error
    if (llmProvidersQuery.data == null) return true // no data yet — assume available
    // Unwrap ApiSendResponse envelope: actual payload is in .data
    const raw = llmProvidersQuery.data as any
    const data = raw?.data ?? raw
    if (Array.isArray(data?.providers)) return data.providers.length > 0
    if (typeof data?.total_configured === "number") return data.total_configured > 0
    return true // fallback: assume available if shape unknown
  }, [llmProvidersQuery.data, llmProvidersQuery.isLoading, llmProvidersQuery.isError])

  const sourceContext = React.useMemo<GenerateSourceContext | null>(() => {
    if (!initialIntent) return null
    if (
      initialIntent.sourceType !== "media" &&
      initialIntent.sourceType !== "note" &&
      initialIntent.sourceType !== "message"
    ) {
      return null
    }
    return {
      sourceType: initialIntent.sourceType,
      sourceId: initialIntent.sourceId?.trim() || null,
      sourceTitle: initialIntent.sourceTitle?.trim() || null
    }
  }, [initialIntent])

  const [sourceText, setSourceText] = React.useState(() => initialIntent?.text || "")
  const [numCards, setNumCards] = React.useState(10)
  const [cardType, setCardType] = React.useState<"basic" | "basic_reverse" | "cloze">("basic")
  const [difficulty, setDifficulty] = React.useState<"easy" | "medium" | "hard" | "mixed">("mixed")
  const [provider, setProvider] = React.useState("")
  const [model, setModel] = React.useState("")
  const [focusTopicsInput, setFocusTopicsInput] = React.useState("")
  const [targetDeckId, setTargetDeckId] = React.useState<DeckSelectionValue>(undefined)
  const [newDeckName, setNewDeckName] = React.useState(() =>
    t("option:flashcards.generatedDeckName", {
      defaultValue: "Generated Flashcards"
    })
  )
  const [reviewPromptSide, setReviewPromptSide] = React.useState<DeckReviewPromptSide>("front")
  const [generatedCards, setGeneratedCards] = React.useState<GeneratedCardDraft[]>([])
  const [generationError, setGenerationError] = React.useState<string | null>(null)
  const [isSaving, setIsSaving] = React.useState(false)
  const generatedDeckSchedulerDraft = useDeckSchedulerDraft()
  const selectedDeck = React.useMemo(
    () => (typeof targetDeckId === "number" ? decks.find((deck) => deck.id === targetDeckId) ?? null : null),
    [decks, targetDeckId]
  )
  const deckOptions = React.useMemo(
    () => [
      ...decks.map((deck) => ({
        label: deck.name,
        value: deck.id
      })),
      {
        label: t("option:flashcards.createNewDeck", {
          defaultValue: "Create new deck"
        }),
        value: NEW_DECK_OPTION_VALUE
      }
    ],
    [decks, t]
  )

  React.useEffect(() => {
    if (targetDeckId != null) return
    if (decks.length > 0) {
      setTargetDeckId(decks[0].id)
      return
    }
    setTargetDeckId(NEW_DECK_OPTION_VALUE)
  }, [decks, targetDeckId])

  const updateGeneratedCard = React.useCallback(
    (id: string, patch: Partial<GeneratedCardDraft>) => {
      setGeneratedCards((prev) =>
        prev.map((card) => (card.id === id ? { ...card, ...patch } : card))
      )
    },
    []
  )

  const removeGeneratedCard = React.useCallback((id: string) => {
    setGeneratedCards((prev) => prev.filter((card) => card.id !== id))
  }, [])

  const handleGenerate = React.useCallback(async () => {
    try {
      setGenerationError(null)
      const result = await generateMutation.mutateAsync({
        text: sourceText,
        numCards,
        cardType,
        difficulty,
        focusTopics: focusTopicsInput
          .split(",")
          .map((topic) => topic.trim())
          .filter((topic) => topic.length > 0),
        provider: provider.trim() || undefined,
        model: model.trim() || undefined
      })
      const drafts = normalizeGeneratedCards(result.flashcards)
      setGeneratedCards(drafts)
      if (drafts.length === 0) {
        const warningCopy = t("option:flashcards.generateEmptyResult", {
          defaultValue:
            "No cards were generated. Try reducing card count, simplifying text, or adjusting provider/model."
        })
        setGenerationError(warningCopy)
        onTransferAction?.({
          area: "generate",
          status: "warning",
          message: warningCopy
        })
        return
      }
      const successCopy = t("option:flashcards.generateSuccess", {
        defaultValue: "Generated {{count}} cards.",
        count: drafts.length
      })
      message.success(successCopy)
      onTransferAction?.({
        area: "generate",
        status: "success",
        message: successCopy
      })
    } catch (e: unknown) {
      const baseMessage = e instanceof Error ? e.message : "Generation failed"
      const errorCopy = t("option:flashcards.generateErrorWithHelp", {
        defaultValue:
          "{{message}}. Check provider/model settings, then retry with shorter text or fewer cards.",
        message: baseMessage
      })
      setGenerationError(errorCopy)
      onTransferAction?.({
        area: "generate",
        status: "error",
        message: errorCopy
      })
    }
  }, [
    cardType,
    difficulty,
    focusTopicsInput,
    generateMutation,
    message,
    model,
    numCards,
    onTransferAction,
    provider,
    sourceText,
    t
  ])

  const resolveTargetDeckId = React.useCallback(async (): Promise<number> => {
    if (typeof targetDeckId === "number") return targetDeckId
    if (targetDeckId === undefined && decks.length > 0) return decks[0].id
    if (targetDeckId === NEW_DECK_OPTION_VALUE || (targetDeckId == null && decks.length === 0)) {
      const name = newDeckName.trim()
      if (!name) {
        throw new Error(
          t("option:flashcards.newDeckNameRequired", {
            defaultValue: "Enter a deck name."
          })
        )
      }
      const schedulerSettings = generatedDeckSchedulerDraft.getValidatedSettings()
      if (!schedulerSettings) {
        throw new Error(
          t("option:flashcards.schedulerDraftInvalid", {
            defaultValue: "Draft has validation errors."
          })
        )
      }
      const createdDeck = await createDeckMutation.mutateAsync({
        name,
        review_prompt_side: reviewPromptSide,
        scheduler_type: schedulerSettings.scheduler_type,
        scheduler_settings: schedulerSettings.scheduler_settings
      })
      setTargetDeckId(createdDeck.id)
      return createdDeck.id
    }
    if (targetDeckId == null && decks.length > 0) return decks[0].id
    throw new Error(
      t("option:flashcards.newDeckNameRequired", {
        defaultValue: "Enter a deck name."
      })
    )
  }, [createDeckMutation, decks, generatedDeckSchedulerDraft, newDeckName, reviewPromptSide, t, targetDeckId])

  const handleSaveGeneratedCards = React.useCallback(async () => {
    if (generatedCards.length === 0) return
    setIsSaving(true)
    try {
      const deckId = await resolveTargetDeckId()
      let created = 0
      let failed = 0
      const successfulDraftIds = new Set<string>()
      for (const card of generatedCards) {
        try {
          await createMutation.mutateAsync({
            deck_id: deckId,
            front: card.front,
            back: card.back,
            tags: card.tags,
            notes: card.notes || undefined,
            extra: card.extra || undefined,
            model_type: card.model_type,
            reverse: card.model_type === "basic_reverse",
            is_cloze: card.model_type === "cloze",
            source_ref_type: sourceContext?.sourceType,
            source_ref_id: sourceContext?.sourceId || undefined
          })
          created += 1
          successfulDraftIds.add(card.id)
        } catch {
          failed += 1
        }
      }

      await qc.invalidateQueries({
        predicate: (query) =>
          Array.isArray(query.queryKey) &&
          typeof query.queryKey[0] === "string" &&
          query.queryKey[0].startsWith("flashcards:")
      })

      if (created > 0 && failed === 0) {
        const successCopy = t("option:flashcards.generateSaveSuccess", {
          defaultValue: "Saved {{count}} generated cards.",
          count: created
        })
        message.success(successCopy)
        onTransferAction?.({
          area: "generate",
          status: "success",
          message: successCopy
        })
        setGeneratedCards([])
        return
      }

      if (created > 0 && failed > 0) {
        const warningCopy = t("option:flashcards.generateSavePartial", {
          defaultValue: "Saved {{created}} cards; {{failed}} failed.",
          created,
          failed
        })
        message.warning(warningCopy)
        onTransferAction?.({
          area: "generate",
          status: "warning",
          message: warningCopy
        })
        setGeneratedCards((prev) =>
          prev.filter((card) => !successfulDraftIds.has(card.id))
        )
        return
      }

      const errorCopy = t("option:flashcards.generateSaveFailed", {
        defaultValue: "Failed to save generated cards."
      })
      message.error(errorCopy)
      onTransferAction?.({
        area: "generate",
        status: "error",
        message: errorCopy
      })
    } finally {
      setIsSaving(false)
    }
  }, [
    createMutation,
    generatedCards,
    message,
    onTransferAction,
    qc,
    resolveTargetDeckId,
    sourceContext,
    t
  ])

  return (
    <div className="flex flex-col gap-3">
      <Text type="secondary">
        {t("option:flashcards.generateHelp", {
          defaultValue:
            "Generate cards from pasted text, review/edit them, then save to a deck."
        })}
      </Text>
      {sourceContext && (
        <Alert
          type="info"
          showIcon
          data-testid="flashcards-generate-source-context"
          title={t("option:flashcards.generateSourceContextTitle", {
            defaultValue: "Source context attached"
          })}
          description={t("option:flashcards.generateSourceContextDescription", {
            defaultValue:
              "Cards saved from this draft will be linked to {{sourceType}} {{sourceId}}.",
            sourceType: sourceContext.sourceType,
            sourceId:
              sourceContext.sourceId ||
              sourceContext.sourceTitle ||
              t("option:flashcards.unknownSource", {
                defaultValue: "unknown source"
              })
          })}
        />
      )}
      <Input.TextArea
        rows={6}
        value={sourceText}
        onChange={(event) => setSourceText(event.target.value)}
        placeholder={t("option:flashcards.generateTextPlaceholder", {
          defaultValue: "Paste transcript, notes, or study material..."
        })}
        data-testid="flashcards-generate-text"
      />
      <Typography.Text type="secondary" className="text-xs -mt-1">
        {t("option:flashcards.generateQualityTip", {
          defaultValue:
            "Tip: Longer, more detailed source text produces higher quality flashcards. Aim for at least a paragraph."
        })}
      </Typography.Text>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        <Form.Item
          label={t("option:flashcards.generateNumCards", {
            defaultValue: "Number of cards"
          })}
          className="!mb-2"
        >
          <Input
            value={String(numCards)}
            type="number"
            min={1}
            max={100}
            onChange={(event) => {
              const next = Number(event.target.value)
              if (!Number.isFinite(next)) return
              setNumCards(Math.max(1, Math.min(100, Math.round(next))))
            }}
            data-testid="flashcards-generate-count"
          />
        </Form.Item>
        <Form.Item
          label={t("option:flashcards.generateCardType", {
            defaultValue: "Card type"
          })}
          className="!mb-2"
        >
          <Select
            value={cardType}
            onChange={(value) => setCardType(value)}
            data-testid="flashcards-generate-card-type"
            options={[
              { value: "basic", label: "Basic" },
              { value: "basic_reverse", label: "Basic (reverse)" },
              { value: "cloze", label: "Cloze" }
            ]}
          />
        </Form.Item>
        <Form.Item
          label={t("option:flashcards.generateDifficulty", {
            defaultValue: "Difficulty"
          })}
          className="!mb-2"
        >
          <Select
            value={difficulty}
            onChange={(value) => setDifficulty(value)}
            data-testid="flashcards-generate-difficulty"
            options={[
              { value: "easy", label: "Easy" },
              { value: "medium", label: "Medium" },
              { value: "hard", label: "Hard" },
              { value: "mixed", label: "Mixed" }
            ]}
          />
        </Form.Item>
        <Form.Item
          label={t("option:flashcards.deck", { defaultValue: "Deck" })}
          className="!mb-2"
        >
          <Select
            allowClear
            value={targetDeckId ?? undefined}
            onChange={(value) => setTargetDeckId((value as DeckSelectionValue) ?? null)}
            data-testid="flashcards-generate-deck"
            options={deckOptions}
          />
        </Form.Item>
        {targetDeckId === NEW_DECK_OPTION_VALUE ? (
          <NewDeckConfigurationFields
            deckName={newDeckName}
            onDeckNameChange={setNewDeckName}
            reviewPromptSide={reviewPromptSide}
            onReviewPromptSideChange={setReviewPromptSide}
            schedulerDraft={generatedDeckSchedulerDraft}
            nameTestId="flashcards-generate-new-deck-name"
          />
        ) : selectedDeck?.scheduler_settings ? (
          <Text
            type="secondary"
            className="block text-xs -mt-2 mb-2"
            data-testid="flashcards-generate-selected-deck-summary"
          >
            {formatSchedulerSummary(selectedDeck.scheduler_type, selectedDeck.scheduler_settings)}
          </Text>
        ) : null}
        {!hasLlmProviders && (
          <Alert
            type="info"
            showIcon
            className="mb-3 md:col-span-2"
            data-testid="flashcards-generate-no-llm-banner"
            message={t("option:flashcards.generateNoLlmBanner", {
              defaultValue:
                "Flashcard generation requires an LLM provider. Configure one in Settings \u2192 LLM Providers."
            })}
          />
        )}
        <Form.Item
          label={t("option:flashcards.generateProvider", {
            defaultValue: "Provider (optional)"
          })}
          className="!mb-2"
        >
          <Input
            value={provider}
            onChange={(event) => setProvider(event.target.value)}
            data-testid="flashcards-generate-provider"
          />
        </Form.Item>
        <Form.Item
          label={t("option:flashcards.generateModel", {
            defaultValue: "Model (optional)"
          })}
          className="!mb-2"
        >
          <Input
            value={model}
            onChange={(event) => setModel(event.target.value)}
            data-testid="flashcards-generate-model"
          />
        </Form.Item>
      </div>
      <Form.Item
        label={t("option:flashcards.generateFocusTopics", {
          defaultValue: "Focus topics (comma-separated)"
        })}
        className="!mb-2"
      >
        <Input
          value={focusTopicsInput}
          onChange={(event) => setFocusTopicsInput(event.target.value)}
          placeholder={t("option:flashcards.generateFocusTopicsPlaceholder", {
            defaultValue: "exam 1, chapter 3, formulas"
          })}
          data-testid="flashcards-generate-focus-topics"
        />
      </Form.Item>
      {generationError && (
        <Alert
          type="error"
          showIcon
          title={generationError}
          description={
            <span className="text-xs">
              {t("option:flashcards.generateProviderKeyHint", {
                defaultValue: "If this is a provider or API key issue, "
              })}
              <Link to="/settings/provider-keys" className="text-primary hover:text-primaryStrong underline">
                {t("option:flashcards.generateProviderKeyLink", {
                  defaultValue: "configure provider keys in Settings"
                })}
              </Link>
              .
            </span>
          }
        />
      )}
      <Tooltip
        title={
          !hasLlmProviders
            ? t("option:flashcards.generateNoLlmTooltip", {
                defaultValue: "No LLM provider configured"
              })
            : undefined
        }
      >
        <span>
          <Button
            type="primary"
            onClick={handleGenerate}
            loading={generateMutation.isPending}
            disabled={!sourceText.trim() || !hasLlmProviders}
            data-testid="flashcards-generate-button"
          >
            {t("option:flashcards.generateButton", { defaultValue: "Generate cards" })}
          </Button>
        </span>
      </Tooltip>

      {generatedCards.length > 0 && (
        <div className="space-y-2">
          <Text strong>
            {t("option:flashcards.generatePreviewTitle", {
              defaultValue: "Generated cards (editable before save)"
            })}
          </Text>
          {generatedCards.map((card, index) => (
            <Card
              key={card.id}
              size="small"
              title={t("option:flashcards.generatedCardTitle", {
                defaultValue: "Card {{index}}",
                index: index + 1
              })}
              extra={
                <Button
                  type="text"
                  danger
                  size="small"
                  onClick={() => removeGeneratedCard(card.id)}
                >
                  {t("common:remove", { defaultValue: "Remove" })}
                </Button>
              }
            >
              <Space orientation="vertical" className="w-full">
                <Input.TextArea
                  value={card.front}
                  rows={2}
                  onChange={(event) =>
                    updateGeneratedCard(card.id, { front: event.target.value })
                  }
                />
                <Input.TextArea
                  value={card.back}
                  rows={3}
                  onChange={(event) =>
                    updateGeneratedCard(card.id, { back: event.target.value })
                  }
                />
                <Input
                  value={card.tags.join(", ")}
                  onChange={(event) =>
                    updateGeneratedCard(card.id, {
                      tags: event.target.value
                        .split(",")
                        .map((tag) => tag.trim())
                        .filter((tag) => tag.length > 0)
                    })
                  }
                  placeholder={t("option:flashcards.tagsPlaceholder", {
                    defaultValue: "tag-1, tag-2"
                  })}
                />
              </Space>
            </Card>
          ))}
          <Button
            type="primary"
            onClick={handleSaveGeneratedCards}
            loading={isSaving || createMutation.isPending || createDeckMutation.isPending}
            data-testid="flashcards-generate-save-button"
          >
            {t("option:flashcards.generateSaveButton", {
              defaultValue: "Save generated cards"
            })}
          </Button>
        </div>
      )}
    </div>
  )
}

/**
 * Import/Export tab for flashcards.
 */
type ImportExportTabProps = {
  generateIntent?: FlashcardsGenerateIntent | null
  studyPackIntent?: StudyPackIntent | null
}

export const ImportExportTab: React.FC<ImportExportTabProps> = ({
  generateIntent,
  studyPackIntent
}) => {
  const { t } = useTranslation(["option", "common"])
  const limitsQuery = useImportLimitsQuery()
  const [lastTransferAction, setLastTransferAction] =
    React.useState<TransferActionSummary | null>(null)
  const [studyPackDrawerOpen, setStudyPackDrawerOpen] = React.useState(false)

  React.useEffect(() => {
    if (studyPackIntent) {
      setStudyPackDrawerOpen(true)
    }
  }, [studyPackIntent])

  const handleTransferAction = React.useCallback((summary: TransferActionSummaryInput) => {
    setLastTransferAction({
      ...summary,
      at: new Date().toISOString()
    })
  }, [])

  const lastTransferActionText = React.useMemo(() => {
    if (!lastTransferAction) {
      return t("option:flashcards.transferSummaryNoAction", {
        defaultValue: "No transfer actions yet in this session."
      })
    }
    const areaLabel =
      lastTransferAction.area === "import"
        ? t("option:flashcards.importTitle", { defaultValue: "Import Flashcards" })
        : lastTransferAction.area === "export"
          ? t("option:flashcards.exportTitle", { defaultValue: "Export Flashcards" })
          : lastTransferAction.area === "occlusion"
            ? t("option:flashcards.occlusionTitle", {
                defaultValue: "Image Occlusion"
              })
            : t("option:flashcards.generateTitle", {
                defaultValue: "Generate Flashcards"
              })
    return t("option:flashcards.transferSummaryLastAction", {
      defaultValue: "{{area}} · {{message}} · {{time}}",
      area: areaLabel,
      message: lastTransferAction.message,
      time: new Date(lastTransferAction.at).toLocaleTimeString()
    })
  }, [lastTransferAction, t])

  return (
    <div className="grid gap-4 grid-cols-1 xl:grid-cols-4">
      <Card
        className="xl:col-span-4"
        title={t("option:flashcards.studyPackLauncherTitle", {
          defaultValue: "Study packs"
        })}
        data-testid="flashcards-study-pack-launcher"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="max-w-3xl space-y-1">
            <Text strong className="block">
              {t("option:flashcards.studyPackLauncherHeadline", {
                defaultValue: "Turn media or notes into a review queue."
              })}
            </Text>
            <Text type="secondary">
              {t("option:flashcards.studyPackLauncherBody", {
                defaultValue:
                  "Create a study pack from supported sources, then review the generated deck in Flashcards."
              })}
            </Text>
          </div>
          <Button
            type="primary"
            onClick={() => setStudyPackDrawerOpen(true)}
            data-testid="study-pack-launcher-button"
          >
            {t("option:flashcards.studyPackLaunchButton", {
              defaultValue: "Create study pack"
            })}
          </Button>
        </div>
      </Card>
      <Card
        className="xl:col-span-4"
        title={t("option:flashcards.transferSummaryTitle", {
          defaultValue: "Transfer summary"
        })}
        data-testid="flashcards-transfer-summary"
      >
        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <div data-testid="flashcards-transfer-summary-formats">
            <Text strong className="block">
              {t("option:flashcards.transferSummaryFormatsLabel", {
                defaultValue: "Supported formats"
              })}
            </Text>
            <Text type="secondary">
              {t("option:flashcards.transferSummaryFormatsValue", {
                defaultValue:
                  "Import: CSV, TSV, JSON, JSONL, Structured Q&A, APKG · Author: Generate, Image Occlusion · Export: TSV, CSV, JSON, APKG"
              })}
            </Text>
          </div>
          <div data-testid="flashcards-transfer-summary-limits">
            <Text strong className="block">
              {t("option:flashcards.transferSummaryLimitsLabel", {
                defaultValue: "Current import limits"
              })}
            </Text>
            <Text type="secondary">
              {limitsQuery.data
                ? t("option:flashcards.transferSummaryLimitsValue", {
                    defaultValue: "{{cards}} cards · {{bytes}} bytes",
                    cards: limitsQuery.data.max_cards_per_import,
                    bytes: limitsQuery.data.max_content_size_bytes
                  })
                : t("option:flashcards.transferSummaryLimitsUnknown", {
                    defaultValue: "Limits unavailable"
                  })}
            </Text>
          </div>
          <div data-testid="flashcards-transfer-summary-last-action">
            <Text strong className="block">
              {t("option:flashcards.transferSummaryLastActionLabel", {
                defaultValue: "Last action"
              })}
            </Text>
            <Text
              type={
                lastTransferAction?.status === "error"
                  ? "danger"
                  : lastTransferAction?.status === "warning"
                    ? "warning"
                    : "secondary"
              }
            >
              {lastTransferActionText}
            </Text>
          </div>
        </div>
      </Card>
      <Card
        title={t("option:flashcards.importTitle", {
          defaultValue: "Import Flashcards"
        })}
      >
        <ImportPanel onTransferAction={handleTransferAction} />
      </Card>
      <Card
        title={t("option:flashcards.exportTitle", {
          defaultValue: "Export Flashcards"
        })}
      >
        <ExportPanel onTransferAction={handleTransferAction} />
      </Card>
      <Card
        title={t("option:flashcards.generateTitle", {
          defaultValue: "Generate Flashcards"
        })}
      >
        <GeneratePanel
          initialIntent={generateIntent || null}
          onTransferAction={handleTransferAction}
        />
      </Card>
      <Card
        title={t("option:flashcards.occlusionTitle", {
          defaultValue: "Image Occlusion"
        })}
      >
        <ImageOcclusionTransferPanel onTransferAction={handleTransferAction} />
      </Card>
      <StudyPackCreateDrawer
        open={studyPackDrawerOpen}
        onClose={() => setStudyPackDrawerOpen(false)}
        initialIntent={studyPackIntent || null}
      />
    </div>
  )
}

export default ImportExportTab
