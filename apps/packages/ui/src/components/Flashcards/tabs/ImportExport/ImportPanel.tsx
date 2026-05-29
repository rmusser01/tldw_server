import React from "react"
import { useQueryClient } from "@tanstack/react-query"
import {
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

import { Alert } from "@/components/ui/primitives"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { useUndoNotification } from "@/hooks/useUndoNotification"
import { processInChunks } from "@/utils/chunk-processing"
import {
  deleteFlashcard,
  getFlashcard,
  type DeckReviewPromptSide,
  type FlashcardsImportError
} from "@/services/flashcards"

import { FileDropZone } from "../../components"
import { NewDeckConfigurationFields } from "../../components/NewDeckConfigurationFields"
import { FLASHCARDS_HELP_LINKS } from "../../constants"
import {
  useCreateDeckMutation,
  useCreateFlashcardsBulkMutation,
  useDecksQuery,
  useImportFlashcardsMutation,
  useImportFlashcardsApkgMutation,
  useImportFlashcardsJsonMutation,
  useImportLimitsQuery,
  usePreviewStructuredQaImportMutation
} from "../../hooks"
import { useDeckSchedulerDraft } from "../../hooks/useDeckSchedulerDraft"
import { getUtf8ByteLength } from "../../utils/field-byte-limit"
import { formatSchedulerSummary } from "../../utils/scheduler-settings"
import {
  APKG_ESTIMATED_BYTES_PER_CARD,
  DeckSelectionValue,
  IMPORT_HELP_ANCHORS,
  IMPORT_UNDO_CHUNK_SIZE,
  IMPORT_UNDO_SECONDS,
  LARGE_IMPORT_CONFIRM_THRESHOLD_APKG_BYTES,
  LARGE_IMPORT_CONFIRM_THRESHOLD_ROWS,
  NEW_DECK_OPTION_VALUE,
  SUPPORTED_DELIMITERS,
  SupportedDelimiter,
  TransferActionReporterProps,
  buildStructuredDraftSaveError,
  countDelimiterOccurrences,
  detectJsonImportFormat,
  estimateJsonItemCount,
  getImportErrorGuidance,
  normalizeHeaderToken,
  normalizeImportLimits,
  normalizeImportErrors,
  normalizeImportedItems,
  normalizeStructuredDrafts,
  type ImportMode,
  type ImportResultSummary,
  type StructuredImportDraft
} from "./shared"

const { Text } = Typography

/**
 * Import panel for CSV/TSV flashcard import.
 */
export const ImportPanel: React.FC<TransferActionReporterProps> = ({ onTransferAction }) => {
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
  const importLimits = React.useMemo(
    () => normalizeImportLimits(limitsQuery.data),
    [limitsQuery.data]
  )
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
      {importLimits && (
        <Text type="secondary" className="text-xs">
          {t("option:flashcards.importLimitsValue", {
            defaultValue:
              "Limits: max {{maxLines}} lines, {{maxLineBytes}} bytes per line, {{maxFieldBytes}} bytes per field",
            maxLines: importLimits.maxLines.toLocaleString(),
            maxLineBytes: importLimits.maxLineLengthBytes.toLocaleString(),
            maxFieldBytes: importLimits.maxFieldLengthBytes.toLocaleString()
          })}
        </Text>
      )}
      {importPreflightWarning && (
        <Alert
          variant="warning"
          data-testid="flashcards-import-preflight-warning"
          title={t("option:flashcards.importPreflightTitle", {
            defaultValue: "Check import format before continuing"
          })}
        >
          {importPreflightWarning}
        </Alert>
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
          variant="warning"
          data-testid="flashcards-structured-preview-errors"
          title={t("option:flashcards.structuredPreviewErrorsTitle", {
            defaultValue: "Preview warnings"
          })}
        >
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
        </Alert>
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
          variant={lastResult.errors.length > 0 ? "warning" : "success"}
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
        >
          {lastResult.errors.length > 0 && (
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
          )}
        </Alert>
      )}
    </div>
  )
}
