import { Alert, Button, Card, Form, Input, Select, Space, Typography } from "antd"
import { useQueryClient } from "@tanstack/react-query"
import React from "react"
import { useTranslation } from "react-i18next"

import { useAntdMessage } from "@/hooks/useAntdMessage"
import { useUndoNotification } from "@/hooks/useUndoNotification"
import { processInChunks } from "@/utils/chunk-processing"
import {
  deleteFlashcard,
  getFlashcard,
  type DeckReviewPromptSide,
  type Deck,
  type FlashcardCreate
} from "@/services/flashcards"
import { uploadFlashcardAsset } from "@/services/flashcard-assets"
import {
  useCreateDeckMutation,
  useCreateFlashcardsBulkMutation,
  useDecksQuery
} from "../hooks"
import { useDeckSchedulerDraft } from "../hooks/useDeckSchedulerDraft"
import { NewDeckConfigurationFields } from "../components/NewDeckConfigurationFields"
import { formatSchedulerSummary } from "../utils/scheduler-settings"
import { generateImageOcclusionAssets } from "../utils/image-occlusion-canvas"
import { ImageOcclusionPanel, type ImageOcclusionRegion } from "./ImageOcclusionPanel"

const { Text } = Typography

const OCCLUSION_UNDO_SECONDS = 30
const OCCLUSION_UNDO_CHUNK_SIZE = 50
const IMAGE_OCCLUSION_SYSTEM_TAG = "image-occlusion"
const MAX_OCCLUSION_REGIONS = 25
const NEW_DECK_OPTION_VALUE = "__new__" as const
type DeckSelectionValue = number | typeof NEW_DECK_OPTION_VALUE | null | undefined

type TransferActionStatus = "success" | "warning" | "error"

interface ImageOcclusionTransferPanelProps {
  onTransferAction?: (summary: {
    area: "occlusion"
    status: TransferActionStatus
    message: string
  }) => void
}

interface ImageOcclusionDraft {
  id: string
  front: string
  back: string
  notes: string
  tags: string[]
  source_ref_type: "manual"
  source_ref_id: string
}

interface ImageOcclusionPanelState {
  sourceFile: File | null
  sourceUrl: string | null
  regions: ImageOcclusionRegion[]
  selectedRegionId: string | null
}

const parseTagInput = (value: string): string[] => {
  const seen = new Set<string>()
  const tags: string[] = []
  value
    .split(",")
    .map((tag) => tag.trim())
    .filter((tag) => tag.length > 0)
    .forEach((tag) => {
      const normalized = tag.toLowerCase()
      if (seen.has(normalized)) return
      seen.add(normalized)
      tags.push(tag)
    })
  return tags
}

const appendSystemTag = (tags: string[], systemTag: string): string[] => {
  const seen = new Set<string>()
  const combined: string[] = []
  ;[...tags, systemTag].forEach((tag) => {
    const normalized = String(tag || "").trim()
    if (!normalized) return
    const key = normalized.toLowerCase()
    if (seen.has(key)) return
    seen.add(key)
    combined.push(normalized)
  })
  return combined
}

const toUploadFile = (blob: Blob, filename: string): File =>
  new File([blob], filename, {
    type: blob.type || "application/octet-stream"
  })

const toOcclusionNotes = (
  sourceReference: string,
  region: Pick<ImageOcclusionRegion, "x" | "y" | "width" | "height" | "label">
): string =>
  [
    "[image-occlusion]",
    `source=${sourceReference}`,
    `region=${region.x.toFixed(4)},${region.y.toFixed(4)},${region.width.toFixed(4)},${region.height.toFixed(4)}`,
    `label=${region.label.trim()}`
  ].join("\n")

const normalizeCreatedItems = (value: unknown): Array<{ uuid: string }> => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null
      const uuid = (entry as Record<string, unknown>).uuid
      if (typeof uuid !== "string" || uuid.trim().length === 0) {
        return null
      }
      return { uuid }
    })
    .filter((entry): entry is { uuid: string } => entry !== null)
}

export const ImageOcclusionTransferPanel: React.FC<ImageOcclusionTransferPanelProps> = ({
  onTransferAction
}) => {
  const { t } = useTranslation(["option", "common"])
  const message = useAntdMessage()
  const { showUndoNotification } = useUndoNotification()
  const queryClient = useQueryClient()
  const decksQuery = useDecksQuery()
  const createDeckMutation = useCreateDeckMutation()
  const createBulkMutation = useCreateFlashcardsBulkMutation()
  const decks = decksQuery.data || []

  const [panelState, setPanelState] = React.useState<ImageOcclusionPanelState>({
    sourceFile: null,
    sourceUrl: null,
    regions: [],
    selectedRegionId: null
  })
  const [targetDeckId, setTargetDeckId] = React.useState<DeckSelectionValue>(undefined)
  const [newDeckName, setNewDeckName] = React.useState(() =>
    t("option:flashcards.occlusionDeckName", {
      defaultValue: "Image Occlusion"
    })
  )
  const [reviewPromptSide, setReviewPromptSide] = React.useState<DeckReviewPromptSide>("front")
  const [tagsInput, setTagsInput] = React.useState("")
  const [drafts, setDrafts] = React.useState<ImageOcclusionDraft[]>([])
  const [error, setError] = React.useState<string | null>(null)
  const [isGenerating, setIsGenerating] = React.useState(false)
  const [isSaving, setIsSaving] = React.useState(false)
  const schedulerDraft = useDeckSchedulerDraft()
  const selectedDeck = React.useMemo(
    () => (typeof targetDeckId === "number" ? decks.find((deck) => deck.id === targetDeckId) ?? null : null),
    [decks, targetDeckId]
  )
  const deckOptions = React.useMemo(
    () => [
      ...decks.map((deck: Deck) => ({
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
      const schedulerSettings = schedulerDraft.getValidatedSettings()
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
  }, [createDeckMutation, decks, newDeckName, reviewPromptSide, schedulerDraft, t, targetDeckId])

  const updateDraft = React.useCallback((id: string, patch: Partial<ImageOcclusionDraft>) => {
    setDrafts((current) =>
      current.map((draft) => (draft.id === id ? { ...draft, ...patch } : draft))
    )
  }, [])

  const removeDraft = React.useCallback((id: string) => {
    setDrafts((current) => current.filter((draft) => draft.id !== id))
  }, [])

  const handleGenerateDrafts = React.useCallback(async () => {
    const sourceFile = panelState.sourceFile
    const regions = panelState.regions

    if (!sourceFile) {
      const warningCopy = t("option:flashcards.occlusionMissingSource", {
        defaultValue: "Select a source image before generating drafts."
      })
      setError(warningCopy)
      message.warning(warningCopy)
      return
    }
    if (regions.length === 0) {
      const warningCopy = t("option:flashcards.occlusionMissingRegions", {
        defaultValue: "Draw at least one region before generating drafts."
      })
      setError(warningCopy)
      message.warning(warningCopy)
      return
    }
    if (regions.length > MAX_OCCLUSION_REGIONS) {
      const warningCopy = t("option:flashcards.occlusionTooManyRegions", {
        defaultValue: "Limit each generation run to {{count}} regions or fewer.",
        count: MAX_OCCLUSION_REGIONS
      })
      setError(warningCopy)
      message.warning(warningCopy)
      return
    }

    const unlabeled = regions.filter((region) => region.label.trim().length === 0)
    if (unlabeled.length > 0) {
      const warningCopy = t("option:flashcards.occlusionMissingLabels", {
        defaultValue: "Every region needs an answer label before you generate drafts."
      })
      setError(warningCopy)
      message.warning(warningCopy)
      return
    }

    setIsGenerating(true)
    setError(null)

    try {
      const assets = await generateImageOcclusionAssets(sourceFile, regions)
      const batchTags = appendSystemTag(parseTagInput(tagsInput), IMAGE_OCCLUSION_SYSTEM_TAG)

      const sourceAsset = await uploadFlashcardAsset(
        toUploadFile(assets.source.blob, "image-occlusion-source.webp")
      )

      const nextDrafts: ImageOcclusionDraft[] = []
      for (const [index, generatedRegion] of assets.regions.entries()) {
        const region = regions.find((item) => item.id === generatedRegion.regionId)
        if (!region) continue
        const promptAsset = await uploadFlashcardAsset(
          toUploadFile(generatedRegion.promptBlob, `image-occlusion-prompt-${index + 1}.webp`)
        )
        const answerAsset = await uploadFlashcardAsset(
          toUploadFile(generatedRegion.answerBlob, `image-occlusion-answer-${index + 1}.webp`)
        )

        nextDrafts.push({
          id: `occlusion-${generatedRegion.regionId}`,
          front: `Identify the occluded region.\n\n${promptAsset.markdown_snippet}`,
          back: `${region.label.trim()}\n\n${answerAsset.markdown_snippet}`,
          notes: toOcclusionNotes(sourceAsset.reference, region),
          tags: batchTags,
          source_ref_type: "manual",
          source_ref_id: `image-occlusion:${sourceAsset.asset_uuid}:${index}`
        })
      }

      setDrafts(nextDrafts)
      const successCopy = t("option:flashcards.occlusionDraftsReady", {
        defaultValue: "Prepared {{count}} image occlusion drafts.",
        count: nextDrafts.length
      })
      message.success(successCopy)
      onTransferAction?.({
        area: "occlusion",
        status: "success",
        message: successCopy
      })
    } catch (generationError: unknown) {
      const errorMessage =
        generationError instanceof Error
          ? generationError.message
          : t("option:flashcards.occlusionGenerationFailed", {
              defaultValue: "Image occlusion generation failed."
            })
      setError(errorMessage)
      message.error(errorMessage)
      onTransferAction?.({
        area: "occlusion",
        status: "error",
        message: errorMessage
      })
    } finally {
      setIsGenerating(false)
    }
  }, [message, onTransferAction, panelState, t, tagsInput])

  const handleSaveDrafts = React.useCallback(async () => {
    const validDrafts = drafts.filter((draft) => draft.front.trim() && draft.back.trim())
    if (validDrafts.length === 0) {
      message.warning(
        t("option:flashcards.occlusionNoDraftsToSave", {
          defaultValue: "Generate at least one valid draft before saving."
        })
      )
      return
    }

    setIsSaving(true)
    setError(null)
    try {
      const deckId = await resolveTargetDeckId()
      const payload: FlashcardCreate[] = validDrafts.map((draft) => ({
        deck_id: deckId,
        front: draft.front.trim(),
        back: draft.back.trim(),
        notes: draft.notes || undefined,
        extra: undefined,
        tags: draft.tags,
        model_type: "basic",
        reverse: false,
        is_cloze: false,
        source_ref_type: draft.source_ref_type,
        source_ref_id: draft.source_ref_id
      }))

      const created = await createBulkMutation.mutateAsync(payload)
      const createdItems = normalizeCreatedItems(created.items)
      setDrafts([])

      const successCopy = t("option:flashcards.occlusionSaveSuccess", {
        defaultValue: "Saved {{count}} image occlusion cards.",
        count: createdItems.length
      })
      message.success(successCopy)
      onTransferAction?.({
        area: "occlusion",
        status: "success",
        message: successCopy
      })

      if (createdItems.length > 0) {
        showUndoNotification({
          title: t("option:flashcards.occlusionUndoTitle", {
            defaultValue: "Image occlusion saved"
          }),
          description: t("option:flashcards.importUndoHint", {
            defaultValue: "Undo within {{seconds}}s to remove {{count}} imported cards.",
            seconds: OCCLUSION_UNDO_SECONDS,
            count: createdItems.length
          }),
          duration: OCCLUSION_UNDO_SECONDS,
          onUndo: async () => {
            let failedRollbacks = 0
            await processInChunks(createdItems, OCCLUSION_UNDO_CHUNK_SIZE, async (chunk) => {
              const results = await Promise.allSettled(
                chunk.map(async (item) => {
                  const latest = await getFlashcard(item.uuid)
                  await deleteFlashcard(item.uuid, latest.version)
                })
              )
              failedRollbacks += results.filter((result) => result.status === "rejected").length
            })
            await queryClient.invalidateQueries({
              predicate: (query) =>
                Array.isArray(query.queryKey) &&
                typeof query.queryKey[0] === "string" &&
                query.queryKey[0].startsWith("flashcards:")
            })
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
    } catch (saveError: unknown) {
      const errorMessage =
        saveError instanceof Error
          ? saveError.message
          : t("option:flashcards.occlusionSaveFailed", {
              defaultValue: "Saving image occlusion cards failed."
            })
      setError(errorMessage)
      message.error(errorMessage)
      onTransferAction?.({
        area: "occlusion",
        status: "error",
        message: errorMessage
      })
    } finally {
      setIsSaving(false)
    }
  }, [
    createBulkMutation,
    drafts,
    message,
    onTransferAction,
    queryClient,
    resolveTargetDeckId,
    showUndoNotification,
    t
  ])

  return (
    <div className="flex flex-col gap-3">
      <Text type="secondary">
        {t("option:flashcards.occlusionTransferHelp", {
          defaultValue:
            "Upload one image, draw labeled rectangular occlusions, generate drafts, then save them in bulk."
        })}
      </Text>

      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        <Form.Item
          label={t("option:flashcards.deck", { defaultValue: "Deck" })}
          className="!mb-2"
        >
          <Select
            allowClear
            value={targetDeckId ?? undefined}
            onChange={(value) => setTargetDeckId((value as DeckSelectionValue) ?? null)}
            data-testid="flashcards-occlusion-deck"
            options={deckOptions}
          />
        </Form.Item>
        {targetDeckId === NEW_DECK_OPTION_VALUE ? (
          <NewDeckConfigurationFields
            deckName={newDeckName}
            onDeckNameChange={setNewDeckName}
            reviewPromptSide={reviewPromptSide}
            onReviewPromptSideChange={setReviewPromptSide}
            schedulerDraft={schedulerDraft}
            nameTestId="flashcards-occlusion-new-deck-name"
          />
        ) : selectedDeck?.scheduler_settings ? (
          <Text
            type="secondary"
            className="block text-xs -mt-2 mb-2"
            data-testid="flashcards-occlusion-selected-deck-summary"
          >
            {formatSchedulerSummary(selectedDeck.scheduler_type, selectedDeck.scheduler_settings)}
          </Text>
        ) : null}
        <Form.Item
          label={t("option:flashcards.tags", { defaultValue: "Tags" })}
          className="!mb-2"
        >
          <Input
            value={tagsInput}
            onChange={(event) => setTagsInput(event.target.value)}
            data-testid="flashcards-occlusion-tags"
            placeholder={t("option:flashcards.tagsPlaceholder", {
              defaultValue: "tag-1, tag-2"
            })}
          />
        </Form.Item>
      </div>

      <ImageOcclusionPanel onChange={setPanelState} />

      {error && <Alert type="error" showIcon title={error} />}

      <Button
        type="primary"
        onClick={handleGenerateDrafts}
        loading={isGenerating}
        data-testid="flashcards-occlusion-generate-button"
      >
        {t("option:flashcards.occlusionGenerateButton", {
          defaultValue: "Generate occlusion drafts"
        })}
      </Button>

      {drafts.length > 0 && (
        <div className="space-y-2">
          <Text strong>
            {t("option:flashcards.occlusionDraftsTitle", {
              defaultValue: "Generated occlusion drafts"
            })}
          </Text>
          {drafts.map((draft, index) => (
            <Card
              key={draft.id}
              size="small"
              title={t("option:flashcards.occlusionDraftTitle", {
                defaultValue: "Occlusion {{index}}",
                index: index + 1
              })}
              extra={
                <Button
                  type="text"
                  danger
                  size="small"
                  onClick={() => removeDraft(draft.id)}
                >
                  {t("common:remove", { defaultValue: "Remove" })}
                </Button>
              }
            >
              <Space orientation="vertical" className="w-full">
                <Input.TextArea
                  rows={3}
                  value={draft.front}
                  data-testid={`flashcards-occlusion-draft-front-${draft.id}`}
                  onChange={(event) => updateDraft(draft.id, { front: event.target.value })}
                />
                <Input.TextArea
                  rows={3}
                  value={draft.back}
                  data-testid={`flashcards-occlusion-draft-back-${draft.id}`}
                  onChange={(event) => updateDraft(draft.id, { back: event.target.value })}
                />
                <Input
                  value={draft.tags.join(", ")}
                  onChange={(event) =>
                    updateDraft(draft.id, {
                      tags: appendSystemTag(
                        parseTagInput(event.target.value),
                        IMAGE_OCCLUSION_SYSTEM_TAG
                      )
                    })
                  }
                />
              </Space>
            </Card>
          ))}
          <Button
            type="primary"
            onClick={handleSaveDrafts}
            loading={isSaving || createBulkMutation.isPending || createDeckMutation.isPending}
            data-testid="flashcards-occlusion-save-button"
          >
            {t("option:flashcards.occlusionSaveButton", {
              defaultValue: "Save occlusion cards"
            })}
          </Button>
        </div>
      )}
    </div>
  )
}

export default ImageOcclusionTransferPanel
