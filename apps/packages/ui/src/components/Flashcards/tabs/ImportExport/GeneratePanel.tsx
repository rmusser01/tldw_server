import React from "react"
import { Link } from "react-router-dom"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Button, Card, Form, Input, Select, Space, Tooltip, Typography } from "antd"
import { useTranslation } from "react-i18next"

import { Alert, type AlertVariant } from "@/components/ui/primitives"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { getLlmProviders } from "@/services/prompt-studio"
import type { DeckReviewPromptSide } from "@/services/flashcards"

import { NewDeckConfigurationFields } from "../../components/NewDeckConfigurationFields"
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useDecksQuery,
  useGenerateFlashcardsMutation
} from "../../hooks"
import { useDeckSchedulerDraft } from "../../hooks/useDeckSchedulerDraft"
import { formatSchedulerSummary } from "../../utils/scheduler-settings"
import {
  DeckSelectionValue,
  GeneratePanelProps,
  GenerateSourceContext,
  GeneratedCardDraft,
  NEW_DECK_OPTION_VALUE,
  TransferActionReporterProps,
  normalizeGeneratedCards
} from "./shared"

const { Text } = Typography

type GeneratedSaveStatus = {
  variant: AlertVariant
  title: string
  detail: string
  retryable: boolean
}

/**
 * Generate panel for LLM-assisted card generation from free text.
 */
export const GeneratePanel: React.FC<GeneratePanelProps & TransferActionReporterProps> = ({
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
      initialIntent.sourceType !== "manual" &&
      initialIntent.sourceType !== "media" &&
      initialIntent.sourceType !== "note" &&
      initialIntent.sourceType !== "message"
    ) {
      return null
    }
    const sourceId = initialIntent.sourceId?.trim() || null
    const sourceTitle = initialIntent.sourceTitle?.trim() || null
    if (initialIntent.sourceType === "manual" && !sourceId && !sourceTitle) {
      return null
    }
    return {
      sourceType: initialIntent.sourceType,
      sourceId: sourceId || (initialIntent.sourceType === "manual" ? sourceTitle : null),
      sourceTitle
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
  const [saveStatus, setSaveStatus] = React.useState<GeneratedSaveStatus | null>(null)
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

  const clearRetryableSaveStatus = React.useCallback(() => {
    setSaveStatus((current) => (current?.retryable ? null : current))
  }, [])

  const updateGeneratedCard = React.useCallback(
    (id: string, patch: Partial<GeneratedCardDraft>) => {
      clearRetryableSaveStatus()
      setGeneratedCards((prev) =>
        prev.map((card) => (card.id === id ? { ...card, ...patch } : card))
      )
    },
    [clearRetryableSaveStatus]
  )

  const removeGeneratedCard = React.useCallback((id: string) => {
    clearRetryableSaveStatus()
    setGeneratedCards((prev) => prev.filter((card) => card.id !== id))
  }, [clearRetryableSaveStatus])

  const handleGenerate = React.useCallback(async () => {
    try {
      setGenerationError(null)
      setSaveStatus(null)
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
    setSaveStatus(null)
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
        setSaveStatus({
          variant: "success",
          title: successCopy,
          detail: t("option:flashcards.generateSaveSuccessDetail", {
            defaultValue: "All generated drafts were saved to the selected deck."
          }),
          retryable: false
        })
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
        setSaveStatus({
          variant: "warning",
          title: warningCopy,
          detail: t("option:flashcards.generateSavePartialDetail", {
            defaultValue:
              "Only failed drafts remain below. Review them, then retry saving."
          }),
          retryable: true
        })
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
      setSaveStatus({
        variant: "error",
        title: errorCopy,
        detail: t("option:flashcards.generateSaveFailedDetail", {
          defaultValue:
            "All generated drafts are still available. Check the deck and draft content, then retry saving."
        }),
        retryable: true
      })
      onTransferAction?.({
        area: "generate",
        status: "error",
        message: errorCopy
      })
    } catch (e: unknown) {
      const errorCopy =
        e instanceof Error && e.message
          ? e.message
          : t("option:flashcards.generateSaveFailed", {
              defaultValue: "Failed to save generated cards."
            })
      message.error(errorCopy)
      setSaveStatus({
        variant: "error",
        title: errorCopy,
        detail: t("option:flashcards.generateSaveFatalDetail", {
          defaultValue:
            "Generated drafts are still available. Check the deck settings and draft content, then retry saving."
        }),
        retryable: generatedCards.length > 0
      })
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

  const saveInProgress = isSaving || createMutation.isPending || createDeckMutation.isPending

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
          variant="info"
          data-testid="flashcards-generate-source-context"
          title={t("option:flashcards.generateSourceContextTitle", {
            defaultValue: "Source context attached"
          })}
        >
          {t("option:flashcards.generateSourceContextDescription", {
            defaultValue:
              "Cards saved from this draft will be linked to {{sourceType}} {{sourceId}}.",
            sourceType: sourceContext.sourceType,
            sourceId:
              sourceContext.sourceTitle ||
              sourceContext.sourceId ||
              t("option:flashcards.unknownSource", {
                defaultValue: "unknown source"
              })
          })}
        </Alert>
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
            variant="info"
            className="mb-3 md:col-span-2"
            data-testid="flashcards-generate-no-llm-banner"
          >
            {t("option:flashcards.generateNoLlmBanner", {
              defaultValue:
                "Flashcard generation requires an LLM provider. Configure one in Settings \u2192 LLM Providers."
            })}
          </Alert>
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
          variant="error"
          title={generationError}
        >
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
        </Alert>
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

      {saveStatus && (
        <Alert
          variant={saveStatus.variant}
          title={saveStatus.title}
          action={
            saveStatus.retryable
              ? {
                  label: t("option:flashcards.generateSaveRetry", {
                    defaultValue: "Retry saving remaining drafts"
                  }),
                  onClick: handleSaveGeneratedCards,
                  loading: saveInProgress,
                  disabled: generatedCards.length === 0 || saveInProgress,
                  "data-testid": "flashcards-generate-save-retry"
                }
              : undefined
          }
          data-testid="flashcards-generate-save-status"
        >
          {saveStatus.detail}
        </Alert>
      )}

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
            loading={saveInProgress}
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
