import React from "react"
import { Alert, Button, Card, Collapse, Empty, Input, List, Space, Typography } from "antd"
import { useTranslation } from "react-i18next"

import {
  useDecksQuery,
  useDueCountsQuery,
  useUpdateDeckMutation
} from "../hooks/useFlashcardQueries"
import { DeckSchedulerSettingsEditor } from "../components/DeckSchedulerSettingsEditor"
import { useDeckSchedulerDraft } from "../hooks/useDeckSchedulerDraft"
import { DeckStudyDefaultsFields } from "../components/DeckStudyDefaultsFields"
import type { Deck } from "@/services/flashcards"
import type { SchedulerSettingsDraft } from "../utils/scheduler-settings"
import { createSchedulerDraft, formatSchedulerSummary } from "../utils/scheduler-settings"

const { Text, Title } = Typography

type EditorStatus = "idle" | "dirty" | "saving" | "saved" | "conflict"

export interface SchedulerTabProps {
  isActive?: boolean
  onDirtyChange?: (dirty: boolean) => void
  discardSignal?: number
}

const matchesDeckQuery = (deck: Deck, query: string): boolean => {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) return true
  const haystack = `${deck.name} ${deck.description ?? ""}`.trim().toLowerCase()
  return haystack.includes(normalizedQuery)
}

const isVersionConflict = (error: unknown): boolean =>
  typeof error === "object" &&
  error !== null &&
  "response" in error &&
  typeof (error as { response?: { status?: unknown } }).response?.status === "number" &&
  (error as { response?: { status?: number } }).response?.status === 409

const cloneDraft = (draft: SchedulerSettingsDraft): SchedulerSettingsDraft => ({
  scheduler_type: draft.scheduler_type,
  sm2_plus: { ...draft.sm2_plus },
  fsrs: { ...draft.fsrs }
})

export const SchedulerTab: React.FC<SchedulerTabProps> = ({
  isActive = false,
  onDirtyChange,
  discardSignal = 0
}) => {
  const { t } = useTranslation(["option", "common"])
  const decksQuery = useDecksQuery()
  const updateDeckMutation = useUpdateDeckMutation()
  const schedulerDraft = useDeckSchedulerDraft()

  const [searchText, setSearchText] = React.useState("")
  const [selectedDeckId, setSelectedDeckId] = React.useState<number | null>(null)
  const [copyDeckId, setCopyDeckId] = React.useState("")
  const [editorStatus, setEditorStatus] = React.useState<EditorStatus>("idle")
  const [baseDeckId, setBaseDeckId] = React.useState<number | null>(null)
  const [baseVersion, setBaseVersion] = React.useState<number | null>(null)
  const [baseDraft, setBaseDraft] = React.useState<SchedulerSettingsDraft | null>(null)
  const [conflictDraft, setConflictDraft] = React.useState<SchedulerSettingsDraft | null>(null)
  const [reviewPromptSide, setReviewPromptSide] =
    React.useState<Deck["review_prompt_side"]>("front")
  const [baseReviewPromptSide, setBaseReviewPromptSide] =
    React.useState<Deck["review_prompt_side"]>("front")
  const [conflictReviewPromptSide, setConflictReviewPromptSide] =
    React.useState<Deck["review_prompt_side"] | null>(null)
  const [saveError, setSaveError] = React.useState<string | null>(null)

  const allDecks = decksQuery.data ?? []
  const visibleDecks = React.useMemo(
    () => allDecks.filter((deck) => matchesDeckQuery(deck, searchText)),
    [allDecks, searchText]
  )

  React.useEffect(() => {
    if (allDecks.length === 0) {
      setSelectedDeckId(null)
      return
    }

    const stillExists = selectedDeckId != null && allDecks.some((deck) => deck.id === selectedDeckId)
    if (!stillExists) {
      setSelectedDeckId(visibleDecks[0]?.id ?? allDecks[0]?.id ?? null)
    }
  }, [allDecks, selectedDeckId, visibleDecks])

  const activeDeck = React.useMemo(
    () => allDecks.find((deck) => deck.id === selectedDeckId) ?? null,
    [allDecks, selectedDeckId]
  )
  const activeDeckCounts = useDueCountsQuery(activeDeck?.id, {
    enabled: isActive && !!activeDeck
  })

  const syncDraftFromDeck = React.useCallback(
    (deck: Deck, status: EditorStatus = "idle") => {
      const nextDraft = createSchedulerDraft({
        schedulerType: deck.scheduler_type,
        settings: deck.scheduler_settings
      })
      schedulerDraft.replaceDraftState(nextDraft)
      setReviewPromptSide(deck.review_prompt_side)
      setBaseDeckId(deck.id)
      setBaseVersion(deck.version)
      setBaseDraft(cloneDraft(nextDraft))
      setBaseReviewPromptSide(deck.review_prompt_side)
      setCopyDeckId("")
      setConflictDraft(null)
      setConflictReviewPromptSide(null)
      setSaveError(null)
      setEditorStatus(status)
    },
    [schedulerDraft]
  )

  const refreshDeckFromServer = React.useCallback(
    async (deckId: number, status: EditorStatus = "idle") => {
      const response = await decksQuery.refetch()
      const refreshedDecks = response.data ?? decksQuery.data ?? []
      const latestDeck = refreshedDecks.find((deck) => deck.id === deckId) ?? null
      if (latestDeck) {
        const latestDraft = createSchedulerDraft({
          schedulerType: latestDeck.scheduler_type,
          settings: latestDeck.scheduler_settings
        })
        schedulerDraft.replaceDraftState(latestDraft)
        setReviewPromptSide(latestDeck.review_prompt_side)
        setBaseDeckId(latestDeck.id)
        setBaseVersion(latestDeck.version)
        setBaseDraft(cloneDraft(latestDraft))
        setBaseReviewPromptSide(latestDeck.review_prompt_side)
        setCopyDeckId("")
        setConflictDraft(null)
        setConflictReviewPromptSide(null)
        setSaveError(null)
        setEditorStatus(status)
      }
      return latestDeck
    },
    [decksQuery, schedulerDraft]
  )

  React.useEffect(() => {
    if (!activeDeck) {
      setBaseDeckId(null)
      setBaseVersion(null)
      setBaseDraft(null)
      setReviewPromptSide("front")
      setBaseReviewPromptSide("front")
      setCopyDeckId("")
      setConflictDraft(null)
      setConflictReviewPromptSide(null)
      setSaveError(null)
      setEditorStatus("idle")
      return
    }

    if (baseDeckId !== activeDeck.id) {
      syncDraftFromDeck(activeDeck)
      return
    }

    if (
      (editorStatus === "idle" || editorStatus === "saved") &&
      baseVersion != null &&
      activeDeck.version > baseVersion
    ) {
      syncDraftFromDeck(activeDeck, editorStatus)
    }
  }, [activeDeck, baseDeckId, baseVersion, editorStatus, syncDraftFromDeck])

  const draftChanged = React.useMemo(() => {
    if (!baseDraft) return false
    return JSON.stringify(schedulerDraft.draft) !== JSON.stringify(baseDraft)
  }, [baseDraft, schedulerDraft.draft])
  const reviewPromptSideChanged = reviewPromptSide !== baseReviewPromptSide

  React.useEffect(() => {
    if (editorStatus === "saving" || editorStatus === "conflict") return
    if (draftChanged || reviewPromptSideChanged) {
      setEditorStatus("dirty")
    } else if (editorStatus === "dirty") {
      setEditorStatus("idle")
    }
  }, [draftChanged, editorStatus, reviewPromptSideChanged])

  const otherDecks = React.useMemo(
    () => allDecks.filter((deck) => deck.id !== activeDeck?.id),
    [activeDeck?.id, allDecks]
  )

  const draftPreview = schedulerDraft.summary

  const applyPreset = React.useCallback(
    (presetId: Parameters<typeof schedulerDraft.applyPreset>[0]) => {
      schedulerDraft.applyPreset(presetId)
      setConflictDraft(null)
      setSaveError(null)
      setEditorStatus("dirty")
    },
    [schedulerDraft]
  )

  const copyFromSelectedDeck = React.useCallback(() => {
    const sourceDeck = otherDecks.find((deck) => String(deck.id) === copyDeckId)
    if (!sourceDeck) return
    schedulerDraft.replaceDraft({
      schedulerType: sourceDeck.scheduler_type,
      settings: sourceDeck.scheduler_settings
    })
    setConflictDraft(null)
    setConflictReviewPromptSide(null)
    setSaveError(null)
    setEditorStatus("dirty")
  }, [copyDeckId, otherDecks, schedulerDraft])

  const resetToDefaults = React.useCallback(() => {
    schedulerDraft.resetToDefaults()
    setConflictDraft(null)
    setConflictReviewPromptSide(null)
    setSaveError(null)
    setEditorStatus("dirty")
  }, [schedulerDraft])

  const reloadLatest = React.useCallback(async () => {
    if (!activeDeck) return
    setSaveError(null)

    try {
      await refreshDeckFromServer(activeDeck.id)
    } catch (_error) {
      setSaveError(
        t("option:flashcards.schedulerReloadError", {
          defaultValue: "Failed to reload scheduler settings."
        })
      )
    }
  }, [activeDeck, refreshDeckFromServer, t])

  const reapplyConflict = React.useCallback(() => {
    if (!conflictDraft) return
    schedulerDraft.replaceDraftState(cloneDraft(conflictDraft))
    if (conflictReviewPromptSide) {
      setReviewPromptSide(conflictReviewPromptSide)
    }
    setEditorStatus("dirty")
    setSaveError(null)
  }, [conflictDraft, conflictReviewPromptSide, schedulerDraft])

  const handleSave = React.useCallback(async () => {
    if (!activeDeck) return

    const parsed = schedulerDraft.getValidatedSettings()
    if (!parsed) return

    setEditorStatus("saving")
    setSaveError(null)

    try {
      const updatedDeck = await updateDeckMutation.mutateAsync({
        deckId: activeDeck.id,
        update: {
          review_prompt_side: reviewPromptSide,
          scheduler_type: parsed.scheduler_type,
          scheduler_settings: parsed.scheduler_settings,
          expected_version: baseVersion ?? activeDeck.version
        }
      })
      syncDraftFromDeck(updatedDeck, "saved")
    } catch (error: unknown) {
      if (isVersionConflict(error)) {
        const pendingDraft = cloneDraft(schedulerDraft.draft)
        const pendingReviewPromptSide = reviewPromptSide

        try {
          const latestDeck = await refreshDeckFromServer(activeDeck.id)
          if (!latestDeck) {
            syncDraftFromDeck(activeDeck)
          }
        } catch (_refreshError) {
          syncDraftFromDeck(activeDeck)
        }

        setConflictDraft(pendingDraft)
        setConflictReviewPromptSide(pendingReviewPromptSide)
        setEditorStatus("conflict")
        return
      }

      setEditorStatus("dirty")
      setSaveError(
        t("option:flashcards.schedulerSaveError", {
          defaultValue: "Failed to save deck settings."
        })
      )
    }
  }, [
    activeDeck,
    baseVersion,
    refreshDeckFromServer,
    reviewPromptSide,
    schedulerDraft,
    t,
    updateDeckMutation
  ])

  const isDirty = draftChanged || reviewPromptSideChanged || editorStatus === "conflict"

  React.useEffect(() => {
    onDirtyChange?.(isDirty)
  }, [isDirty, onDirtyChange])

  React.useEffect(() => {
    if (discardSignal <= 0) return

    if (!activeDeck) {
      setBaseDeckId(null)
      setBaseVersion(null)
      setBaseDraft(null)
      setReviewPromptSide("front")
      setBaseReviewPromptSide("front")
      setCopyDeckId("")
      setConflictDraft(null)
      setConflictReviewPromptSide(null)
      setSaveError(null)
      setEditorStatus("idle")
      return
    }

    syncDraftFromDeck(activeDeck)
  }, [activeDeck, discardSignal, syncDraftFromDeck])

  const confirmDiscardChanges = React.useCallback(() => {
    if (!isDirty) return true
    return window.confirm(
      t("option:flashcards.schedulerDiscardChangesPrompt", {
        defaultValue: "Discard unsaved deck changes?"
      })
    )
  }, [isDirty, t])

  return (
    <div className="space-y-4">
      <div>
        <Title level={4} className="!mb-1">
          {t("option:flashcards.schedulerTabTitle", { defaultValue: "Scheduler" })}
        </Title>
        <Text type="secondary">
          {t("option:flashcards.schedulerTabDescription", {
            defaultValue: "Choose a deck to inspect and edit its spaced-repetition policy."
          })}
        </Text>
      </div>

      <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
        <Card size="small" title={t("option:flashcards.schedulerDecks", { defaultValue: "Decks" })}>
          <Space orientation="vertical" size={12} className="w-full">
            <Input.Search
              value={searchText}
              onChange={(event) => setSearchText(event.target.value)}
              placeholder={t("option:flashcards.searchDecks", { defaultValue: "Search decks" })}
              allowClear
            />

            {visibleDecks.length === 0 ? (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={t("option:flashcards.noDecksForScheduler", {
                  defaultValue: "No decks match the current filter."
                })}
              />
            ) : (
              <List
                dataSource={visibleDecks}
                renderItem={(deck) => (
                  <List.Item className="!px-0">
                    <Button
                      type={deck.id === selectedDeckId ? "primary" : "text"}
                      className="!h-auto !w-full !justify-start"
                      onClick={() => {
                        if (deck.id === selectedDeckId) return
                        if (!confirmDiscardChanges()) return
                        setSelectedDeckId(deck.id)
                      }}
                    >
                      <div className="min-w-0 text-left">
                        <div className="font-medium">{deck.name}</div>
                        {deck.description ? (
                          <div className="text-xs text-text-muted">{deck.description}</div>
                        ) : null}
                        <div className="text-xs text-text-subtle">
                          {formatSchedulerSummary(deck.scheduler_type, deck.scheduler_settings)}
                        </div>
                      </div>
                    </Button>
                  </List.Item>
                )}
              />
            )}
          </Space>
        </Card>

        <Card size="small">
          {activeDeck ? (
            <Space orientation="vertical" size={16} className="w-full">
              <div>
                <Title level={5} className="!mb-0">
                  {t("option:flashcards.schedulerDeckHeading", {
                    defaultValue: "{{name}} Scheduler",
                    name: activeDeck.name
                  })}
                </Title>
                <Text type="secondary">
                  {t("option:flashcards.schedulerDeckVersion", {
                    defaultValue: "Version {{version}}",
                    version: baseVersion ?? activeDeck.version
                  })}
                </Text>
              </div>

              {editorStatus === "conflict" && (
                <div
                  className="rounded border border-amber-300 bg-amber-50 p-3"
                  data-testid="flashcards-scheduler-conflict"
                >
                  <div className="font-medium">
                    {t("option:flashcards.schedulerConflictTitle", {
                      defaultValue: "Deck settings changed elsewhere."
                    })}
                  </div>
                  <div className="text-sm text-text-muted">
                    {t("option:flashcards.schedulerConflictDescription", {
                      defaultValue: "Reload the latest saved settings or reapply your draft before saving again."
                    })}
                  </div>
                  <Space className="mt-3">
                    <Button size="small" onClick={reloadLatest}>
                      {t("option:flashcards.schedulerReloadLatest", {
                        defaultValue: "Reload latest"
                      })}
                    </Button>
                    <Button size="small" type="primary" onClick={reapplyConflict}>
                      {t("option:flashcards.schedulerReapplyDraft", {
                        defaultValue: "Reapply my draft"
                      })}
                    </Button>
                  </Space>
                </div>
              )}

              {saveError && <Alert type="error" message={saveError} />}

              {schedulerDraft.draft.scheduler_type === "fsrs" &&
                activeDeck.scheduler_type !== "fsrs" && (
                  <Alert
                    type="info"
                    message={t("option:flashcards.fsrsSwitchInfoTitle", {
                      defaultValue: "Switching this deck to FSRS"
                    })}
                    description={t("option:flashcards.fsrsSwitchInfoBody", {
                      defaultValue:
                        "Existing cards keep their review history. FSRS state will be derived conservatively as cards are reviewed."
                    })}
                  />
                )}

              <Card
                size="small"
                title={t("option:flashcards.schedulerSummaryCard", {
                  defaultValue: "Current draft"
                })}
              >
                <Text type={draftPreview ? "secondary" : "danger"}>
                  {draftPreview ??
                    t("option:flashcards.schedulerDraftInvalid", {
                      defaultValue: "Draft has validation errors."
                    })}
                </Text>
              </Card>

              <Card
                size="small"
                title={t("option:flashcards.schedulerCountsCard", {
                  defaultValue: "Active deck counts"
                })}
              >
                <div className="grid gap-3 sm:grid-cols-4">
                  <div>
                    <Text type="secondary">Due review</Text>
                    <div className="text-lg font-semibold">{activeDeckCounts.data?.due ?? 0}</div>
                  </div>
                  <div>
                    <Text type="secondary">New</Text>
                    <div className="text-lg font-semibold">{activeDeckCounts.data?.new ?? 0}</div>
                  </div>
                  <div>
                    <Text type="secondary">Learning</Text>
                    <div className="text-lg font-semibold">{activeDeckCounts.data?.learning ?? 0}</div>
                  </div>
                  <div>
                    <Text type="secondary">Total due</Text>
                    <div className="text-lg font-semibold">{activeDeckCounts.data?.total ?? 0}</div>
                  </div>
                </div>
              </Card>

              <Card
                size="small"
                title={t("option:flashcards.schedulerTools", { defaultValue: "Tools" })}
              >
                <Space orientation="vertical" size={12} className="w-full">
                  <div className="flex flex-wrap items-end gap-2">
                    <label className="flex min-w-[220px] flex-col gap-1 text-sm text-text-muted">
                      <span>
                        {t("option:flashcards.schedulerCopyFromDeck", {
                          defaultValue: "Copy from deck"
                        })}
                      </span>
                      <select
                        value={copyDeckId}
                        onChange={(event) => setCopyDeckId(event.target.value)}
                        data-testid="flashcards-scheduler-copy-select"
                        className="rounded border border-border bg-background px-2 py-1"
                      >
                        <option value="">
                          {t("option:flashcards.schedulerCopySelectPlaceholder", {
                            defaultValue: "Choose another deck"
                          })}
                        </option>
                        {otherDecks.map((deck) => (
                          <option key={deck.id} value={String(deck.id)}>
                            {deck.name}
                          </option>
                        ))}
                      </select>
                    </label>
                    <Button onClick={copyFromSelectedDeck} disabled={!copyDeckId}>
                      {t("option:flashcards.schedulerCopyAction", {
                        defaultValue: "Copy settings"
                      })}
                    </Button>
                    <Button onClick={resetToDefaults}>
                      {t("option:flashcards.schedulerResetAction", {
                        defaultValue: "Reset to defaults"
                      })}
                    </Button>
                  </div>
                </Space>
              </Card>

              <Card size="small" title={t("option:flashcards.studyDefaults", { defaultValue: "Study defaults" })}>
                <DeckStudyDefaultsFields
                  reviewPromptSide={reviewPromptSide}
                  onReviewPromptSideChange={setReviewPromptSide}
                />
              </Card>

              <DeckSchedulerSettingsEditor schedulerDraft={schedulerDraft} advancedDefaultOpen />

              <Collapse
                ghost
                items={[
                  {
                    key: "scheduler-explainer",
                    label: t("option:flashcards.schedulerExplainerTitle", {
                      defaultValue: "What's the difference between SM-2+ and FSRS?"
                    }),
                    children: (
                      <div className="space-y-2 text-xs text-text-muted">
                        <p>
                          {t("option:flashcards.schedulerSm2Description", {
                            defaultValue:
                              "SM-2+ is the classic algorithm used by Anki for decades. It adjusts review intervals based on a simple ease factor. Predictable and well-understood."
                          })}
                        </p>
                        <p>
                          {t("option:flashcards.schedulerFsrsDescription", {
                            defaultValue:
                              "FSRS is a newer algorithm that uses a memory model to predict when you'll forget each card. Often more efficient — fewer reviews for the same retention."
                          })}
                        </p>
                        <p>
                          {t("option:flashcards.schedulerRecommendation", {
                            defaultValue:
                              "If you're unsure, start with SM-2+ (the default). You can switch later without losing your review history."
                          })}
                        </p>
                      </div>
                    )
                  }
                ]}
              />

              <div className="flex flex-wrap items-center gap-3">
                <Button
                  type="primary"
                  onClick={() => void handleSave()}
                  loading={editorStatus === "saving" || updateDeckMutation.isPending}
                >
                  {t("option:flashcards.schedulerSaveAction", {
                    defaultValue: "Save changes"
                  })}
                </Button>
                {(draftChanged || reviewPromptSideChanged) && editorStatus !== "conflict" && (
                  <Text type="warning">
                    {t("option:flashcards.schedulerDirtyState", {
                      defaultValue: "Unsaved changes"
                    })}
                  </Text>
                )}
                {editorStatus === "saved" && !draftChanged && !reviewPromptSideChanged && (
                  <Text type="success">
                    {t("option:flashcards.schedulerSavedState", {
                      defaultValue: "All changes saved"
                    })}
                  </Text>
                )}
              </div>
            </Space>
          ) : (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={t("option:flashcards.selectDeckForScheduler", {
                defaultValue: "Select a deck to inspect its scheduler."
              })}
            />
          )}
        </Card>
      </div>
    </div>
  )
}

export default SchedulerTab
