import React from "react"
import { useTranslation } from "react-i18next"
import {
  BookOpenCheck,
  ExternalLink,
  LayoutTemplate,
  Layers,
  MessageSquareText,
  Save,
  Sparkles,
  Trash2,
  WandSparkles
} from "lucide-react"
import { Button, Input, Select, Typography } from "antd"
import { browser } from "wxt/browser"
import {
  useCreateFlashcardMutation,
  useDecksQuery,
  useFlashcardsEnabled,
  useGenerateFlashcardsMutation,
  useReviewFlashcardMutation,
  useReviewQuery
} from "@/components/Flashcards/hooks"
import { FlashcardTemplateValueModal } from "@/components/Flashcards/components/FlashcardTemplateValueModal"
import type { GeneratedCardDraft } from "@/components/Flashcards/tabs/ImportExport/shared"
import { normalizeGeneratedCards } from "@/components/Flashcards/tabs/ImportExport/shared"
import { normalizeFlashcardTemplateFields } from "@/components/Flashcards/utils/template-helpers"
import type { FlashcardCreate } from "@/services/flashcards"
import { buildFlashcardsGenerateRoute } from "@/services/tldw/flashcards-generate-handoff"

const { Text, Title } = Typography
const { TextArea } = Input
const SIDE_PANEL_GENERATE_CARD_COUNT = 3

type CaptureDraft = {
  id: string
  front: string
  back: string
  sourceId: string
  sourceTitle?: string
  tags?: string[]
  modelType?: "basic" | "basic_reverse" | "cloze"
  notes?: string | null
  extra?: string | null
}

type CapturedPageSelection = {
  selectedText: string
  sourceId: string
  sourceTitle?: string
  draftFront: string
}

export const readSelectedTextFromPage = () => {
  const activeElement = document.activeElement
  if (
    activeElement instanceof HTMLInputElement ||
    activeElement instanceof HTMLTextAreaElement
  ) {
    const { selectionStart, selectionEnd, value } = activeElement
    if (
      typeof selectionStart === "number" &&
      typeof selectionEnd === "number" &&
      selectionEnd > selectionStart
    ) {
      return value.slice(selectionStart, selectionEnd)
    }
  }

  return window.getSelection()?.toString() ?? ""
}

export default function SidepanelFlashcards() {
  const { t } = useTranslation()
  const draftIdRef = React.useRef(0)
  const [captureError, setCaptureError] = React.useState<string | null>(null)
  const [captureLoading, setCaptureLoading] = React.useState(false)
  const [generateHandoffLoading, setGenerateHandoffLoading] =
    React.useState(false)
  const [draftGenerateLoading, setDraftGenerateLoading] = React.useState(false)
  const [saveStatus, setSaveStatus] = React.useState<{
    type: "success" | "error"
    message: string
  } | null>(null)
  const generationInFlightRef = React.useRef(false)
  const [drafts, setDrafts] = React.useState<CaptureDraft[]>([])
  const [reviewOpen, setReviewOpen] = React.useState(false)
  const [reviewAnswerRevealed, setReviewAnswerRevealed] =
    React.useState(false)
  const [sidepanelReviewedCount, setSidepanelReviewedCount] = React.useState(0)
  const [submittedReviewUuid, setSubmittedReviewUuid] = React.useState<
    string | null
  >(null)
  const reviewAnswerStartRef = React.useRef<number | null>(null)
  const reviewSubmitInFlightRef = React.useRef(false)
  const [templateDraftId, setTemplateDraftId] = React.useState<string | null>(
    null
  )
  const [savingDraftIds, setSavingDraftIds] = React.useState<Set<string>>(
    () => new Set()
  )
  const savingDraftIdsRef = React.useRef<Set<string>>(new Set())
  const [selectedDeckId, setSelectedDeckId] = React.useState<number | null>(null)
  const flashcardsAvailability = useFlashcardsEnabled()
  const decksQuery = useDecksQuery()
  const createFlashcardMutation = useCreateFlashcardMutation()
  const generateFlashcardsMutation = useGenerateFlashcardsMutation()
  const reviewFlashcardMutation = useReviewFlashcardMutation()
  const decks = React.useMemo(() => decksQuery.data ?? [], [decksQuery.data])
  const selectedDeck = decks.find((deck) => deck.id === selectedDeckId) ?? null
  const flashcardsUnavailable =
    !flashcardsAvailability.isOnline ||
    flashcardsAvailability.flashcardsUnsupported
  const decksLoading = decksQuery.isLoading || flashcardsAvailability.capsLoading
  const decksLoadFailed = decksQuery.isError
  const hasSelectableDecks =
    !flashcardsUnavailable &&
    !decksLoading &&
    !decksLoadFailed &&
    decks.length > 0
  const showNoDecksMessage =
    !flashcardsUnavailable &&
    !decksLoading &&
    !decksLoadFailed &&
    decks.length === 0
  const hasDeck = selectedDeckId != null && hasSelectableDecks
  const reviewQuery = useReviewQuery(selectedDeckId, {
    enabled: reviewOpen && hasDeck,
    refetchOnWindowFocus: false
  })
  const reviewCard = reviewQuery.data ?? null
  const isReviewLoading = reviewQuery.isLoading
  const isReviewFetching = reviewQuery.isFetching
  const isReviewSubmitting = reviewFlashcardMutation.isPending
  const submitReviewFlashcard = reviewFlashcardMutation.mutateAsync
  const refetchReviewCard = reviewQuery.refetch
  const reviewCardUuid = reviewCard?.uuid ?? null
  const isReviewAdvancePending =
    submittedReviewUuid != null && reviewCardUuid === submittedReviewUuid
  const isReviewAdvanceFailed =
    submittedReviewUuid != null && reviewQuery.isError

  const sidepanelRatingOptions = React.useMemo(
    () => [
      {
        value: 0,
        label: t("option:flashcards.ratingAgain", { defaultValue: "Again" })
      },
      {
        value: 2,
        label: t("option:flashcards.ratingHard", { defaultValue: "Hard" })
      },
      {
        value: 3,
        label: t("option:flashcards.ratingGood", { defaultValue: "Good" })
      },
      {
        value: 5,
        label: t("option:flashcards.ratingEasy", { defaultValue: "Easy" })
      }
    ],
    [t]
  )

  React.useEffect(() => {
    if (
      selectedDeckId != null &&
      decks.some((deck) => deck.id === selectedDeckId)
    ) {
      return
    }
    setSelectedDeckId(decks[0]?.id ?? null)
  }, [decks, selectedDeckId])

  React.useEffect(() => {
    setReviewAnswerRevealed(false)
    reviewAnswerStartRef.current = null
  }, [reviewCard?.uuid, selectedDeckId])

  React.useEffect(() => {
    if (submittedReviewUuid == null) return
    if (reviewCardUuid != null && reviewCardUuid !== submittedReviewUuid) {
      setSubmittedReviewUuid(null)
      return
    }
    if (!reviewCardUuid && !isReviewFetching && !reviewQuery.isError) {
      setSubmittedReviewUuid(null)
    }
  }, [
    isReviewFetching,
    reviewCardUuid,
    reviewQuery.isError,
    submittedReviewUuid
  ])

  const openOptionsHashRoute = React.useCallback(async (route: string) => {
    setCaptureError(null)
    const normalizedRoute = route.startsWith("/") ? route : `/${route}`
    const url = browser.runtime.getURL(`/options.html#${normalizedRoute}`)
    const openFailedMessage = t(
      "sidepanel:flashcards.openFailed",
      "Could not open Flashcards. Check popup permissions and try again."
    )
    const openFallbackWindow = () => {
      const openedWindow = window.open(url, "_blank")
      if (openedWindow) return true
      setCaptureError(openFailedMessage)
      return false
    }

    if (browser.tabs?.create) {
      try {
        await browser.tabs.create({ url })
        return true
      } catch {
        return openFallbackWindow()
      }
    }
    return openFallbackWindow()
  }, [t])

  const openFlashcards = React.useCallback(() => {
    void openOptionsHashRoute("/flashcards")
  }, [openOptionsHashRoute])

  const captureMessages = React.useMemo(() => ({
    unavailable: t(
      "sidepanel:flashcards.selectionCaptureUnavailable",
      "Page selection capture is unavailable in this browser."
    ),
    activeTabUnavailable: t(
      "sidepanel:flashcards.activeTabUnavailable",
      "Open a page tab before capturing page selection."
    ),
    noSelection: t(
      "sidepanel:flashcards.noPageSelection",
      "Select text on the page first."
    ),
    failed: t(
      "sidepanel:flashcards.selectionCaptureFailed",
      "Could not read the current page selection. Use the Save to Notes context menu or paste text into Create & Import."
    )
  }), [t])

  const formatCaptureErrorMessage = React.useCallback(
    (error: unknown) => {
      const message =
        error instanceof Error && error.message.trim() ? error.message : ""
      const validationMessages = [
        captureMessages.unavailable,
        captureMessages.activeTabUnavailable,
        captureMessages.noSelection
      ]
      return validationMessages.includes(message) ? message : captureMessages.failed
    },
    [captureMessages]
  )

  const readActivePageSelection = React.useCallback(async (): Promise<CapturedPageSelection> => {
    if (!browser.tabs?.query || !browser.scripting?.executeScript) {
      throw new Error(captureMessages.unavailable)
    }

    const activeTabs = await browser.tabs.query({
      active: true,
      currentWindow: true
    })
    const activeTab = activeTabs[0]
    const tabId = activeTab?.id
    if (typeof tabId !== "number") {
      throw new Error(captureMessages.activeTabUnavailable)
    }

    const results = await browser.scripting.executeScript({
      target: { tabId },
      func: readSelectedTextFromPage
    })
    const selectedText = String(results?.[0]?.result ?? "").trim()
    if (!selectedText) {
      throw new Error(captureMessages.noSelection)
    }

    const sourceId = activeTab.url || String(tabId)
    const sourceTitle = activeTab.title || activeTab.url || undefined
    return {
      selectedText,
      sourceId,
      sourceTitle,
      draftFront:
        activeTab.title ||
        activeTab.url ||
        t("sidepanel:flashcards.defaultFront", "Page selection")
    }
  }, [captureMessages, t])

  const handleCapturePageSelection = React.useCallback(async () => {
    setCaptureError(null)
    setSaveStatus(null)
    setCaptureLoading(true)
    try {
      const captured = await readActivePageSelection()

      draftIdRef.current += 1
      const nextDraft: CaptureDraft = {
        id: `capture-${draftIdRef.current}`,
        front: captured.draftFront,
        back: captured.selectedText,
        sourceId: captured.sourceId,
        sourceTitle: captured.sourceTitle
      }
      setDrafts((current) => [...current, nextDraft])
    } catch (error) {
      setCaptureError(formatCaptureErrorMessage(error))
    } finally {
      setCaptureLoading(false)
    }
  }, [formatCaptureErrorMessage, readActivePageSelection])

  const handleGenerateFromSelection = React.useCallback(async () => {
    if (generationInFlightRef.current) return
    generationInFlightRef.current = true
    setCaptureError(null)
    setSaveStatus(null)
    setGenerateHandoffLoading(true)
    try {
      const captured = await readActivePageSelection()
      await openOptionsHashRoute(
        buildFlashcardsGenerateRoute({
          text: captured.selectedText,
          sourceType: "manual",
          sourceId: captured.sourceId,
          sourceTitle: captured.sourceTitle
        })
      )
    } catch (error) {
      setCaptureError(formatCaptureErrorMessage(error))
    } finally {
      generationInFlightRef.current = false
      setGenerateHandoffLoading(false)
    }
  }, [formatCaptureErrorMessage, openOptionsHashRoute, readActivePageSelection])

  const buildGeneratedDrafts = React.useCallback(
    (
      cards: GeneratedCardDraft[],
      captured: CapturedPageSelection
    ): CaptureDraft[] =>
      cards.map((card) => {
        draftIdRef.current += 1
        return {
          id: `generated-${draftIdRef.current}`,
          front: card.front,
          back: card.back,
          sourceId: captured.sourceId,
          sourceTitle: captured.sourceTitle,
          tags: card.tags,
          modelType: card.model_type,
          notes: card.notes,
          extra: card.extra
        }
      }),
    []
  )

  const formatGenerateDraftErrorMessage = React.useCallback(
    (error: unknown) => {
      const message =
        error instanceof Error && error.message.trim() ? error.message : ""
      const validationMessages = [
        captureMessages.unavailable,
        captureMessages.activeTabUnavailable,
        captureMessages.noSelection
      ]
      if (validationMessages.includes(message)) return message
      return t("sidepanel:flashcards.generateDraftsFailed", {
        defaultValue: "Could not generate draft cards. {{message}}",
        message:
          message ||
          t(
            "sidepanel:flashcards.generateDraftsFailedFallback",
            "Check provider settings and try again."
          )
      })
    },
    [captureMessages, t]
  )

  const formatGeneratedDraftsSuccess = React.useCallback(
    (count: number) =>
      count === 1
        ? t("sidepanel:flashcards.generateDraftsSuccess_one", {
            defaultValue: "Generated 1 draft card.",
            count
          })
        : t("sidepanel:flashcards.generateDraftsSuccess_other", {
            defaultValue: "Generated {{count}} draft cards.",
            count
          }),
    [t]
  )

  const formatDraftQueueCount = React.useCallback(
    (count: number) =>
      count === 1
        ? t("sidepanel:flashcards.draftQueueCount_one", {
            defaultValue: "1 draft card ready",
            count
          })
        : t("sidepanel:flashcards.draftQueueCount_other", {
            defaultValue: "{{count}} draft cards ready",
            count
          }),
    [t]
  )

  const formatSaveAllSuccess = React.useCallback(
    (savedCount: number, deckName: string) =>
      savedCount === 1
        ? t("sidepanel:flashcards.saveAllSuccess_one", {
            defaultValue: "Saved 1 card to {{deckName}}.",
            savedCount,
            deckName
          })
        : t("sidepanel:flashcards.saveAllSuccess_other", {
            defaultValue: "Saved {{savedCount}} cards to {{deckName}}.",
            savedCount,
            deckName
          }),
    [t]
  )

  const formatSaveAllPartial = React.useCallback(
    (savedCount: number, failedCount: number, deckName: string) => {
      if (savedCount === 1 && failedCount === 1) {
        return t("sidepanel:flashcards.saveAllPartial_one_one", {
          defaultValue:
            "Saved 1 card to {{deckName}}. 1 draft still needs attention.",
          savedCount,
          failedCount,
          deckName
        })
      }
      if (savedCount === 1) {
        return t("sidepanel:flashcards.saveAllPartial_one_other", {
          defaultValue:
            "Saved 1 card to {{deckName}}. {{failedCount}} drafts still need attention.",
          savedCount,
          failedCount,
          deckName
        })
      }
      if (failedCount === 1) {
        return t("sidepanel:flashcards.saveAllPartial_other_one", {
          defaultValue:
            "Saved {{savedCount}} cards to {{deckName}}. 1 draft still needs attention.",
          savedCount,
          failedCount,
          deckName
        })
      }
      return t("sidepanel:flashcards.saveAllPartial_other_other", {
        defaultValue:
          "Saved {{savedCount}} cards to {{deckName}}. {{failedCount}} drafts still need attention.",
        savedCount,
        failedCount,
        deckName
      })
    },
    [t]
  )

  const formatSidepanelReviewProgress = React.useCallback(
    (reviewedCount: number) =>
      reviewedCount === 1
        ? t("sidepanel:flashcards.sidepanelReviewProgress_one", {
            defaultValue: "Reviewed 1 card in this sidepanel session.",
            reviewedCount
          })
        : t("sidepanel:flashcards.sidepanelReviewProgress_other", {
            defaultValue:
              "Reviewed {{reviewedCount}} cards in this sidepanel session.",
            reviewedCount
          }),
    [t]
  )

  const handleGenerateDraftCards = React.useCallback(async () => {
    if (generationInFlightRef.current) return
    generationInFlightRef.current = true
    setCaptureError(null)
    setSaveStatus(null)
    setDraftGenerateLoading(true)
    try {
      const captured = await readActivePageSelection()
      const result = await generateFlashcardsMutation.mutateAsync({
        text: captured.selectedText,
        numCards: SIDE_PANEL_GENERATE_CARD_COUNT,
        cardType: "basic",
        difficulty: "mixed"
      })
      const generatedDrafts = buildGeneratedDrafts(
        normalizeGeneratedCards(result?.flashcards),
        captured
      )
      if (generatedDrafts.length === 0) {
        setCaptureError(
          t(
            "sidepanel:flashcards.generateDraftsEmpty",
            "No draft cards were generated. Try selecting a longer passage."
          )
        )
        return
      }
      setDrafts((current) => [...current, ...generatedDrafts])
      setSaveStatus({
        type: "success",
        message: formatGeneratedDraftsSuccess(generatedDrafts.length)
      })
    } catch (error) {
      setCaptureError(formatGenerateDraftErrorMessage(error))
    } finally {
      generationInFlightRef.current = false
      setDraftGenerateLoading(false)
    }
  }, [
    buildGeneratedDrafts,
    formatGenerateDraftErrorMessage,
    formatGeneratedDraftsSuccess,
    generateFlashcardsMutation,
    readActivePageSelection,
    t
  ])

  const getDeckName = React.useCallback(
    () => selectedDeck?.name || t("sidepanel:flashcards.defaultDeck", "deck"),
    [selectedDeck?.name, t]
  )

  const setActiveSavingDraftIds = React.useCallback(
    (draftIds: Iterable<string>) => {
      const nextDraftIds = new Set(draftIds)
      savingDraftIdsRef.current = nextDraftIds
      setSavingDraftIds(nextDraftIds)
    },
    []
  )

  const buildSavePayload = React.useCallback(
    (draft: CaptureDraft): FlashcardCreate | null => {
      if (selectedDeckId == null) return null

      const front = draft.front.trim()
      const back = draft.back.trim()
      if (!front || !back) return null

      return {
        ...(draft.tags?.length ? { tags: draft.tags } : {}),
        ...(draft.notes ? { notes: draft.notes } : {}),
        ...(draft.extra ? { extra: draft.extra } : {}),
        deck_id: selectedDeckId,
        front,
        back,
        model_type: draft.modelType ?? "basic",
        is_cloze: draft.modelType === "cloze",
        reverse: draft.modelType === "basic_reverse",
        source_ref_type: "manual",
        source_ref_id: draft.sourceId
      }
    },
    [selectedDeckId]
  )

  const handleUpdateDraft = React.useCallback(
    (draftId: string, field: "front" | "back", value: string) => {
      setDrafts((current) =>
        current.map((draft) =>
          draft.id === draftId ? { ...draft, [field]: value } : draft
        )
      )
    },
    []
  )

  const handleRemoveDraft = React.useCallback((draftId: string) => {
    setSaveStatus(null)
    setDrafts((current) => current.filter((draft) => draft.id !== draftId))
  }, [])

  const handleOpenTemplateDraft = React.useCallback((draftId: string) => {
    setCaptureError(null)
    setSaveStatus(null)
    setTemplateDraftId(draftId)
  }, [])

  const handleToggleReview = React.useCallback(() => {
    setCaptureError(null)
    setSaveStatus(null)
    setReviewAnswerRevealed(false)
    setSubmittedReviewUuid(null)
    reviewAnswerStartRef.current = null
    if (!reviewOpen) {
      setSidepanelReviewedCount(0)
    }
    setReviewOpen((current) => !current)
  }, [reviewOpen])

  const handleRevealReviewAnswer = React.useCallback(() => {
    setCaptureError(null)
    setSaveStatus(null)
    setReviewAnswerRevealed(true)
    reviewAnswerStartRef.current = Date.now()
  }, [])

  const handleSubmitSidepanelReview = React.useCallback(
    async (rating: number) => {
      if (
        !reviewCardUuid ||
        isReviewSubmitting ||
        submittedReviewUuid === reviewCardUuid ||
        reviewSubmitInFlightRef.current
      ) {
        return
      }
      reviewSubmitInFlightRef.current = true

      const answerTimeMs =
        reviewAnswerStartRef.current == null
          ? undefined
          : Math.max(0, Date.now() - reviewAnswerStartRef.current)

      try {
        await submitReviewFlashcard({
          cardUuid: reviewCardUuid,
          rating,
          answerTimeMs
        })
      } catch (error) {
        console.error("Failed to submit sidepanel flashcard review:", {
          cardUuid: reviewCardUuid,
          rating,
          error
        })
        setSaveStatus({
          type: "error",
          message: t(
            "sidepanel:flashcards.sidepanelReviewSubmitFailed",
            "Could not submit review. Check your connection and try again."
          )
        })
        return
      } finally {
        reviewSubmitInFlightRef.current = false
      }

      const nextReviewedCount = sidepanelReviewedCount + 1
      setSidepanelReviewedCount(nextReviewedCount)
      setSubmittedReviewUuid(reviewCardUuid)
      setSaveStatus({
        type: "success",
        message: formatSidepanelReviewProgress(nextReviewedCount)
      })
      setReviewAnswerRevealed(false)
      reviewAnswerStartRef.current = null
    },
    [
      formatSidepanelReviewProgress,
      isReviewSubmitting,
      reviewCardUuid,
      sidepanelReviewedCount,
      submitReviewFlashcard,
      submittedReviewUuid,
      t
    ]
  )

  const handleRetryLoadReviewCard = React.useCallback(async () => {
    setCaptureError(null)
    setSaveStatus(null)
    try {
      const result = await refetchReviewCard()
      if (result.error) {
        throw result.error
      }
      const nextCard = result.data ?? null
      if (!nextCard || nextCard.uuid !== submittedReviewUuid) {
        setSubmittedReviewUuid(null)
      }
    } catch (error) {
      console.error("Failed to retry sidepanel flashcard review load:", {
        cardUuid: submittedReviewUuid,
        error
      })
      setSaveStatus({
        type: "error",
        message: t(
          "sidepanel:flashcards.sidepanelReviewAdvanceFailed",
          "Review saved, but could not load the next card. Try again."
        )
      })
    }
  }, [refetchReviewCard, submittedReviewUuid, t])

  const handleCloseTemplateDraft = React.useCallback(() => {
    setTemplateDraftId(null)
  }, [])

  const handleApplyTemplateDraft = React.useCallback(
    (
      templateDraft: Pick<
        FlashcardCreate,
        "deck_id" | "tags" | "model_type" | "front" | "back" | "notes" | "extra"
      >
    ) => {
      setSaveStatus(null)
      setDrafts((current) =>
        current.map((draft) => {
          if (draft.id !== templateDraftId) return draft

          const normalized = normalizeFlashcardTemplateFields(templateDraft)
          return {
            ...draft,
            front: normalized.front ?? draft.front,
            back: normalized.back ?? draft.back,
            modelType: normalized.model_type,
            tags: normalized.tags ?? draft.tags,
            notes: normalized.notes ?? draft.notes ?? null,
            extra: normalized.extra ?? draft.extra ?? null
          }
        })
      )
      setTemplateDraftId(null)
    },
    [templateDraftId]
  )

  const handleSaveDraft = React.useCallback(
    async (draftId: string) => {
      if (!hasDeck || savingDraftIdsRef.current.size > 0) return

      const draft = drafts.find((candidate) => candidate.id === draftId)
      if (!draft) return

      const payload = buildSavePayload(draft)
      if (!payload) return

      setSaveStatus(null)
      setActiveSavingDraftIds([draftId])
      try {
        await createFlashcardMutation.mutateAsync(payload)
        setSaveStatus({
          type: "success",
          message: t("sidepanel:flashcards.saveSuccess", {
            defaultValue: "Saved to {{deckName}}.",
            deckName: getDeckName()
          })
        })
        setDrafts((current) =>
          current.filter((candidate) => candidate.id !== draftId)
        )
      } catch {
        setSaveStatus({
          type: "error",
          message: t(
            "sidepanel:flashcards.saveFailed",
            "Could not save flashcard. Check your connection and try again."
          )
        })
      } finally {
        setActiveSavingDraftIds([])
      }
    },
    [
      buildSavePayload,
      createFlashcardMutation,
      drafts,
      getDeckName,
      hasDeck,
      setActiveSavingDraftIds,
      t
    ]
  )

  const handleSaveAllDrafts = React.useCallback(async () => {
    if (!hasDeck || savingDraftIdsRef.current.size > 0) return

    const savedDraftIds = new Set<string>()

    try {
      const validDrafts = drafts.filter((draft) => buildSavePayload(draft))
      if (!validDrafts.length) return

      setSaveStatus(null)
      setActiveSavingDraftIds(validDrafts.map((draft) => draft.id))

      for (const draft of validDrafts) {
        const payload = buildSavePayload(draft)
        if (!payload) continue

        try {
          await createFlashcardMutation.mutateAsync(payload)
          savedDraftIds.add(draft.id)
        } catch {
          // Keep saving the queue so users get a precise partial-success state.
        }
      }

      if (savedDraftIds.size > 0) {
        setDrafts((current) =>
          current.filter((draft) => !savedDraftIds.has(draft.id))
        )
      }

      const failedCount = validDrafts.length - savedDraftIds.size
      const deckName = getDeckName()
      if (failedCount === 0) {
        setSaveStatus({
          type: "success",
          message: formatSaveAllSuccess(savedDraftIds.size, deckName)
        })
      } else if (savedDraftIds.size > 0) {
        setSaveStatus({
          type: "error",
          message: formatSaveAllPartial(savedDraftIds.size, failedCount, deckName)
        })
      } else {
        setSaveStatus({
          type: "error",
          message: t(
            "sidepanel:flashcards.saveAllFailed",
            "Could not save any flashcards. Check your connection and try again."
          )
        })
      }
    } catch {
      setSaveStatus({
        type: "error",
        message: t(
          "sidepanel:flashcards.saveAllUnexpectedFailed",
          "Could not finish saving flashcards. Check your connection and try again."
        )
      })
    } finally {
      setActiveSavingDraftIds([])
    }
  }, [
    buildSavePayload,
    createFlashcardMutation,
    drafts,
    formatSaveAllPartial,
    formatSaveAllSuccess,
    getDeckName,
    hasDeck,
    setActiveSavingDraftIds,
    t
  ])

  const isSaving = createFlashcardMutation.isPending || savingDraftIds.size > 0
  const isGenerating = generateHandoffLoading || draftGenerateLoading
  const templateTargetDraft =
    drafts.find((draft) => draft.id === templateDraftId) ?? null
  const canSaveDraft = React.useCallback(
    (draft: CaptureDraft) => !!buildSavePayload(draft) && hasDeck && !isSaving,
    [buildSavePayload, hasDeck, isSaving]
  )
  const validDraftCount = drafts.filter(
    (draft) => buildSavePayload(draft)
  ).length
  const canSaveAllDrafts = validDraftCount > 0 && hasDeck && !isSaving

  return (
    <main className="flex min-h-full flex-col gap-4 p-4 text-left">
      <div className="flex items-start gap-3">
        <div className="rounded-full border border-border bg-surface2 p-3">
          <Layers className="size-6 text-text-muted" aria-hidden="true" />
        </div>
        <div className="min-w-0">
          <Title level={4} className="!mb-1">
            {t("sidepanel:flashcards.title", "Flashcards")}
          </Title>
          <Text type="secondary">
            {t(
              "sidepanel:flashcards.workspaceDescription",
              "Capture selected page text into a deck, or open the full Flashcards workspace."
            )}
          </Text>
        </div>
      </div>
      <div className="flex w-full flex-col gap-2">
        <Button
          type="primary"
          block
          icon={<ExternalLink className="size-4" aria-hidden="true" />}
          onClick={openFlashcards}
        >
          {t("sidepanel:flashcards.openFull", "Open full Flashcards")}
        </Button>
        <Button
          block
          disabled={isSaving || isGenerating}
          icon={<BookOpenCheck className="size-4" aria-hidden="true" />}
          onClick={handleToggleReview}
        >
          {reviewOpen
            ? t("sidepanel:flashcards.closeSidepanelReview", "Close review")
            : t("sidepanel:flashcards.reviewDueCard", "Review due card")}
        </Button>
        <Button
          block
          loading={captureLoading}
          disabled={isSaving || isGenerating}
          icon={<MessageSquareText className="size-4" aria-hidden="true" />}
          onClick={handleCapturePageSelection}
        >
          {t(
            "sidepanel:flashcards.capturePageSelection",
            "Capture page selection"
          )}
        </Button>
        <Button
          block
          loading={draftGenerateLoading}
          disabled={isSaving || captureLoading || generateHandoffLoading}
          icon={<WandSparkles className="size-4" aria-hidden="true" />}
          onClick={() => void handleGenerateDraftCards()}
        >
          {t(
            "sidepanel:flashcards.generateDraftCards",
            "Generate draft cards"
          )}
        </Button>
        <Button
          block
          loading={generateHandoffLoading}
          disabled={isSaving || captureLoading || draftGenerateLoading}
          icon={<Sparkles className="size-4" aria-hidden="true" />}
          onClick={() => void handleGenerateFromSelection()}
        >
          {t(
            "sidepanel:flashcards.generateFromSelection",
            "Generate from selection"
          )}
        </Button>
      </div>
      <Text type="secondary" className="text-xs">
        {t(
          "sidepanel:flashcards.selectionHint",
          "Create one editable card, generate a small draft batch, apply templates to queued drafts, review due cards here, or use full Flashcards for imports and management."
        )}
      </Text>
      {reviewOpen ? (
        <section
          className="flex flex-col gap-3 rounded-lg border border-border bg-surface p-3"
          aria-label={t(
            "sidepanel:flashcards.sidepanelReviewSection",
            "Sidepanel flashcard review"
          )}
        >
          <div>
            <Title level={5} className="!mb-1">
              {t("sidepanel:flashcards.reviewDueCard", "Review due card")}
            </Title>
            <Text type="secondary" className="text-xs">
              {t(
                "sidepanel:flashcards.sidepanelReviewDescription",
                "Review the next due card from the selected deck."
              )}
            </Text>
          </div>
          <label className="flex flex-col gap-1 text-sm font-medium">
            {t("sidepanel:flashcards.deckLabel", "Deck")}
            <Select
              data-testid="sidepanel-flashcards-review-deck-select"
              aria-label={t("sidepanel:flashcards.reviewDeckLabel", "Review deck")}
              loading={decksLoading}
              disabled={
                !hasSelectableDecks || isReviewSubmitting
              }
              placeholder={t(
                "sidepanel:flashcards.deckPlaceholder",
                "Choose a deck"
              )}
              value={selectedDeckId ?? undefined}
              options={decks.map((deck) => ({
                label: deck.name,
                value: deck.id
              }))}
              onChange={(value) => setSelectedDeckId(value)}
            />
          </label>
          {flashcardsUnavailable ? (
            <Text type="danger" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.unavailable",
                "Flashcards are unavailable. Check the server connection and try again."
              )}
            </Text>
          ) : decksLoadFailed ? (
            <Text type="danger" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.decksLoadFailed",
                "Could not load decks. Check your connection and try again."
              )}
            </Text>
          ) : decksLoading ? (
            <Text type="secondary" className="text-xs" role="status">
              {t("sidepanel:flashcards.decksLoading", "Loading decks...")}
            </Text>
          ) : showNoDecksMessage ? (
            <Text type="warning" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.noDecks",
                "Create a deck in full Flashcards before saving here."
              )}
            </Text>
          ) : isReviewAdvanceFailed ? (
            <div className="flex flex-col gap-2" role="status">
              <Text type="warning">
                {t(
                  "sidepanel:flashcards.sidepanelReviewAdvanceFailed",
                  "Review saved, but could not load the next card. Try again."
                )}
              </Text>
              <Button onClick={() => void handleRetryLoadReviewCard()}>
                {t(
                  "sidepanel:flashcards.retryLoadNextReviewCard",
                  "Retry loading next card"
                )}
              </Button>
            </div>
          ) : isReviewAdvancePending ? (
            <Text type="secondary" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.sidepanelReviewAdvanceLoading",
                "Review saved. Loading next card..."
              )}
            </Text>
          ) : reviewQuery.isError ? (
            <Text type="danger" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.sidepanelReviewLoadFailed",
                "Could not load the next review card. Check your connection and try again."
              )}
            </Text>
          ) : hasDeck && isReviewLoading ? (
            <Text type="secondary" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.sidepanelReviewLoading",
                "Loading review card..."
              )}
            </Text>
          ) : hasDeck && reviewCard ? (
            <div className="flex flex-col gap-3">
              <div className="flex flex-col gap-1">
                <Text type="secondary" className="text-xs">
                  {t("sidepanel:flashcards.frontLabel", "Front")}
                </Text>
                <div className="whitespace-pre-wrap rounded-md border border-border bg-surface2 p-3 text-sm">
                  {reviewCard.front}
                </div>
              </div>
              {reviewAnswerRevealed ? (
                <>
                  <div className="flex flex-col gap-1">
                    <Text type="secondary" className="text-xs">
                      {t("sidepanel:flashcards.backLabel", "Back")}
                    </Text>
                    <div className="whitespace-pre-wrap rounded-md border border-border bg-surface2 p-3 text-sm">
                      {reviewCard.back}
                    </div>
                  </div>
                  <div
                    className="grid grid-cols-2 gap-2"
                    role="group"
                    aria-label={t(
                      "sidepanel:flashcards.ratingOptions",
                      "Rating options"
                    )}
                  >
                    {sidepanelRatingOptions.map((option) => (
                      <Button
                        key={option.value}
                        loading={isReviewSubmitting}
                        disabled={isReviewSubmitting}
                        onClick={() =>
                          void handleSubmitSidepanelReview(option.value)
                        }
                      >
                        {option.label}
                      </Button>
                    ))}
                  </div>
                </>
              ) : (
                <Button
                  type="primary"
                  block
                  disabled={isReviewSubmitting}
                  onClick={handleRevealReviewAnswer}
                >
                  {t("sidepanel:flashcards.revealAnswer", "Reveal answer")}
                </Button>
              )}
            </div>
          ) : hasDeck ? (
            <div className="flex flex-col gap-1" role="status">
              <Text>
                {t(
                  "sidepanel:flashcards.sidepanelReviewCaughtUp",
                  "No due cards right now."
                )}
              </Text>
              <Text type="secondary" className="text-xs">
                {t(
                  "sidepanel:flashcards.sidepanelReviewFullWorkspaceHint",
                  "Use full Flashcards for imports, management, and richer review controls."
                )}
              </Text>
            </div>
          ) : null}
        </section>
      ) : null}
      {drafts.length > 0 ? (
        <section
          className="flex flex-col gap-3"
          aria-label={t("sidepanel:flashcards.draftQueue", "Draft queue")}
        >
          <div className="flex items-center justify-between gap-2">
            <Text strong>
              {formatDraftQueueCount(drafts.length)}
            </Text>
            <Button
              size="small"
              icon={<Save className="size-4" aria-hidden="true" />}
              loading={isSaving}
              disabled={!canSaveAllDrafts}
              onClick={() => void handleSaveAllDrafts()}
            >
              {t("sidepanel:flashcards.saveAllCards", "Save all cards")}
            </Button>
          </div>
          <label className="flex flex-col gap-1 text-sm font-medium">
            {t("sidepanel:flashcards.deckLabel", "Deck")}
            <Select
              data-testid="sidepanel-flashcards-deck-select"
              aria-label={t("sidepanel:flashcards.deckLabel", "Deck")}
              loading={decksLoading}
              disabled={!hasSelectableDecks || isSaving}
              placeholder={t(
                "sidepanel:flashcards.deckPlaceholder",
                "Choose a deck"
              )}
              value={selectedDeckId ?? undefined}
              options={decks.map((deck) => ({
                label: deck.name,
                value: deck.id
              }))}
              onChange={(value) => setSelectedDeckId(value)}
            />
          </label>
          {flashcardsUnavailable ? (
            <Text type="danger" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.unavailable",
                "Flashcards are unavailable. Check the server connection and try again."
              )}
            </Text>
          ) : decksLoadFailed ? (
            <Text type="danger" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.decksLoadFailed",
                "Could not load decks. Check your connection and try again."
              )}
            </Text>
          ) : decksLoading ? (
            <Text type="secondary" className="text-xs" role="status">
              {t("sidepanel:flashcards.decksLoading", "Loading decks...")}
            </Text>
          ) : showNoDecksMessage ? (
            <Text type="warning" className="text-xs" role="status">
              {t(
                "sidepanel:flashcards.noDecks",
                "Create a deck in full Flashcards before saving here."
              )}
            </Text>
          ) : null}
          {drafts.map((draft, index) => (
            <section
              key={draft.id}
              className="flex flex-col gap-3 rounded-lg border border-border bg-surface p-3"
              aria-label={t("sidepanel:flashcards.draftSection", {
                defaultValue: "Draft flashcard {{index}}",
                index: index + 1
              })}
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <Title level={5} className="!mb-1">
                    {t("sidepanel:flashcards.draftTitle", "Draft flashcard")}
                  </Title>
                  <Text type="secondary" className="text-xs">
                    {draft.sourceTitle || draft.sourceId}
                  </Text>
                </div>
                <div className="flex shrink-0 items-center gap-1">
                  <Button
                    size="small"
                    icon={<LayoutTemplate className="size-4" aria-hidden="true" />}
                    aria-label={t("sidepanel:flashcards.applyTemplateToDraft", {
                      defaultValue: "Apply template to draft {{index}}",
                      index: index + 1
                    })}
                    disabled={isSaving}
                    onClick={() => handleOpenTemplateDraft(draft.id)}
                  />
                  <Button
                    size="small"
                    icon={<Trash2 className="size-4" aria-hidden="true" />}
                    aria-label={t("sidepanel:flashcards.removeDraft", {
                      defaultValue: "Remove draft {{index}}",
                      index: index + 1
                    })}
                    disabled={isSaving}
                    onClick={() => handleRemoveDraft(draft.id)}
                  />
                </div>
              </div>
              <label className="flex flex-col gap-1 text-sm font-medium">
                {t("sidepanel:flashcards.frontLabel", "Front")}
                <TextArea
                  aria-label={t("sidepanel:flashcards.frontLabel", "Front")}
                  autoSize={{ minRows: 2, maxRows: 4 }}
                  disabled={isSaving}
                  value={draft.front}
                  onChange={(event) =>
                    handleUpdateDraft(draft.id, "front", event.target.value)
                  }
                />
              </label>
              <label className="flex flex-col gap-1 text-sm font-medium">
                {t("sidepanel:flashcards.backLabel", "Back")}
                <TextArea
                  aria-label={t("sidepanel:flashcards.backLabel", "Back")}
                  autoSize={{ minRows: 4, maxRows: 8 }}
                  disabled={isSaving}
                  value={draft.back}
                  onChange={(event) =>
                    handleUpdateDraft(draft.id, "back", event.target.value)
                  }
                />
              </label>
              <Button
                type="primary"
                block
                icon={<Save className="size-4" aria-hidden="true" />}
                loading={savingDraftIds.has(draft.id)}
                disabled={!canSaveDraft(draft)}
                onClick={() => void handleSaveDraft(draft.id)}
              >
                {t("sidepanel:flashcards.saveCard", "Save card")}
              </Button>
            </section>
          ))}
        </section>
      ) : null}
      {saveStatus ? (
        <Text
          type={saveStatus.type === "error" ? "danger" : "success"}
          className="text-xs"
          role="status"
        >
          {saveStatus.message}
        </Text>
      ) : null}
      {captureError ? (
        <Text type="danger" className="text-xs" role="status">
          {captureError}
        </Text>
      ) : null}
      {templateTargetDraft ? (
        <FlashcardTemplateValueModal
          open
          onClose={handleCloseTemplateDraft}
          onApply={handleApplyTemplateDraft}
          draftDefaults={{
            deck_id: selectedDeckId ?? undefined,
            front: templateTargetDraft.front,
            back: templateTargetDraft.back,
            tags: templateTargetDraft.tags,
            notes: templateTargetDraft.notes ?? null,
            extra: templateTargetDraft.extra ?? null
          }}
        />
      ) : null}
    </main>
  )
}
