import React from "react"
import {
  Alert,
  Button,
  Card,
  Empty,
  Input,
  Segmented,
  Select,
  Space,
  Switch,
  Tag,
  Tooltip,
  Typography
} from "antd"
import { X, Minus, Check, Star, Calendar, Undo2 } from "lucide-react"
import dayjs from "dayjs"
import relativeTime from "dayjs/plugin/relativeTime"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import type { DeckReviewPromptSide, Flashcard, FlashcardUpdate } from "@/services/flashcards"
import { getSetting, setSetting } from "@/services/settings/registry"
import { FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING } from "@/services/settings/ui-settings"
import { trackFlashcardsShortcutHintTelemetry } from "@/utils/flashcards-shortcut-hint-telemetry"
import { FLASHCARDS_HELP_LINKS } from "../constants"
import {
  useDecksQuery,
  useCramQueueQuery,
  useReviewQuery,
  useReviewFlashcardMutation,
  useEndFlashcardReviewSessionMutation,
  useFlashcardAssistantQuery,
  useFlashcardAssistantRespondMutation,
  useUpdateFlashcardMutation,
  useResetFlashcardSchedulingMutation,
  useDeleteFlashcardMutation,
  useFlashcardShortcuts,
  useDueCountsQuery,
  useDeckDueCountsQuery,
  useReviewAnalyticsSummaryQuery,
  useHasCardsQuery,
  useNextDueQuery
} from "../hooks"
import {
  MarkdownWithBoundary,
  ReviewProgress,
  ReviewAnalyticsSummary,
  FlashcardEditDrawer,
  FlashcardStudyAssistantPanel
} from "../components"
import { RecentStudySessions } from "../components/RecentStudySessions"
import { calculateIntervals } from "../utils/calculateIntervals"
import { formatCardType } from "../utils/model-type-labels"
import { FlashcardQueueStateBadge } from "../utils/queue-state-badges"
import { buildReviewUndoState } from "../utils/review-undo"
import { getFlashcardSourceMeta } from "../utils/source-reference"
import {
  formatFlashcardsUiErrorMessage,
  mapFlashcardsUiError,
  type FlashcardsUiErrorCode
} from "../utils/error-taxonomy"
import type { StudyAssistantRespondRequest } from "@/services/flashcards"
import { trackFlashcardsErrorRecoveryTelemetry } from "@/utils/flashcards-error-recovery-telemetry"
import { FeatureHint } from "@/components/Common/FeatureHint"
import { StudySuggestionsPanel } from "@/components/StudySuggestions/StudySuggestionsPanel"
import {
  parseStudySuggestionTargetId,
  type StudySuggestionActionRequest,
  type StudySuggestionActionResponse
} from "@/services/studySuggestions"
import { buildQuizAssessmentRouteFromFlashcards } from "@/services/tldw/quiz-flashcards-handoff"
import { useFlashcardsShortcutHintDensity } from "../hooks/useFlashcardsShortcutHintDensity"

dayjs.extend(relativeTime)

const { Text, Title } = Typography

interface ReviewTabProps {
  onNavigateToCreate: () => void
  onNavigateToImport: () => void
  reviewDeckId: number | null | undefined
  onReviewDeckChange: (deckId: number | null | undefined) => void
  reviewOverrideCard?: Flashcard | null
  onClearOverride?: () => void
  isActive: boolean
  forceShowWorkspaceItems?: boolean
}

interface ReviewFailureState {
  code: FlashcardsUiErrorCode
  message: string
  retry: {
    cardUuid: string
    rating: number
    answerTimeMs?: number
  }
  canReload: boolean
}

/**
 * Review tab for studying flashcards with spaced repetition.
 */
export const ReviewTab: React.FC<ReviewTabProps> = ({
  onNavigateToCreate,
  onNavigateToImport,
  reviewDeckId,
  onReviewDeckChange,
  reviewOverrideCard,
  onClearOverride,
  isActive,
  forceShowWorkspaceItems = false
}) => {
  const { t } = useTranslation(["option", "common"])
  const navigate = useNavigate()
  const message = useAntdMessage()
  const reportUiError = React.useCallback(
    (error: unknown, operation: string, fallback: string) => {
      const mapped = mapFlashcardsUiError(error, {
        operation,
        fallback
      })
      console.warn("[flashcards:error]", {
        code: mapped.code,
        status: mapped.status,
        operation,
        raw: mapped.rawMessage
      })
      void trackFlashcardsErrorRecoveryTelemetry({
        type: "flashcards_mutation_failed",
        surface: "review",
        operation,
        error_code: mapped.code,
        status: mapped.status ?? null,
        retriable: mapped.code !== "FLASHCARDS_VALIDATION"
      })
      message.error(formatFlashcardsUiErrorMessage(mapped))
      return mapped
    },
    [message]
  )

  // State
  const [showAnswer, setShowAnswer] = React.useState(false)
  const [reviewedCount, setReviewedCount] = React.useState(0)
  const [localOverrideCard, setLocalOverrideCard] = React.useState<Flashcard | null>(null)
  const [editDrawerOpen, setEditDrawerOpen] = React.useState(false)
  const [showRatingRationale, setShowRatingRationale] = React.useState(false)
  const [reviewMode, setReviewMode] = React.useState<"due" | "cram">("due")
  const [cramTag, setCramTag] = React.useState("")
  const [cramUpdatesSchedule, setCramUpdatesSchedule] = React.useState(false)
  const [cramQueueIndex, setCramQueueIndex] = React.useState(0)
  const [reviewFailure, setReviewFailure] = React.useState<ReviewFailureState | null>(
    null
  )
  const [isReloadingFailedCard, setIsReloadingFailedCard] = React.useState(false)
  const [isReviewOnboardingDismissed, setIsReviewOnboardingDismissed] = React.useState(false)
  const [activeReviewSessionId, setActiveReviewSessionId] = React.useState<number | null>(null)
  const [selectedStudySessionId, setSelectedStudySessionId] = React.useState<number | null>(null)
  const [shortcutHintDensity, setShortcutHintDensity] = useFlashcardsShortcutHintDensity()
  const [sessionReviewPromptSide, setSessionReviewPromptSide] = React.useState<DeckReviewPromptSide | null>(null)

  // Undo state - stores the last reviewed card for potential re-rating
  const [lastReviewedCard, setLastReviewedCard] = React.useState<Flashcard | null>(null)
  const [showUndoButton, setShowUndoButton] = React.useState(false)
  const [undoCountdown, setUndoCountdown] = React.useState(0)
  const undoTimeoutRef = React.useRef<number | null>(null)
  const undoIntervalRef = React.useRef<number | null>(null)
  const autoRevealAnswerRef = React.useRef(false)
  const activeReviewSessionIdRef = React.useRef<number | null>(null)
  const previousActiveCardUuidRef = React.useRef<string | null>(null)
  const previousReviewScopeKeyRef = React.useRef<string | null>(null)
  const completeReviewSessionRef = React.useRef<(
    reason: "manual" | "auto" | "scope_change",
    options?: {
      revealSnapshot?: boolean
      sessionId?: number | null
    }
  ) => Promise<void>>()
  const autoEndSessionAttemptedRef = React.useRef<number | null>(null)

  // Auto-track answer time - stores the timestamp when answer was revealed
  const answerStartTimeRef = React.useRef<number | null>(null)

  // Queries and mutations
  const directPathVisibilityOptions = React.useMemo(
    () => (forceShowWorkspaceItems && reviewDeckId != null ? { includeWorkspaceItems: true } : undefined),
    [forceShowWorkspaceItems, reviewDeckId]
  )
  const decksQuery = useDecksQuery()
  const directPathDecksQuery = useDecksQuery(
    forceShowWorkspaceItems && reviewDeckId != null
      ? { includeWorkspaceItems: true }
      : { enabled: false }
  )
  const reviewQuery = useReviewQuery(reviewDeckId, {
    enabled: reviewMode === "due",
    ...(directPathVisibilityOptions ?? {})
  })
  const cramTagFilter = cramTag.trim() || undefined
  const cramQueueQuery = useCramQueueQuery(reviewDeckId, cramTagFilter, {
    enabled: reviewMode === "cram",
    ...(directPathVisibilityOptions ?? {})
  })
  const reviewMutation = useReviewFlashcardMutation()
  const updateMutation = useUpdateFlashcardMutation()
  const resetSchedulingMutation = useResetFlashcardSchedulingMutation()
  const deleteMutation = useDeleteFlashcardMutation()
  const dueCountsQuery = useDueCountsQuery(reviewDeckId, directPathVisibilityOptions)
  const deckDueCountsQuery = useDeckDueCountsQuery()
  const analyticsSummaryQuery = useReviewAnalyticsSummaryQuery(reviewDeckId, directPathVisibilityOptions)
  const hasCardsQuery = useHasCardsQuery(directPathVisibilityOptions)
  const nextDueQuery = useNextDueQuery(reviewDeckId, directPathVisibilityOptions)
  const endReviewSessionMutation = useEndFlashcardReviewSessionMutation()
  const nextDueInfo = nextDueQuery.data
  const cramQueue = cramQueueQuery.data || []
  const cramQueueCard =
    reviewMode === "cram" && cramQueueIndex < cramQueue.length
      ? cramQueue[cramQueueIndex]
      : null
  const activeCard =
    localOverrideCard ??
    reviewOverrideCard ??
    (reviewMode === "cram" ? cramQueueCard : reviewQuery.data)
  const assistantQuery = useFlashcardAssistantQuery(activeCard?.uuid, {
    enabled: isActive && !!activeCard
  })
  const assistantRespondMutation = useFlashcardAssistantRespondMutation()
  const reviewProgressTotal =
    reviewMode === "cram" ? cramQueue.length : dueCountsQuery.data?.total ?? 0
  const isCramMode = reviewMode === "cram"
  const showTopBarCreateCta = !activeCard
  const reviewScopeKey = React.useMemo(
    () => [
      reviewMode,
      reviewDeckId != null ? `deck:${reviewDeckId}` : "global",
      cramTagFilter ? `tag:${cramTagFilter}` : "untagged"
    ].join(":"),
    [cramTagFilter, reviewDeckId, reviewMode]
  )
  const activeCardSource = React.useMemo(
    () => (activeCard ? getFlashcardSourceMeta(activeCard) : null),
    [activeCard]
  )
  React.useEffect(() => {
    if (!isActive || !activeCard || shortcutHintDensity === "hidden") return
    void trackFlashcardsShortcutHintTelemetry({
      type: "flashcards_shortcut_hints_exposed",
      surface: "review",
      density: shortcutHintDensity
    })
  }, [isActive, activeCard?.uuid, shortcutHintDensity])
  React.useEffect(() => {
    let active = true
    void getSetting(FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING).then((stored) => {
      if (!active) return
      setIsReviewOnboardingDismissed(stored)
    })
    return () => {
      active = false
    }
  }, [])
  const setReviewOnboardingDismissed = React.useCallback((dismissed: boolean) => {
    setIsReviewOnboardingDismissed(dismissed)
    void setSetting(FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING, dismissed)
  }, [])
  const showReviewOnboardingGuide = hasCardsQuery.data === false && !isReviewOnboardingDismissed
  const cycleShortcutHintDensity = React.useCallback(() => {
    void setShortcutHintDensity((prev) => {
      const next =
        prev === "expanded"
          ? "compact"
          : prev === "compact"
            ? "hidden"
            : "expanded"
      void trackFlashcardsShortcutHintTelemetry({
        type: "flashcards_shortcut_hint_density_changed",
        surface: "review",
        from_density: prev,
        to_density: next
      })
      if (next === "hidden" && prev !== "hidden") {
        void trackFlashcardsShortcutHintTelemetry({
          type: "flashcards_shortcut_hints_dismissed",
          surface: "review",
          from_density: prev
        })
      }
      return next
    })
  }, [setShortcutHintDensity])
  const shortcutHintToggleLabel =
    shortcutHintDensity === "expanded"
      ? t("option:flashcards.shortcutHintsCompact", {
          defaultValue: "Compact hints"
        })
      : shortcutHintDensity === "compact"
        ? t("option:flashcards.shortcutHintsHide", {
            defaultValue: "Hide hints"
          })
        : t("option:flashcards.shortcutHintsShow", {
            defaultValue: "Show hints"
          })

  const availableDecks = React.useMemo(() => {
    const items = [...(decksQuery.data || [])]
    const directPreviewDeck =
      directPathDecksQuery.data?.find((deck) => deck.id === reviewDeckId) ?? null
    if (directPreviewDeck && !items.some((deck) => deck.id === directPreviewDeck.id)) {
      items.unshift(directPreviewDeck)
    }
    return items
  }, [decksQuery.data, directPathDecksQuery.data, reviewDeckId])

  // Get deck name for progress display
  const currentDeckName = React.useMemo(() => {
    if (!reviewDeckId) return undefined
    return availableDecks.find((d) => d.id === reviewDeckId)?.name ?? `Deck ${reviewDeckId}`
  }, [availableDecks, reviewDeckId])
  const selectedDeck = React.useMemo(
    () => (reviewDeckId ? availableDecks.find((deck) => deck.id === reviewDeckId) ?? null : null),
    [availableDecks, reviewDeckId]
  )
  const isPromptSideLockedToFront = activeCard?.model_type === "cloze"
  const effectivePromptSide: DeckReviewPromptSide =
    isPromptSideLockedToFront
      ? "front"
      : sessionReviewPromptSide ?? selectedDeck?.review_prompt_side ?? "front"
  const promptLabel =
    effectivePromptSide === "back"
      ? t("option:flashcards.back", { defaultValue: "Back" })
      : t("option:flashcards.front", { defaultValue: "Front" })
  const promptContent =
    effectivePromptSide === "back" ? activeCard?.back ?? "" : activeCard?.front ?? ""
  const answerLabel =
    effectivePromptSide === "back"
      ? t("option:flashcards.front", { defaultValue: "Front" })
      : t("option:flashcards.back", { defaultValue: "Back" })
  const answerContent =
    effectivePromptSide === "back" ? activeCard?.front ?? "" : activeCard?.back ?? ""

  const activeSchedulerLabel = React.useMemo(() => {
    if (!activeCard?.scheduler_type) return null
    return activeCard.scheduler_type === "fsrs" ? "FSRS" : "SM-2+"
  }, [activeCard?.scheduler_type])

  // Calculate intervals for current card
  const intervals = React.useMemo(() => {
    if (!activeCard) return null
    return activeCard.next_intervals ?? calculateIntervals(activeCard)
  }, [activeCard])

  // Rating options for Anki-style review with colors, shortcuts, icons, and interval previews
  const ratingOptions = React.useMemo(
    () => [
      {
        value: 0,
        key: "1",
        label: t("option:flashcards.ratingAgain", { defaultValue: "Again" }),
        description: t("option:flashcards.ratingAgainHelp", {
          defaultValue: "I didn't remember this card."
        }),
        interval: intervals?.again ?? "< 1 min",
        // WCAG AA compliant: 5.6:1 contrast ratio
        bgClass: "bg-danger hover:bg-danger/90 border-danger text-white",
        icon: X
      },
      {
        value: 2,
        key: "2",
        label: t("option:flashcards.ratingHard", { defaultValue: "Hard" }),
        description: t("option:flashcards.ratingHardHelp", {
          defaultValue: "I barely remembered; it felt difficult."
        }),
        interval: intervals?.hard ?? "< 10 min",
        // WCAG AA compliant: 9.3:1 contrast ratio (dark text on light bg)
        bgClass: "bg-warn/10 hover:bg-warn/20 border-warn/30 !text-warn",
        icon: Minus
      },
      {
        value: 3,
        key: "3",
        label: t("option:flashcards.ratingGood", { defaultValue: "Good" }),
        description: t("option:flashcards.ratingGoodHelp", {
          defaultValue: "I remembered with a bit of effort."
        }),
        interval: intervals?.good ?? "1 day",
        // WCAG AA compliant: 4.8:1 contrast ratio
        bgClass: "bg-success hover:bg-success/90 border-success text-white",
        primary: true,
        icon: Check
      },
      {
        value: 5,
        key: "4",
        label: t("option:flashcards.ratingEasy", { defaultValue: "Easy" }),
        description: t("option:flashcards.ratingEasyHelp", {
          defaultValue: "I remembered easily; no problem."
        }),
        interval: intervals?.easy ?? "4 days",
        // WCAG AA compliant: 5.8:1 contrast ratio
        bgClass: "bg-primary hover:bg-primaryStrong border-primary text-white",
        icon: Star
      }
    ],
    [t, intervals]
  )

  const onSubmitReview = React.useCallback(
    async (
      rating: number,
      retryOptions?: {
        cardUuid?: string
        answerTimeMs?: number
      }
    ): Promise<boolean> => {
      let attemptedAnswerTimeMs: number | undefined = retryOptions?.answerTimeMs
      try {
        const card = activeCard
        if (!card) return false
        if (retryOptions?.cardUuid && retryOptions.cardUuid !== card.uuid) {
          setReviewFailure((prev) =>
            prev
              ? {
                  ...prev,
                  message: t("option:flashcards.retryCardChanged", {
                    defaultValue:
                      "The review queue changed before retry. Reload and submit again."
                  })
                }
              : prev
          )
          return false
        }

        // Auto-calculate answer time from when the answer was shown
        if (
          typeof attemptedAnswerTimeMs !== "number" &&
          answerStartTimeRef.current
        ) {
          attemptedAnswerTimeMs = Date.now() - answerStartTimeRef.current
        }

        // Store the card for potential undo before submitting
        const cardForUndo = { ...card }

        const advanceCramQueue = () => {
          if (reviewMode !== "cram") return
          const currentIndex = cramQueue.findIndex((queued) => queued.uuid === card.uuid)
          if (currentIndex >= 0) {
            setCramQueueIndex(Math.min(currentIndex + 1, cramQueue.length))
            return
          }
          setCramQueueIndex((idx) => Math.min(idx + 1, cramQueue.length))
        }

        if (reviewMode === "cram" && !cramUpdatesSchedule) {
          setShowAnswer(false)
          answerStartTimeRef.current = null
          setReviewedCount((c) => c + 1)
          setShowUndoButton(false)
          setLastReviewedCard(null)
          setUndoCountdown(0)
          if (undoTimeoutRef.current) {
            window.clearTimeout(undoTimeoutRef.current)
          }
          if (undoIntervalRef.current) {
            window.clearInterval(undoIntervalRef.current)
          }
          if (reviewOverrideCard) {
            onClearOverride?.()
          }
          if (localOverrideCard) {
            setLocalOverrideCard(null)
          }
          advanceCramQueue()
          message.success(
            t("option:flashcards.cramPracticeSaved", {
              defaultValue: "Practice saved. Scheduling unchanged."
            })
          )
          setReviewFailure(null)
          return true
        }

        const reviewResult = await reviewMutation.mutateAsync({
          cardUuid: card.uuid,
          rating,
          answerTimeMs: attemptedAnswerTimeMs
        })
        if (typeof reviewResult.review_session_id === "number") {
          setActiveReviewSessionId(reviewResult.review_session_id)
          setSelectedStudySessionId(null)
        }
        setReviewFailure(null)
        setShowAnswer(false)
        answerStartTimeRef.current = null
        if (reviewedCount === 0) {
          message.success(
            t("option:flashcards.milestoneFirstReview", {
              defaultValue: "First card reviewed! Review daily for best results."
            })
          )
        }
        setReviewedCount((c) => c + 1)
        if (reviewOverrideCard) {
          onClearOverride?.()
        }
        if (localOverrideCard) {
          setLocalOverrideCard(null)
        }
        advanceCramQueue()

        // Enable undo for 10 seconds with visible countdown
        setLastReviewedCard(cardForUndo)
        setShowUndoButton(true)
        setUndoCountdown(10)
        if (undoTimeoutRef.current) {
          window.clearTimeout(undoTimeoutRef.current)
        }
        if (undoIntervalRef.current) {
          window.clearInterval(undoIntervalRef.current)
        }
        // Countdown interval for visual feedback
        undoIntervalRef.current = window.setInterval(() => {
          setUndoCountdown((prev) => {
            if (prev <= 1) {
              if (undoIntervalRef.current) {
                window.clearInterval(undoIntervalRef.current)
              }
              return 0
            }
            return prev - 1
          })
        }, 1000)
        // Timeout to hide undo button
        undoTimeoutRef.current = window.setTimeout(() => {
          setShowUndoButton(false)
          setLastReviewedCard(null)
          setUndoCountdown(0)
          if (undoIntervalRef.current) {
            window.clearInterval(undoIntervalRef.current)
          }
        }, 10000) // 10 second undo window

        const dueLabel = reviewResult.due_at
          ? dayjs(reviewResult.due_at).fromNow()
          : t("option:flashcards.nextReviewUnknown", {
              defaultValue: "soon"
            })
        const intervalLabel =
          reviewResult.interval_days === 1
            ? t("option:flashcards.intervalOneDay", { defaultValue: "1 day" })
            : t("option:flashcards.intervalManyDays", {
                defaultValue: "{{count}} days",
                count: reviewResult.interval_days
              })

        message.success(
          t("option:flashcards.reviewSavedWithSchedule", {
            defaultValue: "Saved. Next review {{due}} (next review gap: {{interval}}).",
            due: dueLabel,
            interval: intervalLabel
          })
        )
        return true
      } catch (e: unknown) {
        const mapped = reportUiError(
          e,
          "submitting your review",
          t("option:flashcards.reviewSubmitFailed", {
            defaultValue: "Failed to submit review."
          })
        )
        if (activeCard) {
          setReviewFailure({
            code: mapped.code,
            message: formatFlashcardsUiErrorMessage(mapped),
            retry: {
              cardUuid: activeCard.uuid,
              rating,
              answerTimeMs: attemptedAnswerTimeMs
            },
            canReload:
              mapped.code === "FLASHCARDS_VERSION_CONFLICT" ||
              mapped.code === "FLASHCARDS_NOT_FOUND"
          })
        }
        return false
      }
    },
    [
      activeCard,
      cramQueue,
      cramUpdatesSchedule,
      localOverrideCard,
      message,
      onClearOverride,
      reportUiError,
      reviewMode,
      reviewMutation,
      reviewOverrideCard,
      t
    ]
  )

  const handleRetryFailedReview = React.useCallback(async () => {
    if (!reviewFailure || reviewMutation.isPending) return
    void trackFlashcardsErrorRecoveryTelemetry({
      type: "flashcards_retry_requested",
      surface: "review",
      operation: "submitting your review",
      error_code: reviewFailure.code
    })
    const succeeded = await onSubmitReview(reviewFailure.retry.rating, {
      cardUuid: reviewFailure.retry.cardUuid,
      answerTimeMs: reviewFailure.retry.answerTimeMs
    })
    if (succeeded) {
      void trackFlashcardsErrorRecoveryTelemetry({
        type: "flashcards_retry_succeeded",
        surface: "review",
        operation: "submitting your review",
        error_code: reviewFailure.code
      })
    }
  }, [onSubmitReview, reviewFailure, reviewMutation.isPending])

  const handleReloadFailedReviewCard = React.useCallback(async () => {
    if (!reviewFailure?.retry.cardUuid) return
    setIsReloadingFailedCard(true)
    try {
      await Promise.all([reviewQuery.refetch(), dueCountsQuery.refetch()])
      setReviewFailure((prev) =>
        prev
          ? {
              ...prev,
              canReload: false,
              message: t("option:flashcards.retryReloadReady", {
                defaultValue:
                  "Latest card data loaded. Retry the review to submit your rating."
              })
            }
          : prev
      )
      message.info(
        t("option:flashcards.cardReloaded", {
          defaultValue: "Card reloaded."
        })
      )
      void trackFlashcardsErrorRecoveryTelemetry({
        type: "flashcards_recovered_by_reload",
        surface: "review",
        operation: "submitting your review",
        error_code: reviewFailure.code
      })
    } catch (error: unknown) {
      reportUiError(
        error,
        "reloading the latest card state",
        t("option:flashcards.cardReloadFailed", {
          defaultValue: "Failed to reload card."
        })
      )
    } finally {
      setIsReloadingFailedCard(false)
    }
  }, [dueCountsQuery, message, reportUiError, reviewFailure, reviewQuery, t])

  const completeReviewSession = React.useCallback(
    async (
      reason: "manual" | "auto" | "scope_change",
      options?: {
        revealSnapshot?: boolean
        sessionId?: number | null
      }
    ) => {
      const sessionId = options?.sessionId ?? activeReviewSessionId
      if (sessionId == null || endReviewSessionMutation.isPending) {
        return
      }
      if (reason === "auto" && autoEndSessionAttemptedRef.current === sessionId) {
        return
      }
      if (reason === "auto") {
        autoEndSessionAttemptedRef.current = sessionId
      }

      try {
        const completedSession = await endReviewSessionMutation.mutateAsync(sessionId)
        setActiveReviewSessionId((current) => (current === sessionId ? null : current))
        if (options?.revealSnapshot ?? reason !== "scope_change") {
          setSelectedStudySessionId(completedSession.id)
        }
        autoEndSessionAttemptedRef.current = null
      } catch (error: unknown) {
        reportUiError(
          error,
          "ending the review session",
          t("option:flashcards.reviewSessionEndFailed", {
            defaultValue: "Failed to end session."
          })
        )
      }
    },
    [activeReviewSessionId, endReviewSessionMutation, reportUiError, t]
  )

  const handleStudySuggestionActionResult = React.useCallback(
    async (response: StudySuggestionActionResponse, request: StudySuggestionActionRequest) => {
      const targetId = parseStudySuggestionTargetId(response.target_id)

      if (response.target_service === "flashcards" && targetId) {
        setSelectedStudySessionId(null)
        onReviewDeckChange(targetId)
        return
      }

      if (response.target_service === "quiz" && request.actionKind === "follow_up_quiz" && targetId) {
        navigate(
          buildQuizAssessmentRouteFromFlashcards({
            startQuizId: targetId,
            highlightQuizId: targetId,
            deckId: reviewDeckId ?? undefined,
            deckName: currentDeckName,
            forceShowWorkspaceItems
          })
        )
      }
    },
    [currentDeckName, forceShowWorkspaceItems, navigate, onReviewDeckChange, reviewDeckId]
  )

  React.useEffect(() => {
    completeReviewSessionRef.current = completeReviewSession
  }, [completeReviewSession])

  // Handle undo - re-present the last reviewed card
  const handleUndoReview = React.useCallback(() => {
    const undoState = buildReviewUndoState(lastReviewedCard, reviewedCount)
    if (!undoState) return

    // Clear the undo state and countdown
    if (undoTimeoutRef.current) {
      window.clearTimeout(undoTimeoutRef.current)
    }
    if (undoIntervalRef.current) {
      window.clearInterval(undoIntervalRef.current)
    }
    setShowUndoButton(false)
    setUndoCountdown(0)
    setReviewedCount(undoState.nextReviewedCount)

    const shouldRevealOnCurrent = activeCard?.uuid === undoState.overrideCard.uuid
    if (shouldRevealOnCurrent) {
      autoRevealAnswerRef.current = false
      setShowAnswer(true)
      answerStartTimeRef.current = Date.now()
    } else {
      autoRevealAnswerRef.current = true
    }
    setLocalOverrideCard(undoState.overrideCard)

    message.info(
      t("option:flashcards.undoReviewHint", {
        defaultValue: "Rate this card again to update your response"
      })
    )
  }, [activeCard?.uuid, lastReviewedCard, reviewedCount, message, t])

  // Cleanup timeout and interval on unmount
  React.useEffect(() => {
    return () => {
      if (undoTimeoutRef.current) {
        window.clearTimeout(undoTimeoutRef.current)
      }
      if (undoIntervalRef.current) {
        window.clearInterval(undoIntervalRef.current)
      }
    }
  }, [])

  React.useEffect(() => {
    if (autoRevealAnswerRef.current) {
      autoRevealAnswerRef.current = false
      setShowAnswer(true)
      answerStartTimeRef.current = Date.now()
      return
    }
    setShowAnswer(false)
    setShowRatingRationale(false)
    answerStartTimeRef.current = null
  }, [activeCard?.uuid])

  React.useEffect(() => {
    const previousActiveCardUuid = previousActiveCardUuidRef.current
    const nextActiveCardUuid = activeCard?.uuid ?? null
    if (previousActiveCardUuid && !nextActiveCardUuid && activeReviewSessionId != null) {
      void completeReviewSession("auto")
    }
    previousActiveCardUuidRef.current = nextActiveCardUuid
  }, [activeCard?.uuid, activeReviewSessionId, completeReviewSession])

  // Track when the answer is shown (for auto-timing)
  const handleShowAnswer = React.useCallback(() => {
    setShowAnswer(true)
    answerStartTimeRef.current = Date.now()
  }, [])

  const handleOpenEdit = React.useCallback(() => {
    if (!activeCard) return
    setEditDrawerOpen(true)
  }, [activeCard])

  const handleSaveEdit = React.useCallback(
    async (values: FlashcardUpdate) => {
      if (!activeCard) return
      try {
        await updateMutation.mutateAsync({
          uuid: activeCard.uuid,
          update: values
        })
        message.success(
          t("option:flashcards.cardUpdated", {
            defaultValue: "Card updated."
          })
        )
        setEditDrawerOpen(false)
      } catch (error: unknown) {
        reportUiError(
          error,
          "saving card edits",
          t("option:flashcards.cardUpdateFailed", {
            defaultValue: "Failed to update card."
          })
        )
      }
    },
    [activeCard, updateMutation, message, reportUiError, t]
  )

  const handleDeleteFromReview = React.useCallback(async () => {
    if (!activeCard) return
    try {
      await deleteMutation.mutateAsync({
        uuid: activeCard.uuid,
        version: activeCard.version
      })
      message.success(
        t("option:flashcards.cardDeleted", {
          defaultValue: "Card deleted."
        })
      )
      setEditDrawerOpen(false)
      setShowAnswer(false)
      setShowRatingRationale(false)
      answerStartTimeRef.current = null
      if (reviewOverrideCard) {
        onClearOverride?.()
      }
      if (localOverrideCard?.uuid === activeCard.uuid) {
        setLocalOverrideCard(null)
      }
    } catch (error: unknown) {
      reportUiError(
        error,
        "deleting this card",
        t("option:flashcards.cardDeleteFailed", {
          defaultValue: "Failed to delete card."
        })
      )
    }
  }, [
    activeCard,
    deleteMutation,
    message,
    reportUiError,
    t,
    reviewOverrideCard,
    onClearOverride,
    localOverrideCard?.uuid
  ])

  const handleResetSchedulingFromReview = React.useCallback(async () => {
    if (!activeCard) return
    try {
      await resetSchedulingMutation.mutateAsync({
        uuid: activeCard.uuid,
        expectedVersion: activeCard.version
      })
      message.success(
        t("option:flashcards.schedulingResetSuccess", {
          defaultValue: "Scheduling reset to new-card defaults."
        })
      )
      setEditDrawerOpen(false)
    } catch (error: unknown) {
      reportUiError(
        error,
        "resetting scheduling",
        t("option:flashcards.schedulingResetFailed", {
          defaultValue: "Failed to reset scheduling."
        })
      )
    }
  }, [activeCard, resetSchedulingMutation, message, reportUiError, t])

  const handleAssistantRespond = React.useCallback(
    async (request: StudyAssistantRespondRequest) => {
      if (!activeCard) return
      await assistantRespondMutation.mutateAsync({
        cardUuid: activeCard.uuid,
        request
      })
    },
    [activeCard, assistantRespondMutation]
  )

  React.useEffect(() => {
    if (reviewMode !== "cram") return
    setCramQueueIndex((idx) => Math.min(idx, cramQueue.length))
  }, [reviewMode, cramQueue.length])

  React.useEffect(() => {
    activeReviewSessionIdRef.current = activeReviewSessionId
  }, [activeReviewSessionId])

  // Reset reviewed/session state when review scope changes
  React.useEffect(() => {
    const previousReviewScopeKey = previousReviewScopeKeyRef.current
    const scopeChanged =
      previousReviewScopeKey != null && previousReviewScopeKey !== reviewScopeKey
    const sessionIdToClose = scopeChanged ? activeReviewSessionIdRef.current : null

    setReviewedCount(0)
    setCramQueueIndex(0)
    setLocalOverrideCard(null)
    setShowUndoButton(false)
    setLastReviewedCard(null)
    setUndoCountdown(0)
    setSelectedStudySessionId(null)
    setSessionReviewPromptSide(null)
    autoEndSessionAttemptedRef.current = null
    autoRevealAnswerRef.current = false
    previousReviewScopeKeyRef.current = reviewScopeKey

    if (sessionIdToClose != null) {
      void (async () => {
        try {
          await completeReviewSessionRef.current?.("scope_change", {
            sessionId: sessionIdToClose,
            revealSnapshot: false
          })
        } finally {
          setActiveReviewSessionId((current) =>
            current === sessionIdToClose ? null : current
          )
        }
      })()
    }
  }, [reviewScopeKey])

  React.useEffect(() => {
    if (reviewOverrideCard) {
      setLocalOverrideCard(null)
    }
  }, [reviewOverrideCard])

  // Keyboard shortcuts for review
  useFlashcardShortcuts({
    enabled: isActive && !!activeCard,
    showingAnswer: showAnswer,
    onFlip: handleShowAnswer,
    onRate: onSubmitReview,
    onEdit: handleOpenEdit,
    onUndo: showUndoButton ? handleUndoReview : undefined
  })

  return (
    <div>
      <div
        className={`mb-3 flex flex-wrap items-center gap-2 ${
          activeCard ? "rounded border border-border/70 bg-surface2/60 p-2" : ""
        }`}
        data-testid="flashcards-review-topbar"
      >
        <Select
          placeholder={t("option:flashcards.selectDeck", {
            defaultValue: "Select deck (optional)"
          })}
          allowClear
          loading={decksQuery.isLoading || directPathDecksQuery.isLoading}
          value={reviewDeckId ?? undefined}
          className="min-w-64 max-w-full flex-1"
          onChange={(v) => {
            onReviewDeckChange(v)
            if (reviewOverrideCard) {
              onClearOverride?.()
            }
          }}
          data-testid="flashcards-review-deck-select"
          options={availableDecks.map((d) => ({
            label:
              ((deckDueCountsQuery.data?.[d.id]?.due ?? 0) > 0)
                ? t("option:flashcards.deckWithDueCount", {
                    defaultValue: "{{deckName}} ({{count}} due)",
                    deckName: d.name,
                    count: deckDueCountsQuery.data?.[d.id]?.due ?? 0
                  })
                : d.name,
            value: d.id
          }))}
        />
        <Segmented
          value={reviewMode}
          onChange={(value) => {
            setReviewMode(value as "due" | "cram")
          }}
          options={[
            {
              label: t("option:flashcards.reviewModeDueOnly", {
                defaultValue: "Due only"
              }),
              value: "due"
            },
            {
              label: t("option:flashcards.reviewModeCram", {
                defaultValue: "Cram"
              }),
              value: "cram"
            }
          ]}
          data-testid="flashcards-review-mode-toggle"
        />
        <Segmented
          value={effectivePromptSide}
          onChange={(value) => {
            setSessionReviewPromptSide(value as DeckReviewPromptSide)
          }}
          disabled={isPromptSideLockedToFront}
          options={[
            {
              label: t("option:flashcards.reviewPromptSideFront", {
                defaultValue: "Front first"
              }),
              value: "front"
            },
            {
              label: t("option:flashcards.reviewPromptSideBack", {
                defaultValue: "Back first"
              }),
              value: "back"
            }
          ]}
          data-testid="flashcards-review-prompt-side-toggle"
        />
        {reviewMode === "cram" && (
          <>
            <Input
              allowClear
              value={cramTag}
              onChange={(event) => setCramTag(event.target.value)}
              placeholder={t("option:flashcards.cramTagFilterPlaceholder", {
                defaultValue: "Filter cram by tag (optional)"
              })}
              className="min-w-56 max-w-full flex-1"
              data-testid="flashcards-review-cram-tag"
            />
            <Space size={6}>
              <Text type="secondary" className="text-xs">
                {t("option:flashcards.cramUpdateSchedule", {
                  defaultValue: "Update schedule"
                })}
              </Text>
              <Switch
                checked={cramUpdatesSchedule}
                onChange={setCramUpdatesSchedule}
                data-testid="flashcards-review-cram-update-schedule"
              />
            </Space>
          </>
        )}
        {showTopBarCreateCta && (
          <Button
            type="primary"
            className="min-h-11"
            onClick={onNavigateToCreate}
            data-testid="flashcards-review-create-cta"
          >
            {t("option:flashcards.noDueCreateCta", { defaultValue: "Create card" })}
          </Button>
        )}
      </div>

      <ReviewAnalyticsSummary
        summary={analyticsSummaryQuery.data}
        isLoading={analyticsSummaryQuery.isLoading}
        selectedDeckId={reviewDeckId ?? null}
      />

      {reviewProgressTotal > 0 && (
        <ReviewProgress
          dueCount={reviewProgressTotal}
          reviewedCount={reviewedCount}
          deckName={currentDeckName}
        />
      )}

      {activeCard ? (
        <Card data-testid="flashcards-review-active-card">
          <div className="flex flex-col gap-3">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <Tag>{formatCardType(activeCard, t)}</Tag>
                <FlashcardQueueStateBadge
                  card={activeCard}
                  testId="flashcards-review-queue-state"
                />
                {activeSchedulerLabel && (
                  <Tag data-testid="flashcards-review-scheduler-type">
                    {activeSchedulerLabel}
                  </Tag>
                )}
                {activeCard.tags?.map((tag) => (
                  <Tag key={tag}>{tag}</Tag>
                ))}
                {activeCardSource && (
                  <Tag
                    color={
                      activeCardSource.unavailable
                        ? "default"
                        : activeCardSource.type === "media"
                          ? "blue"
                          : activeCardSource.type === "note"
                            ? "gold"
                            : "green"
                    }
                  >
                    {activeCardSource.href ? (
                      <a href={activeCardSource.href}>{activeCardSource.label}</a>
                    ) : (
                      activeCardSource.label
                    )}
                  </Tag>
                )}
                <Tooltip
                  title={t("option:flashcards.shortcutEditTooltip", {
                    defaultValue: "Shortcut: E"
                  })}
                >
                  <Button
                    size="small"
                    onClick={handleOpenEdit}
                    data-testid="flashcards-review-edit-card"
                    aria-label={t("option:flashcards.shortcutEditAria", {
                      defaultValue: "Edit card (E)"
                    })}
                  >
                    {t("option:flashcards.editCardAction", {
                      defaultValue: "Edit"
                    })}
                  </Button>
                </Tooltip>
              </div>
            </div>

            <div>
              <Title level={5} className="!mb-2">
                {promptLabel}
              </Title>
              <div className="rounded border border-border bg-surface p-3 text-sm text-text">
                <MarkdownWithBoundary
                  content={promptContent}
                  size="sm"
                  className="prose-headings:!text-text prose-p:!text-text prose-li:!text-text prose-strong:!text-text"
                />
              </div>
            </div>

            {showAnswer && (
              <div>
                <Title level={5} className="!mb-2">
                  {answerLabel}
                </Title>
                <div className="rounded border border-border bg-surface p-3 text-sm text-text">
                  <MarkdownWithBoundary
                    content={answerContent}
                    size="sm"
                    className="prose-headings:!text-text prose-p:!text-text prose-li:!text-text prose-strong:!text-text"
                  />
                </div>
                {activeCard.extra && (
                  <div className="mt-2 rounded border border-border bg-surface p-3 text-sm text-text/80">
                    <MarkdownWithBoundary
                      content={activeCard.extra}
                      size="xs"
                      className="prose-headings:!text-text prose-p:!text-text prose-li:!text-text prose-strong:!text-text"
                    />
                  </div>
                )}
              </div>
            )}

            <div className="relative">
              <FlashcardStudyAssistantPanel
                cardUuid={activeCard.uuid}
                threadVersion={assistantQuery.data?.thread.version ?? null}
                messages={assistantQuery.data?.messages ?? []}
                availableActions={assistantQuery.data?.available_actions ?? null}
                assistantContext={assistantQuery.data}
                isLoading={assistantQuery.isLoading}
                isError={assistantQuery.isError}
                queryError={assistantQuery.error}
                isResponding={assistantRespondMutation.isPending}
                onReloadContext={() => assistantQuery.refetch()}
                onRespond={handleAssistantRespond}
              />
              <FeatureHint
                featureKey="flashcards_study_assistant_discovery"
                title={t("option:flashcards.studyAssistantHintTitle", {
                  defaultValue: "Study assistant"
                })}
                description={t("option:flashcards.studyAssistantHintDescription", {
                  defaultValue:
                    "Need help understanding a card? Ask the study assistant."
                })}
                position="top"
              />
            </div>

            {activeReviewSessionId != null && (
              <div className="flex justify-end">
                <Button
                  type="default"
                  loading={endReviewSessionMutation.isPending}
                  onClick={() => {
                    void completeReviewSession("manual")
                  }}
                  data-testid="flashcards-review-end-session"
                >
                  End Session
                </Button>
              </div>
            )}

            <div className="mt-2 flex flex-col gap-3">
              {!showAnswer ? (
                <div className="flex flex-col gap-2">
                  <Tooltip
                    title={t("option:flashcards.shortcutFlipTooltip", {
                      defaultValue: "Shortcut: Space"
                    })}
                  >
                    <Button
                      type="primary"
                      onClick={handleShowAnswer}
                      data-testid="flashcards-review-show-answer"
                      aria-label={t("option:flashcards.shortcutFlipAria", {
                        defaultValue: "Show answer (Space)"
                      })}
                    >
                      {t("option:flashcards.showAnswer", {
                        defaultValue: "Show Answer"
                      })}
                    </Button>
                  </Tooltip>
                  <div
                    className="flex flex-wrap items-center gap-2"
                    data-testid="flashcards-review-shortcut-chips-question"
                  >
                    {shortcutHintDensity === "expanded" && (
                      <>
                        <Tag className="!m-0">
                          {t("option:flashcards.shortcutChipFlip", {
                            defaultValue: "Space Flip"
                          })}
                        </Tag>
                        <Tag className="!m-0">
                          {t("option:flashcards.shortcutChipEdit", {
                            defaultValue: "E Edit"
                          })}
                        </Tag>
                      </>
                    )}
                    {shortcutHintDensity === "compact" && (
                      <Tag className="!m-0">
                        {t("option:flashcards.shortcutChipReviewCompactQuestion", {
                          defaultValue: "Space / E"
                        })}
                      </Tag>
                    )}
                    <Button
                      type="link"
                      size="small"
                      className="!h-auto !px-0 text-xs"
                      onClick={cycleShortcutHintDensity}
                      data-testid="flashcards-review-shortcut-hints-toggle"
                    >
                      {shortcutHintToggleLabel}
                    </Button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex flex-col gap-2">
                    <Text>
                      {t("option:flashcards.rate", {
                        defaultValue: "How well did you remember this card?"
                      })}
                    </Text>
                    <div className="flex flex-wrap gap-2 justify-center" role="group" aria-label={t("option:flashcards.ratingGroup", { defaultValue: "Rating options" })}>
                      {ratingOptions.map((opt) => {
                        const Icon = opt.icon
                        return (
                          <Tooltip
                            key={opt.value}
                            title={`${opt.description} (${opt.key})`}
                          >
                            <Button
                              onClick={() => onSubmitReview(opt.value)}
                              aria-label={`${opt.label}: ${opt.description} Press ${opt.key}`}
                              className={`${opt.bgClass} ${opt.primary ? "!px-6" : ""} min-h-11 focus:ring-2 focus:ring-offset-2 focus:ring-current focus:outline-none`}
                              data-testid={`flashcards-review-rate-${opt.key}`}
                            >
                              <div className="flex flex-col items-center gap-0.5">
                                {/* Larger icons (24px) for colorblind accessibility */}
                                <Icon className="size-6" aria-hidden="true" />
                                <span className="font-medium">
                                  {opt.label}
                                  {/* Keyboard shortcut badge for additional visual differentiation */}
                                  <span className="ml-1.5 inline-flex items-center justify-center w-5 h-5 rounded text-xs font-bold bg-black/20">
                                    {opt.key}
                                  </span>
                                </span>
                                <span className="text-xs opacity-80">
                                  {opt.interval}
                                </span>
                              </div>
                            </Button>
                          </Tooltip>
                        )
                      })}
                    </div>
                    {reviewFailure && (
                      <Alert
                        showIcon
                        type={
                          reviewFailure.code === "FLASHCARDS_VERSION_CONFLICT"
                            ? "warning"
                            : "error"
                        }
                        title={t("option:flashcards.reviewRetryTitle", {
                          defaultValue: "Review not saved"
                        })}
                        description={reviewFailure.message}
                        data-testid="flashcards-review-retry-alert"
                        action={
                          <Space orientation="vertical" size={8}>
                            <Button
                              size="small"
                              type="primary"
                              loading={reviewMutation.isPending}
                              onClick={handleRetryFailedReview}
                              data-testid="flashcards-review-retry-button"
                            >
                              {t("option:flashcards.retryAction", {
                                defaultValue: "Retry"
                              })}
                            </Button>
                            {reviewFailure.canReload && (
                              <Button
                                size="small"
                                onClick={handleReloadFailedReviewCard}
                                loading={isReloadingFailedCard}
                                data-testid="flashcards-review-reload-button"
                              >
                                {t("option:flashcards.reloadCardAction", {
                                  defaultValue: "Reload card"
                                })}
                              </Button>
                            )}
                          </Space>
                        }
                      />
                    )}
                    <Text type="secondary" className="text-xs text-center">
                      {t("option:flashcards.ratingIntervalsMeaning", {
                        defaultValue:
                          "Again = shortest gap, Hard = short gap, Good = medium gap, Easy = longest gap."
                      })}
                    </Text>
                    {isCramMode && !cramUpdatesSchedule && (
                      <Text type="secondary" className="text-xs text-center">
                        {t("option:flashcards.cramNoScheduleHint", {
                          defaultValue:
                            "Practice-only mode: ratings do not change your review schedule."
                        })}
                      </Text>
                    )}
                    <Button
                      type="link"
                      size="small"
                      className="self-start !px-0"
                      onClick={() => setShowRatingRationale((prev) => !prev)}
                    >
                      {t("option:flashcards.whyTheseRatings", {
                        defaultValue: "Why these ratings?"
                      })}
                    </Button>
                    {showRatingRationale && (
                      <div className="rounded border border-border bg-surface p-2 text-xs text-text-muted">
                        <Text className="block text-xs">
                          {t("option:flashcards.ratingMapAgain", {
                            defaultValue: "Again = 0: forgot it, repeat very soon."
                          })}
                        </Text>
                        <Text className="block text-xs">
                          {t("option:flashcards.ratingMapHard", {
                            defaultValue: "Hard = 2: remembered with strain, keep gap short."
                          })}
                        </Text>
                        <Text className="block text-xs">
                          {t("option:flashcards.ratingMapGood", {
                            defaultValue: "Good = 3: normal recall, use the default schedule step."
                          })}
                        </Text>
                        <Text className="block text-xs">
                          {t("option:flashcards.ratingMapEasy", {
                            defaultValue: "Easy = 5: effortless recall, jump to a longer gap."
                          })}
                        </Text>
                      </div>
                    )}
                    <div
                      className="flex flex-wrap items-center gap-2"
                      data-testid="flashcards-review-shortcut-chips-answer"
                    >
                      {shortcutHintDensity === "expanded" && (
                        <>
                          <Tag className="!m-0">
                            {t("option:flashcards.shortcutChipRate", {
                              defaultValue: "1-4 Rate"
                            })}
                          </Tag>
                          <Tag className="!m-0">
                            {t("option:flashcards.shortcutChipEdit", {
                              defaultValue: "E Edit"
                            })}
                          </Tag>
                          <Tag className="!m-0">
                            {t("option:flashcards.shortcutChipUndo", {
                              defaultValue: "Ctrl+Z Re-rate"
                            })}
                          </Tag>
                        </>
                      )}
                      {shortcutHintDensity === "compact" && (
                        <Tag className="!m-0">
                          {t("option:flashcards.shortcutChipReviewCompactAnswer", {
                            defaultValue: "1-4 / E / Ctrl+Z"
                          })}
                        </Tag>
                      )}
                      <Button
                        type="link"
                        size="small"
                        className="!h-auto !px-0 text-xs"
                        onClick={cycleShortcutHintDensity}
                        data-testid="flashcards-review-shortcut-hints-toggle"
                      >
                        {shortcutHintToggleLabel}
                      </Button>
                    </div>

                    {/* Re-rate button - appears briefly after rating with countdown */}
                    {showUndoButton && lastReviewedCard && (
                      <div className="mt-3 pt-3 border-t border-border">
                        <Button
                          type="text"
                          icon={<Undo2 className="size-4" />}
                          onClick={handleUndoReview}
                          className="text-text-muted hover:text-text min-h-11 focus:ring-2 focus:ring-primary focus:ring-offset-2"
                          aria-label={t("option:flashcards.undoRatingAria", {
                            defaultValue: "Re-rate last card, {{seconds}} seconds remaining",
                            seconds: undoCountdown
                          })}
                        >
                          <span className="flex items-center gap-2">
                            {t("option:flashcards.undoRating", {
                              defaultValue: "Re-rate last card"
                            })}
                            <span
                              className="inline-flex items-center justify-center min-w-6 h-6 px-1.5 rounded-full bg-surface2 text-xs font-medium tabular-nums"
                              role="timer"
                              aria-live="polite"
                            >
                              {undoCountdown}s
                            </span>
                          </span>
                        </Button>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </Card>
      ) : (
        <Card data-testid="flashcards-review-empty-card">
          <Empty
            description={
              hasCardsQuery.data === false
                ? t("option:flashcards.noCardsYet", {
                    defaultValue: "No flashcards yet"
                  })
                : isCramMode && cramTagFilter
                  ? t("option:flashcards.cramNoCardsForTag", {
                      defaultValue: "No cards match this cram tag filter."
                    })
                  : isCramMode
                    ? t("option:flashcards.cramComplete", {
                        defaultValue: "Cram session complete!"
                      })
                : t("option:flashcards.allCaughtUp", {
                    defaultValue: "You're all caught up!"
                  })
            }
          >
            <Space orientation="vertical" align="center">
              {hasCardsQuery.data === false ? (
                <>
                  {showReviewOnboardingGuide ? (
                    <Alert
                      type="info"
                      showIcon
                      className="w-full max-w-2xl text-left"
                      data-testid="flashcards-review-onboarding-guide"
                      title={t("option:flashcards.onboardingTitle", {
                        defaultValue: "How flashcard study works"
                      })}
                      description={
                        <div className="space-y-2">
                          <Text type="secondary" className="block">
                            {t("option:flashcards.onboardingDescription", {
                              defaultValue:
                                "Spaced repetition focuses on cards right before you forget them, so short daily sessions are enough."
                            })}
                          </Text>
                          <ol className="mb-0 list-decimal pl-5 text-xs text-text-muted">
                            <li>
                              {t("option:flashcards.onboardingStepCreate", {
                                defaultValue:
                                  "Add cards manually, import a deck, or generate cards from text."
                              })}
                            </li>
                            <li>
                              {t("option:flashcards.onboardingStepReview", {
                                defaultValue:
                                  "Review due cards each day and reveal answers with Space."
                              })}
                            </li>
                            <li>
                              {t("option:flashcards.onboardingStepRate", {
                                defaultValue:
                                  "Rate recall with Again/Hard/Good/Easy so the next review is scheduled automatically."
                              })}
                            </li>
                            <li className="text-text-muted">
                              {t("option:flashcards.onboardingStepLlmNote", {
                                defaultValue:
                                  "Card generation and the study assistant require an LLM provider (configure in Settings)."
                              })}
                            </li>
                          </ol>
                          <a
                            href={FLASHCARDS_HELP_LINKS.ratings}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-primary hover:underline"
                            data-testid="flashcards-review-onboarding-doc-link"
                          >
                            {t("option:flashcards.onboardingDocLink", {
                              defaultValue: "Open the full Flashcards study guide"
                            })}
                          </a>
                        </div>
                      }
                      action={
                        <Button
                          size="small"
                          onClick={() => setReviewOnboardingDismissed(true)}
                          data-testid="flashcards-review-onboarding-dismiss"
                        >
                          {t("option:flashcards.onboardingDismiss", {
                            defaultValue: "Hide guide"
                          })}
                        </Button>
                      }
                    />
                  ) : (
                    <Button
                      type="link"
                      size="small"
                      className="!px-0"
                      onClick={() => setReviewOnboardingDismissed(false)}
                      data-testid="flashcards-review-onboarding-reopen"
                    >
                      {t("option:flashcards.onboardingReopen", {
                        defaultValue: "Show study guide"
                      })}
                    </Button>
                  )}
                  <Text type="secondary">
                    {t("option:flashcards.noCardsDescription", {
                      defaultValue:
                        "Get started by creating cards, importing a deck, or generating flashcards from your content."
                    })}
                  </Text>
                  <Space>
                    <Button
                      type="primary"
                      onClick={onNavigateToCreate}
                      data-testid="flashcards-review-empty-create-cta"
                    >
                      {t("option:flashcards.createFirstCard", {
                        defaultValue: "Create card"
                      })}
                    </Button>
                    <Button
                      onClick={onNavigateToImport}
                      data-testid="flashcards-review-empty-import-cta"
                    >
                      {t("option:flashcards.noDueImportCta", {
                        defaultValue: "Import a deck"
                      })}
                    </Button>
                    <Button
                      onClick={onNavigateToImport}
                      data-testid="flashcards-review-empty-generate-cta"
                    >
                      {t("option:flashcards.generateFromTextCta", {
                        defaultValue: "Generate from text"
                      })}
                    </Button>
                  </Space>
                </>
              ) : (
                <>
                  {/* Celebratory state with session stats */}
                  {reviewedCount > 0 && (
                    <div className="text-center mb-3">
                      <Text className="text-4xl">🎉</Text>
                      <div className="mt-2">
                        <Text strong className="text-lg">
                          {isCramMode
                            ? t("option:flashcards.reviewedThisCramSession", {
                                defaultValue:
                                  "{{count}} cards practiced in this cram session",
                                count: reviewedCount
                              })
                            : t("option:flashcards.reviewedThisSession", {
                                defaultValue: "{{count}} cards reviewed this session",
                                count: reviewedCount
                              })}
                        </Text>
                      </div>
                    </div>
                  )}
                  <Text type="secondary">
                    {isCramMode
                      ? t("option:flashcards.cramCompleteDescription", {
                          defaultValue:
                            "You reached the end of your cram queue."
                        })
                      : t("option:flashcards.allCaughtUpDescription", {
                          defaultValue:
                            "No cards are due for review. Great job!"
                        })}
                  </Text>

                  {/* Next due date information */}
                  {!isCramMode && nextDueInfo && (
                    <div className="mt-4 p-3 rounded-lg bg-surface2 border border-border">
                      {nextDueInfo.nextDueAt ? (
                        <>
                          <div className="flex items-center gap-2 text-sm">
                            <Calendar className="size-4 text-primary" aria-hidden="true" />
                            <Text strong>
                              {t("option:flashcards.nextDueAt", {
                                defaultValue: "Next review: {{time}}",
                                time: dayjs(nextDueInfo.nextDueAt).fromNow()
                              })}
                            </Text>
                          </div>
                          <Text type="secondary" className="text-xs mt-1 block">
                            {dayjs(nextDueInfo.nextDueAt).format("dddd, MMMM D [at] h:mm A")}
                            {" · "}
                            {t("option:flashcards.nextDueCardCount", {
                              defaultValue: "{{count}} cards due",
                              count: nextDueInfo.cardsDue
                            })}
                          </Text>
                        </>
                      ) : (
                        <Text strong className="text-sm">
                          {t("option:flashcards.nextDueUnavailable", {
                            defaultValue: "Next review estimate unavailable"
                          })}
                        </Text>
                      )}
                      {nextDueInfo.isCapped && (
                        <Text type="secondary" className="text-xs mt-2 block">
                          {t("option:flashcards.nextDueCapped", {
                            defaultValue:
                              "Next review is beyond the first {{count}} cards. Narrow filters to improve the estimate.",
                            count: nextDueInfo.scanned
                          })}
                        </Text>
                      )}
                    </div>
                  )}

                  <Button type="link" onClick={onNavigateToCreate}>
                    {t("option:flashcards.createMoreCards", {
                      defaultValue: "Create card"
                    })}
                  </Button>
                  {activeReviewSessionId != null && (
                    <Button
                      type="default"
                      loading={endReviewSessionMutation.isPending}
                      onClick={() => {
                        void completeReviewSession("manual")
                      }}
                      data-testid="flashcards-review-end-session-empty"
                    >
                      {t("option:flashcards.endSession", {
                        defaultValue: "End Session"
                      })}
                    </Button>
                  )}
                </>
              )}
          </Space>
        </Empty>
      </Card>
      )}
      {selectedStudySessionId != null && (
        <div className="mt-4">
          <StudySuggestionsPanel
            anchorType="flashcard_review_session"
            anchorId={selectedStudySessionId}
            onActionResult={handleStudySuggestionActionResult}
          />
        </div>
      )}
      <div className="mt-4">
        <RecentStudySessions
          deckId={reviewDeckId ?? null}
          selectedSessionId={selectedStudySessionId}
          onOpenSession={(sessionId) => {
            setSelectedStudySessionId(sessionId)
            setActiveReviewSessionId(null)
          }}
          isActive={isActive}
        />
      </div>
      <FlashcardEditDrawer
        open={editDrawerOpen}
        onClose={() => setEditDrawerOpen(false)}
        card={activeCard ?? null}
        onSave={handleSaveEdit}
        onDelete={handleDeleteFromReview}
        onResetScheduling={handleResetSchedulingFromReview}
        isLoading={
          updateMutation.isPending ||
          deleteMutation.isPending ||
          resetSchedulingMutation.isPending
        }
        decks={decksQuery.data || []}
        decksLoading={decksQuery.isLoading}
      />
    </div>
  )
}

export default ReviewTab
