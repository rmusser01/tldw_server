import React from "react"
import {
  Button,
  Card,
  Checkbox,
  Select,
  Descriptions,
  Empty,
  List,
  Modal,
  Progress,
  Skeleton,
  Statistic,
  Tag,
  Typography,
  message
} from "antd"
import { useTranslation } from "react-i18next"
import {
  CheckCircleOutlined,
  BookOutlined,
  ClockCircleOutlined,
  CloseCircleOutlined,
  EyeOutlined,
  RedoOutlined,
  TrophyOutlined
} from "@ant-design/icons"
import { useNavigate } from "react-router-dom"
import {
  useAllAttemptsQuery,
  useAttemptQuery,
  useAttemptRemediationConversionsQuery,
  useConvertAttemptRemediationQuestionsMutation,
  useGenerateRemediationQuizMutation,
  useQuizzesQuery
} from "../hooks"
import { useDecksQuery } from "@/components/Flashcards/hooks/useFlashcardQueries"
import { NewDeckConfigurationFields } from "@/components/Flashcards/components/NewDeckConfigurationFields"
import { useDeckSchedulerDraft } from "@/components/Flashcards/hooks/useDeckSchedulerDraft"
import { formatSchedulerSummary } from "@/components/Flashcards/utils/scheduler-settings"
import type { DeckReviewPromptSide } from "@/services/flashcards"
import { StudySuggestionsPanel } from "@/components/StudySuggestions/StudySuggestionsPanel"
import type {
  AnswerValue,
  QuestionPublic,
  QuizAnswer,
  QuizRemediationConversionSummary
} from "@/services/quizzes"
import {
  parseStudySuggestionTargetId,
  type StudySuggestionActionRequest,
  type StudySuggestionActionResponse
} from "@/services/studySuggestions"
import { buildFlashcardsStudyRouteFromQuiz } from "@/services/tldw/quiz-flashcards-handoff"
import type { TakeTabNavigationIntent } from "../navigation"
import { RESULTS_FILTER_PREFS_KEY } from "../stateKeys"
import { QuizRemediationPanel } from "../components/QuizRemediationPanel"
import { QuizMarkdown } from "../components/QuizMarkdown"
import { SourceCitations } from "../components/SourceCitations"
import { formatFillBlankAcceptedAnswers } from "../utils/fillBlankAnswer"
import { formatMatchingAnswer } from "../utils/matchingAnswer"

const { Text } = Typography

const DEFAULT_PASSING_SCORE = 70
type PassFilterKey = "all" | "pass" | "fail"
type DateRangeFilterKey = "all" | "7d" | "30d" | "90d"
type DeckTargetValue = number | "__new__" | null

const normalizeMultiSelectAnswer = (value: unknown): number[] => {
  if (Array.isArray(value)) {
    return Array.from(new Set(
      value
        .map((entry) => Number(entry))
        .filter((entry) => Number.isFinite(entry) && entry >= 0)
    )).sort((a, b) => a - b)
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return []
    try {
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed)) return normalizeMultiSelectAnswer(parsed)
    } catch {
      // Fall through to comma-separated parsing.
    }
    return normalizeMultiSelectAnswer(trimmed.split(","))
  }
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return [Math.floor(value)]
  }
  return []
}

type ResultsFilterPrefs = {
  page: number
  pageSize: number
  quizFilterId: number | null
  passFilter: PassFilterKey
  dateRangeFilter: DateRangeFilterKey
}

const DEFAULT_RESULTS_FILTER_PREFS: ResultsFilterPrefs = {
  page: 1,
  pageSize: 10,
  quizFilterId: null,
  passFilter: "all",
  dateRangeFilter: "all"
}

const readStoredResultsFilterPrefs = (): ResultsFilterPrefs => {
  if (typeof window === "undefined") {
    return DEFAULT_RESULTS_FILTER_PREFS
  }
  try {
    const raw = window.sessionStorage.getItem(RESULTS_FILTER_PREFS_KEY)
    if (!raw) return DEFAULT_RESULTS_FILTER_PREFS
    const parsed = JSON.parse(raw) as Partial<ResultsFilterPrefs>
    const page = typeof parsed.page === "number" && parsed.page > 0 ? parsed.page : DEFAULT_RESULTS_FILTER_PREFS.page
    const pageSize = typeof parsed.pageSize === "number" && parsed.pageSize > 0
      ? parsed.pageSize
      : DEFAULT_RESULTS_FILTER_PREFS.pageSize
    const quizFilterId = typeof parsed.quizFilterId === "number" ? parsed.quizFilterId : null
    const passFilter = parsed.passFilter === "pass" || parsed.passFilter === "fail" ? parsed.passFilter : "all"
    const dateRangeFilter = parsed.dateRangeFilter === "7d"
      || parsed.dateRangeFilter === "30d"
      || parsed.dateRangeFilter === "90d"
      ? parsed.dateRangeFilter
      : "all"
    return {
      page,
      pageSize,
      quizFilterId,
      passFilter,
      dateRangeFilter
    }
  } catch {
    return DEFAULT_RESULTS_FILTER_PREFS
  }
}

interface ResultsTabProps {
  onRetakeQuiz?: (intent: TakeTabNavigationIntent) => void
}

type RemediationConversionState = {
  active: QuizRemediationConversionSummary | null
}

export const ResultsTab: React.FC<ResultsTabProps> = ({ onRetakeQuiz }) => {
  const { t } = useTranslation(["option", "common"])
  const navigate = useNavigate()
  const [messageApi, contextHolder] = message.useMessage()

  const [page, setPage] = React.useState(() => readStoredResultsFilterPrefs().page)
  const [pageSize, setPageSize] = React.useState(() => readStoredResultsFilterPrefs().pageSize)
  const [quizFilterId, setQuizFilterId] = React.useState<number | null>(
    () => readStoredResultsFilterPrefs().quizFilterId
  )
  const [passFilter, setPassFilter] = React.useState<PassFilterKey>(
    () => readStoredResultsFilterPrefs().passFilter
  )
  const [dateRangeFilter, setDateRangeFilter] = React.useState<DateRangeFilterKey>(
    () => readStoredResultsFilterPrefs().dateRangeFilter
  )
  const [selectedAttemptId, setSelectedAttemptId] = React.useState<number | null>(null)
  const [flashcardModalOpen, setFlashcardModalOpen] = React.useState(false)
  const [selectedMissedQuestions, setSelectedMissedQuestions] = React.useState<Record<number, boolean>>({})
  const [deckTarget, setDeckTarget] = React.useState<DeckTargetValue>(null)
  const [newDeckName, setNewDeckName] = React.useState("")
  const [reviewPromptSide, setReviewPromptSide] = React.useState<DeckReviewPromptSide>("front")
  const [replaceExistingQuestionIds, setReplaceExistingQuestionIds] = React.useState<number[]>([])
  const newDeckSchedulerDraft = useDeckSchedulerDraft()

  const { data: decksData, isLoading: decksLoading } = useDecksQuery({
    enabled: selectedAttemptId != null
  })
  const remediationConversionsQuery = useAttemptRemediationConversionsQuery(
    selectedAttemptId,
    { enabled: selectedAttemptId != null }
  )
  const convertRemediationQuestionsMutation = useConvertAttemptRemediationQuestionsMutation()
  const generateRemediationQuizMutation = useGenerateRemediationQuizMutation()
  const flashcardMutationPending = convertRemediationQuestionsMutation.isPending

  const attemptsQueryParams = React.useMemo(() => ({
    limit: 200,
    offset: 0,
    quiz_id: quizFilterId ?? undefined
  }), [quizFilterId])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.sessionStorage.setItem(RESULTS_FILTER_PREFS_KEY, JSON.stringify({
        page,
        pageSize,
        quizFilterId,
        passFilter,
        dateRangeFilter
      } satisfies ResultsFilterPrefs))
    } catch {
      // Ignore session-storage persistence failures for filter prefs.
    }
  }, [page, pageSize, quizFilterId, passFilter, dateRangeFilter])

  React.useEffect(() => {
    if (selectedAttemptId == null) {
      setFlashcardModalOpen(false)
      setReplaceExistingQuestionIds([])
    }
  }, [selectedAttemptId])

  React.useEffect(() => {
    setSelectedMissedQuestions({})
    setReplaceExistingQuestionIds([])
  }, [selectedAttemptId])

  const { data: attemptsData, isLoading: attemptsLoading } = useAllAttemptsQuery({
    quiz_id: attemptsQueryParams.quiz_id
  })
  const attemptedQuizIds = React.useMemo(() => {
    const ids = new Set((attemptsData?.items ?? []).map((attempt) => attempt.quiz_id))
    return Array.from(ids)
  }, [attemptsData?.items])
  const quizLookupLimit = React.useMemo(() => {
    if (attemptedQuizIds.length === 0) return 20
    return Math.min(100, Math.max(20, attemptedQuizIds.length * 3))
  }, [attemptedQuizIds.length])
  const { data: quizzesData, isLoading: quizzesLoading } = useQuizzesQuery({
    limit: quizLookupLimit,
    offset: 0
  }, {
    enabled: !attemptsLoading
  })
  const {
    data: selectedAttemptDetails,
    isLoading: selectedAttemptLoading,
    isFetching: selectedAttemptFetching
  } = useAttemptQuery(
    selectedAttemptId,
    { includeQuestions: true },
    { enabled: selectedAttemptId != null }
  )

  const decks = decksData ?? []
  const selectedDeck = React.useMemo(
    () => (typeof deckTarget === "number" ? decks.find((deck) => deck.id === deckTarget) ?? null : null),
    [deckTarget, decks]
  )
  const remediationConversions = remediationConversionsQuery.data?.items ?? []
  const remediationConversionStateByQuestionId = React.useMemo(() => {
    return remediationConversions.reduce<Map<number, RemediationConversionState>>((acc, item) => {
      if (item.status === "active") {
        acc.set(item.question_id, { active: item })
      }
      return acc
    }, new Map())
  }, [remediationConversions])
  const linkedStudyDeckId = React.useMemo(() => {
    const activeConversions = remediationConversions.filter(
      (item) => item.status === "active" && !item.orphaned
    )
    if (activeConversions.length === 0) return undefined

    const liveDeckIds = new Set(decks.map((deck) => deck.id))
    const firstDeckId = activeConversions[0]?.target_deck_id

    if (
      typeof firstDeckId !== "number" ||
      !liveDeckIds.has(firstDeckId) ||
      !activeConversions.every(
        (item) => item.target_deck_id === firstDeckId && liveDeckIds.has(firstDeckId)
      )
    ) {
      return undefined
    }

    return firstDeckId
  }, [decks, remediationConversions])

  const handleNavigateToFlashcardsStudy = React.useCallback(
    (params: { quizId: number; attemptId: number; deckId?: number }) => {
      const route = buildFlashcardsStudyRouteFromQuiz({
        quizId: params.quizId,
        attemptId: params.attemptId,
        deckId: params.deckId ?? linkedStudyDeckId
      })
      navigate(route)
    },
    [linkedStudyDeckId, navigate]
  )

  const attempts = attemptsData?.items ?? []
  const selectedAttemptSummary = React.useMemo(() => {
    if (selectedAttemptId == null) {
      return null
    }

    const listAttempt = attempts.find((attempt) => attempt.id === selectedAttemptId)
    if (listAttempt) {
      return {
        id: listAttempt.id,
        quizId: listAttempt.quiz_id
      }
    }

    if (selectedAttemptDetails) {
      return {
        id: selectedAttemptDetails.id,
        quizId: selectedAttemptDetails.quiz_id
      }
    }

    return null
  }, [attempts, selectedAttemptDetails, selectedAttemptId])

  const handleStudySuggestionActionResult = React.useCallback(
    async (response: StudySuggestionActionResponse, request: StudySuggestionActionRequest) => {
      if (!selectedAttemptSummary) return

      const targetId = parseStudySuggestionTargetId(response.target_id)

      if (response.target_service === "flashcards" && targetId) {
        handleNavigateToFlashcardsStudy({
          quizId: selectedAttemptSummary.quizId,
          attemptId: selectedAttemptSummary.id,
          deckId: targetId
        })
        return
      }

      if (response.target_service === "quiz" && request.actionKind === "follow_up_quiz" && targetId) {
        if (onRetakeQuiz) {
          onRetakeQuiz({
            startQuizId: targetId,
            highlightQuizId: targetId
          })
        } else {
          const params = new URLSearchParams()
          params.set("tab", "take")
          params.set("start_quiz_id", String(targetId))
          params.set("highlight_quiz_id", String(targetId))
          navigate(`/quiz?${params.toString()}`)
        }
      }
    },
    [handleNavigateToFlashcardsStudy, navigate, onRetakeQuiz, selectedAttemptSummary]
  )

  const quizzes = quizzesData?.items ?? []

  const quizMap = React.useMemo(() => {
    const map = new Map<number, { name: string; passingScore: number | null; mediaId: number | null }>()
    quizzes.forEach((q) => {
      map.set(q.id, {
        name: q.name,
        passingScore: typeof q.passing_score === "number" ? q.passing_score : null,
        mediaId: typeof q.media_id === "number" ? q.media_id : null
      })
    })
    attemptedQuizIds.forEach((id) => {
      if (map.has(id)) return
      map.set(id, {
        name: `Quiz #${id}`,
        passingScore: null,
        mediaId: null
      })
    })
    return map
  }, [attemptedQuizIds, quizzes])
  const getPassingScoreForQuiz = React.useCallback((quizId: number) => {
    return quizMap.get(quizId)?.passingScore ?? DEFAULT_PASSING_SCORE
  }, [quizMap])
  const quizFilterOptions = React.useMemo(() => ([
    {
      value: "all" as const,
      label: t("option:quiz.filterAllQuizzes", { defaultValue: "All quizzes" })
    },
    ...[...attemptedQuizIds]
      .sort((left, right) => left - right)
      .map((quizId) => ({
        value: quizId,
        label: quizMap.get(quizId)?.name ?? `Quiz #${quizId}`
      }))
  ]), [attemptedQuizIds, quizMap, t])

  const filteredAttempts = React.useMemo(() => {
    const nowMs = Date.now()
    const dateRangeDays = dateRangeFilter === "7d"
      ? 7
      : dateRangeFilter === "30d"
        ? 30
        : dateRangeFilter === "90d"
          ? 90
          : null

    return attempts.filter((attempt) => {
      if (quizFilterId != null && attempt.quiz_id !== quizFilterId) {
        return false
      }

      if (passFilter !== "all") {
        if (!attempt.completed_at) return false
        const percentage = attempt.total_possible > 0
          ? Math.round(((attempt.score ?? 0) / attempt.total_possible) * 100)
          : 0
        const passingScore = getPassingScoreForQuiz(attempt.quiz_id)
        if (passFilter === "pass" && percentage < passingScore) return false
        if (passFilter === "fail" && percentage >= passingScore) return false
      }

      if (dateRangeDays != null) {
        const referenceDate = attempt.completed_at ?? attempt.started_at
        if (!referenceDate) return false
        const referenceMs = new Date(referenceDate).getTime()
        if (Number.isNaN(referenceMs)) return false
        const windowMs = dateRangeDays * 24 * 60 * 60 * 1000
        if (nowMs - referenceMs > windowMs) return false
      }

      return true
    })
  }, [attempts, dateRangeFilter, getPassingScoreForQuiz, passFilter, quizFilterId])

  const totalFilteredAttempts = filteredAttempts.length
  const paginatedAttempts = React.useMemo(() => {
    const offset = (page - 1) * pageSize
    return filteredAttempts.slice(offset, offset + pageSize)
  }, [filteredAttempts, page, pageSize])

  React.useEffect(() => {
    const maxPage = Math.max(1, Math.ceil(totalFilteredAttempts / pageSize))
    if (page > maxPage) {
      setPage(maxPage)
    }
  }, [page, pageSize, totalFilteredAttempts])

  const isLoading = attemptsLoading || quizzesLoading
  const hasActiveFilters = quizFilterId != null || passFilter !== "all" || dateRangeFilter !== "all"

  // Calculate stats
  const stats = React.useMemo(() => {
    if (filteredAttempts.length === 0) return null

    const completedAttempts = filteredAttempts.filter((a) => a.completed_at)
    const totalScore = completedAttempts.reduce((sum, a) => sum + (a.score ?? 0), 0)
    const totalPossible = completedAttempts.reduce((sum, a) => sum + a.total_possible, 0)
    const avgScore = totalPossible > 0 ? Math.round((totalScore / totalPossible) * 100) : 0

    const totalTime = completedAttempts.reduce((sum, a) => sum + (a.time_spent_seconds ?? 0), 0)
    const avgTime = completedAttempts.length > 0 ? Math.round(totalTime / completedAttempts.length) : 0

    return {
      totalAttempts: completedAttempts.length,
      avgScore,
      avgTime,
      uniqueQuizzes: new Set(completedAttempts.map((a) => a.quiz_id)).size
    }
  }, [filteredAttempts])

  const scoreTrend = React.useMemo(() => {
    return filteredAttempts
      .filter((attempt) => attempt.completed_at && attempt.total_possible > 0)
      .map((attempt) => ({
        attemptId: attempt.id,
        completedAt: attempt.completed_at ?? attempt.started_at,
        scorePct: Math.round(((attempt.score ?? 0) / attempt.total_possible) * 100)
      }))
      .sort((left, right) => new Date(left.completedAt).getTime() - new Date(right.completedAt).getTime())
      .slice(-20)
  }, [filteredAttempts])

  const scoreDistribution = React.useMemo(() => {
    const buckets = [
      { label: "0-49%", min: 0, max: 49, count: 0 },
      { label: "50-69%", min: 50, max: 69, count: 0 },
      { label: "70-84%", min: 70, max: 84, count: 0 },
      { label: "85-100%", min: 85, max: 100, count: 0 }
    ]

    filteredAttempts
      .filter((attempt) => attempt.completed_at && attempt.total_possible > 0)
      .forEach((attempt) => {
        const scorePct = Math.round(((attempt.score ?? 0) / attempt.total_possible) * 100)
        const bucket = buckets.find((candidate) => scorePct >= candidate.min && scorePct <= candidate.max)
        if (bucket) {
          bucket.count += 1
        }
      })

    return buckets
  }, [filteredAttempts])

  const scoreTrendPolylinePoints = React.useMemo(() => {
    if (scoreTrend.length < 2) return null
    const width = 360
    const height = 84
    const padding = 8
    const drawableWidth = width - padding * 2
    const drawableHeight = height - padding * 2
    return scoreTrend
      .map((entry, index) => {
        const x = padding + (index / (scoreTrend.length - 1)) * drawableWidth
        const y = padding + ((100 - entry.scorePct) / 100) * drawableHeight
        return `${x},${y}`
      })
      .join(" ")
  }, [scoreTrend])

  const scoreDistributionMaxCount = React.useMemo(() => {
    const max = Math.max(...scoreDistribution.map((bucket) => bucket.count), 0)
    return max > 0 ? max : 1
  }, [scoreDistribution])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    })
  }

  const resetFilters = () => {
    setQuizFilterId(null)
    setPassFilter("all")
    setDateRangeFilter("all")
    setPage(1)
  }

  const buildCsvCell = (value: string | number | null | undefined) => {
    const normalized = value == null ? "" : String(value)
    return `"${normalized.replace(/"/g, "\"\"")}"`
  }

  const exportFilteredAttemptsCsv = () => {
    if (filteredAttempts.length === 0) return

    const now = new Date()
    const fileDate = now.toISOString().slice(0, 10)
    const dateRangeEndIso = now.toISOString()
    const dateRangeStartIso = dateRangeFilter === "7d"
      ? new Date(now.getTime() - (7 * 24 * 60 * 60 * 1000)).toISOString()
      : dateRangeFilter === "30d"
        ? new Date(now.getTime() - (30 * 24 * 60 * 60 * 1000)).toISOString()
        : dateRangeFilter === "90d"
          ? new Date(now.getTime() - (90 * 24 * 60 * 60 * 1000)).toISOString()
          : ""

    const headers = [
      "attempt_id",
      "quiz_id",
      "quiz_name",
      "started_at",
      "completed_at",
      "time_spent_seconds",
      "score",
      "total_possible",
      "percentage",
      "passing_score",
      "is_passing",
      "filter_quiz_id",
      "filter_pass_state",
      "filter_date_range",
      "filter_date_start_iso",
      "filter_date_end_iso"
    ]

    const lines = filteredAttempts.map((attempt) => {
      const score = attempt.score ?? 0
      const total = attempt.total_possible
      const percentage = total > 0 ? Math.round((score / total) * 100) : 0
      const passingScore = getPassingScoreForQuiz(attempt.quiz_id)
      const isPassing = percentage >= passingScore
      const quizName = quizMap.get(attempt.quiz_id)?.name ?? `Quiz #${attempt.quiz_id}`

      return [
        attempt.id,
        attempt.quiz_id,
        quizName,
        attempt.started_at,
        attempt.completed_at ?? "",
        attempt.time_spent_seconds ?? "",
        score,
        total,
        percentage,
        passingScore,
        isPassing,
        quizFilterId ?? "",
        passFilter,
        dateRangeFilter,
        dateRangeStartIso,
        dateRangeEndIso
      ]
        .map((value) =>
          buildCsvCell(typeof value === "boolean" ? String(value) : value)
        )
        .join(",")
    })

    const csv = [headers.join(","), ...lines].join("\n")
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement("a")
    const quizSegment = quizFilterId != null ? `quiz-${quizFilterId}` : "all-quizzes"
    const dateSegment = dateRangeFilter === "all" ? "all-dates" : dateRangeFilter
    link.href = url
    link.download = `quiz-results-${quizSegment}-${dateSegment}-${fileDate}.csv`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  }

  const formatAnswerValue = React.useCallback((
    value: AnswerValue | null | undefined,
    question?: QuestionPublic
  ) => {
    if (value === null || value === undefined) {
      return t("option:quiz.noAnswer", { defaultValue: "No answer" })
    }
    if (question?.question_type === "fill_blank") {
      const accepted = formatFillBlankAcceptedAnswers(value)
      if (accepted.length > 1) {
        return accepted.join(" / ")
      }
      return accepted[0] ?? String(value)
    }
    if (question?.question_type === "multi_select") {
      const indices = normalizeMultiSelectAnswer(value)
      if (indices.length === 0) return String(value)
      const labels = indices.map((index) => question.options?.[index] ?? String(index))
      return labels.join(" / ")
    }
    if (question?.question_type === "multiple_choice") {
      const index = Number(value)
      if (!Number.isNaN(index)) {
        return question.options?.[index] ?? String(value)
      }
      return String(value)
    }
    if (question?.question_type === "matching") {
      return formatMatchingAnswer(value)
    }
    if (Array.isArray(value)) {
      return value.join(" / ")
    }
    if (question?.question_type === "true_false") {
      const normalized = String(value).trim().toLowerCase()
      if (normalized === "true") return t("option:quiz.true", { defaultValue: "True" })
      if (normalized === "false") return t("option:quiz.false", { defaultValue: "False" })
    }
    return String(value)
  }, [t])

  const detailRows = React.useMemo(() => {
    if (!selectedAttemptDetails) return []

    const questions = selectedAttemptDetails.questions ?? []
    const answers = selectedAttemptDetails.answers ?? []
    const answerMap = new Map<number, QuizAnswer>(answers.map((answer) => [answer.question_id, answer]))

    if (questions.length > 0) {
      return questions.map((question, index) => {
        const answer = answerMap.get(question.id)
        return {
          key: `q-${question.id}`,
          order: index + 1,
          questionText: question.question_text || `Question #${question.id}`,
          question,
          answer
        }
      })
    }

    return answers.map((answer, index) => ({
      key: `a-${answer.question_id}-${index}`,
      order: index + 1,
      questionText: t("option:quiz.questionFallbackLabel", {
        defaultValue: "Question #{{id}}",
        id: answer.question_id
      }),
      question: undefined,
      answer
    }))
  }, [selectedAttemptDetails, t])
  const selectedAttemptMediaId = React.useMemo(() => {
    if (!selectedAttemptDetails) return null
    return quizMap.get(selectedAttemptDetails.quiz_id)?.mediaId ?? null
  }, [quizMap, selectedAttemptDetails])

  const missedQuestionEntries = React.useMemo(() => {
    if (!selectedAttemptDetails) return []
    const questionMap = new Map(
      (selectedAttemptDetails.questions ?? []).map((question) => [question.id, question])
    )

    return (selectedAttemptDetails.answers ?? [])
      .filter((answer) => answer.is_correct === false)
      .map((answer) => {
        const question = questionMap.get(answer.question_id)
        const conversionState = remediationConversionStateByQuestionId.get(answer.question_id)
        const activeConversion = conversionState?.active ?? null
        const orphaned = Boolean(activeConversion?.orphaned)
        return {
          questionId: answer.question_id,
          questionText:
            question?.question_text ||
            t("option:quiz.questionFallbackLabel", {
              defaultValue: "Question #{{id}}",
              id: answer.question_id
          }),
          correctAnswerText: formatAnswerValue(answer.correct_answer, question),
          userAnswerText: formatAnswerValue(answer.user_answer, question),
          explanation: answer.explanation ?? null,
          alreadyConverted: Boolean(activeConversion && !orphaned),
          orphaned,
          convertedDeckName: activeConversion?.target_deck_name_snapshot ?? null,
          linkedFlashcardCount: activeConversion?.flashcard_count ?? 0,
          hasSupersededHistory: (activeConversion?.superseded_count ?? 0) > 0
        }
      })
  }, [formatAnswerValue, remediationConversionStateByQuestionId, selectedAttemptDetails, t])

  const hasMissedQuestions = missedQuestionEntries.length > 0
  const selectedMissedQuestionCount = React.useMemo(() => {
    return missedQuestionEntries.filter((entry) => selectedMissedQuestions[entry.questionId]).length
  }, [missedQuestionEntries, selectedMissedQuestions])

  const updateSelectedMissedQuestions = React.useCallback((
    updater: Record<number, boolean> | ((previous: Record<number, boolean>) => Record<number, boolean>)
  ) => {
    setReplaceExistingQuestionIds([])
    setSelectedMissedQuestions(updater)
  }, [])

  React.useEffect(() => {
    if (!flashcardModalOpen) return
    if (deckTarget === null) {
      setDeckTarget(decks[0]?.id ?? "__new__")
    }
  }, [deckTarget, decks, flashcardModalOpen])

  const openFlashcardModal = React.useCallback(() => {
    if (!selectedAttemptDetails) return
    if (missedQuestionEntries.length === 0) {
      messageApi.info(
        t("option:quiz.noMissedQuestionsToConvert", {
          defaultValue: "No missed questions to convert for this attempt."
        })
      )
      return
    }

    const hasExistingSelection = missedQuestionEntries.some(
      (entry) => selectedMissedQuestions[entry.questionId]
    )
    const nextSelection = hasExistingSelection
      ? { ...selectedMissedQuestions }
      : missedQuestionEntries.reduce<Record<number, boolean>>((acc, entry) => {
        acc[entry.questionId] = !entry.alreadyConverted || entry.orphaned
        return acc
      }, {})

    updateSelectedMissedQuestions(nextSelection)
    setDeckTarget(decks[0]?.id ?? "__new__")
    setReplaceExistingQuestionIds([])
    newDeckSchedulerDraft.resetToDefaults()

    const quizName =
      quizMap.get(selectedAttemptDetails.quiz_id)?.name ?? `Quiz #${selectedAttemptDetails.quiz_id}`
    setNewDeckName(`${quizName} - Missed Questions`)
    setFlashcardModalOpen(true)
  }, [
    decks,
    messageApi,
    missedQuestionEntries,
    newDeckSchedulerDraft,
    quizMap,
    selectedAttemptDetails,
    selectedMissedQuestions,
    t,
    updateSelectedMissedQuestions
  ])

  const handleCreateRemediationQuiz = React.useCallback(async (questionIds: number[]) => {
    if (!selectedAttemptDetails) return
    if (questionIds.length === 0) {
      messageApi.warning(
        t("option:quiz.selectAtLeastOneMissedQuestion", {
          defaultValue: "Select at least one missed question."
        })
      )
      return
    }

    try {
      const result = await generateRemediationQuizMutation.mutateAsync({
        attemptId: selectedAttemptDetails.id,
        questionIds
      })

      onRetakeQuiz?.({
        startQuizId: result.quiz.id,
        highlightQuizId: result.quiz.id,
        sourceTab: "results",
        attemptId: selectedAttemptDetails.id
      })

      messageApi.success(
        t("option:quiz.remediationQuizCreated", {
          defaultValue: "Created remediation quiz {{name}}.",
          name: result.quiz.name
        })
      )
    } catch (error) {
      messageApi.error(
        t("option:quiz.remediationQuizCreateError", {
          defaultValue: "Failed to create remediation quiz."
        })
      )
      if (error instanceof Error && error.message) {
        console.error("[ResultsTab] Failed generating remediation quiz:", error.message)
      }
    }
  }, [generateRemediationQuizMutation, messageApi, onRetakeQuiz, selectedAttemptDetails, t])

  const handleCreateFlashcardsFromMissedQuestions = React.useCallback(async () => {
    if (!selectedAttemptDetails) return

    const selectedEntries = missedQuestionEntries.filter(
      (entry) => selectedMissedQuestions[entry.questionId]
    )

    if (selectedEntries.length === 0) {
      messageApi.warning(
        t("option:quiz.selectAtLeastOneMissedQuestion", {
          defaultValue: "Select at least one missed question."
        })
      )
      return
    }

    const replaceActive = replaceExistingQuestionIds.length > 0
    const questionIds = replaceActive
      ? selectedEntries
          .map((entry) => entry.questionId)
          .filter((questionId) => replaceExistingQuestionIds.includes(questionId))
      : selectedEntries.map((entry) => entry.questionId)

    if (questionIds.length === 0) {
      messageApi.info(
        t("option:quiz.missedQuestionsAlreadyConverted", {
          defaultValue: "Selected questions were already converted for this attempt."
        })
      )
      return
    }

    const request =
      deckTarget === "__new__" || deckTarget == null
        ? (() => {
            const trimmedDeckName = newDeckName.trim()
            if (!trimmedDeckName) {
              messageApi.warning(
                t("option:quiz.deckNameRequired", {
                  defaultValue: "Enter a deck name."
                })
              )
              return null
            }
            const schedulerSettings = newDeckSchedulerDraft.getValidatedSettings()
            if (!schedulerSettings) {
              messageApi.warning(
                t("option:flashcards.schedulerSettingsInvalid", {
                  defaultValue: "Fix the scheduler settings before creating the deck."
                })
              )
              return null
            }
            return {
              question_ids: questionIds,
              create_deck_name: trimmedDeckName,
              create_deck_review_prompt_side: reviewPromptSide,
              create_deck_scheduler_type: schedulerSettings.scheduler_type,
              create_deck_scheduler_settings: schedulerSettings.scheduler_settings,
              replace_active: replaceActive
            } as const
          })()
        : ({
            question_ids: questionIds,
            target_deck_id: deckTarget,
            replace_active: replaceActive
          } as const)

    if (request == null) {
      return
    }

    try {
      const response = await convertRemediationQuestionsMutation.mutateAsync({
        attemptId: selectedAttemptDetails.id,
        request
      })

      if (response.target_deck?.id && (deckTarget === "__new__" || deckTarget == null)) {
        setDeckTarget(response.target_deck.id)
        setNewDeckName(response.target_deck.name)
      }

      const createdResults = response.results.filter(
        (result) => result.status === "created" || result.status === "superseded_and_created"
      )
      const existingResults = response.results.filter((result) => result.status === "already_exists")
      const failedResults = response.results.filter((result) => result.status === "failed")

      if (createdResults.length > 0) {
        messageApi.success(
          t("option:quiz.flashcardsCreatedFromMissed", {
            defaultValue: "Created {{count}} flashcards from missed questions.",
            count: response.created_flashcard_uuids.length || createdResults.length
          })
        )
      }

      if (existingResults.length > 0) {
        const conflictingQuestionIds = existingResults.map((result) => result.question_id)
        setReplaceExistingQuestionIds(conflictingQuestionIds)
        setSelectedMissedQuestions(
          conflictingQuestionIds.reduce<Record<number, boolean>>((acc, questionId) => {
            acc[questionId] = true
            return acc
          }, {})
        )
        messageApi.info(
          t("option:quiz.remediationConvertAgainPrompt", {
            defaultValue: "Some selected questions already have active remediation flashcards. Use Convert Again Anyway to supersede them."
          })
        )
        return
      }

      if (failedResults.length > 0) {
        messageApi.error(
          t("option:quiz.flashcardsCreateFromMissedError", {
            defaultValue: "Failed to create flashcards from missed questions."
          })
        )
        return
      }

      setReplaceExistingQuestionIds([])
      setFlashcardModalOpen(false)
    } catch (error) {
      messageApi.error(
        t("option:quiz.flashcardsCreateFromMissedError", {
          defaultValue: "Failed to create flashcards from missed questions."
        })
      )
      if (error instanceof Error && error.message) {
        console.error("[ResultsTab] Failed converting missed questions to flashcards:", error.message)
      }
    }
  }, [
    convertRemediationQuestionsMutation,
    deckTarget,
    messageApi,
    missedQuestionEntries,
    newDeckSchedulerDraft,
    newDeckName,
    reviewPromptSide,
    replaceExistingQuestionIds,
    selectedAttemptDetails,
    selectedMissedQuestions,
    t,
  ])

  const renderFlashcardConversionModal = () => (
    <Modal
      title={t("option:quiz.createFlashcardsFromMissed", {
        defaultValue: "Create Flashcards from Missed Questions"
      })}
      open={flashcardModalOpen}
      onCancel={() => {
        setReplaceExistingQuestionIds([])
        setFlashcardModalOpen(false)
      }}
      onOk={() => {
        void handleCreateFlashcardsFromMissedQuestions()
      }}
      okText={replaceExistingQuestionIds.length > 0
        ? t("option:quiz.convertAgainAnyway", { defaultValue: "Convert Again Anyway" })
        : t("option:quiz.createFlashcards", { defaultValue: "Create Flashcards" })}
      okButtonProps={{
        loading: flashcardMutationPending
      }}
      destroyOnHidden
    >
      {missedQuestionEntries.length === 0 ? (
        <Empty
          description={t("option:quiz.noMissedQuestionsToConvert", {
            defaultValue: "No missed questions to convert for this attempt."
          })}
        />
      ) : (
        <div className="space-y-4">
          {replaceExistingQuestionIds.length > 0 && (
            <div className="rounded border border-warning/40 bg-warning/10 p-3 text-sm text-text">
              {t("option:quiz.remediationConvertAgainPrompt", {
                defaultValue: "Some selected questions already have active remediation flashcards. Convert Again Anyway will supersede those prior conversions."
              })}
            </div>
          )}
          <div className="space-y-2">
            <div className="text-sm font-medium text-text">
              {t("option:quiz.flashcardDestination", {
                defaultValue: "Destination deck"
              })}
            </div>
            <Select<number | "__new__">
              value={deckTarget ?? undefined}
              onChange={(value) => setDeckTarget(value)}
              loading={decksLoading}
              options={[
                ...decks.map((deck) => ({
                  value: deck.id,
                  label: deck.name
                })),
                {
                  value: "__new__" as const,
                  label: t("option:quiz.createNewDeck", {
                    defaultValue: "Create new deck"
                  })
                }
              ]}
            />
            {deckTarget === "__new__" && (
              <NewDeckConfigurationFields
                deckName={newDeckName}
                onDeckNameChange={setNewDeckName}
                reviewPromptSide={reviewPromptSide}
                onReviewPromptSideChange={setReviewPromptSide}
                schedulerDraft={newDeckSchedulerDraft}
                nameTestId="quiz-remediation-new-deck-name"
                hint={t("option:quiz.newDeckSchedulerHint", {
                  defaultValue: "These scheduler settings will be applied to the new remediation deck."
                })}
              />
            )}
            {deckTarget !== "__new__" && selectedDeck?.scheduler_settings ? (
              <Text
                type="secondary"
                className="block text-xs"
                data-testid="quiz-remediation-selected-deck-summary"
              >
                {t("option:flashcards.selectedDeckSchedulerSummary", {
                  defaultValue: "Scheduler: {{summary}}",
                  summary: formatSchedulerSummary(
                    selectedDeck.scheduler_type,
                    selectedDeck.scheduler_settings
                  )
                })}
              </Text>
            ) : null}
          </div>

          <div className="space-y-2">
            <Checkbox
              checked={
                missedQuestionEntries.length > 0 &&
                selectedMissedQuestionCount === missedQuestionEntries.length
              }
              indeterminate={
                selectedMissedQuestionCount > 0 &&
                selectedMissedQuestionCount < missedQuestionEntries.length
              }
              onChange={(event) => {
                const checked = event.target.checked
                const next = missedQuestionEntries.reduce<Record<number, boolean>>((acc, entry) => {
                  acc[entry.questionId] = checked
                  return acc
                }, {})
                updateSelectedMissedQuestions(next)
              }}
            >
              {t("option:quiz.selectAllMissedQuestions", {
                defaultValue: "Select all missed questions"
              })}
            </Checkbox>

            <List
              dataSource={missedQuestionEntries}
              renderItem={(entry) => (
                <List.Item>
                  <div className="w-full space-y-1">
                    <Checkbox
                      checked={Boolean(selectedMissedQuestions[entry.questionId])}
                      onChange={(event) => {
                        updateSelectedMissedQuestions((previous) => ({
                          ...previous,
                          [entry.questionId]: event.target.checked
                        }))
                      }}
                    >
                      <span className="font-medium">{entry.questionText}</span>
                    </Checkbox>
                    <div className="pl-6 text-xs text-text-muted">
                      {t("option:quiz.correctAnswerLabel", { defaultValue: "Correct answer" })}:{" "}
                      {entry.correctAnswerText}
                    </div>
                    <div className="pl-6 text-xs text-text-subtle">
                      {t("option:quiz.yourAnswer", { defaultValue: "Your answer" })}:{" "}
                      {entry.userAnswerText}
                    </div>
                    {entry.orphaned ? (
                      <div className="pl-6 text-xs text-warning">
                        {t("option:quiz.linkedCardsDeleted", {
                          defaultValue: "Linked cards were deleted. Convert again to recreate them."
                        })}
                      </div>
                    ) : entry.alreadyConverted ? (
                      <div className="pl-6 text-xs text-text-subtle">
                        {entry.convertedDeckName
                          ? t("option:quiz.alreadyConvertedDeckLabel", {
                              defaultValue: "Converted in deck {{deckName}}.",
                              deckName: entry.convertedDeckName
                            })
                          : t("option:quiz.alreadyConverted", {
                              defaultValue: "Remediation flashcards already exist for this miss."
                            })}
                      </div>
                    ) : null}
                    {entry.hasSupersededHistory && (
                      <div className="pl-6 text-xs text-text-subtle">
                        {t("option:quiz.supersededRemediationHistory", {
                          defaultValue: "Superseded remediation history exists for this question."
                        })}
                      </div>
                    )}
                  </div>
                </List.Item>
              )}
            />
          </div>
        </div>
      )}
    </Modal>
  )

  const renderAttemptDetailsModal = () => (
    <Modal
      title={t("option:quiz.attemptDetails", { defaultValue: "Attempt Details" })}
      open={selectedAttemptId != null}
      onCancel={() => {
        setFlashcardModalOpen(false)
        setSelectedAttemptId(null)
      }}
      footer={(
        <div className="flex w-full flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <Button
              icon={<BookOutlined />}
              disabled={selectedAttemptDetails == null}
              onClick={() => {
                if (!selectedAttemptDetails) return
                handleNavigateToFlashcardsStudy({
                  quizId: selectedAttemptDetails.quiz_id,
                  attemptId: selectedAttemptDetails.id
                })
              }}
            >
              {t("option:quiz.studyWithFlashcards", {
                defaultValue: "Study with Flashcards"
              })}
            </Button>
            <Button
              type="primary"
              disabled={!hasMissedQuestions}
              onClick={openFlashcardModal}
            >
              {t("option:quiz.createFlashcardsFromMissed", {
                defaultValue: "Create Flashcards from Missed Questions"
              })}
            </Button>
          </div>
          <Button
            onClick={() => {
              setFlashcardModalOpen(false)
              setSelectedAttemptId(null)
            }}
          >
            {t("common:close", { defaultValue: "Close" })}
          </Button>
        </div>
      )}
      width={920}
      destroyOnHidden
    >
      {selectedAttemptId == null ? null : (
        <div className="space-y-4">
          <StudySuggestionsPanel
            anchorType="quiz_attempt"
            anchorId={selectedAttemptId}
            onActionResult={handleStudySuggestionActionResult}
          />

          {selectedAttemptLoading || selectedAttemptFetching ? (
            <div className="py-4" data-testid="results-detail-loading-skeleton">
              <Skeleton active paragraph={{ rows: 8 }} />
            </div>
          ) : !selectedAttemptDetails ? (
            <Empty
              description={t("option:quiz.attemptDetailsUnavailable", {
                defaultValue: "Attempt details are unavailable."
              })}
            />
          ) : (
            <>
              <Descriptions size="small" bordered column={1}>
                <Descriptions.Item label={t("option:quiz.quiz", { defaultValue: "Quiz" })}>
                  {quizMap.get(selectedAttemptDetails.quiz_id)?.name ?? `Quiz #${selectedAttemptDetails.quiz_id}`}
                </Descriptions.Item>
                <Descriptions.Item label={t("option:quiz.started", { defaultValue: "Started" })}>
                  {formatDate(selectedAttemptDetails.started_at)}
                </Descriptions.Item>
                <Descriptions.Item label={t("option:quiz.completed", { defaultValue: "Completed" })}>
                  {selectedAttemptDetails.completed_at
                    ? formatDate(selectedAttemptDetails.completed_at)
                    : t("option:quiz.inProgress", { defaultValue: "In progress" })}
                </Descriptions.Item>
                <Descriptions.Item label={t("option:quiz.timeSpent", { defaultValue: "Time Spent" })}>
                  {selectedAttemptDetails.time_spent_seconds != null
                    ? formatTime(selectedAttemptDetails.time_spent_seconds)
                    : t("option:quiz.notAvailable", { defaultValue: "Not available" })}
                </Descriptions.Item>
              </Descriptions>

              {hasMissedQuestions && (selectedAttemptDetails.questions?.length ?? 0) > 0 ? (
                <QuizRemediationPanel
                  attemptId={selectedAttemptDetails.id}
                  quizId={selectedAttemptDetails.quiz_id}
                  missedQuestionEntries={missedQuestionEntries}
                  selectedMissedQuestions={selectedMissedQuestions}
                  onSelectedMissedQuestionsChange={updateSelectedMissedQuestions}
                  onCreateRemediationQuiz={handleCreateRemediationQuiz}
                  onCreateRemediationFlashcards={openFlashcardModal}
                  onStudyLinkedCards={() => {
                    handleNavigateToFlashcardsStudy({
                      quizId: selectedAttemptDetails.quiz_id,
                      attemptId: selectedAttemptDetails.id
                    })
                  }}
                  remediationQuizPending={generateRemediationQuizMutation.isPending}
                />
              ) : null}

              <List
                dataSource={detailRows}
                locale={{
                  emptyText: t("option:quiz.noAnswersSubmitted", {
                    defaultValue: "No answer breakdown available."
                  })
                }}
                renderItem={(entry) => {
                  const wasAnswered = entry.answer != null
                  const isCorrect = Boolean(entry.answer?.is_correct)
                  const citations = Array.isArray(entry.answer?.source_citations)
                    ? entry.answer?.source_citations
                    : Array.isArray(entry.question?.source_citations)
                      ? entry.question?.source_citations
                      : null

                  return (
                    <List.Item>
                      <div className="w-full space-y-2">
                        <div className="flex items-start justify-between gap-3">
                          <div className="font-medium">
                            <span className="block text-xs text-text-muted">
                              {t("option:quiz.questionNumberLabel", {
                                defaultValue: "Question {{number}}",
                                number: entry.order
                              })}
                            </span>
                            <QuizMarkdown
                              content={entry.questionText}
                              className="[&>p]:my-1"
                            />
                          </div>
                          {!wasAnswered ? (
                            <Tag>{t("option:quiz.unanswered", { defaultValue: "Unanswered" })}</Tag>
                          ) : (
                            <Tag
                              color={isCorrect ? "success" : "error"}
                              icon={isCorrect ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                            >
                              {isCorrect
                                ? t("option:quiz.correct", { defaultValue: "Correct" })
                                : t("option:quiz.incorrect", { defaultValue: "Incorrect" })}
                            </Tag>
                          )}
                        </div>

                        <div className="text-sm text-text-muted">
                          {t("option:quiz.yourAnswer", { defaultValue: "Your answer" })}:{" "}
                          <span className="font-medium">
                            {formatAnswerValue(entry.answer?.user_answer, entry.question)}
                          </span>
                        </div>

                        <div className="text-sm text-text-muted">
                          {t("option:quiz.correctAnswerLabel", { defaultValue: "Correct answer" })}:{" "}
                          <span className="font-medium">
                            {formatAnswerValue(entry.answer?.correct_answer, entry.question)}
                          </span>
                        </div>

                        {entry.answer && (
                          <div className="text-sm text-text-muted">
                            {t("option:quiz.points", { defaultValue: "Points" })}:{" "}
                            <span className="font-medium">
                              {Number(entry.answer.points_awarded ?? 0)}
                              {entry.question ? ` / ${entry.question.points}` : ""}
                            </span>
                          </div>
                        )}

                        {entry.answer?.hint_used && (
                          <Typography.Text className="block text-xs text-text-muted">
                            {Number(entry.answer.hint_penalty_points ?? 0) > 0
                              ? t("option:quiz.hintPenaltyResult", {
                                defaultValue: "Hint used (-{{points}} point(s)).",
                                points: Number(entry.answer.hint_penalty_points ?? 0)
                              })
                              : t("option:quiz.hintUsedNoPenalty", {
                                defaultValue: "Hint used."
                              })}
                          </Typography.Text>
                        )}

                        {entry.answer?.explanation && (
                          <QuizMarkdown
                            content={entry.answer.explanation}
                            className="text-sm text-text-subtle"
                          />
                        )}
                        <SourceCitations
                          citations={citations}
                          fallbackMediaId={selectedAttemptMediaId}
                        />
                      </div>
                    </List.Item>
                  )
                }}
              />
            </>
          )}
        </div>
      )}
    </Modal>
  )

  if (isLoading) {
    return (
      <>
        {contextHolder}
        <div className="space-y-4 py-2" data-testid="results-loading-skeleton">
          <Card size="small">
            <Skeleton active paragraph={{ rows: 2 }} />
          </Card>
          <Card size="small">
            <Skeleton active paragraph={{ rows: 3 }} />
          </Card>
          <Card size="small">
            <Skeleton active paragraph={{ rows: 6 }} />
          </Card>
        </div>
      </>
    )
  }

  if (attempts.length === 0 && !hasActiveFilters) {
    return (
      <>
        {contextHolder}
        <Empty
          description={
            <div className="space-y-2">
              <p className="text-text-muted">
                {t("option:quiz.noAttempts", { defaultValue: "No quiz attempts yet" })}
              </p>
              <p className="text-sm text-text-subtle">
                {t("option:quiz.noAttemptsHint", {
                  defaultValue: "Complete a quiz to see your results here"
                })}
              </p>
            </div>
          }
        >
          {onRetakeQuiz && (
            <Button
              type="primary"
              onClick={() => onRetakeQuiz({
                startQuizId: null,
                highlightQuizId: null,
                forceShowWorkspaceItems: false,
                sourceTab: "results",
                attemptId: null,
                assignmentMode: null,
                assignmentDueAt: null,
                assignmentNote: null,
                assignedByRole: null
              })}
            >
              {t("option:quiz.takeAQuiz", { defaultValue: "Take a Quiz" })}
            </Button>
          )}
        </Empty>
      </>
    )
  }

  return (
    <>
      {contextHolder}
      <div className="space-y-6">
        {renderAttemptDetailsModal()}
        {renderFlashcardConversionModal()}
        {/* Stats summary */}
        {stats && (
          <Card size="small">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Statistic
                title={t("option:quiz.totalAttempts", { defaultValue: "Total Attempts" })}
                value={stats.totalAttempts}
                prefix={<TrophyOutlined />}
              />
              <Statistic
                title={t("option:quiz.avgScore", { defaultValue: "Average Score" })}
                value={stats.avgScore}
                suffix="%"
                styles={{
                  content: {
                    color:
                      stats.avgScore >= 70
                        ? "var(--color-success)"
                        : stats.avgScore >= 50
                          ? "var(--color-warn)"
                          : "var(--color-danger)"
                  }
                }}
              />
              <Statistic
                title={t("option:quiz.avgTime", { defaultValue: "Average Time" })}
                value={formatTime(stats.avgTime)}
                prefix={<ClockCircleOutlined />}
              />
              <Statistic
                title={t("option:quiz.uniqueQuizzes", { defaultValue: "Quizzes Taken" })}
                value={stats.uniqueQuizzes}
              />
            </div>
          </Card>
        )}

        <Card size="small" title={t("option:quiz.scoreTrend", { defaultValue: "Score Trend" })}>
          {scoreTrendPolylinePoints ? (
            <div className="space-y-4">
              <div className="overflow-x-auto">
                <svg
                  viewBox="0 0 360 84"
                  role="img"
                  aria-label={t("option:quiz.scoreTrendAria", {
                    defaultValue: "Score percentage trend over recent attempts"
                  })}
                  className="w-full min-w-[280px] h-24"
                >
                  <line x1="8" y1="76" x2="352" y2="76" stroke="var(--color-border)" strokeWidth="1" />
                  <polyline
                    fill="none"
                    stroke="var(--color-primary)"
                    strokeWidth="2"
                    points={scoreTrendPolylinePoints}
                  />
                </svg>
              </div>
              <div className="grid gap-2">
                {scoreDistribution.map((bucket) => (
                  <div key={bucket.label} className="flex items-center gap-3">
                    <span className="w-16 text-xs text-text-muted">{bucket.label}</span>
                    <div className="h-2 flex-1 rounded bg-surface2 overflow-hidden">
                      <div
                        className="h-full bg-primary"
                        style={{ width: `${(bucket.count / scoreDistributionMaxCount) * 100}%` }}
                      />
                    </div>
                    <span className="w-8 text-right text-xs text-text-subtle">{bucket.count}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <Text type="secondary">
              {t("option:quiz.scoreTrendNotEnoughData", {
                defaultValue: "Complete at least two attempts to see a trend."
              })}
            </Text>
          )}
        </Card>

        {/* Attempt history */}
        <div>
        <h3 className="text-lg font-medium mb-4">
          {t("option:quiz.attemptHistory", { defaultValue: "Attempt History" })}
        </h3>

        <div className="mb-4 flex flex-wrap items-center gap-3">
          <Select<number | "all">
            value={quizFilterId ?? "all"}
            onChange={(next) => {
              setQuizFilterId(next === "all" ? null : Number(next))
              setPage(1)
            }}
            style={{ minWidth: 220 }}
            options={quizFilterOptions}
          />

          <Select<PassFilterKey>
            value={passFilter}
            onChange={(next) => {
              setPassFilter(next)
              setPage(1)
            }}
            style={{ minWidth: 170 }}
            options={[
              {
                value: "all",
                label: t("option:quiz.filterAllResults", { defaultValue: "All results" })
              },
              {
                value: "pass",
                label: t("option:quiz.filterPassingOnly", { defaultValue: "Passing only" })
              },
              {
                value: "fail",
                label: t("option:quiz.filterFailingOnly", { defaultValue: "Failing only" })
              }
            ]}
          />

          <Select<DateRangeFilterKey>
            value={dateRangeFilter}
            onChange={(next) => {
              setDateRangeFilter(next)
              setPage(1)
            }}
            style={{ minWidth: 180 }}
            options={[
              {
                value: "all",
                label: t("option:quiz.filterAllDates", { defaultValue: "All dates" })
              },
              {
                value: "7d",
                label: t("option:quiz.filterLast7Days", { defaultValue: "Last 7 days" })
              },
              {
                value: "30d",
                label: t("option:quiz.filterLast30Days", { defaultValue: "Last 30 days" })
              },
              {
                value: "90d",
                label: t("option:quiz.filterLast90Days", { defaultValue: "Last 90 days" })
              }
            ]}
          />

          <Button onClick={resetFilters}>
            {t("common:reset", { defaultValue: "Reset" })}
          </Button>
          <Button onClick={exportFilteredAttemptsCsv} disabled={filteredAttempts.length === 0}>
            {t("option:quiz.exportCsv", { defaultValue: "Export CSV" })}
          </Button>
        </div>

        <List
          dataSource={paginatedAttempts}
          locale={{
            emptyText: t("option:quiz.noAttemptsMatchFilters", {
              defaultValue: "No attempts match the selected filters."
            })
          }}
          pagination={{
            current: page,
            pageSize,
            total: totalFilteredAttempts,
            showSizeChanger: true,
            locale: {
              items_per_page: t("option:quiz.itemsPerPage", { defaultValue: "items/page" })
            },
            onChange: (nextPage, nextPageSize) => {
              setPage(nextPage)
              if (nextPageSize && nextPageSize !== pageSize) {
                setPageSize(nextPageSize)
                setPage(1)
              }
            }
          }}
          renderItem={(attempt) => {
            const quizMeta = quizMap.get(attempt.quiz_id)
            const quizName = quizMeta?.name ?? `Quiz #${attempt.quiz_id}`
            const score = attempt.score ?? 0
            const total = attempt.total_possible
            const percentage = total > 0 ? Math.round((score / total) * 100) : 0
            const passingScore = getPassingScoreForQuiz(attempt.quiz_id)
            const isPassing = percentage >= passingScore

            return (
              <List.Item>
                <div className="w-full">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <Text strong>{quizName}</Text>
                      <div className="text-sm text-text-subtle">
                        {attempt.completed_at
                          ? formatDate(attempt.completed_at)
                          : t("option:quiz.inProgress", { defaultValue: "In progress" })}
                      </div>
                    </div>
                    <div className="text-right">
                      {attempt.completed_at ? (
                        <div className="space-y-1">
                          <Tag
                            color={isPassing ? "success" : "error"}
                            icon={isPassing ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                          >
                            {score}/{total} ({percentage}%)
                          </Tag>
                          <div>
                            <Button
                              type="link"
                              size="small"
                              icon={<EyeOutlined />}
                              onClick={() => setSelectedAttemptId(attempt.id)}
                            >
                              {t("option:quiz.viewDetails", { defaultValue: "View Details" })}
                            </Button>
                            <Button
                              type="link"
                              size="small"
                              icon={<BookOutlined />}
                              onClick={() =>
                                handleNavigateToFlashcardsStudy({
                                  quizId: attempt.quiz_id,
                                  attemptId: attempt.id
                                })
                              }
                            >
                              {t("option:quiz.studyWithFlashcards", {
                                defaultValue: "Study with Flashcards"
                              })}
                            </Button>
                            {onRetakeQuiz && (
                              <Button
                                type="link"
                                size="small"
                                icon={<RedoOutlined />}
                                onClick={() => onRetakeQuiz({
                                  startQuizId: attempt.quiz_id,
                                  highlightQuizId: attempt.quiz_id,
                                  sourceTab: "results",
                                  attemptId: attempt.id
                                })}
                              >
                                {t("option:quiz.retake", { defaultValue: "Retake" })}
                              </Button>
                            )}
                          </div>
                        </div>
                      ) : (
                        <Tag color="processing">
                          {t("option:quiz.incomplete", { defaultValue: "Incomplete" })}
                        </Tag>
                      )}
                    </div>
                  </div>

                  {attempt.completed_at && (
                    <div className="flex items-center gap-4">
                      <Progress
                        percent={percentage}
                        size="small"
                        aria-label={t("option:quiz.progressAria", {
                          defaultValue: "Quiz completion progress"
                        })}
                        strokeColor={
                          isPassing
                            ? "var(--color-success)"
                            : percentage >= 50
                              ? "var(--color-warn)"
                              : "var(--color-danger)"
                        }
                        className="flex-1"
                      />
                      {attempt.time_spent_seconds && (
                        <Text type="secondary" className="text-sm whitespace-nowrap">
                          <ClockCircleOutlined className="mr-1" />
                          {formatTime(attempt.time_spent_seconds)}
                        </Text>
                      )}
                    </div>
                  )}
                </div>
              </List.Item>
            )
          }}
        />
        </div>
      </div>
    </>
  )
}

export default ResultsTab
