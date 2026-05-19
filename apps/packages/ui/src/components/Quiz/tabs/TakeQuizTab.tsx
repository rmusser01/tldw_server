import React from "react"
import { Link } from "react-router-dom"
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Descriptions,
  Empty,
  Input,
  List,
  Modal,
  Progress,
  Radio,
  Select,
  Skeleton,
  Space,
  Tag,
  Tooltip,
  Typography,
  message
} from "antd"
import { useTranslation } from "react-i18next"
import {
  PlayCircleOutlined,
  ClockCircleOutlined,
  QuestionCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined
} from "@ant-design/icons"
import {
  useAttemptsQuery,
  useQuizzesQuery,
  useQuizQuery,
  useStartAttemptMutation,
  useSubmitAttemptMutation
} from "../hooks"
import { useQuizTimer } from "../hooks/useQuizTimer"
import { useQuizAutoSave } from "../hooks/useQuizAutoSave"
import {
  clearQueuedQuizSubmission,
  readQueuedQuizSubmission,
  writeQueuedQuizSubmission,
  type QueuedQuizSubmission
} from "../hooks/quizSubmissionQueue"
import { useServerOnline } from "@/hooks/useServerOnline"
import {
  listQuestions,
  type AnswerValue,
  type QuestionPublic,
  type Quiz,
  type QuizAnswer,
  type QuizAnswerInput,
  type QuizAttempt
} from "@/services/quizzes"
import type { TakeTabNavigationSource } from "../navigation"
import { TAKE_QUIZ_LIST_PREFS_KEY } from "../stateKeys"
import {
  buildShuffledOptionEntries,
  drawDeterministicQuestionPool,
  type ShuffledOptionEntry
} from "../utils/optionShuffle"
import {
  formatFillBlankAcceptedAnswers,
  isFillBlankAnswerCorrect
} from "../utils/fillBlankAnswer"
import {
  buildMatchingPairs,
  formatMatchingAnswer,
  isMatchingAnswerCorrect,
  normalizeMatchingAnswerMap
} from "../utils/matchingAnswer"
import { summarizeQuizSources } from "../utils/sourceBundle"
import { QuizMarkdown } from "../components/QuizMarkdown"
import { SourceCitations } from "../components/SourceCitations"

interface TakeQuizTabProps {
  onNavigateToGenerate: () => void
  onNavigateToCreate: () => void
  startQuizId?: number | null
  highlightQuizId?: number | null
  navigationSource?: TakeTabNavigationSource | null
  assignmentMode?: "shared" | null
  assignmentDueAt?: string | null
  assignmentNote?: string | null
  assignedByRole?: string | null
  externalSearchQuery?: string | null
  externalSearchToken?: number | null
  onStartHandled?: () => void
  onHighlightHandled?: () => void
  onExternalSearchHandled?: () => void
}

const DEFAULT_PASSING_SCORE = 70
type QuizSortKey = "name_asc" | "date_desc" | "questions_desc"
type TakeSessionMode = "graded" | "practice" | "review"
type StudyPoolSizePreference = "all" | number
type PracticeQuestionTimerPreference = "off" | number
const TOUCH_TARGET_CLASS = "min-h-11 px-4"

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
      if (Array.isArray(parsed)) {
        return normalizeMultiSelectAnswer(parsed)
      }
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

export const TakeQuizTab: React.FC<TakeQuizTabProps> = ({
  onNavigateToGenerate,
  onNavigateToCreate,
  startQuizId,
  highlightQuizId,
  navigationSource,
  assignmentMode,
  assignmentDueAt,
  assignmentNote,
  assignedByRole,
  externalSearchQuery,
  externalSearchToken,
  onStartHandled,
  onHighlightHandled,
  onExternalSearchHandled
}) => {
  const { t } = useTranslation(["option", "common"])
  const [messageApi, contextHolder] = message.useMessage()
  const [page, setPage] = React.useState(() => {
    if (typeof window === "undefined") return 1
    try {
      const raw = window.sessionStorage.getItem(TAKE_QUIZ_LIST_PREFS_KEY)
      if (!raw) return 1
      const parsed = JSON.parse(raw) as { page?: number }
      return typeof parsed.page === "number" && parsed.page > 0 ? parsed.page : 1
    } catch {
      return 1
    }
  })
  const [pageSize, setPageSize] = React.useState(() => {
    if (typeof window === "undefined") return 12
    try {
      const raw = window.sessionStorage.getItem(TAKE_QUIZ_LIST_PREFS_KEY)
      if (!raw) return 12
      const parsed = JSON.parse(raw) as { pageSize?: number }
      return typeof parsed.pageSize === "number" && parsed.pageSize > 0 ? parsed.pageSize : 12
    } catch {
      return 12
    }
  })
  const [searchQuery, setSearchQuery] = React.useState(() => {
    if (typeof window === "undefined") return ""
    try {
      const raw = window.sessionStorage.getItem(TAKE_QUIZ_LIST_PREFS_KEY)
      if (!raw) return ""
      const parsed = JSON.parse(raw) as { searchQuery?: string }
      return typeof parsed.searchQuery === "string" ? parsed.searchQuery : ""
    } catch {
      return ""
    }
  })
  const [sortBy, setSortBy] = React.useState<QuizSortKey>(() => {
    if (typeof window === "undefined") return "date_desc"
    try {
      const raw = window.sessionStorage.getItem(TAKE_QUIZ_LIST_PREFS_KEY)
      if (!raw) return "date_desc"
      const parsed = JSON.parse(raw) as { sortBy?: QuizSortKey }
      if (parsed.sortBy === "name_asc" || parsed.sortBy === "date_desc" || parsed.sortBy === "questions_desc") {
        return parsed.sortBy
      }
      return "date_desc"
    } catch {
      return "date_desc"
    }
  })
  const [modePreference, setModePreference] = React.useState<TakeSessionMode>(() => {
    if (typeof window === "undefined") return "graded"
    try {
      const raw = window.sessionStorage.getItem(TAKE_QUIZ_LIST_PREFS_KEY)
      if (!raw) return "graded"
      const parsed = JSON.parse(raw) as { modePreference?: TakeSessionMode }
      if (
        parsed.modePreference === "graded" ||
        parsed.modePreference === "practice" ||
        parsed.modePreference === "review"
      ) {
        return parsed.modePreference
      }
      return "graded"
    } catch {
      return "graded"
    }
  })
  const [studyPoolSizePreference, setStudyPoolSizePreference] = React.useState<StudyPoolSizePreference>(() => {
    if (typeof window === "undefined") return "all"
    try {
      const raw = window.sessionStorage.getItem(TAKE_QUIZ_LIST_PREFS_KEY)
      if (!raw) return "all"
      const parsed = JSON.parse(raw) as { studyPoolSize?: number | "all" }
      if (parsed.studyPoolSize === "all") {
        return "all"
      }
      if (typeof parsed.studyPoolSize === "number" && Number.isFinite(parsed.studyPoolSize) && parsed.studyPoolSize > 0) {
        return Math.floor(parsed.studyPoolSize)
      }
      return "all"
    } catch {
      return "all"
    }
  })
  const [studyPoolSeedOverride] = React.useState<number | null>(() => {
    if (typeof window === "undefined") return null
    try {
      const raw = window.sessionStorage.getItem(TAKE_QUIZ_LIST_PREFS_KEY)
      if (!raw) return null
      const parsed = JSON.parse(raw) as { studyPoolSeedOverride?: number }
      const rawValue = parsed.studyPoolSeedOverride
      if (!Number.isFinite(rawValue)) return null
      const normalized = Math.floor(Math.abs(rawValue)) >>> 0
      return normalized === 0 ? null : normalized
    } catch {
      return null
    }
  })
  const [practiceQuestionTimerPreference, setPracticeQuestionTimerPreference] = React.useState<PracticeQuestionTimerPreference>(() => {
    if (typeof window === "undefined") return "off"
    try {
      const raw = window.sessionStorage.getItem(TAKE_QUIZ_LIST_PREFS_KEY)
      if (!raw) return "off"
      const parsed = JSON.parse(raw) as { practiceQuestionTimerSeconds?: number | "off" }
      if (parsed.practiceQuestionTimerSeconds === "off") {
        return "off"
      }
      if (
        typeof parsed.practiceQuestionTimerSeconds === "number" &&
        Number.isFinite(parsed.practiceQuestionTimerSeconds) &&
        parsed.practiceQuestionTimerSeconds > 0
      ) {
        return Math.floor(parsed.practiceQuestionTimerSeconds)
      }
      return "off"
    } catch {
      return "off"
    }
  })
  const [activeQuizId, setActiveQuizId] = React.useState<number | null>(null)
  const [activeSessionMode, setActiveSessionMode] = React.useState<TakeSessionMode | null>(null)
  const [studySessionLoading, setStudySessionLoading] = React.useState(false)
  const [pendingQuizId, setPendingQuizId] = React.useState<number | null>(null)
  const [autoSaveWarningDismissed, setAutoSaveWarningDismissed] = React.useState(false)
  const [assignmentAlertDismissed, setAssignmentAlertDismissed] = React.useState(false)
  const [focusedQuestionId, setFocusedQuestionId] = React.useState<number | null>(null)
  const [unansweredQuestionNumbers, setUnansweredQuestionNumbers] = React.useState<number[]>([])
  const [attempt, setAttempt] = React.useState<QuizAttempt | null>(null)
  const [result, setResult] = React.useState<QuizAttempt | null>(null)
  const [questions, setQuestions] = React.useState<QuestionPublic[]>([])
  const [answers, setAnswers] = React.useState<Record<number, AnswerValue>>({})
  const [hintUsedByQuestionId, setHintUsedByQuestionId] = React.useState<Record<number, boolean>>({})
  const [practiceFeedbackByQuestionId, setPracticeFeedbackByQuestionId] = React.useState<Record<number, boolean | null>>({})
  const [practiceQuestionTimerRemainingSeconds, setPracticeQuestionTimerRemainingSeconds] = React.useState<number | null>(null)
  const [studySessionShuffleSeed, setStudySessionShuffleSeed] = React.useState<number | null>(null)
  const [queuedSubmission, setQueuedSubmission] = React.useState<QueuedQuizSubmission | null>(null)
  const [isRetryingQueuedSubmission, setIsRetryingQueuedSubmission] = React.useState(false)
  const [submissionQueueStorageUnavailable, setSubmissionQueueStorageUnavailable] = React.useState(false)
  const [highlightedQuizId, setHighlightedQuizId] = React.useState<number | null>(null)
  const lastAutoStartId = React.useRef<number | null>(null)
  const lastAutoHighlightId = React.useRef<number | null>(null)
  const lastExternalSearchToken = React.useRef<number | null>(null)
  const searchInputRef = React.useRef<any>(null)
  const hasInitializedSearchRef = React.useRef(false)
  const questionRefs = React.useRef<Map<number, HTMLDivElement | null>>(new Map())
  const lastExpiredPracticeQuestionIdRef = React.useRef<number | null>(null)
  const isOnline = useServerOnline()
  const wasOnlineRef = React.useRef(isOnline)
  const offset = (page - 1) * pageSize

  const normalizedSearchQuery = searchQuery.trim()
  const { data, isLoading } = useQuizzesQuery({
    limit: pageSize,
    offset,
    q: normalizedSearchQuery.length > 0 ? normalizedSearchQuery : undefined
  })
  const { data: attemptsData } = useAttemptsQuery({ limit: 200, offset: 0 })
  const detailQuizId = pendingQuizId ?? activeQuizId
  const { data: quizDetails } = useQuizQuery(detailQuizId, { enabled: detailQuizId != null })
  const directPreviewQuizId = startQuizId ?? highlightQuizId ?? null
  const { data: directPreviewQuiz } = useQuizQuery(directPreviewQuizId, {
    enabled: directPreviewQuizId != null && detailQuizId == null
  })
  const startAttemptMutation = useStartAttemptMutation()
  const submitAttemptMutation = useSubmitAttemptMutation()

  const quizzes = data?.items ?? []
  const attempts = attemptsData?.items ?? []
  const total = data?.count ?? 0
  const injectedDirectPreview = directPreviewQuiz != null && !quizzes.some((quiz) => quiz.id === directPreviewQuiz.id)
  const visibleTotal = total + (injectedDirectPreview ? 1 : 0)
  const sortedQuizzes = React.useMemo(() => {
    const items = [...quizzes]
    if (directPreviewQuiz && !items.some((quiz) => quiz.id === directPreviewQuiz.id)) {
      items.unshift(directPreviewQuiz)
    }
    if (sortBy === "name_asc") {
      items.sort((left, right) => left.name.localeCompare(right.name))
    } else if (sortBy === "questions_desc") {
      items.sort((left, right) => right.total_questions - left.total_questions)
    } else {
      items.sort((left, right) => {
        const leftTs = new Date(left.created_at ?? 0).getTime()
        const rightTs = new Date(right.created_at ?? 0).getTime()
        return rightTs - leftTs
      })
    }
    return items
  }, [directPreviewQuiz, quizzes, sortBy])

  const {
    storageUnavailable,
    restoreSavedAnswers,
    clearSavedProgress
  } = useQuizAutoSave(attempt?.id ?? null, attempt?.quiz_id ?? activeQuizId, answers, setAnswers)

  const timerState = useQuizTimer({
    timeLimitSeconds: quizDetails?.time_limit_seconds,
    startedAt: attempt?.started_at,
    enabled: attempt != null && result == null,
    onExpire: () => {
      void handleSubmit({ allowPartial: true })
    }
  })

  const timerAnnouncement = React.useMemo(() => {
    if (!timerState || timerState.isExpired || timerState.totalSeconds <= 0) return ""
    if (timerState.totalSeconds <= 60) {
      return t("option:quiz.timerSecondsRemaining", {
        defaultValue: "{{count}} seconds remaining",
        count: timerState.totalSeconds
      })
    }
    if (timerState.totalSeconds % 60 === 0) {
      return t("option:quiz.timerMinutesRemaining", {
        defaultValue: "{{count}} minutes remaining",
        count: Math.floor(timerState.totalSeconds / 60)
      })
    }
    return ""
  }, [timerState, t])

  const statusAnnouncement = React.useMemo(() => {
    if (result) {
      const total = result.total_possible || 0
      const score = result.score ?? 0
      const percentage = total > 0 ? Math.round((score / total) * 100) : 0
      return t("option:quiz.resultAnnouncement", {
        defaultValue: "Quiz submitted. Score {{score}} out of {{total}} ({{percent}} percent).",
        score,
        total,
        percent: percentage
      })
    }
    if (queuedSubmission) {
      return queuedSubmission.allowPartial
        ? t("option:quiz.queueAnnouncementTimer", {
            defaultValue:
              "Time expired and submission is queued. We will retry when connection is restored."
          })
        : t("option:quiz.queueAnnouncement", {
            defaultValue: "Submission failed and is queued locally. Retry is available."
          })
    }
    return ""
  }, [queuedSubmission, result, t])

  const getSubmissionErrorMessage = React.useCallback((error: unknown) => {
    if (error instanceof Error && error.message.trim().length > 0) {
      return error.message
    }
    const messageCandidate = (error as { message?: unknown } | null)?.message
    if (typeof messageCandidate === "string" && messageCandidate.trim().length > 0) {
      return messageCandidate
    }
    return t("option:quiz.submitErrorUnknown", {
      defaultValue: "Unknown submission error"
    })
  }, [t])

  const persistQueuedSubmission = React.useCallback(async (next: QueuedQuizSubmission) => {
    const saved = await writeQueuedQuizSubmission(next)
    setQueuedSubmission(next)
    setSubmissionQueueStorageUnavailable(!saved)
    return saved
  }, [])

  const clearCurrentQueuedSubmission = React.useCallback(async (attemptId: number | null | undefined) => {
    if (attemptId == null) return
    await clearQueuedQuizSubmission(attemptId)
    setQueuedSubmission((current) => (current?.attemptId === attemptId ? null : current))
    setSubmissionQueueStorageUnavailable(false)
  }, [])

  const retryQueuedSubmission = React.useCallback(async (source: "manual" | "online") => {
    if (!queuedSubmission || submitAttemptMutation.isPending || isRetryingQueuedSubmission) return
    setIsRetryingQueuedSubmission(true)
    try {
      const submission = await submitAttemptMutation.mutateAsync({
        attemptId: queuedSubmission.attemptId,
        answers: queuedSubmission.answers
      })
      await clearCurrentQueuedSubmission(queuedSubmission.attemptId)
      if (attempt?.id === queuedSubmission.attemptId) {
        setResult(submission)
        setUnansweredQuestionNumbers([])
        await clearSavedProgress()
      }
      messageApi.success(
        source === "online"
          ? t("option:quiz.submitRetryAutoSuccess", {
              defaultValue: "Connection restored. Queued quiz submission sent."
            })
          : t("option:quiz.submitRetrySuccess", {
              defaultValue: "Submission retry succeeded."
            })
      )
    } catch (error) {
      const updated: QueuedQuizSubmission = {
        ...queuedSubmission,
        retryCount: queuedSubmission.retryCount + 1,
        lastAttemptedAt: Date.now(),
        lastError: getSubmissionErrorMessage(error)
      }
      await persistQueuedSubmission(updated)
      messageApi.error(
        source === "online"
          ? t("option:quiz.submitRetryAutoFailed", {
              defaultValue: "Auto-retry failed. Your answers are still queued locally."
            })
          : t("option:quiz.submitRetryFailed", {
              defaultValue: "Retry failed. Your answers are still queued locally."
            })
      )
    } finally {
      setIsRetryingQueuedSubmission(false)
    }
  }, [
    attempt?.id,
    clearCurrentQueuedSubmission,
    clearSavedProgress,
    getSubmissionErrorMessage,
    isRetryingQueuedSubmission,
    messageApi,
    persistQueuedSubmission,
    queuedSubmission,
    submitAttemptMutation,
    t
  ])

  const resetSession = () => {
    setAttempt(null)
    setResult(null)
    setQuestions([])
    setAnswers({})
    setHintUsedByQuestionId({})
    setPracticeFeedbackByQuestionId({})
    setPracticeQuestionTimerRemainingSeconds(null)
    setStudySessionShuffleSeed(null)
    setQueuedSubmission(null)
    setIsRetryingQueuedSubmission(false)
    setSubmissionQueueStorageUnavailable(false)
    setFocusedQuestionId(null)
    setUnansweredQuestionNumbers([])
    setActiveSessionMode(null)
    setStudySessionLoading(false)
    lastExpiredPracticeQuestionIdRef.current = null
    setActiveQuizId(null)
    setPendingQuizId(null)
  }

  const handleStart = async (quizId: number): Promise<boolean> => {
    try {
      setActiveQuizId(quizId)
      setActiveSessionMode("graded")
      setResult(null)
      setAnswers({})
      setHintUsedByQuestionId({})
      setPracticeFeedbackByQuestionId({})
      setPracticeQuestionTimerRemainingSeconds(null)
      setStudySessionShuffleSeed(null)
      setQueuedSubmission(null)
      setIsRetryingQueuedSubmission(false)
      setSubmissionQueueStorageUnavailable(false)
      setFocusedQuestionId(null)
      setUnansweredQuestionNumbers([])
      setStudySessionLoading(false)
      const newAttempt = await startAttemptMutation.mutateAsync(quizId)
      const newQuestions = newAttempt.questions ?? []
      if (newQuestions.length === 0) {
        setAttempt(null)
        setQuestions([])
        setActiveSessionMode(null)
        setActiveQuizId(null)
        messageApi.error(
          t("option:quiz.noQuestionsToStart", {
            defaultValue: "This quiz has no questions yet."
          })
        )
        return false
      }
      setAttempt(newAttempt)
      setQuestions(newQuestions)
      return true
    } catch (error) {
      setActiveSessionMode(null)
      messageApi.error(
        t("option:quiz.startError", { defaultValue: "Failed to start quiz" })
      )
      return false
    }
  }

  const handleStartStudySession = async (
    quizId: number,
    mode: Exclude<TakeSessionMode, "graded">
  ): Promise<boolean> => {
    try {
      setStudySessionLoading(true)
      setActiveQuizId(quizId)
      setActiveSessionMode(mode)
      setAttempt(null)
      setResult(null)
      setAnswers({})
      setHintUsedByQuestionId({})
      setPracticeFeedbackByQuestionId({})
      setPracticeQuestionTimerRemainingSeconds(null)
      setStudySessionShuffleSeed(null)
      setQueuedSubmission(null)
      setIsRetryingQueuedSubmission(false)
      setSubmissionQueueStorageUnavailable(false)
      setFocusedQuestionId(null)
      setUnansweredQuestionNumbers([])
      setPendingQuizId(null)

      const response = await listQuestions(quizId, {
        include_answers: true,
        limit: 500,
        offset: 0
      })
      const loadedQuestions = (response.items ?? []) as QuestionPublic[]
      if (loadedQuestions.length === 0) {
        setQuestions([])
        setActiveSessionMode(null)
        setActiveQuizId(null)
        setStudySessionShuffleSeed(null)
        messageApi.error(
          t("option:quiz.noQuestionsToStart", {
            defaultValue: "This quiz has no questions yet."
          })
        )
        return false
      }
      const generatedSeed = (Date.now() ^ Math.imul(quizId, 2654435761)) >>> 0
      const candidateSeed = studyPoolSeedOverride ?? generatedSeed
      const nextSessionSeed = (candidateSeed >>> 0) || quizId
      const drawCount = studyPoolSizePreference === "all"
        ? loadedQuestions.length
        : studyPoolSizePreference
      const pooledQuestions = drawDeterministicQuestionPool(
        loadedQuestions,
        drawCount,
        nextSessionSeed
      )
      setStudySessionShuffleSeed(nextSessionSeed === 0 ? quizId : nextSessionSeed)
      setQuestions(pooledQuestions)
      setFocusedQuestionId(pooledQuestions[0]?.id ?? null)
      return true
    } catch {
      setActiveSessionMode(null)
      setStudySessionShuffleSeed(null)
      messageApi.error(
        t("option:quiz.startError", { defaultValue: "Failed to start quiz" })
      )
      return false
    } finally {
      setStudySessionLoading(false)
    }
  }

  const requestGradedStart = React.useCallback((quizId: number) => {
    setModePreference("graded")
    setPendingQuizId(quizId)
  }, [])

  const requestStart = (quizId: number) => {
    if (modePreference === "graded") {
      requestGradedStart(quizId)
      return
    }
    void handleStartStudySession(quizId, modePreference)
  }

  const handleConfirmStart = async () => {
    if (pendingQuizId == null) return
    await handleStart(pendingQuizId)
    setPendingQuizId(null)
  }

  React.useEffect(() => {
    if (startQuizId == null) {
      lastAutoStartId.current = null
      return
    }
    if (lastAutoStartId.current === startQuizId) {
      return
    }
    lastAutoStartId.current = startQuizId
    requestGradedStart(startQuizId)
    onStartHandled?.()
  }, [onStartHandled, requestGradedStart, startQuizId])

  React.useEffect(() => {
    if (highlightQuizId == null) {
      lastAutoHighlightId.current = null
      return
    }
    if (lastAutoHighlightId.current === highlightQuizId) {
      return
    }
    lastAutoHighlightId.current = highlightQuizId
    setHighlightedQuizId(highlightQuizId)
    setSearchQuery("")
    setPage(1)
    onHighlightHandled?.()
  }, [highlightQuizId, onHighlightHandled])

  React.useEffect(() => {
    if (externalSearchToken == null || lastExternalSearchToken.current === externalSearchToken) {
      return
    }
    lastExternalSearchToken.current = externalSearchToken
    setSearchQuery(externalSearchQuery ?? "")
    setPage(1)
    window.setTimeout(() => {
      searchInputRef.current?.focus?.()
    }, 0)
    onExternalSearchHandled?.()
  }, [externalSearchQuery, externalSearchToken, onExternalSearchHandled])

  React.useEffect(() => {
    if (highlightedQuizId == null) return
    const timeoutId = window.setTimeout(() => {
      setHighlightedQuizId(null)
    }, 12000)
    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [highlightedQuizId])

  React.useEffect(() => {
    if (!attempt) return
    void restoreSavedAnswers().then((restored) => {
      if (restored) {
        messageApi.info(
          t("option:quiz.progressRestored", {
            defaultValue: "Restored your saved quiz progress."
          })
        )
      }
    })
  }, [attempt?.id, restoreSavedAnswers, messageApi, t])

  React.useEffect(() => {
    if (!attempt?.id) return
    let cancelled = false
    void readQueuedQuizSubmission(attempt.id).then((queued) => {
      if (cancelled) return
      if (!queued || queued.quizId !== attempt.quiz_id) {
        setQueuedSubmission(null)
        return
      }
      setQueuedSubmission(queued)
    })
    return () => {
      cancelled = true
    }
  }, [attempt?.id, attempt?.quiz_id])

  React.useEffect(() => {
    const becameOnline = !wasOnlineRef.current && isOnline
    wasOnlineRef.current = isOnline
    if (!becameOnline || !queuedSubmission) return
    messageApi.info(
      t("option:quiz.submitRetryingWhenOnline", {
        defaultValue: "Connection restored. Retrying queued quiz submission..."
      })
    )
    void retryQueuedSubmission("online")
  }, [isOnline, queuedSubmission, retryQueuedSubmission, messageApi, t])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.sessionStorage.setItem(
        TAKE_QUIZ_LIST_PREFS_KEY,
        JSON.stringify({
          page,
          pageSize,
          searchQuery,
          sortBy,
          modePreference,
          studyPoolSize: studyPoolSizePreference,
          studyPoolSeedOverride,
          practiceQuestionTimerSeconds: practiceQuestionTimerPreference
        })
      )
    } catch {
      // ignore sessionStorage write errors
    }
  }, [
    modePreference,
    page,
    pageSize,
    practiceQuestionTimerPreference,
    searchQuery,
    sortBy,
    studyPoolSeedOverride,
    studyPoolSizePreference
  ])

  React.useEffect(() => {
    if (!hasInitializedSearchRef.current) {
      hasInitializedSearchRef.current = true
      return
    }
    setPage(1)
  }, [normalizedSearchQuery])

  React.useEffect(() => {
    if (
      activeSessionMode !== "practice" ||
      practiceQuestionTimerPreference === "off" ||
      focusedQuestionId == null ||
      questions.length === 0
    ) {
      setPracticeQuestionTimerRemainingSeconds(null)
      return
    }
    setPracticeQuestionTimerRemainingSeconds(practiceQuestionTimerPreference)
    lastExpiredPracticeQuestionIdRef.current = null
  }, [activeSessionMode, focusedQuestionId, practiceQuestionTimerPreference, questions.length])

  React.useEffect(() => {
    if (
      activeSessionMode !== "practice" ||
      practiceQuestionTimerPreference === "off" ||
      practiceQuestionTimerRemainingSeconds == null ||
      practiceQuestionTimerRemainingSeconds <= 0
    ) {
      return
    }
    const intervalId = window.setInterval(() => {
      setPracticeQuestionTimerRemainingSeconds((previous) => {
        if (previous == null || previous <= 0) return 0
        return previous - 1
      })
    }, 1000)
    return () => {
      window.clearInterval(intervalId)
    }
  }, [activeSessionMode, practiceQuestionTimerPreference, practiceQuestionTimerRemainingSeconds])

  React.useEffect(() => {
    if (
      activeSessionMode !== "practice" ||
      practiceQuestionTimerPreference === "off" ||
      practiceQuestionTimerRemainingSeconds !== 0 ||
      focusedQuestionId == null
    ) {
      return
    }
    if (lastExpiredPracticeQuestionIdRef.current === focusedQuestionId) {
      return
    }
    lastExpiredPracticeQuestionIdRef.current = focusedQuestionId

    const currentIndex = questions.findIndex((question) => question.id === focusedQuestionId)
    if (currentIndex < 0) return
    const nextQuestion = questions[currentIndex + 1]
    if (!nextQuestion) {
      messageApi.warning(
        t("option:quiz.practiceTimerExpiredLastQuestion", {
          defaultValue: "Question timer expired for the final question."
        })
      )
      return
    }

    setFocusedQuestionId(nextQuestion.id)
    window.requestAnimationFrame(() => {
      questionRefs.current.get(nextQuestion.id)?.scrollIntoView({
        behavior: "smooth",
        block: "center"
      })
    })
    messageApi.warning(
      t("option:quiz.practiceTimerAutoAdvance", {
        defaultValue: "Question timer expired. Moving to question {{number}}.",
        number: currentIndex + 2
      })
    )
  }, [
    activeSessionMode,
    focusedQuestionId,
    messageApi,
    practiceQuestionTimerPreference,
    practiceQuestionTimerRemainingSeconds,
    questions,
    t
  ])

  const hasAnswerValue = (value: AnswerValue | null | undefined) => {
    if (value === null || value === undefined) return false
    if (typeof value === "string") return value.trim().length > 0
    if (Array.isArray(value)) return value.length > 0
    return true
  }

  const getQuestionCorrectAnswer = (question: QuestionPublic): AnswerValue | undefined => {
    const raw = (question as QuestionPublic & { correct_answer?: AnswerValue | null }).correct_answer
    return raw ?? undefined
  }

  const getQuestionHint = (question: QuestionPublic): string | null => {
    const raw = (question as QuestionPublic & { hint?: string | null }).hint
    if (typeof raw !== "string") return null
    const trimmed = raw.trim()
    return trimmed.length > 0 ? trimmed : null
  }

  const getQuestionHintPenalty = (question: QuestionPublic): number => {
    const raw = (question as QuestionPublic & { hint_penalty_points?: number | null }).hint_penalty_points
    if (typeof raw !== "number" || !Number.isFinite(raw)) return 0
    return Math.max(0, Math.trunc(raw))
  }

  const getQuestionSourceCitations = (question: QuestionPublic) => {
    const raw = (question as QuestionPublic & { source_citations?: unknown }).source_citations
    return Array.isArray(raw) ? raw : null
  }

  const isPracticeAnswerCorrect = (
    question: QuestionPublic,
    answer: AnswerValue | null | undefined
  ): boolean | null => {
    if (!hasAnswerValue(answer)) return null
    const correct = getQuestionCorrectAnswer(question)
    if (!hasAnswerValue(correct)) return null

    if (question.question_type === "multiple_choice") {
      const userIndex = Number(answer)
      const correctIndex = Number(correct)
      if (!Number.isFinite(userIndex) || !Number.isFinite(correctIndex)) return null
      return userIndex === correctIndex
    }
    if (question.question_type === "multi_select") {
      const userIndices = normalizeMultiSelectAnswer(answer)
      const correctIndices = normalizeMultiSelectAnswer(correct)
      if (userIndices.length === 0 || correctIndices.length === 0) return null
      return userIndices.join(",") === correctIndices.join(",")
    }
    if (question.question_type === "matching") {
      return isMatchingAnswerCorrect(answer, correct)
    }

    if (question.question_type === "true_false") {
      return String(answer).trim().toLowerCase() === String(correct).trim().toLowerCase()
    }

    return isFillBlankAnswerCorrect(answer, correct)
  }

  const updateAnswer = (questionId: number, value: AnswerValue) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }))
    setFocusedQuestionId(questionId)
    setUnansweredQuestionNumbers([])
    if (activeSessionMode !== "practice") return
    const targetQuestion = questions.find((question) => question.id === questionId)
    if (!targetQuestion) return
    setPracticeFeedbackByQuestionId((prev) => ({
      ...prev,
      [questionId]: isPracticeAnswerCorrect(targetQuestion, value)
    }))
  }

  const hasMatchingAnswer = (question: QuestionPublic, value: AnswerValue | null | undefined) => {
    const rows = buildMatchingPairs(question.options, getQuestionCorrectAnswer(question))
    if (rows.length === 0) return false
    const map = normalizeMatchingAnswerMap(value)
    return rows.every(({ left }) => typeof map[left] === "string" && map[left].trim().length > 0)
  }

  const hasAnswer = (questionId: number) => {
    const question = questions.find((entry) => entry.id === questionId)
    if (!question) return false
    const value = answers[questionId]
    if (question.question_type === "matching") {
      return hasMatchingAnswer(question, value)
    }
    return hasAnswerValue(value)
  }

  const answeredCount = questions.filter((q) => hasAnswer(q.id)).length
  const progress = questions.length > 0 ? Math.round((answeredCount / questions.length) * 100) : 0
  const practiceTimerDisplay = React.useMemo(() => {
    if (practiceQuestionTimerRemainingSeconds == null) return null
    const safeSeconds = Math.max(0, practiceQuestionTimerRemainingSeconds)
    const minutes = Math.floor(safeSeconds / 60)
    const seconds = safeSeconds % 60
    return `${minutes}:${seconds.toString().padStart(2, "0")}`
  }, [practiceQuestionTimerRemainingSeconds])
  const optionShuffleSeed = React.useMemo(() => {
    if (activeSessionMode === "graded") {
      return attempt?.id ?? null
    }
    if (activeSessionMode === "practice") {
      return studySessionShuffleSeed
    }
    return null
  }, [activeSessionMode, attempt?.id, studySessionShuffleSeed])

  const getOptionEntriesForQuestion = React.useCallback((question: QuestionPublic): ShuffledOptionEntry[] => {
    if (question.question_type !== "multiple_choice" && question.question_type !== "multi_select") return []
    const options = question.options ?? []
    if (optionShuffleSeed == null || options.length <= 1) {
      return options.map((label, originalIndex) => ({ originalIndex, label }))
    }
    return buildShuffledOptionEntries(options, question.id, optionShuffleSeed)
  }, [optionShuffleSeed])

  const formatQuestionAnswer = React.useCallback((question: QuestionPublic, value: AnswerValue | undefined) => {
    if (value == null) {
      return t("option:quiz.notAnswered", { defaultValue: "Not answered" })
    }
    if (question.question_type === "fill_blank") {
      const accepted = formatFillBlankAcceptedAnswers(value)
      if (accepted.length > 1) {
        return accepted.join(" / ")
      }
      return accepted[0] ?? String(value)
    }
    if (question.question_type === "multi_select") {
      const options = question.options ?? []
      const indices = normalizeMultiSelectAnswer(value)
      if (indices.length === 0) return String(value)
      return indices.map((index) => options[index] ?? String(index)).join(" / ")
    }
    if (question.question_type === "matching") {
      return formatMatchingAnswer(value)
    }
    if (question.question_type !== "multiple_choice") {
      return String(value)
    }
    const options = question.options ?? []
    const index = Number(value)
    if (!Number.isFinite(index)) return String(value)
    return options[index] ?? String(value)
  }, [t])

  const setQuestionRef = React.useCallback((questionId: number, node: HTMLDivElement | null) => {
    questionRefs.current.set(questionId, node)
  }, [])

  const focusQuestion = React.useCallback((questionId: number) => {
    setFocusedQuestionId(questionId)
    questionRefs.current.get(questionId)?.scrollIntoView({
      behavior: "smooth",
      block: "center"
    })
  }, [])

  const handleSubmit = async (options?: { allowPartial?: boolean }) => {
    if (!attempt) return
    if (submitAttemptMutation.isPending || isRetryingQueuedSubmission) return

    const missing = questions.filter((q) => !hasAnswer(q.id))
    if (!options?.allowPartial && missing.length > 0) {
      const missingQuestionNumbers = missing
        .map((question) => questions.findIndex((item) => item.id === question.id) + 1)
        .filter((num) => num > 0)
      setUnansweredQuestionNumbers(missingQuestionNumbers)
      focusQuestion(missing[0].id)
      messageApi.warning(
        t("option:quiz.answerAll", {
          defaultValue: "Please answer all questions before submitting. Unanswered: {{numbers}}",
          numbers: missingQuestionNumbers.join(", ")
        })
      )
      return
    }
    if (options?.allowPartial && missing.length > 0) {
      messageApi.warning(
        t("option:quiz.timerExpiredSubmitting", {
          defaultValue: "Time expired. Submitting your current answers."
        })
      )
    }
    const payload: QuizAnswerInput[] = questions
      .filter((question) => hasAnswer(question.id))
      .map((question) => ({
        question_id: question.id,
        user_answer: answers[question.id],
        hint_used: Boolean(hintUsedByQuestionId[question.id])
      }))

    try {
      const submission = await submitAttemptMutation.mutateAsync({
        attemptId: attempt.id,
        answers: payload
      })
      await clearCurrentQueuedSubmission(attempt.id)
      setResult(submission)
      setUnansweredQuestionNumbers([])
      await clearSavedProgress()
    } catch (error) {
      const previousQueue = queuedSubmission?.attemptId === attempt.id ? queuedSubmission : null
      const didPersist = await persistQueuedSubmission({
        attemptId: attempt.id,
        quizId: attempt.quiz_id,
        answers: payload,
        allowPartial: Boolean(options?.allowPartial),
        queuedAt: previousQueue?.queuedAt ?? Date.now(),
        retryCount: previousQueue ? previousQueue.retryCount + 1 : 0,
        lastAttemptedAt: Date.now(),
        lastError: getSubmissionErrorMessage(error)
      })
      messageApi.error(
        options?.allowPartial
          ? t("option:quiz.submitQueuedForRetryOnTimerExpire", {
            defaultValue:
              "Your time expired and submission failed. We'll retry automatically when connection is restored."
          })
          : didPersist
            ? t("option:quiz.submitQueuedForRetry", {
              defaultValue:
                "Submission failed. Your answers are saved locally. Use Retry to submit when reconnected."
            })
            : t("option:quiz.submitQueuedStorageUnavailable", {
              defaultValue:
                "Submission failed and local retry storage is unavailable. Keep this tab open and retry again."
            })
      )
    }
  }

  const revealHint = (questionId: number) => {
    setHintUsedByQuestionId((prev) => ({ ...prev, [questionId]: true }))
  }

  const renderHintSupport = (question: QuestionPublic) => {
    const hint = getQuestionHint(question)
    if (!hint) return null
    const hintPenaltyPoints = getQuestionHintPenalty(question)
    const hintUsed = Boolean(hintUsedByQuestionId[question.id])
    return (
      <div className="space-y-2">
        {!hintUsed && (
          <Button
            size="small"
            icon={<QuestionCircleOutlined />}
            onClick={() => revealHint(question.id)}
            className="min-h-11 px-3"
            aria-label={t("option:quiz.showHintAria", {
              defaultValue: "Show hint for question {{id}}",
              id: question.id
            })}
          >
            {hintPenaltyPoints > 0
              ? t("option:quiz.showHintWithPenalty", {
                defaultValue: "Show Hint (-{{points}} pts)",
                points: hintPenaltyPoints
              })
              : t("option:quiz.showHint", { defaultValue: "Show Hint" })}
          </Button>
        )}
        {hintUsed && (
          <Alert
            type="info"
            showIcon
            title={t("option:quiz.hint", { defaultValue: "Hint" })}
            description={(
              <div className="space-y-1">
                <QuizMarkdown content={hint} className="text-sm text-text-subtle [&>p]:my-1" />
                {hintPenaltyPoints > 0 && (
                  <Typography.Text className="block text-xs text-text-muted">
                    {t("option:quiz.hintPenaltyAppliedOnCorrect", {
                      defaultValue:
                        "If your answer is correct, {{points}} point(s) will be deducted for using this hint.",
                      points: hintPenaltyPoints
                    })}
                  </Typography.Text>
                )}
              </div>
            )}
          />
        )}
      </div>
    )
  }

  const renderAnswerInput = (question: QuestionPublic) => {
    if (question.question_type === "multiple_choice") {
      const optionEntries = getOptionEntriesForQuestion(question)
      return (
        <fieldset className="border-0 m-0 p-0">
          <legend className="sr-only">{question.question_text}</legend>
          <Radio.Group
            value={answers[question.id]}
            onChange={(e) => updateAnswer(question.id, e.target.value)}
            aria-label={question.question_text}
          >
            <Space orientation="vertical">
              {optionEntries.map((entry, displayIndex) => (
                <Radio key={entry.originalIndex} value={entry.originalIndex}>
                  {entry.label || `${t("option:quiz.option", { defaultValue: "Option" })} ${displayIndex + 1}`}
                </Radio>
              ))}
            </Space>
          </Radio.Group>
        </fieldset>
      )
    }
    if (question.question_type === "multi_select") {
      const optionEntries = getOptionEntriesForQuestion(question)
      const selectedValues = normalizeMultiSelectAnswer(answers[question.id])
      return (
        <fieldset className="border-0 m-0 p-0">
          <legend className="sr-only">{question.question_text}</legend>
          <Checkbox.Group
            value={selectedValues}
            onChange={(checkedValues) => {
              const normalized = normalizeMultiSelectAnswer(checkedValues)
              updateAnswer(question.id, normalized)
            }}
            aria-label={question.question_text}
          >
            <Space orientation="vertical">
              {optionEntries.map((entry, displayIndex) => (
                <Checkbox key={entry.originalIndex} value={entry.originalIndex}>
                  {entry.label || `${t("option:quiz.option", { defaultValue: "Option" })} ${displayIndex + 1}`}
                </Checkbox>
              ))}
            </Space>
          </Checkbox.Group>
        </fieldset>
      )
    }
    if (question.question_type === "matching") {
      const rows = buildMatchingPairs(question.options, getQuestionCorrectAnswer(question))
      const rightOptions = Array.from(new Set(rows.map((row) => row.right).filter((value) => value.length > 0)))
      const selectedMap = normalizeMatchingAnswerMap(answers[question.id])
      return (
        <Space orientation="vertical" className="w-full" size={8}>
          {rows.map((row, index) => (
            <div
              key={`${question.id}-match-${index}`}
              className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_auto_1fr] sm:items-center"
            >
              <Typography.Text>{row.left}</Typography.Text>
              <Typography.Text className="hidden text-text-muted sm:block" aria-hidden>
                {"->"}
              </Typography.Text>
              {rightOptions.length > 0 ? (
                <Select
                  value={selectedMap[row.left]}
                  placeholder={t("option:quiz.selectMatch", { defaultValue: "Select match" })}
                  options={rightOptions.map((value) => ({
                    label: value,
                    value
                  }))}
                  onChange={(nextValue) => {
                    const nextMap = { ...selectedMap }
                    if (typeof nextValue === "string" && nextValue.trim()) {
                      nextMap[row.left] = nextValue
                    } else {
                      delete nextMap[row.left]
                    }
                    updateAnswer(question.id, nextMap)
                  }}
                  allowClear
                  className="w-full"
                />
              ) : (
                <Input
                  value={selectedMap[row.left] ?? ""}
                  placeholder={t("option:quiz.matchingRightPlaceholder", {
                    defaultValue: "Matching item"
                  })}
                  onChange={(event) => {
                    const nextMap = { ...selectedMap }
                    const nextValue = event.target.value.trim()
                    if (nextValue) {
                      nextMap[row.left] = nextValue
                    } else {
                      delete nextMap[row.left]
                    }
                    updateAnswer(question.id, nextMap)
                  }}
                />
              )}
            </div>
          ))}
          {rows.length === 0 && (
            <Typography.Text type="secondary">
              {t("option:quiz.matchingNoPairs", {
                defaultValue: "No matching pairs configured for this question."
              })}
            </Typography.Text>
          )}
        </Space>
      )
    }
    if (question.question_type === "true_false") {
      const legendText = t("option:quiz.trueFalseLegend", {
        defaultValue: "True or false for: {{question}}",
        question: question.question_text
      })
      return (
        <fieldset className="border-0 m-0 p-0">
          <legend className="sr-only">{legendText}</legend>
          <Radio.Group
            value={answers[question.id]}
            onChange={(e) => updateAnswer(question.id, e.target.value)}
            aria-label={legendText}
          >
            <Space orientation="vertical">
              <Radio value="true">{t("option:quiz.true", { defaultValue: "True" })}</Radio>
              <Radio value="false">{t("option:quiz.false", { defaultValue: "False" })}</Radio>
            </Space>
          </Radio.Group>
        </fieldset>
      )
    }
    const helperId = `fill-blank-guidance-${question.id}`
    return (
      <Space orientation="vertical" className="w-full" size={4}>
        <Input
          placeholder={t("option:quiz.correctAnswerPlaceholder", {
            defaultValue: "Enter the correct answer..."
          })}
          aria-label={t("option:quiz.answerForQuestion", {
            defaultValue: "Answer for question: {{question}}",
            question: question.question_text
          })}
          aria-describedby={helperId}
          value={typeof answers[question.id] === "string" ? (answers[question.id] as string) : ""}
          onChange={(e) => updateAnswer(question.id, e.target.value)}
        />
        <Typography.Text id={helperId} type="secondary" className="text-xs">
          {t("option:quiz.fillBlankGuidance", {
            defaultValue:
              "Case-insensitive match. Use `answer1 || answer2` for alternates, and `~answer` or `~0.85:answer` when fuzzy matching is intended."
          })}
        </Typography.Text>
      </Space>
    )
  }

  const renderResults = () => {
    if (!result) return null
    const total = result.total_possible || 0
    const score = result.score ?? 0
    const percentage = total > 0 ? Math.round((score / total) * 100) : 0
    const passingScore = quizDetails?.passing_score ?? DEFAULT_PASSING_SCORE
    const passed = percentage >= passingScore
    const answerMap = new Map(result.answers.map((a) => [a.question_id, a]))

    return (
      <div className="space-y-4">
        <Alert
          type={passed ? "success" : "warning"}
          title={t("option:quiz.scoreSummary", {
            defaultValue: "Score: {{score}} / {{total}} ({{percent}}%)",
            score,
            total,
            percent: percentage
          })}
          description={quizDetails?.passing_score != null
            ? t("option:quiz.passingScoreLabel", { defaultValue: "Pass" }) +
              `: ${quizDetails.passing_score}%`
            : t("option:quiz.noPassingScoreDefault", {
              defaultValue: "No passing score set. Using default: {{score}}%.",
              score: DEFAULT_PASSING_SCORE
            })}
          showIcon
        />

        <List
          dataSource={questions}
          renderItem={(question, index) => {
            const answer = answerMap.get(question.id)
            const isCorrect = answer?.is_correct
            const userAnswer = answer?.user_answer
            const correctAnswer = answer?.correct_answer
            const hintUsed = Boolean(answer?.hint_used)
            const hintPenaltyPoints = Math.max(0, Number(answer?.hint_penalty_points ?? 0))
            const pointsAwarded = Number(answer?.points_awarded ?? 0)

            return (
              <List.Item>
                <div className="w-full space-y-2">
                    <div className="flex items-start justify-between gap-3">
                    <div className="font-medium">
                      <span className="block text-xs text-text-muted">
                        {t("option:quiz.questionNumberLabel", {
                          defaultValue: "Question {{number}}",
                          number: index + 1
                        })}
                      </span>
                      <QuizMarkdown content={question.question_text} className="[&>p]:my-1" />
                    </div>
                    <Tag
                      color={isCorrect ? "green" : "red"}
                      icon={isCorrect ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                    >
                      {isCorrect
                        ? t("option:quiz.correct", { defaultValue: "Correct" })
                        : t("option:quiz.incorrect", { defaultValue: "Incorrect" })}
                    </Tag>
                  </div>
                  <div className="text-sm text-text-muted">
                    {t("option:quiz.yourAnswer", { defaultValue: "Your answer" })}:{" "}
                    <span className="font-medium">{formatQuestionAnswer(question, userAnswer)}</span>
                  </div>
                  <div className="text-sm text-text-muted">
                    {t("option:quiz.correctAnswerLabel", { defaultValue: "Correct answer" })}:{" "}
                    <span className="font-medium">{formatQuestionAnswer(question, correctAnswer)}</span>
                  </div>
                  <div className="text-sm text-text-muted">
                    {t("option:quiz.points", { defaultValue: "Points" })}:{" "}
                    <span className="font-medium">
                      {pointsAwarded} / {question.points}
                    </span>
                  </div>
                  {hintUsed && (
                    <Typography.Text className="block text-xs text-text-muted">
                      {hintPenaltyPoints > 0
                        ? t("option:quiz.hintPenaltyResult", {
                          defaultValue: "Hint used (-{{points}} point(s)).",
                          points: hintPenaltyPoints
                        })
                        : t("option:quiz.hintUsedNoPenalty", {
                          defaultValue: "Hint used."
                        })}
                    </Typography.Text>
                  )}
                  {answer?.explanation && (
                    <QuizMarkdown content={answer.explanation} className="text-sm text-text-subtle" />
                  )}
                  <SourceCitations
                    citations={Array.isArray(answer?.source_citations)
                      ? answer?.source_citations
                      : getQuestionSourceCitations(question)}
                    fallbackMediaId={quizDetails?.media_id ?? null}
                  />
                </div>
              </List.Item>
            )
          }}
        />

        <Space>
          <Button onClick={resetSession} className={TOUCH_TARGET_CLASS}>
            {t("option:quiz.backToList", { defaultValue: "Back to list" })}
          </Button>
          <Tooltip
            title={t("option:quiz.retakeBehavior", {
              defaultValue: "Retake uses the same questions. Answer options may be reshuffled."
            })}
          >
              <Button
              type="primary"
              onClick={() => activeQuizId != null && requestGradedStart(activeQuizId)}
              className={TOUCH_TARGET_CLASS}
            >
              {t("option:quiz.retake", { defaultValue: "Retake Quiz" })}
            </Button>
          </Tooltip>
        </Space>
      </div>
    )
  }

  const formatQuizTimeLimit = (timeLimitSeconds: number | null | undefined) => {
    if (!timeLimitSeconds || timeLimitSeconds <= 0) {
      return t("option:quiz.noTimeLimit", { defaultValue: "No time limit" })
    }
    const minutes = Math.floor(timeLimitSeconds / 60)
    const seconds = timeLimitSeconds % 60
    if (minutes === 0) {
      return t("option:quiz.secondsShort", { defaultValue: "{{count}} sec", count: seconds })
    }
    if (seconds === 0) {
      return `${minutes} ${t("option:quiz.minutes", { defaultValue: "min" })}`
    }
    return `${minutes}m ${seconds}s`
  }

  const formatQuizDate = (dateString: string | null | undefined) => {
    if (!dateString) return null
    const parsed = new Date(dateString)
    if (Number.isNaN(parsed.getTime())) return null
    return parsed.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric"
    })
  }

  const getQuizDifficultyLabel = (quiz: Quiz | null | undefined) => {
    const rawDifficulty = (quiz as Quiz & { difficulty?: string | null } | null)?.difficulty
    if (!rawDifficulty) {
      return t("option:quiz.notSpecified", { defaultValue: "Not specified" })
    }
    return `${rawDifficulty.charAt(0).toUpperCase()}${rawDifficulty.slice(1)}`
  }

  const lastScoreByQuizId = React.useMemo(() => {
    const scoreMap = new Map<number, number>()
    const sortedAttempts = [...attempts].sort((left, right) => {
      const leftDate = new Date(left.completed_at ?? left.started_at).getTime()
      const rightDate = new Date(right.completed_at ?? right.started_at).getTime()
      return rightDate - leftDate
    })
    sortedAttempts.forEach((entry) => {
      if (scoreMap.has(entry.quiz_id)) return
      if (!entry.completed_at) return
      if (!entry.total_possible || entry.total_possible <= 0) return
      const percentage = Math.round(((entry.score ?? 0) / entry.total_possible) * 100)
      scoreMap.set(entry.quiz_id, percentage)
    })
    return scoreMap
  }, [attempts])

  const pendingQuiz = React.useMemo(() => {
    if (pendingQuizId == null) return null
    if (quizDetails?.id === pendingQuizId) return quizDetails
    return quizzes.find((quiz) => quiz.id === pendingQuizId) ?? null
  }, [pendingQuizId, quizDetails, quizzes])

  const highlightedQuiz = React.useMemo(() => {
    if (highlightedQuizId == null) return null
    return sortedQuizzes.find((quiz) => quiz.id === highlightedQuizId) ?? null
  }, [highlightedQuizId, sortedQuizzes])

  const highlightNotice = React.useMemo(() => {
    if (!highlightedQuiz) return null
    if (navigationSource === "generate") {
      return t("option:quiz.generatedHighlightNotice", {
        defaultValue: "Quiz generated successfully: {{name}}. It is highlighted below.",
        name: highlightedQuiz.name
      })
    }
    if (navigationSource === "create") {
      return t("option:quiz.createdHighlightNotice", {
        defaultValue: "Quiz created successfully: {{name}}. It is highlighted below.",
        name: highlightedQuiz.name
      })
    }
    if (navigationSource === "results") {
      return t("option:quiz.retakeHighlightNotice", {
        defaultValue: "Retake ready for {{name}}.",
        name: highlightedQuiz.name
      })
    }
    if (navigationSource === "manage") {
      return t("option:quiz.manageHighlightNotice", {
        defaultValue: "{{name}} selected from Manage tab.",
        name: highlightedQuiz.name
      })
    }
    if (navigationSource === "flashcards") {
      return t("option:quiz.flashcardsHighlightNotice", {
        defaultValue: "Quiz ready from Flashcards context: {{name}}.",
        name: highlightedQuiz.name
      })
    }
    if (navigationSource === "assignment") {
      return t("option:quiz.assignmentHighlightNotice", {
        defaultValue: "Shared assignment ready: {{name}}.",
        name: highlightedQuiz.name
      })
    }
    return t("option:quiz.highlightNotice", {
      defaultValue: "Quiz selected: {{name}}.",
      name: highlightedQuiz.name
    })
  }, [highlightedQuiz, navigationSource, t])

  const hasAssignmentContext = assignmentMode === "shared"
  const normalizedAssignmentNote = React.useMemo(() => {
    if (typeof assignmentNote !== "string") return null
    const trimmed = assignmentNote.trim()
    return trimmed.length > 0 ? trimmed : null
  }, [assignmentNote])
  const normalizedAssignedByRole = React.useMemo(() => {
    if (typeof assignedByRole !== "string") return null
    const trimmed = assignedByRole.trim()
    return trimmed.length > 0 ? trimmed : null
  }, [assignedByRole])
  const parsedAssignmentDueAt = React.useMemo(() => {
    if (!assignmentDueAt) return null
    const parsed = new Date(assignmentDueAt)
    if (Number.isNaN(parsed.getTime())) return null
    return parsed
  }, [assignmentDueAt])
  const assignmentDueAtLabel = React.useMemo(() => {
    if (!parsedAssignmentDueAt) return null
    return parsedAssignmentDueAt.toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short"
    })
  }, [parsedAssignmentDueAt])
  const assignmentIsOverdue = React.useMemo(() => {
    if (!parsedAssignmentDueAt) return false
    return parsedAssignmentDueAt.getTime() < Date.now()
  }, [parsedAssignmentDueAt])
  const assignmentContextKey = React.useMemo(
    () => [
      assignmentMode ?? "",
      assignmentDueAt ?? "",
      assignmentNote ?? "",
      assignedByRole ?? "",
      startQuizId ?? "",
      highlightQuizId ?? ""
    ].join("|"),
    [assignmentDueAt, assignmentMode, assignmentNote, assignedByRole, highlightQuizId, startQuizId]
  )

  React.useEffect(() => {
    if (hasAssignmentContext) {
      setAssignmentAlertDismissed(false)
    }
  }, [assignmentContextKey, hasAssignmentContext])

  const modeStartLabel = React.useMemo(() => {
    if (modePreference === "practice") {
      return t("option:quiz.startPractice", { defaultValue: "Start Practice" })
    }
    if (modePreference === "review") {
      return t("option:quiz.openReview", { defaultValue: "Open Review" })
    }
    return t("option:quiz.startQuiz", { defaultValue: "Start Quiz" })
  }, [modePreference, t])

  const renderAssignmentAlert = () => {
    if (!hasAssignmentContext || assignmentAlertDismissed) return null
    const assignmentPastDueMessage = assignmentIsOverdue
      ? t("option:quiz.assignmentPastDue", {
        defaultValue: "This shared assignment is past due."
      })
      : null
    return (
      <Alert
        data-testid="quiz-assignment-alert"
        type={assignmentIsOverdue ? "warning" : "info"}
        showIcon
        closable
        onClose={() => setAssignmentAlertDismissed(true)}
        title={t("option:quiz.assignmentActive", {
          defaultValue: "This quiz was opened from a shared assignment link."
        })}
        description={(
          <div className="space-y-1 text-sm">
            {assignmentPastDueMessage && (
              <Typography.Text className="block">
                {assignmentPastDueMessage}
              </Typography.Text>
            )}
            {assignmentDueAtLabel && (
              <Typography.Text className="block">
                {t("option:quiz.assignmentDueAt", {
                  defaultValue: "Due: {{date}}",
                  date: assignmentDueAtLabel
                })}
              </Typography.Text>
            )}
            {normalizedAssignedByRole && (
              <Typography.Text className="block">
                {t("option:quiz.assignmentAssignedByRole", {
                  defaultValue: "Assigned by role: {{role}}",
                  role: normalizedAssignedByRole
                })}
              </Typography.Text>
            )}
            {normalizedAssignmentNote && (
              <Typography.Text className="block">
                {t("option:quiz.assignmentNoteLabel", {
                  defaultValue: "Note: {{note}}",
                  note: normalizedAssignmentNote
                })}
              </Typography.Text>
            )}
          </div>
        )}
      />
    )
  }

  const renderStartConfirmationModal = () => (
    <Modal
      title={t("option:quiz.readyToBegin", { defaultValue: "Ready to begin?" })}
      open={pendingQuizId != null}
      onCancel={() => setPendingQuizId(null)}
      onOk={handleConfirmStart}
      okText={t("option:quiz.begin", { defaultValue: "Begin Quiz" })}
      cancelText={t("common:cancel", { defaultValue: "Cancel" })}
      confirmLoading={startAttemptMutation.isPending}
      destroyOnHidden
    >
      <div className="space-y-3">
        <Typography.Text className="block text-text-muted">
          {pendingQuiz?.name ?? t("option:quiz.selectedQuiz", { defaultValue: "Selected quiz" })}
        </Typography.Text>
        <Descriptions size="small" bordered column={1}>
          <Descriptions.Item
            label={t("option:quiz.questions", { defaultValue: "Questions" })}
          >
            {pendingQuiz?.total_questions ?? "-"}
          </Descriptions.Item>
          <Descriptions.Item
            label={t("option:quiz.timeLimit", { defaultValue: "Time limit" })}
          >
            {formatQuizTimeLimit(pendingQuiz?.time_limit_seconds)}
          </Descriptions.Item>
          <Descriptions.Item
            label={t("option:quiz.passingScore", { defaultValue: "Passing score" })}
          >
            {pendingQuiz?.passing_score != null
              ? `${pendingQuiz.passing_score}%`
              : `${DEFAULT_PASSING_SCORE}% (${t("option:quiz.defaultLabel", { defaultValue: "default" })})`}
          </Descriptions.Item>
          <Descriptions.Item
            label={t("option:quiz.difficulty", { defaultValue: "Difficulty" })}
          >
            {getQuizDifficultyLabel(pendingQuiz)}
          </Descriptions.Item>
        </Descriptions>
        {hasAssignmentContext && (
          <Alert
            type={assignmentIsOverdue ? "warning" : "info"}
            showIcon
            title={assignmentIsOverdue
              ? t("option:quiz.assignmentPastDue", {
                defaultValue: "This shared assignment is past due."
              })
              : t("option:quiz.assignmentPreflight", {
                defaultValue: "Shared assignment details"
              })}
            description={(
              <div className="space-y-1 text-sm">
                {assignmentDueAtLabel && (
                  <Typography.Text className="block">
                    {t("option:quiz.assignmentDueAt", {
                      defaultValue: "Due: {{date}}",
                      date: assignmentDueAtLabel
                    })}
                  </Typography.Text>
                )}
                {normalizedAssignmentNote && (
                  <Typography.Text className="block">
                    {t("option:quiz.assignmentNoteLabel", {
                      defaultValue: "Note: {{note}}",
                      note: normalizedAssignmentNote
                    })}
                  </Typography.Text>
                )}
              </div>
            )}
          />
        )}
        <Alert
          type="info"
          showIcon
          title={t("option:quiz.retakeBehavior", {
            defaultValue: "Retake uses the same questions. Answer options may be reshuffled."
          })}
        />
      </div>
    </Modal>
  )

  const renderAutoSaveWarning = () => {
    if (!storageUnavailable || autoSaveWarningDismissed) return null
    return (
      <Alert
        type="warning"
        showIcon
        closable
        onClose={() => setAutoSaveWarningDismissed(true)}
        title={t("option:quiz.autosaveUnavailable", {
          defaultValue:
            "Auto-save unavailable — your progress won't be preserved if you navigate away."
        })}
      />
    )
  }

  const renderSubmissionQueueAlert = () => {
    if (!queuedSubmission) return null
    const queuedAtLabel = new Date(queuedSubmission.queuedAt).toLocaleTimeString()
    return (
      <Alert
        type="error"
        showIcon
        title={queuedSubmission.allowPartial
          ? t("option:quiz.submitQueueTitleTimer", {
            defaultValue: "Time expired. Submission pending retry."
          })
          : t("option:quiz.submitQueueTitle", {
            defaultValue: "Submission failed. Answers queued locally."
          })}
        description={(
          <div className="space-y-2">
            <Typography.Text className="block text-xs text-text-muted">
              {t("option:quiz.submitQueueMeta", {
                defaultValue:
                  "Queued at {{time}}. Retry attempts: {{count}}.",
                time: queuedAtLabel,
                count: queuedSubmission.retryCount
              })}
            </Typography.Text>
            {queuedSubmission.lastError && (
              <Typography.Text className="block text-xs text-text-muted">
                {t("option:quiz.submitQueueLastError", {
                  defaultValue: "Last error: {{error}}",
                  error: queuedSubmission.lastError
                })}
              </Typography.Text>
            )}
            <Typography.Text className="block text-xs text-text-muted">
              {isOnline
                ? t("option:quiz.submitQueueOnlineHint", {
                  defaultValue:
                    "You're online. Retry now to submit immediately, or we'll retry when connectivity changes."
                })
                : t("option:quiz.submitQueueOfflineHint", {
                  defaultValue:
                    "You're offline. We'll retry automatically when your connection returns."
                })}
            </Typography.Text>
            {submissionQueueStorageUnavailable && (
              <Typography.Text className="block text-xs text-text-warning">
                {t("option:quiz.submitQueueStorageUnavailable", {
                  defaultValue:
                    "Local queue storage is unavailable in this browser session."
                })}
              </Typography.Text>
            )}
          </div>
        )}
        action={(
          <Button
            size="small"
            type="primary"
            className="min-h-11 px-3"
            onClick={() => {
              void retryQueuedSubmission("manual")
            }}
            loading={submitAttemptMutation.isPending || isRetryingQueuedSubmission}
            disabled={!isOnline || submitAttemptMutation.isPending || isRetryingQueuedSubmission}
          >
            {t("option:quiz.retrySubmission", { defaultValue: "Retry submission" })}
          </Button>
        )}
      />
    )
  }

  const renderMobileTimerBar = () => {
    if (!timerState) return null
    const timerBarClass = timerState.isDanger
      ? "border-red-300 bg-red-50 text-red-700"
      : timerState.isWarning
        ? "border-amber-300 bg-amber-50 text-amber-700"
        : "border-border bg-background text-text"

    return (
      <div
        data-testid="quiz-mobile-timer-bar"
        className={`sticky z-20 -mx-1 rounded-md border px-3 py-2 shadow-sm md:hidden ${timerBarClass}`}
        style={{ top: "max(env(safe-area-inset-top), 0px)" }}
      >
        <div className="flex items-center justify-between gap-2">
          <div className="text-xs font-medium uppercase tracking-wide">
            {t("option:quiz.timeRemaining", { defaultValue: "Time Remaining" })}
          </div>
          <div
            className="text-sm font-semibold"
            aria-label={t("option:quiz.timerDisplayAria", {
              defaultValue: "Time remaining: {{time}}",
              time: timerState.formattedTime
            })}
          >
            {timerState.formattedTime}
          </div>
        </div>
      </div>
    )
  }

  const renderLiveRegions = () => (
    <>
      {timerAnnouncement && (
        <div
          className="sr-only"
          aria-live={timerState?.isDanger ? "assertive" : "polite"}
          aria-atomic="true"
        >
          {timerAnnouncement}
        </div>
      )}
      {statusAnnouncement && (
        <div className="sr-only" aria-live="polite" aria-atomic="true">
          {statusAnnouncement}
        </div>
      )}
    </>
  )

  if (isLoading) {
    return (
      <div className="space-y-4 py-2" data-testid="take-loading-skeleton">
        {contextHolder}
        {renderLiveRegions()}
        {renderAssignmentAlert()}
        {renderStartConfirmationModal()}
        <Card size="small">
          <Skeleton active paragraph={{ rows: 2 }} />
        </Card>
        <Card size="small">
          <Skeleton active paragraph={{ rows: 6 }} />
        </Card>
      </div>
    )
  }

  if (attempt && result) {
    return (
      <div className="space-y-4">
        {contextHolder}
        {renderLiveRegions()}
        {renderAssignmentAlert()}
        {renderAutoSaveWarning()}
        {renderSubmissionQueueAlert()}
        {renderStartConfirmationModal()}
        {renderResults()}
      </div>
    )
  }

  if ((activeSessionMode === "practice" || activeSessionMode === "review") && studySessionLoading) {
    return (
      <div className="space-y-4 py-2" data-testid="take-study-loading-skeleton">
        {contextHolder}
        {renderLiveRegions()}
        {renderAssignmentAlert()}
        {renderStartConfirmationModal()}
        <Card size="small">
          <Skeleton active paragraph={{ rows: 6 }} />
        </Card>
      </div>
    )
  }

  if (activeSessionMode === "review" && activeQuizId != null && questions.length > 0) {
    return (
      <div className="space-y-4">
        {contextHolder}
        {renderLiveRegions()}
        {renderAssignmentAlert()}
        {renderStartConfirmationModal()}
        <Card
          title={quizDetails?.name || t("option:quiz.reviewMode", { defaultValue: "Review Mode" })}
          extra={<Tag color="blue">{t("option:quiz.reviewMode", { defaultValue: "Review Mode" })}</Tag>}
        >
          <div className="space-y-4">
            <Alert
              type="info"
              showIcon
              title={t("option:quiz.reviewModeHint", {
                defaultValue:
                  "Review mode is read-only. No graded attempt is created and no score is recorded."
              })}
              description={studyPoolSizePreference !== "all"
                ? t("option:quiz.poolSessionSummary", {
                  defaultValue: "Showing {{shown}} questions from a randomized pool.",
                  shown: questions.length
                })
                : undefined}
            />
            <List
              dataSource={questions}
              renderItem={(question, index) => {
                const correctAnswer = getQuestionCorrectAnswer(question)
                return (
                  <List.Item>
                    <div className="w-full space-y-2">
                      <div className="font-medium">
                        <span className="block text-xs text-text-muted">
                          {t("option:quiz.questionNumberLabel", {
                            defaultValue: "Question {{number}}",
                            number: index + 1
                          })}
                        </span>
                        <QuizMarkdown content={question.question_text} className="[&>p]:my-1" />
                      </div>
                      {renderHintSupport(question)}
                      <Typography.Text className="text-sm text-text-muted block">
                        {t("option:quiz.correctAnswerLabel", { defaultValue: "Correct answer" })}:{" "}
                        <span className="font-medium">{formatQuestionAnswer(question, correctAnswer)}</span>
                      </Typography.Text>
                      {(question as QuestionPublic & { explanation?: string | null }).explanation && (
                        <QuizMarkdown
                          content={(question as QuestionPublic & { explanation?: string | null }).explanation || ""}
                          className="text-sm text-text-subtle"
                        />
                      )}
                      <SourceCitations
                        citations={getQuestionSourceCitations(question)}
                        fallbackMediaId={quizDetails?.media_id ?? null}
                      />
                    </div>
                  </List.Item>
                )
              }}
            />
            <Space>
              <Button onClick={resetSession} className={TOUCH_TARGET_CLASS}>
                {t("option:quiz.backToList", { defaultValue: "Back to list" })}
              </Button>
              <Button
                type="primary"
                onClick={() => requestGradedStart(activeQuizId)}
                className={TOUCH_TARGET_CLASS}
              >
                {t("option:quiz.startQuiz", { defaultValue: "Start Quiz" })}
              </Button>
            </Space>
          </div>
        </Card>
      </div>
    )
  }

  if (activeSessionMode === "practice" && activeQuizId != null && questions.length > 0) {
    return (
      <div className="space-y-4">
        {contextHolder}
        {renderLiveRegions()}
        {renderAssignmentAlert()}
        {renderStartConfirmationModal()}
        <Card
          title={quizDetails?.name || t("option:quiz.practiceMode", { defaultValue: "Practice Mode" })}
          extra={
            <Space>
              <Tag color="processing">
                {t("option:quiz.practiceMode", { defaultValue: "Practice Mode" })}
              </Tag>
              <Tag icon={<QuestionCircleOutlined />}>
                {answeredCount}/{questions.length}
              </Tag>
              {practiceQuestionTimerPreference !== "off" && practiceTimerDisplay && (
                <Tag
                  color={practiceQuestionTimerRemainingSeconds != null && practiceQuestionTimerRemainingSeconds <= 10 ? "red" : "gold"}
                  aria-label={t("option:quiz.practiceTimerAria", {
                    defaultValue: "Question timer remaining: {{time}}",
                    time: practiceTimerDisplay
                  })}
                >
                  {t("option:quiz.practiceTimerTag", {
                    defaultValue: "Question timer: {{time}}",
                    time: practiceTimerDisplay
                  })}
                </Tag>
              )}
            </Space>
          }
        >
          <div className="space-y-4">
            <Alert
              type="info"
              showIcon
              title={t("option:quiz.practiceModeHint", {
                defaultValue:
                  "Practice mode gives immediate feedback after each answer and does not create a graded attempt."
              })}
              description={(
                <div className="space-y-1">
                  {studyPoolSizePreference !== "all" && (
                    <Typography.Text className="block text-sm">
                      {t("option:quiz.poolSessionSummary", {
                        defaultValue: "Showing {{shown}} questions from a randomized pool.",
                        shown: questions.length
                      })}
                    </Typography.Text>
                  )}
                  {practiceQuestionTimerPreference !== "off" && (
                    <Typography.Text className="block text-sm">
                      {t("option:quiz.practiceTimerHint", {
                        defaultValue:
                          "Per-question timer is enabled ({{seconds}}s). Focus auto-advances when time expires.",
                        seconds: practiceQuestionTimerPreference
                      })}
                    </Typography.Text>
                  )}
                </div>
              )}
            />
            <Card size="small">
              <div className="space-y-2">
                <Typography.Text className="text-xs text-text-muted">
                  {t("option:quiz.questionNavigator", { defaultValue: "Question navigator" })}
                </Typography.Text>
                <div className="flex flex-wrap gap-2">
                  {questions.map((question, index) => {
                    const answered = hasAnswer(question.id)
                    const focused = focusedQuestionId === question.id
                    return (
                      <Button
                        key={question.id}
                        size="middle"
                        type={focused ? "primary" : "default"}
                        danger={!answered}
                        onClick={() => focusQuestion(question.id)}
                        className="min-h-11 min-w-11 px-3"
                        aria-label={t("option:quiz.goToQuestion", {
                          defaultValue: "Go to question {{number}}",
                          number: index + 1
                        })}
                      >
                        {index + 1}
                      </Button>
                    )
                  })}
                </div>
              </div>
            </Card>
            <div
              role="progressbar"
              aria-label={t("option:quiz.progressAria", { defaultValue: "Quiz completion progress" })}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progress)}
            >
              <Progress percent={progress} />
            </div>
            <List
              dataSource={questions}
              renderItem={(question, index) => {
                const correctAnswer = getQuestionCorrectAnswer(question)
                const feedback = practiceFeedbackByQuestionId[question.id]
                return (
                  <List.Item>
                    <div
                      ref={(node) => setQuestionRef(question.id, node)}
                      data-testid={`quiz-question-${question.id}`}
                      data-highlighted={focusedQuestionId === question.id}
                      className={`w-full space-y-2 rounded-md p-2 transition-colors ${
                        focusedQuestionId === question.id ? "border border-warning/60 bg-warning/10" : ""
                      }`}
                    >
                      <div className="font-medium">
                        <span className="block text-xs text-text-muted">
                          {t("option:quiz.questionNumberLabel", {
                            defaultValue: "Question {{number}}",
                            number: index + 1
                          })}
                        </span>
                        <QuizMarkdown content={question.question_text} className="[&>p]:my-1" />
                      </div>
                      {renderAnswerInput(question)}
                      {renderHintSupport(question)}
                      {feedback != null && (
                        <Alert
                          type={feedback ? "success" : "error"}
                          showIcon
                          title={feedback
                            ? t("option:quiz.correct", { defaultValue: "Correct" })
                            : t("option:quiz.incorrect", { defaultValue: "Incorrect" })}
                          description={!feedback
                            ? (
                              <div className="space-y-1">
                                <Typography.Text className="block text-sm">
                                  {t("option:quiz.correctAnswerLabel", {
                                    defaultValue: "Correct answer"
                                  })}: {formatQuestionAnswer(question, correctAnswer)}
                                </Typography.Text>
                                {(question as QuestionPublic & { explanation?: string | null }).explanation && (
                                  <QuizMarkdown
                                    content={(question as QuestionPublic & { explanation?: string | null }).explanation || ""}
                                    className="text-sm text-text-subtle"
                                  />
                                )}
                                <SourceCitations
                                  citations={getQuestionSourceCitations(question)}
                                  fallbackMediaId={quizDetails?.media_id ?? null}
                                />
                              </div>
                            )
                            : undefined}
                        />
                      )}
                    </div>
                  </List.Item>
                )
              }}
            />
            <Space>
              <Button onClick={resetSession} className={TOUCH_TARGET_CLASS}>
                {t("option:quiz.backToList", { defaultValue: "Back to list" })}
              </Button>
              <Button
                type="primary"
                onClick={() => requestGradedStart(activeQuizId)}
                className={TOUCH_TARGET_CLASS}
              >
                {t("option:quiz.startQuiz", { defaultValue: "Start Quiz" })}
              </Button>
            </Space>
          </div>
        </Card>
      </div>
    )
  }

  if (attempt && questions.length > 0) {
    return (
      <div className="space-y-4">
        {contextHolder}
        {renderLiveRegions()}
        {renderAssignmentAlert()}
        {renderAutoSaveWarning()}
        {renderSubmissionQueueAlert()}
        {renderStartConfirmationModal()}
        {renderMobileTimerBar()}
        <Card
          title={quizDetails?.name || t("option:quiz.take", { defaultValue: "Take Quiz" })}
          extra={
            <Space>
              <Tag icon={<QuestionCircleOutlined />}>
                {answeredCount}/{questions.length}
              </Tag>
              {timerState && (
                <div className="hidden md:inline-flex">
                  <Tag
                    icon={<ClockCircleOutlined />}
                    color={timerState.isDanger ? "red" : timerState.isWarning ? "orange" : "default"}
                    aria-label={t("option:quiz.timerDisplayAria", {
                      defaultValue: "Time remaining: {{time}}",
                      time: timerState.formattedTime
                    })}
                  >
                    {timerState.formattedTime}
                  </Tag>
                </div>
              )}
            </Space>
          }
        >
          <div className="space-y-4">
            <Card size="small">
              <div className="space-y-2">
                <Typography.Text className="text-xs text-text-muted">
                  {t("option:quiz.questionNavigator", { defaultValue: "Question navigator" })}
                </Typography.Text>
                <div className="flex flex-wrap gap-2">
                  {questions.map((question, index) => {
                    const answered = hasAnswer(question.id)
                    const focused = focusedQuestionId === question.id
                    return (
                      <Button
                        key={question.id}
                        size="middle"
                        type={focused ? "primary" : "default"}
                        danger={!answered}
                        onClick={() => focusQuestion(question.id)}
                        className="min-h-11 min-w-11 px-3"
                        aria-label={t("option:quiz.goToQuestion", {
                          defaultValue: "Go to question {{number}}",
                          number: index + 1
                        })}
                      >
                        {index + 1}
                      </Button>
                    )
                  })}
                </div>
              </div>
            </Card>

            {unansweredQuestionNumbers.length > 0 && (
              <Alert
                type="warning"
                showIcon
                title={t("option:quiz.unansweredSummary", {
                  defaultValue: "Unanswered questions: {{numbers}}",
                  numbers: unansweredQuestionNumbers.join(", ")
                })}
              />
            )}

            <div
              role="progressbar"
              aria-label={t("option:quiz.progressAria", { defaultValue: "Quiz completion progress" })}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progress)}
            >
              <Progress percent={progress} />
            </div>
            <List
              dataSource={questions}
              renderItem={(question, index) => (
                <List.Item>
                  <div
                    ref={(node) => setQuestionRef(question.id, node)}
                    data-testid={`quiz-question-${question.id}`}
                    data-highlighted={focusedQuestionId === question.id}
                    className={`w-full space-y-2 rounded-md p-2 transition-colors ${
                      focusedQuestionId === question.id ? "border border-warning/60 bg-warning/10" : ""
                    }`}
                  >
                    <div className="font-medium">
                      <span className="block text-xs text-text-muted">
                        {t("option:quiz.questionNumberLabel", {
                          defaultValue: "Question {{number}}",
                          number: index + 1
                        })}
                      </span>
                      <QuizMarkdown content={question.question_text} className="[&>p]:my-1" />
                    </div>
                    {renderAnswerInput(question)}
                    {renderHintSupport(question)}
                  </div>
                </List.Item>
              )}
            />
            <Space>
              <Button onClick={resetSession} className={TOUCH_TARGET_CLASS}>
                {t("option:quiz.backToList", { defaultValue: "Back to list" })}
              </Button>
              <Button
                type="primary"
                onClick={() => handleSubmit()}
                loading={submitAttemptMutation.isPending || isRetryingQueuedSubmission}
                disabled={submitAttemptMutation.isPending || isRetryingQueuedSubmission}
                className={TOUCH_TARGET_CLASS}
              >
                {t("common:submit", { defaultValue: "Submit" })}
              </Button>
            </Space>
          </div>
        </Card>
      </div>
    )
  }

  if (quizzes.length === 0) {
    return (
      <>
        {contextHolder}
        {renderLiveRegions()}
        {renderAssignmentAlert()}
        {renderAutoSaveWarning()}
        {renderSubmissionQueueAlert()}
        {renderStartConfirmationModal()}
        <Empty
          description={
            <div className="space-y-2">
              <p className="text-text-muted">
                {t("option:quiz.empty.noQuizzes", {
                  defaultValue: "No quizzes available to take yet"
                })}
              </p>
              <p className="text-sm text-text-subtle">
                {t("option:quiz.empty.createFirstPrefix", {
                  defaultValue: "Generate one from your "
                })}
                <Link to="/media" className="text-primary hover:text-primary/80 underline">
                  {t("option:quiz.empty.mediaLibrary", {
                    defaultValue: "media library"
                  })}
                </Link>
                {t("option:quiz.empty.createFirstSuffix", {
                  defaultValue: " or create one manually, then come back to take it."
                })}
              </p>
            </div>
          }
        >
          <div className="flex gap-2 justify-center">
            <Button type="primary" onClick={onNavigateToGenerate} className={TOUCH_TARGET_CLASS}>
              {t("option:quiz.generateFromMedia", { defaultValue: "Generate from Media" })}
            </Button>
            <Button onClick={onNavigateToCreate} className={TOUCH_TARGET_CLASS}>
              {t("option:quiz.createManually", { defaultValue: "Create Manually" })}
            </Button>
          </div>
        </Empty>
      </>
    )
  }

  return (
    <div className="space-y-4">
      {contextHolder}
      {renderLiveRegions()}
      {renderAssignmentAlert()}
      {renderAutoSaveWarning()}
      {renderSubmissionQueueAlert()}
      {renderStartConfirmationModal()}
      <div className="text-sm text-text-muted">
        {t("option:quiz.selectQuiz", { defaultValue: "Select a quiz to begin" })}
      </div>

      {highlightNotice && (
        <Alert
          type="info"
          showIcon
          title={highlightNotice}
          closable
          onClose={() => setHighlightedQuizId(null)}
        />
      )}

      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <Input.Search
          ref={searchInputRef}
          allowClear
          value={searchQuery}
          onChange={(event) => setSearchQuery(event.target.value)}
          placeholder={t("option:quiz.searchPlaceholder", { defaultValue: "Search quizzes..." })}
          className="w-full md:max-w-md"
        />
        <div className="flex w-full flex-col gap-2 md:w-auto md:flex-row">
          <Select
            className="w-full md:w-56"
            value={modePreference}
            onChange={(nextMode) => setModePreference(nextMode)}
            options={[
              {
                value: "graded",
                label: t("option:quiz.modeGraded", { defaultValue: "Mode: Graded" })
              },
              {
                value: "practice",
                label: t("option:quiz.modePractice", { defaultValue: "Mode: Practice" })
              },
              {
                value: "review",
                label: t("option:quiz.modeReview", { defaultValue: "Mode: Review" })
              }
            ]}
          />
          {modePreference !== "graded" && (
            <Select
              className="w-full md:w-56"
              value={studyPoolSizePreference}
              onChange={(nextValue) => setStudyPoolSizePreference(nextValue)}
              options={[
                {
                  value: "all",
                  label: t("option:quiz.poolAll", { defaultValue: "Question pool: All" })
                },
                {
                  value: 5,
                  label: t("option:quiz.poolFive", { defaultValue: "Question pool: 5" })
                },
                {
                  value: 10,
                  label: t("option:quiz.poolTen", { defaultValue: "Question pool: 10" })
                },
                {
                  value: 20,
                  label: t("option:quiz.poolTwenty", { defaultValue: "Question pool: 20" })
                }
              ]}
            />
          )}
          {modePreference === "practice" && (
            <Select
              className="w-full md:w-56"
              value={practiceQuestionTimerPreference}
              onChange={(nextValue) => setPracticeQuestionTimerPreference(nextValue)}
              options={[
                {
                  value: "off",
                  label: t("option:quiz.practiceTimerOff", { defaultValue: "Question timer: Off" })
                },
                {
                  value: 30,
                  label: t("option:quiz.practiceTimerThirty", { defaultValue: "Question timer: 30s" })
                },
                {
                  value: 60,
                  label: t("option:quiz.practiceTimerSixty", { defaultValue: "Question timer: 60s" })
                },
                {
                  value: 90,
                  label: t("option:quiz.practiceTimerNinety", { defaultValue: "Question timer: 90s" })
                }
              ]}
            />
          )}
          <Select
            className="w-full md:w-56"
            value={sortBy}
            onChange={(nextValue) => setSortBy(nextValue)}
            options={[
              {
                value: "date_desc",
                label: t("option:quiz.sortNewest", { defaultValue: "Newest first" })
              },
              {
                value: "name_asc",
                label: t("option:quiz.sortName", { defaultValue: "Name (A-Z)" })
              },
              {
                value: "questions_desc",
                label: t("option:quiz.sortQuestionCount", { defaultValue: "Most questions" })
              }
            ]}
          />
        </div>
      </div>

      <List
        grid={{ gutter: 16, xs: 1, sm: 2, md: 2, lg: 3, xl: 3, xxl: 4 }}
        dataSource={sortedQuizzes}
        pagination={{
          current: page,
          pageSize,
          total: visibleTotal,
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
        renderItem={(quiz) => {
          const isHighlighted = quiz.id === highlightedQuizId
          const sourceSummary = summarizeQuizSources(quiz)
          return (
            <List.Item>
              <Card
                hoverable
                className="h-full"
                style={isHighlighted
                  ? {
                    borderColor: "var(--color-primary, #1677ff)",
                    boxShadow: "0 0 0 2px rgba(22, 119, 255, 0.2)"
                  }
                  : undefined}
                data-testid={`take-quiz-card-${quiz.id}`}
                data-highlighted={isHighlighted ? "true" : undefined}
                actions={[
                  <Button
                    key="start"
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    className={TOUCH_TARGET_CLASS}
                    loading={studySessionLoading}
                    disabled={startAttemptMutation.isPending || studySessionLoading}
                    onClick={() => {
                      setHighlightedQuizId(null)
                      requestStart(quiz.id)
                    }}
                  >
                    {modeStartLabel}
                  </Button>
                ]}
              >
                <Card.Meta
                  title={quiz.name}
                  description={
                    <div className="space-y-2">
                      {quiz.description && (
                        <p className="text-sm text-text-muted line-clamp-2">
                          {quiz.description}
                        </p>
                      )}
                      <div className="flex flex-wrap gap-2">
                        <Tag icon={<QuestionCircleOutlined />}>
                          {quiz.total_questions}{" "}
                          {t("option:quiz.questions", { defaultValue: "questions" })}
                        </Tag>
                        {quiz.time_limit_seconds && (
                          <Tag icon={<ClockCircleOutlined />}>
                            {formatQuizTimeLimit(quiz.time_limit_seconds)}
                          </Tag>
                        )}
                        {quiz.passing_score != null && (
                          <Tag color="blue">
                            {t("option:quiz.passingScoreLabel", { defaultValue: "Pass" })}: {quiz.passing_score}%
                          </Tag>
                        )}
                        {(quiz as Quiz & { difficulty?: string | null }).difficulty && (
                          <Tag color="purple">
                            {getQuizDifficultyLabel(quiz)}
                          </Tag>
                        )}
                        {sourceSummary.media > 0 && (
                          <Tag color="green" data-testid={`take-quiz-source-media-${quiz.id}`}>
                            {t("option:quiz.sourceBadgeMedia", {
                              defaultValue: "Media {{count}}",
                              count: sourceSummary.media
                            })}
                          </Tag>
                        )}
                        {sourceSummary.notes > 0 && (
                          <Tag color="cyan" data-testid={`take-quiz-source-notes-${quiz.id}`}>
                            {t("option:quiz.sourceBadgeNotes", {
                              defaultValue: "Notes {{count}}",
                              count: sourceSummary.notes
                            })}
                          </Tag>
                        )}
                        {sourceSummary.flashcards > 0 && (
                          <Tag color="magenta" data-testid={`take-quiz-source-flashcards-${quiz.id}`}>
                            {t("option:quiz.sourceBadgeFlashcards", {
                              defaultValue: "Flashcards {{count}}",
                              count: sourceSummary.flashcards
                            })}
                          </Tag>
                        )}
                        {quiz.media_id != null && (
                          <Tag color="green">
                            <Typography.Link href={`/media?id=${quiz.media_id}`}>
                              {t("option:quiz.sourceMedia", { defaultValue: "Source media #{{id}}", id: quiz.media_id })}
                            </Typography.Link>
                          </Tag>
                        )}
                        {lastScoreByQuizId.has(quiz.id) && (
                          <Tag color="geekblue">
                            {t("option:quiz.lastScore", { defaultValue: "Last score: {{score}}%", score: lastScoreByQuizId.get(quiz.id) })}
                          </Tag>
                        )}
                      </div>
                      {formatQuizDate(quiz.created_at) && (
                        <p className="text-xs text-text-subtle">
                          {t("option:quiz.createdOn", { defaultValue: "Created: {{date}}", date: formatQuizDate(quiz.created_at) })}
                        </p>
                      )}
                    </div>
                  }
                />
              </Card>
            </List.Item>
          )
        }}
      />
    </div>
  )
}

export default TakeQuizTab
