import React from "react"
import DOMPurify from "dompurify"
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Divider,
  Empty,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Popconfirm,
  Radio,
  Select,
  Skeleton,
  Space,
  Spin,
  Tag,
  Typography,
  message
} from "antd"
import { useTranslation } from "react-i18next"
import {
  ArrowDownOutlined,
  ArrowUpOutlined,
  CopyOutlined,
  DownloadOutlined,
  DeleteOutlined,
  EditOutlined,
  MinusCircleOutlined,
  PlusOutlined,
  PlayCircleOutlined,
  PrinterOutlined,
  QuestionCircleOutlined,
  SearchOutlined,
  ShareAltOutlined,
  UndoOutlined,
  UploadOutlined
} from "@ant-design/icons"
import {
  useCreateQuizMutation,
  useCreateQuestionMutation,
  useDeleteQuestionMutation,
  useDeleteQuizMutation,
  useQuestionsQuery,
  useQuizzesQuery,
  useUpdateQuestionMutation,
  useUpdateQuizMutation
} from "../hooks"
import { tldwAuth, tldwClient } from "@/services/tldw"
import { importQuizzesJson, listQuestions } from "@/services/quizzes"
import type { AnswerValue, Question, QuestionType, Quiz, SourceCitation } from "@/services/quizzes"
import { normalizeMatchingAnswerMap } from "../utils/matchingAnswer"
import { summarizeQuizSources } from "../utils/sourceBundle"

interface ManageTabProps {
  onNavigateToCreate: () => void
  onNavigateToGenerate: () => void
  onStartQuiz: (quizId: number) => void
  externalSearchQuery?: string | null
  externalSearchToken?: number | null
  onExternalSearchHandled?: () => void
}

export const buildQuizManageQueryParams = ({
  searchQuery,
  page,
  pageSize,
  showWorkspaceQuizzes,
  selectedWorkspaceId
}: {
  searchQuery: string
  page: number
  pageSize: number
  showWorkspaceQuizzes: boolean
  selectedWorkspaceId?: string | null
}) => ({
  q: searchQuery.trim() || undefined,
  limit: pageSize,
  offset: (page - 1) * pageSize,
  workspace_id: selectedWorkspaceId ?? undefined,
  include_workspace_items: selectedWorkspaceId == null ? showWorkspaceQuizzes : false
})

type QuestionDraft = {
  id?: number
  question_type: QuestionType
  question_text: string
  options: string[]
  correct_answer: AnswerValue
  explanation?: string | null
  hint?: string | null
  hint_penalty_points?: number
  points: number
  order_index: number
}

type QuestionValidationErrors = {
  questionText?: string
  options?: string
}

const QUESTION_TEXT_ERROR_ID = "manage-question-text-error"
const QUESTION_OPTIONS_ERROR_ID = "manage-question-options-error"
const MIN_MATCHING_PAIRS = 2
const MAX_MATCHING_PAIRS = 8

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object") return null
  return value as Record<string, unknown>
}

const asNonEmptyString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed ? trimmed : null
}

const getMediaDisplayName = (details: unknown, mediaId: number): string => {
  const record = asRecord(details)
  if (!record) return `Media #${mediaId}`
  return (
    asNonEmptyString(record.title) ??
    asNonEmptyString(record.name) ??
    asNonEmptyString(record.filename) ??
    asNonEmptyString(record.url) ??
    `Media #${mediaId}`
  )
}

type QuizExportEntry = {
  quiz: Quiz
  questions: Question[]
}

type NormalizedImportQuestion = {
  question_type: QuestionType
  question_text: string
  options?: string[]
  correct_answer: AnswerValue
  explanation?: string
  hint?: string
  hint_penalty_points?: number
  source_citations?: SourceCitation[]
  points: number
  order_index: number
  tags?: string[]
}

type QuizImportEntry = {
  quiz: {
    name: string
    description?: string
    workspace_tag?: string
    media_id?: number
    time_limit_seconds?: number
    passing_score?: number
  }
  questions: NormalizedImportQuestion[]
}

const QUIZ_EXPORT_FORMAT = "tldw.quiz.export.v1"
const ASSIGNMENT_PRIVILEGED_ROLES = new Set(["owner", "admin", "lead"])
const SUPPORTED_QUESTION_TYPES: QuestionType[] = [
  "multiple_choice",
  "multi_select",
  "matching",
  "true_false",
  "fill_blank"
]

type ShareAccessState = {
  loading: boolean
  authMode: "single-user" | "multi-user" | null
  role: string | null
  canShareAssignments: boolean
}

const isQuestionType = (value: unknown): value is QuestionType => (
  typeof value === "string" &&
  SUPPORTED_QUESTION_TYPES.includes(value as QuestionType)
)

const toOptionalInteger = (
  value: unknown,
  bounds?: {
    min?: number
    max?: number
  }
): number | undefined => {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined
  const parsed = Math.trunc(value)
  if (typeof bounds?.min === "number" && parsed < bounds.min) return undefined
  if (typeof bounds?.max === "number" && parsed > bounds.max) return undefined
  return parsed
}

const normalizeStringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => String(entry ?? "").trim())
    .filter((entry) => entry.length > 0)
}

const normalizeSourceCitations = (value: unknown): SourceCitation[] | undefined => {
  if (!Array.isArray(value)) return undefined
  const citations = value
    .map((entry) => {
      const record = asRecord(entry)
      if (!record) return null
      const citation: SourceCitation = {}
      const label = asNonEmptyString(record.label)
      if (label) citation.label = label
      const quote = asNonEmptyString(record.quote)
      if (quote) citation.quote = quote
      const chunkId = asNonEmptyString(record.chunk_id)
      if (chunkId) citation.chunk_id = chunkId
      const sourceUrl = asNonEmptyString(record.source_url)
      if (sourceUrl) citation.source_url = sourceUrl
      const mediaId = toOptionalInteger(record.media_id, { min: 1 })
      if (typeof mediaId === "number") citation.media_id = mediaId
      if (typeof record.timestamp_seconds === "number" && Number.isFinite(record.timestamp_seconds)) {
        citation.timestamp_seconds = Math.max(0, Number(record.timestamp_seconds))
      }
      return Object.keys(citation).length > 0 ? citation : null
    })
    .filter((entry): entry is SourceCitation => entry != null)
  return citations.length > 0 ? citations : undefined
}

const normalizeImportQuestion = (
  rawQuestion: unknown,
  fallbackOrderIndex: number
): NormalizedImportQuestion | null => {
  const questionRecord = asRecord(rawQuestion)
  if (!questionRecord) return null

  const questionType = asNonEmptyString(questionRecord.question_type)
  const questionText = asNonEmptyString(questionRecord.question_text)
  if (!questionText || !isQuestionType(questionType)) return null

  const points = toOptionalInteger(questionRecord.points, { min: 0 }) ?? 1
  const orderIndex = toOptionalInteger(questionRecord.order_index, { min: 0 }) ?? fallbackOrderIndex
  const explanation = asNonEmptyString(questionRecord.explanation) ?? undefined
  const hint = asNonEmptyString(questionRecord.hint) ?? undefined
  const hintPenaltyPoints = toOptionalInteger(questionRecord.hint_penalty_points, { min: 0 }) ?? 0
  const sourceCitations = normalizeSourceCitations(questionRecord.source_citations)
  const tags = normalizeStringArray(questionRecord.tags)
  let correctAnswer: AnswerValue = questionRecord.correct_answer as AnswerValue
  let optionsPayload: string[] | undefined

  if (questionType === "multiple_choice" || questionType === "multi_select") {
    const options = normalizeStringArray(questionRecord.options)
    if (options.length < 2) return null
    optionsPayload = options

    if (questionType === "multiple_choice") {
      const parsedIndex = Number(correctAnswer)
      correctAnswer = Number.isFinite(parsedIndex)
        ? Math.max(0, Math.trunc(parsedIndex))
        : 0
    } else {
      const parsedIndexes = Array.isArray(correctAnswer)
        ? correctAnswer
          .map((entry) => Number(entry))
          .filter((entry) => Number.isFinite(entry))
          .map((entry) => Math.max(0, Math.trunc(entry)))
        : []
      correctAnswer = Array.from(new Set(parsedIndexes)).sort((a, b) => a - b)
    }
  } else if (questionType === "matching") {
    const normalizedMap = normalizeMatchingAnswerMap(correctAnswer)
    const optionsFromPayload = normalizeStringArray(questionRecord.options)
    const leftItems = optionsFromPayload.length > 0
      ? optionsFromPayload
      : Object.keys(normalizedMap)
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0)

    if (leftItems.length < MIN_MATCHING_PAIRS) return null

    const normalizedPairs = leftItems
      .map((left) => ({
        left,
        right: String(normalizedMap[left] ?? "").trim()
      }))
      .filter((pair) => pair.left.length > 0 && pair.right.length > 0)

    if (normalizedPairs.length < MIN_MATCHING_PAIRS) return null
    optionsPayload = normalizedPairs.map((pair) => pair.left)
    correctAnswer = Object.fromEntries(normalizedPairs.map((pair) => [pair.left, pair.right]))
  } else if (questionType === "true_false") {
    correctAnswer = String(correctAnswer ?? "true").toLowerCase() === "false" ? "false" : "true"
  } else {
    correctAnswer = String(correctAnswer ?? "").trim()
  }

  return {
    question_type: questionType,
    question_text: questionText,
    options: optionsPayload,
    correct_answer: correctAnswer,
    explanation,
    hint,
    hint_penalty_points: hintPenaltyPoints,
    source_citations: sourceCitations,
    points,
    order_index: orderIndex,
    tags: tags.length > 0 ? tags : undefined
  }
}

const normalizeQuizImportPayload = (rawPayload: unknown): QuizImportEntry[] => {
  const payloadRecord = asRecord(rawPayload)
  if (!payloadRecord || !Array.isArray(payloadRecord.quizzes)) return []

  return payloadRecord.quizzes
    .map((entry) => {
      const entryRecord = asRecord(entry)
      const quizRecord = asRecord(entryRecord?.quiz)
      if (!quizRecord) return null

      const quizName = asNonEmptyString(quizRecord.name)
      if (!quizName) return null

      const questionsRaw = Array.isArray(entryRecord?.questions) ? entryRecord.questions : []
      const questions = questionsRaw
        .map((question, index) => normalizeImportQuestion(question, index))
        .filter((question): question is NormalizedImportQuestion => question != null)

      return {
        quiz: {
          name: quizName,
          description: asNonEmptyString(quizRecord.description) ?? undefined,
          workspace_tag: asNonEmptyString(quizRecord.workspace_tag) ?? undefined,
          media_id: toOptionalInteger(quizRecord.media_id, { min: 1 }),
          time_limit_seconds: toOptionalInteger(quizRecord.time_limit_seconds, { min: 1 }),
          passing_score: toOptionalInteger(quizRecord.passing_score, { min: 0, max: 100 })
        },
        questions
      } as QuizImportEntry
    })
    .filter((entry): entry is QuizImportEntry => entry != null)
}

const escapeHtml = (value: string): string => (
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
)

const PRINTABLE_DOCUMENT_DOCTYPE = "<!doctype html>\n"

const formatPrintableCorrectAnswer = (question: Question): string => {
  if (question.question_type === "multiple_choice") {
    const correctIndex = Number(question.correct_answer)
    const label = Number.isFinite(correctIndex) && Array.isArray(question.options)
      ? question.options[correctIndex]
      : null
    if (typeof label === "string" && label.trim()) {
      return `${correctIndex + 1}. ${label}`
    }
  }

  if (question.question_type === "multi_select") {
    const indexes = Array.isArray(question.correct_answer)
      ? question.correct_answer
        .map((entry) => Number(entry))
        .filter((entry) => Number.isFinite(entry))
        .sort((a, b) => a - b)
      : []
    if (indexes.length > 0 && Array.isArray(question.options)) {
      const labels = indexes
        .map((index) => question.options?.[index])
        .filter((label): label is string => typeof label === "string" && label.trim().length > 0)
      if (labels.length > 0) {
        return labels.map((label, idx) => `${idx + 1}. ${label}`).join(", ")
      }
    }
  }

  if (question.question_type === "matching") {
    const mapping = normalizeMatchingAnswerMap(question.correct_answer)
    return Object.entries(mapping)
      .map(([left, right]) => `${left} -> ${right}`)
      .join("; ")
  }

  if (Array.isArray(question.correct_answer)) {
    return question.correct_answer.map((entry) => String(entry)).join(", ")
  }
  if (question.correct_answer && typeof question.correct_answer === "object") {
    return Object.entries(question.correct_answer as Record<string, string>)
      .map(([key, val]) => `${key}: ${val}`)
      .join("; ")
  }
  return String(question.correct_answer ?? "")
}

const buildPrintableQuizHtml = (entry: QuizExportEntry): string => {
  const questionsMarkup = entry.questions
    .map((question, index) => {
      const optionsMarkup =
        Array.isArray(question.options) && question.options.length > 0
          ? `<ul>${question.options
              .map((option, optionIndex) => `<li>${optionIndex + 1}. ${escapeHtml(option)}</li>`)
              .join("")}</ul>`
          : ""

      const explanationMarkup = question.explanation
        ? `<p><strong>Explanation:</strong> ${escapeHtml(question.explanation)}</p>`
        : ""
      const hintMarkup = question.hint
        ? `<p><strong>Hint:</strong> ${escapeHtml(question.hint)}</p>`
        : ""
      const hintPenaltyMarkup =
        typeof question.hint_penalty_points === "number" && question.hint_penalty_points > 0
          ? `<p><strong>Hint Penalty:</strong> -${question.hint_penalty_points} point(s)</p>`
          : ""

      return `
        <article class="question">
          <h2>${index + 1}. ${escapeHtml(question.question_text)}</h2>
          <p class="meta">Type: ${escapeHtml(question.question_type)} · Points: ${question.points}</p>
          ${optionsMarkup}
          ${hintMarkup}
          ${hintPenaltyMarkup}
          <p><strong>Correct Answer:</strong> ${escapeHtml(formatPrintableCorrectAnswer(question))}</p>
          ${explanationMarkup}
        </article>
      `
    })
    .join("")

  const quizDescription = entry.quiz.description
    ? `<p>${escapeHtml(entry.quiz.description)}</p>`
    : ""
  const timeLimit = typeof entry.quiz.time_limit_seconds === "number"
    ? `<p><strong>Time Limit:</strong> ${Math.round(entry.quiz.time_limit_seconds / 60)} min</p>`
    : ""
  const passingScore = typeof entry.quiz.passing_score === "number"
    ? `<p><strong>Passing Score:</strong> ${entry.quiz.passing_score}%</p>`
    : ""

  const rawDocument = `<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>${escapeHtml(entry.quiz.name)} - Printable Quiz</title>
  <style>
    body { font-family: "Helvetica Neue", Arial, sans-serif; margin: 24px; color: #111827; }
    h1 { margin-bottom: 8px; }
    .summary { margin-bottom: 20px; color: #374151; }
    .question { border-top: 1px solid #d1d5db; padding-top: 12px; margin-top: 12px; break-inside: avoid; }
    .meta { color: #6b7280; font-size: 12px; margin: 4px 0 8px; }
    ul { margin: 8px 0; padding-left: 20px; }
    @media print {
      body { margin: 12mm; }
      .question { page-break-inside: avoid; }
    }
  </style>
</head>
<body>
  <h1>${escapeHtml(entry.quiz.name)}</h1>
  <section class="summary">
    ${quizDescription}
    ${timeLimit}
    ${passingScore}
    <p><strong>Questions:</strong> ${entry.questions.length}</p>
  </section>
  ${questionsMarkup}
</body>
</html>`

  return `${PRINTABLE_DOCUMENT_DOCTYPE}${DOMPurify.sanitize(rawDocument, { WHOLE_DOCUMENT: true })}`
}

export const ManageTab: React.FC<ManageTabProps> = ({
  onNavigateToCreate,
  onNavigateToGenerate,
  onStartQuiz,
  externalSearchQuery,
  externalSearchToken,
  onExternalSearchHandled
}) => {
  const { t } = useTranslation(["option", "common"])
  const [searchQuery, setSearchQuery] = React.useState("")
  const [messageApi, contextHolder] = message.useMessage()
  const [page, setPage] = React.useState(1)
  const [pageSize, setPageSize] = React.useState(10)
  const [showWorkspaceQuizzes, setShowWorkspaceQuizzes] = React.useState(false)
  const [selectedWorkspaceId, setSelectedWorkspaceId] = React.useState<string | null>(null)
  const [editingQuiz, setEditingQuiz] = React.useState<Quiz | null>(null)
  const [editModalOpen, setEditModalOpen] = React.useState(false)
  const [questionModalOpen, setQuestionModalOpen] = React.useState(false)
  const [questionDraft, setQuestionDraft] = React.useState<QuestionDraft | null>(null)
  const [isNewQuestion, setIsNewQuestion] = React.useState(false)
  const [reorderPendingQuestionId, setReorderPendingQuestionId] = React.useState<number | null>(null)
  const [editForm] = Form.useForm()
  const [questionValidationErrors, setQuestionValidationErrors] = React.useState<QuestionValidationErrors>({})
  const editModalTriggerRef = React.useRef<HTMLElement | null>(null)
  const questionModalTriggerRef = React.useRef<HTMLElement | null>(null)
  const lastExternalSearchToken = React.useRef<number | null>(null)
  const searchInputRef = React.useRef<any>(null)

  // Undo deletion state
  const UNDO_GRACE_PERIOD = 8000 // 8 seconds
  const pendingQuizDeletion = React.useRef<{
    quiz: Quiz
    timeoutId: ReturnType<typeof setTimeout>
  } | null>(null)
  const pendingQuestionDeletion = React.useRef<{
    question: Question
    timeoutId: ReturnType<typeof setTimeout>
  } | null>(null)
  const [deletedQuizIds, setDeletedQuizIds] = React.useState<Set<number>>(new Set())
  const [deletedQuestionIds, setDeletedQuestionIds] = React.useState<Set<number>>(new Set())
  const [pendingQuizUndoName, setPendingQuizUndoName] = React.useState<string | null>(null)
  const [pendingQuestionUndoText, setPendingQuestionUndoText] = React.useState<string | null>(null)
  const [selectedQuizIds, setSelectedQuizIds] = React.useState<Set<number>>(new Set())
  const [bulkDeleteInFlight, setBulkDeleteInFlight] = React.useState(false)
  const [bulkExportInFlight, setBulkExportInFlight] = React.useState(false)
  const [importInFlight, setImportInFlight] = React.useState(false)
  const [singleExportInFlightQuizId, setSingleExportInFlightQuizId] = React.useState<number | null>(null)
  const [singlePrintInFlightQuizId, setSinglePrintInFlightQuizId] = React.useState<number | null>(null)
  const [shareInFlightQuizId, setShareInFlightQuizId] = React.useState<number | null>(null)
  const [shareModalOpen, setShareModalOpen] = React.useState(false)
  const [shareTargetQuiz, setShareTargetQuiz] = React.useState<Quiz | null>(null)
  const [shareDueAtLocal, setShareDueAtLocal] = React.useState("")
  const [shareAssignmentNote, setShareAssignmentNote] = React.useState("")
  const [shareAccess, setShareAccess] = React.useState<ShareAccessState>({
    loading: true,
    authMode: null,
    role: null,
    canShareAssignments: false
  })
  const [duplicateInFlightQuizId, setDuplicateInFlightQuizId] = React.useState<number | null>(null)
  const [mediaNameMap, setMediaNameMap] = React.useState<Record<number, string>>({})
  const importInputRef = React.useRef<HTMLInputElement | null>(null)

  // Cleanup pending deletions on unmount
  React.useEffect(() => {
    return () => {
      if (pendingQuizDeletion.current) {
        clearTimeout(pendingQuizDeletion.current.timeoutId)
      }
      if (pendingQuestionDeletion.current) {
        clearTimeout(pendingQuestionDeletion.current.timeoutId)
      }
    }
  }, [])

  React.useEffect(() => {
    setPage(1)
  }, [searchQuery])

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
    let cancelled = false

    void (async () => {
      try {
        const cfg = await tldwClient.getConfig().catch(() => null)
        const authMode = cfg?.authMode === "multi-user" ? "multi-user" : "single-user"
        if (authMode !== "multi-user") {
          if (cancelled) return
          setShareAccess({
            loading: false,
            authMode,
            role: null,
            canShareAssignments: true
          })
          return
        }

        const currentUser = await tldwAuth.getCurrentUser().catch(() => null)
        const normalizedRole = String(currentUser?.role || "").trim().toLowerCase() || "user"
        if (cancelled) return
        setShareAccess({
          loading: false,
          authMode,
          role: normalizedRole,
          canShareAssignments: ASSIGNMENT_PRIVILEGED_ROLES.has(normalizedRole)
        })
      } catch {
        if (cancelled) return
        setShareAccess({
          loading: false,
          authMode: "multi-user",
          role: null,
          canShareAssignments: false
        })
      }
    })()

    return () => {
      cancelled = true
    }
  }, [])

  const queryParams = React.useMemo(
    () =>
      buildQuizManageQueryParams({
        searchQuery,
        page,
        pageSize,
        showWorkspaceQuizzes,
        selectedWorkspaceId
      }),
    [page, pageSize, searchQuery, selectedWorkspaceId, showWorkspaceQuizzes]
  )
  const offset = queryParams.offset
  const { data, isLoading, refetch } = useQuizzesQuery(queryParams)
  const createQuizMutation = useCreateQuizMutation()
  const deleteMutation = useDeleteQuizMutation()
  const updateQuizMutation = useUpdateQuizMutation()
  const createQuestionMutation = useCreateQuestionMutation()
  const updateQuestionMutation = useUpdateQuestionMutation()
  const deleteQuestionMutation = useDeleteQuestionMutation()

  const questionsQuery = useQuestionsQuery(
    editingQuiz?.id,
    {
      include_answers: true,
      limit: 200,
      offset: 0
    },
    { enabled: editModalOpen && !!editingQuiz }
  )

  const allQuizzes = data?.items ?? []
  const quizzes = allQuizzes.filter((q) => !deletedQuizIds.has(q.id))
  const total = data?.count ?? 0
  const workspaceFilterOptions = React.useMemo(() => {
    const workspaceIds = new Set<string>()
    allQuizzes.forEach((quiz) => {
      const rawWorkspaceId =
        typeof quiz.workspace_id === "string" && quiz.workspace_id.trim().length > 0
          ? quiz.workspace_id.trim()
          : typeof quiz.workspace_tag === "string" && quiz.workspace_tag.startsWith("workspace:")
            ? quiz.workspace_tag.slice("workspace:".length).trim()
            : ""
      if (rawWorkspaceId.length > 0) {
        workspaceIds.add(rawWorkspaceId)
      }
    })
    if (selectedWorkspaceId) {
      workspaceIds.add(selectedWorkspaceId)
    }
    return Array.from(workspaceIds)
      .sort((left, right) => left.localeCompare(right))
      .map((workspaceId) => ({
        label: workspaceId,
        value: workspaceId
      }))
  }, [allQuizzes, selectedWorkspaceId])
  const allQuestions = (questionsQuery.data?.items ?? []) as Question[]
  const questions = allQuestions.filter((q) => !deletedQuestionIds.has(q.id))
  const questionTotal = questionsQuery.data?.count ?? 0
  const sortedQuestions = React.useMemo(
    () => [...questions].sort((a, b) => (a.order_index ?? 0) - (b.order_index ?? 0)),
    [questions]
  )
  const reorderBusy = reorderPendingQuestionId != null || updateQuestionMutation.isPending
  const mediaIds = React.useMemo(() => (
    Array.from(new Set(
      quizzes
        .map((quiz) => quiz.media_id)
        .filter((id): id is number => typeof id === "number" && Number.isFinite(id))
    ))
  ), [quizzes])
  const missingMediaIds = React.useMemo(
    () => mediaIds.filter((id) => !mediaNameMap[id]),
    [mediaIds, mediaNameMap]
  )

  React.useEffect(() => {
    setSelectedQuizIds((prev) => {
      const visibleIds = new Set(quizzes.map((quiz) => quiz.id))
      const next = new Set<number>()
      let changed = false
      prev.forEach((id) => {
        if (visibleIds.has(id)) {
          next.add(id)
        } else {
          changed = true
        }
      })
      if (!changed && next.size === prev.size) {
        return prev
      }
      return next
    })
  }, [quizzes])

  React.useEffect(() => {
    if (missingMediaIds.length === 0) return
    let cancelled = false

    void (async () => {
      const entries = await Promise.all(
        missingMediaIds.map(async (mediaId) => {
          try {
            const details = await tldwClient.getMediaDetails(mediaId, {
              include_content: false,
              include_versions: false,
              include_version_content: false
            })
            return [mediaId, getMediaDisplayName(details, mediaId)] as const
          } catch {
            return [mediaId, `Media #${mediaId}`] as const
          }
        })
      )

      if (cancelled) return

      setMediaNameMap((prev) => {
        const next = { ...prev }
        for (const [mediaId, name] of entries) {
          next[mediaId] = name
        }
        return next
      })
    })()

    return () => {
      cancelled = true
    }
  }, [missingMediaIds])

  const questionTypeLabel = (questionType: QuestionType) => {
    if (questionType === "multiple_choice") {
      return t("option:quiz.multipleChoice", { defaultValue: "Multiple Choice" })
    }
    if (questionType === "multi_select") {
      return t("option:quiz.multiSelect", { defaultValue: "Multi-Select" })
    }
    if (questionType === "matching") {
      return t("option:quiz.matching", { defaultValue: "Matching" })
    }
    if (questionType === "true_false") {
      return t("option:quiz.trueFalse", { defaultValue: "True/False" })
    }
    return t("option:quiz.fillBlank", { defaultValue: "Fill in the Blank" })
  }

  const captureFocusTarget = (ref: React.MutableRefObject<HTMLElement | null>) => {
    const activeElement = document.activeElement
    ref.current = activeElement instanceof HTMLElement ? activeElement : null
  }

  const restoreFocusTarget = (ref: React.MutableRefObject<HTMLElement | null>) => {
    const element = ref.current
    if (!element) return
    window.setTimeout(() => {
      if (!element.isConnected) return
      if ("disabled" in element && (element as HTMLButtonElement).disabled) return
      element.focus()
    }, 0)
  }

  const handleUndoQuizDelete = () => {
    if (!pendingQuizDeletion.current) return
    clearTimeout(pendingQuizDeletion.current.timeoutId)
    const quiz = pendingQuizDeletion.current.quiz
    pendingQuizDeletion.current = null
    setPendingQuizUndoName(null)
    setDeletedQuizIds((prev) => {
      const next = new Set(prev)
      next.delete(quiz.id)
      return next
    })
    messageApi.success(
      t("option:quiz.undoSuccess", { defaultValue: "Deletion undone" })
    )
  }

  const executeQuizDelete = async (quiz: Quiz) => {
    try {
      await deleteMutation.mutateAsync({ quizId: quiz.id, version: quiz.version })
      pendingQuizDeletion.current = null
      setPendingQuizUndoName(null)
      setDeletedQuizIds((prev) => {
        const next = new Set(prev)
        next.delete(quiz.id)
        return next
      })
      refetch()
    } catch (error) {
      // Deletion failed, restore the quiz in UI
      setDeletedQuizIds((prev) => {
        const next = new Set(prev)
        next.delete(quiz.id)
        return next
      })
      pendingQuizDeletion.current = null
      setPendingQuizUndoName(null)
      messageApi.error(
        t("option:quiz.deleteError", { defaultValue: "Failed to delete quiz" })
      )
    }
  }

  const handleDelete = (quiz: Quiz) => {
    // Cancel any existing pending deletion
    if (pendingQuizDeletion.current) {
      clearTimeout(pendingQuizDeletion.current.timeoutId)
      // Execute the previous pending deletion immediately
      executeQuizDelete(pendingQuizDeletion.current.quiz)
    }

    // Optimistically hide the quiz from UI
    setDeletedQuizIds((prev) => new Set(prev).add(quiz.id))
    setPendingQuizUndoName(quiz.name)

    // Schedule actual deletion
    const timeoutId = setTimeout(() => {
      executeQuizDelete(quiz)
    }, UNDO_GRACE_PERIOD)

    pendingQuizDeletion.current = { quiz, timeoutId }
  }

  const toggleQuizSelection = (quizId: number, checked: boolean) => {
    setSelectedQuizIds((prev) => {
      const next = new Set(prev)
      if (checked) {
        next.add(quizId)
      } else {
        next.delete(quizId)
      }
      return next
    })
  }

  const clearQuizSelection = () => {
    setSelectedQuizIds(new Set())
  }

  const handleDuplicateQuiz = async (quiz: Quiz) => {
    setDuplicateInFlightQuizId(quiz.id)
    try {
      const response = await listQuestions(quiz.id, {
        include_answers: true,
        limit: 200,
        offset: 0
      })
      const sourceQuestions = (response.items as Question[]).slice()
      sourceQuestions.sort((a, b) => (a.order_index ?? 0) - (b.order_index ?? 0))

      const duplicateQuiz = await createQuizMutation.mutateAsync({
        name: t("option:quiz.duplicateQuizName", {
          defaultValue: "{{name}} (Copy)",
          name: quiz.name
        }),
        description: quiz.description ?? undefined,
        workspace_tag: quiz.workspace_tag ?? undefined,
        media_id: quiz.media_id ?? undefined,
        time_limit_seconds: quiz.time_limit_seconds ?? undefined,
        passing_score: quiz.passing_score ?? undefined
      })

      const failedQuestionNumbers: number[] = []
      for (let i = 0; i < sourceQuestions.length; i++) {
        const sourceQuestion = sourceQuestions[i]
        try {
          await createQuestionMutation.mutateAsync({
            quizId: duplicateQuiz.id,
            question: {
              question_type: sourceQuestion.question_type,
              question_text: sourceQuestion.question_text,
              options:
                sourceQuestion.question_type === "multiple_choice" ||
                sourceQuestion.question_type === "multi_select" ||
                sourceQuestion.question_type === "matching"
                  ? (sourceQuestion.options ?? [])
                  : undefined,
              correct_answer: sourceQuestion.correct_answer,
              explanation: sourceQuestion.explanation || undefined,
              ...(sourceQuestion.hint ? { hint: sourceQuestion.hint } : {}),
              ...((sourceQuestion.hint_penalty_points ?? 0) > 0
                ? { hint_penalty_points: sourceQuestion.hint_penalty_points }
                : {}),
              ...(Array.isArray(sourceQuestion.source_citations) && sourceQuestion.source_citations.length > 0
                ? { source_citations: sourceQuestion.source_citations }
                : {}),
              points: sourceQuestion.points,
              order_index: sourceQuestion.order_index
            }
          })
        } catch {
          failedQuestionNumbers.push(i + 1)
        }
      }

      if (failedQuestionNumbers.length > 0) {
        messageApi.warning(
          t("option:quiz.duplicatePartialFailure", {
            defaultValue:
              "Quiz duplicated, but failed to copy question(s): {{questions}}.",
            questions: failedQuestionNumbers.join(", ")
          })
        )
      } else {
        messageApi.success(
          t("option:quiz.duplicateSuccess", {
            defaultValue: "Quiz duplicated successfully."
          })
        )
      }
      refetch()
    } catch (error) {
      messageApi.error(
        t("option:quiz.duplicateError", {
          defaultValue: "Failed to duplicate quiz."
        })
      )
    } finally {
      setDuplicateInFlightQuizId(null)
    }
  }

  const executeBulkDelete = async () => {
    const selectedQuizzes = quizzes.filter((quiz) => selectedQuizIds.has(quiz.id))
    if (selectedQuizzes.length === 0) {
      return
    }

    setBulkDeleteInFlight(true)
    const failed: string[] = []
    let successCount = 0

    try {
      for (const quiz of selectedQuizzes) {
        try {
          await deleteMutation.mutateAsync({ quizId: quiz.id, version: quiz.version })
          successCount += 1
        } catch {
          failed.push(quiz.name)
        }
      }

      if (successCount > 0) {
        refetch()
      }

      if (failed.length === 0) {
        messageApi.success(
          t("option:quiz.bulkDeleteSuccess", {
            defaultValue: "Deleted {{count}} quiz(es).",
            count: successCount
          })
        )
      } else {
        messageApi.warning(
          t("option:quiz.bulkDeletePartialFailure", {
            defaultValue:
              "Deleted {{success}} quiz(es). Failed to delete: {{failed}}.",
            success: successCount,
            failed: failed.join(", ")
          })
        )
      }
    } finally {
      setBulkDeleteInFlight(false)
      if (failed.length === 0) {
        clearQuizSelection()
      } else {
        const failedSet = new Set(
          selectedQuizzes
            .filter((quiz) => failed.includes(quiz.name))
            .map((quiz) => quiz.id)
        )
        setSelectedQuizIds(failedSet)
      }
    }
  }

  const downloadJsonFile = (filename: string, payload: unknown) => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = filename
    document.body.appendChild(anchor)
    anchor.click()
    document.body.removeChild(anchor)
    URL.revokeObjectURL(url)
  }

  const getQuizExportEntry = async (quiz: Quiz): Promise<QuizExportEntry> => {
    const response = await listQuestions(quiz.id, {
      include_answers: true,
      limit: 200,
      offset: 0
    })

    const questions = (response.items as Question[])
      .slice()
      .sort((a, b) => (a.order_index ?? 0) - (b.order_index ?? 0))

    return { quiz, questions }
  }

  const buildQuizExportPayload = (entries: QuizExportEntry[]) => ({
    export_format: QUIZ_EXPORT_FORMAT,
    exported_at: new Date().toISOString(),
    source: "quiz-manage-tab",
    quiz_count: entries.length,
    quizzes: entries.map(({ quiz, questions }) => ({
      quiz: {
        id: quiz.id,
        name: quiz.name,
        description: quiz.description ?? null,
        workspace_tag: quiz.workspace_tag ?? null,
        media_id: quiz.media_id ?? null,
        time_limit_seconds: quiz.time_limit_seconds ?? null,
        passing_score: quiz.passing_score ?? null,
        total_questions: quiz.total_questions,
        version: quiz.version,
        created_at: quiz.created_at ?? null,
        last_modified: quiz.last_modified ?? null
      },
      questions: questions.map((question) => ({
        id: question.id,
        question_type: question.question_type,
        question_text: question.question_text,
        options: question.options ?? null,
        correct_answer: question.correct_answer,
        explanation: question.explanation ?? null,
        hint: question.hint ?? null,
        hint_penalty_points: question.hint_penalty_points ?? 0,
        source_citations: question.source_citations ?? null,
        points: question.points,
        order_index: question.order_index,
        tags: question.tags ?? null
      }))
    }))
  })

  const handleQuizExport = async (quiz: Quiz) => {
    setSingleExportInFlightQuizId(quiz.id)
    try {
      const entry = await getQuizExportEntry(quiz)
      downloadJsonFile(`quiz-${quiz.id}-export.json`, buildQuizExportPayload([entry]))
      messageApi.success(
        t("option:quiz.exportSuccess", {
          defaultValue: "Quiz exported successfully."
        })
      )
    } catch {
      messageApi.error(
        t("option:quiz.exportError", {
          defaultValue: "Failed to export quiz."
        })
      )
    } finally {
      setSingleExportInFlightQuizId(null)
    }
  }

  const handleQuizPrint = async (quiz: Quiz) => {
    setSinglePrintInFlightQuizId(quiz.id)
    try {
      const entry = await getQuizExportEntry(quiz)
      const printWindow = window.open("", "_blank", "noopener,noreferrer,width=1024,height=768")
      if (!printWindow) {
        throw new Error("print-window-unavailable")
      }
      printWindow.document.open()
      printWindow.document.write(buildPrintableQuizHtml(entry))
      printWindow.document.close()
      printWindow.focus()
      printWindow.print()

      messageApi.success(
        t("option:quiz.printOpened", {
          defaultValue: "Printable quiz opened."
        })
      )
    } catch {
      messageApi.error(
        t("option:quiz.printError", {
          defaultValue: "Failed to open printable quiz."
        })
      )
    } finally {
      setSinglePrintInFlightQuizId(null)
    }
  }

  const executeBulkExport = async () => {
    const selectedQuizzes = quizzes.filter((quiz) => selectedQuizIds.has(quiz.id))
    if (selectedQuizzes.length === 0) {
      return
    }

    setBulkExportInFlight(true)
    const failed: string[] = []
    const exportRows: QuizExportEntry[] = []

    try {
      for (const quiz of selectedQuizzes) {
        try {
          const entry = await getQuizExportEntry(quiz)
          exportRows.push(entry)
        } catch {
          failed.push(quiz.name)
        }
      }

      if (exportRows.length > 0) {
        downloadJsonFile("quizzes-export.json", buildQuizExportPayload(exportRows))
      }

      if (failed.length === 0) {
        messageApi.success(
          t("option:quiz.bulkExportSuccess", {
            defaultValue: "Exported {{count}} quiz(es).",
            count: exportRows.length
          })
        )
      } else {
        messageApi.warning(
          t("option:quiz.bulkExportPartialFailure", {
            defaultValue:
              "Exported {{success}} quiz(es). Failed to export: {{failed}}.",
            success: exportRows.length,
            failed: failed.join(", ")
          })
        )
      }
    } finally {
      setBulkExportInFlight(false)
    }
  }

  const copyTextToClipboard = async (value: string) => {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value)
      return
    }

    const textarea = document.createElement("textarea")
    textarea.value = value
    textarea.setAttribute("readonly", "true")
    textarea.style.position = "absolute"
    textarea.style.left = "-9999px"
    document.body.appendChild(textarea)
    textarea.select()
    document.execCommand("copy")
    document.body.removeChild(textarea)
  }

  const normalizeShareDueAt = (rawValue: string): string | null => {
    const trimmed = rawValue.trim()
    if (!trimmed) return null
    const parsedDate = new Date(trimmed)
    if (Number.isNaN(parsedDate.getTime())) return null
    return parsedDate.toISOString()
  }

  const openShareModal = (quiz: Quiz) => {
    if (shareAccess.loading) {
      return
    }
    if (!shareAccess.canShareAssignments) {
      messageApi.error(
        t("option:quiz.sharePermissionRequired", {
          defaultValue: "Sharing assignments requires owner, admin, or lead role."
        })
      )
      return
    }
    setShareTargetQuiz(quiz)
    setShareDueAtLocal("")
    setShareAssignmentNote("")
    setShareModalOpen(true)
  }

  const closeShareModal = () => {
    setShareModalOpen(false)
    setShareTargetQuiz(null)
    setShareDueAtLocal("")
    setShareAssignmentNote("")
  }

  const handleShareQuiz = async () => {
    if (!shareTargetQuiz) return

    const normalizedDueAt = normalizeShareDueAt(shareDueAtLocal)
    if (shareDueAtLocal.trim() && !normalizedDueAt) {
      messageApi.error(
        t("option:quiz.invalidAssignmentDueDate", {
          defaultValue: "Enter a valid assignment due date."
        })
      )
      return
    }

    setShareInFlightQuizId(shareTargetQuiz.id)
    try {
      const params = new URLSearchParams()
      params.set("tab", "take")
      params.set("start_quiz_id", String(shareTargetQuiz.id))
      params.set("highlight_quiz_id", String(shareTargetQuiz.id))
      params.set("assignment_mode", "shared")
      if (normalizedDueAt) {
        params.set("assignment_due_at", normalizedDueAt)
      }
      const trimmedNote = shareAssignmentNote.trim()
      if (trimmedNote) {
        params.set("assignment_note", trimmedNote)
      }
      if (shareAccess.role) {
        params.set("assigned_by_role", shareAccess.role)
      }
      const path = `/quiz?${params.toString()}`
      const shareUrl = typeof window !== "undefined" && window.location?.origin
        ? `${window.location.origin}${path}`
        : path

      await copyTextToClipboard(shareUrl)
      messageApi.success(
        t("option:quiz.shareLinkCopied", {
          defaultValue: "Quiz share link copied."
        })
      )
      closeShareModal()
    } catch {
      messageApi.error(
        t("option:quiz.shareLinkCopyError", {
          defaultValue: "Failed to copy quiz share link."
        })
      )
    } finally {
      setShareInFlightQuizId(null)
    }
  }

  const handleQuizImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null
    event.target.value = ""
    if (!file) return

    setImportInFlight(true)
    try {
      let payload: unknown
      try {
        payload = JSON.parse(await file.text()) as unknown
      } catch {
        messageApi.error(
          t("option:quiz.importParseError", {
            defaultValue: "Failed to parse quiz import file."
          })
        )
        return
      }

      const payloadRecord = asRecord(payload)
      const exportFormat = asNonEmptyString(payloadRecord?.export_format)
      if (exportFormat && exportFormat !== QUIZ_EXPORT_FORMAT) {
        messageApi.error(
          t("option:quiz.importUnsupportedFormat", {
            defaultValue: "Unsupported quiz export format."
          })
        )
        return
      }

      const entries = normalizeQuizImportPayload(payload)
      if (entries.length === 0) {
        messageApi.error(
          t("option:quiz.importEmpty", {
            defaultValue: "No quizzes found in import file."
          })
        )
        return
      }

      const importResult = await importQuizzesJson({
        export_format: exportFormat ?? QUIZ_EXPORT_FORMAT,
        quizzes: entries.map((entry) => ({
          quiz: entry.quiz,
          questions: [...entry.questions].sort((a, b) => a.order_index - b.order_index)
        }))
      })

      if (importResult.imported_quizzes > 0) {
        refetch()
      }

      if (importResult.imported_quizzes === entries.length && importResult.failed_questions === 0) {
        messageApi.success(
          t("option:quiz.importSuccess", {
            defaultValue: "Imported {{quizzes}} quiz(es) with {{questions}} question(s).",
            quizzes: importResult.imported_quizzes,
            questions: importResult.imported_questions
          })
        )
        return
      }

      if (importResult.imported_quizzes > 0) {
        const failedQuizNames = Array.from(
          new Set(
            importResult.errors
              .filter((error) => typeof error.question_index !== "number")
              .map((error) => error.quiz_name ?? "")
              .filter((name) => name.trim().length > 0)
          )
        )

        messageApi.warning(
          t("option:quiz.importPartial", {
            defaultValue:
              "Imported {{quizzes}} quiz(es). Failed quizzes: {{failedQuizzes}}. Failed questions: {{failedQuestions}}.",
            quizzes: importResult.imported_quizzes,
            failedQuizzes: failedQuizNames.length > 0 ? failedQuizNames.join(", ") : t("common:none", { defaultValue: "none" }),
            failedQuestions: importResult.failed_questions
          })
        )
        return
      }

      messageApi.error(
        t("option:quiz.importError", {
          defaultValue: "Failed to import quizzes."
        })
      )
    } catch {
      messageApi.error(
        t("option:quiz.importError", {
          defaultValue: "Failed to import quizzes."
        })
      )
    } finally {
      setImportInFlight(false)
    }
  }

  React.useEffect(() => {
    if (!editingQuiz) return
    editForm.setFieldsValue({
      name: editingQuiz.name,
      description: editingQuiz.description ?? "",
      timeLimit: editingQuiz.time_limit_seconds
        ? Math.round(editingQuiz.time_limit_seconds / 60)
        : undefined,
      passingScore: editingQuiz.passing_score ?? undefined,
      workspaceId: editingQuiz.workspace_id ?? ""
    })
  }, [editingQuiz, editForm])

  const openEditModal = (quiz: Quiz) => {
    captureFocusTarget(editModalTriggerRef)
    setEditingQuiz(quiz)
    setEditModalOpen(true)
  }

  const closeEditModal = () => {
    setEditModalOpen(false)
    setEditingQuiz(null)
    setQuestionModalOpen(false)
    setQuestionDraft(null)
    setQuestionValidationErrors({})
    editForm.resetFields()
    restoreFocusTarget(editModalTriggerRef)
  }

  const handleSaveQuiz = async () => {
    if (!editingQuiz) return
    try {
      const values = await editForm.validateFields()
      const workspaceId = typeof values.workspaceId === "string"
        ? values.workspaceId.trim()
        : ""
      await updateQuizMutation.mutateAsync({
        quizId: editingQuiz.id,
        update: {
          name: values.name,
          description: values.description || null,
          time_limit_seconds: values.timeLimit ? values.timeLimit * 60 : null,
          passing_score: values.passingScore ?? null,
          workspace_id: workspaceId.length > 0 ? workspaceId : null,
          workspace_tag: workspaceId.length > 0 ? `workspace:${workspaceId}` : null,
          expected_version: editingQuiz.version
        }
      })
      messageApi.success(
        t("option:quiz.updateSuccess", { defaultValue: "Quiz updated successfully" })
      )
      closeEditModal()
      refetch()
    } catch (error) {
      messageApi.error(
        t("option:quiz.updateError", { defaultValue: "Failed to update quiz" })
      )
    }
  }

  const baseQuestionDraft = (): QuestionDraft => ({
    question_type: "multiple_choice",
    question_text: "",
    options: ["", "", "", ""],
    correct_answer: 0,
    explanation: "",
    hint: "",
    hint_penalty_points: 0,
    points: 1,
    order_index: questionTotal
  })

  const openQuestionModal = (question?: Question) => {
    captureFocusTarget(questionModalTriggerRef)
    if (question) {
      setIsNewQuestion(false)
      const matchingMap = normalizeMatchingAnswerMap(question.correct_answer)
      const matchingOptions = (question.options ?? []).length > 0
        ? (question.options ?? [])
        : Object.keys(matchingMap)
      setQuestionDraft({
        id: question.id,
        question_type: question.question_type,
        question_text: question.question_text,
        options:
          question.question_type === "multiple_choice" ||
          question.question_type === "multi_select"
            ? (question.options ?? ["", "", "", ""])
            : question.question_type === "matching"
              ? (matchingOptions.length > 0 ? matchingOptions : ["", ""])
              : ["", "", "", ""],
        correct_answer:
          question.correct_answer ??
          (question.question_type === "true_false"
            ? "true"
            : question.question_type === "matching"
              ? matchingMap
            : question.question_type === "multi_select"
              ? []
              : ""),
        explanation: question.explanation ?? "",
        hint: question.hint ?? "",
        hint_penalty_points: question.hint_penalty_points ?? 0,
        points: question.points ?? 1,
        order_index: question.order_index ?? 0
      })
    } else {
      setIsNewQuestion(true)
      setQuestionDraft(baseQuestionDraft())
    }
    setQuestionValidationErrors({})
    setQuestionModalOpen(true)
  }

  const closeQuestionModal = () => {
    setQuestionModalOpen(false)
    setQuestionDraft(null)
    setQuestionValidationErrors({})
    restoreFocusTarget(questionModalTriggerRef)
  }

  const updateQuestionDraft = (updates: Partial<QuestionDraft>) => {
    setQuestionDraft((prev) => (prev ? { ...prev, ...updates } : prev))
  }

  const normalizeOptions = (options: string[]) => {
    const trimmed = options.map((opt) => opt.trim())
    const filtered: string[] = []
    const indexMap = new Map<number, number>()
    trimmed.forEach((opt, idx) => {
      if (!opt) return
      indexMap.set(idx, filtered.length)
      filtered.push(opt)
    })
    return { filtered, indexMap }
  }

  const handleSaveQuestion = async () => {
    if (!editingQuiz || !questionDraft) return

    const validationErrors: QuestionValidationErrors = {}
    if (!questionDraft.question_text.trim()) {
      validationErrors.questionText = t("option:quiz.questionTextRequired", {
        defaultValue: "Question text is required."
      })
    }

    let optionsPayload: string[] | undefined
    let correctAnswer: AnswerValue = questionDraft.correct_answer

    if (questionDraft.question_type === "multiple_choice" || questionDraft.question_type === "multi_select") {
      const { filtered, indexMap } = normalizeOptions(questionDraft.options)
      if (filtered.length < 2) {
        validationErrors.options = t("option:quiz.optionsRequired", {
          defaultValue: "Please provide at least two options."
        })
      }
      if (questionDraft.question_type === "multiple_choice") {
        const rawIndex = Number(correctAnswer)
        const mapped = indexMap.get(Number.isNaN(rawIndex) ? 0 : rawIndex)
        correctAnswer = mapped ?? 0
      } else {
        const selectedRaw = Array.isArray(correctAnswer) ? correctAnswer : []
        const selectedMapped = selectedRaw
          .map((entry) => Number(entry))
          .filter((entry) => Number.isFinite(entry))
          .map((entry) => indexMap.get(entry))
          .filter((entry): entry is number => typeof entry === "number")
        const selectedUnique = Array.from(new Set(selectedMapped)).sort((a, b) => a - b)
        if (selectedUnique.length === 0) {
          validationErrors.options = t("option:quiz.multiSelectCorrectRequired", {
            defaultValue: "Select at least one correct option."
          })
        }
        correctAnswer = selectedUnique
      }
      optionsPayload = filtered
    } else if (questionDraft.question_type === "matching") {
      const leftOptions = questionDraft.options
        .map((option) => option.trim())
        .filter((option) => option.length > 0)
      const answerMap = normalizeMatchingAnswerMap(questionDraft.correct_answer)
      const normalizedPairs = leftOptions
        .map((left) => {
          const right = String(answerMap[left] ?? "").trim()
          return { left, right }
        })
        .filter((pair) => pair.left.length > 0 && pair.right.length > 0)

      if (normalizedPairs.length < MIN_MATCHING_PAIRS) {
        validationErrors.options = t("option:quiz.matchingPairsRequired", {
          defaultValue: "Provide at least two complete matching pairs."
        })
      }

      optionsPayload = normalizedPairs.map((pair) => pair.left)
      correctAnswer = Object.fromEntries(normalizedPairs.map((pair) => [pair.left, pair.right]))
    } else if (questionDraft.question_type === "true_false") {
      correctAnswer = String(correctAnswer || "true").toLowerCase() === "true" ? "true" : "false"
    } else {
      correctAnswer = String(correctAnswer || "").trim()
    }

    if (validationErrors.questionText || validationErrors.options) {
      setQuestionValidationErrors(validationErrors)
      return
    }
    setQuestionValidationErrors({})
    const hintText = (questionDraft.hint ?? "").trim()
    const hintPenaltyPoints = Math.max(0, Number(questionDraft.hint_penalty_points ?? 0) || 0)

    try {
      if (isNewQuestion) {
        await createQuestionMutation.mutateAsync({
          quizId: editingQuiz.id,
          question: {
            question_type: questionDraft.question_type,
            question_text: questionDraft.question_text,
            options: optionsPayload,
            correct_answer: correctAnswer,
            explanation: questionDraft.explanation || undefined,
            ...(hintText ? { hint: hintText } : {}),
            ...(hintPenaltyPoints > 0 ? { hint_penalty_points: hintPenaltyPoints } : {}),
            points: questionDraft.points,
            order_index: questionDraft.order_index
          }
        })
        messageApi.success(
          t("option:quiz.questionSaveSuccess", { defaultValue: "Question created successfully." })
        )
      } else if (questionDraft.id != null) {
        await updateQuestionMutation.mutateAsync({
          quizId: editingQuiz.id,
          questionId: questionDraft.id,
          update: {
            question_type: questionDraft.question_type,
            question_text: questionDraft.question_text,
            options: optionsPayload,
            correct_answer: correctAnswer,
            explanation: questionDraft.explanation || undefined,
            ...(hintText ? { hint: hintText } : {}),
            ...(hintPenaltyPoints > 0 ? { hint_penalty_points: hintPenaltyPoints } : {}),
            points: questionDraft.points,
            order_index: questionDraft.order_index
          }
        })
        messageApi.success(
          t("option:quiz.questionUpdateSuccess", { defaultValue: "Question updated successfully." })
        )
      }
      closeQuestionModal()
      questionsQuery.refetch()
    } catch (error) {
      messageApi.error(
        t("option:quiz.questionSaveError", { defaultValue: "Failed to save question." })
      )
    }
  }

  const handleUndoQuestionDelete = () => {
    if (!pendingQuestionDeletion.current) return
    clearTimeout(pendingQuestionDeletion.current.timeoutId)
    const question = pendingQuestionDeletion.current.question
    pendingQuestionDeletion.current = null
    setPendingQuestionUndoText(null)
    setDeletedQuestionIds((prev) => {
      const next = new Set(prev)
      next.delete(question.id)
      return next
    })
    messageApi.success(
      t("option:quiz.undoSuccess", { defaultValue: "Deletion undone" })
    )
  }

  const executeQuestionDelete = async (question: Question, quizId: number) => {
    try {
      await deleteQuestionMutation.mutateAsync({
        quizId,
        questionId: question.id,
        version: question.version
      })
      pendingQuestionDeletion.current = null
      setPendingQuestionUndoText(null)
      setDeletedQuestionIds((prev) => {
        const next = new Set(prev)
        next.delete(question.id)
        return next
      })
      questionsQuery.refetch()
    } catch (error) {
      // Deletion failed, restore the question in UI
      setDeletedQuestionIds((prev) => {
        const next = new Set(prev)
        next.delete(question.id)
        return next
      })
      pendingQuestionDeletion.current = null
      setPendingQuestionUndoText(null)
      messageApi.error(
        t("option:quiz.questionDeleteError", { defaultValue: "Failed to delete question." })
      )
    }
  }

  const handleDeleteQuestion = (question: Question) => {
    if (!editingQuiz) return

    // Cancel any existing pending question deletion
    if (pendingQuestionDeletion.current && editingQuiz) {
      clearTimeout(pendingQuestionDeletion.current.timeoutId)
      executeQuestionDelete(pendingQuestionDeletion.current.question, editingQuiz.id)
    }

    // Optimistically hide the question from UI
    setDeletedQuestionIds((prev) => new Set(prev).add(question.id))
    setPendingQuestionUndoText(question.question_text || t("option:quiz.question", { defaultValue: "Question" }))

    // Schedule actual deletion
    const quizId = editingQuiz.id
    const timeoutId = setTimeout(() => {
      executeQuestionDelete(question, quizId)
    }, UNDO_GRACE_PERIOD)

    pendingQuestionDeletion.current = { question, timeoutId }
  }

  const handleReorderQuestion = async (questionId: number, direction: "up" | "down") => {
    if (!editingQuiz) return
    const currentIndex = sortedQuestions.findIndex((question) => question.id === questionId)
    if (currentIndex < 0) return

    const nextIndex = direction === "up" ? currentIndex - 1 : currentIndex + 1
    if (nextIndex < 0 || nextIndex >= sortedQuestions.length) return

    const currentQuestion = sortedQuestions[currentIndex]
    const targetQuestion = sortedQuestions[nextIndex]
    const currentOrder = currentQuestion.order_index ?? currentIndex
    const targetOrder = targetQuestion.order_index ?? nextIndex

    setReorderPendingQuestionId(questionId)
    try {
      await updateQuestionMutation.mutateAsync({
        quizId: editingQuiz.id,
        questionId: currentQuestion.id,
        update: {
          order_index: targetOrder
        }
      })
      await updateQuestionMutation.mutateAsync({
        quizId: editingQuiz.id,
        questionId: targetQuestion.id,
        update: {
          order_index: currentOrder
        }
      })
      messageApi.success(
        t("option:quiz.reorderQuestionsSuccess", {
          defaultValue: "Question order updated."
        })
      )
      questionsQuery.refetch()
    } catch (error) {
      messageApi.error(
        t("option:quiz.reorderQuestionsError", {
          defaultValue: "Failed to reorder questions."
        })
      )
    } finally {
      setReorderPendingQuestionId(null)
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-4 py-2" data-testid="manage-loading-skeleton">
        <Card size="small">
          <Skeleton active paragraph={{ rows: 2 }} />
        </Card>
        <Card size="small">
          <Skeleton active paragraph={{ rows: 5 }} />
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {contextHolder}

      {pendingQuizUndoName && (
        <Alert
          type="info"
          showIcon
          title={t("option:quiz.quizDeletedUndoPrompt", {
            defaultValue: "Quiz deleted: {{name}}",
            name: pendingQuizUndoName
          })}
          action={(
            <Button
              type="link"
              size="small"
              icon={<UndoOutlined />}
              onClick={handleUndoQuizDelete}
            >
              {t("option:quiz.undo", { defaultValue: "Undo" })}
            </Button>
          )}
        />
      )}

      {pendingQuestionUndoText && (
        <Alert
          type="info"
          showIcon
          title={t("option:quiz.questionDeletedUndoPrompt", {
            defaultValue: "Question deleted: {{question}}",
            question: pendingQuestionUndoText
          })}
          action={(
            <Button
              type="link"
              size="small"
              icon={<UndoOutlined />}
              onClick={handleUndoQuestionDelete}
            >
              {t("option:quiz.undo", { defaultValue: "Undo" })}
            </Button>
          )}
        />
      )}

      <input
        ref={importInputRef}
        type="file"
        accept="application/json"
        className="hidden"
        data-testid="manage-import-input"
        onChange={(event) => {
          void handleQuizImport(event)
        }}
      />

      <div className="flex justify-between items-center">
        <div className="flex flex-wrap items-center gap-3">
          <Input
            ref={searchInputRef}
            placeholder={t("option:quiz.searchQuizzes", { defaultValue: "Search quizzes..." })}
            prefix={<SearchOutlined />}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="max-w-xs"
            allowClear
          />
          <Checkbox
            checked={showWorkspaceQuizzes}
            onChange={(event) => {
              setShowWorkspaceQuizzes(event.target.checked)
              if (!event.target.checked) {
                setSelectedWorkspaceId(null)
              }
              setPage(1)
            }}
            aria-label={t("option:quiz.showWorkspaceQuizzes", { defaultValue: "Show workspace quizzes" })}
            data-testid="quiz-manage-show-workspace-quizzes"
          >
            {t("option:quiz.showWorkspaceQuizzes", { defaultValue: "Show workspace quizzes" })}
          </Checkbox>
          <Select<string>
            allowClear
            showSearch
            placeholder={t("option:quiz.filterWorkspace", { defaultValue: "Filter workspace" })}
            value={selectedWorkspaceId ?? undefined}
            onChange={(value) => {
              setSelectedWorkspaceId(value ?? null)
              setPage(1)
            }}
            disabled={!showWorkspaceQuizzes && selectedWorkspaceId == null}
            options={workspaceFilterOptions}
            className="min-w-44"
            data-testid="quiz-manage-workspace-filter"
          />
        </div>
        <Space>
          <Button onClick={onNavigateToGenerate}>
            {t("option:quiz.generateNew", { defaultValue: "Generate New" })}
          </Button>
          <Button
            icon={<UploadOutlined />}
            onClick={() => importInputRef.current?.click()}
            loading={importInFlight}
            data-testid="manage-import-trigger"
          >
            {t("option:quiz.importJson", { defaultValue: "Import JSON" })}
          </Button>
          <Button type="primary" onClick={onNavigateToCreate}>
            {t("option:quiz.createNew", { defaultValue: "Create New" })}
          </Button>
        </Space>
      </div>

      {selectedQuizIds.size > 0 && (
        <Alert
          type="info"
          showIcon
          title={t("option:quiz.bulkSelectionMessage", {
            defaultValue: "{{count}} quiz(es) selected",
            count: selectedQuizIds.size
          })}
          action={(
            <Space>
              <Button
                size="small"
                onClick={clearQuizSelection}
                disabled={bulkDeleteInFlight || bulkExportInFlight}
              >
                {t("common:clear", { defaultValue: "Clear" })}
              </Button>
              <Button
                size="small"
                onClick={() => executeBulkExport()}
                loading={bulkExportInFlight}
                disabled={bulkDeleteInFlight || bulkExportInFlight}
                data-testid="manage-bulk-export"
              >
                {t("option:quiz.exportSelected", { defaultValue: "Export Selected" })}
              </Button>
              <Popconfirm
                title={t("option:quiz.bulkDeleteConfirmTitle", {
                  defaultValue: "Delete selected quizzes?"
                })}
                description={t("option:quiz.bulkDeleteConfirmDescription", {
                  defaultValue: "This will permanently delete {{count}} quiz(es).",
                  count: selectedQuizIds.size
                })}
                okText={t("common:delete", { defaultValue: "Delete" })}
                cancelText={t("common:cancel", { defaultValue: "Cancel" })}
                onConfirm={() => executeBulkDelete()}
              >
                <Button
                  size="small"
                  danger
                  loading={bulkDeleteInFlight}
                  disabled={bulkExportInFlight}
                  data-testid="manage-bulk-delete"
                >
                  {t("option:quiz.deleteSelected", { defaultValue: "Delete Selected" })}
                </Button>
              </Popconfirm>
            </Space>
          )}
        />
      )}

      {quizzes.length === 0 ? (
        <Empty
          description={
            searchQuery
              ? t("option:quiz.noSearchResults", { defaultValue: "No quizzes match your search" })
              : t("option:quiz.noQuizzesYet", { defaultValue: "No quizzes yet" })
          }
        >
          {!searchQuery && (
            <Space>
              <Button type="primary" onClick={onNavigateToGenerate}>
                {t("option:quiz.generateFromMedia", { defaultValue: "Generate from Media" })}
              </Button>
              <Button onClick={onNavigateToCreate}>
                {t("option:quiz.createManually", { defaultValue: "Create Manually" })}
              </Button>
            </Space>
          )}
        </Empty>
      ) : (
        <List
          dataSource={quizzes}
          pagination={{
            current: page,
            pageSize,
            total,
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
            const sourceSummary = summarizeQuizSources(quiz)
            return (
              <List.Item
              actions={[
                <Button
                  key="start"
                  type="link"
                  icon={<PlayCircleOutlined />}
                  onClick={() => {
                    onStartQuiz(quiz.id)
                  }}
                  data-testid={`quiz-start-${quiz.id}`}
                >
                  {t("option:quiz.start", { defaultValue: "Start" })}
                </Button>,
                <Button
                  key="duplicate"
                  type="link"
                  icon={<CopyOutlined />}
                  onClick={() => {
                    void handleDuplicateQuiz(quiz)
                  }}
                  loading={duplicateInFlightQuizId === quiz.id}
                  data-testid={`quiz-duplicate-${quiz.id}`}
                >
                  {t("option:quiz.duplicate", { defaultValue: "Duplicate" })}
                </Button>,
                <Button
                  key="export"
                  type="link"
                  icon={<DownloadOutlined />}
                  onClick={() => {
                    void handleQuizExport(quiz)
                  }}
                  loading={singleExportInFlightQuizId === quiz.id}
                  data-testid={`quiz-export-${quiz.id}`}
                >
                  {t("option:quiz.export", { defaultValue: "Export" })}
                </Button>,
                <Button
                  key="print"
                  type="link"
                  icon={<PrinterOutlined />}
                  onClick={() => {
                    void handleQuizPrint(quiz)
                  }}
                  loading={singlePrintInFlightQuizId === quiz.id}
                  data-testid={`quiz-print-${quiz.id}`}
                >
                  {t("option:quiz.print", { defaultValue: "Print" })}
                </Button>,
                <Button
                  key="share"
                  type="link"
                  icon={<ShareAltOutlined />}
                  onClick={() => {
                    openShareModal(quiz)
                  }}
                  loading={shareInFlightQuizId === quiz.id}
                  disabled={shareAccess.loading || !shareAccess.canShareAssignments}
                  title={
                    shareAccess.loading
                      ? t("option:quiz.checkingSharePermissions", {
                        defaultValue: "Checking share permissions..."
                      })
                      : !shareAccess.canShareAssignments
                        ? t("option:quiz.sharePermissionRequired", {
                          defaultValue: "Sharing assignments requires owner, admin, or lead role."
                        })
                        : undefined
                  }
                  data-testid={`quiz-share-${quiz.id}`}
                >
                  {t("option:quiz.share", { defaultValue: "Share" })}
                </Button>,
                <Button
                  key="edit"
                  type="link"
                  icon={<EditOutlined />}
                  onClick={() => {
                    openEditModal(quiz)
                  }}
                  data-testid={`quiz-edit-${quiz.id}`}
                >
                  {t("option:quiz.edit", { defaultValue: "Edit" })}
                </Button>,
                <Button
                  key="delete"
                  type="link"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() => handleDelete(quiz)}
                >
                  {t("option:quiz.delete", { defaultValue: "Delete" })}
                </Button>
              ]}
              >
                <List.Item.Meta
                  title={
                    <Space size="small" align="center">
                      <Checkbox
                        checked={selectedQuizIds.has(quiz.id)}
                        onChange={(event) => toggleQuizSelection(quiz.id, event.target.checked)}
                        aria-label={t("option:quiz.selectQuiz", {
                          defaultValue: "Select quiz {{name}}",
                          name: quiz.name
                        })}
                      />
                      <span className="font-medium">{quiz.name}</span>
                    </Space>
                  }
                  description={
                    <div className="space-y-1">
                      <div className="flex flex-wrap gap-2">
                        {sourceSummary.media > 0 && (
                          <Tag color="green" data-testid={`manage-quiz-source-media-${quiz.id}`}>
                            {t("option:quiz.sourceBadgeMedia", {
                              defaultValue: "Media {{count}}",
                              count: sourceSummary.media
                            })}
                          </Tag>
                        )}
                        {sourceSummary.notes > 0 && (
                          <Tag color="cyan" data-testid={`manage-quiz-source-notes-${quiz.id}`}>
                            {t("option:quiz.sourceBadgeNotes", {
                              defaultValue: "Notes {{count}}",
                              count: sourceSummary.notes
                            })}
                          </Tag>
                        )}
                        {sourceSummary.flashcards > 0 && (
                          <Tag color="magenta" data-testid={`manage-quiz-source-flashcards-${quiz.id}`}>
                            {t("option:quiz.sourceBadgeFlashcards", {
                              defaultValue: "Flashcards {{count}}",
                              count: sourceSummary.flashcards
                            })}
                          </Tag>
                        )}
                      </div>
                    {quiz.description && (
                      <p className="text-sm text-text-muted line-clamp-1">
                        {quiz.description}
                      </p>
                    )}
                    <div className="flex gap-2">
                      <Tag icon={<QuestionCircleOutlined />}>
                        {quiz.total_questions}{" "}
                        {t("option:quiz.questions", { defaultValue: "questions" })}
                      </Tag>
                      {quiz.passing_score && (
                        <Tag color="blue">
                          {t("option:quiz.passingScoreLabel", { defaultValue: "Pass" })}: {quiz.passing_score}%
                        </Tag>
                      )}
                      {quiz.media_id != null && (
                        <Tag color="green">
                          <Typography.Link
                            href={`/media?id=${quiz.media_id}`}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {mediaNameMap[quiz.media_id] ??
                              t("option:quiz.sourceMedia", {
                                defaultValue: "Source media #{{id}}",
                                id: quiz.media_id
                              })}
                          </Typography.Link>
                        </Tag>
                      )}
                    </div>
                  </div>
                  }
                />
              </List.Item>
            )
          }}
        />
      )}

      <Modal
        title={t("option:quiz.editQuizTitle", { defaultValue: "Edit Quiz" })}
        open={editModalOpen}
        onCancel={closeEditModal}
        onOk={handleSaveQuiz}
        okText={t("common:save", { defaultValue: "Save" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        confirmLoading={updateQuizMutation.isPending}
        width="95vw"
        style={{ top: 16 }}
        modalRender={(node) => <div data-testid="manage-edit-quiz-modal">{node}</div>}
      >
        <Form form={editForm} layout="vertical">
          <Form.Item
            name="name"
            label={t("option:quiz.quizName", { defaultValue: "Quiz Name" })}
            rules={[
              {
                required: true,
                message: t("option:quiz.nameRequired", { defaultValue: "Please enter a quiz name" })
              }
            ]}
          >
            <Input />
          </Form.Item>

          <Form.Item
            name="description"
            label={t("option:quiz.description", { defaultValue: "Description" })}
          >
            <Input.TextArea rows={2} />
          </Form.Item>

          <Form.Item
            name="workspaceId"
            label={t("option:quiz.workspaceId", { defaultValue: "Workspace ID" })}
          >
            <Input
              aria-label={t("option:quiz.workspaceId", { defaultValue: "Workspace ID" })}
              placeholder={t("option:quiz.workspaceIdPlaceholder", { defaultValue: "Leave blank for general scope" })}
            />
          </Form.Item>

          <div className="grid grid-cols-2 gap-4">
            <Form.Item
              name="timeLimit"
              label={t("option:quiz.timeLimit", { defaultValue: "Time Limit (minutes)" })}
            >
              <InputNumber min={1} max={180} className="w-full" />
            </Form.Item>
            <Form.Item
              name="passingScore"
              label={t("option:quiz.passingScore", { defaultValue: "Passing Score (%)" })}
            >
              <InputNumber min={1} max={100} className="w-full" />
            </Form.Item>
          </div>
        </Form>

        <Divider />

        <div className="flex items-center justify-between mb-3">
          <Typography.Title level={5} className="!mb-0">
            {t("option:quiz.questionsSection", { defaultValue: "Questions" })}
          </Typography.Title>
          <Button
            type="dashed"
            icon={<PlusOutlined />}
            onClick={() => openQuestionModal()}
            data-testid="manage-add-question"
          >
            {t("option:quiz.addQuestion", { defaultValue: "Add Question" })}
          </Button>
        </div>

        {questionsQuery.isLoading ? (
          <div className="flex justify-center py-6">
            <Spin />
          </div>
        ) : questions.length === 0 ? (
          <Empty
            description={t("option:quiz.noQuestionsYet", { defaultValue: "No questions added yet" })}
          />
        ) : (
          <div className="max-h-[50vh] overflow-y-auto pr-1" data-testid="manage-questions-scroll-container">
            <List
              dataSource={sortedQuestions}
              renderItem={(question, index) => (
                <List.Item
                  actions={[
                    <Button
                      key="move-up"
                      type="text"
                      icon={<ArrowUpOutlined />}
                      aria-label={t("option:quiz.moveQuestionUp", {
                        defaultValue: "Move question {{number}} up",
                        number: index + 1
                      })}
                      disabled={reorderBusy || index === 0}
                      onClick={() => handleReorderQuestion(question.id, "up")}
                    />,
                    <Button
                      key="move-down"
                      type="text"
                      icon={<ArrowDownOutlined />}
                      aria-label={t("option:quiz.moveQuestionDown", {
                        defaultValue: "Move question {{number}} down",
                        number: index + 1
                      })}
                      disabled={reorderBusy || index === sortedQuestions.length - 1}
                      onClick={() => handleReorderQuestion(question.id, "down")}
                    />,
                    <Button
                      key="edit"
                      type="link"
                      onClick={() => openQuestionModal(question)}
                      data-testid={`manage-edit-question-${question.id}`}
                    >
                      {t("common:edit", { defaultValue: "Edit" })}
                    </Button>,
                    <Button
                      key="delete"
                      type="link"
                      danger
                      onClick={() => handleDeleteQuestion(question)}
                    >
                      {t("option:quiz.delete", { defaultValue: "Delete" })}
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    title={question.question_text}
                    description={(
                      <div className="flex flex-wrap gap-2">
                        <Tag>{questionTypeLabel(question.question_type)}</Tag>
                        <Tag>{t("option:quiz.points", { defaultValue: "Points" })}: {question.points}</Tag>
                        <Tag>{t("option:quiz.orderIndex", { defaultValue: "Order" })}: {(question.order_index ?? 0) + 1}</Tag>
                      </div>
                    )}
                  />
                </List.Item>
              )}
            />
          </div>
        )}
      </Modal>

      <Modal
        title={t("option:quiz.shareQuizAssignment", { defaultValue: "Share Quiz Assignment" })}
        open={shareModalOpen}
        onCancel={closeShareModal}
        onOk={() => {
          void handleShareQuiz()
        }}
        okText={t("option:quiz.copyAssignmentLink", { defaultValue: "Copy Link" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        confirmLoading={shareInFlightQuizId != null}
      >
        <Space orientation="vertical" className="w-full" size={12}>
          <Typography.Text type="secondary">
            {t("option:quiz.shareAssignmentDescription", {
              defaultValue:
                "Create a shareable assignment link for this quiz. Add an optional due date and note."
            })}
          </Typography.Text>
          <div className="space-y-1">
            <Typography.Text strong>
              {t("option:quiz.assignmentDueDate", { defaultValue: "Due date (optional)" })}
            </Typography.Text>
            <Input
              type="datetime-local"
              value={shareDueAtLocal}
              onChange={(event) => setShareDueAtLocal(event.target.value)}
              data-testid="quiz-share-due-at-input"
            />
            <Typography.Text type="secondary" className="text-xs">
              {t("option:quiz.assignmentDueDateHint", {
                defaultValue:
                  "When provided, the link includes an `assignment_due_at` timestamp in UTC."
              })}
            </Typography.Text>
          </div>
          <div className="space-y-1">
            <Typography.Text strong>
              {t("option:quiz.assignmentNote", { defaultValue: "Assignment note (optional)" })}
            </Typography.Text>
            <Input.TextArea
              value={shareAssignmentNote}
              onChange={(event) => setShareAssignmentNote(event.target.value)}
              rows={2}
              maxLength={280}
              data-testid="quiz-share-note-input"
              placeholder={t("option:quiz.assignmentNotePlaceholder", {
                defaultValue: "Add context for the learner..."
              })}
            />
          </div>
        </Space>
      </Modal>

      <Modal
        title={
          isNewQuestion
            ? t("option:quiz.addQuestion", { defaultValue: "Add Question" })
            : t("option:quiz.editQuestion", { defaultValue: "Edit Question" })
        }
        open={questionModalOpen}
        onCancel={closeQuestionModal}
        onOk={handleSaveQuestion}
        okText={t("common:save", { defaultValue: "Save" })}
        cancelText={t("common:cancel", { defaultValue: "Cancel" })}
        confirmLoading={createQuestionMutation.isPending || updateQuestionMutation.isPending}
        width={760}
      >
        {questionDraft && (
          <Space orientation="vertical" className="w-full">
            <Select
              value={questionDraft.question_type}
              onChange={(value: QuestionType) => {
                const updates: Partial<QuestionDraft> = { question_type: value }
                if (value === "multiple_choice") {
                  updates.correct_answer = 0
                  if (questionDraft.options.length === 0) {
                    updates.options = ["", "", "", ""]
                  }
                } else if (value === "multi_select") {
                  updates.correct_answer = []
                  if (questionDraft.options.length === 0) {
                    updates.options = ["", "", "", ""]
                  }
                } else if (value === "matching") {
                  updates.correct_answer = {}
                  const padded = [...questionDraft.options]
                  while (padded.length < MIN_MATCHING_PAIRS) {
                    padded.push("")
                  }
                  updates.options = padded
                } else if (value === "true_false") {
                  updates.correct_answer = "true"
                } else {
                  updates.correct_answer = ""
                }
                updateQuestionDraft(updates)
                setQuestionValidationErrors({})
              }}
              options={[
                { label: t("option:quiz.multipleChoice", { defaultValue: "Multiple Choice" }), value: "multiple_choice" },
                { label: t("option:quiz.multiSelect", { defaultValue: "Multi-Select" }), value: "multi_select" },
                { label: t("option:quiz.matching", { defaultValue: "Matching" }), value: "matching" },
                { label: t("option:quiz.trueFalse", { defaultValue: "True/False" }), value: "true_false" },
                { label: t("option:quiz.fillBlank", { defaultValue: "Fill in the Blank" }), value: "fill_blank" }
              ]}
              className="w-60"
            />

            <Input.TextArea
              placeholder={t("option:quiz.questionText", { defaultValue: "Enter your question..." })}
              value={questionDraft.question_text}
              onChange={(e) => {
                updateQuestionDraft({ question_text: e.target.value })
                if (questionValidationErrors.questionText) {
                  setQuestionValidationErrors((prev) => ({ ...prev, questionText: undefined }))
                }
              }}
              aria-invalid={questionValidationErrors.questionText ? true : undefined}
              aria-describedby={questionValidationErrors.questionText ? QUESTION_TEXT_ERROR_ID : undefined}
              rows={2}
            />
            {questionValidationErrors.questionText && (
              <div
                id={QUESTION_TEXT_ERROR_ID}
                className="text-sm text-red-600"
                role="alert"
              >
                {questionValidationErrors.questionText}
              </div>
            )}

            {(questionDraft.question_type === "multiple_choice" || questionDraft.question_type === "multi_select") && (
              <div className="space-y-2">
                <div className="text-sm font-medium">
                  {t("option:quiz.options", { defaultValue: "Options" })}
                </div>
                {questionDraft.options.map((option, optIndex) => (
                  <div key={optIndex} className="flex items-center gap-2">
                    {questionDraft.question_type === "multi_select" ? (
                      <Checkbox
                        checked={Array.isArray(questionDraft.correct_answer) && questionDraft.correct_answer.includes(optIndex)}
                        onChange={(event) => {
                          const existing = Array.isArray(questionDraft.correct_answer)
                            ? questionDraft.correct_answer
                              .map((entry) => Number(entry))
                              .filter((entry) => Number.isFinite(entry))
                            : []
                          const next = event.target.checked
                            ? Array.from(new Set([...existing, optIndex])).sort((a, b) => a - b)
                            : existing.filter((entry) => entry !== optIndex)
                          updateQuestionDraft({ correct_answer: next })
                          if (questionValidationErrors.options) {
                            setQuestionValidationErrors((prev) => ({ ...prev, options: undefined }))
                          }
                        }}
                      />
                    ) : (
                      <input
                        type="radio"
                        name="edit-correct"
                        checked={Number(questionDraft.correct_answer) === optIndex}
                        onChange={() => updateQuestionDraft({ correct_answer: optIndex })}
                      />
                    )}
                    <Input
                      placeholder={`${t("option:quiz.option", { defaultValue: "Option" })} ${optIndex + 1}`}
                      value={option}
                      onChange={(e) => {
                        const newOptions = [...questionDraft.options]
                        newOptions[optIndex] = e.target.value
                        updateQuestionDraft({ options: newOptions })
                        if (questionValidationErrors.options) {
                          setQuestionValidationErrors((prev) => ({ ...prev, options: undefined }))
                        }
                      }}
                      aria-invalid={questionValidationErrors.options ? true : undefined}
                      aria-describedby={questionValidationErrors.options ? QUESTION_OPTIONS_ERROR_ID : undefined}
                      className="flex-1"
                    />
                  </div>
                ))}
                {questionValidationErrors.options && (
                  <div
                    id={QUESTION_OPTIONS_ERROR_ID}
                    className="text-sm text-red-600"
                    role="alert"
                  >
                    {questionValidationErrors.options}
                  </div>
                )}
              </div>
            )}

            {questionDraft.question_type === "matching" && (
              <div className="space-y-2">
                <div className="text-sm font-medium">
                  {t("option:quiz.matchingPairs", { defaultValue: "Matching Pairs" })}
                </div>
                {questionDraft.options.map((leftOption, pairIndex) => {
                  const answerMap = normalizeMatchingAnswerMap(questionDraft.correct_answer)
                  const leftTrimmed = leftOption.trim()
                  const rightValue = leftTrimmed ? (answerMap[leftTrimmed] ?? "") : ""
                  return (
                    <div
                      key={pairIndex}
                      className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_1fr_auto] sm:items-center"
                    >
                      <Input
                        placeholder={t("option:quiz.matchingLeftPlaceholder", {
                          defaultValue: "Left item"
                        })}
                        value={leftOption}
                        onChange={(e) => {
                          const nextLeftRaw = e.target.value
                          const previousLeft = questionDraft.options[pairIndex] ?? ""
                          const previousTrimmed = previousLeft.trim()
                          const nextTrimmed = nextLeftRaw.trim()
                          const nextOptions = [...questionDraft.options]
                          nextOptions[pairIndex] = nextLeftRaw
                          const nextMap = { ...normalizeMatchingAnswerMap(questionDraft.correct_answer) }
                          const carryValue = previousTrimmed ? nextMap[previousTrimmed] : undefined
                          if (previousTrimmed) {
                            delete nextMap[previousTrimmed]
                          }
                          if (nextTrimmed && carryValue) {
                            nextMap[nextTrimmed] = carryValue
                          }
                          updateQuestionDraft({
                            options: nextOptions,
                            correct_answer: nextMap
                          })
                          if (questionValidationErrors.options) {
                            setQuestionValidationErrors((prev) => ({ ...prev, options: undefined }))
                          }
                        }}
                      />
                      <Input
                        placeholder={t("option:quiz.matchingRightPlaceholder", {
                          defaultValue: "Matching item"
                        })}
                        value={rightValue}
                        onChange={(e) => {
                          const mapKey = (questionDraft.options[pairIndex] ?? "").trim()
                          const nextMap = { ...normalizeMatchingAnswerMap(questionDraft.correct_answer) }
                          if (!mapKey) {
                            updateQuestionDraft({ correct_answer: nextMap })
                            return
                          }
                          const nextRight = e.target.value.trim()
                          if (nextRight) {
                            nextMap[mapKey] = nextRight
                          } else {
                            delete nextMap[mapKey]
                          }
                          updateQuestionDraft({ correct_answer: nextMap })
                          if (questionValidationErrors.options) {
                            setQuestionValidationErrors((prev) => ({ ...prev, options: undefined }))
                          }
                        }}
                      />
                      <Button
                        type="text"
                        icon={<MinusCircleOutlined />}
                        className="min-h-11 min-w-11 self-start sm:self-auto"
                        aria-label={t("option:quiz.removeMatchingPair", {
                          defaultValue: "Remove matching pair {{pair}}",
                          pair: pairIndex + 1
                        })}
                        onClick={() => {
                          if (questionDraft.options.length <= MIN_MATCHING_PAIRS) return
                          const removedLeft = questionDraft.options[pairIndex]?.trim()
                          const nextOptions = questionDraft.options.filter((_, idx) => idx !== pairIndex)
                          const nextMap = { ...normalizeMatchingAnswerMap(questionDraft.correct_answer) }
                          if (removedLeft) {
                            delete nextMap[removedLeft]
                          }
                          updateQuestionDraft({
                            options: nextOptions,
                            correct_answer: nextMap
                          })
                        }}
                        disabled={questionDraft.options.length <= MIN_MATCHING_PAIRS}
                      />
                    </div>
                  )
                })}
                {questionValidationErrors.options && (
                  <div
                    id={QUESTION_OPTIONS_ERROR_ID}
                    className="text-sm text-red-600"
                    role="alert"
                  >
                    {questionValidationErrors.options}
                  </div>
                )}
                <Button
                  type="dashed"
                  icon={<PlusOutlined />}
                  className="min-h-11"
                  onClick={() => {
                    if (questionDraft.options.length >= MAX_MATCHING_PAIRS) return
                    updateQuestionDraft({
                      options: [...questionDraft.options, ""]
                    })
                  }}
                  disabled={questionDraft.options.length >= MAX_MATCHING_PAIRS}
                >
                  {t("option:quiz.addMatchingPair", { defaultValue: "Add Pair" })}
                </Button>
              </div>
            )}

            {questionDraft.question_type === "true_false" && (
              <fieldset className="border-0 m-0 p-0">
                <legend className="sr-only">
                  {t("option:quiz.trueFalseLegend", {
                    defaultValue: "True or false for: {{question}}",
                    question: questionDraft.question_text || t("option:quiz.question", { defaultValue: "question" })
                  })}
                </legend>
                <Radio.Group
                  value={questionDraft.correct_answer}
                  onChange={(e) => updateQuestionDraft({ correct_answer: e.target.value })}
                >
                  <Space orientation="vertical">
                    <Radio value="true">{t("option:quiz.true", { defaultValue: "True" })}</Radio>
                    <Radio value="false">{t("option:quiz.false", { defaultValue: "False" })}</Radio>
                  </Space>
                </Radio.Group>
              </fieldset>
            )}

            {questionDraft.question_type === "fill_blank" && (
              <Space orientation="vertical" className="w-full" size={4}>
                <Input
                  placeholder={t("option:quiz.correctAnswerPlaceholder", {
                    defaultValue: "Enter the correct answer..."
                  })}
                  value={typeof questionDraft.correct_answer === "string" ? questionDraft.correct_answer : ""}
                  onChange={(e) => updateQuestionDraft({ correct_answer: e.target.value })}
                />
                <Typography.Text type="secondary" className="text-xs">
                  {t("option:quiz.fillBlankAuthoringHelp", {
                    defaultValue:
                      "Use `answer1 || answer2` for alternates. Prefix with `~` (or `~0.85:`) to allow fuzzy matching."
                  })}
                </Typography.Text>
              </Space>
            )}

            <div className="grid grid-cols-2 gap-4">
              <InputNumber
                min={1}
                className="w-full"
                value={questionDraft.points}
                onChange={(value) => updateQuestionDraft({ points: Number(value) || 1 })}
                placeholder={t("option:quiz.points", { defaultValue: "Points" })}
              />
              <InputNumber
                min={0}
                className="w-full"
                value={questionDraft.order_index}
                onChange={(value) => updateQuestionDraft({ order_index: Number(value) || 0 })}
                placeholder={t("option:quiz.orderIndex", { defaultValue: "Order" })}
              />
            </div>

            <Input.TextArea
              placeholder={t("option:quiz.explanationPlaceholder", {
                defaultValue: "Explanation (shown after answering)..."
              })}
              value={questionDraft.explanation ?? ""}
              onChange={(e) => updateQuestionDraft({ explanation: e.target.value })}
              rows={2}
            />
            <Input.TextArea
              placeholder={t("option:quiz.hintPlaceholder", {
                defaultValue: "Optional hint (learner can reveal during quiz)..."
              })}
              value={questionDraft.hint ?? ""}
              onChange={(e) => updateQuestionDraft({ hint: e.target.value })}
              rows={2}
            />
            <InputNumber
              min={0}
              className="w-full sm:w-60"
              value={questionDraft.hint_penalty_points ?? 0}
              onChange={(value) => updateQuestionDraft({ hint_penalty_points: Number(value) || 0 })}
              placeholder={t("option:quiz.hintPenaltyPoints", {
                defaultValue: "Hint penalty (points)"
              })}
            />
            <Typography.Text type="secondary" className="text-xs">
              {t("option:quiz.hintPenaltyHelp", {
                defaultValue: "Applied only when the learner answers correctly after revealing the hint."
              })}
            </Typography.Text>
          </Space>
        )}
      </Modal>
    </div>
  )
}

export default ManageTab
