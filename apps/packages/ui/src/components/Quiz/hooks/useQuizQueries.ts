import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import {
  listQuizzes,
  createQuiz,
  getQuiz,
  updateQuiz,
  deleteQuiz,
  listQuestions,
  createQuestion,
  updateQuestion,
  deleteQuestion,
  startAttempt,
  submitAttempt,
  listAttempts,
  getAttempt,
  listAttemptRemediationConversions,
  convertAttemptRemediationQuestions,
  getQuizAttemptQuestionAssistant,
  generateQuiz,
  generateRemediationQuiz,
  respondQuizAttemptQuestionAssistant,
  type Quiz,
  type Question,
  type QuestionListParams,
  type QuizCreate,
  type QuizUpdate,
  type QuestionCreate,
  type QuestionUpdate,
  type QuestionType,
  type QuizGenerateRequest,
  type QuizRemediationGenerateRequest,
  type QuizRemediationConvertRequest,
  type QuizRemediationConvertResponse,
  type QuizRemediationConversionListResponse,
  type QuizAnswerInput,
  type QuizListParams,
  type AttemptListParams,
  type QuizAttempt
} from "@/services/quizzes"
import type {
  StudyAssistantAction,
  StudyAssistantContextResponse,
  StudyAssistantRespondRequest
} from "@/services/flashcards"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useMilestoneStore } from "@/store/milestones"

export interface UseQuizQueriesOptions {
  enabled?: boolean
}

const QUIZ_QUERY_STALE_TIME_MS = 30_000
const ATTEMPT_QUERY_STALE_TIME_MS = 30_000
const DEFAULT_STUDY_ASSISTANT_ACTIONS: StudyAssistantAction[] = [
  "explain",
  "mnemonic",
  "follow_up",
  "fact_check",
  "freeform"
]

type QuizListCacheValue = {
  items: Quiz[]
  count: number
}

const normalizeWorkspaceId = (value?: string | null): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const normalizeWorkspaceTag = (value?: string | null): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const quizHasWorkspaceOwnership = (quiz: Quiz): boolean => (
  normalizeWorkspaceId(quiz.workspace_id) != null ||
  normalizeWorkspaceTag(quiz.workspace_tag) != null
)

const quizMatchesVisibility = (quiz: Quiz, params: QuizListParams): boolean => {
  const workspaceId = normalizeWorkspaceId(params.workspace_id)
  const includeWorkspaceItems = params.include_workspace_items ?? false
  const quizWorkspaceId = normalizeWorkspaceId(quiz.workspace_id)
  const quizWorkspaceTag = normalizeWorkspaceTag(quiz.workspace_tag)
  const isWorkspaceOwned = quizHasWorkspaceOwnership(quiz)
  const matchesSpecificWorkspace = workspaceId != null && (
    quizWorkspaceId === workspaceId ||
    quizWorkspaceTag === `workspace:${workspaceId}`
  )

  if (workspaceId != null) {
    if (includeWorkspaceItems) {
      return !isWorkspaceOwned || matchesSpecificWorkspace
    }
    return matchesSpecificWorkspace
  }

  if (includeWorkspaceItems) {
    return true
  }

  return !isWorkspaceOwned
}

type OptimisticQuizContext = {
  previousLists: Array<[readonly unknown[], QuizListCacheValue | undefined]>
  previousDetail: Quiz | undefined
  tempId?: number
}

const extractQuizListParams = (queryKey: readonly unknown[]): QuizListParams => {
  if (!Array.isArray(queryKey) || queryKey[0] !== "quizzes:list") return {}
  const rawParams = queryKey[1]
  if (!rawParams || typeof rawParams !== "object") return {}
  return rawParams as QuizListParams
}

const sanitizeQuizPatch = (patch: QuizUpdate): Partial<Quiz> => {
  const { expected_version: _ignoredExpectedVersion, ...rest } = patch
  return Object.fromEntries(
    Object.entries(rest).filter(([, value]) => value !== undefined)
  ) as Partial<Quiz>
}

const applyQuizPatch = (quiz: Quiz, patch: Partial<Quiz>): Quiz => ({
  ...quiz,
  ...patch
})

/**
 * Helper to check if quizzes feature is available
 */
export function useQuizzesEnabled() {
  const isOnline = useServerOnline()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const quizzesUnsupported = !capsLoading && !!capabilities && !capabilities.hasQuizzes

  return {
    isOnline,
    capsLoading,
    quizzesUnsupported,
    quizzesEnabled: isOnline && !quizzesUnsupported
  }
}

// --- Quiz Queries ---

/**
 * Hook for fetching quiz list
 */
export function useQuizzesQuery(params: QuizListParams = {}, options?: UseQuizQueriesOptions) {
  const { quizzesEnabled } = useQuizzesEnabled()

  return useQuery({
    queryKey: ["quizzes:list", params],
    queryFn: () => listQuizzes(params),
    enabled: options?.enabled ?? quizzesEnabled,
    staleTime: QUIZ_QUERY_STALE_TIME_MS,
    refetchOnWindowFocus: false
  })
}

/**
 * Hook for fetching a single quiz with its questions
 */
export function useQuizQuery(quizId: number | null | undefined, options?: UseQuizQueriesOptions) {
  const { quizzesEnabled } = useQuizzesEnabled()

  return useQuery({
    queryKey: ["quizzes:detail", quizId],
    queryFn: () => getQuiz(quizId!),
    enabled: (options?.enabled ?? quizzesEnabled) && quizId != null,
    staleTime: QUIZ_QUERY_STALE_TIME_MS,
    refetchOnWindowFocus: false
  })
}

/**
 * Hook for fetching questions for a quiz
 */
export function useQuestionsQuery(
  quizId: number | null | undefined,
  params: QuestionListParams = {},
  options?: UseQuizQueriesOptions
) {
  const { quizzesEnabled } = useQuizzesEnabled()

  return useQuery({
    queryKey: ["quizzes:questions", quizId, params],
    queryFn: () => listQuestions(quizId!, params),
    enabled: (options?.enabled ?? quizzesEnabled) && quizId != null,
    staleTime: QUIZ_QUERY_STALE_TIME_MS,
    refetchOnWindowFocus: false
  })
}

// --- Quiz Mutations ---

/**
 * Hook for creating a quiz
 */
export function useCreateQuizMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["quizzes:create"],
    mutationFn: (payload: QuizCreate) => createQuiz(payload),
    onMutate: async (payload): Promise<OptimisticQuizContext> => {
      await qc.cancelQueries({ queryKey: ["quizzes:list"] })
      const previousLists = qc.getQueriesData<QuizListCacheValue>({ queryKey: ["quizzes:list"] })
      const tempId = -Date.now()
      const optimisticQuiz: Quiz = {
        id: tempId,
        name: payload.name,
        description: payload.description ?? null,
        workspace_id: payload.workspace_id ?? null,
        workspace_tag: payload.workspace_tag ?? null,
        media_id: payload.media_id ?? null,
        total_questions: 0,
        time_limit_seconds: payload.time_limit_seconds ?? null,
        passing_score: payload.passing_score ?? null,
        deleted: false,
        client_id: "optimistic",
        version: 0,
        created_at: new Date().toISOString(),
        last_modified: new Date().toISOString()
      }

      previousLists.forEach(([queryKey, previous]) => {
        if (!previous) return
        const params = extractQuizListParams(queryKey)
        const offset = typeof params.offset === "number" ? params.offset : 0
        if (offset > 0) return

        const limit = typeof params.limit === "number" && params.limit > 0 ? params.limit : undefined
        const nextItems = [optimisticQuiz, ...previous.items].filter((quiz) => quizMatchesVisibility(quiz, params))
        qc.setQueryData(queryKey, {
          ...previous,
          items: limit ? nextItems.slice(0, limit) : nextItems,
          count: previous.count + (quizMatchesVisibility(optimisticQuiz, params) ? 1 : 0)
        } satisfies QuizListCacheValue)
      })

      return {
        previousLists,
        previousDetail: undefined,
        tempId
      }
    },
    onError: (_error, _payload, context) => {
      context?.previousLists?.forEach(([queryKey, previous]) => {
        qc.setQueryData(queryKey, previous)
      })
    },
    onSuccess: (quiz, _payload, context) => {
      qc.setQueriesData<QuizListCacheValue>({ queryKey: ["quizzes:list"] }, (current) => {
        if (!current) return current
        const replaced = current.items.map((item) =>
          item.id === context?.tempId ? quiz : item
        )
        return {
          ...current,
          items: replaced
        }
      })
      qc.setQueryData(["quizzes:detail", quiz.id], quiz)
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["quizzes:list"], refetchType: "active" })
    }
  })
}

/**
 * Hook for updating a quiz
 */
export function useUpdateQuizMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["quizzes:update"],
    mutationFn: (params: { quizId: number; update: QuizUpdate }) =>
      updateQuiz(params.quizId, params.update),
    onMutate: async (variables): Promise<OptimisticQuizContext> => {
      await qc.cancelQueries({ queryKey: ["quizzes:list"] })
      await qc.cancelQueries({ queryKey: ["quizzes:detail", variables.quizId] })

      const previousLists = qc.getQueriesData<QuizListCacheValue>({ queryKey: ["quizzes:list"] })
      const previousDetail = qc.getQueryData<Quiz>(["quizzes:detail", variables.quizId])
      const patch = sanitizeQuizPatch(variables.update)

      qc.setQueryData(["quizzes:detail", variables.quizId], (current: Quiz | undefined) => (
        current ? { ...current, ...patch } : current
      ))
      previousLists.forEach(([queryKey, previous]) => {
        if (!previous) return
        const params = extractQuizListParams(queryKey)
        const previousQuiz =
          previous.items.find((quiz) => quiz.id === variables.quizId) ?? previousDetail
        const nextQuiz = previousQuiz ? applyQuizPatch(previousQuiz, patch) : null
        const previousVisible = previousQuiz ? quizMatchesVisibility(previousQuiz, params) : false
        const nextVisible = nextQuiz ? quizMatchesVisibility(nextQuiz, params) : false
        const countDelta = Number(nextVisible) - Number(previousVisible)
        const nextItems = previous.items
          .map((quiz) => (quiz.id === variables.quizId ? applyQuizPatch(quiz, patch) : quiz))
          .filter((quiz) => quizMatchesVisibility(quiz, params))

        qc.setQueryData(queryKey, {
          ...previous,
          items: nextItems,
          count: Math.max(0, previous.count + countDelta)
        } satisfies QuizListCacheValue)
      })

      return {
        previousLists,
        previousDetail
      }
    },
    onError: (_error, variables, context) => {
      context?.previousLists?.forEach(([queryKey, previous]) => {
        qc.setQueryData(queryKey, previous)
      })
      qc.setQueryData(["quizzes:detail", variables.quizId], context?.previousDetail)
    },
    onSettled: (_data, _error, variables) => {
      qc.invalidateQueries({ queryKey: ["quizzes:list"], refetchType: "active" })
      qc.invalidateQueries({ queryKey: ["quizzes:detail", variables.quizId], refetchType: "active" })
    }
  })
}

/**
 * Hook for deleting a quiz
 */
export function useDeleteQuizMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["quizzes:delete"],
    mutationFn: (params: { quizId: number; version: number }) =>
      deleteQuiz(params.quizId, params.version),
    onMutate: async (variables): Promise<OptimisticQuizContext> => {
      await qc.cancelQueries({ queryKey: ["quizzes:list"] })
      await qc.cancelQueries({ queryKey: ["quizzes:detail", variables.quizId] })

      const previousLists = qc.getQueriesData<QuizListCacheValue>({ queryKey: ["quizzes:list"] })
      const previousDetail = qc.getQueryData<Quiz>(["quizzes:detail", variables.quizId])

      qc.setQueriesData<QuizListCacheValue>({ queryKey: ["quizzes:list"] }, (current) => {
        if (!current) return current
        const nextItems = current.items.filter((quiz) => quiz.id !== variables.quizId)
        return {
          ...current,
          items: nextItems,
          count: Math.max(0, current.count - (nextItems.length === current.items.length ? 0 : 1))
        }
      })
      qc.removeQueries({ queryKey: ["quizzes:detail", variables.quizId], exact: true })

      return {
        previousLists,
        previousDetail
      }
    },
    onError: (_error, variables, context) => {
      context?.previousLists?.forEach(([queryKey, previous]) => {
        qc.setQueryData(queryKey, previous)
      })
      if (context?.previousDetail) {
        qc.setQueryData(["quizzes:detail", variables.quizId], context.previousDetail)
      }
    },
    onSettled: (_data, _error, variables) => {
      qc.invalidateQueries({ queryKey: ["quizzes:list"], refetchType: "active" })
      qc.invalidateQueries({ queryKey: ["quizzes:detail", variables.quizId], refetchType: "active" })
    }
  })
}

// --- Question Mutations ---

/**
 * Hook for creating a question
 */
export function useCreateQuestionMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["quizzes:question:create"],
    mutationFn: (params: { quizId: number; question: QuestionCreate }) =>
      createQuestion(params.quizId, params.question),
    onSuccess: (_, variables) => {
      qc.invalidateQueries({ queryKey: ["quizzes:questions", variables.quizId] })
      qc.invalidateQueries({ queryKey: ["quizzes:detail", variables.quizId] })
    }
  })
}

/**
 * Hook for updating a question
 */
export function useUpdateQuestionMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["quizzes:question:update"],
    mutationFn: (params: { quizId: number; questionId: number; update: QuestionUpdate }) =>
      updateQuestion(params.quizId, params.questionId, params.update),
    onSuccess: (_, variables) => {
      qc.invalidateQueries({ queryKey: ["quizzes:questions", variables.quizId] })
    }
  })
}

/**
 * Hook for deleting a question
 */
export function useDeleteQuestionMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["quizzes:question:delete"],
    mutationFn: (params: { quizId: number; questionId: number; version: number }) =>
      deleteQuestion(params.quizId, params.questionId, params.version),
    onSuccess: (_, variables) => {
      qc.invalidateQueries({ queryKey: ["quizzes:questions", variables.quizId] })
      qc.invalidateQueries({ queryKey: ["quizzes:detail", variables.quizId] })
    }
  })
}

// --- Attempt Queries & Mutations ---

/**
 * Hook for fetching attempt list
 */
export function useAttemptsQuery(params: AttemptListParams = {}, options?: UseQuizQueriesOptions) {
  const { quizzesEnabled } = useQuizzesEnabled()

  return useQuery({
    queryKey: ["quizzes:attempts", params],
    queryFn: () => listAttempts(params),
    enabled: options?.enabled ?? quizzesEnabled,
    staleTime: ATTEMPT_QUERY_STALE_TIME_MS,
    refetchOnWindowFocus: false
  })
}

/**
 * Hook for fetching all attempts (auto-paginates through the attempts API)
 */
export function useAllAttemptsQuery(params: Omit<AttemptListParams, "limit" | "offset"> = {}, options?: UseQuizQueriesOptions) {
  const { quizzesEnabled } = useQuizzesEnabled()

  return useQuery({
    queryKey: ["quizzes:attempts:all", params],
    queryFn: async () => {
      const pageSize = 200
      let offset = 0
      let total = 0
      const items: QuizAttempt[] = []

      do {
        const page = await listAttempts({
          ...params,
          limit: pageSize,
          offset
        })
        total = page.count
        items.push(...page.items)

        if (page.items.length === 0) {
          break
        }

        offset += page.items.length
      } while (items.length < total)

      return {
        items,
        count: total
      }
    },
    enabled: options?.enabled ?? quizzesEnabled,
    staleTime: ATTEMPT_QUERY_STALE_TIME_MS,
    refetchOnWindowFocus: false
  })
}

/**
 * Hook for fetching a single attempt
 */
export function useAttemptQuery(
  attemptId: number | null | undefined,
  params: {
    includeQuestions?: boolean
    includeAnswers?: boolean
  } = {},
  options?: UseQuizQueriesOptions
) {
  const { quizzesEnabled } = useQuizzesEnabled()

  return useQuery({
    queryKey: ["quizzes:attempt", attemptId, params],
    queryFn: () => getAttempt(attemptId!, {
      include_questions: params.includeQuestions ? true : undefined,
      include_answers: params.includeAnswers ? true : undefined
    }),
    enabled: (options?.enabled ?? quizzesEnabled) && attemptId != null,
    staleTime: ATTEMPT_QUERY_STALE_TIME_MS,
    refetchOnWindowFocus: false
  })
}

export function useAttemptRemediationConversionsQuery(
  attemptId: number | null | undefined,
  options?: UseQuizQueriesOptions
) {
  const { quizzesEnabled } = useQuizzesEnabled()

  return useQuery({
    queryKey: ["quizzes:attempt:remediation-conversions", attemptId ?? null],
    queryFn: ({ signal }) => listAttemptRemediationConversions(attemptId!, { signal }),
    enabled: (options?.enabled ?? quizzesEnabled) && attemptId != null,
    staleTime: ATTEMPT_QUERY_STALE_TIME_MS,
    refetchOnWindowFocus: false
  })
}

export function useQuizAttemptQuestionAssistantQuery(
  attemptId: number | null | undefined,
  questionId: number | null | undefined,
  options?: UseQuizQueriesOptions
) {
  const { quizzesEnabled } = useQuizzesEnabled()

  return useQuery({
    queryKey: ["quizzes:assistant", attemptId ?? null, questionId ?? null],
    queryFn: ({ signal }) => getQuizAttemptQuestionAssistant(attemptId!, questionId!, { signal }),
    enabled: (options?.enabled ?? quizzesEnabled) && attemptId != null && questionId != null,
    staleTime: ATTEMPT_QUERY_STALE_TIME_MS,
    refetchOnWindowFocus: false
  })
}

export function useQuizAttemptQuestionAssistantRespondMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["quizzes:assistant:respond"],
    mutationFn: (params: {
      attemptId: number
      questionId: number
      request: StudyAssistantRespondRequest
      signal?: AbortSignal
    }) => {
      const cached = qc.getQueryData<StudyAssistantContextResponse>([
        "quizzes:assistant",
        params.attemptId,
        params.questionId
      ])
      const request = params.request.expected_thread_version != null
        ? params.request
        : cached?.thread?.version != null
          ? {
              ...params.request,
              expected_thread_version: cached.thread.version
            }
          : params.request

      return respondQuizAttemptQuestionAssistant(
        params.attemptId,
        params.questionId,
        request,
        params.signal ? { signal: params.signal } : undefined
      )
    },
    onSuccess: (response, variables) => {
      qc.setQueryData<StudyAssistantContextResponse>(
        ["quizzes:assistant", variables.attemptId, variables.questionId],
        (current) => ({
          thread: response.thread,
          messages: current
            ? [...current.messages, response.user_message, response.assistant_message]
            : [response.user_message, response.assistant_message],
          context_snapshot: response.context_snapshot,
          available_actions: current?.available_actions ?? [...DEFAULT_STUDY_ASSISTANT_ACTIONS]
        })
      )
    },
    onError: (error) => {
      console.error("Failed to respond with quiz question assistant:", error)
    }
  })
}

export function useConvertAttemptRemediationQuestionsMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["quizzes:attempt:remediation-convert"],
    mutationFn: (params: {
      attemptId: number
      request: QuizRemediationConvertRequest
      signal?: AbortSignal
    }) => (
      convertAttemptRemediationQuestions(
        params.attemptId,
        params.request,
        params.signal ? { signal: params.signal } : undefined
      )
    ),
    onSuccess: (response: QuizRemediationConvertResponse, variables) => {
      qc.invalidateQueries({ queryKey: ["flashcards:decks"], refetchType: "active" })
      qc.setQueryData<QuizRemediationConversionListResponse | undefined>(
        ["quizzes:attempt:remediation-conversions", variables.attemptId],
        (current) => {
          if (!current) return current

          const nextItems = current.items
            .filter((item) => {
              const incomingQuestionIds = new Set(response.results.map((result) => result.question_id))
              if (!incomingQuestionIds.has(item.question_id)) return true
              return item.status !== "active"
            })

          response.results.forEach((result) => {
            if (result.conversion) {
              nextItems.push(result.conversion)
            }
          })

          const supersededCount = current.superseded_count
            + response.results.filter((result) => result.status === "superseded_and_created").length
          return {
            attempt_id: current.attempt_id,
            items: nextItems,
            count: nextItems.length,
            superseded_count: supersededCount
          }
        }
      )
      qc.invalidateQueries({
        queryKey: ["quizzes:attempt:remediation-conversions", variables.attemptId],
        refetchType: "active"
      })
    }
  })
}

/**
 * Hook for starting a quiz attempt
 */
export function useStartAttemptMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["quizzes:attempt:start"],
    mutationFn: (quizId: number) => startAttempt(quizId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["quizzes:attempts"] })
    }
  })
}

/**
 * Hook for submitting a quiz attempt
 */
export function useSubmitAttemptMutation() {
  const qc = useQueryClient()
  const markMilestone = useMilestoneStore((state) => state.markMilestone)

  return useMutation({
    mutationKey: ["quizzes:attempt:submit"],
    mutationFn: (params: { attemptId: number; answers: QuizAnswerInput[] }) =>
      submitAttempt(params.attemptId, params.answers),
    onSuccess: () => {
      markMilestone("first_quiz_taken")
      qc.invalidateQueries({ queryKey: ["quizzes:attempts"] })
    }
  })
}

// --- AI Generation ---

/**
 * Hook for generating a quiz from media
 */
export function useGenerateQuizMutation() {
  const qc = useQueryClient()

  return useMutation({
    mutationKey: ["quizzes:generate"],
    mutationFn: (params: { request: QuizGenerateRequest; signal?: AbortSignal }) =>
      generateQuiz(params.request, { signal: params.signal }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["quizzes:list"] })
    }
  })
}

export function useGenerateRemediationQuizMutation() {
  const qc = useQueryClient()
  type GenerateRemediationQuizMutationInput = {
    attemptId: number
    questionIds: number[]
    numQuestions?: number
    questionTypes?: QuestionType[]
    difficulty?: "easy" | "medium" | "hard" | "mixed"
    focusTopics?: string[]
    model?: string
    apiProvider?: string
    workspaceTag?: string | null
    signal?: AbortSignal
  }

  return useMutation({
    mutationKey: ["quizzes:generate:remediation"],
    mutationFn: (params: GenerateRemediationQuizMutationInput) => {
      const request: QuizRemediationGenerateRequest = {
        attemptId: params.attemptId,
        questionIds: params.questionIds
      }
      if (params.numQuestions !== undefined) request.num_questions = params.numQuestions
      if (params.questionTypes !== undefined) request.question_types = params.questionTypes
      if (params.difficulty !== undefined) request.difficulty = params.difficulty
      if (params.focusTopics !== undefined) request.focus_topics = params.focusTopics
      if (params.model !== undefined) request.model = params.model
      if (params.apiProvider !== undefined) request.api_provider = params.apiProvider
      if (params.workspaceTag !== undefined) request.workspace_tag = params.workspaceTag

      return params.signal
        ? generateRemediationQuiz(request, { signal: params.signal })
        : generateRemediationQuiz(request)
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["quizzes:list"] })
    }
  })
}

// Re-export types
export type {
  Quiz,
  Question,
  QuizCreate,
  QuizUpdate,
  QuestionCreate,
  QuestionUpdate,
  QuizGenerateRequest
}
