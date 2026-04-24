import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import { createResourceClient } from "@/services/resource-client"
import type {
  DeckSchedulerSettingsEnvelope,
  DeckSchedulerType,
  DeckReviewPromptSide,
  StudyAssistantContextResponse,
  StudyAssistantRespondRequest,
  StudyAssistantRespondResponse
} from "@/services/flashcards"

const quizzesClient = createResourceClient({
  basePath: "/api/v1/quizzes" as AllowedPath
})

export const QUIZ_GENERATION_TIMEOUT_MS = 120000

const quizAttemptsClient = createResourceClient({
  basePath: "/api/v1/quizzes/attempts" as AllowedPath,
  updateMethod: "PUT"
})

const getQuestionsClient = (quizId: number) =>
  createResourceClient({
    basePath: `/api/v1/quizzes/${quizId}/questions` as AllowedPath
  })

const getQuizAttemptsClient = (quizId: number) =>
  createResourceClient({
    basePath: `/api/v1/quizzes/${quizId}/attempts` as AllowedPath
  })

// Question types
export type QuestionType = "multiple_choice" | "multi_select" | "matching" | "true_false" | "fill_blank"
export type AnswerValue = number | string | number[] | Record<string, string>
export type QuizGenerateSourceType =
  | "media"
  | "note"
  | "flashcard_deck"
  | "flashcard_card"
  | "quiz_attempt"
  | "quiz_attempt_question"
export type QuizGenerateSource = {
  source_type: QuizGenerateSourceType
  source_id: string
}
export type SourceCitation = {
  source_type?: QuizGenerateSourceType | null
  source_id?: string | null
  label?: string | null
  quote?: string | null
  media_id?: number | null
  chunk_id?: string | null
  timestamp_seconds?: number | null
  source_url?: string | null
}

// Quiz container
export type Quiz = {
  id: number
  name: string
  description?: string | null
  workspace_id?: string | null
  workspace_tag?: string | null
  media_id?: number | null
  source_bundle_json?: QuizGenerateSource[] | null
  total_questions: number
  time_limit_seconds?: number | null
  passing_score?: number | null
  deleted: boolean
  client_id: string
  version: number
  created_at?: string | null
  last_modified?: string | null
}

// Individual question
export type QuestionBase = {
  id: number
  quiz_id: number
  question_type: QuestionType
  question_text: string
  options?: string[] | null
  hint?: string | null
  hint_penalty_points?: number | null
  source_citations?: SourceCitation[] | null
  points: number
  order_index: number
  tags?: string[] | null
  deleted: boolean
  client_id: string
  version: number
  created_at?: string | null
  last_modified?: string | null
}

export type QuestionPublic = QuestionBase

export type QuestionAdmin = QuestionBase & {
  correct_answer: AnswerValue
  explanation?: string | null
}

export type Question = QuestionAdmin

// Quiz attempt answer
export type QuizAnswer = {
  question_id: number
  user_answer: AnswerValue
  is_correct: boolean
  correct_answer?: AnswerValue
  explanation?: string | null
  hint_used?: boolean | null
  hint_penalty_points?: number | null
  source_citations?: SourceCitation[] | null
  points_awarded?: number | null
  time_spent_ms?: number | null
}

export type QuizAnswerInput = {
  question_id: number
  user_answer: AnswerValue
  hint_used?: boolean | null
  time_spent_ms?: number | null
}

// Quiz attempt/session
export type QuizAttempt = {
  id: number
  quiz_id: number
  started_at: string
  completed_at?: string | null
  score?: number | null
  total_possible: number
  time_spent_seconds?: number | null
  answers: QuizAnswer[]
  questions?: QuestionPublic[]
}

// Create types
export type QuizCreate = {
  name: string
  description?: string | null
  workspace_id?: string | null
  workspace_tag?: string | null
  media_id?: number | null
  source_bundle_json?: QuizGenerateSource[] | null
  time_limit_seconds?: number | null
  passing_score?: number | null
}

export type QuizUpdate = {
  name?: string | null
  description?: string | null
  workspace_id?: string | null
  workspace_tag?: string | null
  media_id?: number | null
  source_bundle_json?: QuizGenerateSource[] | null
  time_limit_seconds?: number | null
  passing_score?: number | null
  expected_version?: number | null
}

export type QuestionCreate = {
  question_type: QuestionType
  question_text: string
  options?: string[] | null
  correct_answer: AnswerValue
  explanation?: string | null
  hint?: string | null
  hint_penalty_points?: number | null
  source_citations?: SourceCitation[] | null
  points?: number
  order_index?: number
  tags?: string[] | null
}

export type QuestionUpdate = {
  question_type?: QuestionType | null
  question_text?: string | null
  options?: string[] | null
  correct_answer?: AnswerValue | null
  explanation?: string | null
  hint?: string | null
  hint_penalty_points?: number | null
  source_citations?: SourceCitation[] | null
  points?: number | null
  order_index?: number | null
  tags?: string[] | null
  expected_version?: number | null
}

// AI generation request
type QuizGenerateRequestBase = {
  num_questions?: number
  question_types?: QuestionType[]
  difficulty?: "easy" | "medium" | "hard" | "mixed"
  focus_topics?: string[]
  model?: string
  workspace_id?: string | null
  workspace_tag?: string | null
}

type QuizGenerateRequestWithMedia = QuizGenerateRequestBase & {
  media_id: number
  sources?: QuizGenerateSource[]
}

type QuizGenerateRequestWithSources = QuizGenerateRequestBase & {
  sources: QuizGenerateSource[]
  media_id?: number
}

export type QuizGenerateRequest = QuizGenerateRequestWithMedia | QuizGenerateRequestWithSources

export type QuizRemediationGenerateRequest = {
  attemptId: number
  questionIds: number[]
  num_questions?: number
  question_types?: QuestionType[]
  difficulty?: "easy" | "medium" | "hard" | "mixed"
  focus_topics?: string[]
  model?: string
  api_provider?: string
  workspace_tag?: string | null
}

export type QuizRemediationConversionSummary = {
  id: number
  attempt_id: number
  quiz_id: number
  question_id: number
  status: "active" | "superseded"
  orphaned: boolean
  superseded_count: number
  superseded_by_id?: number | null
  target_deck_id?: number | null
  target_deck_name_snapshot?: string | null
  flashcard_count: number
  flashcard_uuids_json: string[]
  source_ref_id?: string | null
  created_at?: string | null
  last_modified?: string | null
  client_id: string
  version: number
}

export type QuizRemediationConversionListResponse = {
  attempt_id: number
  items: QuizRemediationConversionSummary[]
  count: number
  superseded_count: number
}

export type QuizRemediationTargetDeck = {
  id: number
  name: string
}

export type QuizRemediationConvertRequest = {
  question_ids: number[]
  target_deck_id?: number | null
  create_deck_name?: string | null
  create_deck_review_prompt_side?: DeckReviewPromptSide
  create_deck_scheduler_type?: DeckSchedulerType | null
  create_deck_scheduler_settings?: DeckSchedulerSettingsEnvelope | null
  replace_active?: boolean
}

export type QuizRemediationConvertResult = {
  question_id: number
  status: "created" | "already_exists" | "superseded_and_created" | "failed"
  conversion?: QuizRemediationConversionSummary | null
  flashcard_uuids: string[]
  error?: string | null
}

export type QuizRemediationConvertResponse = {
  attempt_id: number
  quiz_id: number
  target_deck?: QuizRemediationTargetDeck | null
  results: QuizRemediationConvertResult[]
  created_flashcard_uuids: string[]
}

// List response types
export type QuizListResponse = {
  items: Quiz[]
  count: number
}

export type QuestionListResponse = {
  items: Array<QuestionPublic | QuestionAdmin>
  count: number
}

export type AttemptListResponse = {
  items: QuizAttempt[]
  count: number
}

// List params
export type QuizListParams = {
  media_id?: number | null
  q?: string | null
  workspace_id?: string | null
  workspace_tag?: string | null
  include_workspace_items?: boolean | null
  limit?: number
  offset?: number
}

export type QuestionListParams = {
  include_answers?: boolean
  q?: string | null
  limit?: number
  offset?: number
}

export type AttemptListParams = {
  quiz_id?: number | null
  limit?: number
  offset?: number
}

export type QuizGenerateResponse = {
  quiz: Quiz
  questions: QuestionAdmin[]
}

export type QuizImportQuestion = {
  question_type: QuestionType
  question_text: string
  options?: string[] | null
  correct_answer: AnswerValue
  explanation?: string | null
  hint?: string | null
  hint_penalty_points?: number | null
  source_citations?: SourceCitation[] | null
  points?: number
  order_index?: number
  tags?: string[] | null
}

export type QuizImportEntry = {
  quiz: QuizCreate
  questions: QuizImportQuestion[]
}

export type QuizImportRequest = {
  export_format?: string | null
  quizzes: QuizImportEntry[]
}

export type QuizImportItemResult = {
  source_index: number
  quiz_id: number
  imported_questions: number
  failed_questions: number
}

export type QuizImportError = {
  source_index: number
  quiz_name?: string | null
  question_index?: number | null
  error: string
}

export type QuizImportResponse = {
  imported_quizzes: number
  failed_quizzes: number
  imported_questions: number
  failed_questions: number
  items: QuizImportItemResult[]
  errors: QuizImportError[]
}

// --- Quizzes CRUD ---

export async function listQuizzes(params: QuizListParams = {}): Promise<QuizListResponse> {
  return await quizzesClient.list<QuizListResponse>({
    media_id: params.media_id,
    q: params.q,
    workspace_id: params.workspace_id,
    workspace_tag: params.workspace_tag,
    include_workspace_items: params.include_workspace_items ?? false,
    limit: params.limit,
    offset: params.offset
  })
}

export async function createQuiz(input: QuizCreate): Promise<Quiz> {
  return await quizzesClient.create<Quiz>(input)
}

export async function getQuiz(quizId: number): Promise<Quiz> {
  return await quizzesClient.get<Quiz>(quizId)
}

export async function updateQuiz(quizId: number, input: QuizUpdate): Promise<void> {
  await quizzesClient.update<void>(quizId, input)
}

export async function deleteQuiz(quizId: number, expectedVersion: number): Promise<void> {
  await quizzesClient.remove<void>(quizId, {
    expected_version: expectedVersion
  })
}

// --- Questions CRUD ---

export async function listQuestions(
  quizId: number,
  params: QuestionListParams = {}
): Promise<QuestionListResponse> {
  return await getQuestionsClient(quizId).list<QuestionListResponse>({
    include_answers: params.include_answers ? true : undefined,
    q: params.q,
    limit: params.limit,
    offset: params.offset
  })
}

export async function createQuestion(quizId: number, input: QuestionCreate): Promise<Question> {
  return await getQuestionsClient(quizId).create<Question>(input)
}

export async function updateQuestion(
  quizId: number,
  questionId: number,
  input: QuestionUpdate
): Promise<void> {
  await getQuestionsClient(quizId).update<void>(questionId, input)
}

export async function deleteQuestion(
  quizId: number,
  questionId: number,
  expectedVersion: number
): Promise<void> {
  await getQuestionsClient(quizId).remove<void>(questionId, {
    expected_version: expectedVersion
  })
}

// --- Quiz Attempts ---

export async function startAttempt(quizId: number): Promise<QuizAttempt> {
  return await getQuizAttemptsClient(quizId).create<QuizAttempt>({})
}

export async function submitAttempt(
  attemptId: number,
  answers: QuizAnswerInput[]
): Promise<QuizAttempt> {
  return await quizAttemptsClient.update<QuizAttempt>(attemptId, { answers })
}

export async function listAttempts(params: AttemptListParams = {}): Promise<AttemptListResponse> {
  return await quizAttemptsClient.list<AttemptListResponse>({
    quiz_id: params.quiz_id,
    limit: params.limit,
    offset: params.offset
  })
}

export async function getAttempt(
  attemptId: number,
  params?: {
    include_questions?: boolean
    include_answers?: boolean
  }
): Promise<QuizAttempt> {
  return await quizAttemptsClient.get<QuizAttempt>(attemptId, {
    include_questions: params?.include_questions ? true : undefined,
    include_answers: params?.include_answers ? true : undefined
  })
}

export async function listAttemptRemediationConversions(
  attemptId: number,
  options?: { signal?: AbortSignal }
): Promise<QuizRemediationConversionListResponse> {
  return await bgRequest<QuizRemediationConversionListResponse, AllowedPath, "GET">({
    path: `/api/v1/quizzes/attempts/${attemptId}/remediation-conversions` as AllowedPath,
    method: "GET",
    abortSignal: options?.signal
  })
}

export async function convertAttemptRemediationQuestions(
  attemptId: number,
  input: QuizRemediationConvertRequest,
  options?: { signal?: AbortSignal }
): Promise<QuizRemediationConvertResponse> {
  return await bgRequest<QuizRemediationConvertResponse, AllowedPath, "POST">({
    path: `/api/v1/quizzes/attempts/${attemptId}/remediation-conversions/convert` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input,
    abortSignal: options?.signal
  })
}

export async function getQuizAttemptQuestionAssistant(
  attemptId: number,
  questionId: number,
  options?: { signal?: AbortSignal }
): Promise<StudyAssistantContextResponse> {
  return await bgRequest<StudyAssistantContextResponse, AllowedPath, "GET">({
    path: `/api/v1/quizzes/attempts/${attemptId}/questions/${questionId}/assistant` as AllowedPath,
    method: "GET",
    abortSignal: options?.signal
  })
}

export async function respondQuizAttemptQuestionAssistant(
  attemptId: number,
  questionId: number,
  input: StudyAssistantRespondRequest,
  options?: { signal?: AbortSignal }
): Promise<StudyAssistantRespondResponse> {
  return await bgRequest<StudyAssistantRespondResponse, AllowedPath, "POST">({
    path: `/api/v1/quizzes/attempts/${attemptId}/questions/${questionId}/assistant/respond` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input,
    abortSignal: options?.signal
  })
}

// --- AI Generation ---

export async function generateQuiz(
  request: QuizGenerateRequest,
  options?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<QuizGenerateResponse> {
  const timeoutMs =
    typeof options?.timeoutMs === "number" && options.timeoutMs > 0
      ? options.timeoutMs
      : QUIZ_GENERATION_TIMEOUT_MS

  return await bgRequest<QuizGenerateResponse, AllowedPath, "POST">({
    path: "/api/v1/quizzes/generate",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: request,
    abortSignal: options?.signal,
    timeoutMs
  })
}

export function buildQuizAttemptQuestionSources(
  attemptId: number,
  questionIds: number[]
): QuizGenerateSource[] {
  return questionIds.map((questionId) => ({
    source_type: "quiz_attempt_question",
    source_id: `${attemptId}:${questionId}`
  }))
}

export async function generateRemediationQuiz(
  request: QuizRemediationGenerateRequest,
  options?: { signal?: AbortSignal; timeoutMs?: number }
): Promise<QuizGenerateResponse> {
  return await generateQuiz(
    {
      num_questions: request.num_questions,
      question_types: request.question_types,
      difficulty: request.difficulty,
      focus_topics: request.focus_topics,
      model: request.model,
      api_provider: request.api_provider,
      workspace_tag: request.workspace_tag,
      sources: buildQuizAttemptQuestionSources(request.attemptId, request.questionIds)
    },
    options
  )
}

export async function importQuizzesJson(
  request: QuizImportRequest,
  options?: { signal?: AbortSignal }
): Promise<QuizImportResponse> {
  return await bgRequest<QuizImportResponse, AllowedPath, "POST">({
    path: "/api/v1/quizzes/import/json",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: request,
    abortSignal: options?.signal
  })
}
