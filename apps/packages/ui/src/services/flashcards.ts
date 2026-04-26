import { bgRequest, bgUpload } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import {
  buildQuery,
  createResourceClient
} from "@/services/resource-client"

const decksClient = createResourceClient({
  basePath: "/api/v1/flashcards/decks" as AllowedPath
})

const flashcardsClient = createResourceClient({
  basePath: "/api/v1/flashcards" as AllowedPath
})

const flashcardTemplatesClient = createResourceClient({
  basePath: "/api/v1/flashcards/templates" as AllowedPath
})

const flashcardTagsClient = createResourceClient({
  basePath: "/api/v1/flashcards/tags" as AllowedPath
})

export const FLASHCARD_GENERATION_TIMEOUT_MS = 120000

export type DeckSchedulerSettings = {
  new_steps_minutes: number[]
  relearn_steps_minutes: number[]
  graduating_interval_days: number
  easy_interval_days: number
  easy_bonus: number
  interval_modifier: number
  max_interval_days: number
  leech_threshold: number
  enable_fuzz: boolean
}

export type DeckSchedulerType = "sm2_plus" | "fsrs"
export type DeckReviewPromptSide = "front" | "back"

export type FsrsSchedulerSettings = {
  target_retention: number
  maximum_interval_days: number
  enable_fuzz: boolean
}

export type DeckSchedulerSettingsEnvelope = {
  sm2_plus: DeckSchedulerSettings
  fsrs: FsrsSchedulerSettings
}

export type DeckSchedulerSettingsEnvelopeUpdate = {
  sm2_plus?: Partial<DeckSchedulerSettings>
  fsrs?: Partial<FsrsSchedulerSettings>
}

export type FlashcardIntervalPreviews = {
  again: string
  hard: string
  good: string
  easy: string
}

export type StudyAssistantAction =
  | "explain"
  | "mnemonic"
  | "follow_up"
  | "fact_check"
  | "freeform"

export type StudyAssistantInputModality = "text" | "voice_transcript"

export type StudyAssistantThreadSummary = {
  id: number
  context_type: "flashcard" | "quiz_attempt_question"
  flashcard_uuid?: string | null
  quiz_attempt_id?: number | null
  question_id?: number | null
  last_message_at?: string | null
  message_count: number
  deleted: boolean
  client_id: string
  version: number
  created_at?: string | null
  last_modified?: string | null
}

export type StudyAssistantMessage = {
  id: number
  thread_id: number
  role: "user" | "assistant"
  action_type: StudyAssistantAction
  input_modality: StudyAssistantInputModality
  content: string
  structured_payload: Record<string, unknown>
  context_snapshot: Record<string, unknown>
  provider?: string | null
  model?: string | null
  created_at?: string | null
  client_id: string
}

export type StudyAssistantFactCheckPayload = {
  verdict: "correct" | "partially_correct" | "incorrect"
  corrections: string[]
  missing_points: string[]
  next_prompt: string
}

export type StudyAssistantRespondRequest = {
  action: StudyAssistantAction
  message?: string | null
  input_modality?: StudyAssistantInputModality
  provider?: string | null
  model?: string | null
  expected_thread_version?: number | null
}

export type StudyAssistantContextResponse = {
  thread: StudyAssistantThreadSummary
  messages: StudyAssistantMessage[]
  context_snapshot: Record<string, unknown>
  available_actions: StudyAssistantAction[]
}

export type StudyAssistantRespondResponse = {
  thread: StudyAssistantThreadSummary
  user_message: StudyAssistantMessage
  assistant_message: StudyAssistantMessage
  structured_payload: Record<string, unknown>
  context_snapshot: Record<string, unknown>
}

export type StudyPackSourceType = "note" | "media" | "message"

export type StudyPackSourceSelection = {
  source_type: StudyPackSourceType
  source_id: string
  label?: string | null
  source_title?: string | null
  excerpt_text?: string | null
  locator?: Record<string, unknown> | null
}

export type StudyPackCreateJobRequest = {
  title: string
  workspace_id?: string | null
  deck_mode?: "new" | null
  source_items: StudyPackSourceSelection[]
}

export type StudyPackStatus = "active" | "superseded"

export type StudyPackJobApiStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"

export type StudyPackSummaryResponse = {
  id: number
  workspace_id?: string | null
  title: string
  deck_id?: number | null
  source_bundle_json: Record<string, unknown>
  generation_options_json?: Record<string, unknown> | null
  status: StudyPackStatus
  superseded_by_pack_id?: number | null
  created_at?: string | null
  last_modified?: string | null
  deleted: boolean
  client_id: string
  version: number
}

export type StudyPackJobSummaryResponse = {
  id: number
  status: StudyPackJobApiStatus
  domain: string
  queue: string
  job_type: string
}

export type StudyPackJobAcceptedResponse = {
  job: StudyPackJobSummaryResponse
}

export type StudyPackJobStatusResponse = {
  job: StudyPackJobSummaryResponse
  study_pack?: StudyPackSummaryResponse | null
  error?: string | null
}

// Minimal client types based on openapi.json
export type Deck = {
  id: number
  name: string
  description?: string | null
  workspace_id?: string | null
  review_prompt_side: DeckReviewPromptSide
  deleted: boolean
  client_id: string
  version: number
  created_at?: string | null
  last_modified?: string | null
  scheduler_type: DeckSchedulerType
  scheduler_settings_json?: string | null
  scheduler_settings: DeckSchedulerSettingsEnvelope
}

export type Flashcard = {
  uuid: string
  deck_id?: number | null
  front: string
  back: string
  notes?: string | null
  extra?: string | null
  is_cloze: boolean
  tags?: string[] | null
  ef: number
  interval_days: number
  repetitions: number
  lapses: number
  due_at?: string | null
  created_at?: string | null
  last_reviewed_at?: string | null
  queue_state: "new" | "learning" | "review" | "relearning" | "suspended"
  step_index?: number | null
  suspended_reason?: "manual" | "leech" | null
  last_modified?: string | null
  deleted: boolean
  client_id: string
  version: number
  model_type: "basic" | "basic_reverse" | "cloze"
  reverse: boolean
  scheduler_type?: DeckSchedulerType | null
  source_ref_type?: "media" | "message" | "note" | "manual" | null
  source_ref_id?: string | null
  conversation_id?: string | null
  message_id?: string | null
  next_intervals?: FlashcardIntervalPreviews | null
}

export type FlashcardTemplateFieldTarget =
  | "front_template"
  | "back_template"
  | "notes_template"
  | "extra_template"

export type FlashcardTemplateModelType = "basic" | "basic_reverse" | "cloze"

export type FlashcardTemplatePlaceholderDefinition = {
  key: string
  label: string
  help_text?: string | null
  default_value?: string | null
  required?: boolean
  targets: FlashcardTemplateFieldTarget[]
}

export type FlashcardTemplate = {
  id: number
  name: string
  model_type: FlashcardTemplateModelType
  front_template: string
  back_template?: string | null
  notes_template?: string | null
  extra_template?: string | null
  placeholder_definitions: FlashcardTemplatePlaceholderDefinition[]
  created_at?: string | null
  last_modified?: string | null
  deleted: boolean
  client_id: string
  version: number
}

export type FlashcardTemplateCreate = {
  name: string
  model_type: FlashcardTemplateModelType
  front_template: string
  back_template?: string | null
  notes_template?: string | null
  extra_template?: string | null
  placeholder_definitions?: FlashcardTemplatePlaceholderDefinition[]
}

export type FlashcardTemplateUpdate = {
  name?: string | null
  model_type?: FlashcardTemplateModelType | null
  front_template?: string | null
  back_template?: string | null
  notes_template?: string | null
  extra_template?: string | null
  placeholder_definitions?: FlashcardTemplatePlaceholderDefinition[] | null
  expected_version?: number | null
}

export type FlashcardTemplateListResponse = {
  items: FlashcardTemplate[]
  count: number
  total?: number | null
}

export type DeckUpdate = {
  name?: string | null
  description?: string | null
  workspace_id?: string | null
  review_prompt_side?: DeckReviewPromptSide
  scheduler_type?: DeckSchedulerType | null
  scheduler_settings?: DeckSchedulerSettingsEnvelopeUpdate | null
  expected_version?: number | null
}

export type DeckCreateInput = {
  name: string
  description?: string | null
  workspace_id?: string | null
  review_prompt_side?: DeckReviewPromptSide
  scheduler_type?: DeckSchedulerType | null
  scheduler_settings?: DeckSchedulerSettingsEnvelope | null
}

export type FlashcardCreate = {
  deck_id?: number | null
  front: string
  back: string
  notes?: string | null
  extra?: string | null
  is_cloze?: boolean | null
  tags?: string[] | null
  source_ref_type?: "media" | "message" | "note" | "manual" | null
  source_ref_id?: string | null
  model_type?: Flashcard["model_type"] | null
  reverse?: boolean | null
}

export type FlashcardUpdate = {
  deck_id?: number | null
  front?: string | null
  back?: string | null
  notes?: string | null
  extra?: string | null
  is_cloze?: boolean | null
  tags?: string[] | null
  expected_version?: number | null
  model_type?: Flashcard["model_type"] | null
  reverse?: boolean | null
}

export type FlashcardBulkUpdateItem = FlashcardUpdate & {
  uuid: string
}

export type FlashcardBulkUpdateError = {
  code: "validation_error" | "not_found" | "conflict"
  message: string
  invalid_fields?: string[]
  invalid_deck_ids?: number[]
}

export type FlashcardBulkUpdateResult = {
  uuid: string
  status: "updated" | "validation_error" | "not_found" | "conflict"
  flashcard?: Flashcard | null
  error?: FlashcardBulkUpdateError | null
}

export type FlashcardBulkUpdateResponse = {
  results: FlashcardBulkUpdateResult[]
}

export type FlashcardResetSchedulingRequest = {
  expected_version: number
}

export type FlashcardListResponse = {
  items: Flashcard[]
  count: number
  total?: number | null
}

export type FlashcardReviewRequest = {
  card_uuid: string
  rating: number // 0-5
  answer_time_ms?: number | null
}

export type FlashcardGeneratedDraft = {
  front: string
  back: string
  tags?: string[] | null
  model_type?: "basic" | "basic_reverse" | "cloze"
  notes?: string | null
  extra?: string | null
}

export type FlashcardsGenerateRequest = {
  text: string
  num_cards?: number
  card_type?: "basic" | "basic_reverse" | "cloze"
  difficulty?: "easy" | "medium" | "hard" | "mixed"
  focus_topics?: string[] | null
  provider?: string | null
  model?: string | null
}

export type FlashcardsGenerateResponse = {
  flashcards: FlashcardGeneratedDraft[]
  count: number
}

export type FlashcardReviewResponse = {
  uuid: string
  ef: number
  interval_days: number
  repetitions: number
  lapses: number
  due_at?: string | null
  last_reviewed_at?: string | null
  last_modified?: string | null
  version: number
  scheduler_type: DeckSchedulerType
  queue_state: Flashcard["queue_state"]
  step_index?: number | null
  suspended_reason?: Flashcard["suspended_reason"]
  next_intervals: FlashcardIntervalPreviews
  review_session_id?: number | null
}

export type FlashcardNextReviewResponse = {
  card?: Flashcard | null
  selection_reason?: "learning_due" | "review_due" | "new" | "none" | null
}

export type FlashcardReviewSessionSummary = {
  id: number
  deck_id?: number | null
  review_mode: string
  tag_filter?: string | null
  scope_key: string
  status: string
  started_at?: string | null
  last_activity_at?: string | null
  completed_at?: string | null
  client_id: string
}

export type FlashcardTagSuggestionItem = {
  tag: string
  count: number
}

export type FlashcardTagSuggestionsResponse = {
  items: FlashcardTagSuggestionItem[]
  count: number
}

export type FlashcardsImportRequest = {
  content: string
  delimiter?: string | null
  has_header?: boolean | null
}

export type FlashcardsImportJsonRequest = {
  content: string
  filename?: string | null
}

export type FlashcardsImportApkgRequest = {
  bytes: Uint8Array
  filename?: string | null
}

export type DeckListParams = {
  workspace_id?: string | null
  include_workspace_items?: boolean | null
  signal?: AbortSignal
}

export type StructuredQaImportPreviewRequest = {
  content: string
}

export type FlashcardsImportError = {
  line?: number | null
  index?: number | null
  error: string
}

export type FlashcardsImportResponse = {
  imported: number
  items: Array<{
    uuid: string
    deck_id: number
  }>
  errors: FlashcardsImportError[]
}

export type StructuredQaImportPreviewDraft = {
  front: string
  back: string
  line_start: number
  line_end: number
  notes?: string | null
  extra?: string | null
  tags?: string[] | null
}

export type StructuredQaImportPreviewResponse = {
  drafts: StructuredQaImportPreviewDraft[]
  errors: Array<{ line?: number | null; error: string }>
  detected_format: "qa_labels"
  skipped_blocks: number
}

export type FlashcardsExportParams = {
  deck_id?: number | null
  tag?: string | null
  q?: string | null
  format?: "csv" | "apkg" | "json" | null
  include_reverse?: boolean | null
  delimiter?: string | null
  include_header?: boolean | null
  extended_header?: boolean | null
}

export type FlashcardDeckProgress = {
  deck_id: number
  deck_name: string
  total: number
  new: number
  learning: number
  due: number
  mature: number
}

export type FlashcardAnalyticsSummary = {
  reviewed_today: number
  retention_rate_today?: number | null
  lapse_rate_today?: number | null
  avg_answer_time_ms_today?: number | null
  study_streak_days: number
  generated_at: string
  decks: FlashcardDeckProgress[]
}

// Decks
export async function listDecks(options?: DeckListParams): Promise<Deck[]> {
  return await decksClient.list<Deck[]>({
    workspace_id: options?.workspace_id,
    include_workspace_items: options?.include_workspace_items ?? false
  }, {
    abortSignal: options?.signal
  })
}

export async function createDeck(
  input: DeckCreateInput,
  options?: { signal?: AbortSignal }
): Promise<Deck> {
  return await decksClient.create<Deck>(input, {
    abortSignal: options?.signal
  })
}

export async function updateDeck(
  deck_id: number,
  input: DeckUpdate,
  options?: { signal?: AbortSignal }
): Promise<Deck> {
  return await decksClient.update<Deck>(String(deck_id), input, {
    abortSignal: options?.signal
  })
}

// Flashcard templates
export async function listFlashcardTemplates(options?: {
  signal?: AbortSignal
}): Promise<FlashcardTemplateListResponse> {
  return await flashcardTemplatesClient.list<FlashcardTemplateListResponse>({}, {
    abortSignal: options?.signal
  })
}

export async function getFlashcardTemplate(
  template_id: number,
  options?: { signal?: AbortSignal }
): Promise<FlashcardTemplate> {
  return await flashcardTemplatesClient.get<FlashcardTemplate>(template_id, undefined, {
    abortSignal: options?.signal
  })
}

export async function createFlashcardTemplate(
  input: FlashcardTemplateCreate,
  options?: { signal?: AbortSignal }
): Promise<FlashcardTemplate> {
  return await flashcardTemplatesClient.create<FlashcardTemplate>(input, {
    abortSignal: options?.signal
  })
}

export async function updateFlashcardTemplate(
  template_id: number,
  input: FlashcardTemplateUpdate,
  options?: { signal?: AbortSignal }
): Promise<FlashcardTemplate> {
  return await flashcardTemplatesClient.update<FlashcardTemplate>(String(template_id), input, {
    abortSignal: options?.signal
  })
}

export async function deleteFlashcardTemplate(
  template_id: number,
  expected_version: number,
  options?: { signal?: AbortSignal }
): Promise<void> {
  await flashcardTemplatesClient.remove<void>(String(template_id), {
    expected_version
  }, {
    abortSignal: options?.signal
  })
}

// Flashcards CRUD
export async function listFlashcards(params: {
  deck_id?: number | null
  tag?: string | null
  due_status?: "new" | "learning" | "due" | "all" | null
  q?: string | null
  workspace_id?: string | null
  include_workspace_items?: boolean | null
  limit?: number
  offset?: number
  order_by?: "due_at" | "created_at" | null
}): Promise<FlashcardListResponse> {
  return await flashcardsClient.list<FlashcardListResponse>({
    deck_id: params.deck_id,
    tag: params.tag,
    due_status: params.due_status,
    q: params.q,
    workspace_id: params.workspace_id,
    include_workspace_items: params.include_workspace_items ?? false,
    limit: params.limit,
    offset: params.offset,
    order_by: params.order_by
  })
}

export async function listFlashcardTagSuggestions(params?: {
  q?: string | null
  limit?: number | null
  signal?: AbortSignal
}): Promise<FlashcardTagSuggestionsResponse> {
  const normalizedQuery = params?.q?.trim()

  return await flashcardTagsClient.list<FlashcardTagSuggestionsResponse>({
    ...(normalizedQuery ? { q: normalizedQuery } : {}),
    limit: params?.limit
  }, {
    abortSignal: params?.signal
  })
}

export async function createFlashcard(
  input: FlashcardCreate,
  options?: { signal?: AbortSignal }
): Promise<Flashcard> {
  return await flashcardsClient.create<Flashcard>(input, {
    abortSignal: options?.signal
  })
}

export async function createFlashcardsBulk(
  input: FlashcardCreate[],
  options?: { signal?: AbortSignal }
): Promise<FlashcardListResponse> {
  return await bgRequest<FlashcardListResponse, AllowedPath, "POST">({
    path: "/api/v1/flashcards/bulk",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input,
    abortSignal: options?.signal
  })
}

export async function updateFlashcardsBulk(
  input: FlashcardBulkUpdateItem[]
): Promise<FlashcardBulkUpdateResponse> {
  return await bgRequest<FlashcardBulkUpdateResponse, AllowedPath, "PATCH">({
    path: "/api/v1/flashcards/bulk",
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: input
  })
}

export async function getFlashcard(card_uuid: string): Promise<Flashcard> {
  return await flashcardsClient.get<Flashcard>(card_uuid)
}

export async function getFlashcardAssistant(
  card_uuid: string,
  options?: { signal?: AbortSignal }
): Promise<StudyAssistantContextResponse> {
  return await bgRequest<StudyAssistantContextResponse, AllowedPath, "GET">({
    path: `/api/v1/flashcards/${card_uuid}/assistant` as AllowedPath,
    method: "GET",
    abortSignal: options?.signal
  })
}

export async function respondFlashcardAssistant(
  card_uuid: string,
  input: StudyAssistantRespondRequest,
  options?: { signal?: AbortSignal }
): Promise<StudyAssistantRespondResponse> {
  return await bgRequest<StudyAssistantRespondResponse, AllowedPath, "POST">({
    path: `/api/v1/flashcards/${card_uuid}/assistant/respond` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input,
    abortSignal: options?.signal
  })
}

export async function updateFlashcard(card_uuid: string, input: FlashcardUpdate): Promise<void> {
  await flashcardsClient.update<void>(card_uuid, input)
}

export async function deleteFlashcard(card_uuid: string, expected_version: number): Promise<void> {
  await flashcardsClient.remove<void>(card_uuid, {
    expected_version
  })
}

export async function resetFlashcardScheduling(
  card_uuid: string,
  input: FlashcardResetSchedulingRequest
): Promise<Flashcard> {
  return await bgRequest<Flashcard, AllowedPath, "POST">({
    path: `/api/v1/flashcards/${card_uuid}/reset-scheduling` as AllowedPath,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input
  })
}

// Review
export async function reviewFlashcard(input: FlashcardReviewRequest): Promise<FlashcardReviewResponse> {
  return await bgRequest<FlashcardReviewResponse, AllowedPath, "POST">({
    path: "/api/v1/flashcards/review",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input
  })
}

export async function getNextReviewCard(
  deck_id?: number | null,
  params?: {
    workspace_id?: string | null
    include_workspace_items?: boolean | null
  }
): Promise<FlashcardNextReviewResponse> {
  const query = buildQuery({
    deck_id,
    workspace_id: params?.workspace_id,
    include_workspace_items: params?.include_workspace_items ?? false
  })
  return await bgRequest<FlashcardNextReviewResponse, AllowedPath, "GET">({
    path: `/api/v1/flashcards/review/next${query}` as AllowedPath,
    method: "GET"
  })
}

export async function listRecentFlashcardReviewSessions(params?: {
  deck_id?: number | null
  scope_key?: string | null
  status?: string | null
  limit?: number | null
  signal?: AbortSignal
}): Promise<FlashcardReviewSessionSummary[]> {
  const query = buildQuery({
    deck_id: params?.deck_id,
    scope_key: params?.scope_key,
    status: params?.status,
    limit: params?.limit ?? 20
  })
  return await bgRequest<FlashcardReviewSessionSummary[], AllowedPath, "GET">({
    path: `/api/v1/flashcards/review-sessions${query}` as any,
    method: "GET",
    abortSignal: params?.signal
  })
}

export async function endFlashcardReviewSession(
  reviewSessionId: number,
  options?: { signal?: AbortSignal }
): Promise<FlashcardReviewSessionSummary> {
  return await bgRequest<FlashcardReviewSessionSummary, AllowedPath, "POST">({
    path: "/api/v1/flashcards/review-sessions/end" as any,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: {
      review_session_id: reviewSessionId
    },
    abortSignal: options?.signal
  })
}

export async function generateFlashcards(
  input: FlashcardsGenerateRequest
): Promise<FlashcardsGenerateResponse> {
  return await bgRequest<FlashcardsGenerateResponse, AllowedPath, "POST">({
    path: "/api/v1/flashcards/generate",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: input,
    timeoutMs: FLASHCARD_GENERATION_TIMEOUT_MS
  })
}

const sanitizeStudyPackSourceItem = (
  item: StudyPackSourceSelection
): StudyPackSourceSelection => {
  const sanitized: StudyPackSourceSelection = {
    source_type: item.source_type,
    source_id: item.source_id
  }

  const label = typeof item.label === "string" ? item.label.trim() : ""
  const sourceTitle = typeof item.source_title === "string" ? item.source_title.trim() : ""
  const excerptText = typeof item.excerpt_text === "string" ? item.excerpt_text.trim() : ""

  if (label) sanitized.label = label
  else if (sourceTitle) sanitized.source_title = sourceTitle

  if (excerptText) sanitized.excerpt_text = excerptText
  if (item.locator && typeof item.locator === "object" && !Array.isArray(item.locator)) {
    sanitized.locator = item.locator
  }

  return sanitized
}

const sanitizeStudyPackRequest = (
  request: StudyPackCreateJobRequest
): Omit<StudyPackCreateJobRequest, "source_items"> & {
  deck_mode: "new"
  source_items: StudyPackSourceSelection[]
} => ({
  title: request.title,
  workspace_id: request.workspace_id,
  deck_mode: "new",
  source_items: request.source_items.map(sanitizeStudyPackSourceItem)
})

export async function createStudyPackJob(
  request: StudyPackCreateJobRequest,
  options?: { signal?: AbortSignal }
): Promise<StudyPackJobAcceptedResponse> {
  return await bgRequest<StudyPackJobAcceptedResponse, AllowedPath, "POST">({
    path: "/api/v1/flashcards/study-packs/jobs",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: sanitizeStudyPackRequest(request),
    abortSignal: options?.signal
  })
}

export async function getStudyPackJob(
  jobId: number,
  options?: { signal?: AbortSignal }
): Promise<StudyPackJobStatusResponse> {
  return await bgRequest<StudyPackJobStatusResponse, AllowedPath, "GET">({
    path: `/api/v1/flashcards/study-packs/jobs/${jobId}` as AllowedPath,
    method: "GET",
    abortSignal: options?.signal
  })
}

export async function getStudyPack(
  packId: number,
  options?: { signal?: AbortSignal }
): Promise<StudyPackSummaryResponse> {
  return await bgRequest<StudyPackSummaryResponse, AllowedPath, "GET">({
    path: `/api/v1/flashcards/study-packs/${packId}` as AllowedPath,
    method: "GET",
    abortSignal: options?.signal
  })
}

export async function regenerateStudyPackJob(
  packId: number,
  options?: { signal?: AbortSignal }
): Promise<StudyPackJobAcceptedResponse> {
  return await bgRequest<StudyPackJobAcceptedResponse, AllowedPath, "POST">({
    path: `/api/v1/flashcards/study-packs/${packId}/regenerate` as AllowedPath,
    method: "POST",
    abortSignal: options?.signal
  })
}

export const regenerateStudyPack = regenerateStudyPackJob

// Import
export async function getFlashcardsImportLimits(): Promise<any> {
  return await bgRequest<any, AllowedPath, "GET">({
    path: "/api/v1/config/flashcards-import-limits",
    method: "GET"
  })
}

export async function importFlashcards(payload: FlashcardsImportRequest, overrides?: {
  max_lines?: number | null
  max_line_length?: number | null
  max_field_length?: number | null
}): Promise<FlashcardsImportResponse> {
  const query = buildQuery({
    max_lines: overrides?.max_lines,
    max_line_length: overrides?.max_line_length,
    max_field_length: overrides?.max_field_length
  })
  const path = `/api/v1/flashcards/import${query}` as AllowedPath
  return await bgRequest<FlashcardsImportResponse, AllowedPath, "POST">({
    path,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload
  })
}

export async function previewStructuredQaImport(
  payload: StructuredQaImportPreviewRequest,
  overrides?: {
    max_lines?: number | null
    max_line_length?: number | null
    max_field_length?: number | null
  }
): Promise<StructuredQaImportPreviewResponse> {
  const query = buildQuery({
    max_lines: overrides?.max_lines,
    max_line_length: overrides?.max_line_length,
    max_field_length: overrides?.max_field_length
  })
  const path = `/api/v1/flashcards/import/structured/preview${query}` as AllowedPath
  return await bgRequest<StructuredQaImportPreviewResponse, AllowedPath, "POST">({
    path,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload
  })
}

export async function importFlashcardsJson(
  payload: FlashcardsImportJsonRequest,
  overrides?: {
    max_items?: number | null
    max_field_length?: number | null
  }
): Promise<FlashcardsImportResponse> {
  const query = buildQuery({
    max_items: overrides?.max_items,
    max_field_length: overrides?.max_field_length
  })
  const path = `/api/v1/flashcards/import/json${query}` as AllowedPath
  const filename = (payload.filename || "flashcards.json").trim() || "flashcards.json"
  const lowerName = filename.toLowerCase()
  const mimeType =
    lowerName.endsWith(".jsonl") || lowerName.endsWith(".ndjson")
      ? "application/x-ndjson"
      : "application/json"
  const bytes = new TextEncoder().encode(payload.content)

  return await bgUpload<FlashcardsImportResponse, AllowedPath, "POST">({
    path,
    method: "POST",
    fileFieldName: "file",
    file: {
      name: filename,
      type: mimeType,
      data: bytes
    }
  })
}

export async function importFlashcardsApkg(
  payload: FlashcardsImportApkgRequest,
  overrides?: {
    max_items?: number | null
    max_field_length?: number | null
  }
): Promise<FlashcardsImportResponse> {
  const query = buildQuery({
    max_items: overrides?.max_items,
    max_field_length: overrides?.max_field_length
  })
  const path = `/api/v1/flashcards/import/apkg${query}` as AllowedPath
  const filename = (payload.filename || "flashcards.apkg").trim() || "flashcards.apkg"

  return await bgUpload<FlashcardsImportResponse, AllowedPath, "POST">({
    path,
    method: "POST",
    fileFieldName: "file",
    file: {
      name: filename,
      type: "application/apkg",
      data: payload.bytes
    }
  })
}

export async function getFlashcardsAnalyticsSummary(params?: {
  deck_id?: number | null
  workspace_id?: string | null
  include_workspace_items?: boolean | null
  signal?: AbortSignal
}): Promise<FlashcardAnalyticsSummary> {
  const query = buildQuery({
    deck_id: params?.deck_id,
    workspace_id: params?.workspace_id,
    include_workspace_items: params?.include_workspace_items ?? false
  })
  const path = `/api/v1/flashcards/analytics/summary${query}` as AllowedPath
  return await bgRequest<FlashcardAnalyticsSummary, AllowedPath, "GET">({
    path,
    method: "GET",
    abortSignal: params?.signal
  })
}

// Export (returns text/csv or file-like payload)
export async function exportFlashcards(params: FlashcardsExportParams = {}): Promise<string> {
  const query = buildQuery({
    deck_id: params.deck_id,
    tag: params.tag,
    q: params.q,
    format: params.format,
    include_reverse: params.include_reverse,
    delimiter: params.delimiter,
    include_header: params.include_header,
    extended_header: params.extended_header
  })
  const path = `/api/v1/flashcards/export${query}` as AllowedPath
  // Force accept text so bgRequest returns text
  return await bgRequest<string, AllowedPath, "GET">({
    path,
    method: "GET",
    headers: { Accept: "text/plain, text/csv, application/octet-stream, application/json;q=0.5" }
  })
}

// Export binary (APKG). Uses direct fetch to preserve binary payload.
export async function exportFlashcardsFile(params: FlashcardsExportParams & { format: 'apkg' }): Promise<Blob> {
  const query = buildQuery({
    deck_id: params.deck_id,
    tag: params.tag,
    q: params.q,
    format: "apkg",
    include_reverse: params.include_reverse,
    // CSV specific options ignored for apkg on server side, but safe to pass
    delimiter: params.delimiter,
    include_header: params.include_header,
    extended_header: params.extended_header
  })
  const path = `/api/v1/flashcards/export${query}` as AllowedPath
  const response = await bgRequest<{
    ok: boolean
    status: number
    data?: ArrayBuffer
    error?: string
    headers?: Record<string, string>
  }>({
    path,
    method: "GET",
    headers: { Accept: "application/octet-stream" },
    responseType: "arrayBuffer",
    returnResponse: true
  })
  if (!response) {
    throw new Error("Export failed")
  }
  if (!response.ok) {
    throw new Error(response.error || `Export failed: ${response.status}`)
  }
  const headers = new Headers(response.headers || {})
  return new Blob([response.data ?? new Uint8Array()], {
    type: headers.get("content-type") || "application/octet-stream"
  })
}
