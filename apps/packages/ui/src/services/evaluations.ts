import { apiSend } from "@/services/api-send"

// Lightweight client for the Evaluations module.
// Shapes are intentionally loose for now; we can tighten them
// using OpenAPI-derived types later if needed.

export type EvaluationSummary = {
  id: string
  name?: string
  description?: string
  eval_type?: string
  created?: number
  created_at?: string | number | null
  dataset_id?: string | null
  metadata?: Record<string, any> | null
}

export type EvaluationDetail = EvaluationSummary & {
  eval_spec?: Record<string, any>
  updated_at?: string | number | null
  deleted?: boolean
}

export type EvaluationRunSummary = {
  id: string
  status?: string
  created_at?: string
  completed_at?: string | null
}

export type EvaluationListResponse = {
  object?: "list"
  data: EvaluationSummary[]
  has_more?: boolean
  first_id?: string | null
  last_id?: string | null
}

export type DatasetSample = {
  input?: any
  expected?: any
  sample_id?: string
  query?: string
  expected_behavior?: any
  reference_answer?: string
  sample_payload?: Record<string, any>
  metadata?: Record<string, any>
}

export type DatasetResponse = {
  id: string
  object?: string
  name: string
  description?: string | null
  sample_count: number
  created: number
  created_at?: number | null
  created_by: string
  metadata?: Record<string, any> | null
  samples?: DatasetSample[]
}

export type DatasetListResponse = {
  object?: "list"
  data: DatasetResponse[]
  has_more?: boolean
  first_id?: string | null
  last_id?: string | null
  total?: number | null
}

export type EvaluationRateLimitStatus = {
  tier: string
  limits: {
    evaluations_per_minute: number
    evaluations_per_day: number
    tokens_per_day: number
    cost_per_day: number
    cost_per_month: number
  }
  usage: {
    evaluations_today: number
    tokens_today: number
    cost_today: number
    cost_month: number
  }
  remaining: {
    daily_evaluations: number
    daily_tokens: number
    daily_cost: number
    monthly_cost: number
  }
  reset_at: string
}

export type EvaluationRunDetail = {
  id: string
  eval_id: string
  status: string
  target_model: string
  created: number
  created_at?: number | null
  started_at?: number | null
  completed_at?: number | null
  progress?: Record<string, any> | null
  error_message?: string | null
  results?: Record<string, any> | null
  usage?: Record<string, any> | null
}

export type CreateEvaluationPayload = {
  name: string
  description?: string
  eval_type: string
  eval_spec: any
  dataset_id?: string
  dataset?: DatasetSample[]
  metadata?: Record<string, any>
}

export type CreateRunPayload = {
  target_model: string
  config?: Record<string, any>
  dataset_override?: { samples: DatasetSample[] }
  webhook_url?: string
}

export type EvaluationHistoryFilters = {
  user_id?: string
  evaluation_type?: string
  start_date?: string
  end_date?: string
  limit?: number
  offset?: number
}

export type EvaluationHistoryItem = {
  id: string
  name?: string
  user_id?: string
  created_by?: string
  type?: string
  eval_type?: string
  evaluation_type?: string
  created_at?: string
  eval_id?: string
  evaluation_id?: string
  run_id?: string
  detail?: Record<string, any>
  metadata?: Record<string, any>
}

export type EvaluationWebhook = {
  id?: string
  webhook_id?: string | number
  url: string
  events: string[]
  created_at?: string
  secret?: string
  status?: string
  is_active?: boolean
  failure_count?: number
}

export type RecipeEvaluationMode = "fixed_context" | "live_end_to_end"
export type RecipeSupervisionMode =
  | "rubric"
  | "reference_answer"
  | "pairwise"
  | "mixed"
export type RecipeCandidateDimension =
  | "generation_model"
  | "prompt_variant"
  | "formatting_citation_mode"

export type RecipeManifestCapabilities = {
  evaluation_modes?: RecipeEvaluationMode[]
  supervision_modes?: RecipeSupervisionMode[]
  candidate_dimensions?: RecipeCandidateDimension[]
  [key: string]: any
}

export type RecipeManifestDefaultRunConfig = {
  evaluation_mode?: RecipeEvaluationMode
  supervision_mode?: RecipeSupervisionMode
  candidate_dimensions?: RecipeCandidateDimension[]
  [key: string]: any
}

export type RecipeManifest = {
  recipe_id: string
  recipe_version: string
  name: string
  description: string
  supported_modes: Array<"labeled" | "unlabeled">
  tags: string[]
  launchable: boolean
  capabilities?: RecipeManifestCapabilities
  default_run_config?: RecipeManifestDefaultRunConfig
}

export type RecipeLaunchReadiness = {
  recipe_id: string
  ready: boolean
  can_enqueue_runs: boolean
  can_reuse_completed_runs: boolean
  runtime_checks: Record<string, boolean>
  message?: string | null
}

export type RecipeDatasetValidation = {
  valid: boolean
  errors: string[]
  dataset_mode?: "labeled" | "unlabeled" | "mixed" | null
  sample_count?: number
  review_sample?: Record<string, any>
  dataset_snapshot_ref?: string | null
  dataset_content_hash?: string | null
  [key: string]: any
}

export type RecipeConfidenceSummary = {
  kind?: "aggregate" | "bootstrap" | "judge" | "heuristic"
  confidence: number
  sample_count: number
  spread?: number | null
  margin?: number | null
  judge_agreement?: number | null
  notes?: string | null
}

export type RecipeRecommendationSlot = {
  candidate_run_id?: string | null
  reason_code?: string | null
  explanation?: string | null
  confidence?: number | null
  metadata?: Record<string, any>
}

export type RecipeRunRecord = {
  run_id: string
  recipe_id: string
  recipe_version: string
  status: string
  review_state?: string
  dataset_snapshot_ref?: string | null
  dataset_content_hash?: string | null
  confidence_summary?: RecipeConfidenceSummary | null
  recommendation_slots?: Record<string, RecipeRecommendationSlot>
  child_run_ids?: string[]
  created_at: string
  updated_at?: string | null
  metadata?: Record<string, any>
}

export type RecipeRunReport = {
  run: RecipeRunRecord
  confidence_summary?: RecipeConfidenceSummary | null
  recommendation_slots: Record<string, RecipeRecommendationSlot>
}

export type EmbeddingRecipeCandidateStatus =
  | "ready"
  | "missing_key"
  | "disallowed_provider"
  | "disallowed_model"
  | "quota_risk"
  | "unknown"

export type EmbeddingRecipeCandidateHint = {
  provider: string
  model: string
  is_local: boolean
  default: boolean
  status: EmbeddingRecipeCandidateStatus
  status_reason?: string | null
  dimensions?: number | null
  revision?: string | null
  cost_hint?: string | null
}

export type EmbeddingRecipeCandidatesResponse = {
  recipe_id: "embeddings_model_selection"
  current?: EmbeddingRecipeCandidateHint | null
  candidates: EmbeddingRecipeCandidateHint[]
  warnings: string[]
}

export type RecipeRecommendationApplyPreviewPayload = {
  slot_name: string
  candidate_run_id?: string | null
}

export type RecipeRecommendationApplyPreviewResponse = {
  run_id: string
  recipe_id: string
  slot_name: string
  candidate_run_id?: string | null
  apply_eligible: boolean
  apply_available: boolean
  blocked_reason?: string | null
  warnings: string[]
  current: Record<string, string | null>
  proposed: Record<string, string | null>
  affected_config: Record<string, string>
  copy_config: Record<string, Record<string, string>>
  reindex_required: boolean
}

export type SyntheticEvalDraftSample = {
  sample_id: string
  recipe_kind: string
  provenance: string
  review_state: string
  sample_payload?: Record<string, any>
  sample_metadata?: Record<string, any>
  source_kind?: string | null
  created_by?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export type SyntheticEvalQueueResponse = {
  data: SyntheticEvalDraftSample[]
  total: number
}

export type SyntheticEvalGenerationRequest = {
  recipe_kind: string
  corpus_scope?: Record<string, any> | string[]
  generation_metadata?: Record<string, any>
  context_snapshot_ref?: string
  retrieval_baseline_ref?: string
  reference_answer?: string
  real_examples?: Record<string, any>[]
  seed_examples?: Record<string, any>[]
  target_sample_count: number
}

export type SyntheticEvalGenerationResponse = {
  generation_batch_id?: string | null
  samples: SyntheticEvalDraftSample[]
  source_breakdown?: Record<string, number>
  coverage?: Record<string, string[]>
  missing_coverage?: Record<string, string[]>
  corpus_scope?: Record<string, any>
}

export type SyntheticEvalReviewActionRecord = {
  action_id: string
  sample_id: string
  action: string
  reviewer_id?: string | null
  notes?: string | null
  action_payload?: Record<string, any>
  resulting_review_state?: string | null
  created_at?: string | null
}

export type SyntheticEvalPromotionResponse = {
  dataset_id: string
  dataset_snapshot_ref: string
  promotion_ids: string[]
  sample_count: number
}

export type BenchmarkInfo = {
  name: string
  description?: string
  evaluation_type?: string
  dataset_source?: string
  metadata?: Record<string, any>
}

const withIdempotency = (
  key?: string | null
): Record<string, string> | undefined => {
  if (!key) return undefined
  return { "Idempotency-Key": key }
}

export async function listEvaluations(params?: {
  limit?: number
  after?: string
  eval_type?: string
}) {
  const query = new URLSearchParams()
  if (params?.limit != null) query.set("limit", String(params.limit))
  if (params?.after) query.set("after", params.after)
  if (params?.eval_type) query.set("eval_type", params.eval_type)

  const path =
    "/api/v1/evaluations" + (query.toString() ? `?${query.toString()}` : "")

  return await apiSend<EvaluationListResponse>({
    path: path as any,
    method: "GET"
  })
}

export async function getEvaluation(evalId: string) {
  return await apiSend<EvaluationDetail>({
    path: `/api/v1/evaluations/${encodeURIComponent(evalId)}` as any,
    method: "GET"
  })
}

export async function updateEvaluation(
  evalId: string,
  payload: Partial<CreateEvaluationPayload>
) {
  return await apiSend<EvaluationDetail>({
    path: `/api/v1/evaluations/${encodeURIComponent(evalId)}` as any,
    method: "PATCH",
    body: payload
  })
}

export async function deleteEvaluation(evalId: string) {
  return await apiSend<void>({
    path: `/api/v1/evaluations/${encodeURIComponent(evalId)}` as any,
    method: "DELETE"
  })
}

export async function listRuns(evalId: string, params?: { limit?: number }) {
  const query = new URLSearchParams()
  if (params?.limit != null) query.set("limit", String(params.limit))
  const path =
    `/api/v1/evaluations/${encodeURIComponent(evalId)}/runs` +
    (query.toString() ? `?${query.toString()}` : "")

  return await apiSend<{
    object?: "list"
    data: EvaluationRunSummary[]
    has_more?: boolean
    first_id?: string | null
    last_id?: string | null
  }>({
    path: path as any,
    method: "GET"
  })
}

export async function cancelRun(runId: string) {
  return await apiSend({
    path: `/api/v1/evaluations/runs/${encodeURIComponent(runId)}/cancel` as any,
    method: "POST"
  })
}

export async function getRateLimits() {
  return await apiSend<EvaluationRateLimitStatus>({
    path: "/api/v1/evaluations/rate-limits" as any,
    method: "GET"
  })
}

export async function listDatasets(params?: {
  limit?: number
  offset?: number
}) {
  const query = new URLSearchParams()
  if (params?.limit != null) query.set("limit", String(params.limit))
  if (params?.offset != null) query.set("offset", String(params.offset))

  const path =
    "/api/v1/evaluations/datasets" +
    (query.toString() ? `?${query.toString()}` : "")

  return await apiSend<DatasetListResponse>({
    path: path as any,
    method: "GET"
  })
}

export async function getDataset(
  datasetId: string,
  params?: {
    include_samples?: boolean
    limit?: number
    offset?: number
  }
) {
  const search = new URLSearchParams()
  if (params?.include_samples) search.set("include_samples", "true")
  if (params?.limit != null) search.set("limit", String(params.limit))
  if (params?.offset != null) search.set("offset", String(params.offset))
  const path =
    `/api/v1/evaluations/datasets/${encodeURIComponent(datasetId)}` +
    (search.toString() ? `?${search.toString()}` : "")

  return await apiSend<DatasetResponse>({
    path: path as any,
    method: "GET"
  })
}

export async function createDataset(payload: {
  name: string
  description?: string
  samples: DatasetSample[]
  metadata?: Record<string, any>
}) {
  return await apiSend<DatasetResponse>({
    path: "/api/v1/evaluations/datasets" as any,
    method: "POST",
    body: payload
  })
}

export async function deleteDataset(datasetId: string) {
  return await apiSend<void>({
    path: `/api/v1/evaluations/datasets/${encodeURIComponent(
      datasetId
    )}` as any,
    method: "DELETE"
  })
}

export async function createEvaluation(
  payload: CreateEvaluationPayload,
  options?: { idempotencyKey?: string }
) {
  return await apiSend({
    path: "/api/v1/evaluations" as any,
    method: "POST",
    headers: withIdempotency(options?.idempotencyKey),
    body: payload
  })
}

export async function createSpecializedEvaluation(
  endpoint:
    | "geval"
    | "rag"
    | "response-quality"
    | "propositions"
    | "ocr"
    | "ocr-pdf"
    | "batch"
    | "embeddings"
    | "embeddings-ab"
    | "embeddings-batch"
    | string,
  payload: Record<string, any>,
  options?: { idempotencyKey?: string }
) {
  return await apiSend({
    path: `/api/v1/evaluations/${endpoint}` as any,
    method: "POST",
    headers: withIdempotency(options?.idempotencyKey),
    body: payload
  })
}

export async function listBenchmarks() {
  return await apiSend<{
    object?: "list"
    data: BenchmarkInfo[]
    total?: number
  }>({
    path: "/api/v1/evaluations/benchmarks" as any,
    method: "GET"
  })
}

export async function listRecipeManifests() {
  return await apiSend<RecipeManifest[]>({
    path: "/api/v1/evaluations/recipes" as any,
    method: "GET"
  })
}

export async function getRecipeManifest(recipeId: string) {
  return await apiSend<RecipeManifest>({
    path: `/api/v1/evaluations/recipes/${encodeURIComponent(recipeId)}` as any,
    method: "GET"
  })
}

export async function getRecipeLaunchReadiness(recipeId: string) {
  return await apiSend<RecipeLaunchReadiness>({
    path: `/api/v1/evaluations/recipes/${encodeURIComponent(recipeId)}/launch-readiness` as any,
    method: "GET"
  })
}

export async function validateRecipeDataset(
  recipeId: string,
  payload: {
    dataset_id?: string
    dataset?: DatasetSample[]
    run_config?: Record<string, any>
  }
) {
  return await apiSend<RecipeDatasetValidation>({
    path: `/api/v1/evaluations/recipes/${encodeURIComponent(recipeId)}/validate-dataset` as any,
    method: "POST",
    body: payload
  })
}

export async function createRecipeRun(
  recipeId: string,
  payload: {
    dataset_id?: string
    dataset?: DatasetSample[]
    run_config: Record<string, any>
    force_rerun?: boolean
  }
) {
  return await apiSend<RecipeRunRecord>({
    path: `/api/v1/evaluations/recipes/${encodeURIComponent(recipeId)}/runs` as any,
    method: "POST",
    body: payload
  })
}

export async function getRecipeRun(runId: string) {
  return await apiSend<RecipeRunRecord>({
    path: `/api/v1/evaluations/recipe-runs/${encodeURIComponent(runId)}` as any,
    method: "GET"
  })
}

export async function getRecipeRunReport(runId: string) {
  return await apiSend<RecipeRunReport>({
    path: `/api/v1/evaluations/recipe-runs/${encodeURIComponent(runId)}/report` as any,
    method: "GET"
  })
}

export async function getEmbeddingRecipeCandidates() {
  return await apiSend<EmbeddingRecipeCandidatesResponse>({
    path: "/api/v1/evaluations/recipes/embeddings_model_selection/candidates" as any,
    method: "GET"
  })
}

export async function previewRecipeRecommendationApply(
  runId: string,
  payload: RecipeRecommendationApplyPreviewPayload
) {
  return await apiSend<RecipeRecommendationApplyPreviewResponse>({
    path: `/api/v1/evaluations/recipe-runs/${encodeURIComponent(runId)}/apply-preview` as any,
    method: "POST",
    body: payload
  })
}

export async function listSyntheticEvalQueue(params?: {
  recipe_kind?: string
  review_state?: string
  source_kind?: string
  generation_batch_id?: string
  limit?: number
  offset?: number
}) {
  const query = new URLSearchParams()
  if (params?.recipe_kind) query.set("recipe_kind", params.recipe_kind)
  if (params?.review_state) query.set("review_state", params.review_state)
  if (params?.source_kind) query.set("source_kind", params.source_kind)
  if (params?.generation_batch_id) {
    query.set("generation_batch_id", params.generation_batch_id)
  }
  if (params?.limit != null) query.set("limit", String(params.limit))
  if (params?.offset != null) query.set("offset", String(params.offset))

  const path =
    "/api/v1/evaluations/synthetic/queue" +
    (query.toString() ? `?${query.toString()}` : "")

  return await apiSend<SyntheticEvalQueueResponse>({
    path: path as any,
    method: "GET"
  })
}

export async function generateSyntheticEvalDrafts(
  payload: SyntheticEvalGenerationRequest
) {
  return await apiSend<SyntheticEvalGenerationResponse>({
    path: "/api/v1/evaluations/synthetic/drafts/generate" as any,
    method: "POST",
    body: payload
  })
}

export async function reviewSyntheticEvalSample(
  sampleId: string,
  payload: {
    action: string
    reviewer_id?: string
    notes?: string
    action_payload?: Record<string, any>
    resulting_review_state?: string
  }
) {
  return await apiSend<SyntheticEvalReviewActionRecord>({
    path: `/api/v1/evaluations/synthetic/queue/${encodeURIComponent(sampleId)}/review` as any,
    method: "POST",
    body: payload
  })
}

export async function promoteSyntheticEvalSamples(payload: {
  sample_ids: string[]
  dataset_name: string
  dataset_description?: string
  dataset_metadata?: Record<string, any>
  promoted_by?: string
  promotion_reason?: string
}) {
  return await apiSend<SyntheticEvalPromotionResponse>({
    path: "/api/v1/evaluations/synthetic/promotions" as any,
    method: "POST",
    body: payload
  })
}

export async function runBenchmark(
  benchmarkName: string,
  payload: Record<string, any>
) {
  return await apiSend({
    path: `/api/v1/evaluations/benchmarks/${encodeURIComponent(benchmarkName)}/run` as any,
    method: "POST",
    body: payload
  })
}

export async function createRun(
  evalId: string,
  payload: CreateRunPayload,
  options?: { idempotencyKey?: string }
) {
  return await apiSend({
    path: `/api/v1/evaluations/${encodeURIComponent(evalId)}/runs` as any,
    method: "POST",
    headers: withIdempotency(options?.idempotencyKey),
    body: payload
  })
}

export async function getRun(runId: string) {
  return await apiSend<EvaluationRunDetail>({
    path: `/api/v1/evaluations/runs/${encodeURIComponent(runId)}` as any,
    method: "GET"
  })
}

export async function getHistory(filters?: EvaluationHistoryFilters) {
  return await apiSend<{
    total_count: number
    items: EvaluationHistoryItem[]
    aggregations?: Record<string, any>
  }>({
    path: "/api/v1/evaluations/history" as any,
    method: "POST",
    body: filters || {}
  })
}

export async function registerWebhook(payload: {
  url: string
  events: string[]
}) {
  return await apiSend<EvaluationWebhook>({
    path: "/api/v1/evaluations/webhooks" as any,
    method: "POST",
    body: payload
  })
}

export async function listWebhooks() {
  return await apiSend<EvaluationWebhook[]>({
    path: "/api/v1/evaluations/webhooks" as any,
    method: "GET"
  })
}

export async function deleteWebhook(url: string) {
  const query = new URLSearchParams({ url })
  return await apiSend<void>({
    path: `/api/v1/evaluations/webhooks?${query.toString()}` as any,
    method: "DELETE"
  })
}
