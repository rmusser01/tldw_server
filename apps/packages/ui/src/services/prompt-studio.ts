import { apiSend } from "@/services/api-send"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"
import type { ApiResponseEnvelope } from "@/services/response-envelope"

// Prompt Studio client – aligns with tldw_server prompt_studio endpoints.

export type PaginationMeta = {
  page: number
  per_page: number
  total: number
  total_pages?: number
}

export type StandardResponse<T> = ApiResponseEnvelope<T>

export type ListResponse<T> = StandardResponse<T[]> & {
  metadata?: PaginationMeta
}

export type Project = {
  id: number
  uuid?: string
  name: string
  description?: string | null
  status?: string
  created_at?: string
  updated_at?: string
  prompt_count?: number
  test_case_count?: number
  metadata?: Record<string, any> | null
}

export type ProjectCreatePayload = {
  name: string
  description?: string | null
  status?: string
  metadata?: Record<string, any> | null
}

export type ProjectUpdatePayload = Partial<ProjectCreatePayload>

export type PromptModule = {
  type: string
  enabled?: boolean
  config?: Record<string, any> | null
}

export type FewShotExample = {
  inputs: Record<string, any>
  outputs: Record<string, any>
  explanation?: string | null
}

export type PromptFormat = "legacy" | "structured"
export type StructuredPromptDefinition = Record<string, any>

export type Prompt = {
  id: number
  uuid?: string
  project_id: number
  name: string
  system_prompt?: string | null
  user_prompt?: string | null
  prompt_format?: PromptFormat
  prompt_schema_version?: number | null
  prompt_definition?: StructuredPromptDefinition | null
  few_shot_examples?: FewShotExample[] | null
  modules_config?: PromptModule[] | null
  signature_id?: number | null
  version_number: number
  change_description?: string | null
  parent_version_id?: number | null
  created_at?: string
  updated_at?: string
}

export type PromptCreatePayload = {
  project_id: number
  name: string
  system_prompt?: string | null
  user_prompt?: string | null
  prompt_format?: PromptFormat
  prompt_schema_version?: number | null
  prompt_definition?: StructuredPromptDefinition | null
  few_shot_examples?: FewShotExample[] | null
  modules_config?: PromptModule[] | null
  change_description?: string | null
  signature_id?: number | null
  parent_version_id?: number | null
}

export type PromptUpdatePayload = {
  name?: string
  system_prompt?: string | null
  user_prompt?: string | null
  prompt_format?: PromptFormat
  prompt_schema_version?: number | null
  prompt_definition?: StructuredPromptDefinition | null
  few_shot_examples?: FewShotExample[] | null
  modules_config?: PromptModule[] | null
  change_description: string
}

export type StructuredPromptPreviewRequest = {
  project_id: number
  signature_id?: number | null
  prompt_format: PromptFormat
  system_prompt?: string | null
  user_prompt?: string | null
  prompt_schema_version?: number | null
  prompt_definition?: StructuredPromptDefinition | null
  few_shot_examples?: FewShotExample[] | null
  modules_config?: PromptModule[] | null
  variables?: Record<string, any>
}

export type StructuredPromptPreviewResponse = {
  prompt_format: PromptFormat
  prompt_schema_version?: number | null
  assembled_messages: Array<{
    role: string
    content: string
  }>
  legacy_system_prompt: string
  legacy_user_prompt: string
}

export type PromptVersion = {
  id: number
  uuid?: string
  version_number: number
  name: string
  change_description?: string | null
  created_at?: string
  parent_version_id?: number | null
}

export type ExecutePromptPayload = {
  prompt_id: number
  inputs?: Record<string, any>
  provider?: string
  model?: string
}

export type ExecutePromptResult = {
  output: string
  tokens_used?: number
  execution_time?: number
}

export type TestCase = {
  id: number
  project_id: number
  name?: string | null
  description?: string | null
  inputs: Record<string, any>
  expected_outputs?: Record<string, any> | null
  tags?: string[] | null
  is_golden?: boolean
  signature_id?: number | null
  created_at?: string
  updated_at?: string
}

export type TestCaseCreatePayload = {
  project_id: number
  name?: string | null
  description?: string | null
  inputs: Record<string, any>
  expected_outputs?: Record<string, any> | null
  tags?: string[] | null
  is_golden?: boolean
  signature_id?: number | null
}

export type TestCaseUpdatePayload = Partial<Omit<TestCaseCreatePayload, "project_id">>

export type TestCaseBulkCreatePayload = {
  project_id: number
  signature_id?: number | null
  test_cases: Array<Omit<TestCaseCreatePayload, "project_id">>
}

export type EvaluationConfig = {
  model_name?: string
  temperature?: number
  max_tokens?: number
  top_p?: number
  frequency_penalty?: number
  presence_penalty?: number
  api_endpoint?: string
  api_key_name?: string
  timeout_seconds?: number
  retry_count?: number
  parallel_requests?: number
}

export type EvaluationCreatePayload = {
  project_id: number
  prompt_id: number
  test_case_ids: number[]
  name?: string | null
  description?: string | null
  config?: EvaluationConfig
  model_configs?: EvaluationConfig[]
  run_async?: boolean
  tags?: string[]
}

export type PromptStudioEvaluation = {
  id: number
  uuid?: string
  project_id: number
  prompt_id: number
  name?: string | null
  description?: string | null
  status: string
  metrics?: Record<string, any>
  aggregate_metrics?: Record<string, any>
  config?: Record<string, any>
  model_configs?: EvaluationConfig[]
  test_case_ids?: number[]
  test_run_ids?: number[]
  error_message?: string | null
  created_at?: string
  completed_at?: string | null
}

export type EvaluationListResponse = {
  evaluations: PromptStudioEvaluation[]
  total: number
  limit: number
  offset: number
}

export type PromptStudioStatus = {
  queue_depth: number
  processing: number
  leases: Record<string, number>
  by_status?: Record<string, number>
  by_type?: Record<string, number>
  avg_processing_time_seconds?: number
  success_rate?: number
}

export type LlmProviderModelInfo = {
  name?: string
  id?: string
  model_id?: string
  display_name?: string
  deprecated?: boolean
  [key: string]: any
}

export type LlmProviderConfig = {
  name: string
  display_name?: string
  models?: string[]
  models_info?: LlmProviderModelInfo[]
  default_model?: string | null
  [key: string]: any
}

export type LlmProvidersResponse = {
  providers: LlmProviderConfig[]
  default_provider?: string | null
  total_configured?: number
  [key: string]: any
}

const withIdempotency = (
  key?: string | null
): Record<string, string> | undefined => {
  if (!key) return undefined
  return { "Idempotency-Key": key }
}

const buildQuery = (params: Record<string, any>) => {
  const qs = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => {
    if (v === undefined || v === null) return
    qs.set(k, String(v))
  })
  const query = qs.toString()
  return query ? `?${query}` : ""
}

// Capability probe
export async function hasPromptStudio(): Promise<boolean> {
  try {
    const res = await getPromptStudioStatus()
    const status = (res as any)?.data
    return Boolean((res as any)?.ok && (status?.success ?? true))
  } catch {
    return false
  }
}

// Projects
export async function listProjects(params?: {
  page?: number
  per_page?: number
  include_deleted?: boolean
  search?: string
}) {
  const query = buildQuery({
    page: params?.page ?? 1,
    per_page: params?.per_page ?? 20,
    include_deleted: params?.include_deleted,
    search: params?.search
  })
  return await apiSend<ListResponse<Project>>({
    path: appendPathQuery("/api/v1/prompt-studio/projects", query),
    method: "GET"
  })
}

export async function createProject(
  payload: ProjectCreatePayload,
  idempotencyKey?: string | null
) {
  return await apiSend<StandardResponse<Project>>({
    path: "/api/v1/prompt-studio/projects/",
    method: "POST",
    body: payload,
    headers: withIdempotency(idempotencyKey)
  })
}

export async function getProject(projectId: number) {
  return await apiSend<StandardResponse<Project>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/projects/${encodeURIComponent(projectId)}`
    ),
    method: "GET"
  })
}

export async function updateProject(projectId: number, payload: ProjectUpdatePayload) {
  return await apiSend<StandardResponse<Project>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/projects/${encodeURIComponent(projectId)}`
    ),
    method: "PUT",
    body: payload
  })
}

// Prompts
export async function listPrompts(projectId: number, params?: { page?: number; per_page?: number; include_deleted?: boolean }) {
  const query = buildQuery({
    page: params?.page ?? 1,
    per_page: params?.per_page ?? 20,
    include_deleted: params?.include_deleted
  })
  return await apiSend<ListResponse<Prompt>>({
    path: appendPathQuery(
      toAllowedPath(
        `/api/v1/prompt-studio/prompts/list/${encodeURIComponent(projectId)}`
      ),
      query
    ),
    method: "GET"
  })
}

export async function createPrompt(
  payload: PromptCreatePayload,
  idempotencyKey?: string | null
) {
  return await apiSend<StandardResponse<Prompt>>({
    path: "/api/v1/prompt-studio/prompts/create",
    method: "POST",
    body: payload,
    headers: withIdempotency(idempotencyKey)
  })
}

export async function getPrompt(promptId: number) {
  return await apiSend<StandardResponse<Prompt>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/prompts/get/${encodeURIComponent(promptId)}`
    ),
    method: "GET"
  })
}

export async function updatePrompt(promptId: number, payload: PromptUpdatePayload) {
  return await apiSend<StandardResponse<Prompt>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/prompts/update/${encodeURIComponent(promptId)}`
    ),
    method: "PUT",
    body: payload
  })
}

export async function previewPromptDefinition(
  payload: StructuredPromptPreviewRequest
) {
  return await apiSend<StandardResponse<StructuredPromptPreviewResponse>>({
    path: "/api/v1/prompt-studio/prompts/preview",
    method: "POST",
    body: payload
  })
}

export async function getPromptHistory(promptId: number) {
  return await apiSend<StandardResponse<PromptVersion[]>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/prompts/history/${encodeURIComponent(promptId)}`
    ),
    method: "GET"
  })
}

export async function revertPrompt(promptId: number, version: number) {
  return await apiSend<StandardResponse<Prompt>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/prompts/revert/${encodeURIComponent(
        promptId
      )}/${encodeURIComponent(version)}`
    ),
    method: "POST"
  })
}

export async function executePrompt(payload: ExecutePromptPayload) {
  return await apiSend<ExecutePromptResult>({
    path: "/api/v1/prompt-studio/prompts/execute",
    method: "POST",
    body: payload
  })
}

// Test cases
export async function listTestCases(
  projectId: number,
  params?: {
    page?: number
    per_page?: number
    is_golden?: boolean
    tags?: string
    search?: string
    signature_id?: number
  }
) {
  const query = buildQuery({
    page: params?.page ?? 1,
    per_page: params?.per_page ?? 20,
    is_golden: params?.is_golden,
    tags: params?.tags,
    search: params?.search,
    signature_id: params?.signature_id
  })
  return await apiSend<ListResponse<TestCase>>({
    path: appendPathQuery(
      toAllowedPath(
        `/api/v1/prompt-studio/test-cases/list/${encodeURIComponent(projectId)}`
      ),
      query
    ),
    method: "GET"
  })
}

export async function createTestCase(payload: TestCaseCreatePayload) {
  return await apiSend<StandardResponse<TestCase>>({
    path: "/api/v1/prompt-studio/test-cases/create",
    method: "POST",
    body: payload
  })
}

export async function createBulkTestCases(payload: TestCaseBulkCreatePayload) {
  return await apiSend<StandardResponse<TestCase[]>>({
    path: "/api/v1/prompt-studio/test-cases/bulk",
    method: "POST",
    body: payload
  })
}

export async function getTestCase(testCaseId: number) {
  return await apiSend<StandardResponse<TestCase>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/test-cases/get/${encodeURIComponent(testCaseId)}`
    ),
    method: "GET"
  })
}

export async function updateTestCase(testCaseId: number, payload: TestCaseUpdatePayload) {
  return await apiSend<StandardResponse<TestCase>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/test-cases/update/${encodeURIComponent(testCaseId)}`
    ),
    method: "PUT",
    body: payload
  })
}

// Evaluations
export async function createEvaluation(payload: EvaluationCreatePayload) {
  return await apiSend<PromptStudioEvaluation>({
    path: "/api/v1/prompt-studio/evaluations",
    method: "POST",
    body: payload
  })
}

export async function listEvaluations(params: {
  project_id: number
  prompt_id?: number
  limit?: number
  offset?: number
}) {
  const query = buildQuery({
    project_id: params.project_id,
    prompt_id: params.prompt_id,
    limit: params.limit ?? 100,
    offset: params.offset ?? 0
  })
  return await apiSend<EvaluationListResponse>({
    path: appendPathQuery("/api/v1/prompt-studio/evaluations", query),
    method: "GET"
  })
}

export async function getEvaluation(evaluationId: number) {
  return await apiSend<PromptStudioEvaluation>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/evaluations/${encodeURIComponent(evaluationId)}`
    ),
    method: "GET"
  })
}

export async function deleteEvaluation(evaluationId: number) {
  return await apiSend<{ message: string }>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/evaluations/${encodeURIComponent(evaluationId)}`
    ),
    method: "DELETE"
  })
}

// Status
export async function getPromptStudioStatus(params?: { warn_seconds?: number }) {
  const query = buildQuery({ warn_seconds: params?.warn_seconds })
  return await apiSend<StandardResponse<PromptStudioStatus>>({
    path: appendPathQuery("/api/v1/prompt-studio/status", query),
    method: "GET"
  })
}

export async function getLlmProviders(params?: { include_deprecated?: boolean }) {
  const query = buildQuery({
    include_deprecated: params?.include_deprecated
  })
  return await apiSend<LlmProvidersResponse>({
    path: appendPathQuery("/api/v1/llm/providers", query),
    method: "GET"
  })
}

// --- Extended Project Operations ---

export async function archiveProject(projectId: number) {
  return await apiSend<StandardResponse<Project>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/projects/archive/${encodeURIComponent(projectId)}`
    ),
    method: "POST"
  })
}

export async function unarchiveProject(projectId: number) {
  return await apiSend<StandardResponse<Project>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/projects/unarchive/${encodeURIComponent(projectId)}`
    ),
    method: "POST"
  })
}

export async function deleteProject(projectId: number, permanent?: boolean) {
  const query = buildQuery({ permanent })
  return await apiSend<{ message: string }>({
    path: appendPathQuery(
      toAllowedPath(
        `/api/v1/prompt-studio/projects/delete/${encodeURIComponent(projectId)}`
      ),
      query
    ),
    method: "DELETE"
  })
}

export type ProjectStats = {
  prompt_count: number
  test_case_count: number
  evaluation_count: number
  optimization_count?: number
}

export async function getProjectStats(projectId: number) {
  // This uses getProject which already includes counts
  return await getProject(projectId)
}

// --- Extended Test Case Operations ---

export async function deleteTestCase(testCaseId: number, permanent?: boolean) {
  const query = buildQuery({ permanent })
  return await apiSend<{ message: string }>({
    path: appendPathQuery(
      toAllowedPath(
        `/api/v1/prompt-studio/test-cases/delete/${encodeURIComponent(testCaseId)}`
      ),
      query
    ),
    method: "DELETE"
  })
}

export type TestCaseExportFormat = "json" | "csv"

export type ExportTestCasesPayload = {
  format: TestCaseExportFormat
  include_golden_only?: boolean
  tag_filter?: string[]
}

export async function exportTestCases(
  projectId: number,
  payload: ExportTestCasesPayload
) {
  return await apiSend<TestCase[] | string>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/test-cases/export/${encodeURIComponent(projectId)}`
    ),
    method: "POST",
    body: payload
  })
}

export type TestCaseImportPayload = {
  project_id: number
  format: TestCaseExportFormat
  data: string | TestCaseCreatePayload[]
  signature_id?: number | null
}

export async function importTestCases(payload: TestCaseImportPayload) {
  return await apiSend<StandardResponse<TestCase[]>>({
    path: "/api/v1/prompt-studio/test-cases/import",
    method: "POST",
    body: payload
  })
}

export type GenerateTestCasesPayload = {
  project_id: number
  prompt_id?: number
  count?: number
  seed_examples?: Array<{ inputs: Record<string, any>; expected_outputs?: Record<string, any> }>
  provider?: string
  model?: string
  signature_id?: number | null
}

export async function generateTestCases(payload: GenerateTestCasesPayload) {
  return await apiSend<StandardResponse<TestCase[]>>({
    path: "/api/v1/prompt-studio/test-cases/generate",
    method: "POST",
    body: payload
  })
}

export type RunTestCasesPayload = {
  prompt_id: number
  test_case_ids: number[]
  config?: EvaluationConfig
}

export type TestRunResult = {
  test_case_id: number
  output: string
  passed?: boolean
  execution_time?: number
  error?: string | null
}

export async function runTestCases(payload: RunTestCasesPayload) {
  return await apiSend<StandardResponse<TestRunResult[]>>({
    path: "/api/v1/prompt-studio/test-cases/run",
    method: "POST",
    body: payload
  })
}

// --- Optimization Operations ---

export type OptimizationStrategy =
  | "iterative"
  | "mipro"
  | "bootstrap"
  | "hyperparameter"
  | "genetic"
  | "beam_search"
  | "simulated_annealing"
  | "random_search"
  | "hill_climbing"
  | "mcts"

export type OptimizationStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"

export type OptimizationConfig = {
  strategy: OptimizationStrategy
  max_iterations?: number
  population_size?: number
  beam_width?: number
  temperature?: number
  learning_rate?: number
  early_stopping_rounds?: number
  metric?: string
  minimize?: boolean
  custom_params?: Record<string, any>
}

export type Optimization = {
  id: number
  uuid?: string
  project_id: number
  prompt_id: number
  name?: string | null
  description?: string | null
  status: OptimizationStatus
  config: OptimizationConfig
  model_config?: EvaluationConfig
  test_case_ids?: number[]
  best_prompt_id?: number | null
  best_score?: number | null
  current_iteration?: number
  total_iterations?: number
  error_message?: string | null
  cancel_reason?: string | null
  created_at?: string
  started_at?: string | null
  completed_at?: string | null
}

export type OptimizationIteration = {
  iteration: number
  prompt_id: number
  score: number
  metrics?: Record<string, any>
  changes?: string
  timestamp?: string
}

export type OptimizationCreatePayload = {
  project_id: number
  prompt_id: number
  name?: string | null
  description?: string | null
  config: OptimizationConfig
  model_config?: EvaluationConfig
  test_case_ids?: number[]
}

export type OptimizationListResponse = {
  optimizations: Optimization[]
  total: number
  limit: number
  offset: number
}

export async function listOptimizations(params: {
  project_id: number
  prompt_id?: number
  status?: OptimizationStatus
  limit?: number
  offset?: number
}) {
  const query = buildQuery({
    project_id: params.project_id,
    prompt_id: params.prompt_id,
    status: params.status,
    limit: params.limit ?? 50,
    offset: params.offset ?? 0
  })
  return await apiSend<OptimizationListResponse>({
    path: appendPathQuery("/api/v1/prompt-studio/optimizations", query),
    method: "GET"
  })
}

export async function createOptimization(payload: OptimizationCreatePayload) {
  return await apiSend<StandardResponse<Optimization>>({
    path: "/api/v1/prompt-studio/optimizations",
    method: "POST",
    body: payload
  })
}

export async function getOptimization(optimizationId: number) {
  return await apiSend<StandardResponse<Optimization>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/optimizations/${encodeURIComponent(optimizationId)}`
    ),
    method: "GET"
  })
}

export async function cancelOptimization(optimizationId: number, reason?: string) {
  return await apiSend<StandardResponse<Optimization>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/optimizations/${encodeURIComponent(optimizationId)}/cancel`
    ),
    method: "POST",
    body: reason ? { reason } : undefined
  })
}

export async function deleteOptimization(optimizationId: number) {
  return await apiSend<{ message: string }>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/optimizations/${encodeURIComponent(optimizationId)}`
    ),
    method: "DELETE"
  })
}

export async function getOptimizationIterations(optimizationId: number) {
  return await apiSend<StandardResponse<OptimizationIteration[]>>({
    path: toAllowedPath(
      `/api/v1/prompt-studio/optimizations/${encodeURIComponent(optimizationId)}/iterations`
    ),
    method: "GET"
  })
}

export type StrategyInfo = {
  name: OptimizationStrategy
  display_name: string
  description: string
  supported_params: string[]
  default_params: Record<string, any>
  requires_test_cases: boolean
  supports_early_stopping: boolean
}

export async function getOptimizationStrategies() {
  return await apiSend<StandardResponse<StrategyInfo[]>>({
    path: "/api/v1/prompt-studio/optimizations/strategies",
    method: "GET"
  })
}

export type CompareStrategiesPayload = {
  project_id: number
  prompt_id: number
  strategies: OptimizationStrategy[]
  test_case_ids: number[]
  iterations_per_strategy?: number
  model_config?: EvaluationConfig
}

export type StrategyComparisonResult = {
  strategy: OptimizationStrategy
  best_score: number
  avg_score: number
  iterations_completed: number
  best_prompt_id?: number
  execution_time_seconds: number
}

export async function compareStrategies(payload: CompareStrategiesPayload) {
  return await apiSend<StandardResponse<StrategyComparisonResult[]>>({
    path: "/api/v1/prompt-studio/optimizations/compare",
    method: "POST",
    body: payload
  })
}
