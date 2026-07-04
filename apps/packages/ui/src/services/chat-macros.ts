import { apiSend, type ApiSendResponse } from "@/services/api-send"

const CHAT_MACROS_BASE = "/api/v1/chat/macros"

const encodePathPart = (value: string): string => encodeURIComponent(value)

export type ChatMacroSource = "builtin" | "user"

export interface ChatMacroSummary {
  name: string
  command: string
  description?: string | null
  enabled: boolean
  source: ChatMacroSource
  immutable: boolean
  digest: string
  builtin_version?: number | null
  schema_version: number
}

export interface ChatMacroListResponse {
  macros: ChatMacroSummary[]
  count: number
}

export interface ChatMacroDetail {
  summary: ChatMacroSummary
  definition: Record<string, unknown>
  raw: string
  supporting_files: Record<string, string>
}

export interface ChatMacroCreateRequest {
  name: string
  raw: string
  supporting_files?: Record<string, string> | null
}

export interface ChatMacroUpdateRequest {
  raw?: string
  supporting_files?: Record<string, string> | null
  enabled?: boolean
}

export interface ChatMacroCloneRequest {
  name: string
  command?: string | null
}

export interface ChatMacroSettingsResponse {
  settings: Record<string, unknown>
}

export interface ChatMacroValidateResponse {
  valid: boolean
  macro?: Record<string, unknown> | null
  error?: string | null
}

export interface ChatMacroRunRequest {
  macro_name: string
  args?: Record<string, unknown>
  mode?: "background" | "chat_native" | "foreground"
  surface?: string | null
  conversation_id?: string | null
  workspace_id?: string | null
  acp_session_id?: string | null
  output_profile?: string | null
  context_snapshot?: Record<string, unknown> | null
  model_selection?: Record<string, unknown> | null
}

export interface ChatMacroRunResponse {
  run_id: string
  status: string
  detail_url: string
  job_id?: string | null
}

export interface ChatMacroRunRecord {
  run_id: string
  macro_name: string
  macro_command: string
  macro_source?: string | null
  macro_version?: number | null
  macro_digest?: string | null
  normalized_args?: Record<string, unknown>
  status: string
  surface?: string | null
  conversation_id?: string | null
  workspace_id?: string | null
  acp_session_id?: string | null
  job_id?: string | null
  output_profile?: string | null
  status_message_id?: string | null
  final_message_id?: string | null
  final_output?: string | null
  final_output_format?: string | null
  final_post_status?: string | null
  cancel_requested_at?: string | null
  error_code?: string | null
  error?: string | null
  created_at?: string | null
  started_at?: string | null
  completed_at?: string | null
  updated_at?: string | null
}

export interface ChatMacroBranchSummary {
  branch_id: string
  step_id?: string
  label?: string | null
  output_name?: string | null
  status: string
  attempt_count?: number
  output?: string | null
  retained?: boolean
  error_code?: string | null
  error?: string | null
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
}

export interface ChatMacroRunDetailResponse {
  run: ChatMacroRunRecord
  branches: ChatMacroBranchSummary[]
}

export interface ChatMacroCancelResponse {
  run_id: string
  status: string
  cancel_requested_at?: string | null
}

export const listChatMacros = (): Promise<ApiSendResponse<ChatMacroListResponse>> =>
  apiSend<ChatMacroListResponse>({
    path: CHAT_MACROS_BASE,
    method: "GET"
  })

export const getChatMacro = (
  name: string
): Promise<ApiSendResponse<ChatMacroDetail>> =>
  apiSend<ChatMacroDetail>({
    path: `${CHAT_MACROS_BASE}/${encodePathPart(name)}`,
    method: "GET"
  })

export const createChatMacro = (
  request: ChatMacroCreateRequest
): Promise<ApiSendResponse<ChatMacroDetail>> =>
  apiSend<ChatMacroDetail>({
    path: CHAT_MACROS_BASE,
    method: "POST",
    body: request
  })

export const updateChatMacro = (
  name: string,
  request: ChatMacroUpdateRequest
): Promise<ApiSendResponse<ChatMacroDetail>> =>
  apiSend<ChatMacroDetail>({
    path: `${CHAT_MACROS_BASE}/${encodePathPart(name)}`,
    method: "PUT",
    body: request
  })

export const setChatMacroEnabled = (
  name: string,
  enabled: boolean
): Promise<ApiSendResponse<ChatMacroDetail>> =>
  updateChatMacro(name, { enabled })

export const deleteChatMacro = (name: string): Promise<ApiSendResponse<void>> =>
  apiSend<void>({
    path: `${CHAT_MACROS_BASE}/${encodePathPart(name)}`,
    method: "DELETE"
  })

export const validateChatMacro = (
  raw: string
): Promise<ApiSendResponse<ChatMacroValidateResponse>> =>
  apiSend<ChatMacroValidateResponse>({
    path: `${CHAT_MACROS_BASE}/validate`,
    method: "POST",
    body: { raw }
  })

export const getChatMacroSettings = (): Promise<
  ApiSendResponse<ChatMacroSettingsResponse>
> =>
  apiSend<ChatMacroSettingsResponse>({
    path: `${CHAT_MACROS_BASE}/settings`,
    method: "GET"
  })

export const updateChatMacroSettings = (
  settings: Record<string, unknown>
): Promise<ApiSendResponse<ChatMacroSettingsResponse>> =>
  apiSend<ChatMacroSettingsResponse>({
    path: `${CHAT_MACROS_BASE}/settings`,
    method: "PUT",
    body: { settings }
  })

export const cloneChatMacro = (
  name: string,
  request: ChatMacroCloneRequest
): Promise<ApiSendResponse<ChatMacroDetail>> =>
  apiSend<ChatMacroDetail>({
    path: `${CHAT_MACROS_BASE}/${encodePathPart(name)}/clone`,
    method: "POST",
    body: request
  })

export const runChatMacro = (
  request: ChatMacroRunRequest
): Promise<ApiSendResponse<ChatMacroRunResponse>> =>
  apiSend<ChatMacroRunResponse>({
    path: `${CHAT_MACROS_BASE}/run`,
    method: "POST",
    body: request
  })

export const getChatMacroRun = (
  runId: string
): Promise<ApiSendResponse<ChatMacroRunDetailResponse>> =>
  apiSend<ChatMacroRunDetailResponse>({
    path: `${CHAT_MACROS_BASE}/runs/${encodePathPart(runId)}`,
    method: "GET"
  })

export const cancelChatMacroRun = (
  runId: string
): Promise<ApiSendResponse<ChatMacroCancelResponse>> =>
  apiSend<ChatMacroCancelResponse>({
    path: `${CHAT_MACROS_BASE}/runs/${encodePathPart(runId)}/cancel`,
    method: "POST"
  })
