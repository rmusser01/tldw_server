/**
 * Node-Based Workflow Editor Types
 * Type definitions for the visual workflow canvas editor
 */

import type { Node, Edge, XYPosition } from "@xyflow/react"

// ─────────────────────────────────────────────────────────────────────────────
// Step Types (Node Categories)
// ─────────────────────────────────────────────────────────────────────────────

export type WorkflowStepType = string

export type StepCategory =
  | "ai"       // AI & LLM
  | "search"   // Search & RAG
  | "media"    // Media & Documents
  | "text"     // Text & Data Transform
  | "research" // Research & Academic
  | "audio"    // Audio
  | "video"    // Video & Subtitles
  | "control"  // Control Flow
  | "io"       // Integrations
  | "utility"  // Utility

// ─────────────────────────────────────────────────────────────────────────────
// Workflow Step Schema (Server-Driven)
// ─────────────────────────────────────────────────────────────────────────────

export type WorkflowStepSchema = {
  type?: string | string[]
  description?: string
  properties?: Record<string, WorkflowStepSchema>
  required?: string[]
  enum?: Array<string | number | boolean | null>
  items?: WorkflowStepSchema
  default?: unknown
  anyOf?: WorkflowStepSchema[]
  oneOf?: WorkflowStepSchema[]
  allOf?: WorkflowStepSchema[]
  additionalProperties?: boolean | WorkflowStepSchema
}

export interface WorkflowStepTypeInfo {
  name: string
  description?: string
  schema?: WorkflowStepSchema
  example?: unknown
  min_engine_version?: string
}

export interface WorkflowDefinitionStepPayload {
  id: string
  name?: string
  type: string
  config?: Record<string, unknown>
  retry?: number | null
  timeout_seconds?: number | null
  on_success?: string | null
  on_failure?: string | null
  on_timeout?: string | null
}

export interface WorkflowDefinitionPayload {
  name: string
  version?: number
  description?: string | null
  tags?: string[]
  inputs?: Record<string, unknown>
  on_completion_webhook?: Record<string, unknown> | string | null
  steps: WorkflowDefinitionStepPayload[]
  metadata?: Record<string, unknown>
  visibility?: "private"
}

export interface SavedWorkflowSummary {
  id: number
  name: string
  version: number
  description?: string | null
  tags?: string[]
  is_active: boolean
}

export interface SavedWorkflowDefinition {
  id: number
  name: string
  version: number
  is_active: boolean
  definition: WorkflowDefinitionPayload
}

export interface ServerWorkflowRun {
  run_id: string
  workflow_id?: number | null
  user_id?: string | null
  status: "queued" | "running" | "paused" | "waiting_human" | "waiting_approval" | "succeeded" | "failed" | "cancelled"
  inputs: Record<string, unknown>
  outputs?: Record<string, unknown> | null
  error?: string | null
  status_reason?: string | null
  definition_version?: number | null
}

export type WorkflowInputValidator = (data: unknown, schema: Record<string, unknown>) => Promise<string[]>

export interface WorkflowFailureSummary {
  reason_code_core?: string | null
  reason_code_detail?: string | null
  category?: string | null
  blame_scope?: string | null
  retryable?: boolean | null
  retry_recommendation?: string | null
  error_summary?: string | null
  internal_detail?: Record<string, unknown> | null
}

export interface WorkflowStepAttempt {
  attempt_id: string
  step_run_id: string
  step_id: string
  attempt_number: number
  status: string
  reason_code_core?: string | null
  reason_code_detail?: string | null
  retryable?: boolean | null
  error_summary?: string | null
  started_at?: string | null
  ended_at?: string | null
  duration_ms?: number | null
  metadata?: Record<string, unknown>
  [key: string]: unknown
}

export interface WorkflowRunStepSummary {
  step_id: string
  step_run_id?: string
  name?: string | null
  type?: string | null
  status?: string
  attempt_count?: number
  latest_failure?: WorkflowFailureSummary | null
  [key: string]: unknown
}

export interface WorkflowRunInvestigation {
  run_id: string
  status: string
  schema_version: number
  derived_from_event_seq: number
  failed_step?: WorkflowRunStepSummary | null
  primary_failure?: WorkflowFailureSummary | null
  attempts: WorkflowStepAttempt[]
  evidence: Record<string, unknown>
  recommended_actions: string[]
}

export interface WorkflowStepAttemptsResponse {
  run_id: string
  step_id: string
  attempts: WorkflowStepAttempt[]
}

export interface WorkflowValidationIssue {
  code: string
  message: string
  step_id?: string | null
  step_type?: string | null
}

export interface WorkflowPreflightRequest {
  definition: WorkflowDefinitionPayload
  validation_mode?: "block" | "non-block"
}

export interface WorkflowPreflightResult {
  valid: boolean
  errors: WorkflowValidationIssue[]
  warnings: WorkflowValidationIssue[]
}

// StepTypeMetadata is defined in step-registry.ts to avoid circular deps

// ─────────────────────────────────────────────────────────────────────────────
// Port/Handle Definitions
// ─────────────────────────────────────────────────────────────────────────────

export type PortDataType =
  | "any"
  | "string"
  | "number"
  | "boolean"
  | "array"
  | "object"
  | "file"
  | "audio"
  | "control" // For control flow (branch conditions)

export interface PortDefinition {
  id: string
  label: string
  dataType: PortDataType
  required?: boolean
  multiple?: boolean // Can accept multiple connections
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Configuration Schemas
// ─────────────────────────────────────────────────────────────────────────────

export type ConfigFieldType =
  | "text"
  | "textarea"
  | "number"
  | "select"
  | "multiselect"
  | "checkbox"
  | "model-picker"
  | "collection-picker"
  | "template-editor"
  | "json-editor"
  | "url"
  | "duration"

export interface ConfigFieldSchema {
  key: string
  type: ConfigFieldType
  label: string
  description?: string
  required?: boolean
  default?: unknown
  options?: Array<{ value: string; label: string }>
  validation?: {
    min?: number
    max?: number
    pattern?: string
    message?: string
  }
  showWhen?: {
    field: string
    value: unknown
  }
}

export type StepConfigSchema = ConfigFieldSchema[]

// ─────────────────────────────────────────────────────────────────────────────
// Node Data (Step Instance Data)
// ─────────────────────────────────────────────────────────────────────────────

export interface BaseStepData {
  label: string
  stepType: WorkflowStepType
  config: Record<string, unknown>
  isExpanded?: boolean
}

// Specific step data types

export interface PromptStepData extends BaseStepData {
  stepType: "prompt"
  config: {
    model?: string
    systemPrompt?: string
    userPromptTemplate?: string
    temperature?: number
    maxTokens?: number
    stopSequences?: string[]
  }
}

export interface RagSearchStepData extends BaseStepData {
  stepType: "rag_search"
  config: {
    collectionId?: string
    queryTemplate?: string
    topK?: number
    minScore?: number
  }
}

export interface MediaIngestStepData extends BaseStepData {
  stepType: "media_ingest"
  config: {
    sourceType?: "url" | "file"
    url?: string
    extractAudio?: boolean
    transcribe?: boolean
    chunkingStrategy?: "sentence" | "paragraph" | "fixed"
  }
}

export interface BranchStepData extends BaseStepData {
  stepType: "branch"
  config: {
    conditions: Array<{
      id: string
      expression: string
      outputId: string
    }>
    defaultOutputId?: string
  }
}

export interface MapStepData extends BaseStepData {
  stepType: "map"
  config: {
    arrayPath?: string
    itemVariable?: string
    maxParallel?: number
  }
}

export interface WaitForHumanStepData extends BaseStepData {
  stepType: "wait_for_human"
  config: {
    promptMessage?: string
    allowEdit?: boolean
    editableFields?: string[]
    timeoutSeconds?: number
    defaultAction?: "approve" | "reject"
  }
}

export interface WebhookStepData extends BaseStepData {
  stepType: "webhook"
  config: {
    url?: string
    method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE"
    headers?: Record<string, string>
    bodyTemplate?: string
    responseMapping?: string
  }
}

export interface TtsStepData extends BaseStepData {
  stepType: "tts"
  config: {
    voice?: string
    speed?: number
    format?: "mp3" | "opus" | "aac" | "flac" | "wav" | "pcm"
  }
}

export interface SttTranscribeStepData extends BaseStepData {
  stepType: "stt_transcribe"
  config: {
    model?: string
    language?: string
    punctuate?: boolean
  }
}

export interface DelayStepData extends BaseStepData {
  stepType: "delay"
  config: {
    durationSeconds?: number
  }
}

export interface LogStepData extends BaseStepData {
  stepType: "log"
  config: {
    level?: "debug" | "info" | "warn" | "error"
    messageTemplate?: string
  }
}

export interface StartStepData extends BaseStepData {
  stepType: "start"
  config: {
    inputSchema?: Record<string, unknown>
  }
}

export interface EndStepData extends BaseStepData {
  stepType: "end"
  config: {
    outputMapping?: string
  }
}

export type StepData =
  | PromptStepData
  | RagSearchStepData
  | MediaIngestStepData
  | BranchStepData
  | MapStepData
  | WaitForHumanStepData
  | WebhookStepData
  | TtsStepData
  | SttTranscribeStepData
  | DelayStepData
  | LogStepData
  | StartStepData
  | EndStepData

// ─────────────────────────────────────────────────────────────────────────────
// React Flow Node/Edge Types
// ─────────────────────────────────────────────────────────────────────────────

// Use Record<string, unknown> for React Flow compatibility while retaining StepData shape
export interface WorkflowNodeData extends Record<string, unknown> {
  label: string
  stepType: WorkflowStepType
  config: Record<string, unknown>
  isExpanded?: boolean
}

export type WorkflowNode = Node<WorkflowNodeData, WorkflowStepType>
export type WorkflowEdge = Edge<{
  dataType?: PortDataType
  animated?: boolean
}>

// ─────────────────────────────────────────────────────────────────────────────
// Execution State Types
// ─────────────────────────────────────────────────────────────────────────────

export type StepExecutionStatus =
  | "idle"
  | "queued"
  | "running"
  | "success"
  | "failed"
  | "skipped"
  | "waiting_human"
  | "waiting_approval"
  | "cancelled"

export interface StepExecutionState {
  nodeId: string
  status: StepExecutionStatus
  startedAt?: number
  completedAt?: number
  durationMs?: number
  tokensUsed?: number
  error?: string
  output?: unknown
  streamingOutput?: string // For streaming prompt outputs
  artifacts?: StepArtifact[]
}

export interface StepArtifact {
  id: string
  type: "file" | "audio" | "image" | "text"
  name: string
  url?: string
  data?: string // Base64 or inline data
  mimeType?: string
}

// ─────────────────────────────────────────────────────────────────────────────
// Human-in-the-Loop Types
// ─────────────────────────────────────────────────────────────────────────────

export interface HumanApprovalRequest {
  id: string
  nodeId: string
  nodeName: string
  promptMessage: string
  dataToReview: unknown
  editableFields?: string[]
  allowEdit: boolean
  createdAt: number
  timeoutAt?: number
}

export interface HumanApprovalResponse {
  requestId: string
  action: "approve" | "reject"
  editedData?: unknown
  reason?: string
  respondedAt: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Workflow Definition (Server Format)
// ─────────────────────────────────────────────────────────────────────────────

export interface ServerWorkflowDefinition {
  id?: string
  name: string
  description?: string
  version?: number
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
  metadata?: {
    createdAt?: number
    updatedAt?: number
    createdBy?: string
    tags?: string[]
    thumbnail?: string
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Workflow Run State
// ─────────────────────────────────────────────────────────────────────────────

export type WorkflowRunStatus =
  | "idle"
  | "queued"
  | "running"
  | "paused"
  | "waiting_human"
  | "waiting_approval"
  | "completed"
  | "failed"
  | "cancelled"

export interface WorkflowRunState {
  runId: string | null
  status: WorkflowRunStatus
  startedAt?: number
  completedAt?: number
  nodeStates: Record<string, StepExecutionState>
  currentNodeId?: string
  pendingApproval?: HumanApprovalRequest
  error?: string
}

// ─────────────────────────────────────────────────────────────────────────────
// Editor UI State
// ─────────────────────────────────────────────────────────────────────────────

export type EditorTool =
  | "select"
  | "pan"
  | "connect"

export type SidebarPanel =
  | "palette"
  | "config"
  | "execution"
  | "templates"
  | null

export interface EditorUIState {
  selectedNodeIds: string[]
  selectedEdgeIds: string[]
  activeTool: EditorTool
  sidebarPanel: SidebarPanel
  isMiniMapVisible: boolean
  isGridVisible: boolean
  zoom: number
  panPosition: XYPosition
  isDragging: boolean
  isConnecting: boolean
}

// ─────────────────────────────────────────────────────────────────────────────
// Undo/Redo History
// ─────────────────────────────────────────────────────────────────────────────

export interface EditorHistoryEntry {
  id: string
  timestamp: number
  description: string
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
}

export interface EditorHistory {
  past: EditorHistoryEntry[]
  present: EditorHistoryEntry
  future: EditorHistoryEntry[]
  maxHistorySize: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Workflow Template
// ─────────────────────────────────────────────────────────────────────────────

export interface WorkflowTemplate {
  id: string
  name: string
  description: string
  category: string
  thumbnail?: string
  tags: string[]
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
  difficulty?: "beginner" | "intermediate" | "advanced"
  estimatedDuration?: string
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation Types
// ─────────────────────────────────────────────────────────────────────────────

export type ValidationSeverity = "error" | "warning" | "info"

export interface ValidationIssue {
  id: string
  nodeId?: string
  edgeId?: string
  severity: ValidationSeverity
  message: string
  field?: string
}

export interface WorkflowValidation {
  isValid: boolean
  issues: ValidationIssue[]
}
