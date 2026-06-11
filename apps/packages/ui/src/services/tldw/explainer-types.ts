/**
 * Explainer workspace API contract: request payloads and response shapes.
 *
 * Single source of truth for both halves of the boundary — the client methods
 * in TldwApiClient and the workspace UI consume these same types.
 */

export type ExplainerMode = "goal" | "sources"
export type ExplainerOutputIntent = "explain" | "plan" | "both"
export type ExplainerGrounding = "source_only" | "source_led" | "open"
export type ExplainerDepthPreset = "quick" | "standard" | "deep"
export type ExplainerNodeKind = "question" | "answer" | "explanation" | "step" | "summary"
export type ExplainerNodeStatus = "idle" | "queued" | "generating" | "error" | "complete"
export type ExplainerEvidenceState =
  | "supported"
  | "partially_supported"
  | "uncited"
  | "insufficient"

export interface ExplainerSelectedSourcePayload {
  sourceId: string
  sourceType: string
  title: string
  snapshotVersion?: string | null
  metadata?: Record<string, unknown> | null
}

export interface ExplainerSessionCreatePayload {
  title: string
  mode: ExplainerMode
  outputIntent: ExplainerOutputIntent
  grounding: ExplainerGrounding
  depthPreset: ExplainerDepthPreset
  selectedSources: ExplainerSelectedSourcePayload[]
  rootPrompt: string
}

export interface ExplainerSessionPatchPayload {
  title?: string
  outputIntent?: ExplainerOutputIntent
  grounding?: ExplainerGrounding
  depthPreset?: ExplainerDepthPreset
  selectedSources?: ExplainerSelectedSourcePayload[]
}

export interface ExplainerNodeCreatePayload {
  parentId?: string | null
  title: string
  body?: string | null
  kind?: ExplainerNodeKind
  intent?: ExplainerOutputIntent
  status?: ExplainerNodeStatus
  evidenceState?: ExplainerEvidenceState
  outsideKnowledgeUsed?: boolean
  citations?: Record<string, unknown>[]
}

export interface ExplainerNodePatchPayload {
  title?: string | null
  body?: string | null
  status?: ExplainerNodeStatus
  evidenceState?: ExplainerEvidenceState
  outsideKnowledgeUsed?: boolean
  selectedOptionId?: string | null
  selectedCustomAnswer?: string | null
  questionOptions?: Record<string, unknown>[] | null
  generationMetadata?: Record<string, unknown> | null
  citations?: Record<string, unknown>[] | null
}

export interface ExplainerNodeExpandPayload {
  intent?: ExplainerOutputIntent | null
}

export interface ExplainerQuestionAnswerPayload {
  selectedOptionId?: string | null
  selectedCustomAnswer?: string | null
}

export interface ExplainerChatbookExportPayload {
  name?: string | null
  description?: string | null
  asyncMode?: boolean
}

export interface ExplainerSelectedSource {
  sourceId: string
  sourceType: string
  title: string
  addedAt?: string | null
  snapshotVersion?: string | null
  metadata?: Record<string, unknown> | null
}

export interface ExplainerCitation {
  id: string
  sourceId: string
  sourceType: string
  title: string
  excerpt: string
  locationLabel?: string | null
  startOffset?: number | null
  endOffset?: number | null
  url?: string | null
  snapshotHash?: string | null
}

export interface ExplainerNode {
  id: string
  sessionId: string
  parentId: string | null
  ordinal: number
  title: string
  body: string | null
  kind: ExplainerNodeKind
  intent: ExplainerOutputIntent
  status: ExplainerNodeStatus
  evidenceState: ExplainerEvidenceState
  outsideKnowledgeUsed: boolean
  citations: ExplainerCitation[]
  questionOptions?: Array<Record<string, unknown>> | null
  selectedOptionId?: string | null
  selectedCustomAnswer?: string | null
  generationMetadata?: Record<string, unknown> | null
  childNodeIds: string[]
  createdAt: string
  updatedAt: string
}

export interface ExplainerSession {
  id: string
  ownerUserId: string
  title: string
  mode: ExplainerMode
  status: string
  outputIntent: ExplainerOutputIntent
  grounding: ExplainerGrounding
  depthPreset: ExplainerDepthPreset
  selectedSources: ExplainerSelectedSource[]
  rootNodeIds: string[]
  nodes: Record<string, ExplainerNode>
  createdAt: string
  updatedAt: string
  archivedAt?: string | null
}

export interface ExplainerSessionSummary {
  id: string
  ownerUserId: string
  title: string
  mode: ExplainerMode
  status: string
  outputIntent: ExplainerOutputIntent
  grounding: ExplainerGrounding
  depthPreset: ExplainerDepthPreset
  nodeCount: number
  selectedSourceCount: number
  createdAt: string
  updatedAt: string
  archivedAt?: string | null
}

export interface ExplainerSessionListResponse {
  items: ExplainerSessionSummary[]
  total: number
  limit: number
  offset: number
}

export interface ExplainerJobAccepted {
  jobId: string
  sessionId: string
  nodeId: string
  status: string
}

export interface ExplainerJobStatus {
  jobId: string
  sessionId?: string | null
  nodeId?: string | null
  status: string
  progressPercent?: number | null
  progressMessage?: string | null
  error?: string | null
}

export interface ExplainerDeleteNodeResponse {
  id: string
  status: string
}

export interface ExplainerExportResponse {
  success: boolean
  message: string
  job_id?: string | null
  download_url?: string | null
}
