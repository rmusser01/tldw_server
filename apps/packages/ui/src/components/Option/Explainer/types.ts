import type {
  ExplainerDepthPreset,
  ExplainerEvidenceState,
  ExplainerGrounding,
  ExplainerMode,
  ExplainerNodeKind,
  ExplainerNodeStatus,
  ExplainerOutputIntent
} from "@/services/tldw/TldwApiClient"

export type {
  ExplainerDepthPreset,
  ExplainerEvidenceState,
  ExplainerGrounding,
  ExplainerMode,
  ExplainerNodeKind,
  ExplainerNodeStatus,
  ExplainerOutputIntent
}

export interface ExplainerSelectedSource {
  sourceId: string
  sourceType: string
  title: string
  addedAt?: string | null
  snapshotVersion?: string | null
  metadata?: Record<string, unknown> | null
}

export interface ExplainerSourceCandidate extends ExplainerSelectedSource {
  description?: string | null
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

export interface ExplainerJobStatus {
  jobId: string
  sessionId?: string | null
  nodeId?: string | null
  status: string
  progressPercent?: number | null
  progressMessage?: string | null
  error?: string | null
}

export interface ExplainerExportResponse {
  success: boolean
  message: string
  job_id?: string | null
  download_url?: string | null
}

export interface FlattenedExplainerNode {
  node: ExplainerNode
  depth: number
}
