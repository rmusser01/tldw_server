import type {
  ExplainerNode,
  ExplainerSelectedSource
} from "@/services/tldw/explainer-types"

export type {
  ExplainerCitation,
  ExplainerDepthPreset,
  ExplainerEvidenceState,
  ExplainerExportResponse,
  ExplainerGrounding,
  ExplainerJobStatus,
  ExplainerMode,
  ExplainerNode,
  ExplainerNodeKind,
  ExplainerNodeStatus,
  ExplainerOutputIntent,
  ExplainerSelectedSource,
  ExplainerSession,
  ExplainerSessionListResponse,
  ExplainerSessionSummary
} from "@/services/tldw/explainer-types"

export interface ExplainerSourceCandidate extends ExplainerSelectedSource {
  description?: string | null
}

export interface FlattenedExplainerNode {
  node: ExplainerNode
  depth: number
}
