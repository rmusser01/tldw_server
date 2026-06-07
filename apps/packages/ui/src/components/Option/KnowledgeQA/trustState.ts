import type { CitationRef, KnowledgeAnswerTrustState, RagResult } from "./types"

export const KNOWLEDGE_ANSWER_TRUST_STATES: KnowledgeAnswerTrustState[] = [
  "cited_answer",
  "uncited_degraded_answer",
  "no_answer_insufficient_evidence",
  "no_results",
  "failed_search",
  "unsynced_local_result",
  "unknown_trust",
]

export type KnowledgeTrustInput = {
  answer: string | null
  results: RagResult[]
  citations: CitationRef[]
  hasRequiredMetadata?: boolean
  transportFailed?: boolean
  syncFailed?: boolean
  weakEvidence?: boolean
}

export type KnowledgeTrustResult = {
  state: KnowledgeAnswerTrustState
}

export type KnowledgeTrustTone = "success" | "warning" | "danger" | "muted"

export type KnowledgeTrustPresentation = {
  label: string
  tone: KnowledgeTrustTone
  degraded: boolean
}

const TRUST_PRESENTATION: Record<KnowledgeAnswerTrustState, KnowledgeTrustPresentation> = {
  cited_answer: {
    label: "Cited answer",
    tone: "success",
    degraded: false,
  },
  uncited_degraded_answer: {
    label: "Uncited answer",
    tone: "warning",
    degraded: true,
  },
  no_answer_insufficient_evidence: {
    label: "Insufficient evidence",
    tone: "warning",
    degraded: true,
  },
  no_results: {
    label: "No results",
    tone: "muted",
    degraded: true,
  },
  failed_search: {
    label: "Failed search",
    tone: "danger",
    degraded: true,
  },
  unsynced_local_result: {
    label: "Unsynced local result",
    tone: "warning",
    degraded: true,
  },
  unknown_trust: {
    label: "Trust unknown",
    tone: "warning",
    degraded: true,
  },
}

export function normalizeKnowledgeAnswerTrust(
  input: KnowledgeTrustInput
): KnowledgeTrustResult {
  if (input.transportFailed) return { state: "failed_search" }
  if (input.syncFailed) return { state: "unsynced_local_result" }
  if (!input.hasRequiredMetadata) return { state: "unknown_trust" }
  if (input.results.length === 0) return { state: "no_results" }
  if (input.weakEvidence && !input.answer) {
    return { state: "no_answer_insufficient_evidence" }
  }
  if (input.answer && input.citations.length === 0) {
    return { state: "uncited_degraded_answer" }
  }
  if (input.answer && input.citations.length > 0) {
    return { state: "cited_answer" }
  }
  return { state: "unknown_trust" }
}

export function isKnowledgeAnswerTrustState(
  value: unknown
): value is KnowledgeAnswerTrustState {
  return (
    typeof value === "string" &&
    KNOWLEDGE_ANSWER_TRUST_STATES.includes(value as KnowledgeAnswerTrustState)
  )
}

export function getKnowledgeAnswerTrustPresentation(
  state: KnowledgeAnswerTrustState
): KnowledgeTrustPresentation {
  return TRUST_PRESENTATION[state]
}

export function getKnowledgeAnswerTrustLabel(
  state: KnowledgeAnswerTrustState
): string {
  return getKnowledgeAnswerTrustPresentation(state).label
}
