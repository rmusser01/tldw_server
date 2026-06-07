import type {
  CitationRef,
  EvidenceOrigin,
  KnowledgeAnswerTrustState,
  KnowledgeTrustMetadata,
  KnowledgeTrustReasonCode,
  RagResult,
} from "./types"

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
  backendTrust?: KnowledgeTrustMetadata | null
  hasRequiredMetadata?: boolean
  transportFailed?: boolean
  syncFailed?: boolean
  weakEvidence?: boolean
}

export type KnowledgeTrustResult = {
  state: KnowledgeAnswerTrustState
  reasonCodes: KnowledgeTrustReasonCode[]
  evidenceOrigin: EvidenceOrigin
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

const TRUST_REASON_MESSAGES: Record<string, string> = {
  missing_citations:
    "Generated answer is missing citations that map to returned sources.",
  citation_source_not_returned:
    "Generated citations do not map to returned sources.",
  missing_inspectable_evidence:
    "Cited sources do not include inspectable excerpts.",
  low_relevance:
    "Retrieved matches are below the relevance threshold.",
  web_fallback_used:
    "Web fallback contributed evidence to this answer.",
  no_evidence:
    "No searchable evidence was returned.",
  unclassified:
    "Answer trust could not be classified from the available metadata.",
}

function normalizeEvidenceOrigin(value: unknown): EvidenceOrigin {
  return value === "local_library" ||
    value === "web_fallback" ||
    value === "mixed" ||
    value === "unknown_origin"
    ? value
    : "unknown_origin"
}

function normalizeReasonCodes(value: unknown): KnowledgeTrustReasonCode[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((entry): entry is string => typeof entry === "string" && entry.length > 0)
}

function normalizeBackendTrust(
  backendTrust: KnowledgeTrustMetadata | null | undefined
): KnowledgeTrustResult | null {
  if (!backendTrust || !isKnowledgeAnswerTrustState(backendTrust.state)) {
    return null
  }
  return {
    state: backendTrust.state,
    reasonCodes: normalizeReasonCodes(
      backendTrust.reasonCodes ?? backendTrust.reason_codes
    ),
    evidenceOrigin: normalizeEvidenceOrigin(
      backendTrust.evidenceOrigin ?? backendTrust.evidence_origin
    ),
  }
}

function trustResult(
  state: KnowledgeAnswerTrustState,
  reasonCodes: KnowledgeTrustReasonCode[] = [],
  evidenceOrigin: EvidenceOrigin = "unknown_origin"
): KnowledgeTrustResult {
  return { state, reasonCodes, evidenceOrigin }
}

export function normalizeKnowledgeAnswerTrust(
  input: KnowledgeTrustInput
): KnowledgeTrustResult {
  if (input.transportFailed) return trustResult("failed_search")
  if (input.syncFailed) return trustResult("unsynced_local_result")

  const backendTrust = normalizeBackendTrust(input.backendTrust)
  if (backendTrust) return backendTrust

  if (!input.hasRequiredMetadata) return trustResult("unknown_trust")
  if (input.results.length === 0) {
    return trustResult("no_results", ["no_evidence"], "local_library")
  }
  if (input.weakEvidence && !input.answer) {
    return trustResult("no_answer_insufficient_evidence", ["low_relevance"])
  }
  if (input.answer && input.citations.length === 0) {
    return trustResult("uncited_degraded_answer", ["missing_citations"])
  }
  if (input.answer && input.citations.length > 0) {
    return trustResult("cited_answer")
  }
  return trustResult("unknown_trust", ["unclassified"])
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

export function getKnowledgeTrustReasonMessage(
  reasonCode: KnowledgeTrustReasonCode
): string | null {
  return TRUST_REASON_MESSAGES[reasonCode] ?? null
}

export function getKnowledgeTrustReasonMessages(
  reasonCodes: KnowledgeTrustReasonCode[]
): string[] {
  const messages = reasonCodes
    .map(getKnowledgeTrustReasonMessage)
    .filter((message): message is string => Boolean(message))
  return Array.from(new Set(messages))
}
