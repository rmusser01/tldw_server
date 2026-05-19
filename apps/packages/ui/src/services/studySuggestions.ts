import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type SuggestionAnchorType = "quiz_attempt" | "flashcard_review_session"
export type SuggestionStatus = "none" | "pending" | "ready" | "failed"
export type SuggestionTargetService = "quiz" | "flashcards"
export type SuggestionDisposition = "opened_existing" | "generated"

export type StudySuggestionStatusResponse = {
  anchor_type: SuggestionAnchorType
  anchor_id: number
  status: SuggestionStatus
  job_id?: number | null
  snapshot_id?: number | null
}

export type StudySuggestionSnapshotTopic = {
  id: string
  display_label?: string | null
  topic_key?: string | null
  canonical_label?: string | null
  evidence_reasons?: string[] | null
  type?: string | null
  status?: string | null
  selected?: boolean | null
  source_type?: string | null
  source_id?: string | null
}

export type StudySuggestionSnapshotResource = {
  id: number
  service: string
  activity_type: string
  anchor_type: SuggestionAnchorType
  anchor_id: number
  suggestion_type: string
  status: string
  payload: {
    summary?: Record<string, unknown>
    counts?: Record<string, unknown>
    topics?: StudySuggestionSnapshotTopic[]
    [key: string]: unknown
  } | Record<string, unknown> | string | null
  user_selection?: Record<string, unknown> | unknown[] | string | null
  refreshed_from_snapshot_id?: number | null
  created_at?: string | null
  last_modified?: string | null
}

export type StudySuggestionSnapshotResponse = {
  snapshot: StudySuggestionSnapshotResource
  live_evidence: Record<string, unknown>
}

export type StudySuggestionRefreshRequest = {
  reason?: string | null
}

export type StudySuggestionRefreshResponse = {
  job: {
    id: number
    status: string
  }
}

export type StudySuggestionActionRequest = {
  targetService: SuggestionTargetService
  targetType: string
  actionKind: string
  selectedTopicIds: string[]
  selectedTopicEdits?: Array<{
    id: string
    label: string
  }>
  manualTopicLabels?: string[]
  hasExplicitSelection?: boolean
  generatorVersion?: string
  forceRegenerate?: boolean
}

export type StudySuggestionActionResponse = {
  disposition: SuggestionDisposition
  snapshot_id: number
  selection_fingerprint: string
  target_service: SuggestionTargetService
  target_type: string
  target_id: string
}

const encodeSegment = (value: string | number): string =>
  encodeURIComponent(String(value))

export const parseStudySuggestionTargetId = (
  targetId: string | null | undefined
): number | undefined => {
  if (typeof targetId !== "string") {
    return undefined
  }

  const normalized = targetId.trim()
  if (!normalized) {
    return undefined
  }

  if (/^\d+$/.test(normalized)) {
    const parsed = Number.parseInt(normalized, 10)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
  }

  const prefixedMatch = /^(?:quiz|deck)-(\d+)$/i.exec(normalized)
  if (!prefixedMatch) {
    return undefined
  }

  const parsed = Number.parseInt(prefixedMatch[1], 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
}

export async function getStudySuggestionAnchorStatus(
  anchorType: SuggestionAnchorType,
  anchorId: number,
  options?: { signal?: AbortSignal }
): Promise<StudySuggestionStatusResponse> {
  return await bgRequest<StudySuggestionStatusResponse, AllowedPath, "GET">({
    path: `/api/v1/study-suggestions/anchors/${encodeSegment(anchorType)}/${encodeSegment(anchorId)}/status` as any,
    method: "GET",
    abortSignal: options?.signal
  })
}

export async function getStudySuggestionSnapshot(
  snapshotId: number,
  options?: { signal?: AbortSignal }
): Promise<StudySuggestionSnapshotResponse> {
  return await bgRequest<StudySuggestionSnapshotResponse, AllowedPath, "GET">({
    path: `/api/v1/study-suggestions/snapshots/${encodeSegment(snapshotId)}` as any,
    method: "GET",
    abortSignal: options?.signal
  })
}

export async function refreshStudySuggestionSnapshot(
  snapshotId: number,
  request: StudySuggestionRefreshRequest = {},
  options?: { signal?: AbortSignal }
): Promise<StudySuggestionRefreshResponse> {
  return await bgRequest<StudySuggestionRefreshResponse, AllowedPath, "POST">({
    path: `/api/v1/study-suggestions/snapshots/${encodeSegment(snapshotId)}/refresh` as any,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: request,
    abortSignal: options?.signal
  })
}

export async function performStudySuggestionAction(
  snapshotId: number,
  request: StudySuggestionActionRequest,
  options?: { signal?: AbortSignal }
): Promise<StudySuggestionActionResponse> {
  return await bgRequest<StudySuggestionActionResponse, AllowedPath, "POST">({
    path: `/api/v1/study-suggestions/snapshots/${encodeSegment(snapshotId)}/actions` as any,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: {
      target_service: request.targetService,
      target_type: request.targetType,
      action_kind: request.actionKind,
      selected_topic_ids: request.selectedTopicIds,
      selected_topic_edits: request.selectedTopicEdits ?? [],
      manual_topic_labels: request.manualTopicLabels ?? [],
      has_explicit_selection: request.hasExplicitSelection ?? false,
      generator_version: request.generatorVersion,
      force_regenerate: request.forceRegenerate ?? false
    },
    abortSignal: options?.signal
  })
}
