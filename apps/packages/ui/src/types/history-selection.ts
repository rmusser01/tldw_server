/** H1 wire values. Owners construct snapshots and admissions; views only select them. */
export type HistoryCursorV1 =
  | { readonly kind: "after_message"; readonly message_id: string }
  | { readonly kind: "before_message"; readonly message_id: string }
  | { readonly kind: "empty" }

export type HistoryInterpretationV1 =
  | { readonly kind: "parent_graph_v1" }
  | { readonly kind: "legacy_linear_v1"; readonly projection_id: string }

export type HistoryInterpretationStatusV1 =
  | { readonly kind: "parent_graph_v1" }
  | { readonly kind: "legacy_linear_v1"; readonly projection_id: string; readonly ordered_path_ids: readonly string[] }
  | { readonly kind: "legacy_review_required" }

export type HistoryMessageRevisionV1 = { readonly id: string; readonly revision: string }
export type HistoryFencesV1 = { readonly conversation: string; readonly history: string; readonly settings: string }
export type HistoryRequiredReferenceV1 = { readonly id: string; readonly revision: string; readonly kind: string }

export type HistoryNodeV1 = HistoryMessageRevisionV1 & {
  readonly preview?: string | null
  readonly legacy_projection_id?: string | null
  readonly parent_id: string | null
  readonly role: string
  readonly settled: boolean
  readonly conversation_id?: string | null
  readonly metadata?: readonly HistoryRequiredReferenceV1[]
  readonly assets?: readonly HistoryRequiredReferenceV1[]
  readonly comparison?: { readonly cluster_id: string; readonly model_id: string | null; readonly common: boolean } | null
}

/** Coherently loaded content for one selected manifest member.
 * `message` and the complete `images` array match the existing chat Message adapter.
 */
export type HistorySelectedContentV1 = HistoryMessageRevisionV1 & {
  readonly message: string
  readonly images: readonly string[]
  readonly tool_calls?: readonly Record<string, unknown>[] | null
  readonly extra_metadata?: Readonly<Record<string, unknown>> | null
}

export type NativeForkContextV1 = {
  readonly policy: "plain_v1"
  readonly storage_context_digest: string
  readonly supported: boolean
}

export type HistorySelectionSnapshotV1 = {
  readonly native_fork_context?: NativeForkContextV1 | null
  readonly version: 1
  readonly owner_key: string
  readonly conversation_id: string
  readonly fences: HistoryFencesV1
  readonly nodes: readonly HistoryNodeV1[]
  readonly source_digest: string
  readonly interpretation_status: HistoryInterpretationStatusV1
  readonly storage_context_digest: string
}

export type HistoryViewSelectionV1 = {
  readonly view_session_id: string
  readonly owner_key: string
  readonly conversation_id: string
  readonly interpretation: HistoryInterpretationV1
  readonly cursor: HistoryCursorV1
  readonly selection_revision: number
}

export type HistorySelectionV1 = {
  readonly version: 1
  readonly owner_key: string
  readonly conversation_id: string
  readonly interpretation: HistoryInterpretationV1
  readonly cursor: HistoryCursorV1
  readonly selection_revision: number
  readonly purpose: "send" | "fork"
  readonly messages: readonly HistoryMessageRevisionV1[]
  readonly fences: HistoryFencesV1
  readonly storage_context_digest: string
  readonly request_context_digest: string
  readonly selection_digest: string
}

export type HistoryResolutionFailureV1 = {
  readonly status: "legacy_review_required" | "stale_selection" | "invalid_history" | "unsupported_history_capability"
  readonly code: string
}
export type HistoryResolutionV1 =
  | { readonly status: "ready"; readonly selection: HistorySelectionV1; readonly rows: readonly HistoryNodeV1[] }
  | HistoryResolutionFailureV1

export type HistorySelectionCaptureV1 = {
  readonly status: "captured"
  readonly snapshot: HistorySelectionSnapshotV1
  readonly rows: readonly HistoryNodeV1[]
  readonly selected_content: readonly HistorySelectedContentV1[]
  readonly view: HistoryViewSelectionV1
  readonly purpose: "send" | "fork"
  readonly storage_context_digest: string
}
export type HistoryCaptureResultV1 = HistorySelectionCaptureV1 | (HistoryResolutionFailureV1 & {
  readonly snapshot: HistorySelectionSnapshotV1
  readonly view: HistoryViewSelectionV1
})

/** Null owner is allowed only for a fresh read-only bootstrap under a verified client lease. */
export type HistoryCaptureRequestV1 = {
  readonly view: Omit<HistoryViewSelectionV1, "owner_key"> & { readonly owner_key?: string | null }
  readonly purpose: "send" | "fork"
}

export type PreparedHistoryContextV1 = {
  readonly payload: unknown
  readonly request_context_digest: string
  /** Existing composer/request/connection lease, rechecked before dispatch. */
  readonly validate_lease: () => boolean
}

export type LegacyHistoryProjectionConfirmV1 = {
  readonly version: 1
  readonly projection_id: string
  readonly owner_key: string
  readonly conversation_id: string
  readonly source_digest: string
  readonly fences: HistoryFencesV1
  readonly source_members: readonly HistoryMessageRevisionV1[]
  readonly ordered_path_ids: readonly string[]
  readonly cursor: HistoryCursorV1
  readonly selection_revision: number
}

export type LegacyHistoryProjectionV1 = LegacyHistoryProjectionConfirmV1 & {
  readonly projection_digest: string
  readonly created_at: string
}

export type HistoryAdmissionReferenceV1 = {
  readonly version: 1
  readonly owner_key: string
  readonly conversation_id: string
  readonly input_message_id: string
  readonly input_message_revision: string
  readonly selection_digest: string
}
export type HistoryAdmissionV1 = HistoryAdmissionReferenceV1 & {
  readonly messages: readonly HistoryMessageRevisionV1[]
  readonly originating_selection_revision: number
}

export type CompareHistorySelectionV1 = {
  readonly version: 1
  readonly owner_key: string
  readonly conversation_id: string
  readonly model_id: string
  readonly cluster_id: string | null
  readonly cursor: HistoryCursorV1
  readonly messages: readonly HistoryMessageRevisionV1[]
  readonly fences: HistoryFencesV1
  readonly storage_context_digest: string
  readonly request_context_digest: string
  readonly selection_digest: string
}

export type ForkRequestV1 = {
  readonly operation_id: string
  readonly owner_key: string
  readonly destination_owner_key: string
  readonly request_digest: string
  readonly input:
    | { readonly kind: "normal"; readonly selection: HistorySelectionV1 }
    | { readonly kind: "comparison"; readonly selection: CompareHistorySelectionV1 }
}

export type LocalForkProjectionV1 = {
  readonly request: ForkRequestV1
  readonly source_fences: HistoryFencesV1
  readonly source_members: readonly HistoryMessageRevisionV1[]
  readonly id_map: Readonly<Record<string, string>>
  readonly child_rows: readonly unknown[]
  readonly child_files: readonly unknown[]
}

export type ForkResultV1 =
  | {
      readonly state: "committed"
      readonly operation_id: string
      readonly owner_key: string
      readonly child_id: string
      readonly message_map: Record<string, string>
    }
  | {
      readonly state: "legacy_completed"
      readonly operation_id: string
      readonly owner_key: string
      readonly child_id: string
    }
  | {
      readonly state: "rejected" | "blocked"
      readonly operation_id: string
      readonly owner_key: string
      readonly code: string
    }
  | {
      readonly state: "unknown" | "partial"
      readonly operation_id: string
      readonly owner_key: string
      readonly code: string
      readonly candidate_child_id?: string
    }
