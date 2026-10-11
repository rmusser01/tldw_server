import type { HistoryAdmissionReferenceV1, HistorySelectionV1, PreparedHistoryContextV1 } from "./history-selection"

export type HistoryDurableSourceMetadataV1 = {
  readonly source?: string
  readonly title?: string
  readonly chunk_id?: string
  readonly retrieval_strategy?: string
  readonly source_type?: string
  readonly selection_reason?: string
  readonly score?: number
  readonly page?: number
  readonly media_id?: string
  readonly author?: string
  readonly chunk_index?: number
  readonly total_chunks?: number
  readonly start_char?: number
  readonly end_char?: number
  readonly chunk_start?: number
  readonly chunk_end?: number
  readonly loc?: { readonly lines: { readonly from: number; readonly to: number } }
}

export type HistoryDurableSourceV1 = {
  readonly name: string
  readonly type: string
  readonly mode: "rag"
  readonly url: string
  readonly pageContent: string
  readonly metadata: HistoryDurableSourceMetadataV1
}

export type HistoryDurableResultV1 = {
  readonly version: 1
  readonly sources: readonly HistoryDurableSourceV1[]
}

/** Server-owned observation; meaningful only after matching the dispatched request. */
export type HistoryDurableResultReceiptV1 = HistoryDurableResultV1 & {
  readonly result_message_id: string
  readonly result_message_revision: "1"
  readonly admission: HistoryAdmissionReferenceV1
  readonly request_context_digest: string
}

export type HistoryDurableEnvelopeV1 =
  | { readonly version: 1; readonly kind: "selection"; readonly selection: HistorySelectionV1 }
  | { readonly version: 1; readonly kind: "admission"; readonly admission: HistoryAdmissionReferenceV1; readonly request_context_digest: string }

/** Exact dispatched body, not transport scope, credentials or capability authority. */
export type HistoryDurableRequestBodyV1 = Readonly<Record<string, unknown>> & {
  readonly stream: boolean
  readonly model: string
  readonly api_provider: string
  readonly save_to_db: true
  readonly conversation_id: string
  readonly messages: readonly { readonly role: "system" | "user"; readonly content: string }[]
  readonly tldw_turn: {
    readonly user_message_id: string
    readonly result_v1: HistoryDurableResultV1
    readonly history_v1: HistoryDurableEnvelopeV1
  }
}

export type PreparedHistoryDurableTurnV1 = {
  readonly body: HistoryDurableRequestBodyV1
  readonly prepared: PreparedHistoryContextV1
  readonly request_context_digest: string
}
