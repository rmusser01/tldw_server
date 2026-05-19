export type WebClipperDestination = "note" | "workspace" | "both"

export type WebClipperOutcomeState = "saved" | "saved_with_warnings" | "partially_saved" | "failed"

export type WebClipperEnrichmentStatus = "pending" | "running" | "complete" | "failed"

export type WebClipperEnrichmentType = "ocr" | "vlm"

export interface WebClipperNotePayload {
  title?: string | null
  comment?: string | null
  folder_id?: number | null
  keywords: string[]
}

export interface WebClipperWorkspacePayload {
  workspace_id: string
}

export interface WebClipperContentPayload {
  visible_body?: string | null
  full_extract?: string | null
  selected_text?: string | null
}

export interface WebClipperAttachmentPayload {
  slot: string
  file_name?: string | null
  media_type: string
  text_content?: string | null
  content_base64?: string | null
  source_url?: string | null
}

export interface WebClipperEnhancementOptions {
  run_ocr?: boolean
  run_vlm?: boolean
}

export interface WebClipperSaveRequest {
  clip_id: string
  clip_type: string
  source_url: string
  source_title: string
  destination_mode?: WebClipperDestination
  note?: WebClipperNotePayload
  workspace?: WebClipperWorkspacePayload | null
  content?: WebClipperContentPayload
  attachments?: WebClipperAttachmentPayload[]
  enhancements?: WebClipperEnhancementOptions
  capture_metadata?: Record<string, unknown>
  source_note_version?: number | null
}

export interface WebClipperEnrichmentPayload {
  clip_id: string
  enrichment_type: WebClipperEnrichmentType
  status: WebClipperEnrichmentStatus
  inline_summary?: string | null
  structured_payload?: Record<string, unknown>
  source_note_version: number
  error?: string | null
}

export interface WebClipperSavedNote {
  id: string
  title: string
  version: number
}

export interface WebClipperWorkspacePlacement {
  workspace_id: string
  workspace_note_id: number
  source_note_id: string
  source_note_version?: number | null
}

export interface WebClipperAttachmentRecord {
  slot: string
  file_name: string
  original_file_name: string
  content_type?: string | null
  size_bytes: number
  uploaded_at: string
  url: string
}

export interface WebClipperSaveResponse {
  clip_id: string
  status: WebClipperOutcomeState
  note: WebClipperSavedNote | null
  workspace_placement: WebClipperWorkspacePlacement | null
  attachments: WebClipperAttachmentRecord[]
  warnings: string[]
  note_id: string
  workspace_placement_saved: boolean
  workspace_placement_count: number
}

export interface WebClipperStatusResponse {
  clip_id: string
  status: WebClipperOutcomeState
  note: WebClipperSavedNote
  workspace_placements: WebClipperWorkspacePlacement[]
  attachments: WebClipperAttachmentRecord[]
  analysis: Record<string, unknown>
  content_budget: Record<string, unknown>
}

export interface WebClipperEnrichmentResponse {
  clip_id: string
  enrichment_type: WebClipperEnrichmentType
  status: WebClipperEnrichmentStatus
  source_note_version: number
  inline_applied: boolean
  inline_summary?: string | null
  conflict_reason?: string | null
  warnings: string[]
}
