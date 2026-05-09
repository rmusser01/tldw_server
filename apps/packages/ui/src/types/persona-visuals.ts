export type PersonaVisualStateId =
  | "idle"
  | "wake_armed"
  | "listening"
  | "thinking"
  | "speaking"
  | "tool_running"
  | "approval_needed"
  | "error"
  | "offline"

export type PersonaVisualRendererType =
  | "sprite_frames"
  | "sprite_sheet"
  | "static_image"
  | "live2d"

export type PersonaVisualPackStatus =
  | "draft"
  | "review"
  | "active"
  | "archived"
  | "failed"

export type PersonaVisualAssetRole =
  | "frame"
  | "still_pose"
  | "sprite_sheet"
  | "preview"
  | "generated_candidate"

export type PersonaVisualCandidateStatus =
  | "review"
  | "accepted"
  | "rejected"
  | "failed"

export interface PersonaVisualRegion {
  x: number
  y: number
  width: number
  height: number
}

export interface PersonaVisualFrame {
  asset_id: string
  region?: PersonaVisualRegion | null
  duration_ms?: number
}

export interface PersonaVisualAnimation {
  frames?: PersonaVisualFrame[]
  asset_ids?: string[]
  frame_rate?: number
  loop?: boolean
  alignment?: { x: number; y: number }
  preview_frame?: number
  preview_asset_id?: string
}

export interface PersonaVisualAuthoredTrigger {
  id: string
  source: "live_state" | "tool_category" | "mcp_runtime"
  match: string
  state: PersonaVisualStateId
  duration_ms: number
  priority: number
}

export interface PersonaVisualManifest {
  manifest_version: 1
  renderer_type: PersonaVisualRendererType
  states: Partial<Record<PersonaVisualStateId, { animation_id: string }>>
  animations: Record<string, PersonaVisualAnimation>
  fallbacks?: Partial<Record<PersonaVisualStateId, PersonaVisualStateId[]>>
  authored_triggers?: PersonaVisualAuthoredTrigger[]
}

export interface PersonaVisualAsset {
  id: string
  pack_id?: string
  persona_id?: string
  asset_role: PersonaVisualAssetRole | string
  storage_key?: string
  url: string
  original_filename?: string | null
  mime_type: string
  byte_size?: number
  checksum_sha256?: string
  width?: number | null
  height?: number | null
  duration_ms?: number | null
  provenance?: string
  created_at?: string
  last_modified?: string
  version?: number
}

export interface PersonaVisualPack {
  id: string
  persona_id: string
  user_id?: string
  title: string
  renderer_type: PersonaVisualRendererType
  status: PersonaVisualPackStatus
  manifest_version?: number
  manifest: PersonaVisualManifest
  parent_pack_id?: string | null
  revision_number?: number
  provenance?: string
  active_at?: string | null
  assets?: PersonaVisualAsset[]
  assets_by_id?: Record<string, PersonaVisualAsset>
  created_at?: string
  last_modified?: string
  version?: number
}

export interface PersonaVisualPackCreate {
  title: string
  manifest?: Partial<PersonaVisualManifest> | Record<string, unknown>
}

export interface PersonaVisualManifestUpdate {
  manifest: PersonaVisualManifest | Record<string, unknown>
  expected_version?: number | null
}

export interface PersonaVisualPackListResponse {
  packs: PersonaVisualPack[]
  active_pack?: PersonaVisualPack | null
}

export interface PersonaVisualCandidate {
  id: string
  pack_id: string
  persona_id: string
  user_id?: string
  job_id?: string | null
  status: PersonaVisualCandidateStatus
  proposed_manifest_patch?: Record<string, unknown>
  generated_asset_ids?: string[]
  prompt?: string | null
  failure_reason?: string | null
  created_at?: string
  last_modified?: string
  version?: number
}

export interface PersonaVisualCandidateReviewRequest {
  status: "accepted" | "rejected" | "failed"
  failure_reason?: string | null
}

export interface PersonaVisualDeactivateResponse {
  status: "deactivated"
  persona_id: string
}
