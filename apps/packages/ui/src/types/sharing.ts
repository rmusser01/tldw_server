/**
 * Sharing Types
 * Types for workspace sharing, share tokens, and shared-with-me views
 */

// ─────────────────────────────────────────────────────────────────────────────
// Enums
// ─────────────────────────────────────────────────────────────────────────────

export type ShareScopeType = "team" | "org"
export type AccessLevel = "view_chat" | "view_chat_add" | "full_edit"
export type ShareResourceType = "chatbook" | "workspace"

export const ACCESS_LEVEL_LABELS: Record<AccessLevel, string> = {
  view_chat: "Read-only",
  view_chat_add: "Can add sources",
  full_edit: "Full access",
}

export const ACCESS_LEVEL_COLORS: Record<AccessLevel, string> = {
  view_chat: "blue",
  view_chat_add: "green",
  full_edit: "orange",
}

export const getAccessLevelLabel = (accessLevel: string): string =>
  ACCESS_LEVEL_LABELS[accessLevel as AccessLevel] ?? accessLevel

export const getAccessLevelColor = (accessLevel: string): string =>
  ACCESS_LEVEL_COLORS[accessLevel as AccessLevel] ?? "default"

export interface ShareWorkspaceRequest {
  share_scope_type: ShareScopeType
  share_scope_id: number
  access_level: AccessLevel
  allow_clone: boolean
}

export interface ShareResponse {
  id: number
  workspace_id: string
  owner_user_id: number
  share_scope_type: ShareScopeType
  share_scope_id: number
  access_level: AccessLevel
  allow_clone: boolean
  created_by: number
  created_at?: string | null
  updated_at?: string | null
  revoked_at?: string | null
  is_revoked: boolean
}

export interface ShareListResponse {
  shares: ShareResponse[]
  total: number
}

export interface SharedWithMeItem {
  share_id: number
  workspace_id: string
  workspace_name?: string
  owner_user_id: number
  owner_username?: string
  access_level: AccessLevel
  allow_clone: boolean
  shared_at?: string | null
  workspace_description?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface SharedWithMeResponse {
  items: SharedWithMeItem[]
  total: number
}

export interface CreateTokenRequest {
  resource_type: ShareResourceType
  resource_id: string
  access_level: AccessLevel
  allow_clone: boolean
  password?: string
  max_uses?: number
  expires_at?: string
}

export interface TokenResponse {
  id: number
  token_prefix: string
  token?: string
  resource_type: ShareResourceType
  resource_id: string
  access_level: AccessLevel
  allow_clone: boolean
  is_password_protected: boolean
  max_uses?: number | null
  use_count: number
  expires_at?: string | null
  created_at?: string | null
  revoked_at?: string | null
  is_revoked: boolean
  raw_token?: string
}

export interface TokenListResponse {
  tokens: TokenResponse[]
  total: number
}

export interface PublicSharePreview {
  resource_type: ShareResourceType
  resource_name?: string | null
  resource_description?: string | null
  is_password_protected: boolean
  access_level: AccessLevel
  allow_clone: boolean
}
