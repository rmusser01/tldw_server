/**
 * Integrations control-plane API client.
 */

import { bgRequest } from "@/services/background-proxy"
import { toAllowedPath } from "@/services/tldw/path-utils"

export type IntegrationProvider = "slack" | "discord" | "telegram"
export type PersonalIntegrationProvider = "slack" | "discord"
export type IntegrationScope = "personal" | "workspace"
export type IntegrationStatus = "connected" | "disconnected" | "disabled" | "degraded" | "needs_config"
export type IntegrationCommand = "help" | "ask" | "rag" | "summarize" | "status"

export interface IntegrationConnection {
  id: string
  provider: IntegrationProvider
  scope: IntegrationScope
  display_name: string
  status: IntegrationStatus
  enabled: boolean
  connected_at?: string | null
  updated_at?: string | null
  health?: Record<string, unknown> | null
  metadata: Record<string, unknown>
  actions: string[]
}

export interface IntegrationOverviewResponse {
  scope: IntegrationScope
  items: IntegrationConnection[]
}

export interface PersonalIntegrationConnectResponse {
  provider: PersonalIntegrationProvider
  connection_id: string
  status: "ready"
  auth_url: string
  auth_session_id: string
  expires_at: string
}

export interface PersonalIntegrationUpdatePayload {
  enabled: boolean
}

export interface PersonalIntegrationDeleteResponse {
  deleted: boolean
  provider: PersonalIntegrationProvider
  connection_id: string
}

export interface SlackWorkspacePolicy {
  allowed_commands: IntegrationCommand[]
  channel_allowlist: string[]
  channel_denylist: string[]
  default_response_mode: "ephemeral" | "thread" | "channel"
  strict_user_mapping: boolean
  service_user_id: string | null
  user_mappings: Record<string, string>
  workspace_quota_per_minute: number
  user_quota_per_minute: number
  status_scope: "workspace" | "workspace_and_user"
}

export interface SlackWorkspacePolicyUpdate {
  allowed_commands?: IntegrationCommand[]
  channel_allowlist?: string[]
  channel_denylist?: string[]
  default_response_mode?: "ephemeral" | "thread" | "channel"
  strict_user_mapping?: boolean
  service_user_id?: string
  user_mappings?: Record<string, string>
  workspace_quota_per_minute?: number
  user_quota_per_minute?: number
  status_scope?: "workspace" | "workspace_and_user"
}

export interface SlackWorkspacePolicyResponse {
  provider: "slack"
  scope: "workspace"
  installation_ids: string[]
  uniform: boolean
  policy: SlackWorkspacePolicy
}

export interface DiscordWorkspacePolicy {
  allowed_commands: IntegrationCommand[]
  channel_allowlist: string[]
  channel_denylist: string[]
  default_response_mode: "ephemeral" | "channel"
  strict_user_mapping: boolean
  service_user_id: string | null
  user_mappings: Record<string, string>
  guild_quota_per_minute: number
  user_quota_per_minute: number
  status_scope: "guild" | "guild_and_user"
}

export interface DiscordWorkspacePolicyUpdate {
  allowed_commands?: IntegrationCommand[]
  channel_allowlist?: string[]
  channel_denylist?: string[]
  default_response_mode?: "ephemeral" | "channel"
  strict_user_mapping?: boolean
  service_user_id?: string
  user_mappings?: Record<string, string>
  guild_quota_per_minute?: number
  user_quota_per_minute?: number
  status_scope?: "guild" | "guild_and_user"
}

export interface DiscordWorkspacePolicyResponse {
  provider: "discord"
  scope: "workspace"
  installation_ids: string[]
  uniform: boolean
  policy: DiscordWorkspacePolicy
}

export interface TelegramBotConfigResponse {
  ok: boolean
  provider: "telegram"
  scope_type: "org" | "team"
  scope_id: number
  bot_username: string
  enabled: boolean
}

export interface TelegramBotConfigUpdate {
  bot_token: string
  webhook_secret: string
  bot_username?: string | null
  enabled: boolean
}

const assertNoNullPolicyFields = (payload: Record<string, unknown>): void => {
  for (const [key, value] of Object.entries(payload)) {
    if (value === null) {
      throw new Error(`Null policy fields are not supported: ${key}`)
    }
  }
}

const normalizePersonalConnectionId = (
  provider: PersonalIntegrationProvider,
  connectionId: string
): string => {
  const normalized = String(connectionId || "").trim()
  const expected = `personal:${provider}`
  if (!normalized) {
    throw new Error("connectionId is required")
  }
  if (normalized === provider || normalized === expected) {
    return expected
  }
  throw new Error(`connectionId must target ${expected}`)
}

export interface TelegramLinkedActorItem {
  id: number
  scope_type: "org" | "team"
  scope_id: number
  telegram_user_id: number
  auth_user_id: number
  telegram_username?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface TelegramLinkedActorListResponse {
  ok: boolean
  scope_type: "org" | "team"
  scope_id: number
  items: TelegramLinkedActorItem[]
}

export interface TelegramLinkedActorRevokeResponse {
  ok: boolean
  deleted: boolean
  id: number
  scope_type: "org" | "team"
  scope_id: number
}

export interface TelegramPairingCodeResponse {
  ok: boolean
  pairing_code: string
  scope_type: "org" | "team"
  scope_id: number
  expires_at: string
}

export async function listPersonalIntegrations(): Promise<IntegrationOverviewResponse> {
  return await bgRequest<IntegrationOverviewResponse>({
    path: "/api/v1/integrations/personal",
    method: "GET"
  })
}

export async function connectPersonalIntegration(
  provider: PersonalIntegrationProvider
): Promise<PersonalIntegrationConnectResponse> {
  return await bgRequest<PersonalIntegrationConnectResponse>({
    path: `/api/v1/integrations/personal/${provider}/connect`,
    method: "POST"
  })
}

export async function updatePersonalIntegration(
  provider: PersonalIntegrationProvider,
  connectionId: string,
  payload: PersonalIntegrationUpdatePayload
): Promise<IntegrationConnection> {
  if (typeof payload.enabled !== "boolean") {
    throw new Error("enabled must be set explicitly")
  }
  return await bgRequest<IntegrationConnection>({
    path: toAllowedPath(
      `/api/v1/integrations/personal/${provider}/${encodeURIComponent(normalizePersonalConnectionId(provider, connectionId))}`
    ),
    method: "PATCH",
    body: payload
  })
}

export async function deletePersonalIntegration(
  provider: PersonalIntegrationProvider,
  connectionId: string
): Promise<PersonalIntegrationDeleteResponse> {
  return await bgRequest<PersonalIntegrationDeleteResponse>({
    path: toAllowedPath(
      `/api/v1/integrations/personal/${provider}/${encodeURIComponent(normalizePersonalConnectionId(provider, connectionId))}`
    ),
    method: "DELETE"
  })
}

export async function listWorkspaceIntegrations(): Promise<IntegrationOverviewResponse> {
  return await bgRequest<IntegrationOverviewResponse>({
    path: "/api/v1/integrations/workspace",
    method: "GET",
    // Capability probe: absent on servers without the module (#2896).
    expectedStatuses: [404, 405, 410, 501]
  })
}

export async function getWorkspaceSlackPolicy(): Promise<SlackWorkspacePolicyResponse> {
  return await bgRequest<SlackWorkspacePolicyResponse>({
    path: "/api/v1/integrations/workspace/slack/policy",
    method: "GET",
    // Capability probe: absent on servers without the module (#2896).
    expectedStatuses: [404, 405, 410, 501]
  })
}

export async function updateWorkspaceSlackPolicy(
  payload: SlackWorkspacePolicyUpdate
): Promise<SlackWorkspacePolicyResponse> {
  assertNoNullPolicyFields(payload as Record<string, unknown>)
  return await bgRequest<SlackWorkspacePolicyResponse>({
    path: "/api/v1/integrations/workspace/slack/policy",
    method: "PUT",
    body: payload
  })
}

export async function getWorkspaceDiscordPolicy(): Promise<DiscordWorkspacePolicyResponse> {
  return await bgRequest<DiscordWorkspacePolicyResponse>({
    path: "/api/v1/integrations/workspace/discord/policy",
    method: "GET",
    // Capability probe: absent on servers without the module (#2896).
    expectedStatuses: [404, 405, 410, 501]
  })
}

export async function updateWorkspaceDiscordPolicy(
  payload: DiscordWorkspacePolicyUpdate
): Promise<DiscordWorkspacePolicyResponse> {
  assertNoNullPolicyFields(payload as Record<string, unknown>)
  return await bgRequest<DiscordWorkspacePolicyResponse>({
    path: "/api/v1/integrations/workspace/discord/policy",
    method: "PUT",
    body: payload
  })
}

export async function getWorkspaceTelegramBot(): Promise<TelegramBotConfigResponse> {
  return await bgRequest<TelegramBotConfigResponse>({
    path: "/api/v1/integrations/workspace/telegram/bot",
    method: "GET",
    // Capability probe: absent on servers without the module (#2896).
    expectedStatuses: [404, 405, 410, 501]
  })
}

export async function updateWorkspaceTelegramBot(
  payload: TelegramBotConfigUpdate
): Promise<TelegramBotConfigResponse> {
  if (typeof payload.enabled !== "boolean") {
    throw new Error("enabled must be set explicitly")
  }
  return await bgRequest<TelegramBotConfigResponse>({
    path: "/api/v1/integrations/workspace/telegram/bot",
    method: "PUT",
    body: payload
  })
}

export async function createWorkspaceTelegramPairingCode(): Promise<TelegramPairingCodeResponse> {
  return await bgRequest<TelegramPairingCodeResponse>({
    path: "/api/v1/integrations/workspace/telegram/pairing-code",
    method: "POST"
  })
}

export async function listWorkspaceTelegramLinkedActors(): Promise<TelegramLinkedActorListResponse> {
  return await bgRequest<TelegramLinkedActorListResponse>({
    path: "/api/v1/integrations/workspace/telegram/linked-actors",
    method: "GET",
    // Capability probe: absent on servers without the module (#2896).
    expectedStatuses: [404, 405, 410, 501]
  })
}

export async function revokeWorkspaceTelegramLinkedActor(
  actorId: number
): Promise<TelegramLinkedActorRevokeResponse> {
  return await bgRequest<TelegramLinkedActorRevokeResponse>({
    path: toAllowedPath(`/api/v1/integrations/workspace/telegram/linked-actors/${encodeURIComponent(actorId)}`),
    method: "DELETE"
  })
}
