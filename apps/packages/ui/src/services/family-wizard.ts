import { bgRequest } from "@/services/background-proxy"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type HouseholdMode = "family" | "institutional"
export type WizardActivationStatus =
  | "draft"
  | "invites_pending"
  | "partially_active"
  | "active"
  | "needs_attention"
export type RelationshipType = "parent" | "legal_guardian" | "institutional"
export type MemberRole = "guardian" | "dependent" | "caregiver"
export type PlanStatus = "queued" | "active" | "failed"
export type AccountMode = "existing_account" | "invite_new"
export type ProvisioningStatus =
  | "not_started"
  | "invite_ready"
  | "sent"
  | "accepted"
  | "expired"
  | "failed"
export type RelationshipDraftStatus =
  | "pending"
  | "pending_provisioning"
  | "active"
  | "declined"
  | "revoked"
export type InviteStatus =
  | "not_started"
  | "ready"
  | "sent"
  | "accepted"
  | "expired"
  | "revoked"
  | "failed"

export interface HouseholdDraft {
  id: string
  owner_user_id: string
  name: string
  mode: HouseholdMode
  status: WizardActivationStatus
  created_at: string
  updated_at: string
}

export interface CreateHouseholdDraftBody {
  name: string
  mode: HouseholdMode
}

export interface UpdateHouseholdDraftBody {
  name?: string
  mode?: HouseholdMode
  status?: WizardActivationStatus
}

export interface HouseholdMemberDraft {
  id: string
  household_draft_id: string
  role: MemberRole
  display_name: string
  user_id: string | null
  email: string | null
  invite_required: boolean
  account_mode: AccountMode
  provisioning_status: ProvisioningStatus
  metadata: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface AddHouseholdMemberDraftBody {
  role: MemberRole
  display_name: string
  user_id?: string
  email?: string
  invite_required?: boolean
  account_mode?: AccountMode
  provisioning_status?: ProvisioningStatus
  metadata?: Record<string, unknown>
}

export interface RelationshipDraft {
  id: string
  household_draft_id: string
  guardian_member_draft_id: string
  dependent_member_draft_id: string
  relationship_type: RelationshipType
  dependent_visible: boolean
  status: RelationshipDraftStatus
  relationship_id: string | null
  created_at: string
  updated_at: string
}

export interface SaveRelationshipDraftBody {
  guardian_member_draft_id: string
  dependent_member_draft_id: string
  relationship_type?: RelationshipType
  dependent_visible?: boolean
}

export interface GuardrailPlanDraft {
  id: string
  household_draft_id: string
  dependent_member_draft_id: string
  dependent_user_id: string | null
  relationship_draft_id: string
  template_id: string
  overrides: Record<string, unknown>
  status: PlanStatus
  materialized_policy_id: string | null
  failure_reason: string | null
  created_at: string
  updated_at: string
}

export interface SaveGuardrailPlanDraftBody {
  dependent_member_draft_id?: string
  dependent_user_id?: string
  relationship_draft_id: string
  template_id: string
  overrides?: Record<string, unknown>
}

export interface ActivationSummaryItem {
  dependent_member_draft_id: string
  dependent_user_id: string | null
  relationship_status: RelationshipDraftStatus
  plan_status: PlanStatus
  message: string | null
}

export interface ActivationSummary {
  household_draft_id: string
  status: WizardActivationStatus
  active_count: number
  pending_count: number
  failed_count: number
  items: ActivationSummaryItem[]
}

export interface ResendPendingInvitesBody {
  dependent_user_ids: string[]
  member_draft_ids: string[]
}

export interface ResendPendingInvitesResponse {
  household_draft_id: string
  resent_count: number
  skipped_count: number
  resent_user_ids: string[]
  skipped_user_ids: string[]
  resent_member_draft_ids: string[]
  skipped_member_draft_ids: string[]
}

export interface HouseholdMemberInvite {
  id: string
  household_draft_id: string
  member_draft_id: string
  status: InviteStatus
  delivery_channel: string
  delivery_target: string | null
  invite_token: string
  resend_count: number
  last_sent_at: string | null
  accepted_at: string | null
  expires_at: string | null
  revoked_at: string | null
  failure_reason: string | null
  created_at: string
  updated_at: string
}

export interface HouseholdInviteTrackerItem {
  member_draft_id: string
  display_name: string
  account_mode: AccountMode
  dependent_user_id: string | null
  relationship_draft_id: string | null
  relationship_status: RelationshipDraftStatus | null
  plan_draft_id: string | null
  plan_status: PlanStatus | null
  invite_id: string | null
  invite_status: InviteStatus
  invite_delivery_channel: string | null
  invite_delivery_target: string | null
  invite_last_sent_at: string | null
  invite_accepted_at: string | null
  invite_expires_at: string | null
  blocker_codes: string[]
  available_actions: string[]
}

export interface HouseholdInviteTracker {
  household_draft_id: string
  active_count: number
  pending_count: number
  failed_count: number
  items: HouseholdInviteTrackerItem[]
}

export interface HouseholdInvitePreview {
  invite_id: string
  household_draft_id: string
  member_draft_id: string
  household_name: string
  dependent_display_name: string
  invite_status: InviteStatus
  expires_at: string | null
  requires_registration: boolean
}

export interface HouseholdInviteAcceptRegisterBody {
  token: string
  username: string
  email: string
  password: string
}

export interface HouseholdInviteAcceptClaimBody {
  token: string
}

export interface HouseholdInviteAcceptResponse {
  household_draft_id: string
  member_draft_id: string
  invite_id: string
  user_id: string
  relationship_id: string | null
  materialized_plan_count: number
  was_existing_user: boolean
}

export interface HouseholdDraftSnapshot {
  household: HouseholdDraft
  members: HouseholdMemberDraft[]
  relationships: RelationshipDraft[]
  plans: GuardrailPlanDraft[]
}

const getInvitePreviewPath = (token: string): AllowedPath => {
  const query = new URLSearchParams({ token }).toString()
  return appendPathQuery(toAllowedPath("/api/v1/guardian/wizard/invites/preview"), `?${query}`)
}

export async function createHouseholdDraft(
  body: CreateHouseholdDraftBody
): Promise<HouseholdDraft> {
  return bgRequest<HouseholdDraft>({
    path: toAllowedPath("/api/v1/guardian/wizard/drafts"),
    method: "POST",
    body
  })
}

export async function listHouseholdDrafts(): Promise<HouseholdDraft[]> {
  return bgRequest<HouseholdDraft[]>({
    path: toAllowedPath("/api/v1/guardian/wizard/drafts"),
    method: "GET"
  })
}

export const getHouseholdDrafts = listHouseholdDrafts

export async function getHouseholdDraft(draftId: string): Promise<HouseholdDraft> {
  return bgRequest<HouseholdDraft>({
    path: toAllowedPath(`/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}`),
    method: "GET"
  })
}

export async function getLatestHouseholdDraft(): Promise<HouseholdDraft | null> {
  try {
    return await bgRequest<HouseholdDraft>({
      path: toAllowedPath("/api/v1/guardian/wizard/drafts/latest"),
      method: "GET"
    })
  } catch (error) {
    const status = (error as { status?: number } | null)?.status
    if (status === 404) return null
    throw error
  }
}

export async function getHouseholdDraftSnapshot(
  draftId: string
): Promise<HouseholdDraftSnapshot> {
  return bgRequest<HouseholdDraftSnapshot>({
    path: toAllowedPath(`/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/snapshot`),
    method: "GET"
  })
}

export async function updateHouseholdDraft(
  draftId: string,
  body: UpdateHouseholdDraftBody
): Promise<HouseholdDraft> {
  return bgRequest<HouseholdDraft>({
    path: toAllowedPath(`/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}`),
    method: "PATCH",
    body
  })
}

export async function addHouseholdMemberDraft(
  draftId: string,
  body: AddHouseholdMemberDraftBody
): Promise<HouseholdMemberDraft> {
  return bgRequest<HouseholdMemberDraft>({
    path: toAllowedPath(`/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/members`),
    method: "POST",
    body
  })
}

export async function removeHouseholdMemberDraft(
  draftId: string,
  memberId: string
): Promise<{ detail: string }> {
  return bgRequest<{ detail: string }>({
    path: toAllowedPath(
      `/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/members/${encodeURIComponent(memberId)}`
    ),
    method: "DELETE"
  })
}

export async function provisionHouseholdMemberInvite(
  draftId: string,
  memberId: string
): Promise<HouseholdMemberInvite> {
  return bgRequest<HouseholdMemberInvite>({
    path: toAllowedPath(
      `/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/members/${encodeURIComponent(memberId)}/invite/provision`
    ),
    method: "POST"
  })
}

export async function saveRelationshipDraft(
  draftId: string,
  body: SaveRelationshipDraftBody
): Promise<RelationshipDraft> {
  return bgRequest<RelationshipDraft>({
    path: toAllowedPath(`/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/relationships`),
    method: "POST",
    body
  })
}

export async function saveGuardrailPlanDraft(
  draftId: string,
  body: SaveGuardrailPlanDraftBody
): Promise<GuardrailPlanDraft> {
  return bgRequest<GuardrailPlanDraft>({
    path: toAllowedPath(`/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/plans`),
    method: "POST",
    body
  })
}

export async function getHouseholdInviteTracker(
  draftId: string
): Promise<HouseholdInviteTracker> {
  return bgRequest<HouseholdInviteTracker>({
    path: toAllowedPath(`/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/tracker`),
    method: "GET"
  })
}

export async function getActivationSummary(
  draftId: string
): Promise<ActivationSummary> {
  return bgRequest<ActivationSummary>({
    path: toAllowedPath(
      `/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/activation-summary`
    ),
    method: "GET"
  })
}

export async function resendHouseholdMemberInvite(
  draftId: string,
  inviteId: string
): Promise<HouseholdMemberInvite> {
  return bgRequest<HouseholdMemberInvite>({
    path: toAllowedPath(
      `/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/invites/${encodeURIComponent(inviteId)}/resend`
    ),
    method: "POST"
  })
}

export async function reissueHouseholdMemberInvite(
  draftId: string,
  inviteId: string
): Promise<HouseholdMemberInvite> {
  return bgRequest<HouseholdMemberInvite>({
    path: toAllowedPath(
      `/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/invites/${encodeURIComponent(inviteId)}/reissue`
    ),
    method: "POST"
  })
}

export async function resendPendingInvites(
  draftId: string,
  body: ResendPendingInvitesBody
): Promise<ResendPendingInvitesResponse> {
  return bgRequest<ResendPendingInvitesResponse>({
    path: toAllowedPath(
      `/api/v1/guardian/wizard/drafts/${encodeURIComponent(draftId)}/invites/resend`
    ),
    method: "POST",
    body
  })
}

export async function getHouseholdInvitePreview(
  token: string
): Promise<HouseholdInvitePreview> {
  return bgRequest<HouseholdInvitePreview>({
    path: getInvitePreviewPath(token),
    method: "GET"
  })
}

export async function acceptHouseholdInviteRegister(
  body: HouseholdInviteAcceptRegisterBody
): Promise<HouseholdInviteAcceptResponse> {
  return bgRequest<HouseholdInviteAcceptResponse>({
    path: toAllowedPath("/api/v1/guardian/wizard/invites/accept/register"),
    method: "POST",
    body
  })
}

export async function acceptHouseholdInviteClaim(
  body: HouseholdInviteAcceptClaimBody
): Promise<HouseholdInviteAcceptResponse> {
  return bgRequest<HouseholdInviteAcceptResponse>({
    path: toAllowedPath("/api/v1/guardian/wizard/invites/accept/claim"),
    method: "POST",
    body
  })
}
