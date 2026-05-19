'use client';

import { ApiError, requestJson, requestText } from './http';
import { normalizeListResponse, normalizePagedResponse } from './normalize';
import type {
  ApiKey,
  ApiKeyMutationResponse,
  ApiKeyUsageSummary,
  ApiKeyUsageTopResponse,
  AuditLog,
  BackupScheduleListResponse,
  BackupScheduleMutationResponse,
  BackupsResponse,
  BillingAnalytics,
  ByokValidationRunCreateRequest,
  ByokValidationRunItem,
  ByokValidationRunListResponse,
  CompliancePosture,
  ComplianceReportSchedule,
  DigestPreference,
  EffectivePermissionsResponse,
  EmailDeliveryListResponse,
  FeatureRegistryEntry,
  IncidentItem,
  IncidentNotifyResponse,
  IncidentsResponse,
  Invoice,
  MaintenanceRotationRunCreateRequest,
  MaintenanceRotationRunCreateResponse,
  MaintenanceRotationRunItem,
  MaintenanceRotationRunListResponse,
  OrgMember,
  Organization,
  OrgMembership,
  OrgUsageSummary,
  Plan,
  ProviderSecret,
  RegistrationCode,
  RetentionPoliciesResponse,
  RetentionPolicyPreviewResponse,
  SecurityAlertStatus,
  SecurityHealthData,
  Subscription,
  DependencyUptimeStats,
  SystemDependenciesResponse,
  Team,
  TeamMembership,
  User,
  UserWithKeyCount,
  VoiceAnalyticsSummary,
  VoiceCommand,
  VoiceCommandListResponse,
  VoiceCommandUsage,
  VoiceCommandValidationResponse,
  VoiceSession,
  VoiceSessionListResponse,
  WatchlistSettings,
  WebhookCreateResponse,
  WebhookDeliveryItem,
  WebhookDeliveryListResponse,
  WebhookItem,
  WebhookListResponse,
  AdminWebhook,
  AdminWebhooksResponse,
  AdminWebhookDeliveryLogResponse,
  AdminWebhookTestResult,
} from '@/types';
export { ApiError };

type QueryParamPrimitive = string | number | boolean;
type QueryParamValue = QueryParamPrimitive | QueryParamPrimitive[] | null | undefined;

type CreatePlanInput = {
  name: string;
  tier: string;
  monthly_price_cents: number;
  included_token_credits: number;
  overage_rate_per_1k_tokens_cents: number;
  stripe_product_id?: string;
  stripe_price_id?: string;
  features?: string[];
  is_default?: boolean;
};

type UpdatePlanInput = Partial<CreatePlanInput>;

type AddTeamMemberInput =
  | { email: string; role?: string }
  | { userId: string | number; role?: string }
  | { user_id: number; role?: string };

function normalizeTeamMemberInput(member: AddTeamMemberInput): Record<string, unknown> {
  if ('email' in member) {
    return { email: member.email, role: member.role };
  }
  if ('user_id' in member) {
    return { user_id: member.user_id, role: member.role };
  }
  if ('userId' in member) {
    const userId = typeof member.userId === 'number'
      ? member.userId
      : Number(member.userId);
    if (!Number.isFinite(userId)) {
      throw new Error('Invalid userId');
    }
    return { user_id: userId, role: member.role };
  }
  throw new Error('Invalid team member payload');
}

function buildQueryString(params?: Record<string, QueryParamValue>): string {
  if (!params) return '';
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === null || value === undefined) return;
    if (Array.isArray(value)) {
      value.forEach((entry) => {
        query.append(key, String(entry));
      });
      return;
    }
    query.append(key, String(value));
  });
  return query.toString();
}

function requestRouterAnalytics(path: string, params?: Record<string, string>) {
  const queryParams = params ? new URLSearchParams(params).toString() : '';
  return requestJson(`/admin/router-analytics/${path}${queryParams ? `?${queryParams}` : ''}`);
}

export async function getTeam(teamId: string) {
  return await requestJson(`/admin/teams/${encodeURIComponent(teamId)}`);
}

export async function getTeamMembers(teamId: string) {
  return await requestJson(`/admin/teams/${encodeURIComponent(teamId)}/members`);
}

export async function addTeamMember(teamId: string, member: AddTeamMemberInput) {
  const payload = normalizeTeamMemberInput(member);
  return await requestJson(`/admin/teams/${encodeURIComponent(teamId)}/members`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function removeTeamMember(teamId: string, memberId: string | number) {
  const memberValue = String(memberId).trim();
  if (!memberValue) {
    throw new Error('memberId is required');
  }
  return await requestJson(`/admin/teams/${encodeURIComponent(teamId)}/members/${encodeURIComponent(memberValue)}`, {
    method: 'DELETE',
  });
}

/**
 * API client for tldw_server admin operations
 */
export const api = {
  // ============================================
  // Dashboard & Stats
  // ============================================
  getDashboardStats: () => requestJson('/admin/stats'),
  getRealtimeStats: () =>
    requestJson<{
      active_sessions: number;
      tokens_today: { prompt: number; completion: number; total: number };
    }>('/admin/stats/realtime'),
  getDashboardActivity: (days = 7, params?: { granularity?: 'hour' | 'day' }) => {
    const query = new URLSearchParams({ days: String(days) });
    if (params?.granularity) {
      query.set('granularity', params.granularity);
    }
    return requestJson(`/admin/activity?${query.toString()}`);
  },

  // ============================================
  // User Management
  // ============================================
  getUsersPage: async (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    const response = await requestJson(`/admin/users${queryParams ? `?${queryParams}` : ''}`);
    const { items, total, limit } = normalizePagedResponse<UserWithKeyCount>(response, ['users', 'items']);
    const record = response && typeof response === 'object'
      ? (response as Record<string, unknown>)
      : {};
    const page = typeof record.page === 'number' ? record.page : 1;
    const pages = typeof record.pages === 'number'
      ? record.pages
      : (typeof limit === 'number' && limit > 0 ? Math.ceil(total / limit) : 1);
    return {
      items,
      total,
      page,
      limit: limit ?? items.length,
      pages,
    };
  },
  getUsers: async (params?: Record<string, string>) => {
    const response = await api.getUsersPage(params);
    return response.items;
  },
  createUser: (data: Record<string, unknown>) => requestJson('/admin/users', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  getUser: (userId: string) => requestJson<User>(`/admin/users/${userId}`),
  getUserOrgMemberships: (userId: string) => requestJson<OrgMembership[]>(`/admin/users/${userId}/org-memberships`),
  getUserTeamMemberships: (userId: string) => requestJson<TeamMembership[]>(`/admin/users/${userId}/team-memberships`),
  getUserEffectivePermissions: (userId: string) =>
    requestJson<EffectivePermissionsResponse>(`/admin/users/${userId}/effective-permissions`),
  getUserSessions: (userId: string) => requestJson(`/admin/users/${userId}/sessions`),
  revokeUserSession: (userId: string, sessionId: string, data: { reason: string; admin_password?: string | null }) =>
    requestJson(`/admin/users/${userId}/sessions/${sessionId}`, {
      method: 'DELETE',
      body: JSON.stringify(data),
    }),
  revokeAllUserSessions: (userId: string, data: { reason: string; admin_password?: string | null }) =>
    requestJson(`/admin/users/${userId}/sessions/revoke-all`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  getUserMfaStatus: (userId: string) => requestJson(`/admin/users/${userId}/mfa`),
  getUserMfaStatusBulk: (userIds: number[]) =>
    requestJson<{ mfa_status: Record<string, boolean>; failed_user_ids: number[] }>(`/admin/users/mfa/bulk?ids=${userIds.join(',')}`),
  disableUserMfa: (userId: string, data: { reason: string; admin_password?: string | null }) =>
    requestJson(`/admin/users/${userId}/mfa/disable`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  setUserMfaRequirement: (userId: string, data: { require_mfa: boolean; reason: string; admin_password?: string | null }) =>
    requestJson(`/admin/users/${userId}/mfa/require`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  updateUser: (userId: string, data: Record<string, unknown>) => requestJson(`/admin/users/${userId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  resetUserPassword: (
    userId: string,
    data: { temporary_password: string; force_password_change?: boolean; reason: string; admin_password?: string | null }
  ) => requestJson(`/admin/users/${userId}/reset-password`, {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  deleteUser: (userId: string, data: { reason: string; admin_password?: string | null }) => requestJson(`/admin/users/${userId}`, {
    method: 'DELETE',
    body: JSON.stringify(data),
  }),
  inviteUser: (data: { email: string; role: string; expiry_days?: number }) => requestJson('/admin/users/invite', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  getInvitations: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/users/invitations${queryParams ? `?${queryParams}` : ''}`);
  },
  revokeInvitation: (invitationId: string) => requestJson(`/admin/users/invitations/${encodeURIComponent(invitationId)}`, {
    method: 'DELETE',
  }),
  resendInvitation: (invitationId: string) => requestJson(`/admin/users/invitations/${encodeURIComponent(invitationId)}/resend`, {
    method: 'POST',
  }),
  getCurrentUser: () => requestJson<User>('/users/me'),

  // ============================================
  // Registration Codes
  // ============================================
  getRegistrationSettings: () => requestJson('/admin/registration-settings'),
  updateRegistrationSettings: (data: Record<string, unknown>) => requestJson('/admin/registration-settings', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  getRegistrationCodes: async (includeExpired: boolean = false) => {
    const response = await requestJson(`/admin/registration-codes?include_expired=${includeExpired}`);
    return normalizeListResponse<RegistrationCode>(response, ['codes', 'items']);
  },
  createRegistrationCode: (data: Record<string, unknown>) => requestJson('/admin/registration-codes', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  deleteRegistrationCode: (codeId: number | string) => requestJson(`/admin/registration-codes/${codeId}`, {
    method: 'DELETE',
  }),

  // ============================================
  // API Key Management
  // ============================================
  getUserApiKeys: (
    userId: string,
    params?: {
      include_revoked?: boolean;
    }
  ) => {
    const query = new URLSearchParams();
    if (params?.include_revoked !== undefined) {
      query.set('include_revoked', String(params.include_revoked));
    }
    const suffix = query.toString() ? `?${query.toString()}` : '';
    return requestJson<ApiKey[]>(`/admin/users/${userId}/api-keys${suffix}`);
  },
  createApiKey: (userId: string, data: Record<string, unknown>) => requestJson<ApiKeyMutationResponse>(`/admin/users/${userId}/api-keys`, {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  rotateApiKey: (userId: string, keyId: string) => requestJson<ApiKeyMutationResponse>(`/admin/users/${userId}/api-keys/${keyId}/rotate`, {
    method: 'POST',
  }),
  revokeApiKey: (userId: string, keyId: string) => requestJson(`/admin/users/${userId}/api-keys/${keyId}`, {
    method: 'DELETE',
  }),
  getApiKeyAuditLog: (keyId: string) => requestJson(`/admin/api-keys/${keyId}/audit-log`),
  getApiKeyUsage: (keyId: string) => requestJson<ApiKeyUsageSummary>(`/admin/api-keys/${keyId}/usage`),
  getTopApiKeyUsage: (limit: number = 10) => requestJson<ApiKeyUsageTopResponse>(`/admin/api-keys/usage/top?limit=${limit}`),

  // ============================================
  // Organizations
  // ============================================
  getOrganizations: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/orgs${queryParams ? `?${queryParams}` : ''}`);
  },
  getOrganization: (orgId: string) => requestJson<Organization>(`/orgs/${orgId}`),
  createOrganization: (data: Record<string, unknown>) => requestJson('/admin/orgs', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  updateOrganization: (orgId: string, data: Record<string, unknown>) => requestJson(`/orgs/${orgId}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  }),
  deleteOrganization: (orgId: string) => requestJson(`/orgs/${orgId}`, {
    method: 'DELETE',
  }),
  getOrgMembers: (orgId: string) => requestJson<OrgMember[]>(`/admin/orgs/${orgId}/members`),
  addOrgMember: (orgId: string, data: Record<string, unknown>) => requestJson(`/admin/orgs/${orgId}/members`, {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  removeOrgMember: (orgId: string, userId: string) => requestJson(`/admin/orgs/${orgId}/members/${userId}`, {
    method: 'DELETE',
  }),
  updateOrgMemberRole: (orgId: string, userId: string, data: Record<string, unknown>) => requestJson(`/admin/orgs/${orgId}/members/${userId}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  }),
  createOrgInvite: (orgId: string, data: Record<string, unknown>) => requestJson(`/orgs/${orgId}/invite`, {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  getOrgInvites: (orgId: string, params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/orgs/${encodeURIComponent(orgId)}/invites${queryParams ? `?${queryParams}` : ''}`);
  },
  revokeOrgInvite: (orgId: string, inviteId: string) =>
    requestJson(`/orgs/${encodeURIComponent(orgId)}/invites/${encodeURIComponent(inviteId)}`, {
      method: 'DELETE',
    }),

  // ============================================
  // Teams
  // ============================================
  getTeam,
  getTeams: (orgId: string) => requestJson<Team[]>(`/admin/orgs/${orgId}/teams`),
  createTeam: (orgId: string, data: Record<string, unknown>) => requestJson(`/admin/orgs/${orgId}/teams`, {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  updateTeam: (orgId: string, teamId: string, data: Record<string, unknown>) =>
    requestJson(`/orgs/${orgId}/teams/${teamId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),
  deleteTeam: (orgId: string, teamId: string) =>
    requestJson(`/orgs/${orgId}/teams/${teamId}`, {
      method: 'DELETE',
    }),
  getTeamMembers,
  addTeamMember,
  updateTeamMemberRole: (teamId: string, memberId: string | number, data: Record<string, unknown>) =>
    requestJson(`/admin/teams/${encodeURIComponent(teamId)}/members/${encodeURIComponent(String(memberId))}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),
  removeTeamMember,

  // ============================================
  // Roles & Permissions (RBAC)
  // ============================================
  getRoles: () => requestJson('/admin/roles'),
  getRole: (roleId: string) => requestJson(`/admin/roles/${roleId}`),
  createRole: (data: Record<string, unknown>) => requestJson('/admin/roles', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  updateRole: (roleId: string, data: Record<string, unknown>) => requestJson(`/admin/roles/${roleId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  deleteRole: (roleId: string) => requestJson(`/admin/roles/${roleId}`, {
    method: 'DELETE',
  }),
  getRolePermissions: (roleId: string) => requestJson(`/admin/roles/${roleId}/permissions`),
  assignPermissionToRole: (roleId: string, permissionId: string) => requestJson(`/admin/roles/${roleId}/permissions/${permissionId}`, {
    method: 'POST',
  }),
  removePermissionFromRole: (roleId: string, permissionId: string) => requestJson(`/admin/roles/${roleId}/permissions/${permissionId}`, {
    method: 'DELETE',
  }),
  getRoleUsers: (roleId: string) => requestJson(`/admin/roles/${roleId}/users`),
  getPermissions: () => requestJson('/admin/permissions'),
  getPermission: (permId: string) => requestJson(`/admin/permissions/${permId}`),
  createPermission: (data: Record<string, unknown>) => requestJson('/admin/permissions', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  updatePermission: (permId: string, data: Record<string, unknown>) => requestJson(`/admin/permissions/${permId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  deletePermission: (permId: string) => requestJson(`/admin/permissions/${permId}`, {
    method: 'DELETE',
  }),
  // Tool permissions
  getToolPermissions: () => requestJson('/admin/tool-permissions'),
  assignToolPermission: (data: Record<string, unknown>) => requestJson('/admin/tool-permissions', {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  // ============================================
  // Provider Secrets (BYOK)
  // ============================================
  getUserByokKeys: (userId: string) => requestJson(`/admin/users/${userId}/byok-keys`),
  getAdminUserByokKeys: (userId: string) => requestJson(`/admin/keys/users/${userId}`),
  createUserByokKey: (userId: string, data: Record<string, unknown>) => requestJson(`/admin/users/${userId}/byok-keys`, {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  deleteUserByokKey: (userId: string, provider: string) => requestJson(`/admin/users/${userId}/byok-keys/${provider}`, {
    method: 'DELETE',
  }),
  getOrgByokKeys: (orgId: string) => requestJson<ProviderSecret[]>(`/admin/orgs/${orgId}/byok-keys`),
  createOrgByokKey: (orgId: string, data: Record<string, unknown>) => requestJson(`/admin/orgs/${orgId}/byok-keys`, {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  deleteOrgByokKey: (orgId: string, provider: string) => requestJson(`/admin/orgs/${orgId}/byok-keys/${provider}`, {
    method: 'DELETE',
  }),
  getOpenAIOAuthStatus: () => requestJson('/users/keys/openai/oauth/status'),
  startOpenAIOAuth: (data?: { credential_fields?: Record<string, unknown>; return_path?: string }) =>
    requestJson('/users/keys/openai/oauth/authorize', {
      method: 'POST',
      body: JSON.stringify(data ?? {}),
    }),
  refreshOpenAIOAuth: () => requestJson('/users/keys/openai/oauth/refresh', {
    method: 'POST',
  }),
  disconnectOpenAIOAuth: () => requestJson('/users/keys/openai/oauth', {
    method: 'DELETE',
  }),
  switchOpenAICredentialSource: (authSource: 'api_key' | 'oauth') =>
    requestJson('/users/keys/openai/source', {
      method: 'POST',
      body: JSON.stringify({ auth_source: authSource }),
    }),

  // ============================================
  // Budgets
  // ============================================
  getBudgets: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/budgets${queryParams ? `?${queryParams}` : ''}`);
  },
  updateBudget: async (orgId: string, data: Record<string, unknown>) => {
    const normalizedOrgId = Number(orgId);
    if (!Number.isInteger(normalizedOrgId) || normalizedOrgId <= 0) {
      throw new Error('Invalid organization ID');
    }
    try {
      return await requestJson(`/admin/budgets/${encodeURIComponent(String(normalizedOrgId))}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      });
    } catch (error: unknown) {
      if (error instanceof ApiError && [404, 405].includes(error.status)) {
        return requestJson('/admin/budgets', {
          method: 'POST',
          body: JSON.stringify({
            org_id: normalizedOrgId,
            ...data,
          }),
        });
      }
      throw error;
    }
  },

  // ============================================
  // Data Ops
  // ============================================
  getBackups: (params?: Record<string, string>, options?: RequestInit) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson<BackupsResponse>(`/admin/backups${queryParams ? `?${queryParams}` : ''}`, options);
  },
  createBackup: (data: Record<string, unknown>) => requestJson('/admin/backups', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  restoreBackup: (backupId: string, data: Record<string, unknown>) =>
    requestJson(`/admin/backups/${encodeURIComponent(backupId)}/restore`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  listBackupSchedules: (params?: Record<string, string>, options?: RequestInit) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson<BackupScheduleListResponse>(
      `/admin/backup-schedules${queryParams ? `?${queryParams}` : ''}`,
      options
    );
  },
  createBackupSchedule: (data: Record<string, unknown>) =>
    requestJson<BackupScheduleMutationResponse>('/admin/backup-schedules', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  updateBackupSchedule: (scheduleId: string, data: Record<string, unknown>) =>
    requestJson<BackupScheduleMutationResponse>(`/admin/backup-schedules/${encodeURIComponent(scheduleId)}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),
  pauseBackupSchedule: (scheduleId: string) =>
    requestJson<BackupScheduleMutationResponse>(`/admin/backup-schedules/${encodeURIComponent(scheduleId)}/pause`, {
      method: 'POST',
    }),
  resumeBackupSchedule: (scheduleId: string) =>
    requestJson<BackupScheduleMutationResponse>(`/admin/backup-schedules/${encodeURIComponent(scheduleId)}/resume`, {
      method: 'POST',
    }),
  deleteBackupSchedule: (scheduleId: string) =>
    requestJson<BackupScheduleMutationResponse>(`/admin/backup-schedules/${encodeURIComponent(scheduleId)}`, {
      method: 'DELETE',
    }),
  getRetentionPolicies: () => requestJson<RetentionPoliciesResponse>('/admin/retention-policies'),
  previewRetentionPolicyImpact: (policyKey: string, data: Record<string, unknown>) =>
    requestJson<RetentionPolicyPreviewResponse>(`/admin/retention-policies/${encodeURIComponent(policyKey)}/preview`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  updateRetentionPolicy: (policyKey: string, data: Record<string, unknown>) =>
    requestJson(`/admin/retention-policies/${encodeURIComponent(policyKey)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  previewDataSubjectRequest: (data: Record<string, unknown>) =>
    requestJson('/admin/data-subject-requests/preview', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  listDataSubjectRequests: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/data-subject-requests${queryParams ? `?${queryParams}` : ''}`);
  },
  createDataSubjectRequest: (data: Record<string, unknown>) =>
    requestJson('/admin/data-subject-requests', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // ============================================
  // System Ops
  // ============================================
  getSystemLogs: (params?: Record<string, string>, options?: RequestInit) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/system/logs${queryParams ? `?${queryParams}` : ''}`, options);
  },
  getMaintenanceMode: (options?: RequestInit) => requestJson('/admin/maintenance', options),
  updateMaintenanceMode: (data: Record<string, unknown>) => requestJson('/admin/maintenance', {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  getFeatureFlags: (params?: Record<string, string>, options?: RequestInit) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/feature-flags${queryParams ? `?${queryParams}` : ''}`, options);
  },
  upsertFeatureFlag: (flagKey: string, data: Record<string, unknown>) =>
    requestJson(`/admin/feature-flags/${encodeURIComponent(flagKey)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  deleteFeatureFlag: (flagKey: string, params: Record<string, string>) => {
    const queryParams = new URLSearchParams(params).toString();
    return requestJson(`/admin/feature-flags/${encodeURIComponent(flagKey)}?${queryParams}`, {
      method: 'DELETE',
    });
  },
  getIncidents: (params?: Record<string, string>, options?: RequestInit) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson<IncidentsResponse>(
      `/admin/incidents${queryParams ? `?${queryParams}` : ''}`,
      options
    );
  },
  createIncident: (data: Record<string, unknown>) => requestJson<IncidentItem>('/admin/incidents', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  updateIncident: (incidentId: string, data: Record<string, unknown>) =>
    requestJson<IncidentItem>(`/admin/incidents/${encodeURIComponent(incidentId)}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),
  addIncidentEvent: (incidentId: string, data: Record<string, unknown>) =>
    requestJson<IncidentItem>(`/admin/incidents/${encodeURIComponent(incidentId)}/events`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  deleteIncident: (incidentId: string) =>
    requestJson(`/admin/incidents/${encodeURIComponent(incidentId)}`, {
      method: 'DELETE',
    }),
  notifyIncidentStakeholders: (incidentId: string, data: { recipients: string[]; message?: string }) =>
    requestJson<IncidentNotifyResponse>(`/admin/incidents/${encodeURIComponent(incidentId)}/notify`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  notifyIncident: (incidentId: string) =>
    requestJson<{ notified: boolean; incident_id: string; webhooks_delivered: number }>(
      `/admin/incidents/${encodeURIComponent(incidentId)}/notify`,
      { method: 'POST' },
    ),
  getIncidentSlaMetrics: () =>
    requestJson<{
      total_incidents: number;
      resolved_count: number;
      acknowledged_count: number;
      avg_mtta_minutes: number | null;
      avg_mttr_minutes: number | null;
      p95_mtta_minutes: number | null;
      p95_mttr_minutes: number | null;
    }>('/admin/incidents/metrics/sla'),

  // ============================================
  // Email Delivery Log
  // ============================================
  getEmailDeliveries: (params?: { limit?: number; offset?: number; status?: string }) => {
    const qs = buildQueryString(params);
    return requestJson<EmailDeliveryListResponse>(`/admin/email/deliveries${qs ? `?${qs}` : ''}`);
  },

  // ============================================
  // Audit Logs
  // ============================================
  getAuditLogs: async (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    const response = await requestJson(`/admin/audit-log${queryParams ? `?${queryParams}` : ''}`);
    const { items, total, limit, offset } = normalizePagedResponse(
      response,
      ['entries', 'items']
    );
    const mapped: AuditLog[] = items.map((entry) => {
      const record = entry as Record<string, unknown>;
      const rawDetails = record.details;
      let details: Record<string, unknown> | undefined;
      if (rawDetails && typeof rawDetails === 'string') {
        try {
          const parsed = JSON.parse(rawDetails);
          details = (parsed && typeof parsed === 'object')
            ? (parsed as Record<string, unknown>)
            : { value: parsed };
        } catch {
          details = { value: rawDetails };
        }
      } else if (rawDetails && typeof rawDetails === 'object') {
        details = rawDetails as Record<string, unknown>;
      }
      return {
        id: String(record.id ?? ''),
        timestamp: (record.timestamp ?? record.created_at ?? '') as string,
        user_id: Number(record.user_id ?? 0),
        action: String(record.action ?? ''),
        resource: String(record.resource ?? record.resource_type ?? ''),
        details,
        ip_address: record.ip_address ? String(record.ip_address) : undefined,
        username: record.username ? String(record.username) : undefined,
        request_id: record.request_id ? String(record.request_id)
          : record.context_request_id ? String(record.context_request_id)
          : undefined,
        raw: record,
      };
    });
    return { entries: mapped, total, limit, offset };
  },

  // ============================================
  // Error Breakdown (10.4)
  // ============================================
  getErrorBreakdown: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/errors/breakdown${queryParams ? `?${queryParams}` : ''}`);
  },

  // ============================================
  // Rate Limit Summary (10.5)
  // ============================================
  getRateLimitSummary: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/rate-limits/summary${queryParams ? `?${queryParams}` : ''}`);
  },

  // ============================================
  // Configuration
  // ============================================
  getSetupStatus: () => requestJson('/setup/status'),
  getConfig: () => requestJson('/setup/config'),
  updateConfig: (data: Record<string, unknown>) => requestJson('/setup/config', {
    method: 'POST',
    body: JSON.stringify(data),
  }),

  // ============================================
  // Config Profiles & Editing
  // ============================================
  getEffectiveConfig: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/config/effective${queryParams ? `?${queryParams}` : ''}`);
  },
  getConfigProfiles: () => requestJson('/admin/config/profiles'),
  snapshotConfigProfile: (data: { name: string; description?: string }) =>
    requestJson('/admin/config/profiles/snapshot', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  getConfigProfile: (name: string) =>
    requestJson(`/admin/config/profiles/${encodeURIComponent(name)}`),
  restoreConfigProfile: (name: string) =>
    requestJson(`/admin/config/profiles/${encodeURIComponent(name)}/restore`, {
      method: 'POST',
    }),
  deleteConfigProfile: (name: string) =>
    requestJson(`/admin/config/profiles/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),
  updateConfigSection: (section: string, values: Record<string, string>) =>
    requestJson(`/admin/config/sections/${encodeURIComponent(section)}`, {
      method: 'PUT',
      body: JSON.stringify({ values }),
    }),
  exportConfig: () => requestJson('/admin/config/export'),
  importConfig: (sections: Record<string, Record<string, string>>) =>
    requestJson('/admin/config/import', {
      method: 'POST',
      body: JSON.stringify({ sections }),
    }),

  // ============================================
  // LLM Providers
  // ============================================
  getLLMProviders: () => requestJson('/llm/providers'),
  getLLMProviderOverrides: () => requestJson('/admin/llm/providers/overrides'),
  getLLMProviderOverride: (provider: string) => requestJson(`/admin/llm/providers/overrides/${encodeURIComponent(provider)}`),
  updateLLMProviderOverride: (provider: string, data: Record<string, unknown>) => requestJson(`/admin/llm/providers/overrides/${encodeURIComponent(provider)}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  deleteLLMProviderOverride: (provider: string) => requestJson(`/admin/llm/providers/overrides/${encodeURIComponent(provider)}`, {
    method: 'DELETE',
  }),
  testLLMProvider: (data: Record<string, unknown>) => requestJson('/admin/llm/providers/test', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  getLLMProvidersHealth: () => requestJson('/admin/llm/providers/health'),

  // ============================================
  // Monitoring
  // ============================================
  getWatchlists: () => requestJson('/monitoring/watchlists'),
  createWatchlist: <T extends object>(data: T) => requestJson('/monitoring/watchlists', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  updateWatchlist: <T extends object>(watchlistId: string, data: T) => requestJson(`/monitoring/watchlists/${watchlistId}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  deleteWatchlist: (watchlistId: string) => requestJson(`/monitoring/watchlists/${watchlistId}`, {
    method: 'DELETE',
  }),
  getAlerts: () => requestJson('/monitoring/alerts'),
  getAdminAlertHistory: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/monitoring/alerts/history${queryParams ? `?${queryParams}` : ''}`);
  },
  getAdminAlertRules: () => requestJson('/admin/monitoring/alert-rules'),
  createAdminAlertRule: (data: Record<string, unknown>) => requestJson('/admin/monitoring/alert-rules', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  deleteAdminAlertRule: (ruleId: string) => requestJson(`/admin/monitoring/alert-rules/${encodeURIComponent(ruleId)}`, {
    method: 'DELETE',
  }),
  acknowledgeAlert: (alertId: string) => requestJson(`/monitoring/alerts/${alertId}/acknowledge`, {
    method: 'POST',
  }),
  dismissAlert: (alertId: string) => requestJson(`/monitoring/alerts/${alertId}`, {
    method: 'DELETE',
  }),
  assignAdminAlert: (alertIdentity: string, data: Record<string, unknown>) => requestJson(
    `/admin/monitoring/alerts/${encodeURIComponent(alertIdentity)}/assign`,
    {
      method: 'POST',
      body: JSON.stringify(data),
    }
  ),
  snoozeAdminAlert: (alertIdentity: string, data: Record<string, unknown>) => requestJson(
    `/admin/monitoring/alerts/${encodeURIComponent(alertIdentity)}/snooze`,
    {
      method: 'POST',
      body: JSON.stringify(data),
    }
  ),
  escalateAdminAlert: (alertIdentity: string, data: Record<string, unknown>) => requestJson(
    `/admin/monitoring/alerts/${encodeURIComponent(alertIdentity)}/escalate`,
    {
      method: 'POST',
      body: JSON.stringify(data),
    }
  ),
  getMonitoringMetrics: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/monitoring/metrics${queryParams ? `?${queryParams}` : ''}`);
  },
  getHealth: () => requestJson('/health'),
  getHealthMetrics: () => requestJson('/health/metrics'),
  getLlmHealth: () => requestJson('/llm/health'),
  getTtsHealth: () => requestJson('/audio/health'),
  getSttHealth: () => requestJson('/audio/transcriptions/health'),
  getEmbeddingsHealth: () => requestJson('/embeddings/health'),
  getMetrics: () => requestJson('/metrics'),
  getMetricsText: () => requestText('/metrics/text'),
  getRagHealth: () => requestJson('/rag/health'),

  // ============================================
  // Jobs
  // ============================================
  getJobs: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/jobs/list${queryParams ? `?${queryParams}` : ''}`);
  },
  getJobDetail: (jobId: string | number, params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/jobs/${encodeURIComponent(String(jobId))}${queryParams ? `?${queryParams}` : ''}`);
  },
  getJobsStats: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/jobs/stats${queryParams ? `?${queryParams}` : ''}`);
  },
  getJobsStale: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/jobs/stale${queryParams ? `?${queryParams}` : ''}`);
  },
  cancelJobs: (data: Record<string, unknown>) => requestJson('/jobs/batch/cancel', {
    method: 'POST',
    headers: { 'X-Confirm': 'true' },
    body: JSON.stringify(data),
  }),
  retryJobsNow: (data: Record<string, unknown>) => requestJson('/jobs/retry-now', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  requeueQuarantinedJobs: (data: Record<string, unknown>) => requestJson('/jobs/batch/requeue_quarantined', {
    method: 'POST',
    headers: { 'X-Confirm': 'true' },
    body: JSON.stringify(data),
  }),

  // ============================================
  // Usage Analytics
  // ============================================
  getUsageDaily: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/usage/daily${queryParams ? `?${queryParams}` : ''}`);
  },
  getUsageTop: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/usage/top${queryParams ? `?${queryParams}` : ''}`);
  },
  getLlmUsageSummary: (params?: Record<string, QueryParamValue>) => {
    const queryParams = buildQueryString(params);
    return requestJson(`/admin/llm-usage/summary${queryParams ? `?${queryParams}` : ''}`);
  },
  getLlmUsage: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/llm-usage${queryParams ? `?${queryParams}` : ''}`);
  },
  getLlmTopSpenders: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/llm-usage/top-spenders${queryParams ? `?${queryParams}` : ''}`);
  },
  getRouterAnalyticsStatus: (params?: Record<string, string>) => requestRouterAnalytics('status', params),
  getRouterAnalyticsStatusBreakdowns: (params?: Record<string, string>) =>
    requestRouterAnalytics('status/breakdowns', params),
  getRouterAnalyticsQuota: (params?: Record<string, string>) => requestRouterAnalytics('quota', params),
  getRouterAnalyticsProviders: (params?: Record<string, string>) => requestRouterAnalytics('providers', params),
  getRouterAnalyticsAccess: (params?: Record<string, string>) => requestRouterAnalytics('access', params),
  getRouterAnalyticsNetwork: (params?: Record<string, string>) => requestRouterAnalytics('network', params),
  getRouterAnalyticsModels: (params?: Record<string, string>) => requestRouterAnalytics('models', params),
  getRouterAnalyticsConversations: (params?: Record<string, string>) => requestRouterAnalytics('conversations', params),
  getRouterAnalyticsLog: (params?: Record<string, string>) => requestRouterAnalytics('log', params),
  getRouterAnalyticsMeta: (params?: Record<string, string>) => requestRouterAnalytics('meta', params),

  // ============================================
  // Resource Governor
  // ============================================
  getRateLimitEvents: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/rate-limit-events${queryParams ? `?${queryParams}` : ''}`);
  },
  getResourceGovernorPolicy: (params?: { include_ids?: boolean }, signal?: AbortSignal) => {
    const queryParams = params ? new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)])
    ).toString() : '';
    return requestJson(`/resource-governor/policy${queryParams ? `?${queryParams}` : ''}`, { signal });
  },
  simulateResourceGovernorPolicy: (data: Record<string, unknown>) =>
    requestJson('/resource-governor/policy/simulate', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  updateResourceGovernorPolicy: (data: Record<string, unknown>) =>
    requestJson('/resource-governor/policy', {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  deleteResourceGovernorPolicy: (policyId: string) =>
    requestJson(`/resource-governor/policy/${encodeURIComponent(policyId)}`, {
      method: 'DELETE',
    }),

  // ============================================
  // Rate Limiting
  // ============================================
  setRoleRateLimits: (roleId: string, data: { resource: string; limit_per_min?: number | null; burst?: number | null }) =>
    requestJson(`/admin/roles/${encodeURIComponent(roleId)}/rate-limits`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  clearRoleRateLimits: (roleId: string) =>
    requestJson(`/admin/roles/${encodeURIComponent(roleId)}/rate-limits`, {
      method: 'DELETE',
    }),
  getUserRateLimits: (userId: string) =>
    requestJson(`/admin/users/${encodeURIComponent(userId)}/rate-limits`),
  setUserRateLimits: (userId: string, data: { resource: string; limit_per_min?: number | null; burst?: number | null }) =>
    requestJson(`/admin/users/${encodeURIComponent(userId)}/rate-limits`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // ============================================
  // Notification Settings
  // ============================================
  getNotificationSettings: () => requestJson('/monitoring/notifications/settings'),
  updateNotificationSettings: (data: Record<string, unknown>) =>
    requestJson('/monitoring/notifications/settings', {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  testNotification: (data?: Record<string, unknown>) =>
    requestJson('/monitoring/notifications/test', {
      method: 'POST',
      body: JSON.stringify(data ?? {}),
    }),
  getRecentNotifications: () => requestJson('/monitoring/notifications/recent'),

  // ============================================
  // User Permission Overrides
  // ============================================
  getUserPermissionOverrides: (userId: string) =>
    requestJson(`/admin/users/${encodeURIComponent(userId)}/overrides`),
  addUserPermissionOverride: (userId: string, data: { permission_id: number; grant: boolean }) =>
    requestJson(`/admin/users/${encodeURIComponent(userId)}/overrides`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  removeUserPermissionOverride: (userId: string, permissionId: string) =>
    requestJson(`/admin/users/${encodeURIComponent(userId)}/overrides/${encodeURIComponent(permissionId)}`, {
      method: 'DELETE',
    }),

  // ============================================
  // Shared Provider Keys (Org/Team BYOK)
  // ============================================
  getSharedProviderKeys: (params?: { scope_type?: string; scope_id?: number; provider?: string }) => {
    const queryParams = params ? new URLSearchParams(
      Object.entries(params)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => [k, String(v)])
    ).toString() : '';
    return requestJson(`/admin/keys/shared${queryParams ? `?${queryParams}` : ''}`);
  },
  createSharedProviderKey: (data: { scope_type: string; scope_id: number; provider: string; api_key: string }) =>
    requestJson('/admin/keys/shared', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  deleteSharedProviderKey: (scopeType: string, scopeId: number, provider: string) =>
    requestJson(`/admin/keys/shared/${encodeURIComponent(scopeType)}/${encodeURIComponent(String(scopeId))}/${encodeURIComponent(provider)}`, {
      method: 'DELETE',
    }),
  testSharedProviderKey: (data: { scope_type: string; scope_id: number; provider: string; model?: string }) =>
    requestJson('/admin/keys/shared/test', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  createByokValidationRun: (data: ByokValidationRunCreateRequest) =>
    requestJson<ByokValidationRunItem>('/admin/byok/validation-runs', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  getByokValidationRuns: (params?: { limit?: number; offset?: number }) => {
    const queryString = buildQueryString(params);
    return requestJson<ByokValidationRunListResponse>(
      `/admin/byok/validation-runs${queryString ? `?${queryString}` : ''}`
    );
  },
  getByokValidationRun: (runId: string) =>
    requestJson<ByokValidationRunItem>(`/admin/byok/validation-runs/${encodeURIComponent(runId)}`),

  // ============================================
  // Security Health
  // ============================================
  getSecurityHealth: () => requestJson<SecurityHealthData>('/health/security'),
  getSecurityAlertStatus: () => requestJson<SecurityAlertStatus>('/admin/security/alert-status'),
  getKeyAgeStats: () => requestJson<{
    buckets: Array<{ label: string; count: number; color: string }>;
    total: number;
  }>('/admin/security/key-age-stats'),
  debugSimulateRateLimit: (data: { user_id: number; endpoint: string }) =>
    requestJson<{
      user_id: number;
      endpoint: string;
      effective_limit_per_min: number | null;
      effective_burst: number | null;
      limit_source: string;
      would_allow: boolean;
      user_limits: Array<Record<string, unknown>>;
      role_limits: Array<Record<string, unknown>>;
    }>('/admin/debug/simulate-rate-limit', { method: 'POST', body: JSON.stringify(data) }),
  getCostAttribution: (groupBy = 'user', rangeDays = 7) =>
    requestJson<{
      group_by: string;
      range_days: number;
      items: Array<{ entity_id: number; request_count: number; total_tokens: number; estimated_cost_usd: number }>;
    }>(`/admin/usage/cost-attribution?group_by=${groupBy}&range_days=${rangeDays}`),
  getRiskWeights: () => requestJson<{
    weights: Record<string, { weight: number; cap: number }>;
  }>('/admin/security/risk-weights'),
  setRiskWeights: (weights: Record<string, { weight: number; cap: number }>) =>
    requestJson<{ weights: Record<string, { weight: number; cap: number }> }>('/admin/security/risk-weights', {
      method: 'POST',
      body: JSON.stringify(weights),
    }),
  getAllDependenciesHealth: () => requestJson<{
    status: string;
    dependencies: Array<{ name: string; status: string; latency_ms: number; error?: string; detail?: string }>;
    checked_at: string;
  }>('/admin/dependencies/health'),
  getDependenciesUptimeHistory: (params?: Record<string, QueryParamValue>) => {
    const qs = buildQueryString(params);
    return requestJson<{
      range_days: number;
      services: Record<string, Array<{ bucket: string; uptime_pct: number; probes: number }>>;
    }>(`/admin/dependencies/uptime-history${qs ? `?${qs}` : ''}`);
  },
  getBudgetForecast: (orgId: number) => requestJson<{
    org_id: number;
    forecast_available: boolean;
    burn_rate_usd_per_day?: number;
    projected_monthly_usd?: number;
    days_until_exhaustion?: number | null;
  }>(`/admin/budgets/forecast?org_id=${orgId}`),

  // ============================================
  // Compliance Posture
  // ============================================
  getCompliancePosture: () => requestJson<CompliancePosture>('/admin/compliance/posture'),

  // ============================================
  // Compliance Report Schedules
  // ============================================
  getReportSchedules: () =>
    requestJson<{ items: ComplianceReportSchedule[]; total: number }>('/admin/compliance/report-schedules'),
  createReportSchedule: (data: { frequency: string; recipients: string[]; format: string; enabled: boolean }) =>
    requestJson<ComplianceReportSchedule>('/admin/compliance/report-schedules', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  updateReportSchedule: (scheduleId: string, data: Record<string, unknown>) =>
    requestJson<ComplianceReportSchedule>(`/admin/compliance/report-schedules/${encodeURIComponent(scheduleId)}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),
  deleteReportSchedule: (scheduleId: string) =>
    requestJson(`/admin/compliance/report-schedules/${encodeURIComponent(scheduleId)}`, {
      method: 'DELETE',
    }),
  sendReportNow: (scheduleId: string) =>
    requestJson<{ sent_count: number; total_recipients: number; errors: string[] }>(
      `/admin/compliance/report-schedules/${encodeURIComponent(scheduleId)}/send-now`,
      { method: 'POST' },
    ),

  // ============================================
  // Email Digest Preferences
  // ============================================
  getDigestPreference: () => requestJson<DigestPreference>('/admin/digest/preference'),
  setDigestPreference: (data: { email: string; frequency: string }) =>
    requestJson<DigestPreference>('/admin/digest/preference', {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  // ============================================
  // System Dependencies Health
  // ============================================
  getSystemDependencies: () => requestJson<SystemDependenciesResponse>('/admin/dependencies'),
  getDependencyUptime: (name: string, days: number = 7) =>
    requestJson<DependencyUptimeStats>(`/admin/dependencies/${encodeURIComponent(name)}/uptime?days=${days}`),

  // ============================================
  // Virtual API Keys
  // ============================================
  getUserVirtualKeys: (userId: string) =>
    requestJson(`/admin/users/${encodeURIComponent(userId)}/virtual-keys`),
  createUserVirtualKey: (userId: string, data: { name: string; scopes: string[] }) =>
    requestJson(`/admin/users/${encodeURIComponent(userId)}/virtual-keys`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  deleteUserVirtualKey: (userId: string, keyId: string) =>
    requestJson(`/admin/users/${encodeURIComponent(userId)}/virtual-keys/${encodeURIComponent(keyId)}`, {
      method: 'DELETE',
    }),

  // ============================================
  // Tool Permissions (Role-specific)
  // ============================================
  getRoleToolPermissions: (roleId: string) =>
    requestJson(`/admin/roles/${encodeURIComponent(roleId)}/permissions/tools`),
  batchGrantToolPermissions: (roleId: string, data: { tools: string[] }) =>
    requestJson(`/admin/roles/${encodeURIComponent(roleId)}/permissions/tools/batch`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  batchRevokeToolPermissions: (roleId: string, data: { tools: string[] }) =>
    requestJson(`/admin/roles/${encodeURIComponent(roleId)}/permissions/tools/batch/revoke`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  grantToolPermissionsByPrefix: (roleId: string, prefix: string) =>
    requestJson(`/admin/roles/${encodeURIComponent(roleId)}/permissions/tools/prefix/grant`, {
      method: 'POST',
      body: JSON.stringify({ prefix }),
    }),
  revokeToolPermissionsByPrefix: (roleId: string, prefix: string) =>
    requestJson(`/admin/roles/${encodeURIComponent(roleId)}/permissions/tools/prefix/revoke`, {
      method: 'POST',
      body: JSON.stringify({ prefix }),
    }),

  // ============================================
  // Cleanup Settings
  // ============================================
  getCleanupSettings: () => requestJson('/admin/cleanup-settings'),
  updateCleanupSettings: (data: Record<string, unknown>) =>
    requestJson('/admin/cleanup-settings', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // ============================================
  // Notes Title Settings
  // ============================================
  getNotesTitleSettings: () => requestJson('/admin/notes/title-settings'),
  updateNotesTitleSettings: (data: Record<string, unknown>) =>
    requestJson('/admin/notes/title-settings', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // ============================================
  // Kanban FTS Maintenance
  // ============================================
  runKanbanFtsMaintenance: () =>
    requestJson('/admin/kanban/fts-maintenance', {
      method: 'POST',
    }),

  // ============================================
  // Job SLA & Attachments
  // ============================================
  getJobSlaPolicies: () => requestJson('/admin/jobs/sla/policies'),
  createJobSlaPolicy: (data: Record<string, unknown>) =>
    requestJson('/admin/jobs/sla/policy', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  deleteJobSlaPolicy: (data: { domain: string; queue: string; job_type: string }) =>
    requestJson('/admin/jobs/sla/policy', {
      method: 'DELETE',
      body: JSON.stringify(data),
    }),
  getJobSlaBreaches: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/jobs/sla/breaches${queryParams ? `?${queryParams}` : ''}`);
  },
  getJobAttachments: (jobId: string | number, params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson(`/admin/jobs/${encodeURIComponent(String(jobId))}/attachments${queryParams ? `?${queryParams}` : ''}`);
  },
  addJobAttachment: (jobId: string, data: FormData) =>
    requestJson(`/admin/jobs/${encodeURIComponent(jobId)}/attachments`, {
      method: 'POST',
      body: data,
    }),
  rotateJobCrypto: () =>
    requestJson('/admin/jobs/crypto/rotate', {
      method: 'POST',
    }),
  createMaintenanceRotationRun: (data: MaintenanceRotationRunCreateRequest) =>
    requestJson<MaintenanceRotationRunCreateResponse>('/admin/maintenance/rotation-runs', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  getMaintenanceRotationRuns: (params?: { limit?: number; offset?: number }) => {
    const search = new URLSearchParams();
    if (typeof params?.limit === 'number') search.set('limit', String(params.limit));
    if (typeof params?.offset === 'number') search.set('offset', String(params.offset));
    const query = search.toString();
    return requestJson<MaintenanceRotationRunListResponse>(
      `/admin/maintenance/rotation-runs${query ? `?${query}` : ''}`
    );
  },
  getMaintenanceRotationRun: (runId: string) =>
    requestJson<MaintenanceRotationRunItem>(`/admin/maintenance/rotation-runs/${encodeURIComponent(runId)}`),

  // ============================================
  // Organization Watchlist Settings
  // ============================================
  getOrgWatchlistSettings: (orgId: string) =>
    requestJson<WatchlistSettings>(`/admin/orgs/${encodeURIComponent(orgId)}/watchlists/settings`),
  updateOrgWatchlistSettings: (orgId: string, data: Record<string, unknown>) =>
    requestJson(`/admin/orgs/${encodeURIComponent(orgId)}/watchlists/settings`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  // ============================================
  // Debug Tools
  // ============================================
  debugResolveApiKey: (value: string, options?: { mode?: 'raw_key' | 'key_id' | 'user_id' }) => {
    const mode = options?.mode ?? 'raw_key';
    if (mode === 'key_id') {
      return requestJson(`/authnz/debug/api-key-id?key_id=${encodeURIComponent(value)}`);
    }
    if (mode === 'user_id') {
      return requestJson(`/authnz/debug/api-key-id?user_id=${encodeURIComponent(value)}`);
    }
    return requestJson('/authnz/debug/api-key-id', {
      headers: { 'X-API-KEY': value },
    });
  },
  debugGetBudgetSummary: (apiKey: string) =>
    requestJson('/authnz/debug/budget-summary', {
      headers: { 'X-API-KEY': apiKey },
    }),
  debugResolvePermissions: (userId: string) =>
    requestJson(`/authnz/debug/permissions?user_id=${encodeURIComponent(userId)}`),
  debugValidateToken: (token: string) =>
    requestJson('/authnz/debug/validate-token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    }),

  // ============================================
  // ACP Sessions (Admin)
  // ============================================
  getACPSessions: (params?: Record<string, QueryParamValue>) => {
    const queryParams = buildQueryString(params);
    return requestJson(`/admin/acp/sessions${queryParams ? `?${queryParams}` : ''}`);
  },
  getACPSessionUsage: (sessionId: string) =>
    requestJson(`/admin/acp/sessions/${encodeURIComponent(sessionId)}/usage`),
  closeACPSession: (sessionId: string) =>
    requestJson(`/admin/acp/sessions/${encodeURIComponent(sessionId)}/close`, {
      method: 'POST',
    }),
  setSessionBudget: (sessionId: string, data: { token_budget: number; auto_terminate_at_budget?: boolean }) =>
    requestJson(`/admin/acp/sessions/${encodeURIComponent(sessionId)}/budget`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  // ============================================
  // ACP Agent Configs (Admin)
  // ============================================
  getACPAgentConfigs: (params?: Record<string, QueryParamValue>) => {
    const queryParams = buildQueryString(params);
    return requestJson(`/admin/acp/agents${queryParams ? `?${queryParams}` : ''}`);
  },
  getACPAgentConfig: (configId: number) =>
    requestJson(`/admin/acp/agents/${configId}`),
  createACPAgentConfig: (data: Record<string, unknown>) =>
    requestJson('/admin/acp/agents', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  updateACPAgentConfig: (configId: number, data: Record<string, unknown>) =>
    requestJson(`/admin/acp/agents/${configId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  deleteACPAgentConfig: (configId: number) =>
    requestJson(`/admin/acp/agents/${configId}`, {
      method: 'DELETE',
    }),
  getACPAgentMetrics: () =>
    requestJson<{ items: Array<{
      agent_type: string;
      session_count: number;
      active_sessions: number;
      total_prompt_tokens: number;
      total_completion_tokens: number;
      total_tokens: number;
      total_messages: number;
      last_used_at: string | null;
      total_estimated_cost_usd: number | null;
    }> }>('/admin/acp/agents/metrics'),
  getACPAgentUsage: (rangeDays = 7) =>
    requestJson<{
      agents: Array<{
        agent_type: string;
        invocation_count: number;
        total_tokens: number;
        prompt_tokens: number;
        completion_tokens: number;
        estimated_cost_usd: number;
        error_count: number;
        avg_tokens_per_session: number;
      }>;
      range_days: number;
    }>(`/admin/acp/agents/usage?range_days=${rangeDays}`),

  // ============================================
  // ACP Permission Policies (Admin)
  // ============================================
  getACPPermissionPolicies: (params?: Record<string, QueryParamValue>) => {
    const queryParams = buildQueryString(params);
    return requestJson(`/admin/acp/permission-policies${queryParams ? `?${queryParams}` : ''}`);
  },
  createACPPermissionPolicy: (data: Record<string, unknown>) =>
    requestJson('/admin/acp/permission-policies', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  updateACPPermissionPolicy: (policyId: number, data: Record<string, unknown>) =>
    requestJson(`/admin/acp/permission-policies/${policyId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  deleteACPPermissionPolicy: (policyId: number) =>
    requestJson(`/admin/acp/permission-policies/${policyId}`, {
      method: 'DELETE',
    }),

  // ============================================
  // MCP Servers (Admin)
  // ============================================
  getMCPStatus: () => requestJson('/mcp/status'),
  getMCPMetrics: () => requestJson('/mcp/metrics'),
  getMCPTools: () => requestJson('/mcp/tools'),
  getMCPModules: () => requestJson('/mcp/modules'),
  getMCPModulesHealth: () => requestJson('/mcp/modules/health'),
  getMCPHealth: () => requestJson('/mcp/health'),
  getMCPToolUsage: (params?: Record<string, QueryParamValue>) => {
    const qs = buildQueryString(params);
    return requestJson<{
      period_seconds: number;
      modules: Record<string, { calls: number; avg_latency_ms: number }>;
      tools: Record<string, { calls: number; avg_latency_ms: number }>;
    }>(`/admin/mcp/tool-usage${qs ? `?${qs}` : ''}`);
  },

  // ============================================
  // Voice Commands & Assistant
  // ============================================
  getVoiceCommands: async (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson<VoiceCommandListResponse | VoiceCommand[]>(`/voice/commands${queryParams ? `?${queryParams}` : ''}`);
  },
  getVoiceCommand: (commandId: string, signal?: AbortSignal) =>
    requestJson<VoiceCommand>(`/voice/commands/${encodeURIComponent(commandId)}`, { signal }),
  createVoiceCommand: (data: Record<string, unknown>) =>
    requestJson<VoiceCommand>('/voice/commands', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  updateVoiceCommand: (commandId: string, data: Record<string, unknown>) =>
    requestJson<VoiceCommand>(`/voice/commands/${encodeURIComponent(commandId)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  deleteVoiceCommand: (commandId: string) =>
    requestJson(`/voice/commands/${encodeURIComponent(commandId)}`, {
      method: 'DELETE',
    }),
  toggleVoiceCommand: (commandId: string, enabled: boolean) =>
    requestJson<VoiceCommand>(`/voice/commands/${encodeURIComponent(commandId)}/toggle`, {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    }),
  validateVoiceCommand: (commandId: string) =>
    requestJson<VoiceCommandValidationResponse>(`/voice/commands/${encodeURIComponent(commandId)}/validate`, {
      method: 'POST',
    }),

  // Voice Sessions
  getVoiceSessions: (params?: Record<string, string>) => {
    const queryParams = params ? new URLSearchParams(params).toString() : '';
    return requestJson<VoiceSessionListResponse | VoiceSession[]>(`/voice/sessions${queryParams ? `?${queryParams}` : ''}`);
  },
  getVoiceSession: (sessionId: string) =>
    requestJson<VoiceSession>(`/voice/sessions/${encodeURIComponent(sessionId)}`),
  deleteVoiceSession: (sessionId: string) =>
    requestJson(`/voice/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'DELETE',
    }),

  // Voice Analytics
  getVoiceAnalytics: (params?: { days?: number; user_id?: number }) => {
    const queryParams = params ? new URLSearchParams(
      Object.entries(params)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => [k, String(v)])
    ).toString() : '';
    return requestJson<VoiceAnalyticsSummary>(`/voice/analytics${queryParams ? `?${queryParams}` : ''}`);
  },
  getVoiceCommandUsage: (commandId: string, params?: { days?: number }, signal?: AbortSignal) => {
    const queryParams = params ? new URLSearchParams(
      Object.entries(params)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => [k, String(v)])
    ).toString() : '';
    return requestJson<VoiceCommandUsage>(`/voice/commands/${encodeURIComponent(commandId)}/usage${queryParams ? `?${queryParams}` : ''}`, { signal });
  },

  // Voice Command Dry-Run
  dryRunVoiceCommand: (data: { phrase: string; command_id?: string }) =>
    requestJson<{
      dry_run: boolean;
      phrase: string;
      matched: boolean;
      match_method: string;
      matched_phrase: string | null;
      confidence: number | null;
      action_type: string;
      action_config: Record<string, unknown>;
      processing_time_ms: number;
      alternatives: Array<{ action_type: string; confidence: number | null; raw_text: string | null }>;
    }>('/voice/commands/dry-run', { method: 'POST', body: JSON.stringify(data) }),

  // Voice Workflow Templates
  getVoiceWorkflowTemplates: () =>
    requestJson('/voice/workflows/templates'),

  // ============================================
  // Plans & Billing
  // ============================================
  getBillingAnalytics: () =>
    requestJson<BillingAnalytics>('/admin/billing/analytics'),

  getPlans: (params?: Record<string, QueryParamValue>) => {
    const qs = buildQueryString(params);
    return requestJson<Plan[]>(`/billing/plans${qs ? `?${qs}` : ''}`);
  },
  getPlan: (planId: string) =>
    requestJson<Plan>(`/billing/plans/${encodeURIComponent(planId)}`),
  createPlan: (data: CreatePlanInput) =>
    requestJson<Plan>('/billing/plans', { method: 'POST', body: JSON.stringify(data) }),
  updatePlan: (planId: string, data: UpdatePlanInput) =>
    requestJson<Plan>(`/billing/plans/${encodeURIComponent(planId)}`, { method: 'PUT', body: JSON.stringify(data) }),
  deletePlan: (planId: string) =>
    requestJson(`/billing/plans/${encodeURIComponent(planId)}`, { method: 'DELETE' }),

  // Subscriptions
  getSubscriptions: (params?: Record<string, QueryParamValue>) => {
    const qs = buildQueryString(params);
    return requestJson<Subscription[]>(`/billing/subscriptions${qs ? `?${qs}` : ''}`);
  },
  getOrgSubscription: (orgId: number) =>
    requestJson<Subscription>(`/billing/orgs/${orgId}/subscription`),
  createSubscription: (orgId: number, data: { plan_id: string; trial_days?: number }) =>
    requestJson<{ checkout_url?: string; subscription?: Subscription }>(
      `/billing/orgs/${orgId}/subscription`, { method: 'POST', body: JSON.stringify(data) }),
  updateSubscription: (orgId: number, data: { plan_id: string }) =>
    requestJson<Subscription>(`/billing/orgs/${orgId}/subscription`, { method: 'PUT', body: JSON.stringify(data) }),
  cancelSubscription: (orgId: number) =>
    requestJson(`/billing/orgs/${orgId}/subscription`, { method: 'DELETE' }),

  // Usage & Invoices
  getOrgUsageSummary: (orgId: number, params?: { period?: string }) => {
    const qs = buildQueryString(params);
    return requestJson<OrgUsageSummary>(`/billing/orgs/${orgId}/usage${qs ? `?${qs}` : ''}`);
  },
  getOrgInvoices: (orgId: number, params?: Record<string, QueryParamValue>) => {
    const qs = buildQueryString(params);
    return requestJson<Invoice[]>(`/billing/orgs/${orgId}/invoices${qs ? `?${qs}` : ''}`);
  },

  // Feature Registry
  getFeatureRegistry: () =>
    requestJson<FeatureRegistryEntry[]>('/billing/feature-registry'),
  updateFeatureRegistry: (data: FeatureRegistryEntry[]) =>
    requestJson<FeatureRegistryEntry[]>('/billing/feature-registry', { method: 'PUT', body: JSON.stringify(data) }),

  // Onboarding
  createOnboardingSession: (data: { org_name: string; org_slug: string; plan_id: string; owner_email?: string }) =>
    requestJson<{ checkout_url?: string; org_id?: number }>(
      '/billing/onboarding', { method: 'POST', body: JSON.stringify(data) }),

  // Admin Webhooks
  getWebhooks: (params?: Record<string, QueryParamValue>, options?: RequestInit) => {
    const qs = buildQueryString(params);
    return requestJson<AdminWebhooksResponse>(`/admin/webhooks${qs ? `?${qs}` : ''}`, options);
  },
  createWebhook: (data: { url: string; event_types?: string[]; description?: string; secret?: string; active?: boolean; retry_count?: number; timeout_seconds?: number }) =>
    requestJson<AdminWebhook>('/admin/webhooks', { method: 'POST', body: JSON.stringify(data) }),
  getWebhook: (id: number) =>
    requestJson<AdminWebhook>(`/admin/webhooks/${id}`),
  updateWebhook: (id: number, data: Partial<{ url: string; event_types: string[]; description: string; secret: string; active: boolean; retry_count: number; timeout_seconds: number }>) =>
    requestJson<AdminWebhook>(`/admin/webhooks/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  deleteWebhook: (id: number) =>
    requestJson<{ deleted: boolean; id: number }>(`/admin/webhooks/${id}`, { method: 'DELETE' }),
  testWebhook: (id: number) =>
    requestJson<AdminWebhookTestResult>(`/admin/webhooks/${id}/test`, { method: 'POST' }),
  getWebhookDeliveries: (id: number, params?: Record<string, QueryParamValue>) => {
    const qs = buildQueryString(params);
    return requestJson<AdminWebhookDeliveryLogResponse>(`/admin/webhooks/${id}/deliveries${qs ? `?${qs}` : ''}`);
  },
};

export default api;
