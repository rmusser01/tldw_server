import { bgRequest } from "@/services/background-proxy";
import { buildQuery } from "../client-utils";
import type {
  AdminUserListResponse,
  AdminUserUpdateRequest,
  AdminUserCreateRequest,
  AdminUserSummary,
  AdminRole,
  TldwConfig,
} from "../TldwApiClient";

type UserRolesResponse = { user_id: number; roles: AdminRole[] };
type UserOverridesResponse = {
  user_id: number;
  overrides: Array<{ permission_id: number; permission_name: string; granted: boolean; expires_at?: string | null }>;
};
type UserEffectivePermissionsResponse = { user_id: number; permissions: string[] };

export type BundleImportResult = {
  status: string;
  datasets_restored: string[];
  warnings: string[];
  rollback_failures?: string[];
  validations: Array<{ dataset: string; compatible: boolean; message?: string }>;
  safety_snapshots?: Record<string, string>;
};
type BundleRequestScope = { configSnapshot: TldwConfig; abortSignal: AbortSignal };

export const adminMethods = {
  // ── Admin Users & Roles ──

  async createAdminUser(
    payload: AdminUserCreateRequest,
  ): Promise<AdminUserSummary> {
    return await bgRequest<AdminUserSummary>({
      path: "/api/v1/admin/users",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async listAdminUsers(
    params?: {
      page?: number;
      limit?: number;
      role?: string;
      is_active?: boolean;
      search?: string;
    },
    options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal },
  ): Promise<AdminUserListResponse> {
    const query = buildQuery(params as Record<string, any>);
    return await bgRequest<AdminUserListResponse>({
      path: `/api/v1/admin/users${query}`,
      method: "GET",
      ...options,
    });
  },

  async updateAdminUser(
    userId: number,
    payload: AdminUserUpdateRequest,
    options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal },
  ): Promise<{ message: string }> {
    return await bgRequest<{ message: string }>({
      path: `/api/v1/admin/users/${userId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...options,
    });
  },

  async resetAdminUserPassword(
    userId: number,
    payload: {
      temporary_password: string;
      reason: string;
      force_password_change?: boolean;
    },
  ): Promise<{
    user_id: number;
    force_password_change: boolean;
    message: string;
  }> {
    return await bgRequest<{
      user_id: number;
      force_password_change: boolean;
      message: string;
    }>({
      path: `/api/v1/admin/users/${userId}/reset-password`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        force_password_change: true,
        ...payload,
      },
    });
  },

  async listAdminRoles(options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<AdminRole[]> {
    return await bgRequest<AdminRole[]>({
      path: "/api/v1/admin/roles",
      method: "GET",
      ...options,
    });
  },

  async createAdminRole(name: string,
    description?: string, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<AdminRole> {
    return await bgRequest<AdminRole>({
      path: "/api/v1/admin/roles",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { name, description },
      ...options,
    });
  },

  async deleteAdminRole(roleId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<{ message: string }> {
    return await bgRequest<{ message: string }>({
      path: `/api/v1/admin/roles/${roleId}`,
      method: "DELETE",
      ...options,
    });
  },

  // ── Admin API Key Management ──

  async listUserApiKeys(
    userId: number,
    options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal },
  ): Promise<any[]> {
    return await bgRequest<any[]>({
      path: `/api/v1/admin/users/${userId}/api-keys`,
      method: "GET",
      ...options,
    });
  },

  async createUserApiKey(
    userId: number,
    payload: { name?: string },
    options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/api-keys`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...options,
    });
  },

  async revokeUserApiKey(
    userId: number,
    keyId: number,
    options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal },
  ): Promise<{ message: string }> {
    return await bgRequest<{ message: string }>({
      path: `/api/v1/admin/users/${userId}/api-keys/${keyId}`,
      method: "DELETE",
      ...options,
    });
  },

  async updateUserApiKey(
    userId: number,
    keyId: number,
    payload: { rate_limit?: number; allowed_ips?: string[] },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/api-keys/${keyId}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async rotateUserApiKey(
    userId: number,
    keyId: number,
    options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/api-keys/${keyId}/rotate`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {},
      ...options,
    });
  },

  async listUserVirtualKeys(userId: number): Promise<any[]> {
    return await bgRequest<any[]>({
      path: `/api/v1/admin/users/${userId}/virtual-keys`,
      method: "GET",
    });
  },

  async getApiKeyAuditLog(keyId: number): Promise<any[]> {
    return await bgRequest<any[]>({
      path: `/api/v1/admin/api-keys/${keyId}/audit-log`,
      method: "GET",
    });
  },

  // ── Admin Maintenance ──

  async getMaintenanceState(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/maintenance",
      method: "GET",
    });
  },

  async updateMaintenanceState(payload: {
    enabled: boolean;
    message?: string;
    allowlist_user_ids?: number[];
    allowlist_emails?: string[];
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/maintenance",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async listFeatureFlags(): Promise<any[]> {
    const result = await bgRequest<unknown[] | { items: unknown[] }>({
      path: "/api/v1/admin/feature-flags",
      method: "GET",
    });
    return Array.isArray(result) ? result : result.items;
  },

  async updateFeatureFlag(
    flagKey: string,
    payload: {
      enabled: boolean; scope: "global" | "org" | "user";
      org_id?: number; user_id?: number; target_user_ids: number[];
      rollout_percent: number; variant_value: string | null;
    },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/feature-flags/${encodeURIComponent(flagKey)}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async deleteFeatureFlag(flagKey: string, target: {
    scope: "global" | "org" | "user"; org_id?: number; user_id?: number;
  }): Promise<any> {
    const query = buildQuery(target);
    return await bgRequest<any>({
      path: `/api/v1/admin/feature-flags/${encodeURIComponent(flagKey)}${query}`,
      method: "DELETE",
    });
  },

  async listIncidents(): Promise<any[]> {
    const result = await bgRequest<unknown[] | { items: unknown[] }>({
      path: "/api/v1/admin/incidents",
      method: "GET",
    });
    return Array.isArray(result) ? result : result.items;
  },

  async createIncident(payload: {
    title: string;
    severity?: string;
    summary?: string;
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/incidents",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async updateIncident(
    incidentId: number,
    payload: { status?: string; severity?: string; summary?: string },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/incidents/${incidentId}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async deleteIncident(incidentId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/incidents/${incidentId}`,
      method: "DELETE",
    });
  },

  async listRotationRuns(): Promise<any[]> {
    const result = await bgRequest<unknown[] | { items: unknown[] }>({
      path: "/api/v1/admin/maintenance/rotation-runs",
      method: "GET",
    });
    return Array.isArray(result) ? result : result.items;
  },

  async createRotationRun(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/maintenance/rotation-runs",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { mode: "dry_run" },
    });
  },

  // ── Admin Monitoring & Alerting ──

  async listAlertRules(): Promise<any[]> {
    return await bgRequest<any[]>({
      path: "/api/v1/admin/monitoring/alert-rules",
      method: "GET",
    });
  },

  async createAlertRule(payload: {
    metric: string;
    operator: string;
    threshold: number;
    duration_minutes: number;
    severity: string;
    enabled?: boolean;
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/monitoring/alert-rules",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async deleteAlertRule(ruleId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/monitoring/alert-rules/${ruleId}`,
      method: "DELETE",
    });
  },

  async assignAlert(
    alertIdentity: string,
    payload: { assigned_to_user_id?: number | null },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/monitoring/alerts/${encodeURIComponent(alertIdentity)}/assign`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async snoozeAlert(
    alertIdentity: string,
    payload: { until: string },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/monitoring/alerts/${encodeURIComponent(alertIdentity)}/snooze`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async escalateAlert(alertIdentity: string): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/monitoring/alerts/${encodeURIComponent(alertIdentity)}/escalate`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {},
    });
  },

  async listAlertHistory(params?: { limit?: number }): Promise<any[]> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any[]>({ path: `/api/v1/admin/monitoring/alerts/history${query}`, method: "GET" })
  },

  async getSecurityAlertStatus(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/security/alert-status",
      method: "GET",
    });
  },

  async getDashboardActivity(params?: {
    days?: number;
    granularity?: string;
  }): Promise<any> {
    const query = buildQuery(params as Record<string, any>);
    return await bgRequest<any>({
      path: `/api/v1/admin/activity${query}`,
      method: "GET",
    });
  },

  // ── Admin Usage Analytics ──

  // `start`/`end` are the documented backend params on /usage/daily
  // (YYYY-MM-DD inclusive, admin_usage.py); the old start_date/end_date
  // keys were silently ignored by the daily chart's range selector.
  async getDailyUsage(params?: { start?: string; end?: string }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/usage/daily${query}`, method: "GET" })
  },

  // `start`/`end` pass straight through to the documented backend params:
  // YYYY-MM-DD inclusive on /usage/top, ISO timestamps on the llm-usage*
  // endpoints (admin_usage.py).
  async getTopUsage(params?: {
    metric?: string
    limit?: number
    start?: string
    end?: string
  }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/usage/top${query}`, method: "GET" })
  },

  async exportDailyUsageCsv(): Promise<string> {
    return await bgRequest<string>({
      path: "/api/v1/admin/usage/daily/export.csv",
      method: "GET",
    });
  },

  async exportTopUsageCsv(): Promise<string> {
    return await bgRequest<string>({
      path: "/api/v1/admin/usage/top/export.csv",
      method: "GET",
    });
  },

  async getLlmUsage(params?: {
    provider?: string
    model?: string
    limit?: number
    start?: string
    end?: string
  }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/llm-usage${query}`, method: "GET" })
  },

  async getLlmUsageSummary(params?: {
    group_by?: string
    start?: string
    end?: string
  }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/llm-usage/summary${query}`, method: "GET" })
  },

  async getLlmTopSpenders(params?: {
    limit?: number
    start?: string
    end?: string
  }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/llm-usage/top-spenders${query}`, method: "GET" })
  },

  async getRouterAnalyticsStatus(params?: { range?: string }): Promise<any> {
    const query = buildQuery(params as Record<string, any>);
    return await bgRequest<any>({
      path: `/api/v1/admin/router-analytics/status${query}`,
      method: "GET",
    });
  },

  async getRouterAnalyticsProviders(params?: { range?: string }): Promise<any> {
    const query = buildQuery(params as Record<string, any>);
    return await bgRequest<any>({
      path: `/api/v1/admin/router-analytics/providers${query}`,
      method: "GET",
    });
  },

  // ── Admin Organizations & Teams ──

  async createOrg(payload: { name: string; slug?: string }, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/orgs",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...options,
    });
  },

  async listOrgs(params?: {
    search?: string;
    limit?: number;
    offset?: number;
  }, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    const query = buildQuery(params as Record<string, any>);
    return await bgRequest<any>({
      path: `/api/v1/admin/orgs${query}`,
      method: "GET",
      ...options,
    });
  },

  // `limit`/`offset` page server-side; the backend returns a plain array
  // today, so callers must treat a missing total as "unknown" (see
  // OrgsTeamsPage's fallback-total paging).
  async listOrgMembers(
    orgId: number,
    params?: { role?: string; status?: string; limit?: number; offset?: number },
    options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }
  ): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/orgs/${orgId}/members${query}`, method: "GET", ...options })
  },

  async addOrgMember(orgId: number,
    payload: { user_id: number; role?: string }, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/orgs/${orgId}/members`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...options,
    });
  },

  async removeOrgMember(orgId: number, userId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/orgs/${orgId}/members/${userId}`,
      method: "DELETE",
      ...options,
    });
  },

  async updateOrgMemberRole(orgId: number,
    userId: number,
    payload: { role: string }, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/orgs/${orgId}/members/${userId}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...options,
    });
  },

  async createTeam(orgId: number, payload: { name: string }, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/orgs/${orgId}/teams`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...options,
    });
  },

  // `limit`/`offset` page server-side; same missing-total caveat as
  // listOrgMembers above.
  async listTeams(orgId: number, params?: { limit?: number; offset?: number }, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/orgs/${orgId}/teams${query}`, method: "GET", ...options })
  },

  // The current backend ignores paging params here (returns the full list),
  // but they are sent so server-side paging works once the endpoint adopts
  // limit/offset like its siblings.
  async listTeamMembers(teamId: number, params?: { limit?: number; offset?: number }, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/teams/${teamId}/members${query}`, method: "GET", ...options })
  },

  async addTeamMember(teamId: number,
    payload: { user_id: number; role?: string }, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/teams/${teamId}/members`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...options,
    });
  },

  async removeTeamMember(teamId: number, userId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/teams/${teamId}/members/${userId}`,
      method: "DELETE",
      ...options,
    });
  },

  async updateTeamMemberRole(teamId: number,
    userId: number,
    payload: { role: string }, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/teams/${teamId}/members/${userId}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...options,
    });
  },

  // ── Admin Data Operations ──

  async listBackups(params?: {
    dataset?: string;
    user_id?: number;
  }): Promise<any> {
    const query = buildQuery(params as Record<string, any>);
    return await bgRequest<any>({
      path: `/api/v1/admin/backups${query}`,
      method: "GET",
    });
  },

  async createBackup(payload: {
    dataset: string;
    user_id?: number;
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/backups",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async restoreBackup(
    backupId: string,
    payload: { dataset: string; user_id?: number; confirm: true },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/backups/${encodeURIComponent(backupId)}/restore`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async listBackupSchedules(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/backup-schedules",
      method: "GET",
    });
  },

  async createBackupSchedule(payload: {
    dataset: string;
    target_user_id?: number;
    frequency: "daily" | "weekly" | "monthly";
    time_of_day: string;
    timezone?: string;
    retention_count: number;
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/backup-schedules",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async deleteBackupSchedule(scheduleId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/backup-schedules/${scheduleId}`,
      method: "DELETE",
    });
  },

  async previewDsr(payload: {
    requester_identifier: string;
    request_type?: string;
    categories?: string[];
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/data-subject-requests/preview",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async createDsr(payload: {
    requester_identifier: string;
    request_type: string;
    categories?: string[];
    client_request_id?: string;
    notes?: string;
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/data-subject-requests",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async listDsrs(params?: { limit?: number; offset?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>);
    return await bgRequest<any>({
      path: `/api/v1/admin/data-subject-requests${query}`,
      method: "GET",
    });
  },

  async executeDsr(requestId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/data-subject-requests/${requestId}/execute`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {},
    });
  },

  async listRetentionPolicies(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/retention-policies",
      method: "GET",
    });
  },

  async updateRetentionPolicy(
    policyKey: string,
    payload: { retention_days: number },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/retention-policies/${encodeURIComponent(policyKey)}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async listBundles(scope?: BundleRequestScope): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/backups/bundles",
      method: "GET",
      ...scope,
    });
  },

  async createBundle(payload: { datasets: string[]; user_id?: number }, scope?: BundleRequestScope): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/backups/bundles",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...scope,
    });
  },

  async downloadBundle(bundleId: string, scope: BundleRequestScope): Promise<Blob> {
    const data = await bgRequest<ArrayBuffer>({
      path: `/api/v1/admin/backups/bundles/${encodeURIComponent(bundleId)}/download`,
      method: "GET",
      responseType: "arrayBuffer",
      ...scope,
    });
    if (!(data instanceof ArrayBuffer) || !data.byteLength) throw new Error("Backup download did not return a file.");
    return new Blob([data], { type: "application/zip" });
  },

  async importBundle(file: File, params: { user_id?: number; dry_run: boolean }, scope: BundleRequestScope): Promise<BundleImportResult> {
    if (scope.abortSignal.aborted) throw new DOMException("Backup transfer cancelled", "AbortError");
    const body = new FormData();
    body.append("file", file, file.name);
    return await bgRequest<BundleImportResult>({
      path: `/api/v1/admin/backups/bundles/import${buildQuery(params)}`,
      method: "POST",
      body,
      preferDirect: true,
      timeoutMs: 15000,
      ...scope,
    });
  },

  async deleteBundle(bundleId: string, scope?: BundleRequestScope): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/backups/bundles/${encodeURIComponent(bundleId)}`,
      method: "DELETE",
      ...scope,
    });
  },

  // ── Admin RBAC & Permissions ──

  async listPermissions(options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any[]> {
    return await bgRequest<any[]>({
      path: "/api/v1/admin/permissions",
      method: "GET",
      ...options,
    });
  },

  async listPermissionCategories(options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any[]> {
    return await bgRequest<any[]>({
      path: "/api/v1/admin/permissions/categories",
      method: "GET",
      ...options,
    });
  },

  async getRolePermissionMatrix(options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/roles/matrix",
      method: "GET",
      ...options,
    });
  },

  async listRolePermissions(roleId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any[]> {
    return await bgRequest<any[]>({
      path: `/api/v1/admin/roles/${roleId}/permissions`,
      method: "GET",
      ...options,
    });
  },

  async grantRolePermission(roleId: number,
    permissionId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/roles/${roleId}/permissions/${permissionId}`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {},
      ...options,
    });
  },

  async revokeRolePermission(roleId: number,
    permissionId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/roles/${roleId}/permissions/${permissionId}`,
      method: "DELETE",
      ...options,
    });
  },

  async listUserRoles(userId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<UserRolesResponse> {
    return await bgRequest<UserRolesResponse>({
      path: `/api/v1/admin/users/${userId}/roles`,
      method: "GET",
      ...options,
    });
  },

  async assignUserRole(userId: number, roleId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/roles/${roleId}`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {},
      ...options,
    });
  },

  async removeUserRole(userId: number, roleId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/roles/${roleId}`,
      method: "DELETE",
      ...options,
    });
  },

  async listUserOverrides(userId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<UserOverridesResponse> {
    return await bgRequest<UserOverridesResponse>({
      path: `/api/v1/admin/users/${userId}/overrides`,
      method: "GET",
      ...options,
    });
  },

  async getUserEffectivePermissions(userId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<UserEffectivePermissionsResponse> {
    return await bgRequest<UserEffectivePermissionsResponse>({
      path: `/api/v1/admin/users/${userId}/effective-permissions`,
      method: "GET",
      ...options,
    });
  },

  async addUserOverride(userId: number,
    payload: { permission_id: number; effect: "allow" | "deny" }, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/overrides`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      ...options,
    });
  },

  async deleteUserOverride(userId: number, permissionId: number, options?: { configSnapshot?: TldwConfig; abortSignal?: AbortSignal }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/overrides/${permissionId}`,
      method: "DELETE",
      ...options,
    });
  },

  // ── Admin Billing ──

  async getBillingOverview(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/billing/overview",
      method: "GET",
    });
  },

  async listAllSubscriptions(params?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<any> {
    const query = buildQuery(params as Record<string, any>);
    return await bgRequest<any>({
      path: `/api/v1/admin/billing/subscriptions${query}`,
      method: "GET",
    });
  },

  async getUserSubscription(userId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/billing/subscriptions/${userId}`,
      method: "GET",
    });
  },

  async overrideUserPlan(
    userId: number,
    payload: { plan_id: string; reason?: string },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/billing/subscriptions/${userId}/override`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async grantCredits(
    userId: number,
    payload: { amount: number; reason?: string },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/billing/subscriptions/${userId}/credits`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async listBillingEvents(params?: { limit?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>);
    return await bgRequest<any>({
      path: `/api/v1/admin/billing/events${query}`,
      method: "GET",
    });
  },

  // ── Storage Quotas ──

  async getUserStorageQuota(userId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/storage-quotas/users/${userId}`,
      method: "GET",
    });
  },

  async updateUserStorageQuota(
    userId: number,
    payload: { quota_mb: number | null },
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/storage-quotas/users/${userId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload,
    });
  },

  async getStorageQuotaSummary(params?: {
    limit?: number;
    offset?: number;
  }): Promise<any> {
    const query = buildQuery(params as Record<string, any>);
    return await bgRequest<any>({
      path: `/api/v1/admin/storage-quotas/summary${query}`,
      method: "GET",
    });
  },
};

export type AdminMethods = typeof adminMethods;
