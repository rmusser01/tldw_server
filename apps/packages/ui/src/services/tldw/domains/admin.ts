import { bgRequest } from "@/services/background-proxy"
import { buildQuery } from "../client-utils"
import type {
  AdminUserListResponse,
  AdminUserUpdateRequest,
  AdminRole,
} from "../TldwApiClient"

export const adminMethods = {
  // ── Admin Users & Roles ──

  async listAdminUsers(params?: {
    page?: number
    limit?: number
    role?: string
    is_active?: boolean
    search?: string
  }): Promise<AdminUserListResponse> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<AdminUserListResponse>({
      path: `/api/v1/admin/users${query}`,
      method: "GET"
    })
  },

  async updateAdminUser(
    userId: number,
    payload: AdminUserUpdateRequest
  ): Promise<{ message: string }> {
    return await bgRequest<{ message: string }>({
      path: `/api/v1/admin/users/${userId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async listAdminRoles(): Promise<AdminRole[]> {
    return await bgRequest<AdminRole[]>({
      path: "/api/v1/admin/roles",
      method: "GET"
    })
  },

  async createAdminRole(
    name: string,
    description?: string
  ): Promise<AdminRole> {
    return await bgRequest<AdminRole>({
      path: "/api/v1/admin/roles",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { name, description }
    })
  },

  async deleteAdminRole(roleId: number): Promise<{ message: string }> {
    return await bgRequest<{ message: string }>({
      path: `/api/v1/admin/roles/${roleId}`,
      method: "DELETE"
    })
  },

  // ── Admin API Key Management ──

  async listUserApiKeys(userId: number): Promise<any[]> {
    return await bgRequest<any[]>({
      path: `/api/v1/admin/users/${userId}/api-keys`,
      method: "GET"
    })
  },

  async createUserApiKey(userId: number, payload: { name?: string; rate_limit?: number; allowed_ips?: string[] }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/api-keys`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async revokeUserApiKey(userId: number, keyId: number): Promise<{ message: string }> {
    return await bgRequest<{ message: string }>({
      path: `/api/v1/admin/users/${userId}/api-keys/${keyId}`,
      method: "DELETE"
    })
  },

  async updateUserApiKey(userId: number, keyId: number, payload: { rate_limit?: number; allowed_ips?: string[] }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/api-keys/${keyId}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async rotateUserApiKey(userId: number, keyId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/api-keys/${keyId}/rotate`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async listUserVirtualKeys(userId: number): Promise<any[]> {
    return await bgRequest<any[]>({
      path: `/api/v1/admin/users/${userId}/virtual-keys`,
      method: "GET"
    })
  },

  async getApiKeyAuditLog(keyId: number): Promise<any[]> {
    return await bgRequest<any[]>({
      path: `/api/v1/admin/api-keys/${keyId}/audit-log`,
      method: "GET"
    })
  },

  // ── Admin Maintenance ──

  async getMaintenanceState(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/admin/maintenance", method: "GET" })
  },

  async updateMaintenanceState(payload: { enabled?: boolean; message?: string; allowlist?: string[] }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/maintenance",
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async listFeatureFlags(): Promise<any[]> {
    return await bgRequest<any[]>({ path: "/api/v1/admin/feature-flags", method: "GET" })
  },

  async updateFeatureFlag(flagKey: string, payload: { enabled: boolean }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/feature-flags/${encodeURIComponent(flagKey)}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async deleteFeatureFlag(flagKey: string): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/feature-flags/${encodeURIComponent(flagKey)}`,
      method: "DELETE"
    })
  },

  async listIncidents(): Promise<any[]> {
    return await bgRequest<any[]>({ path: "/api/v1/admin/incidents", method: "GET" })
  },

  async createIncident(payload: { title: string; severity?: string; description?: string }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/incidents",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async updateIncident(incidentId: number, payload: { status?: string; severity?: string; description?: string }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/incidents/${incidentId}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async deleteIncident(incidentId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/incidents/${incidentId}`,
      method: "DELETE"
    })
  },

  async listRotationRuns(): Promise<any[]> {
    return await bgRequest<any[]>({ path: "/api/v1/admin/maintenance/rotation-runs", method: "GET" })
  },

  async createRotationRun(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/maintenance/rotation-runs",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  // ── Admin Monitoring & Alerting ──

  async listAlertRules(): Promise<any[]> {
    return await bgRequest<any[]>({ path: "/api/v1/admin/monitoring/alert-rules", method: "GET" })
  },

  async createAlertRule(payload: {
    metric: string; operator: string; threshold: number;
    duration_minutes: number; severity: string; enabled?: boolean
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/monitoring/alert-rules",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async deleteAlertRule(ruleId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/monitoring/alert-rules/${ruleId}`,
      method: "DELETE"
    })
  },

  async assignAlert(alertIdentity: string, payload: { assigned_to_user_id?: number | null }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/monitoring/alerts/${encodeURIComponent(alertIdentity)}/assign`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async snoozeAlert(alertIdentity: string, payload: { until: string }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/monitoring/alerts/${encodeURIComponent(alertIdentity)}/snooze`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async escalateAlert(alertIdentity: string): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/monitoring/alerts/${encodeURIComponent(alertIdentity)}/escalate`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async listAlertHistory(): Promise<any[]> {
    return await bgRequest<any[]>({ path: "/api/v1/admin/monitoring/alerts/history", method: "GET" })
  },

  async getSecurityAlertStatus(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/admin/security/alert-status", method: "GET" })
  },

  async getDashboardActivity(params?: { days?: number; granularity?: string }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/activity${query}`, method: "GET" })
  },

  // ── Admin Usage Analytics ──

  async getDailyUsage(params?: { start_date?: string; end_date?: string }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/usage/daily${query}`, method: "GET" })
  },

  async getTopUsage(params?: { metric?: string; limit?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/usage/top${query}`, method: "GET" })
  },

  async exportDailyUsageCsv(): Promise<string> {
    return await bgRequest<string>({ path: "/api/v1/admin/usage/daily/export.csv", method: "GET" })
  },

  async exportTopUsageCsv(): Promise<string> {
    return await bgRequest<string>({ path: "/api/v1/admin/usage/top/export.csv", method: "GET" })
  },

  async getLlmUsage(params?: { provider?: string; model?: string; limit?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/llm-usage${query}`, method: "GET" })
  },

  async getLlmUsageSummary(params?: { group_by?: string }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/llm-usage/summary${query}`, method: "GET" })
  },

  async getLlmTopSpenders(params?: { limit?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/llm-usage/top-spenders${query}`, method: "GET" })
  },

  async getRouterAnalyticsStatus(params?: { range?: string }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/router-analytics/status${query}`, method: "GET" })
  },

  async getRouterAnalyticsProviders(params?: { range?: string }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/router-analytics/providers${query}`, method: "GET" })
  },

  // ── Admin Organizations & Teams ──

  async createOrg(payload: { name: string; slug?: string }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/orgs",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async listOrgs(params?: { search?: string; limit?: number; offset?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/orgs${query}`, method: "GET" })
  },

  async listOrgMembers(orgId: number, params?: { role?: string; status?: string }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/orgs/${orgId}/members${query}`, method: "GET" })
  },

  async addOrgMember(orgId: number, payload: { user_id: number; role?: string }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/orgs/${orgId}/members`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async removeOrgMember(orgId: number, userId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/orgs/${orgId}/members/${userId}`,
      method: "DELETE"
    })
  },

  async updateOrgMemberRole(orgId: number, userId: number, payload: { role: string }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/orgs/${orgId}/members/${userId}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async createTeam(orgId: number, payload: { name: string }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/orgs/${orgId}/teams`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async listTeams(orgId: number): Promise<any> {
    return await bgRequest<any>({ path: `/api/v1/admin/orgs/${orgId}/teams`, method: "GET" })
  },

  async listTeamMembers(teamId: number): Promise<any> {
    return await bgRequest<any>({ path: `/api/v1/admin/teams/${teamId}/members`, method: "GET" })
  },

  async addTeamMember(teamId: number, payload: { user_id: number; role?: string }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/teams/${teamId}/members`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async removeTeamMember(teamId: number, userId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/teams/${teamId}/members/${userId}`,
      method: "DELETE"
    })
  },

  async updateTeamMemberRole(teamId: number, userId: number, payload: { role: string }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/teams/${teamId}/members/${userId}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  // ── Admin Data Operations ──

  async listBackups(params?: { dataset?: string; user_id?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/backups${query}`, method: "GET" })
  },

  async createBackup(payload: { dataset: string; user_id?: number }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/backups",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async restoreBackup(backupId: string): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/backups/${encodeURIComponent(backupId)}/restore`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async listBackupSchedules(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/admin/backup-schedules", method: "GET" })
  },

  async createBackupSchedule(payload: { dataset: string; cron?: string; retention_days?: number }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/backup-schedules",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async deleteBackupSchedule(scheduleId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/backup-schedules/${scheduleId}`,
      method: "DELETE"
    })
  },

  async previewDsr(payload: { requester_identifier: string; request_type?: string; categories?: string[] }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/data-subject-requests/preview",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async createDsr(payload: {
    requester_identifier: string; request_type: string;
    categories?: string[]; client_request_id?: string; notes?: string
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/data-subject-requests",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async listDsrs(params?: { limit?: number; offset?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/data-subject-requests${query}`, method: "GET" })
  },

  async executeDsr(requestId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/data-subject-requests/${requestId}/execute`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async listRetentionPolicies(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/admin/retention-policies", method: "GET" })
  },

  async updateRetentionPolicy(policyKey: string, payload: { retention_days: number }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/retention-policies/${encodeURIComponent(policyKey)}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async listBundles(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/admin/backups/bundles", method: "GET" })
  },

  async createBundle(payload: { datasets: string[] }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/admin/backups/bundles",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async deleteBundle(bundleId: string): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/backups/bundles/${encodeURIComponent(bundleId)}`,
      method: "DELETE"
    })
  },

  // ── Admin RBAC & Permissions ──

  async listPermissions(): Promise<any[]> {
    return await bgRequest<any[]>({ path: "/api/v1/admin/permissions", method: "GET" })
  },

  async listPermissionCategories(): Promise<any[]> {
    return await bgRequest<any[]>({ path: "/api/v1/admin/permissions/categories", method: "GET" })
  },

  async getRolePermissionMatrix(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/admin/roles/matrix-boolean", method: "GET" })
  },

  async listRolePermissions(roleId: number): Promise<any[]> {
    return await bgRequest<any[]>({ path: `/api/v1/admin/roles/${roleId}/permissions`, method: "GET" })
  },

  async grantRolePermission(roleId: number, permissionId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/roles/${roleId}/permissions/${permissionId}`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async revokeRolePermission(roleId: number, permissionId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/roles/${roleId}/permissions/${permissionId}`,
      method: "DELETE"
    })
  },

  async listUserRoles(userId: number): Promise<any[]> {
    return await bgRequest<any[]>({ path: `/api/v1/admin/users/${userId}/roles`, method: "GET" })
  },

  async assignUserRole(userId: number, roleId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/roles/${roleId}`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {}
    })
  },

  async removeUserRole(userId: number, roleId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/roles/${roleId}`,
      method: "DELETE"
    })
  },

  async listUserOverrides(userId: number): Promise<any[]> {
    return await bgRequest<any[]>({ path: `/api/v1/admin/users/${userId}/overrides`, method: "GET" })
  },

  async getUserEffectivePermissions(userId: number): Promise<any[]> {
    return await bgRequest<any[]>({ path: `/api/v1/admin/users/${userId}/effective-permissions`, method: "GET" })
  },

  async addUserOverride(userId: number, payload: { permission_id: number; effect: string }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/overrides`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async deleteUserOverride(userId: number, permissionId: number): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/users/${userId}/overrides/${permissionId}`,
      method: "DELETE"
    })
  },

  // ── Admin Billing ──

  async getBillingOverview(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/admin/billing/overview", method: "GET" })
  },

  async listAllSubscriptions(params?: { status?: string; limit?: number; offset?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/billing/subscriptions${query}`, method: "GET" })
  },

  async getUserSubscription(userId: number): Promise<any> {
    return await bgRequest<any>({ path: `/api/v1/admin/billing/subscriptions/${userId}`, method: "GET" })
  },

  async overrideUserPlan(userId: number, payload: { plan_id: string; reason?: string }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/billing/subscriptions/${userId}/override`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async grantCredits(userId: number, payload: { amount: number; reason?: string }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/billing/subscriptions/${userId}/credits`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async listBillingEvents(params?: { limit?: number }): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({ path: `/api/v1/admin/billing/events${query}`, method: "GET" })
  },

  // ── Storage Quotas ──

  async getUserStorageQuota(userId: number): Promise<any> {
    return await bgRequest<any>({ path: `/api/v1/admin/storage-quotas/users/${userId}`, method: "GET" })
  },

  async updateUserStorageQuota(userId: number, payload: { quota_mb: number }): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/admin/storage-quotas/users/${userId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async getStorageQuotaSummary(): Promise<any> {
    return await bgRequest<any>({ path: "/api/v1/admin/storage-quotas/summary", method: "GET" })
  },
}

export type AdminMethods = typeof adminMethods
