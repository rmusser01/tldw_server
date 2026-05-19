# Admin UI API Reference

All endpoints are accessed through the client-side proxy at `/api/proxy/{path}` which forwards to the backend at `NEXT_PUBLIC_API_URL/api/v1/{path}`. Every `/admin/*` endpoint requires platform admin authentication (JWT or API key via httpOnly cookie).

The API client lives in `lib/api-client.ts` and uses `lib/http.ts` for request plumbing.

---

## 1. Dashboard & Stats

### GET /admin/stats
Returns aggregate dashboard statistics (user count, org count, request totals).

**Response:** `{ users: number, orgs: number, ... }`

### GET /admin/stats/realtime
Returns live session and token counters.

**Response:** `{ active_sessions: number, tokens_today: { prompt: number, completion: number, total: number } }`

### GET /admin/activity
Returns time-series activity data.

**Query:** `days` (int, default 7), `granularity` (`hour` | `day`)

**Response:** Array of activity data points.

---

## 2. User Management

### GET /admin/users
Paginated user list.

**Query:** `page`, `limit`, `search`, `role`, `status`

**Response:** `{ items: UserWithKeyCount[], total, page, limit, pages }`

### POST /admin/users
Create a new user.

**Body:** `{ username, email, password, role, ... }`

### GET /admin/users/{userId}
Get single user details.

**Response:** `User`

### PUT /admin/users/{userId}
Update user fields.

**Body:** `{ username?, email?, role?, is_active?, ... }`

### DELETE /admin/users/{userId}
Delete a user. Privileged action.

**Body:** `{ reason: string, admin_password?: string }`

### POST /admin/users/{userId}/reset-password
Reset a user's password.

**Body:** `{ temporary_password: string, force_password_change?: boolean, reason: string, admin_password?: string }`

### GET /admin/users/{userId}/org-memberships
List organizations the user belongs to.

**Response:** `OrgMembership[]`

### GET /admin/users/{userId}/team-memberships
List teams the user belongs to.

**Response:** `TeamMembership[]`

### GET /admin/users/{userId}/effective-permissions
Resolved permission set for the user.

**Response:** `EffectivePermissionsResponse`

### GET /admin/users/{userId}/sessions
List active sessions for a user.

### DELETE /admin/users/{userId}/sessions/{sessionId}
Revoke a specific session.

**Body:** `{ reason: string, admin_password?: string }`

### POST /admin/users/{userId}/sessions/revoke-all
Revoke all sessions for a user.

**Body:** `{ reason: string, admin_password?: string }`

### GET /admin/users/{userId}/mfa
Get MFA enrollment status.

### POST /admin/users/{userId}/mfa/disable
Disable MFA for a user.

**Body:** `{ reason: string, admin_password?: string }`

### POST /admin/users/{userId}/mfa/require
Set MFA requirement.

**Body:** `{ require_mfa: boolean, reason: string, admin_password?: string }`

### GET /users/me
Get current authenticated user. Not admin-scoped.

### POST /admin/users/invite
Send an invitation email.

**Body:** `{ email: string, role: string, expiry_days?: number }`

### GET /admin/users/invitations
List pending invitations.

**Query:** `status`, `page`, `limit`

### DELETE /admin/users/invitations/{invitationId}
Revoke a pending invitation.

### POST /admin/users/invitations/{invitationId}/resend
Resend an invitation email.

---

## 3. Registration Codes

### GET /admin/registration-settings
Current registration settings (open/closed, code-required, etc.).

### POST /admin/registration-settings
Update registration settings.

### GET /admin/registration-codes
List registration codes.

**Query:** `include_expired` (boolean, default false)

**Response:** `RegistrationCode[]`

### POST /admin/registration-codes
Create a new registration code.

### DELETE /admin/registration-codes/{codeId}
Delete a registration code.

---

## 4. API Key Management

### GET /admin/users/{userId}/api-keys
List API keys for a user.

**Query:** `include_revoked` (boolean)

**Response:** `ApiKey[]`

### POST /admin/users/{userId}/api-keys
Create an API key for a user.

**Response:** `ApiKeyMutationResponse` (includes the raw key, shown once)

### POST /admin/users/{userId}/api-keys/{keyId}/rotate
Rotate an API key. Returns new key value.

**Response:** `ApiKeyMutationResponse`

### DELETE /admin/users/{userId}/api-keys/{keyId}
Revoke an API key.

### GET /admin/api-keys/{keyId}/audit-log
Audit trail for a specific key.

### GET /admin/api-keys/{keyId}/usage
Usage summary for a specific key.

**Response:** `ApiKeyUsageSummary`

### GET /admin/api-keys/usage/top
Top API keys by usage.

**Query:** `limit` (int, default 10)

**Response:** `ApiKeyUsageTopResponse`

---

## 5. Organizations

### GET /admin/orgs
List all organizations.

**Query:** `page`, `limit`, `search`

### GET /orgs/{orgId}
Get a single organization.

**Response:** `Organization`

### POST /admin/orgs
Create an organization.

### PATCH /orgs/{orgId}
Update organization fields.

### DELETE /orgs/{orgId}
Delete an organization.

### GET /admin/orgs/{orgId}/members
List organization members.

**Response:** `OrgMember[]`

### POST /admin/orgs/{orgId}/members
Add a member to an organization.

### DELETE /admin/orgs/{orgId}/members/{userId}
Remove a member.

### PATCH /admin/orgs/{orgId}/members/{userId}
Update a member's role within the org.

### POST /orgs/{orgId}/invite
Invite a user to an organization.

### GET /orgs/{orgId}/invites
List pending org invites.

### GET /admin/orgs/{orgId}/watchlists/settings
Get org-specific watchlist settings.

**Response:** `WatchlistSettings`

### PATCH /admin/orgs/{orgId}/watchlists/settings
Update org watchlist settings.

---

## 6. Teams

### GET /admin/orgs/{orgId}/teams
List teams in an organization.

**Response:** `Team[]`

### POST /admin/orgs/{orgId}/teams
Create a team.

### PATCH /orgs/{orgId}/teams/{teamId}
Update a team.

### DELETE /orgs/{orgId}/teams/{teamId}
Delete a team.

### GET /admin/teams/{teamId}
Get team details.

### GET /admin/teams/{teamId}/members
List team members.

### POST /admin/teams/{teamId}/members
Add a member to a team.

**Body:** `{ email: string, role?: string }` or `{ user_id: number, role?: string }`

### PATCH /admin/teams/{teamId}/members/{memberId}
Update a team member's role.

### DELETE /admin/teams/{teamId}/members/{memberId}
Remove a team member.

---

## 7. Roles & Permissions (RBAC)

### GET /admin/roles
List all roles.

### GET /admin/roles/{roleId}
Get a single role.

### POST /admin/roles
Create a role.

### PUT /admin/roles/{roleId}
Update a role.

### DELETE /admin/roles/{roleId}
Delete a role.

### GET /admin/roles/{roleId}/permissions
List permissions assigned to a role.

### POST /admin/roles/{roleId}/permissions/{permissionId}
Assign a permission to a role.

### DELETE /admin/roles/{roleId}/permissions/{permissionId}
Remove a permission from a role.

### GET /admin/roles/{roleId}/users
List users with a given role.

### GET /admin/permissions
List all permissions.

### GET /admin/permissions/{permId}
Get a single permission.

### POST /admin/permissions
Create a permission.

### PUT /admin/permissions/{permId}
Update a permission.

### DELETE /admin/permissions/{permId}
Delete a permission.

### GET /admin/tool-permissions
List tool-level permissions.

### POST /admin/tool-permissions
Assign a tool permission.

### GET /admin/roles/{roleId}/permissions/tools
List tool permissions for a role.

### POST /admin/roles/{roleId}/permissions/tools/batch
Batch-grant tool permissions.

**Body:** `{ tools: string[] }`

### POST /admin/roles/{roleId}/permissions/tools/batch/revoke
Batch-revoke tool permissions.

**Body:** `{ tools: string[] }`

### POST /admin/roles/{roleId}/permissions/tools/prefix/grant
Grant tool permissions by prefix.

**Body:** `{ prefix: string }`

### POST /admin/roles/{roleId}/permissions/tools/prefix/revoke
Revoke tool permissions by prefix.

**Body:** `{ prefix: string }`

---

## 8. User Permission Overrides

### GET /admin/users/{userId}/overrides
List permission overrides for a user.

### POST /admin/users/{userId}/overrides
Add a permission override.

**Body:** `{ permission_id: number, grant: boolean }`

### DELETE /admin/users/{userId}/overrides/{permissionId}
Remove a permission override.

---

## 9. Rate Limiting

### POST /admin/roles/{roleId}/rate-limits
Set rate limits for a role.

**Body:** `{ resource: string, limit_per_min?: number, burst?: number }`

### DELETE /admin/roles/{roleId}/rate-limits
Clear rate limits for a role.

### GET /admin/users/{userId}/rate-limits
Get rate limits for a user.

### POST /admin/users/{userId}/rate-limits
Set rate limits for a user.

**Body:** `{ resource: string, limit_per_min?: number, burst?: number }`

### GET /admin/rate-limits/summary
Aggregated rate-limit summary.

### GET /admin/rate-limit-events
Recent rate-limit violation events.

---

## 10. Provider Secrets (BYOK)

### GET /admin/users/{userId}/byok-keys
List BYOK keys for a user.

### GET /admin/keys/users/{userId}
Admin view of user BYOK keys.

### POST /admin/users/{userId}/byok-keys
Create a BYOK key for a user.

### DELETE /admin/users/{userId}/byok-keys/{provider}
Delete a user BYOK key.

### GET /admin/orgs/{orgId}/byok-keys
List BYOK keys for an org.

**Response:** `ProviderSecret[]`

### POST /admin/orgs/{orgId}/byok-keys
Create a BYOK key for an org.

### DELETE /admin/orgs/{orgId}/byok-keys/{provider}
Delete an org BYOK key.

### GET /users/keys/openai/oauth/status
OpenAI OAuth connection status. Not admin-scoped.

### POST /users/keys/openai/oauth/authorize
Start OpenAI OAuth flow.

### POST /users/keys/openai/oauth/refresh
Refresh OpenAI OAuth token.

### DELETE /users/keys/openai/oauth
Disconnect OpenAI OAuth.

### POST /users/keys/openai/source
Switch OpenAI credential source.

**Body:** `{ auth_source: "api_key" | "oauth" }`

### GET /admin/keys/shared
List shared provider keys.

**Query:** `scope_type`, `scope_id`, `provider`

### POST /admin/keys/shared
Create a shared provider key.

**Body:** `{ scope_type: string, scope_id: number, provider: string, api_key: string }`

### DELETE /admin/keys/shared/{scopeType}/{scopeId}/{provider}
Delete a shared provider key.

### POST /admin/keys/shared/test
Test a shared provider key.

**Body:** `{ scope_type: string, scope_id: number, provider: string, model?: string }`

### POST /admin/byok/validation-runs
Create a BYOK validation run.

### GET /admin/byok/validation-runs
List BYOK validation runs.

### GET /admin/byok/validation-runs/{runId}
Get a single validation run.

---

## 11. Budgets

### GET /admin/budgets
List org budgets.

### PUT /admin/budgets/{orgId}
Update a budget. Falls back to POST if 404/405.

---

## 12. ACP (Agent Client Protocol)

### GET /admin/acp/sessions
List ACP sessions.

**Query:** `agent_type`, `status`, `limit`, `offset`

### GET /admin/acp/sessions/{sessionId}/usage
Get token usage for a session.

### POST /admin/acp/sessions/{sessionId}/close
Close an active session.

### PATCH /admin/acp/sessions/{sessionId}/budget
Set session token budget.

**Body:** `{ token_budget: number, auto_terminate_at_budget?: boolean }`

### GET /admin/acp/agents
List agent configurations.

### GET /admin/acp/agents/{configId}
Get a single agent config.

### POST /admin/acp/agents
Create an agent config.

### PUT /admin/acp/agents/{configId}
Update an agent config.

### DELETE /admin/acp/agents/{configId}
Delete an agent config.

### GET /admin/acp/agents/metrics
Aggregated metrics per agent type.

**Response:** `{ items: [{ agent_type, session_count, active_sessions, total_prompt_tokens, total_completion_tokens, total_tokens, total_messages, last_used_at }] }`

### GET /admin/acp/permission-policies
List ACP permission policies.

### POST /admin/acp/permission-policies
Create an ACP permission policy.

### PUT /admin/acp/permission-policies/{policyId}
Update an ACP permission policy.

### DELETE /admin/acp/permission-policies/{policyId}
Delete an ACP permission policy.

---

## 13. Monitoring & Alerts

### GET /monitoring/watchlists
List watchlists.

### POST /monitoring/watchlists
Create a watchlist.

### PUT /monitoring/watchlists/{watchlistId}
Update a watchlist.

### DELETE /monitoring/watchlists/{watchlistId}
Delete a watchlist.

### GET /monitoring/alerts
List active alerts.

### GET /admin/monitoring/alerts/history
Alert history.

### GET /admin/monitoring/alert-rules
List alert rules.

### POST /admin/monitoring/alert-rules
Create an alert rule.

### DELETE /admin/monitoring/alert-rules/{ruleId}
Delete an alert rule.

### POST /monitoring/alerts/{alertId}/acknowledge
Acknowledge an alert.

### DELETE /monitoring/alerts/{alertId}
Dismiss an alert.

### POST /admin/monitoring/alerts/{alertIdentity}/assign
Assign an alert to a user.

### POST /admin/monitoring/alerts/{alertIdentity}/snooze
Snooze an alert.

### POST /admin/monitoring/alerts/{alertIdentity}/escalate
Escalate an alert.

### GET /monitoring/metrics
Monitoring metrics.

### GET /monitoring/notifications/settings
Get notification settings.

### PUT /monitoring/notifications/settings
Update notification settings.

### POST /monitoring/notifications/test
Send a test notification.

### GET /monitoring/notifications/recent
Recent notifications.

---

## 14. Health & Metrics

### GET /health
Overall backend health.

### GET /health/metrics
Health metrics.

### GET /health/security
Security health assessment.

**Response:** `SecurityHealthData`

### GET /llm/health
LLM subsystem health.

### GET /audio/health
TTS subsystem health.

### GET /audio/transcriptions/health
STT subsystem health.

### GET /embeddings/health
Embeddings subsystem health.

### GET /rag/health
RAG subsystem health.

### GET /metrics
System metrics (JSON).

### GET /metrics/text
System metrics (Prometheus text format).

---

## 15. System Dependencies

### GET /admin/dependencies
List backend dependencies and their status.

**Response:** `SystemDependenciesResponse`

### GET /admin/dependencies/{name}/uptime
Uptime stats for a dependency.

**Query:** `days` (int, default 7)

**Response:** `DependencyUptimeStats`

---

## 16. Incidents

### GET /admin/incidents
List incidents.

**Response:** `IncidentsResponse`

### POST /admin/incidents
Create an incident.

**Response:** `IncidentItem`

### PATCH /admin/incidents/{incidentId}
Update an incident.

**Response:** `IncidentItem`

### POST /admin/incidents/{incidentId}/events
Add an event to an incident timeline.

**Response:** `IncidentItem`

### DELETE /admin/incidents/{incidentId}
Delete an incident.

### POST /admin/incidents/{incidentId}/notify
Notify stakeholders about an incident.

**Body:** `{ recipients: string[], message?: string }`

**Response:** `IncidentNotifyResponse`

### GET /admin/incidents/metrics/sla
SLA metrics across incidents.

**Response:** `{ total_incidents, resolved_count, acknowledged_count, avg_mtta_minutes, avg_mttr_minutes, p95_mtta_minutes, p95_mttr_minutes }`

---

## 17. Webhooks

### GET /admin/webhooks
List webhooks.

**Response:** `WebhookListResponse`

### POST /admin/webhooks
Create a webhook.

**Body:** `{ url: string, events: string[], enabled?: boolean }`

**Response:** `WebhookCreateResponse`

### PATCH /admin/webhooks/{webhookId}
Update a webhook.

**Body:** `{ url?: string, events?: string[], enabled?: boolean }`

**Response:** `WebhookItem`

### DELETE /admin/webhooks/{webhookId}
Delete a webhook.

### GET /admin/webhooks/{webhookId}/deliveries
List deliveries for a webhook.

**Query:** `limit` (int, default 50)

**Response:** `WebhookDeliveryListResponse`

### POST /admin/webhooks/{webhookId}/test
Send a test delivery.

**Response:** `WebhookDeliveryItem`

---

## 18. Email Delivery Log

### GET /admin/email/deliveries
List email delivery records.

**Query:** `limit`, `offset`, `status`

**Response:** `EmailDeliveryListResponse`

---

## 19. Audit Logs

### GET /admin/audit-log
Paginated audit log entries.

**Query:** `page`, `limit`, `action`, `user_id`, `resource`

**Response:** `{ entries: AuditLog[], total, limit, offset }`

---

## 20. Errors & Rate Limits

### GET /admin/errors/breakdown
Error breakdown by category/endpoint.

### GET /admin/rate-limits/summary
Rate-limit summary across the system.

---

## 21. Configuration

### GET /setup/status
Setup wizard status.

### GET /setup/config
Current configuration.

### POST /setup/config
Update configuration.

### GET /admin/config/effective
Effective (merged) configuration.

### GET /admin/config/profiles
List saved config profiles.

### POST /admin/config/profiles/snapshot
Snapshot current config as a named profile.

**Body:** `{ name: string, description?: string }`

### GET /admin/config/profiles/{name}
Get a config profile.

### POST /admin/config/profiles/{name}/restore
Restore a saved profile.

### DELETE /admin/config/profiles/{name}
Delete a saved profile.

### PUT /admin/config/sections/{section}
Update a configuration section.

**Body:** `{ values: Record<string, string> }`

### GET /admin/config/export
Export full configuration.

### POST /admin/config/import
Import configuration sections.

**Body:** `{ sections: Record<string, Record<string, string>> }`

---

## 22. LLM Providers

### GET /llm/providers
List configured LLM providers. Not admin-scoped.

### GET /admin/llm/providers/overrides
List provider overrides.

### GET /admin/llm/providers/overrides/{provider}
Get override for a provider.

### PUT /admin/llm/providers/overrides/{provider}
Set override for a provider.

### DELETE /admin/llm/providers/overrides/{provider}
Remove override for a provider.

### POST /admin/llm/providers/test
Test a provider connection.

### GET /admin/llm/providers/health
Health status of all LLM providers.

---

## 23. System Ops

### GET /admin/system/logs
Paginated system logs.

### GET /admin/maintenance
Get maintenance mode status.

### PUT /admin/maintenance
Enable or disable maintenance mode.

### GET /admin/feature-flags
List feature flags.

### PUT /admin/feature-flags/{flagKey}
Create or update a feature flag.

### DELETE /admin/feature-flags/{flagKey}
Delete a feature flag.

---

## 24. Data Ops

### GET /admin/backups
List backups.

**Response:** `BackupsResponse`

### POST /admin/backups
Create a backup.

### POST /admin/backups/{backupId}/restore
Restore from a backup.

### GET /admin/backup-schedules
List backup schedules.

**Response:** `BackupScheduleListResponse`

### POST /admin/backup-schedules
Create a backup schedule.

### PATCH /admin/backup-schedules/{scheduleId}
Update a backup schedule.

### POST /admin/backup-schedules/{scheduleId}/pause
Pause a schedule.

### POST /admin/backup-schedules/{scheduleId}/resume
Resume a schedule.

### DELETE /admin/backup-schedules/{scheduleId}
Delete a schedule.

### GET /admin/retention-policies
List retention policies.

**Response:** `RetentionPoliciesResponse`

### POST /admin/retention-policies/{policyKey}/preview
Preview impact of a retention policy change.

**Response:** `RetentionPolicyPreviewResponse`

### PUT /admin/retention-policies/{policyKey}
Update a retention policy.

### POST /admin/data-subject-requests/preview
Preview a data subject request.

### GET /admin/data-subject-requests
List data subject requests.

### POST /admin/data-subject-requests
Create a data subject request (GDPR right-to-erasure, export, etc.).

---

## 25. Jobs

### GET /jobs/list
List background jobs.

**Query:** `status`, `domain`, `queue`, `limit`, `offset`

### GET /jobs/{jobId}
Get job details.

### GET /jobs/stats
Job queue statistics.

### GET /jobs/stale
List stale/stuck jobs.

### POST /jobs/batch/cancel
Cancel jobs in bulk.

**Header:** `X-Confirm: true`

### POST /jobs/retry-now
Retry failed jobs immediately.

### POST /jobs/batch/requeue_quarantined
Requeue quarantined jobs.

**Header:** `X-Confirm: true`

### GET /admin/jobs/sla/policies
List job SLA policies.

### POST /admin/jobs/sla/policy
Create a job SLA policy.

### DELETE /admin/jobs/sla/policy
Delete a job SLA policy.

**Body:** `{ domain: string, queue: string, job_type: string }`

### GET /admin/jobs/sla/breaches
List SLA breaches.

### GET /admin/jobs/{jobId}/attachments
List attachments for a job.

### POST /admin/jobs/{jobId}/attachments
Upload an attachment (multipart/form-data).

### POST /admin/jobs/crypto/rotate
Rotate job encryption keys.

---

## 26. Usage Analytics

### GET /admin/usage/daily
Daily usage data.

### GET /admin/usage/top
Top users by usage.

### GET /admin/llm-usage/summary
LLM usage summary.

### GET /admin/llm-usage
Detailed LLM usage records.

### GET /admin/llm-usage/top-spenders
Top LLM spenders.

### GET /admin/router-analytics/{subpath}
Router analytics. Subpaths: `status`, `status/breakdowns`, `quota`, `providers`, `access`, `network`, `models`, `conversations`, `log`, `meta`.

---

## 27. Resource Governor

### GET /resource-governor/policy
Get resource governor policy.

**Query:** `include_ids` (boolean)

### POST /resource-governor/policy/simulate
Simulate a policy change.

### PUT /resource-governor/policy
Update the resource governor policy.

### DELETE /resource-governor/policy/{policyId}
Delete a resource governor policy.

---

## 28. Billing & Plans

### GET /admin/billing/analytics
Billing analytics dashboard data.

**Response:** `BillingAnalytics`

### GET /billing/plans
List billing plans.

**Response:** `Plan[]`

### GET /billing/plans/{planId}
Get a single plan.

### POST /billing/plans
Create a plan.

**Body:** `{ name, tier, monthly_price_cents, included_token_credits, overage_rate_per_1k_tokens_cents, stripe_product_id?, stripe_price_id?, features?, is_default? }`

### PUT /billing/plans/{planId}
Update a plan.

### DELETE /billing/plans/{planId}
Delete a plan.

### GET /billing/subscriptions
List all subscriptions.

### GET /billing/orgs/{orgId}/subscription
Get an org's subscription.

### POST /billing/orgs/{orgId}/subscription
Create a subscription (may return Stripe checkout URL).

**Body:** `{ plan_id: string, trial_days?: number }`

### PUT /billing/orgs/{orgId}/subscription
Change subscription plan.

**Body:** `{ plan_id: string }`

### DELETE /billing/orgs/{orgId}/subscription
Cancel a subscription.

### GET /billing/orgs/{orgId}/usage
Org usage summary.

**Query:** `period`

**Response:** `OrgUsageSummary`

### GET /billing/orgs/{orgId}/invoices
List org invoices.

**Response:** `Invoice[]`

### GET /billing/feature-registry
List feature registry entries.

**Response:** `FeatureRegistryEntry[]`

### PUT /billing/feature-registry
Update the feature registry.

### POST /billing/onboarding
Create an onboarding session (org + subscription).

**Body:** `{ org_name, org_slug, plan_id, owner_email? }`

---

## 29. Compliance

### GET /admin/compliance/posture
Overall compliance posture.

**Response:** `CompliancePosture`

### GET /admin/compliance/report-schedules
List compliance report schedules.

**Response:** `{ items: ComplianceReportSchedule[], total: number }`

### POST /admin/compliance/report-schedules
Create a report schedule.

**Body:** `{ frequency: string, recipients: string[], format: string, enabled: boolean }`

### PATCH /admin/compliance/report-schedules/{scheduleId}
Update a report schedule.

### DELETE /admin/compliance/report-schedules/{scheduleId}
Delete a report schedule.

### POST /admin/compliance/report-schedules/{scheduleId}/send-now
Send a scheduled report immediately.

**Response:** `{ sent_count, total_recipients, errors }`

---

## 30. Security

### GET /health/security
Security health data.

**Response:** `SecurityHealthData`

### GET /admin/security/alert-status
Security alert status.

**Response:** `SecurityAlertStatus`

---

## 31. Email Digest Preferences

### GET /admin/digest/preference
Get admin digest preference.

**Response:** `DigestPreference`

### PUT /admin/digest/preference
Set admin digest preference.

**Body:** `{ email: string, frequency: string }`

---

## 32. MCP Servers

### GET /mcp/status
MCP server status.

### GET /mcp/metrics
MCP metrics.

### GET /mcp/tools
List available MCP tools.

### GET /mcp/modules
List MCP modules.

### GET /mcp/modules/health
MCP module health.

### GET /mcp/health
Overall MCP health.

---

## 33. Voice Commands & Assistant

### GET /voice/commands
List voice commands.

### GET /voice/commands/{commandId}
Get a voice command.

### POST /voice/commands
Create a voice command.

### PUT /voice/commands/{commandId}
Update a voice command.

### DELETE /voice/commands/{commandId}
Delete a voice command.

### POST /voice/commands/{commandId}/toggle
Enable/disable a voice command.

**Body:** `{ enabled: boolean }`

### POST /voice/commands/{commandId}/validate
Validate a voice command configuration.

### GET /voice/commands/{commandId}/usage
Usage stats for a command.

**Query:** `days`

### GET /voice/sessions
List voice sessions.

### GET /voice/sessions/{sessionId}
Get a voice session.

### DELETE /voice/sessions/{sessionId}
Delete a voice session.

### GET /voice/analytics
Voice analytics summary.

**Query:** `days`, `user_id`

**Response:** `VoiceAnalyticsSummary`

### GET /voice/workflows/templates
List workflow templates.

---

## 34. Virtual API Keys

### GET /admin/users/{userId}/virtual-keys
List virtual keys for a user.

### POST /admin/users/{userId}/virtual-keys
Create a virtual key.

**Body:** `{ name: string, scopes: string[] }`

### DELETE /admin/users/{userId}/virtual-keys/{keyId}
Delete a virtual key.

---

## 35. Cleanup & Maintenance

### GET /admin/cleanup-settings
Get cleanup settings.

### POST /admin/cleanup-settings
Update cleanup settings.

### GET /admin/notes/title-settings
Get notes title auto-generation settings.

### POST /admin/notes/title-settings
Update notes title settings.

### POST /admin/kanban/fts-maintenance
Run Kanban FTS index maintenance.

### POST /admin/maintenance/rotation-runs
Create a credential/key rotation run.

### GET /admin/maintenance/rotation-runs
List rotation runs.

### GET /admin/maintenance/rotation-runs/{runId}
Get a rotation run.

---

## 36. Debug Tools

### GET /authnz/debug/api-key-id
Resolve API key details. Pass key via `X-API-KEY` header, or use query params `key_id` or `user_id`.

### GET /authnz/debug/budget-summary
Budget summary for an API key. Pass key via `X-API-KEY` header.

### GET /authnz/debug/permissions
Resolve permissions for a user.

**Query:** `user_id`

### POST /authnz/debug/validate-token
Validate a JWT token.

**Body:** `{ token: string }`
