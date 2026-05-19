'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useToast } from '@/components/ui/toast';
import { useOrgContext } from '@/components/OrgContextSwitcher';
import { AlertsBanner } from '@/components/dashboard/AlertsBanner';
import { CreateOrganizationDialog } from '@/components/dashboard/CreateOrganizationDialog';
import { CreateUserDialog } from '@/components/dashboard/CreateUserDialog';
import { DashboardHeader } from '@/components/dashboard/DashboardHeader';
import { StatsGrid, type RealtimeStats } from '@/components/dashboard/StatsGrid';
import { ActivitySection } from '@/components/dashboard/ActivitySection';
import { RecentActivityCard } from '@/components/dashboard/RecentActivityCard';
import { QuickActionsCard } from '@/components/dashboard/QuickActionsCard';
import {
  Building2, CreditCard, KeyRound, Settings, UserPlus, ShieldAlert
} from 'lucide-react';
import { api } from '@/lib/api-client';
import {
  fetchDashboardBillingStats,
  isBillingEnabled,
} from '@/lib/billing';
import { AuditLog, LLMProvider, Organization, RegistrationCode, type SecurityHealthData, User } from '@/types';
import { buildDashboardUIStats, type DashboardUIStats } from '@/lib/dashboard';
import {
  aggregateUsageDailyRows,
  buildDashboardOperationalKpis,
  DEFAULT_DASHBOARD_OPERATIONAL_KPIS,
  extractLlmDailyCostRows,
  type DashboardOperationalKpis,
  type JobSnapshot,
} from '@/lib/dashboard-kpis';
import { resolveSecurityHealth } from '@/lib/security-health';
import {
  buildDashboardSystemHealth,
  DEFAULT_DASHBOARD_SYSTEM_HEALTH,
  type DashboardSystemHealth,
} from '@/lib/dashboard-health';
import {
  buildDashboardActivityChartData,
  getDashboardActivityQuery,
  mergeOverlayData,
  resolveDashboardActivityPoints,
  type DailyOverlayRow,
  type DashboardActivityPoint,
  type DashboardActivityRange,
} from '@/lib/dashboard-activity';
import {
  buildDashboardUptimeSummary,
  DEFAULT_DASHBOARD_UPTIME_SUMMARY,
  type DashboardUptimeSummary,
} from '@/lib/dashboard-uptime';
import Link from 'next/link';
import { logger } from '@/lib/logger';

type ServerStatusState = 'online' | 'degraded' | 'offline' | 'unknown';

interface DashboardAlert {
  id: string | number;
  message?: string;
  severity?: 'info' | 'warning' | 'error' | 'critical';
  acknowledged: boolean;
  created_at?: string;
}

type ProviderMap = Record<string, Partial<Omit<LLMProvider, 'name'>>>;
const DEFAULT_ACTIVITY_RANGE: DashboardActivityRange = '7d';

const processUsers = (result: PromiseSettledResult<unknown>): User[] => {
  if (result.status !== 'fulfilled') {
    return [];
  }
  return Array.isArray(result.value) ? result.value : [];
};

const processOrganizations = (result: PromiseSettledResult<unknown>): Organization[] => {
  if (result.status !== 'fulfilled') {
    return [];
  }
  return Array.isArray(result.value) ? result.value : [];
};

const processProviders = (result: PromiseSettledResult<unknown>) => {
  const rawProviders: ProviderMap | LLMProvider[] =
    result.status === 'fulfilled' && result.value ? result.value as ProviderMap | LLMProvider[] : {};
  const providerList: LLMProvider[] = Array.isArray(rawProviders)
    ? rawProviders.map((provider) => ({
        name: provider.name,
        enabled: provider.enabled ?? true,
        models: provider.models,
      }))
    : Object.entries(rawProviders).map(([name, value]) => ({
        name,
        enabled: value.enabled ?? true,
        models: value.models,
      }));
  const enabledProviders = providerList.filter((provider) => provider.enabled !== false);
  return { providerList, enabledProviders };
};

const processAuditLogs = (result: PromiseSettledResult<unknown>): AuditLog[] => {
  if (result.status !== 'fulfilled') {
    return [];
  }
  if (Array.isArray(result.value)) {
    return result.value;
  }
  const entries = (result.value as { entries?: AuditLog[] } | null)?.entries;
  if (Array.isArray(entries)) {
    return entries;
  }
  const items = (result.value as { items?: AuditLog[] } | null)?.items;
  return Array.isArray(items) ? items : [];
};

const processAlerts = (result: PromiseSettledResult<unknown>): DashboardAlert[] => {
  if (result.status !== 'fulfilled') {
    return [];
  }
  return Array.isArray(result.value) ? result.value : [];
};

const processRegistrationCodes = (result: PromiseSettledResult<unknown>): RegistrationCode[] => {
  if (result.status !== 'fulfilled') {
    return [];
  }
  return Array.isArray(result.value) ? result.value : [];
};

const computeUserStats = (users: User[]) => ({
  activeUsers: users.filter((user) => user.is_active).length,
  totalStorage: users.reduce((acc, user) => acc + (user.storage_used_mb || 0), 0),
  totalQuota: users.reduce((acc, user) => acc + (user.storage_quota_mb || 0), 0),
});

export default function DashboardPage() {
  const router = useRouter();
  const { selectedOrg } = useOrgContext();
  const { success, error: showError } = useToast();
  const [stats, setStats] = useState<DashboardUIStats>({
    users: 0,
    activeUsers: 0,
    organizations: 0,
    teams: 0,
    apiKeys: 0,
    activeApiKeys: 0,
    providers: 0,
    enabledProviders: 0,
    storageUsedMb: 0,
    storageQuotaMb: 1000,
    activeAcpSessions: null,
    tokensToday: null,
    mcpInvocationsToday: null,
  });
  const [operationalKpis, setOperationalKpis] = useState<DashboardOperationalKpis>(
    DEFAULT_DASHBOARD_OPERATIONAL_KPIS
  );
  const previousJobsSnapshotRef = useRef<JobSnapshot | null>(null);
  const dashboardLoadInFlightRef = useRef(false);
  const dashboardLoadRequestIdRef = useRef(0);
  const [recentActivity, setRecentActivity] = useState<AuditLog[]>([]);
  const [alerts, setAlerts] = useState<DashboardAlert[]>([]);
  const [systemHealth, setSystemHealth] = useState<DashboardSystemHealth>(
    DEFAULT_DASHBOARD_SYSTEM_HEALTH
  );
  const [serverStatus, setServerStatus] = useState<{
    state: ServerStatusState;
    checkedAt?: string;
  }>({ state: 'unknown' });
  const [uptimeSummary, setUptimeSummary] = useState<DashboardUptimeSummary>(
    DEFAULT_DASHBOARD_UPTIME_SUMMARY
  );
  const [activityRange, setActivityRange] = useState<DashboardActivityRange>(DEFAULT_ACTIVITY_RANGE);
  const [activityData, setActivityData] = useState<DashboardActivityPoint[]>(
    resolveDashboardActivityPoints(
      { status: 'rejected', reason: new Error('initial') },
      DEFAULT_ACTIVITY_RANGE
    )
  );
  const [activityOverlayRows, setActivityOverlayRows] = useState<DailyOverlayRow[]>([]);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [registrationCodes, setRegistrationCodes] = useState<RegistrationCode[]>([]);
  const [showCreateUserDialog, setShowCreateUserDialog] = useState(false);
  const [createUserForm, setCreateUserForm] = useState({
    username: '',
    email: '',
    password: '',
    role: 'user',
    is_active: true,
    is_verified: true,
  });
  const [createUserError, setCreateUserError] = useState('');
  const [creatingUser, setCreatingUser] = useState(false);
  const [showOrgDialog, setShowOrgDialog] = useState(false);
  const [orgForm, setOrgForm] = useState({ name: '', slug: '' });
  const [orgError, setOrgError] = useState('');
  const [creatingOrg, setCreatingOrg] = useState(false);
  const [loading, setLoading] = useState(true);
  const loadingRef = useRef(false); // Ref-based guard for auto-refresh overlap
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(true);
  const [securityHealth, setSecurityHealth] = useState<SecurityHealthData | null>(null);
  const [securityHealthError, setSecurityHealthError] = useState('');
  const [realtimeStats, setRealtimeStats] = useState<RealtimeStats | null>(null);
  const [billingStats, setBillingStats] = useState<{
    active_subscriptions: number;
    past_due_count: number;
    plan_distribution: Record<string, number>;
  } | null>(null);

  const loadDashboardData = useCallback(async () => {
    if (dashboardLoadInFlightRef.current) {
      return;
    }

    const requestId = ++dashboardLoadRequestIdRef.current;
    dashboardLoadInFlightRef.current = true;

    try {
      loadingRef.current = true;
      setLoading(true);
      setError(null);
      setSecurityHealthError('');

      const orgParams = selectedOrg ? { org_id: String(selectedOrg.id) } : undefined;
      const auditParams: Record<string, string> = {
        limit: '10',
        ...(selectedOrg ? { org_id: String(selectedOrg.id) } : {}),
      };
      const usageDailyParams: Record<string, string> = {
        limit: '200',
        ...(selectedOrg ? { org_id: String(selectedOrg.id) } : {}),
      };
      const llmUsageSummaryParams: Record<string, string> = {
        group_by: 'day',
        ...(selectedOrg ? { org_id: String(selectedOrg.id) } : {}),
      };
      const activityQuery = getDashboardActivityQuery(activityRange);

      // Fetch all dashboard data in parallel
      const [
        statsResult,
        usersResult,
        orgsResult,
        providersResult,
        auditResult,
        alertsResult,
        activityResult,
        usageDailyResult,
        llmUsageSummaryResult,
        jobsStatsResult,
        metricsTextResult,
        registrationCodesResult,
        healthResult,
        llmHealthResult,
        ragHealthResult,
        ttsHealthResult,
        sttHealthResult,
        embeddingsHealthResult,
        securityHealthResult,
        incidentsResult,
        realtimeStatsResult,
      ] = await Promise.allSettled([
        api.getDashboardStats(),
        api.getUsers(orgParams),
        api.getOrganizations(),
        api.getLLMProviders(),
        api.getAuditLogs(auditParams),
        api.getAlerts(),
        api.getDashboardActivity(activityQuery.days, {
          granularity: activityQuery.granularity,
        }),
        api.getUsageDaily(usageDailyParams),
        api.getLlmUsageSummary(llmUsageSummaryParams),
        api.getJobsStats(),
        api.getMetricsText(),
        api.getRegistrationCodes(),
        api.getHealth(),
        api.getLlmHealth(),
        api.getRagHealth(),
        api.getTtsHealth(),
        api.getSttHealth(),
        api.getEmbeddingsHealth(),
        api.getSecurityHealth(),
        api.getIncidents({ limit: '200' }),
        api.getRealtimeStats(),
      ]);

      const users = processUsers(usersResult);
      const orgs = processOrganizations(orgsResult);
      setOrganizations(orgs);

      const { providerList, enabledProviders } = processProviders(providersResult);

      const logs = processAuditLogs(auditResult);
      setRecentActivity(logs.slice(0, 10));

      const alertsList = processAlerts(alertsResult);
      setAlerts(alertsList.filter((alert) => !alert.acknowledged));

      setActivityData(resolveDashboardActivityPoints(activityResult, activityRange));
      setRegistrationCodes(processRegistrationCodes(registrationCodesResult));

      const operationalKpiModel = buildDashboardOperationalKpis({
        usageDaily: usageDailyResult.status === 'fulfilled' ? usageDailyResult.value : undefined,
        llmUsageSummary: llmUsageSummaryResult.status === 'fulfilled' ? llmUsageSummaryResult.value : undefined,
        jobsStats: jobsStatsResult.status === 'fulfilled' ? jobsStatsResult.value : undefined,
        metricsText: metricsTextResult.status === 'fulfilled' ? metricsTextResult.value : undefined,
        previousJobsSnapshot: previousJobsSnapshotRef.current,
      });
      setOperationalKpis(operationalKpiModel.kpis);
      if (operationalKpiModel.jobsSnapshot) {
        previousJobsSnapshotRef.current = operationalKpiModel.jobsSnapshot;
      }

      // Build overlay data for activity chart (errors, latency, cost per day)
      const usageRows = aggregateUsageDailyRows(
        usageDailyResult.status === 'fulfilled' ? usageDailyResult.value : undefined
      );
      const costRows = extractLlmDailyCostRows(
        llmUsageSummaryResult.status === 'fulfilled' ? llmUsageSummaryResult.value : undefined
      );
      const costByDay = new Map(costRows.map(r => [r.day, r.totalCostUsd]));
      setActivityOverlayRows(
        usageRows.map(r => ({
          day: r.day,
          errors: r.errors,
          latencyAvgMs: r.latencyAvgMs,
          costUsd: costByDay.get(r.day) ?? null,
        }))
      );

      const healthState: ServerStatusState = (() => {
        if (healthResult.status !== 'fulfilled') {
          return 'offline';
        }
        const status = (healthResult.value as { status?: string } | null)?.status;
        if (status === 'ok') return 'online';
        if (status === 'degraded') return 'degraded';
        if (status === 'unhealthy') return 'offline';
        return 'unknown';
      })();
      setServerStatus({ state: healthState, checkedAt: new Date().toISOString() });
      setUptimeSummary(
        incidentsResult.status === 'fulfilled'
          ? buildDashboardUptimeSummary({ incidentsPayload: incidentsResult.value })
          : DEFAULT_DASHBOARD_UPTIME_SUMMARY
      );

      // Process security health
      const resolvedSecurityHealth = resolveSecurityHealth(securityHealthResult);
      setSecurityHealth(resolvedSecurityHealth.data);
      setSecurityHealthError(resolvedSecurityHealth.error);

      // Process realtime stats (active sessions + token consumption)
      if (realtimeStatsResult.status === 'fulfilled' && realtimeStatsResult.value) {
        setRealtimeStats(realtimeStatsResult.value as RealtimeStats);
      } else {
        setRealtimeStats(null);
      }

      const { activeUsers, totalStorage, totalQuota } = computeUserStats(users);

      const computedStats: DashboardUIStats = {
        users: users.length,
        activeUsers,
        organizations: orgs.length,
        // Teams and API key counts provided by stats endpoint.
        teams: 0,
        apiKeys: 0,
        activeApiKeys: 0,
        providers: providerList.length,
        enabledProviders: enabledProviders.length,
        storageUsedMb: totalStorage,
        storageQuotaMb: totalQuota || 1000,
        activeAcpSessions: null,
        tokensToday: null,
        mcpInvocationsToday: null,
      };
      const rawStats = statsResult.status === 'fulfilled' ? statsResult.value : null;
      const nextStats = buildDashboardUIStats({
        computedStats,
        statsResponse: rawStats,
      });
      // Extract non-numeric fields that buildDashboardUIStats can't merge
      if (rawStats && typeof rawStats === 'object') {
        const r = rawStats as Record<string, unknown>;
        nextStats.activeAcpSessions = typeof r.active_acp_sessions === 'number' ? r.active_acp_sessions : null;
        nextStats.mcpInvocationsToday = typeof r.mcp_invocations_today === 'number' ? r.mcp_invocations_today : null;
        if (r.tokens_today && typeof r.tokens_today === 'object') {
          const tt = r.tokens_today as Record<string, unknown>;
          nextStats.tokensToday = {
            prompt: typeof tt.prompt === 'number' ? tt.prompt : 0,
            completion: typeof tt.completion === 'number' ? tt.completion : 0,
            total: typeof tt.total === 'number' ? tt.total : 0,
          };
        }
      }
      setStats(nextStats);

      if (isBillingEnabled()) {
        setBillingStats(
          await fetchDashboardBillingStats(() => api.getSubscriptions())
        );
      } else {
        setBillingStats(null);
      }

      setSystemHealth(buildDashboardSystemHealth({
        healthResult,
        llmHealthResult,
        ragHealthResult,
        ttsHealthResult,
        sttHealthResult,
        embeddingsHealthResult,
        metricsTextResult,
        jobsStatsResult,
      }));

      const optionalHealthFailures = [
        { key: 'llm_health', label: 'LLM health', result: llmHealthResult },
        { key: 'rag_health', label: 'RAG health', result: ragHealthResult },
        { key: 'tts_health', label: 'TTS health', result: ttsHealthResult },
        { key: 'stt_health', label: 'STT health', result: sttHealthResult },
        { key: 'embeddings_health', label: 'embeddings health', result: embeddingsHealthResult },
        { key: 'realtime_stats', label: 'realtime stats', result: realtimeStatsResult },
      ].filter((entry): entry is {
        key: string;
        label: string;
        result: PromiseRejectedResult;
      } => entry.result.status === 'rejected');
      if (optionalHealthFailures.length > 0) {
        logger.warn('Optional dashboard subsystem health fetch failures', {
          component: 'DashboardPage',
          failures: optionalHealthFailures.map((entry) => entry.key).join(', '),
        });
      }

      const failures = [
        { key: 'stats', label: 'stats', result: statsResult },
        { key: 'users', label: 'users', result: usersResult },
        { key: 'organizations', label: 'organizations', result: orgsResult },
        { key: 'providers', label: 'providers', result: providersResult },
        { key: 'audit', label: 'audit logs', result: auditResult },
        { key: 'alerts', label: 'alerts', result: alertsResult },
        { key: 'activity', label: 'activity', result: activityResult },
        { key: 'usage_daily', label: 'usage daily', result: usageDailyResult },
        { key: 'llm_usage_summary', label: 'LLM usage summary', result: llmUsageSummaryResult },
        { key: 'jobs_stats', label: 'jobs stats', result: jobsStatsResult },
        { key: 'metrics_text', label: 'metrics text', result: metricsTextResult },
        { key: 'registration_codes', label: 'registration codes', result: registrationCodesResult },
        { key: 'health', label: 'health', result: healthResult },
        { key: 'security_health', label: 'security health', result: securityHealthResult },
        { key: 'incidents', label: 'incidents', result: incidentsResult },
      ].filter((entry): entry is {
        key: string;
        label: string;
        result: PromiseRejectedResult;
      } => entry.result.status === 'rejected');

      if (failures.length > 0) {
        const failedLabels = failures.map((entry) => entry.label);
        logger.warn('Dashboard data fetch failures', {
          component: 'DashboardPage',
          failures: failedLabels.join(', '),
        });
        setError(`Some dashboard data failed to load: ${failedLabels.join(', ')}`);
      }
    } catch (err: unknown) {
      logger.error('Failed to load dashboard data', { component: 'DashboardPage', error: err instanceof Error ? err.message : String(err) });
      setError('Failed to load dashboard statistics');
      setOperationalKpis(DEFAULT_DASHBOARD_OPERATIONAL_KPIS);
      setUptimeSummary(DEFAULT_DASHBOARD_UPTIME_SUMMARY);
    } finally {
      loadingRef.current = false;
      setLoading(false);
      setLastRefreshed(new Date());
    }
  }, [activityRange, selectedOrg]);

  useEffect(() => {
    void loadDashboardData();
  }, [loadDashboardData]);

  // Auto-refresh: 60-second interval, paused when tab is not visible
  useEffect(() => {
    if (!autoRefreshEnabled) return;

    const intervalId = setInterval(() => {
      if (document.visibilityState === 'visible' && !loadingRef.current) {
        void loadDashboardData();
      }
    }, 60_000);

    return () => clearInterval(intervalId);
  }, [autoRefreshEnabled, loadDashboardData]);


  const formatTimeAgo = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
  };

  const getServerStatusLabel = (state: ServerStatusState) => {
    switch (state) {
      case 'online':
        return 'Online';
      case 'degraded':
        return 'Degraded';
      case 'offline':
        return 'Offline';
      default:
        return 'Unknown';
    }
  };

  const getServerStatusDot = (state: ServerStatusState) => {
    switch (state) {
      case 'online':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      case 'offline':
        return 'bg-red-500';
      default:
        return 'bg-gray-400';
    }
  };

  const isRegistrationCodeActive = (code: RegistrationCode) => {
    if (!code.expires_at) {
      return code.times_used < code.max_uses;
    }
    const expiresAt = new Date(code.expires_at);
    if (Number.isNaN(expiresAt.getTime())) {
      return code.times_used < code.max_uses;
    }
    return expiresAt >= new Date() && code.times_used < code.max_uses;
  };

  const handleCreateUserSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setCreateUserError('');
    setCreatingUser(true);
    try {
      await api.createUser({
        username: createUserForm.username,
        email: createUserForm.email,
        password: createUserForm.password,
        role: createUserForm.role,
        is_active: createUserForm.is_active,
        is_verified: createUserForm.is_verified,
      });
      success('User created', `${createUserForm.username} added.`);
      setShowCreateUserDialog(false);
      setCreateUserForm({
        username: '',
        email: '',
        password: '',
        role: 'user',
        is_active: true,
        is_verified: true,
      });
      await loadDashboardData();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to create user';
      setCreateUserError(message);
      showError('Create user failed', message);
    } finally {
      setCreatingUser(false);
    }
  };

  const handleOrgNameChange = (name: string) => {
    const slug = name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
    setOrgForm({ name, slug });
  };

  const handleOrgSlugChange = (slug: string) => {
    setOrgForm((prev) => ({ ...prev, slug }));
  };

  const handleCreateUserDialogOpenChange = (open: boolean) => {
    setShowCreateUserDialog(open);
    if (!open) setCreateUserError('');
  };

  const handleOrgDialogOpenChange = (open: boolean) => {
    setShowOrgDialog(open);
    if (!open) setOrgError('');
  };

  const handleOrgSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setOrgError('');
    setCreatingOrg(true);
    try {
      await api.createOrganization(orgForm);
      success('Organization created', `${orgForm.name} added.`);
      setShowOrgDialog(false);
      setOrgForm({ name: '', slug: '' });
      await loadDashboardData();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to create organization';
      setOrgError(message);
      showError('Organization failed', message);
    } finally {
      setCreatingOrg(false);
    }
  };

  const handleAcknowledgeAlerts = useCallback(async () => {
    try {
      await Promise.allSettled(
        alerts
          .filter((a) => a.id !== undefined)
          .map((a) => api.acknowledgeAlert(String(a.id)))
      );
      success('Alerts acknowledged', 'All visible alerts acknowledged.');
      await loadDashboardData();
    } catch {
      showError('Acknowledge failed', 'Could not acknowledge alerts.');
    }
  }, [alerts, loadDashboardData, success, showError]);

  const activityChartData = useMemo(
    () => mergeOverlayData(
      buildDashboardActivityChartData(activityData, activityRange),
      activityOverlayRows,
    ),
    [activityData, activityRange, activityOverlayRows]
  );

  const storagePercentage = stats.storageQuotaMb > 0
    ? Math.min((stats.storageUsedMb / stats.storageQuotaMb) * 100, 100)
    : 0;
  const serverStatusLabel = getServerStatusLabel(serverStatus.state);
  const serverStatusDotClass = getServerStatusDot(serverStatus.state);
  const checkedAtLabel = serverStatus.checkedAt ? formatTimeAgo(serverStatus.checkedAt) : null;
  const hasSecurityHealth = securityHealth !== null;

  const activeRegistrationCount = registrationCodes.filter(isRegistrationCodeActive).length;
  const recentOrganizations = organizations.slice(0, 3);
  const handleActivityRangeChange = (nextRange: DashboardActivityRange) => {
    if (nextRange === activityRange) return;
    setActivityRange(nextRange);
  };

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <p
            className="sr-only"
            role="status"
            aria-live="polite"
            aria-atomic="true"
            data-testid="dashboard-alert-count-live"
          >
            {alerts.length} open alert{alerts.length !== 1 ? 's' : ''} on the dashboard.
          </p>
          <DashboardHeader
            serverStatusLabel={serverStatusLabel}
            serverStatusDotClass={serverStatusDotClass}
            checkedAtLabel={checkedAtLabel}
            uptimePercent={uptimeSummary.uptimePercent}
            lastIncidentAt={uptimeSummary.lastIncidentAt}
            uptimeWindowDays={uptimeSummary.windowDays}
            loading={loading}
            onRefresh={loadDashboardData}
            lastRefreshed={lastRefreshed}
            autoRefreshEnabled={autoRefreshEnabled}
            onAutoRefreshToggle={() => setAutoRefreshEnabled((prev) => !prev)}
          />

          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <AlertsBanner alerts={alerts} onAcknowledgeAll={handleAcknowledgeAlerts} />

          <StatsGrid
            loading={loading}
            stats={stats}
            storagePercentage={storagePercentage}
            operationalKpis={operationalKpis}
            realtimeStats={realtimeStats}
            cacheHitRatePct={systemHealth.cache.cacheHitRatePct}
          />

          {isBillingEnabled() && billingStats && (
            <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Subscriptions</CardTitle>
                  <CreditCard className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{billingStats.active_subscriptions}</div>
                  <p className="text-xs text-muted-foreground">
                    {Object.entries(billingStats.plan_distribution).map(
                      ([tier, count]) => `${count} ${tier}`
                    ).join(', ') || 'No subscriptions'}
                  </p>
                </CardContent>
              </Card>
              {billingStats.past_due_count > 0 && (
                <Card className="border-red-200 dark:border-red-800">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-red-600">Past Due</CardTitle>
                    <CreditCard className="h-4 w-4 text-red-600" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-red-600">{billingStats.past_due_count}</div>
                    <p className="text-xs text-muted-foreground">subscriptions need attention</p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          <ActivitySection
            activityChartData={activityChartData}
            systemHealth={systemHealth}
            activityRange={activityRange}
            onActivityRangeChange={handleActivityRangeChange}
            loading={loading}
          />

          <div className="grid gap-6 lg:grid-cols-2">
            <RecentActivityCard
              loading={loading}
              recentActivity={recentActivity}
              formatTimeAgo={formatTimeAgo}
            />
            <QuickActionsCard />
          </div>

            <div className="mt-8 grid gap-6 lg:grid-cols-3">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <UserPlus className="h-5 w-5" />
                    User Registration
                  </CardTitle>
                  <CardDescription>Onboard new users</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-sm text-muted-foreground">Active codes</p>
                      <p className="text-2xl font-bold">{activeRegistrationCount}</p>
                    </div>
                    <CreateUserDialog
                      open={showCreateUserDialog}
                      onOpenChange={handleCreateUserDialogOpenChange}
                      error={createUserError}
                      form={createUserForm}
                      setForm={setCreateUserForm}
                      creating={creatingUser}
                      onSubmit={handleCreateUserSubmit}
                    />
                  </div>

                  <div className="mt-4 flex flex-wrap gap-2">
                    <Link href="/users/registration">
                      <Button variant="outline" size="sm">
                        <KeyRound className="mr-2 h-4 w-4" />
                        Manage Registration Codes
                      </Button>
                    </Link>
                    <Link href="/users">
                      <Button variant="ghost" size="sm">Manage users</Button>
                    </Link>
                    <Link href="/roles">
                      <Button variant="ghost" size="sm">Roles & access</Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Building2 className="h-5 w-5" />
                    Organization Management
                  </CardTitle>
                  <CardDescription>Create, invite, and manage orgs</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Organizations</p>
                      <p className="text-2xl font-bold">{stats.organizations}</p>
                    </div>
                    <CreateOrganizationDialog
                      open={showOrgDialog}
                      onOpenChange={handleOrgDialogOpenChange}
                      error={orgError}
                      form={orgForm}
                      onNameChange={handleOrgNameChange}
                      onSlugChange={handleOrgSlugChange}
                      creating={creatingOrg}
                      onSubmit={handleOrgSubmit}
                    />
                  </div>

                  <div className="mt-4 space-y-3">
                    {recentOrganizations.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No organizations created yet.</p>
                    ) : (
                      recentOrganizations.map((org) => (
                        <div key={org.id} className="flex items-center justify-between rounded-lg border p-3">
                          <div>
                            <p className="text-sm font-medium">{org.name}</p>
                            <p className="text-xs text-muted-foreground">{org.slug}</p>
                          </div>
                          <Link href={`/organizations/${org.id}`}>
                            <Button variant="ghost" size="sm">
                              Manage
                            </Button>
                          </Link>
                        </div>
                      ))
                    )}
                  </div>

                  <div className="mt-4 flex flex-wrap gap-2">
                    <Link href="/organizations">
                      <Button variant="outline" size="sm">Manage orgs</Button>
                    </Link>
                    <Link href="/teams">
                      <Button variant="ghost" size="sm">Teams & invites</Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings className="h-5 w-5" />
                    Server & Governance
                  </CardTitle>
                  <CardDescription>Configuration, providers, and monitoring controls</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">LLM providers</span>
                      <Badge variant="secondary">
                        {stats.enabledProviders}/{stats.providers} enabled
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Active users</span>
                      <Badge variant="secondary">{stats.activeUsers}</Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Open alerts</span>
                      <Badge variant={alerts.length > 0 ? 'destructive' : 'secondary'}>
                        {alerts.length}
                      </Badge>
                    </div>
                  </div>

                  <div className="mt-4 grid gap-2 sm:grid-cols-2">
                    <Link href="/config">
                      <Button variant="outline" size="sm" className="w-full">
                        Server settings
                      </Button>
                    </Link>
                    <Link href="/providers">
                      <Button variant="outline" size="sm" className="w-full">
                        Provider keys
                      </Button>
                    </Link>
                    <Link href="/monitoring">
                      <Button variant="ghost" size="sm" className="w-full">
                        Monitoring
                      </Button>
                    </Link>
                    <Link href="/audit">
                      <Button variant="ghost" size="sm" className="w-full">
                        Audit logs
                      </Button>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Security Health Card */}
            <div className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <ShieldAlert className="h-5 w-5" />
                    Security Overview
                  </CardTitle>
                  <CardDescription>Security posture and recent activity</CardDescription>
                </CardHeader>
                <CardContent>
                  {securityHealthError && (
                    <Alert className="mb-4">
                      <AlertDescription>{securityHealthError}</AlertDescription>
                    </Alert>
                  )}
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground">Risk Score</p>
                      <div className="flex items-center gap-2">
                        <span className={`text-2xl font-bold ${
                          !hasSecurityHealth
                            ? 'text-muted-foreground'
                            : ((securityHealth?.risk_score ?? 0) >= 70 ? 'text-red-500' :
                              (securityHealth?.risk_score ?? 0) >= 40 ? 'text-yellow-500' :
                              'text-green-500')
                        }`}>
                          {hasSecurityHealth ? securityHealth.risk_score : '—'}
                        </span>
                        {hasSecurityHealth && <span className="text-xs text-muted-foreground">/ 100</span>}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground">Security Events (24h)</p>
                      <p className={`text-2xl font-bold ${!hasSecurityHealth ? 'text-muted-foreground' : ''}`}>
                        {hasSecurityHealth ? securityHealth.recent_security_events : '—'}
                      </p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground">Failed Logins (24h)</p>
                      <p className={`text-2xl font-bold ${
                        !hasSecurityHealth ? 'text-muted-foreground' :
                        (securityHealth?.failed_logins_24h ?? 0) > 10 ? 'text-red-500' : ''
                      }`}>
                        {hasSecurityHealth ? securityHealth.failed_logins_24h : '—'}
                      </p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground">Suspicious Activity</p>
                      <p className={`text-2xl font-bold ${
                        !hasSecurityHealth ? 'text-muted-foreground' :
                        (securityHealth?.suspicious_activity ?? 0) > 0 ? 'text-yellow-500' : ''
                      }`}>
                        {hasSecurityHealth ? securityHealth.suspicious_activity : '—'}
                      </p>
                    </div>
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground">MFA Adoption</p>
                      <p className={`text-2xl font-bold ${!hasSecurityHealth ? 'text-muted-foreground' : ''}`}>
                        {hasSecurityHealth ? `${securityHealth.mfa_adoption_rate}%` : '—'}
                      </p>
                    </div>
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <Button variant="outline" size="sm" onClick={() => router.push('/security')}>
                      Security Dashboard
                    </Button>
                    <Button variant="ghost" size="sm" onClick={() => router.push('/audit?filter=security')}>
                      Security Audit Logs
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
