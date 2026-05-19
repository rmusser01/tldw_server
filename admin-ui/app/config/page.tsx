'use client';

import type { ReactNode } from 'react';
import Link from 'next/link';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Cpu,
  Database,
  Flag,
  RefreshCw,
  Server,
  Shield,
  Workflow,
} from 'lucide-react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { api } from '@/lib/api-client';

type UnknownRecord = Record<string, unknown>;

type ProviderSummary = {
  name: string;
  enabled: boolean;
  modelCount: number;
};

type FeatureFlagSummary = {
  key: string;
  enabled: boolean;
  scope: string;
  rolloutPercent: number;
  targetUserCount: number;
};

type StatusValue = 'healthy' | 'degraded' | 'unhealthy' | 'unknown';

type ConfigDataState = {
  health: unknown | null;
  stats: unknown | null;
  featureFlags: unknown | null;
  providers: unknown | null;
};

const numberFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 1,
});

const integerFormatter = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 0,
});

const asRecord = (value: unknown): UnknownRecord | null =>
  typeof value === 'object' && value !== null ? (value as UnknownRecord) : null;

const getString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;

const getNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

const firstString = (...values: unknown[]): string | null => {
  for (const value of values) {
    const parsed = getString(value);
    if (parsed) return parsed;
  }
  return null;
};

const formatError = (error: unknown): string =>
  error instanceof Error && error.message ? error.message : 'Request failed';

const formatMb = (value: number): string => `${numberFormatter.format(value)} MB`;

const formatUptime = (value: unknown): string => {
  const numeric = getNumber(value);
  if (numeric === null) {
    const asText = getString(value);
    return asText ?? 'Unavailable';
  }
  const totalSeconds = Math.max(0, Math.floor(numeric));
  const days = Math.floor(totalSeconds / 86_400);
  const hours = Math.floor((totalSeconds % 86_400) / 3_600);
  const minutes = Math.floor((totalSeconds % 3_600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
};

const normalizeStatus = (value: unknown): StatusValue => {
  const text = getString(value)?.toLowerCase();
  if (!text) return 'unknown';
  if (['ok', 'healthy', 'ready', 'alive', 'enabled'].includes(text)) return 'healthy';
  if (['degraded', 'warning', 'warn'].includes(text)) return 'degraded';
  if (['unhealthy', 'error', 'critical', 'failed', 'not_ready'].includes(text)) return 'unhealthy';
  return 'unknown';
};

const getStatusLabel = (status: StatusValue): string => {
  if (status === 'healthy') return 'Healthy';
  if (status === 'degraded') return 'Degraded';
  if (status === 'unhealthy') return 'Unhealthy';
  return 'Unknown';
};

const getBadgeVariant = (status: StatusValue): 'default' | 'secondary' | 'destructive' | 'outline' => {
  if (status === 'healthy') return 'default';
  if (status === 'degraded') return 'secondary';
  if (status === 'unhealthy') return 'destructive';
  return 'outline';
};

const parseProviders = (payload: unknown): ProviderSummary[] => {
  const payloadRecord = asRecord(payload);
  const rawProviders = payloadRecord ? payloadRecord.providers : null;
  const results: ProviderSummary[] = [];

  const parseProvider = (nameHint: string | null, rawValue: unknown): ProviderSummary | null => {
    const record = asRecord(rawValue);
    if (!record) return null;
    const name = firstString(record.name, record.provider, nameHint) ?? 'Unknown';
    const enabled = typeof record.enabled === 'boolean'
      ? record.enabled
      : typeof record.is_enabled === 'boolean'
        ? record.is_enabled
        : true;
    const models = Array.isArray(record.models) ? record.models : [];
    return { name, enabled, modelCount: models.length };
  };

  if (Array.isArray(rawProviders)) {
    rawProviders.forEach((provider) => {
      const parsed = parseProvider(null, provider);
      if (parsed) results.push(parsed);
    });
    return results;
  }

  if (asRecord(rawProviders)) {
    Object.entries(rawProviders as UnknownRecord).forEach(([providerName, providerValue]) => {
      const parsed = parseProvider(providerName, providerValue);
      if (parsed) results.push(parsed);
    });
    return results;
  }

  if (Array.isArray(payload)) {
    payload.forEach((provider) => {
      const parsed = parseProvider(null, provider);
      if (parsed) results.push(parsed);
    });
    return results;
  }

  if (payloadRecord) {
    const ignoredKeys = new Set(['default_provider', 'total_configured', 'message', 'diagnostics_ui']);
    Object.entries(payloadRecord).forEach(([providerName, providerValue]) => {
      if (ignoredKeys.has(providerName)) return;
      const parsed = parseProvider(providerName, providerValue);
      if (parsed) results.push(parsed);
    });
  }

  return results;
};

const parseFeatureFlags = (payload: unknown): FeatureFlagSummary[] => {
  const payloadRecord = asRecord(payload);
  const items = payloadRecord && Array.isArray(payloadRecord.items)
    ? payloadRecord.items
    : Array.isArray(payload)
      ? payload
      : [];
  return items.reduce<FeatureFlagSummary[]>((acc, item) => {
    const record = asRecord(item);
    if (!record) return acc;
    const key = firstString(record.key, record.name) ?? '';
    if (!key) return acc;
    const enabled = record.enabled === true;
    const scope = firstString(record.scope) ?? 'global';
    const rolloutPercent = getNumber(record.rollout_percent) ?? 100;
    const targetUserCount = Array.isArray(record.target_user_ids)
      ? record.target_user_ids.length
      : 0;
    acc.push({ key, enabled, scope, rolloutPercent, targetUserCount });
    return acc;
  }, []);
};

const formatProviderName = (name: string): string =>
  name
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());

const getCheckRecord = (checks: UnknownRecord | null, keys: string[]): UnknownRecord | null => {
  if (!checks) return null;
  for (const key of keys) {
    const check = asRecord(checks[key]);
    if (check) return check;
  }
  return null;
};

function SectionRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b pb-3 last:border-b-0 last:pb-0">
      <dt className="text-sm font-medium text-muted-foreground">{label}</dt>
      <dd className="text-right text-sm">{value}</dd>
    </div>
  );
}

function SectionLinks({ links }: { links: Array<{ href: string; label: string }> }) {
  return (
    <div className="mt-4 flex flex-wrap gap-4 text-sm">
      {links.map((link) => (
        <Link key={link.href} href={link.href} className="text-primary hover:underline">
          {link.label}
        </Link>
      ))}
    </div>
  );
}

export default function ConfigPage() {
  const [loading, setLoading] = useState(true);
  const [errors, setErrors] = useState<string[]>([]);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [data, setData] = useState<ConfigDataState>({
    health: null,
    stats: null,
    featureFlags: null,
    providers: null,
  });

  const loadData = useCallback(async () => {
    setLoading(true);
    const [healthResult, statsResult, featureFlagsResult, providersResult] = await Promise.allSettled([
      api.getHealth(),
      api.getDashboardStats(),
      api.getFeatureFlags(),
      api.getLLMProviders(),
    ]);

    const nextErrors: string[] = [];
    setData({
      health: healthResult.status === 'fulfilled' ? healthResult.value : null,
      stats: statsResult.status === 'fulfilled' ? statsResult.value : null,
      featureFlags: featureFlagsResult.status === 'fulfilled' ? featureFlagsResult.value : null,
      providers: providersResult.status === 'fulfilled' ? providersResult.value : null,
    });

    if (healthResult.status === 'rejected') {
      nextErrors.push(`Health endpoint unavailable: ${formatError(healthResult.reason)}`);
    }
    if (statsResult.status === 'rejected') {
      nextErrors.push(`Stats endpoint unavailable: ${formatError(statsResult.reason)}`);
    }
    if (featureFlagsResult.status === 'rejected') {
      nextErrors.push(`Feature flags unavailable: ${formatError(featureFlagsResult.reason)}`);
    }
    if (providersResult.status === 'rejected') {
      nextErrors.push(`Providers endpoint unavailable: ${formatError(providersResult.reason)}`);
    }

    setErrors(nextErrors);
    setLastUpdated(new Date().toISOString());
    setLoading(false);
  }, []);

  useEffect(() => {
    const timerId = window.setTimeout(() => {
      void loadData();
    }, 0);
    return () => window.clearTimeout(timerId);
  }, [loadData]);

  const {
    authMode,
    authStatus,
    sessionSummary,
    mfaPolicySummary,
    storageBackend,
    storagePath,
    capacitySummary,
    featureSummary,
    rolloutSummary,
    defaultProviderLabel,
    providerSummary,
    providerItems,
    serverVersion,
    serverUptime,
    pythonVersion,
    operatingSystem,
    deploymentMode,
    serverTimestamp,
    serviceStatuses,
  } = useMemo(() => {
    const health = asRecord(data.health);
    const checks = asRecord(health?.checks);
    const stats = asRecord(data.stats);
    const storageStats = asRecord(stats?.storage);
    const sessionStats = asRecord(stats?.sessions);
    const featureFlags = parseFeatureFlags(data.featureFlags);
    const providers = parseProviders(data.providers);

    const authModeValue = firstString(health?.auth_mode) ?? 'Unavailable';
    const authStatusValue = normalizeStatus(health?.status);
    const sessionsActive = getNumber(sessionStats?.active);
    const sessionsUnique = getNumber(sessionStats?.unique_users);
    const sessionSummaryValue =
      sessionsActive !== null && sessionsUnique !== null
        ? `${integerFormatter.format(sessionsActive)} active / ${integerFormatter.format(sessionsUnique)} unique users`
        : 'Unavailable';

    const mfaFlags = featureFlags.filter((flag) => flag.key.toLowerCase().includes('mfa'));
    const mfaPolicyValue =
      mfaFlags.length === 0
        ? 'No MFA policy flag found'
        : mfaFlags.some((flag) => flag.enabled)
          ? `Enabled by ${mfaFlags.filter((flag) => flag.enabled).length} flag(s)`
          : 'Configured but disabled';

    const databaseCheck = getCheckRecord(checks, ['database']);
    const chachaCheck = getCheckRecord(checks, ['chacha_notes']);

    const storageBackendValue = firstString(
      databaseCheck?.backend,
      databaseCheck?.database,
      databaseCheck?.engine
    ) ?? 'Unavailable';
    const storagePathValue = firstString(
      chachaCheck?.db_path,
      chachaCheck?.path,
      chachaCheck?.database_path
    ) ?? 'Unavailable';

    const usedMb = getNumber(storageStats?.total_used_mb);
    const quotaMb = getNumber(storageStats?.total_quota_mb);
    const usagePercent = usedMb !== null && quotaMb !== null && quotaMb > 0
      ? Math.round((usedMb / quotaMb) * 100)
      : null;
    const capacityValue =
      usedMb !== null && quotaMb !== null
        ? `${formatMb(usedMb)} / ${formatMb(quotaMb)}${usagePercent !== null ? ` (${usagePercent}%)` : ''}`
        : 'Unavailable';

    const enabledFlags = featureFlags.filter((flag) => flag.enabled).length;
    const featureSummaryValue =
      featureFlags.length > 0
        ? `${enabledFlags} enabled of ${featureFlags.length} total`
        : 'No feature flags configured';
    const rolloutRules = featureFlags.filter(
      (flag) => flag.rolloutPercent < 100 || flag.targetUserCount > 0
    ).length;
    const rolloutSummaryValue =
      featureFlags.length > 0
        ? `${rolloutRules} targeted/partial rollout rule(s)`
        : 'No rollout rules configured';

    const defaultProvider = firstString(asRecord(data.providers)?.default_provider);
    const enabledProviders = providers.filter((provider) => provider.enabled).length;
    const providerSummaryValue =
      providers.length > 0
        ? `${enabledProviders} enabled of ${providers.length} configured`
        : 'No providers configured';
    const defaultProviderValue = defaultProvider
      ? formatProviderName(defaultProvider)
      : 'Not set';

    const serviceStatusesValue: Array<{ name: string; status: StatusValue }> = [
      { name: 'API Core', status: authStatusValue },
      { name: 'Database', status: normalizeStatus(getCheckRecord(checks, ['database'])?.status) },
      { name: 'Metrics', status: normalizeStatus(getCheckRecord(checks, ['metrics'])?.status) },
      { name: 'RAG', status: normalizeStatus(getCheckRecord(checks, ['rag'])?.status) },
      { name: 'TTS', status: normalizeStatus(getCheckRecord(checks, ['tts', 'audio_tts'])?.status) },
      { name: 'STT', status: normalizeStatus(getCheckRecord(checks, ['stt', 'audio_stt'])?.status) },
      { name: 'MCP', status: normalizeStatus(getCheckRecord(checks, ['mcp', 'mcp_unified'])?.status) },
    ];

    return {
      authMode: authModeValue,
      authStatus: authStatusValue,
      sessionSummary: sessionSummaryValue,
      mfaPolicySummary: mfaPolicyValue,
      storageBackend: storageBackendValue,
      storagePath: storagePathValue,
      capacitySummary: capacityValue,
      featureSummary: featureSummaryValue,
      rolloutSummary: rolloutSummaryValue,
      defaultProviderLabel: defaultProviderValue,
      providerSummary: providerSummaryValue,
      providerItems: providers,
      serverVersion: firstString(health?.version, health?.app_version, health?.release) ?? 'Unavailable',
      serverUptime: formatUptime(health?.uptime_seconds ?? health?.uptime_sec ?? health?.uptime),
      pythonVersion: firstString(health?.python_version, health?.python) ?? 'Unavailable',
      operatingSystem: firstString(health?.os, health?.platform) ?? 'Unavailable',
      deploymentMode: firstString(health?.deployment_mode, health?.environment) ?? 'Unavailable',
      serverTimestamp: firstString(health?.timestamp) ?? 'Unavailable',
      serviceStatuses: serviceStatusesValue,
    };
  }, [data]);

  const hasLoadedData = data.health !== null || data.stats !== null || data.featureFlags !== null || data.providers !== null;

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-3xl font-bold">System Configuration Overview</h1>
                <Badge
                  variant={
                    process.env.NODE_ENV === 'production' ? 'default'
                    : process.env.NODE_ENV === 'development' ? 'secondary'
                    : 'outline'
                  }
                  className={
                    process.env.NODE_ENV === 'production' ? 'border-red-500 text-red-600' :
                    process.env.NODE_ENV === 'development' ? 'border-green-500 text-green-600' :
                    'border-yellow-500 text-yellow-600'
                  }
                  data-testid="environment-badge"
                >
                  {process.env.NODE_ENV ?? 'unknown'}
                </Badge>
              </div>
              <p className="text-muted-foreground">
                Read-only platform configuration and subsystem status at a glance.
              </p>
            </div>
            <Button variant="outline" onClick={() => void loadData()} loading={loading} loadingText="Refreshing...">
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
          </div>

          {lastUpdated && (
            <p className="mb-4 text-sm text-muted-foreground">
              Last updated: {new Date(lastUpdated).toLocaleString()}
            </p>
          )}

          {errors.length > 0 && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>
                <p className="font-medium">Some configuration data could not be loaded:</p>
                <ul className="mt-2 list-disc pl-4">
                  {errors.map((error) => (
                    <li key={error}>{error}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          {loading && !hasLoadedData ? (
            <Card>
              <CardContent className="pt-6">
                <div className="text-center text-muted-foreground py-8">Loading configuration overview...</div>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-6 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <Shield className="h-5 w-5" />
                    Authentication
                  </CardTitle>
                  <CardDescription>Current authentication mode and session posture.</CardDescription>
                </CardHeader>
                <CardContent>
                  <dl className="space-y-3">
                    <SectionRow label="Auth mode" value={authMode} />
                    <SectionRow
                      label="Auth health"
                      value={<Badge variant={getBadgeVariant(authStatus)}>{getStatusLabel(authStatus)}</Badge>}
                    />
                    <SectionRow label="Active sessions" value={sessionSummary} />
                    <SectionRow label="MFA policy" value={mfaPolicySummary} />
                  </dl>
                  <SectionLinks
                    links={[
                      { href: '/security', label: 'Manage security' },
                      { href: '/users', label: 'Review users' },
                    ]}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <Database className="h-5 w-5" />
                    Storage
                  </CardTitle>
                  <CardDescription>Backend connectivity and aggregate storage utilization.</CardDescription>
                </CardHeader>
                <CardContent>
                  <dl className="space-y-3">
                    <SectionRow label="Database backend" value={storageBackend} />
                    <SectionRow label="Storage path" value={storagePath} />
                    <SectionRow label="Capacity" value={capacitySummary} />
                  </dl>
                  <SectionLinks
                    links={[
                      { href: '/usage', label: 'View usage analytics' },
                      { href: '/data-ops', label: 'Open Data Ops' },
                    ]}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <Flag className="h-5 w-5" />
                    Features
                  </CardTitle>
                  <CardDescription>Feature flag state and rollout targeting summary.</CardDescription>
                </CardHeader>
                <CardContent>
                  <dl className="space-y-3">
                    <SectionRow label="Feature flags" value={featureSummary} />
                    <SectionRow label="Rollout rules" value={rolloutSummary} />
                  </dl>
                  <SectionLinks links={[{ href: '/flags', label: 'Manage feature flags' }]} />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <Cpu className="h-5 w-5" />
                    Providers
                  </CardTitle>
                  <CardDescription>Configured LLM providers and default routing target.</CardDescription>
                </CardHeader>
                <CardContent>
                  <dl className="space-y-3">
                    <SectionRow label="Provider summary" value={providerSummary} />
                    <SectionRow label="Default provider" value={defaultProviderLabel} />
                  </dl>
                  <div className="mt-4 space-y-2">
                    {providerItems.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No providers configured.</p>
                    ) : (
                      providerItems.map((provider) => (
                        <div key={provider.name} className="flex items-center justify-between rounded-md border px-3 py-2">
                          <span className="text-sm">
                            {formatProviderName(provider.name)} ({provider.modelCount} model{provider.modelCount === 1 ? '' : 's'})
                          </span>
                          <Badge variant={provider.enabled ? 'default' : 'outline'}>
                            {provider.enabled ? 'Enabled' : 'Disabled'}
                          </Badge>
                        </div>
                      ))
                    )}
                  </div>
                  <SectionLinks
                    links={[
                      { href: '/providers', label: 'Manage providers' },
                      { href: '/byok', label: 'Manage BYOK keys' },
                    ]}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <Workflow className="h-5 w-5" />
                    Services
                  </CardTitle>
                  <CardDescription>Subsystem health derived from the aggregate health checks.</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {serviceStatuses.map((service) => (
                      <div key={service.name} className="flex items-center justify-between rounded-md border px-3 py-2">
                        <span className="text-sm">{service.name}</span>
                        <Badge variant={getBadgeVariant(service.status)}>
                          {getStatusLabel(service.status)}
                        </Badge>
                      </div>
                    ))}
                  </div>
                  <SectionLinks
                    links={[
                      { href: '/monitoring', label: 'Open monitoring' },
                      { href: '/incidents', label: 'View incidents' },
                    ]}
                  />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-xl">
                    <Server className="h-5 w-5" />
                    Server
                  </CardTitle>
                  <CardDescription>Runtime metadata, versioning, and deployment context.</CardDescription>
                </CardHeader>
                <CardContent>
                  <dl className="space-y-3">
                    <SectionRow label="Version" value={serverVersion} />
                    <SectionRow label="Uptime" value={serverUptime} />
                    <SectionRow label="Python version" value={pythonVersion} />
                    <SectionRow label="Operating system" value={operatingSystem} />
                    <SectionRow label="Deployment mode" value={deploymentMode} />
                    <SectionRow label="Health timestamp" value={serverTimestamp} />
                  </dl>
                  <SectionLinks
                    links={[
                      { href: '/logs', label: 'Inspect logs' },
                      { href: '/monitoring', label: 'Check live metrics' },
                    ]}
                  />
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
