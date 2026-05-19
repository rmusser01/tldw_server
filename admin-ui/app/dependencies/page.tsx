'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Cpu,
  Database,
  RefreshCw,
  Server,
  Wifi,
  WifiOff,
  XCircle,
} from 'lucide-react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { api } from '@/lib/api-client';
import { buildRecentUtcDayKeys, buildSparklinePoints } from '@/lib/provider-token-trends';
import type { DependencyUptimeStats, SystemDependencyItem, SystemDependencyStatus } from '@/types';

type ProviderHealthStatus = 'reachable' | 'unreachable' | 'unknown';

type Provider = {
  name: string;
  enabled: boolean;
  models: string[];
  defaultModel: string | null;
};

type ProviderUsageSummary = {
  requests: number;
  errors: number;
  latencyAvgMs: number | null;
};

type ProviderHealthCheck = {
  status: ProviderHealthStatus;
  testing: boolean;
  lastCheckedAt: string | null;
  lastSuccessAt: string | null;
  responseTimeMs: number | null;
  errorMessage: string | null;
};

type LlmSummaryRow = {
  group_value?: string;
  group_value_secondary?: string | null;
  requests?: number;
  errors?: number;
  latency_avg_ms?: number | null;
};

const TREND_DAYS = 7;

const asRecord = (value: unknown): Record<string, unknown> | null =>
  typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : null;

const getString = (value: unknown): string | null =>
  typeof value === 'string' && value.trim().length > 0 ? value.trim() : null;

const getNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

const formatProviderName = (value: string): string =>
  value
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());

const toProviderList = (payload: unknown): Provider[] => {
  const payloadRecord = asRecord(payload);
  const rawProviders = payloadRecord?.providers;
  const providers: Provider[] = [];

  const parseProvider = (nameHint: string | null, raw: unknown): Provider | null => {
    const record = asRecord(raw);
    if (!record) return null;
    const name = getString(record.name) ?? getString(record.provider) ?? nameHint;
    if (!name) return null;
    const models = Array.isArray(record.models)
      ? record.models.filter((model): model is string => typeof model === 'string' && model.trim().length > 0)
      : [];
    const defaultModel = getString(record.default_model);
    const enabled = typeof record.enabled === 'boolean'
      ? record.enabled
      : typeof record.is_enabled === 'boolean'
        ? record.is_enabled
        : true;
    return { name, enabled, models, defaultModel };
  };

  if (Array.isArray(rawProviders)) {
    rawProviders.forEach((provider) => {
      const parsed = parseProvider(null, provider);
      if (parsed) providers.push(parsed);
    });
    return providers;
  }

  const rawProvidersRecord = asRecord(rawProviders);
  if (rawProvidersRecord) {
    Object.entries(rawProvidersRecord).forEach(([providerName, providerValue]) => {
      const parsed = parseProvider(providerName, providerValue);
      if (parsed) providers.push(parsed);
    });
    return providers;
  }

  if (Array.isArray(payload)) {
    payload.forEach((provider) => {
      const parsed = parseProvider(null, provider);
      if (parsed) providers.push(parsed);
    });
    return providers;
  }

  if (payloadRecord) {
    const ignored = new Set(['default_provider', 'total_configured', 'message', 'diagnostics_ui']);
    Object.entries(payloadRecord).forEach(([providerName, providerValue]) => {
      if (ignored.has(providerName)) return;
      const parsed = parseProvider(providerName, providerValue);
      if (parsed) providers.push(parsed);
    });
  }

  return providers;
};

const toLlmSummaryRows = (payload: unknown): LlmSummaryRow[] => {
  if (Array.isArray(payload)) {
    return payload as LlmSummaryRow[];
  }
  const record = asRecord(payload);
  if (record && Array.isArray(record.items)) {
    return record.items as LlmSummaryRow[];
  }
  return [];
};

const normalizeDayKey = (value: unknown): string | null => {
  const day = getString(value);
  if (!day) return null;
  return day.slice(0, 10);
};

const get24HourWindow = () => {
  const end = new Date();
  const start = new Date(end.getTime() - (24 * 60 * 60 * 1000));
  return { start: start.toISOString(), end: end.toISOString() };
};

const getTrendWindow = (days: number) => {
  const end = new Date();
  const start = new Date(end.getTime() - (days * 24 * 60 * 60 * 1000));
  return { start: start.toISOString(), end: end.toISOString() };
};

const formatPercent = (value: number): string => `${value.toFixed(1)}%`;

const formatErrorRate = (errors: number, requests: number): string =>
  requests > 0 ? formatPercent((errors / requests) * 100) : '0.0%';

const formatCheckTime = (value: string | null): string =>
  value ? new Date(value).toLocaleTimeString() : 'Never';

const formatSinceSuccess = (value: string | null): string => {
  if (!value) return 'Never';
  const diffMs = Date.now() - new Date(value).getTime();
  if (!Number.isFinite(diffMs) || diffMs < 0) return 'Just now';
  const totalSeconds = Math.floor(diffMs / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s ago`;
  const totalMinutes = Math.floor(totalSeconds / 60);
  if (totalMinutes < 60) return `${totalMinutes}m ago`;
  const totalHours = Math.floor(totalMinutes / 60);
  if (totalHours < 24) return `${totalHours}h ago`;
  const totalDays = Math.floor(totalHours / 24);
  return `${totalDays}d ago`;
};

const statusLabel = (status: ProviderHealthStatus): string => {
  if (status === 'reachable') return 'Reachable';
  if (status === 'unreachable') return 'Unreachable';
  return 'Unknown';
};

const statusVariant = (status: ProviderHealthStatus): 'default' | 'secondary' | 'destructive' | 'outline' => {
  if (status === 'reachable') return 'default';
  if (status === 'unreachable') return 'destructive';
  return 'outline';
};

const hasPrometheusProviderData = (metricsText: string): boolean =>
  /provider|llm/i.test(metricsText);

const sysDepStatusLabel = (status: SystemDependencyStatus): string => {
  if (status === 'healthy') return 'Healthy';
  if (status === 'degraded') return 'Degraded';
  if (status === 'down') return 'Down';
  return 'Unknown';
};

const sysDepStatusVariant = (
  status: SystemDependencyStatus,
): 'default' | 'secondary' | 'destructive' | 'outline' => {
  if (status === 'healthy') return 'default';
  if (status === 'degraded') return 'secondary';
  if (status === 'down') return 'destructive';
  return 'outline';
};

const sysDepStatusIcon = (status: SystemDependencyStatus) => {
  if (status === 'healthy') return <CheckCircle2 className="mr-1 h-3 w-3" />;
  if (status === 'degraded') return <AlertTriangle className="mr-1 h-3 w-3" />;
  if (status === 'down') return <XCircle className="mr-1 h-3 w-3" />;
  return <AlertTriangle className="mr-1 h-3 w-3" />;
};

const createDefaultProviderCheck = (): ProviderHealthCheck => ({
  status: 'unknown',
  testing: false,
  lastCheckedAt: null,
  lastSuccessAt: null,
  responseTimeMs: null,
  errorMessage: null,
});

const reconcileProviderChecks = (
  previous: Record<string, ProviderHealthCheck>,
  providerList: Provider[],
): Record<string, ProviderHealthCheck> => {
  const next: Record<string, ProviderHealthCheck> = {};
  providerList.forEach((provider) => {
    const key = provider.name.toLowerCase();
    next[key] = previous[key]
      ? { ...createDefaultProviderCheck(), ...previous[key] }
      : createDefaultProviderCheck();
  });
  return next;
};

function AvailabilitySparkline({
  providerName,
  series,
}: {
  providerName: string;
  series: number[];
}) {
  if (!Array.isArray(series) || series.length === 0) {
    return <span className="text-xs text-muted-foreground">No trend</span>;
  }

  const points = buildSparklinePoints(series, { width: 96, height: 28, padding: 3 });
  const latest = series[series.length - 1] ?? 0;

  return (
    <div className="flex items-center justify-end gap-2">
      <svg
        role="img"
        aria-label={`${providerName} availability trend`}
        viewBox="0 0 96 28"
        className="h-7 w-24"
      >
        <polyline
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />
      </svg>
      <span className="w-12 text-right text-xs text-muted-foreground">{formatPercent(latest)}</span>
    </div>
  );
}

function UptimeBar({ sparkline, label }: { sparkline: number[]; label: string }) {
  if (!Array.isArray(sparkline) || sparkline.length === 0) {
    return <span className="text-xs text-muted-foreground">No history</span>;
  }

  // Downsample to ~56 buckets (one per 3-hour block) for compact display
  const bucketSize = Math.max(1, Math.floor(sparkline.length / 56));
  const buckets: number[] = [];
  for (let i = 0; i < sparkline.length; i += bucketSize) {
    const slice = sparkline.slice(i, i + bucketSize);
    const healthy = slice.filter((v) => v === 1).length;
    buckets.push(healthy / slice.length);
  }

  return (
    <div className="flex items-center gap-1">
      <svg
        role="img"
        aria-label={`${label} uptime sparkline`}
        viewBox={`0 0 ${buckets.length * 2} 12`}
        className="h-3"
        style={{ width: `${Math.min(buckets.length * 2, 112)}px` }}
      >
        {buckets.map((ratio, i) => (
          <rect
            key={i}
            x={i * 2}
            y={0}
            width={1.5}
            height={12}
            rx={0.5}
            fill={ratio >= 1 ? '#22c55e' : ratio > 0 ? '#eab308' : '#ef4444'}
            opacity={ratio >= 1 ? 0.8 : 1}
          />
        ))}
      </svg>
    </div>
  );
}

export default function DependenciesPage() {
  const [loading, setLoading] = useState(true);
  const [runningAllChecks, setRunningAllChecks] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [usageByProvider, setUsageByProvider] = useState<Record<string, ProviderUsageSummary>>({});
  const [availabilityByProvider, setAvailabilityByProvider] = useState<Record<string, number[]>>({});
  const [providerChecks, setProviderChecks] = useState<Record<string, ProviderHealthCheck>>({});
  const [prometheusAvailable, setPrometheusAvailable] = useState(false);
  const [systemDeps, setSystemDeps] = useState<SystemDependencyItem[]>([]);
  const [systemDepsCheckedAt, setSystemDepsCheckedAt] = useState<string | null>(null);
  const [systemDepsLoading, setSystemDepsLoading] = useState(true);
  const [uptimeByDep, setUptimeByDep] = useState<Record<string, DependencyUptimeStats>>({});
  const systemDepsSeqRef = useRef(0);
  const [uptimeByService, setUptimeByService] = useState<Record<string, Array<{ bucket: string; uptime_pct: number; probes: number }>>>({});

  const runProviderCheck = useCallback(async (provider: Provider): Promise<Omit<ProviderHealthCheck, 'testing'>> => {
    const started = performance.now();
    const checkedAt = new Date().toISOString();
    const modelToTest = provider.defaultModel ?? provider.models[0] ?? undefined;

    try {
      await api.testLLMProvider({
        provider: provider.name,
        model: modelToTest,
        use_override: true,
      });
      const elapsed = Math.max(1, Math.round(performance.now() - started));
      return {
        status: 'reachable',
        lastCheckedAt: checkedAt,
        lastSuccessAt: checkedAt,
        responseTimeMs: elapsed,
        errorMessage: null,
      };
    } catch (error: unknown) {
      const elapsed = Math.max(1, Math.round(performance.now() - started));
      return {
        status: 'unreachable',
        lastCheckedAt: checkedAt,
        lastSuccessAt: null,
        responseTimeMs: elapsed,
        errorMessage: error instanceof Error ? error.message : 'Connectivity check failed',
      };
    }
  }, []);

  const runAllProviderChecks = useCallback(async (providerList: Provider[]) => {
    if (providerList.length === 0) return;

    const providerKeys = providerList.map((provider) => provider.name.toLowerCase());
    setProviderChecks((prev) => {
      const next = { ...prev };
      providerKeys.forEach((key) => {
        next[key] = {
          ...createDefaultProviderCheck(),
          ...next[key],
          testing: true,
        };
      });
      return next;
    });

    const checks = await Promise.all(
      providerList.map(async (provider) => {
        const result = await runProviderCheck(provider);
        return { provider, result };
      })
    );

    setProviderChecks((prev) => {
      const next = { ...prev };
      checks.forEach(({ provider, result }) => {
        const key = provider.name.toLowerCase();
        const previous = prev[key];
        next[key] = {
          status: result.status,
          testing: false,
          lastCheckedAt: result.lastCheckedAt,
          lastSuccessAt:
            result.status === 'reachable'
              ? result.lastSuccessAt
              : previous?.lastSuccessAt ?? null,
          responseTimeMs: result.responseTimeMs,
          errorMessage: result.errorMessage,
        };
      });
      return next;
    });
  }, [runProviderCheck]);

  const loadTelemetry = useCallback(async () => {
    setLoading(true);
    setErrors([]);

    const summaryWindow = get24HourWindow();
    const trendWindow = getTrendWindow(TREND_DAYS);

    const [providersResult, usageSummaryResult, usageTrendResult, metricsResult, uptimeResult] = await Promise.allSettled([
      api.getLLMProviders(),
      api.getLlmUsageSummary({
        group_by: 'provider',
        start: summaryWindow.start,
        end: summaryWindow.end,
      }),
      api.getLlmUsageSummary({
        group_by: ['provider', 'day'],
        start: trendWindow.start,
        end: trendWindow.end,
      }),
      api.getMetricsText(),
      api.getDependenciesUptimeHistory({ range_days: '30' }),
    ]);

    const nextErrors: string[] = [];
    const parsedProviders = providersResult.status === 'fulfilled'
      ? toProviderList(providersResult.value)
      : [];
    const usageRows = usageSummaryResult.status === 'fulfilled'
      ? toLlmSummaryRows(usageSummaryResult.value)
      : [];
    const trendRows = usageTrendResult.status === 'fulfilled'
      ? toLlmSummaryRows(usageTrendResult.value)
      : [];

    if (providersResult.status === 'rejected') {
      nextErrors.push(`Providers unavailable: ${providersResult.reason instanceof Error ? providersResult.reason.message : 'Request failed'}`);
    }
    if (usageSummaryResult.status === 'rejected') {
      nextErrors.push(`Usage summary unavailable: ${usageSummaryResult.reason instanceof Error ? usageSummaryResult.reason.message : 'Request failed'}`);
    }
    if (usageTrendResult.status === 'rejected') {
      nextErrors.push(`Usage trend unavailable: ${usageTrendResult.reason instanceof Error ? usageTrendResult.reason.message : 'Request failed'}`);
    }
    if (uptimeResult.status === 'rejected') {
      nextErrors.push(`Uptime history unavailable: ${uptimeResult.reason instanceof Error ? uptimeResult.reason.message : 'Request failed'}`);
    }

    const usageMap: Record<string, ProviderUsageSummary> = {};
    usageRows.forEach((row) => {
      const providerKey = getString(row.group_value)?.toLowerCase();
      if (!providerKey) return;
      usageMap[providerKey] = {
        requests: getNumber(row.requests) ?? 0,
        errors: getNumber(row.errors) ?? 0,
        latencyAvgMs: getNumber(row.latency_avg_ms) ?? null,
      };
    });

    const dayKeys = buildRecentUtcDayKeys(TREND_DAYS, new Date(trendWindow.end));
    const dayIndexByKey = new Map(dayKeys.map((key, index) => [key, index]));
    const requestByProviderAndDay: Record<string, { requests: number[]; errors: number[] }> = {};
    trendRows.forEach((row) => {
      const providerKey = getString(row.group_value)?.toLowerCase();
      const dayKey = normalizeDayKey(row.group_value_secondary);
      if (!providerKey || !dayKey) return;
      const dayIndex = dayIndexByKey.get(dayKey);
      if (dayIndex === undefined) return;

      if (!requestByProviderAndDay[providerKey]) {
        requestByProviderAndDay[providerKey] = {
          requests: Array.from({ length: TREND_DAYS }, () => 0),
          errors: Array.from({ length: TREND_DAYS }, () => 0),
        };
      }
      requestByProviderAndDay[providerKey].requests[dayIndex] += getNumber(row.requests) ?? 0;
      requestByProviderAndDay[providerKey].errors[dayIndex] += getNumber(row.errors) ?? 0;
    });

    const availabilityMap: Record<string, number[]> = {};
    parsedProviders.forEach((provider) => {
      const key = provider.name.toLowerCase();
      const usage = requestByProviderAndDay[key];
      availabilityMap[key] = Array.from({ length: TREND_DAYS }, (_, index) => {
        const requests = usage?.requests[index] ?? 0;
        const errors = usage?.errors[index] ?? 0;
        if (requests <= 0) return 100;
        const availability = ((requests - errors) / requests) * 100;
        return Math.max(0, Math.min(100, availability));
      });
    });

    const metricsText = metricsResult.status === 'fulfilled' && typeof metricsResult.value === 'string'
      ? metricsResult.value
      : '';
    setPrometheusAvailable(metricsText.length > 0 && hasPrometheusProviderData(metricsText));

    const uptimeData = uptimeResult.status === 'fulfilled'
      ? (uptimeResult.value?.services ?? {})
      : {};

    setProviders(parsedProviders);
    setUsageByProvider(usageMap);
    setAvailabilityByProvider(availabilityMap);
    setProviderChecks((prev) => reconcileProviderChecks(prev, parsedProviders));
    setUptimeByService(uptimeData);
    setErrors(nextErrors);
    setLastUpdatedAt(new Date().toISOString());
    setLoading(false);
  }, []);

  const loadSystemDeps = useCallback(async () => {
    const seq = ++systemDepsSeqRef.current;
    setSystemDepsLoading(true);
    try {
      const result = await api.getSystemDependencies();
      if (seq !== systemDepsSeqRef.current) return; // stale response
      const items = Array.isArray(result.items) ? result.items : [];
      setSystemDeps(items);
      setSystemDepsCheckedAt(result.checked_at ?? new Date().toISOString());

      // Fetch 7-day uptime stats for each dependency (best-effort)
      const uptimeResults = await Promise.allSettled(
        items.map((dep) => api.getDependencyUptime(dep.name, 7)),
      );
      if (seq !== systemDepsSeqRef.current) return; // stale response
      const nextUptime: Record<string, DependencyUptimeStats> = {};
      uptimeResults.forEach((settledResult, index) => {
        if (settledResult.status === 'fulfilled' && settledResult.value) {
          nextUptime[items[index].name] = settledResult.value;
        }
      });
      setUptimeByDep(nextUptime);
    } catch {
      if (seq !== systemDepsSeqRef.current) return;
      setSystemDeps([]);
    } finally {
      if (seq === systemDepsSeqRef.current) setSystemDepsLoading(false);
    }
  }, []);

  useEffect(() => {
    const timerId = window.setTimeout(() => {
      void loadTelemetry();
      void loadSystemDeps();
    }, 0);
    return () => window.clearTimeout(timerId);
  }, [loadTelemetry, loadSystemDeps]);

  const handleRunAllChecks = useCallback(async () => {
    setRunningAllChecks(true);
    try {
      await runAllProviderChecks(providers.filter((provider) => provider.enabled));
    } finally {
      setRunningAllChecks(false);
    }
  }, [providers, runAllProviderChecks]);

  const handleTestProvider = useCallback(async (provider: Provider) => {
    const providerKey = provider.name.toLowerCase();
    setProviderChecks((prev) => ({
      ...prev,
      [providerKey]: {
        ...createDefaultProviderCheck(),
        ...prev[providerKey],
        testing: true,
      },
    }));

    const result = await runProviderCheck(provider);
    setProviderChecks((prev) => {
      const previous = prev[providerKey];
      return {
        ...prev,
        [providerKey]: {
          status: result.status,
          testing: false,
          lastCheckedAt: result.lastCheckedAt,
          lastSuccessAt:
            result.status === 'reachable'
              ? result.lastSuccessAt
              : previous?.lastSuccessAt ?? null,
          responseTimeMs: result.responseTimeMs,
          errorMessage: result.errorMessage,
        },
      };
    });
  }, [runProviderCheck]);

  const summary = useMemo(() => {
    const checks = Object.values(providerChecks);
    const reachable = checks.filter((check) => check.status === 'reachable').length;
    const unreachable = checks.filter((check) => check.status === 'unreachable').length;
    return {
      total: providers.length,
      enabled: providers.filter((provider) => provider.enabled).length,
      reachable,
      unreachable,
    };
  }, [providerChecks, providers]);

  const systemDepsSummary = useMemo(() => {
    const healthy = systemDeps.filter((d) => d.status === 'healthy').length;
    const degraded = systemDeps.filter((d) => d.status === 'degraded').length;
    const down = systemDeps.filter((d) => d.status === 'down').length;
    return { total: systemDeps.length, healthy, degraded, down };
  }, [systemDeps]);

  const hasLoadedData = providers.length > 0 || Object.keys(providerChecks).length > 0;

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold">External Dependencies</h1>
              <p className="text-muted-foreground">
                Passive usage telemetry loads on refresh. Live provider checks run only when triggered.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" onClick={() => { void loadTelemetry(); void loadSystemDeps(); }} loading={loading} loadingText="Refreshing...">
                <RefreshCw className="h-4 w-4" />
                Refresh Data
              </Button>
              <Button
                variant="outline"
                onClick={() => void handleRunAllChecks()}
                loading={runningAllChecks}
                loadingText="Running checks..."
              >
                <Activity className="h-4 w-4" />
                Run All Checks
              </Button>
            </div>
          </div>

          {lastUpdatedAt && (
            <p className="mb-4 text-sm text-muted-foreground">
              Last refreshed: {new Date(lastUpdatedAt).toLocaleString()}
            </p>
          )}

          {errors.length > 0 && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>
                <p className="font-medium">Some dependency telemetry is unavailable:</p>
                <ul className="mt-2 list-disc pl-4">
                  {errors.map((error) => (
                    <li key={error}>{error}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          <div className="mb-6 grid gap-4 md:grid-cols-3 lg:grid-cols-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">System Components</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{systemDepsSummary.total}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Components Healthy</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-green-600">{systemDepsSummary.healthy}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Configured Providers</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{summary.total}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Enabled Providers</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{summary.enabled}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Reachable</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-green-600">{summary.reachable}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">Unreachable</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold text-red-600">{summary.unreachable}</p>
              </CardContent>
            </Card>
          </div>

          <Card className="mb-6">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>System Dependencies</CardTitle>
                  <CardDescription>
                    Backend infrastructure health: database, embeddings, workflows, and more.
                  </CardDescription>
                </div>
                {systemDepsCheckedAt && (
                  <span className="text-xs text-muted-foreground">
                    Checked: {new Date(systemDepsCheckedAt).toLocaleTimeString()}
                  </span>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {systemDepsLoading && systemDeps.length === 0 ? (
                <div className="py-8 text-center text-muted-foreground">Loading system dependencies...</div>
              ) : systemDeps.length === 0 ? (
                <div className="py-8 text-center text-muted-foreground">
                  No system dependency data available. The backend may not support this endpoint yet.
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Component</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Latency</TableHead>
                      <TableHead>7d Uptime</TableHead>
                      <TableHead>Trend</TableHead>
                      <TableHead>Error</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {systemDeps.map((dep) => {
                      const uptime = uptimeByDep[dep.name];
                      return (
                        <TableRow
                          key={dep.name}
                          className={dep.status === 'down' ? 'bg-red-50/60' : undefined}
                        >
                          <TableCell>
                            <div className="flex items-center gap-2">
                              {dep.name.toLowerCase().includes('database') ? (
                                <Database className="h-4 w-4 text-muted-foreground" />
                              ) : (
                                <Server className="h-4 w-4 text-muted-foreground" />
                              )}
                              <span className="font-medium">{dep.name}</span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant={sysDepStatusVariant(dep.status)}>
                              {sysDepStatusIcon(dep.status)}
                              {sysDepStatusLabel(dep.status)}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {dep.latency_ms !== null && dep.latency_ms !== undefined
                              ? `${dep.latency_ms} ms`
                              : '\u2014'}
                          </TableCell>
                          <TableCell>
                            {uptime ? (
                              <Badge
                                variant={uptime.uptime_pct >= 99 ? 'default' : uptime.uptime_pct >= 95 ? 'secondary' : 'destructive'}
                              >
                                {uptime.uptime_pct.toFixed(1)}%
                              </Badge>
                            ) : (
                              <span className="text-xs text-muted-foreground">{'\u2014'}</span>
                            )}
                          </TableCell>
                          <TableCell>
                            {uptime ? (
                              <UptimeBar sparkline={uptime.sparkline} label={dep.name} />
                            ) : (
                              <span className="text-xs text-muted-foreground">{'\u2014'}</span>
                            )}
                          </TableCell>
                          <TableCell>
                            {dep.error ? (
                              <span className="text-sm text-red-700">{dep.error}</span>
                            ) : (
                              <span className="text-sm text-muted-foreground">{'\u2014'}</span>
                            )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>LLM Provider Health Grid</CardTitle>
              <CardDescription>
                Usage telemetry loads passively. Trigger live reachability checks per provider or run them in batch.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading && !hasLoadedData ? (
                <div className="py-8 text-center text-muted-foreground">Loading dependency telemetry...</div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Provider</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Last Checked</TableHead>
                      <TableHead>Response Time</TableHead>
                      <TableHead>Error Rate (24h)</TableHead>
                      <TableHead>Last Success</TableHead>
                      <TableHead className="text-right">Availability (7d)</TableHead>
                      <TableHead className="text-right">Uptime (30d)</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {providers.map((provider) => {
                      const key = provider.name.toLowerCase();
                      const usage = usageByProvider[key];
                      const check = providerChecks[key] ?? createDefaultProviderCheck();
                      const unreachable = check.status === 'unreachable';
                      const responseTimeValue = check.responseTimeMs !== null
                        ? `${check.responseTimeMs} ms`
                        : '—';
                      const errorRateValue = usage
                        ? formatErrorRate(usage.errors, usage.requests)
                        : '—';
                      const lastSuccessValue = formatSinceSuccess(check.lastSuccessAt);
                      const availabilitySeries = availabilityByProvider[key] ?? [];

                      return (
                        <TableRow
                          key={provider.name}
                          className={unreachable ? 'bg-red-50/60' : undefined}
                        >
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <Cpu className="h-4 w-4 text-muted-foreground" />
                              <div>
                                <p className="font-medium">{formatProviderName(provider.name)}</p>
                                <p className="text-xs text-muted-foreground">
                                  {provider.enabled ? 'Enabled' : 'Disabled'}
                                </p>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant={statusVariant(check.status)}>
                              {check.status === 'reachable' ? (
                                <Wifi className="mr-1 h-3 w-3" />
                              ) : check.status === 'unreachable' ? (
                                <WifiOff className="mr-1 h-3 w-3" />
                              ) : (
                                <AlertTriangle className="mr-1 h-3 w-3" />
                              )}
                              {statusLabel(check.status)}
                            </Badge>
                            {check.errorMessage && check.status === 'unreachable' ? (
                              <p className="mt-1 text-xs text-red-700">{check.errorMessage}</p>
                            ) : null}
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1 text-sm">
                              <Clock3 className="h-3.5 w-3.5 text-muted-foreground" />
                              {formatCheckTime(check.lastCheckedAt)}
                            </div>
                          </TableCell>
                          <TableCell>{responseTimeValue}</TableCell>
                          <TableCell>{errorRateValue}</TableCell>
                          <TableCell>{lastSuccessValue}</TableCell>
                          <TableCell className="text-right">
                            {prometheusAvailable ? (
                              <AvailabilitySparkline
                                providerName={provider.name}
                                series={availabilitySeries}
                              />
                            ) : (
                              <span className="text-xs text-muted-foreground">
                                Prometheus metrics unavailable
                              </span>
                            )}
                          </TableCell>
                          <TableCell className="text-right">
                            {(() => {
                              const uptimeHistory = uptimeByService[key] || uptimeByService[provider.name];
                              if (!uptimeHistory || uptimeHistory.length === 0) {
                                return <span className="text-xs text-muted-foreground">No data</span>;
                              }
                              const uptimeSeries = uptimeHistory.map((h) => h.uptime_pct);
                              return (
                                <AvailabilitySparkline
                                  providerName={`${provider.name} uptime`}
                                  series={uptimeSeries}
                                />
                              );
                            })()}
                          </TableCell>
                          <TableCell className="text-right">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => void handleTestProvider(provider)}
                              loading={check.testing}
                              loadingText="Testing..."
                              aria-label={`Test ${formatProviderName(provider.name)} connectivity`}
                            >
                              <Activity className="h-4 w-4" />
                              Test
                            </Button>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
