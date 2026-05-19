'use client';

import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { StatsCardSkeleton } from '@/components/ui/skeleton';
import type { DashboardUIStats } from '@/lib/dashboard';
import type { DashboardOperationalKpis, MetricTrend } from '@/lib/dashboard-kpis';
import { formatLatency } from '@/lib/format';
import {
  ArrowDownRight,
  ArrowUpRight,
  Building2,
  Clock3,
  Coins,
  Cpu,
  Database,
  HardDrive,
  Minus,
  Monitor,
  Plug,
  Users,
  Wallet,
  Workflow,
  XCircle,
} from 'lucide-react';

export type RealtimeStats = {
  active_sessions: number;
  tokens_today: { prompt: number; completion: number; total: number };
};

type StatsGridProps = {
  loading: boolean;
  stats: DashboardUIStats;
  storagePercentage: number;
  operationalKpis: DashboardOperationalKpis;
  realtimeStats?: RealtimeStats | null;
  cacheHitRatePct?: number | null;
};

const CARD_COUNT = 12;

function formatCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

const formatTrendValue = (trend: MetricTrend): string => {
  if (trend.percentChange !== null) {
    return `${Math.abs(trend.percentChange).toFixed(1)}%`;
  }
  return Math.abs(trend.delta).toFixed(1);
};

const TrendBadge = ({
  trend,
  label,
  preferLower = false,
  fallback = 'No baseline',
}: {
  trend: MetricTrend | null;
  label: string;
  preferLower?: boolean;
  fallback?: string;
}) => {
  if (!trend) {
    return <p className="text-xs text-muted-foreground">{fallback}</p>;
  }

  if (trend.direction === 'flat') {
    return (
      <p className="inline-flex items-center gap-1 text-xs text-muted-foreground">
        <Minus className="h-3.5 w-3.5" />
        {label} unchanged
      </p>
    );
  }

  const isImprovement = preferLower ? trend.direction === 'down' : trend.direction === 'up';
  const Icon = trend.direction === 'up' ? ArrowUpRight : ArrowDownRight;
  const directionLabel = trend.direction === 'up' ? 'up' : 'down';
  const trendText = formatTrendValue(trend);

  return (
    <p className={`inline-flex items-center gap-1 text-xs ${isImprovement ? 'text-emerald-600' : 'text-amber-600'}`}>
      <Icon className="h-3.5 w-3.5" />
      {trendText} {directionLabel} vs {label}
    </p>
  );
};

const formatUsd = (value: number | null): string => {
  if (value === null) return 'N/A';
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value);
};

const formatPercentage = (value: number | null): string => {
  if (value === null) return 'N/A';
  return `${value.toFixed(2)}%`;
};

const formatLatencyValue = (value: number | null): string =>
  formatLatency(value, { fallback: 'N/A', precision: 0 });

const formatJobsSummary = (
  activeJobs: number | null,
  queuedJobs: number | null,
  failedJobs: number | null
): string => {
  if (activeJobs === null || queuedJobs === null || failedJobs === null) return 'N/A';
  return `${activeJobs} / ${queuedJobs} / ${failedJobs}`;
};

const formatLiveValue = (value: number | null): string => (
  value === null ? 'unavailable' : `${value}`
);

const formatCompactNumber = (value: number): string => {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return `${value}`;
};

const formatCacheHitRateKpi = (value: number | null | undefined): string => {
  if (value === null || value === undefined || !Number.isFinite(value)) return 'N/A';
  return `${value.toFixed(1)}%`;
};

const buildStatsLiveSummary = (
  stats: DashboardUIStats,
  storagePercentage: number,
  operationalKpis: DashboardOperationalKpis,
  realtimeStats?: RealtimeStats | null,
  cacheHitRatePct?: number | null,
): string => {
  const cacheLabel = cacheHitRatePct !== null && cacheHitRatePct !== undefined
    ? `cache hit rate ${cacheHitRatePct.toFixed(1)} percent, `
    : '';
  const base =
    `Dashboard metrics updated. ${stats.users} total users, ${stats.activeUsers} active users, ` +
    `${stats.organizations} organizations, ${stats.enabledProviders} enabled providers, ` +
    `${storagePercentage.toFixed(0)} percent storage used, ` +
    `${cacheLabel}` +
    `error rate ${formatLiveValue(operationalKpis.errorRatePct)} percent, ` +
    `queue depth ${formatLiveValue(operationalKpis.queueDepth)}.`;
  if (realtimeStats) {
    return (
      base +
      ` ${realtimeStats.active_sessions} active sessions, ` +
      `${formatCompactNumber(realtimeStats.tokens_today.total)} tokens consumed.`
    );
  }
  return base;
};

const LATENCY_UNAVAILABLE_TOOLTIP =
  'Latency p95 unavailable. Backend must expose http_request_duration_seconds histogram at /metrics/text.';

export const StatsGrid = ({
  loading,
  stats,
  storagePercentage,
  operationalKpis,
  realtimeStats,
  cacheHitRatePct,
}: StatsGridProps) => (
  <div
    className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4"
    role="status"
    aria-live="polite"
    aria-atomic="true"
    data-testid="dashboard-stats-live-region"
  >
    <p className="sr-only">
      {loading ? 'Dashboard metrics loading.' : buildStatsLiveSummary(stats, storagePercentage, operationalKpis, realtimeStats, cacheHitRatePct)}
    </p>
    {loading ? (
      Array.from({ length: CARD_COUNT }).map((_, index) => (
        <StatsCardSkeleton key={`stats-skeleton-${index}`} />
      ))
    ) : (
      <>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.users}</div>
            <p className="text-xs text-muted-foreground">
              {stats.activeUsers} active
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Organizations</CardTitle>
            <Building2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.organizations}</div>
            <p className="text-xs text-muted-foreground">
              {stats.teams} teams
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">LLM Providers</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.enabledProviders}</div>
            <p className="text-xs text-muted-foreground">
              of {stats.providers} configured
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Storage</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {`${(stats.storageUsedMb / 1024).toFixed(1)} GB`}
            </div>
            <div className="mt-2">
              <div
                className="h-2 w-full rounded-full bg-gray-200"
                role="progressbar"
                aria-valuenow={Math.round(storagePercentage)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`Storage usage: ${storagePercentage.toFixed(0)}%`}
              >
                <div
                  className={`h-2 rounded-full ${
                    storagePercentage > 90 ? 'bg-red-500'
                      : storagePercentage > 70 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${storagePercentage}%` }}
                />
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {storagePercentage.toFixed(0)}% used
              </p>
              <Link
                href="/data-ops"
                className="mt-1 inline-block text-xs text-primary hover:underline"
                data-testid="storage-detail-link"
              >
                View details
              </Link>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Sessions</CardTitle>
            <Monitor className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {realtimeStats ? realtimeStats.active_sessions : 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground">ACP agent sessions</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Token Consumption</CardTitle>
            <Coins className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {realtimeStats
                ? formatCompactNumber(realtimeStats.tokens_today.total)
                : 'N/A'}
            </div>
            {realtimeStats ? (
              <p className="text-xs text-muted-foreground">
                {formatCompactNumber(realtimeStats.tokens_today.prompt)} prompt /{' '}
                {formatCompactNumber(realtimeStats.tokens_today.completion)} completion
              </p>
            ) : (
              <p className="text-xs text-muted-foreground">prompt / completion</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Request Latency (p95)</CardTitle>
            <Clock3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="space-y-1">
            <div
              className="text-2xl font-bold"
              title={operationalKpis.latencyP95Ms === null ? LATENCY_UNAVAILABLE_TOOLTIP : undefined}
            >
              {formatLatencyValue(operationalKpis.latencyP95Ms)}
            </div>
            <TrendBadge
              trend={operationalKpis.latencyTrend}
              label="previous day"
              preferLower
              fallback="Requires /metrics histogram"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Error Rate</CardTitle>
            <XCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="text-2xl font-bold">{formatPercentage(operationalKpis.errorRatePct)}</div>
            <TrendBadge
              trend={operationalKpis.errorRateTrend}
              label="previous day"
              preferLower
            />
            <Link
              href="/audit?action=error"
              className="inline-block text-xs text-primary hover:underline"
              data-testid="error-drilldown-link"
            >
              View errors
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Daily LLM Cost</CardTitle>
            <Wallet className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="text-2xl font-bold">{formatUsd(operationalKpis.dailyCostUsd)}</div>
            <TrendBadge
              trend={operationalKpis.dailyCostTrend}
              label="previous day"
              preferLower
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Jobs &amp; Queue</CardTitle>
            <Workflow className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="text-2xl font-bold">
              {formatJobsSummary(
                operationalKpis.activeJobs,
                operationalKpis.queuedJobs,
                operationalKpis.failedJobs
              )}
            </div>
            <p className="text-xs text-muted-foreground">active / queued / failed</p>
            <TrendBadge
              trend={operationalKpis.queueDepthTrend}
              label="last refresh"
              preferLower
              fallback="No prior queue snapshot"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cache Hit Rate</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="text-2xl font-bold">{formatCacheHitRateKpi(cacheHitRatePct)}</div>
            <p className="text-xs text-muted-foreground">RAG cache hit rate</p>
          </CardContent>
        </Card>

        {stats.mcpInvocationsToday != null ? (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">MCP Calls</CardTitle>
              <Plug className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatCompact(stats.mcpInvocationsToday)}</div>
              <p className="text-xs text-muted-foreground">Tool invocations today</p>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">MCP Tool Invocations</CardTitle>
              <Plug className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent className="space-y-1">
              <div
                className="text-2xl font-bold"
                title="Requires MCP telemetry"
              >
                N/A
              </div>
              <p className="text-xs text-muted-foreground">Requires MCP telemetry</p>
            </CardContent>
          </Card>
        )}
      </>
    )}
  </div>
);
