'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  DASHBOARD_ACTIVITY_RANGE_OPTIONS,
  type DashboardActivityRange,
} from '@/lib/dashboard-activity';
import {
  DASHBOARD_SUBSYSTEMS,
  type DashboardHealthStatus,
  type DashboardSystemHealth,
} from '@/lib/dashboard-health';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle,
  Clock,
  DollarSign,
  TrendingUp,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

type ActivityChartPoint = {
  name: string;
  requests: number;
  users: number;
  errorRate?: number | null;
  latencyP95?: number | null;
  errors?: number;
  latencyAvgMs?: number;
  costUsd?: number;
};

type ActivitySectionProps = {
  activityChartData: ActivityChartPoint[];
  systemHealth: DashboardSystemHealth;
  activityRange: DashboardActivityRange;
  onActivityRangeChange: (range: DashboardActivityRange) => void;
  loading?: boolean;
};

const getHealthIcon = (status: DashboardHealthStatus) => {
  switch (status) {
    case 'healthy':
      return <CheckCircle className="h-4 w-4 text-green-500" aria-hidden="true" />;
    case 'degraded':
      return <AlertTriangle className="h-4 w-4 text-yellow-500" aria-hidden="true" />;
    case 'down':
      return <AlertTriangle className="h-4 w-4 text-red-500" aria-hidden="true" />;
    default:
      return <Clock className="h-4 w-4 text-muted-foreground" aria-hidden="true" />;
  }
};

const getHealthBadge = (status: DashboardHealthStatus) => {
  switch (status) {
    case 'healthy':
      return <Badge className="bg-green-500" role="status">Healthy</Badge>;
    case 'degraded':
      return <Badge className="bg-yellow-500" role="status">Degraded</Badge>;
    case 'down':
      return <Badge variant="destructive" role="status">Down</Badge>;
    default:
      return <Badge variant="secondary" role="status">Unknown</Badge>;
  }
};

const formatCheckedAt = (checkedAt?: string) => {
  if (!checkedAt) {
    return 'Last checked: unavailable';
  }
  const parsed = new Date(checkedAt);
  if (Number.isNaN(parsed.getTime())) {
    return 'Last checked: unavailable';
  }
  return `Last checked: ${parsed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
};

const formatCacheHitRate = (cacheHitRatePct?: number | null): string => {
  if (cacheHitRatePct === undefined || cacheHitRatePct === null || !Number.isFinite(cacheHitRatePct)) {
    return 'Cache hit rate: unavailable';
  }
  return `Cache hit rate: ${cacheHitRatePct.toFixed(1)}%`;
};

const ACTIVITY_RANGE_DESCRIPTION: Record<DashboardActivityRange, string> = {
  '24h': 'Hourly API requests and active users over the past 24 hours',
  '7d': 'Daily API requests and active users over the past 7 days',
  '30d': 'Daily API requests and active users over the past 30 days',
};

export const ActivitySection = ({
  activityChartData,
  systemHealth,
  activityRange,
  onActivityRangeChange,
  loading = false,
}: ActivitySectionProps) => {
  const [showErrorRate, setShowErrorRate] = useState(false);
  const [showLatency, setShowLatency] = useState(false);
  const [showCost, setShowCost] = useState(false);

  const hasErrorData = activityChartData.some(
    (p) => p.errorRate !== undefined && p.errorRate !== null
  );
  const hasLatencyData = activityChartData.some(
    (p) => p.latencyP95 !== undefined && p.latencyP95 !== null
  );
  const hasCostData = activityChartData.some(
    (p) => p.costUsd !== undefined && p.costUsd !== null
  );

  return (
  <div className="grid gap-6 lg:grid-cols-3 mb-8">
    <Card className="lg:col-span-2">
      <CardHeader className="space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Activity Overview
            </CardTitle>
            <CardDescription>{ACTIVITY_RANGE_DESCRIPTION[activityRange]}</CardDescription>
          </div>
          <div className="flex items-center gap-2" role="group" aria-label="Activity range">
            {DASHBOARD_ACTIVITY_RANGE_OPTIONS.map((option) => (
              <Button
                key={option.value}
                type="button"
                size="sm"
                variant={activityRange === option.value ? 'default' : 'outline'}
                onClick={() => onActivityRangeChange(option.value)}
                disabled={loading}
                aria-pressed={activityRange === option.value}
              >
                {option.label}
              </Button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-3 flex-wrap" role="group" aria-label="Chart overlays">
          <Button
            type="button"
            size="sm"
            variant={showErrorRate ? 'default' : 'outline'}
            onClick={() => setShowErrorRate((v) => !v)}
            aria-pressed={showErrorRate}
            disabled={!hasErrorData}
            className="text-xs"
            title={hasErrorData ? 'Toggle error rate overlay' : 'Error rate data unavailable'}
          >
            <span className="inline-block h-2 w-2 rounded-full bg-red-500 mr-1.5" />
            Error Rate
          </Button>
          <Button
            type="button"
            size="sm"
            variant={showLatency ? 'default' : 'outline'}
            onClick={() => setShowLatency((v) => !v)}
            aria-pressed={showLatency}
            disabled={!hasLatencyData}
            className="text-xs"
            title={hasLatencyData ? 'Toggle latency overlay' : 'Latency data unavailable'}
          >
            <span className="inline-block h-2 w-2 rounded-full bg-orange-500 mr-1.5" />
            Latency p95
          </Button>
          <Button
            type="button"
            size="sm"
            variant={showCost ? 'default' : 'outline'}
            onClick={() => setShowCost((v) => !v)}
            aria-pressed={showCost}
            disabled={!hasCostData}
            className="text-xs"
            title={hasCostData ? 'Toggle cost overlay' : 'Cost time-series not available'}
          >
            <DollarSign className="h-3 w-3 mr-1.5 text-emerald-600" />
            Cost
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div role="img" aria-label={`Activity chart showing API requests and active users ${ACTIVITY_RANGE_DESCRIPTION[activityRange].toLowerCase()}`}>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%" minHeight={1} minWidth={1}>
              <AreaChart data={activityChartData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="name" className="text-xs" />
                <YAxis className="text-xs" yAxisId="left" />
                {(showErrorRate || showLatency || showCost) && (
                  <YAxis
                    className="text-xs"
                    yAxisId="right"
                    orientation="right"
                    tickFormatter={(v: number) =>
                      showLatency
                        ? `${v}ms`
                        : showCost
                          ? `$${v}`
                          : showErrorRate
                            ? `${v}%`
                            : `${v}`
                    }
                  />
                )}
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--background))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="requests"
                  stackId="1"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.3}
                  name="Requests"
                  yAxisId="left"
                />
                <Area
                  type="monotone"
                  dataKey="users"
                  stackId="2"
                  stroke="#10b981"
                  fill="#10b981"
                  fillOpacity={0.3}
                  name="Active Users"
                  yAxisId="left"
                />
                {showErrorRate && hasErrorData && (
                  <Line
                    type="monotone"
                    dataKey="errorRate"
                    stroke="#ef4444"
                    strokeWidth={2}
                    dot={false}
                    name="Error Rate (%)"
                    yAxisId="right"
                    connectNulls
                  />
                )}
                {showLatency && hasLatencyData && (
                  <Line
                    type="monotone"
                    dataKey="latencyP95"
                    stroke="#f97316"
                    strokeWidth={2}
                    dot={false}
                    name="Latency p95 (ms)"
                    yAxisId="right"
                    connectNulls
                  />
                )}
                {showCost && hasCostData && (
                  <Line
                    type="monotone"
                    dataKey="costUsd"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Cost ($)"
                    yAxisId="right"
                    connectNulls
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
        <details className="mt-2">
          <summary className="text-xs text-muted-foreground cursor-pointer">
            View chart data as table
          </summary>
          <div className="mt-1 max-h-48 overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  <th className="text-left px-2 py-1">Date</th>
                  <th className="text-right px-2 py-1">Requests</th>
                  <th className="text-right px-2 py-1">Active Users</th>
                  {hasErrorData && <th className="text-right px-2 py-1">Error Rate</th>}
                  {hasLatencyData && <th className="text-right px-2 py-1">Latency p95</th>}
                </tr>
              </thead>
              <tbody>
                {activityChartData.map((point, i) => (
                  <tr key={i} className="border-t border-border">
                    <td className="px-2 py-1">{point.name}</td>
                    <td className="text-right px-2 py-1">{point.requests}</td>
                    <td className="text-right px-2 py-1">{point.users}</td>
                    {hasErrorData && (
                      <td className="text-right px-2 py-1">
                        {point.errorRate != null ? `${point.errorRate.toFixed(2)}%` : '-'}
                      </td>
                    )}
                    {hasLatencyData && (
                      <td className="text-right px-2 py-1">
                        {point.latencyP95 != null ? `${point.latencyP95.toFixed(0)}ms` : '-'}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </CardContent>
    </Card>

    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          System Health
        </CardTitle>
        <CardDescription className="text-xs text-muted-foreground">
          Live health data from API subsystem health endpoints.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {DASHBOARD_SUBSYSTEMS.map((subsystem) => {
          const subsystemHealth = systemHealth[subsystem.key];
          const cacheHitRateLabel = subsystem.key === 'cache'
            ? formatCacheHitRate(subsystemHealth.cacheHitRatePct)
            : null;
          const showError = (subsystemHealth.status === 'degraded' || subsystemHealth.status === 'down')
            && subsystemHealth.errorMessage;
          return (
            <div key={subsystem.key} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
              <div className="min-w-0">
                <div className="flex items-center gap-3">
                  {getHealthIcon(subsystemHealth.status)}
                  <span className="font-medium">{subsystem.label}</span>
                </div>
                <p className="pl-7 text-xs text-muted-foreground">
                  {formatCheckedAt(subsystemHealth.checkedAt)}
                </p>
                {cacheHitRateLabel && (
                  <p className="pl-7 text-xs text-muted-foreground">{cacheHitRateLabel}</p>
                )}
                {showError && (
                  <p className="pl-7 text-xs text-destructive truncate max-w-[250px]" title={subsystemHealth.errorMessage!}>
                    {subsystemHealth.errorMessage}
                  </p>
                )}
              </div>
              {getHealthBadge(subsystemHealth.status)}
            </div>
          );
        })}
        <Link href="/monitoring" className="block">
          <Button variant="outline" className="w-full mt-2">
            View Details
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </Link>
      </CardContent>
    </Card>
  </div>
  );
};
