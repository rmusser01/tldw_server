'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { RefreshCw } from 'lucide-react';

import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { formatLatency } from '@/lib/format';
import {
  getRouterAnalyticsAccess,
  getRouterAnalyticsConversations,
  getRouterAnalyticsLog,
  getRouterAnalyticsMeta,
  getRouterAnalyticsModels,
  getRouterAnalyticsNetwork,
  getRouterAnalyticsProviders,
  getRouterAnalyticsQuota,
  getRouterAnalyticsStatus,
  getRouterAnalyticsStatusBreakdowns,
} from '@/lib/router-analytics-client';
import type {
  RouterAnalyticsAccessResponse,
  RouterAnalyticsBreakdownRow,
  RouterAnalyticsBreakdownsResponse,
  RouterAnalyticsConversationsResponse,
  RouterAnalyticsLogResponse,
  RouterAnalyticsMetaResponse,
  RouterAnalyticsModelsResponse,
  RouterAnalyticsNetworkResponse,
  RouterAnalyticsProvidersResponse,
  RouterAnalyticsQuotaMetric,
  RouterAnalyticsQuotaResponse,
  RouterAnalyticsRange,
  RouterAnalyticsStatusResponse,
} from '@/lib/router-analytics-types';

type UsageTab = 'status' | 'quota' | 'providers' | 'access' | 'network' | 'models' | 'conversations' | 'log';

type UsageTabConfig = {
  value: UsageTab;
  label: string;
  implemented: boolean;
  tier: 'primary' | 'secondary';
};

const TABS: UsageTabConfig[] = [
  { value: 'status', label: 'Status', implemented: true, tier: 'primary' },
  { value: 'quota', label: 'Quota', implemented: true, tier: 'primary' },
  { value: 'models', label: 'Models', implemented: true, tier: 'primary' },
  { value: 'providers', label: 'Providers', implemented: true, tier: 'secondary' },
  { value: 'access', label: 'Access', implemented: true, tier: 'secondary' },
  { value: 'network', label: 'Network', implemented: true, tier: 'secondary' },
  { value: 'conversations', label: 'Conversations', implemented: true, tier: 'secondary' },
  { value: 'log', label: 'Log', implemented: true, tier: 'secondary' },
];

const RANGE_OPTIONS: RouterAnalyticsRange[] = ['realtime', '1h', '8h', '24h', '7d', '30d'];

type TimelineBucket = {
  ts: string;
  requests: number;
  totalTokens: number;
  topModel: string;
};

const normalizeErrorMessage = (error: unknown): string => {
  if (error instanceof Error && error.message.trim()) return error.message;
  if (typeof error === 'string' && error.trim()) return error;
  return 'Failed to load router analytics usage data.';
};

const formatNumber = (value?: number | null): string => {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return Number(value).toLocaleString();
};

const formatTokensPerSecond = (value?: number | null): string => {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return value.toFixed(2);
};

const formatPercent = (value?: number | null): string => {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return `${value.toFixed(1)}%`;
};

const formatQuotaMetric = (metric?: RouterAnalyticsQuotaMetric | null): string => {
  if (!metric) return '—';
  return `${formatNumber(metric.used)} / ${formatNumber(metric.limit)} (${formatPercent(metric.utilization_pct)})`;
};

const formatBucketTime = (value: string): string => {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const formatDateTime = (value?: string | null): string => {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
};

const renderTokenHeader = (shortLabel: string, fullLabel: string) => (
  <TableHead className="text-right" aria-label={fullLabel}>
    <abbr title={fullLabel}>{shortLabel}</abbr>
  </TableHead>
);

function BreakdownCard({ title, rows, keyLabel }: { title: string; rows: RouterAnalyticsBreakdownRow[]; keyLabel: string }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{keyLabel}</TableHead>
              <TableHead className="text-right">Requests</TableHead>
              <TableHead className="text-right" title="Prompt tokens processed">PP</TableHead>
              <TableHead className="text-right" title="Tokens generated (completion)">TG</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} className="text-center text-muted-foreground">
                  No data
                </TableCell>
              </TableRow>
            ) : (
              rows.map((row) => (
                <TableRow key={`${title}-${row.key}`}>
                  <TableCell className="font-medium">{row.label || row.key}</TableCell>
                  <TableCell className="text-right">{formatNumber(row.requests)}</TableCell>
                  <TableCell className="text-right">{formatNumber(row.prompt_tokens)}</TableCell>
                  <TableCell className="text-right">{formatNumber(row.total_tokens)}</TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

export default function UsagePage() {
  const [activeTab, setActiveTab] = useState<UsageTab>('status');
  const [range, setRange] = useState<RouterAnalyticsRange>('8h');
  const [provider, setProvider] = useState('');
  const [model, setModel] = useState('');
  const [tokenFilterValue, setTokenFilterValue] = useState('');

  const [statusPayload, setStatusPayload] = useState<RouterAnalyticsStatusResponse | null>(null);
  const [breakdownsPayload, setBreakdownsPayload] = useState<RouterAnalyticsBreakdownsResponse | null>(null);
  const [quotaPayload, setQuotaPayload] = useState<RouterAnalyticsQuotaResponse | null>(null);
  const [providersPayload, setProvidersPayload] = useState<RouterAnalyticsProvidersResponse | null>(null);
  const [accessPayload, setAccessPayload] = useState<RouterAnalyticsAccessResponse | null>(null);
  const [networkPayload, setNetworkPayload] = useState<RouterAnalyticsNetworkResponse | null>(null);
  const [modelsPayload, setModelsPayload] = useState<RouterAnalyticsModelsResponse | null>(null);
  const [conversationsPayload, setConversationsPayload] = useState<RouterAnalyticsConversationsResponse | null>(null);
  const [logPayload, setLogPayload] = useState<RouterAnalyticsLogResponse | null>(null);
  const [metaPayload, setMetaPayload] = useState<RouterAnalyticsMetaResponse | null>(null);

  const [statusLoading, setStatusLoading] = useState<boolean>(true);
  const [statusError, setStatusError] = useState<string>('');
  const [quotaLoading, setQuotaLoading] = useState<boolean>(false);
  const [quotaError, setQuotaError] = useState<string>('');
  const [providersLoading, setProvidersLoading] = useState<boolean>(false);
  const [providersError, setProvidersError] = useState<string>('');
  const [accessLoading, setAccessLoading] = useState<boolean>(false);
  const [accessError, setAccessError] = useState<string>('');
  const [networkLoading, setNetworkLoading] = useState<boolean>(false);
  const [networkError, setNetworkError] = useState<string>('');
  const [modelsLoading, setModelsLoading] = useState<boolean>(false);
  const [modelsError, setModelsError] = useState<string>('');
  const [conversationsLoading, setConversationsLoading] = useState<boolean>(false);
  const [conversationsError, setConversationsError] = useState<string>('');
  const [logLoading, setLogLoading] = useState<boolean>(false);
  const [logError, setLogError] = useState<string>('');
  const [refreshTick, setRefreshTick] = useState<number>(0);

  const selectedTokenId = useMemo<number | undefined>(() => {
    const rawValue = tokenFilterValue && tokenFilterValue !== '__all__' ? tokenFilterValue : '';
    if (!rawValue) return undefined;
    const parsed = Number(rawValue);
    if (!Number.isInteger(parsed) || parsed <= 0) return undefined;
    return parsed;
  }, [tokenFilterValue]);

  const statusQuery = useMemo(
    () => ({
      range,
      provider: provider || undefined,
      model: model || undefined,
      tokenId: selectedTokenId,
    }),
    [range, provider, model, selectedTokenId]
  );

  const loadStatusData = useCallback(async () => {
    setStatusLoading(true);
    setStatusError('');
    try {
      const [statusResponse, breakdownsResponse, metaResponse] = await Promise.all([
        getRouterAnalyticsStatus(statusQuery),
        getRouterAnalyticsStatusBreakdowns(statusQuery),
        getRouterAnalyticsMeta(),
      ]);
      setStatusPayload(statusResponse);
      setBreakdownsPayload(breakdownsResponse);
      setMetaPayload(metaResponse);
    } catch (loadError) {
      setStatusError(normalizeErrorMessage(loadError));
    } finally {
      setStatusLoading(false);
    }
  }, [statusQuery]);

  const loadQuotaData = useCallback(async () => {
    setQuotaLoading(true);
    setQuotaError('');
    try {
      const [quotaResponse, metaResponse] = await Promise.all([
        getRouterAnalyticsQuota(statusQuery),
        getRouterAnalyticsMeta(),
      ]);
      setQuotaPayload(quotaResponse);
      setMetaPayload(metaResponse);
    } catch (loadError) {
      setQuotaError(normalizeErrorMessage(loadError));
    } finally {
      setQuotaLoading(false);
    }
  }, [statusQuery]);

  const loadProvidersData = useCallback(async () => {
    setProvidersLoading(true);
    setProvidersError('');
    try {
      const [providersResponse, metaResponse] = await Promise.all([
        getRouterAnalyticsProviders(statusQuery),
        getRouterAnalyticsMeta(),
      ]);
      setProvidersPayload(providersResponse);
      setMetaPayload(metaResponse);
    } catch (loadError) {
      setProvidersError(normalizeErrorMessage(loadError));
    } finally {
      setProvidersLoading(false);
    }
  }, [statusQuery]);

  const loadAccessData = useCallback(async () => {
    setAccessLoading(true);
    setAccessError('');
    try {
      const [accessResponse, metaResponse] = await Promise.all([
        getRouterAnalyticsAccess(statusQuery),
        getRouterAnalyticsMeta(),
      ]);
      setAccessPayload(accessResponse);
      setMetaPayload(metaResponse);
    } catch (loadError) {
      setAccessError(normalizeErrorMessage(loadError));
    } finally {
      setAccessLoading(false);
    }
  }, [statusQuery]);

  const loadNetworkData = useCallback(async () => {
    setNetworkLoading(true);
    setNetworkError('');
    try {
      const [networkResponse, metaResponse] = await Promise.all([
        getRouterAnalyticsNetwork(statusQuery),
        getRouterAnalyticsMeta(),
      ]);
      setNetworkPayload(networkResponse);
      setMetaPayload(metaResponse);
    } catch (loadError) {
      setNetworkError(normalizeErrorMessage(loadError));
    } finally {
      setNetworkLoading(false);
    }
  }, [statusQuery]);

  const loadModelsData = useCallback(async () => {
    setModelsLoading(true);
    setModelsError('');
    try {
      const [modelsResponse, metaResponse] = await Promise.all([
        getRouterAnalyticsModels(statusQuery),
        getRouterAnalyticsMeta(),
      ]);
      setModelsPayload(modelsResponse);
      setMetaPayload(metaResponse);
    } catch (loadError) {
      setModelsError(normalizeErrorMessage(loadError));
    } finally {
      setModelsLoading(false);
    }
  }, [statusQuery]);

  const loadConversationsData = useCallback(async () => {
    setConversationsLoading(true);
    setConversationsError('');
    try {
      const [conversationsResponse, metaResponse] = await Promise.all([
        getRouterAnalyticsConversations(statusQuery),
        getRouterAnalyticsMeta(),
      ]);
      setConversationsPayload(conversationsResponse);
      setMetaPayload(metaResponse);
    } catch (loadError) {
      setConversationsError(normalizeErrorMessage(loadError));
    } finally {
      setConversationsLoading(false);
    }
  }, [statusQuery]);

  const loadLogData = useCallback(async () => {
    setLogLoading(true);
    setLogError('');
    try {
      const [logResponse, metaResponse] = await Promise.all([
        getRouterAnalyticsLog(statusQuery),
        getRouterAnalyticsMeta(),
      ]);
      setLogPayload(logResponse);
      setMetaPayload(metaResponse);
    } catch (loadError) {
      setLogError(normalizeErrorMessage(loadError));
    } finally {
      setLogLoading(false);
    }
  }, [statusQuery]);

  useEffect(() => {
    if (activeTab === 'status') {
      void loadStatusData();
      return;
    }
    if (activeTab === 'quota') {
      void loadQuotaData();
      return;
    }
    if (activeTab === 'providers') {
      void loadProvidersData();
      return;
    }
    if (activeTab === 'access') {
      void loadAccessData();
      return;
    }
    if (activeTab === 'network') {
      void loadNetworkData();
      return;
    }
    if (activeTab === 'models') {
      void loadModelsData();
      return;
    }
    if (activeTab === 'conversations') {
      void loadConversationsData();
      return;
    }
    if (activeTab === 'log') {
      void loadLogData();
    }
  }, [
    activeTab,
    loadStatusData,
    loadQuotaData,
    loadProvidersData,
    loadAccessData,
    loadNetworkData,
    loadModelsData,
    loadConversationsData,
    loadLogData,
    refreshTick,
  ]);

  const timelineBuckets = useMemo<TimelineBucket[]>(() => {
    if (!statusPayload?.series?.length) return [];

    const map = new Map<string, { requests: number; totalTokens: number; models: Map<string, number> }>();

    statusPayload.series.forEach((point) => {
      const ts = point.ts;
      const modelKey = point.model && point.model.trim().length > 0 ? point.model : 'unknown';
      const existing = map.get(ts) ?? { requests: 0, totalTokens: 0, models: new Map<string, number>() };
      existing.requests += Number(point.requests || 0);
      existing.totalTokens += Number(point.total_tokens || 0);
      existing.models.set(modelKey, (existing.models.get(modelKey) || 0) + Number(point.total_tokens || 0));
      map.set(ts, existing);
    });

    return [...map.entries()]
      .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime())
      .map(([ts, value]) => {
        let topModel = 'unknown';
        let topValue = -1;
        value.models.forEach((tokenCount, modelName) => {
          if (tokenCount > topValue) {
            topModel = modelName;
            topValue = tokenCount;
          }
        });

        return {
          ts,
          requests: value.requests,
          totalTokens: value.totalTokens,
          topModel,
        };
      });
  }, [statusPayload]);

  const maxTimelineTokens = useMemo(() => {
    if (!timelineBuckets.length) return 0;
    return Math.max(...timelineBuckets.map((bucket) => bucket.totalTokens));
  }, [timelineBuckets]);

  const currentTabLabel = useMemo(() => {
    return TABS.find((tab) => tab.value === activeTab)?.label || 'Tab';
  }, [activeTab]);

  return (
    <PermissionGuard role="admin">
      <ResponsiveLayout>
        <div className="space-y-6" data-testid="usage-router-analytics-page">
          <div className="rounded-lg border bg-card p-4">
            <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
              <div>
                <h1 className="text-2xl font-semibold">Usage Stats</h1>
                <p className="text-sm text-muted-foreground">
                  Router analytics overview for status operations.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
                <div className="space-y-1">
                  <Label htmlFor="usage-time-range">Time Range</Label>
                  <Select
                    id="usage-time-range"
                    aria-label="Time Range"
                    value={range}
                    onChange={(event) => setRange(event.target.value as RouterAnalyticsRange)}
                  >
                    {RANGE_OPTIONS.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="usage-provider-filter">Provider</Label>
                  <Select
                    id="usage-provider-filter"
                    aria-label="Provider"
                    value={provider || '__all__'}
                    onChange={(event) => setProvider(event.target.value === '__all__' ? '' : event.target.value)}
                  >
                    <option value="__all__">All providers</option>
                    {(metaPayload?.providers || []).map((entry) => (
                      <option key={entry.value} value={entry.value}>
                        {entry.label}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="usage-model-filter">Model</Label>
                  <Select
                    id="usage-model-filter"
                    aria-label="Model"
                    value={model || '__all__'}
                    onChange={(event) => setModel(event.target.value === '__all__' ? '' : event.target.value)}
                  >
                    <option value="__all__">All models</option>
                    {(metaPayload?.models || []).map((entry) => (
                      <option key={entry.value} value={entry.value}>
                        {entry.label}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="usage-token-filter">Token</Label>
                  <Select
                    id="usage-token-filter"
                    aria-label="Token"
                    value={tokenFilterValue || '__all__'}
                    onChange={(event) => setTokenFilterValue(event.target.value === '__all__' ? '' : event.target.value)}
                  >
                    <option value="__all__">All tokens</option>
                    {(metaPayload?.tokens || []).map((entry) => (
                      <option key={entry.value} value={entry.value}>
                        {entry.label}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="flex items-end">
                  <Button
                    type="button"
                    variant="outline"
                    className="w-full"
                    onClick={() => setRefreshTick((value) => value + 1)}
                  >
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Refresh
                  </Button>
                </div>
              </div>
            </div>
          </div>

          <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as UsageTab)}>
            <TabsList className="h-auto w-full justify-start overflow-x-auto rounded-md border bg-muted/40 p-1">
              {TABS.map((tab, index) => {
                const prevTab = index > 0 ? TABS[index - 1] : null;
                const showSeparator = prevTab && prevTab.tier === 'primary' && tab.tier === 'secondary';
                return (
                  <span key={tab.value} className="contents">
                    {showSeparator && (
                      <span className="mx-1 self-stretch w-px bg-border" aria-hidden="true" />
                    )}
                    <TabsTrigger value={tab.value} className="relative min-w-fit px-3 py-2">
                      {tab.label}
                      {!tab.implemented && (
                        <Badge variant="secondary" className="ml-2 text-[10px]">
                          Soon
                        </Badge>
                      )}
                    </TabsTrigger>
                  </span>
                );
              })}
            </TabsList>

            <TabsContent value="status" className="space-y-4">
              {statusError && (
                <Alert variant="destructive">
                  <AlertDescription>{statusError}</AlertDescription>
                </Alert>
              )}

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Requests</CardDescription>
                    <CardTitle>{formatNumber(statusPayload?.kpis.requests)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Prompt / Generated</CardDescription>
                    <CardTitle>
                      {formatNumber(statusPayload?.kpis.prompt_tokens)} / {formatNumber(statusPayload?.kpis.generated_tokens)}
                    </CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Avg latency ms</CardDescription>
                    <CardTitle>{formatLatency(statusPayload?.kpis.avg_latency_ms, { fallback: '—', precision: 1 })}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Avg gen tok/s</CardDescription>
                    <CardTitle>{formatTokensPerSecond(statusPayload?.kpis.avg_gen_toks_per_s)}</CardTitle>
                  </CardHeader>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Usage by model (tokens / bucket)</CardTitle>
                  <CardDescription>
                    Providers available: {statusPayload?.providers_available ?? 0} • Online: {statusPayload?.providers_online ?? 0}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {statusLoading ? (
                    <p className="text-sm text-muted-foreground">Loading status timeline...</p>
                  ) : timelineBuckets.length === 0 ? (
                    <p className="text-sm text-muted-foreground">No usage data in this window.</p>
                  ) : (
                    <div className="space-y-3">
                      {timelineBuckets.map((bucket) => {
                        const widthPercent = maxTimelineTokens > 0
                          ? Math.max(6, Math.round((bucket.totalTokens / maxTimelineTokens) * 100))
                          : 0;
                        return (
                          <div key={bucket.ts} className="space-y-1">
                            <div className="flex items-center justify-between text-xs text-muted-foreground">
                              <span>{formatBucketTime(bucket.ts)}</span>
                              <span>{bucket.topModel}</span>
                            </div>
                            <div className="h-2 rounded bg-muted">
                              <div className="h-2 rounded bg-blue-500" style={{ width: `${widthPercent}%` }} />
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {formatNumber(bucket.totalTokens)} tokens • {formatNumber(bucket.requests)} requests
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                <BreakdownCard title="Providers" rows={breakdownsPayload?.providers || []} keyLabel="Provider" />
                <BreakdownCard title="Models" rows={breakdownsPayload?.models || []} keyLabel="Model" />
                <BreakdownCard title="Token Names" rows={breakdownsPayload?.token_names || []} keyLabel="Token Name" />
                <BreakdownCard title="Remote IPs" rows={breakdownsPayload?.remote_ips || []} keyLabel="Remote IP" />
                <BreakdownCard title="User Agents" rows={breakdownsPayload?.user_agents || []} keyLabel="User-Agent" />
              </div>
            </TabsContent>

            <TabsContent value="quota" className="space-y-4">
              {quotaError && (
                <Alert variant="destructive">
                  <AlertDescription>{quotaError}</AlertDescription>
                </Alert>
              )}

              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Keys tracked</CardDescription>
                    <CardTitle>{formatNumber(quotaPayload?.summary.keys_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Keys over budget</CardDescription>
                    <CardTitle>{formatNumber(quotaPayload?.summary.keys_over_budget)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Budgeted keys</CardDescription>
                    <CardTitle>{formatNumber(quotaPayload?.summary.budgeted_keys)}</CardTitle>
                  </CardHeader>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Quota utilization</CardTitle>
                  <CardDescription>Day and 30-day budget usage for active keys in the selected filter window.</CardDescription>
                </CardHeader>
                <CardContent>
                  {quotaLoading ? (
                    <p className="text-sm text-muted-foreground">Loading quota data...</p>
                  ) : (quotaPayload?.items.length || 0) === 0 ? (
                    <p className="text-sm text-muted-foreground">No quota-linked key usage in this window.</p>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Token</TableHead>
                          <TableHead className="text-right">Requests</TableHead>
                          <TableHead className="text-right">Tokens</TableHead>
                          <TableHead className="text-right">Cost USD</TableHead>
                          <TableHead className="text-right">Day Tokens</TableHead>
                          <TableHead className="text-right">30d Tokens</TableHead>
                          <TableHead className="text-right">Day USD</TableHead>
                          <TableHead className="text-right">30d USD</TableHead>
                          <TableHead className="text-right">Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {(quotaPayload?.items || []).map((row) => (
                          <TableRow key={`quota-${row.key_id}`}>
                            <TableCell className="font-medium">{row.token_name}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.requests)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.total_tokens)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.total_cost_usd)}</TableCell>
                            <TableCell className="text-right">{formatQuotaMetric(row.day_tokens)}</TableCell>
                            <TableCell className="text-right">{formatQuotaMetric(row.month_tokens)}</TableCell>
                            <TableCell className="text-right">{formatQuotaMetric(row.day_usd)}</TableCell>
                            <TableCell className="text-right">{formatQuotaMetric(row.month_usd)}</TableCell>
                            <TableCell className="text-right">
                              {row.over_budget ? (
                                <Badge variant="destructive">Exceeded</Badge>
                              ) : (
                                <Badge variant="secondary">OK</Badge>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="providers" className="space-y-4">
              {providersError && (
                <Alert variant="destructive">
                  <AlertDescription>{providersError}</AlertDescription>
                </Alert>
              )}

              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Total providers</CardDescription>
                    <CardTitle>{formatNumber(providersPayload?.summary.providers_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Providers online</CardDescription>
                    <CardTitle>{formatNumber(providersPayload?.summary.providers_online)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Failover events</CardDescription>
                    <CardTitle>{formatNumber(providersPayload?.summary.failover_events)}</CardTitle>
                  </CardHeader>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Provider health and load</CardTitle>
                  <CardDescription>Request volume, latency, and success rate by provider in the selected window.</CardDescription>
                </CardHeader>
                <CardContent>
                  {providersLoading ? (
                    <p className="text-sm text-muted-foreground">Loading provider analytics...</p>
                  ) : (providersPayload?.items.length || 0) === 0 ? (
                    <p className="text-sm text-muted-foreground">No provider usage in this window.</p>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Provider</TableHead>
                          <TableHead className="text-right">Requests</TableHead>
                          <TableHead className="text-right" title="Prompt tokens processed">PP</TableHead>
                          <TableHead className="text-right" title="Tokens generated (completion)">TG</TableHead>
                          <TableHead className="text-right">Cost USD</TableHead>
                          <TableHead className="text-right">Latency ms</TableHead>
                          <TableHead className="text-right">Errors</TableHead>
                          <TableHead className="text-right">Success</TableHead>
                          <TableHead className="text-right">Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {(providersPayload?.items || []).map((row) => (
                          <TableRow key={`providers-${row.provider}`}>
                            <TableCell className="font-medium">{row.provider}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.requests)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.prompt_tokens)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.total_tokens)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.total_cost_usd)}</TableCell>
                            <TableCell className="text-right">{formatLatency(row.avg_latency_ms, { fallback: '—', precision: 1 })}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.errors)}</TableCell>
                            <TableCell className="text-right">{formatPercent(row.success_rate_pct)}</TableCell>
                            <TableCell className="text-right">
                              {row.online ? (
                                <Badge variant="secondary">Online</Badge>
                              ) : (
                                <Badge variant="destructive">Offline</Badge>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="access" className="space-y-4">
              {accessError && (
                <Alert variant="destructive">
                  <AlertDescription>{accessError}</AlertDescription>
                </Alert>
              )}

              <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Token names</CardDescription>
                    <CardTitle>{formatNumber(accessPayload?.summary.token_names_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Remote IPs</CardDescription>
                    <CardTitle>{formatNumber(accessPayload?.summary.remote_ips_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>User agents</CardDescription>
                    <CardTitle>{formatNumber(accessPayload?.summary.user_agents_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Anonymous requests</CardDescription>
                    <CardTitle>{formatNumber(accessPayload?.summary.anonymous_requests)}</CardTitle>
                  </CardHeader>
                </Card>
              </div>

              {accessLoading ? (
                <Card>
                  <CardContent className="pt-6">
                    <p className="text-sm text-muted-foreground">Loading access analytics...</p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                  <BreakdownCard title="Token Names (Access)" rows={accessPayload?.token_names || []} keyLabel="Token Name" />
                  <BreakdownCard title="Remote IPs (Access)" rows={accessPayload?.remote_ips || []} keyLabel="Remote IP" />
                  <BreakdownCard title="User Agents (Access)" rows={accessPayload?.user_agents || []} keyLabel="User-Agent" />
                </div>
              )}
            </TabsContent>

            <TabsContent value="network" className="space-y-4">
              {networkError && (
                <Alert variant="destructive">
                  <AlertDescription>{networkError}</AlertDescription>
                </Alert>
              )}

              <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Remote IPs</CardDescription>
                    <CardTitle>{formatNumber(networkPayload?.summary.remote_ips_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Endpoints</CardDescription>
                    <CardTitle>{formatNumber(networkPayload?.summary.endpoints_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Operations</CardDescription>
                    <CardTitle>{formatNumber(networkPayload?.summary.operations_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Error requests</CardDescription>
                    <CardTitle>{formatNumber(networkPayload?.summary.error_requests)}</CardTitle>
                  </CardHeader>
                </Card>
              </div>

              {networkLoading ? (
                <Card>
                  <CardContent className="pt-6">
                    <p className="text-sm text-muted-foreground">Loading network analytics...</p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                  <BreakdownCard title="Remote IPs (Network)" rows={networkPayload?.remote_ips || []} keyLabel="Remote IP" />
                  <BreakdownCard title="Endpoints (Network)" rows={networkPayload?.endpoints || []} keyLabel="Endpoint" />
                  <BreakdownCard title="Operations (Network)" rows={networkPayload?.operations || []} keyLabel="Operation" />
                </div>
              )}
            </TabsContent>

            <TabsContent value="models" className="space-y-4">
              {modelsError && (
                <Alert variant="destructive">
                  <AlertDescription>{modelsError}</AlertDescription>
                </Alert>
              )}

              <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Total models</CardDescription>
                    <CardTitle>{formatNumber(modelsPayload?.summary.models_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Models online</CardDescription>
                    <CardTitle>{formatNumber(modelsPayload?.summary.models_online)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Providers covered</CardDescription>
                    <CardTitle>{formatNumber(modelsPayload?.summary.providers_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Error requests</CardDescription>
                    <CardTitle>{formatNumber(modelsPayload?.summary.error_requests)}</CardTitle>
                  </CardHeader>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Model health and load</CardTitle>
                  <CardDescription>Request volume and quality metrics by model/provider pair.</CardDescription>
                </CardHeader>
                <CardContent>
                  {modelsLoading ? (
                    <p className="text-sm text-muted-foreground">Loading model analytics...</p>
                  ) : (modelsPayload?.items.length || 0) === 0 ? (
                    <p className="text-sm text-muted-foreground">No model usage in this window.</p>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Model</TableHead>
                          <TableHead>Provider</TableHead>
                          <TableHead className="text-right">Requests</TableHead>
                          <TableHead className="text-right" title="Prompt tokens processed">PP</TableHead>
                          <TableHead className="text-right" title="Tokens generated (completion)">TG</TableHead>
                          <TableHead className="text-right">Cost USD</TableHead>
                          <TableHead className="text-right">Latency ms</TableHead>
                          <TableHead className="text-right">Errors</TableHead>
                          <TableHead className="text-right">Success</TableHead>
                          <TableHead className="text-right">Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {(modelsPayload?.items || []).map((row) => (
                          <TableRow key={`models-${row.provider}-${row.model}`}>
                            <TableCell className="font-medium">{row.model}</TableCell>
                            <TableCell>{row.provider}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.requests)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.prompt_tokens)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.total_tokens)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.total_cost_usd)}</TableCell>
                            <TableCell className="text-right">{formatLatency(row.avg_latency_ms, { fallback: '—', precision: 1 })}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.errors)}</TableCell>
                            <TableCell className="text-right">{formatPercent(row.success_rate_pct)}</TableCell>
                            <TableCell className="text-right">
                              {row.online ? (
                                <Badge variant="secondary">Online</Badge>
                              ) : (
                                <Badge variant="destructive">Offline</Badge>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="conversations" className="space-y-4">
              {conversationsError && (
                <Alert variant="destructive">
                  <AlertDescription>{conversationsError}</AlertDescription>
                </Alert>
              )}

              <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Total conversations</CardDescription>
                    <CardTitle>{formatNumber(conversationsPayload?.summary.conversations_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Active conversations</CardDescription>
                    <CardTitle>{formatNumber(conversationsPayload?.summary.active_conversations)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Avg requests/conversation</CardDescription>
                    <CardTitle>{formatNumber(conversationsPayload?.summary.avg_requests_per_conversation)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Error requests</CardDescription>
                    <CardTitle>{formatNumber(conversationsPayload?.summary.error_requests)}</CardTitle>
                  </CardHeader>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Conversation activity</CardTitle>
                  <CardDescription>Per-conversation request volume and health in the selected filter window.</CardDescription>
                </CardHeader>
                <CardContent>
                  {conversationsLoading ? (
                    <p className="text-sm text-muted-foreground">Loading conversation analytics...</p>
                  ) : (conversationsPayload?.items.length || 0) === 0 ? (
                    <p className="text-sm text-muted-foreground">No conversation activity in this window.</p>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Conversation</TableHead>
                          <TableHead className="text-right">Requests</TableHead>
                          <TableHead className="text-right" title="Prompt tokens processed">PP</TableHead>
                          <TableHead className="text-right" title="Tokens generated (completion)">TG</TableHead>
                          <TableHead className="text-right">Cost USD</TableHead>
                          <TableHead className="text-right">Latency ms</TableHead>
                          <TableHead className="text-right">Errors</TableHead>
                          <TableHead className="text-right">Success</TableHead>
                          <TableHead className="text-right">Last Seen</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {(conversationsPayload?.items || []).map((row) => (
                          <TableRow key={`conversations-${row.conversation_id}`}>
                            <TableCell className="font-medium">{row.conversation_id}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.requests)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.prompt_tokens)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.total_tokens)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.total_cost_usd)}</TableCell>
                            <TableCell className="text-right">{formatLatency(row.avg_latency_ms, { fallback: '—', precision: 1 })}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.errors)}</TableCell>
                            <TableCell className="text-right">{formatPercent(row.success_rate_pct)}</TableCell>
                            <TableCell className="text-right">{formatDateTime(row.last_seen_at)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="log" className="space-y-4">
              {logError && (
                <Alert variant="destructive">
                  <AlertDescription>{logError}</AlertDescription>
                </Alert>
              )}

              <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Requests in window</CardDescription>
                    <CardTitle>{formatNumber(logPayload?.summary.requests_total)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Error requests</CardDescription>
                    <CardTitle>{formatNumber(logPayload?.summary.error_requests)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Estimated requests</CardDescription>
                    <CardTitle>{formatNumber(logPayload?.summary.estimated_requests)}</CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardDescription>Distinct request IDs</CardDescription>
                    <CardTitle>{formatNumber(logPayload?.summary.request_ids_total)}</CardTitle>
                  </CardHeader>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Request log</CardTitle>
                  <CardDescription>Recent router usage entries matching current filters.</CardDescription>
                </CardHeader>
                <CardContent>
                  {logLoading ? (
                    <p className="text-sm text-muted-foreground">Loading request log...</p>
                  ) : (logPayload?.items.length || 0) === 0 ? (
                    <p className="text-sm text-muted-foreground">No requests in this window.</p>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Time</TableHead>
                          <TableHead>Req ID</TableHead>
                          <TableHead>Provider / Model</TableHead>
                          <TableHead className="text-right">Status</TableHead>
                          <TableHead className="text-right">Latency ms</TableHead>
                          <TableHead className="text-right">Tokens</TableHead>
                          <TableHead className="text-right">Cost USD</TableHead>
                          <TableHead className="text-right">Estimated</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {(logPayload?.items || []).map((row) => (
                          <TableRow key={`log-${row.ts}-${row.request_id || 'unknown'}`}>
                            <TableCell>{formatDateTime(row.ts)}</TableCell>
                            <TableCell className="font-mono text-xs">{row.request_id || '—'}</TableCell>
                            <TableCell className="text-sm">
                              <div className="font-medium">{row.provider}</div>
                              <div className="text-muted-foreground">{row.model}</div>
                            </TableCell>
                            <TableCell className="text-right">{row.status ?? '—'}</TableCell>
                            <TableCell className="text-right">{formatLatency(row.latency_ms, { fallback: '—', precision: 1 })}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.total_tokens)}</TableCell>
                            <TableCell className="text-right">{formatNumber(row.total_cost_usd)}</TableCell>
                            <TableCell className="text-right">
                              {row.estimated ? (
                                <Badge variant="secondary">Yes</Badge>
                              ) : (
                                <Badge variant="outline">No</Badge>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {TABS.filter((tab) => !tab.implemented).map((tab) => (
              <TabsContent key={tab.value} value={tab.value}>
                <Card>
                  <CardHeader>
                    <CardTitle>{tab.label}</CardTitle>
                    <CardDescription>{tab.label} tab is coming soon.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground">
                      {currentTabLabel} will be delivered as part of the router analytics staged rollout.
                    </p>
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
