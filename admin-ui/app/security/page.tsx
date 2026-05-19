'use client';

import { useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { api } from '@/lib/api-client';
import { getKeyAgeIndicator, resolveUnifiedApiKeyStatus, type ApiKeyMetadataLike } from '@/lib/api-keys-hub';
import { formatDateTime } from '@/lib/format';
import { isSecurityHealthData } from '@/lib/type-guards';
import type { SecurityHealthData } from '@/types';
import { TableSkeleton } from '@/components/ui/skeleton';
import { ShieldAlert, ShieldCheck, RefreshCw, AlertTriangle, Key, Users, Lock } from 'lucide-react';
import Link from 'next/link';
import { CardSkeleton } from '@/components/ui/skeleton';

type SecurityAlertStatus = {
  total_alerts?: number;
  critical_alerts?: number;
  warning_alerts?: number;
  unacknowledged?: number;
  recent_alerts?: {
    id: string;
    severity: string;
    message: string;
    timestamp: string;
    source?: string;
  }[];
};

type SecurityRiskFactorSeverity = 'low' | 'medium' | 'high';

type SecurityRiskFactor = {
  key: 'users_without_mfa' | 'keys_over_180d' | 'failed_logins_24h' | 'suspicious_activity';
  label: string;
  description: string;
  value: number;
  weight: number;
  cap: number;
  contribution: number;
  severity: SecurityRiskFactorSeverity;
  remediationHref: string;
  remediationLabel: string;
};

type SecurityRiskBreakdownContext = {
  totalUsers: number;
  sampledUsers: number;
  estimatedFromSample: boolean;
  usersWithoutMfa: number;
  agedApiKeys: number;
  failedLogins24h: number;
  suspiciousActivity: number;
};

type SecurityRiskBreakdown = {
  factors: SecurityRiskFactor[];
  estimatedScore: number;
  context: SecurityRiskBreakdownContext;
};

const USER_PAGE_LIMIT = 100;
const MAX_USERS_FOR_KEY_SCAN = 200;

const toNonNegativeInt = (value: unknown) => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, Math.round(parsed));
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const isSecurityAlertStatus = (data: unknown): data is SecurityAlertStatus => {
  if (!isRecord(data)) return false;
  const knownKeys = [
    'total_alerts',
    'critical_alerts',
    'warning_alerts',
    'unacknowledged',
    'recent_alerts',
  ];
  if (!knownKeys.some((key) => key in data)) return false;
  const numberKeys = ['total_alerts', 'critical_alerts', 'warning_alerts', 'unacknowledged'];
  for (const key of numberKeys) {
    const value = data[key];
    if (value !== undefined && typeof value !== 'number') return false;
  }
  if (data.recent_alerts !== undefined) {
    if (!Array.isArray(data.recent_alerts)) return false;
    const validAlerts = data.recent_alerts.every((alert) =>
      isRecord(alert) &&
      typeof alert.id === 'string' &&
      typeof alert.severity === 'string' &&
      typeof alert.message === 'string' &&
      typeof alert.timestamp === 'string'
    );
    if (!validAlerts) return false;
  }
  return true;
};

const formatTimestamp = (dateStr?: string) => formatDateTime(dateStr, { fallback: '—' });

const getRiskLevelLabel = (score: number) => {
  if (score >= 70) return { label: 'High Risk', variant: 'destructive' as const };
  if (score >= 40) return { label: 'Medium Risk', variant: 'secondary' as const };
  return { label: 'Low Risk', variant: 'default' as const };
};

const getRiskLevelColor = (score: number) => {
  if (score >= 70) return 'text-red-500';
  if (score >= 40) return 'text-yellow-500';
  return 'text-green-500';
};

const getRiskBarColor = (score: number) => {
  if (score >= 70) return 'bg-red-500';
  if (score >= 40) return 'bg-yellow-500';
  return 'bg-green-500';
};

const getRiskFactorSeverity = (contribution: number): SecurityRiskFactorSeverity => {
  if (contribution >= 20) return 'high';
  if (contribution >= 10) return 'medium';
  return 'low';
};

const getRiskFactorSeverityVariant = (severity: SecurityRiskFactorSeverity): 'outline' | 'secondary' | 'destructive' => {
  if (severity === 'high') return 'destructive';
  if (severity === 'medium') return 'secondary';
  return 'outline';
};

const normalizeApiKeyList = (value: unknown): ApiKeyMetadataLike[] => {
  if (Array.isArray(value)) {
    return value as ApiKeyMetadataLike[];
  }
  if (!isRecord(value)) return [];
  if (Array.isArray(value.items)) return value.items as ApiKeyMetadataLike[];
  if (Array.isArray(value.keys)) return value.keys as ApiKeyMetadataLike[];
  return [];
};

const countAgedActiveKeys = (keys: ApiKeyMetadataLike[]) => {
  return keys.reduce((count, key) => {
    const status = resolveUnifiedApiKeyStatus(key.status, key.expires_at);
    if (status !== 'active') return count;
    const age = getKeyAgeIndicator(key.created_at ?? null);
    if (!age || age.ageDays <= 180) return count;
    return count + 1;
  }, 0);
};

const buildRiskBreakdown = (context: SecurityRiskBreakdownContext): SecurityRiskBreakdown => {
  const baseFactors = [
    {
      key: 'users_without_mfa',
      label: 'Users without MFA',
      description: 'Accounts without MFA protection increase account takeover risk.',
      value: context.usersWithoutMfa,
      weight: 3,
      cap: 40,
      contribution: Math.min(40, context.usersWithoutMfa * 3),
      severity: 'low',
      remediationHref: '/users?mfa=disabled',
      remediationLabel: 'Review MFA-disabled users',
    },
    {
      key: 'keys_over_180d',
      label: 'API keys older than 180 days',
      description: 'Long-lived active API keys should be rotated regularly.',
      value: context.agedApiKeys,
      weight: 2,
      cap: 25,
      contribution: Math.min(25, context.agedApiKeys * 2),
      severity: 'low',
      remediationHref: '/api-keys?status=active',
      remediationLabel: 'Review active API keys',
    },
    {
      key: 'failed_logins_24h',
      label: 'Failed logins (24h)',
      description: 'Repeated failed login attempts may indicate brute-force activity.',
      value: context.failedLogins24h,
      weight: 1,
      cap: 20,
      contribution: Math.min(20, context.failedLogins24h),
      severity: 'low',
      remediationHref: '/audit?action=login.failed',
      remediationLabel: 'Inspect failed login events',
    },
    {
      key: 'suspicious_activity',
      label: 'Suspicious activity (24h)',
      description: 'Security anomaly count from recent event analysis.',
      value: context.suspiciousActivity,
      weight: 4,
      cap: 20,
      contribution: Math.min(20, context.suspiciousActivity * 4),
      severity: 'low',
      remediationHref: '/monitoring',
      remediationLabel: 'Open monitoring alerts',
    },
  ] satisfies SecurityRiskFactor[];

  const factors: SecurityRiskFactor[] = baseFactors.map((factor) => ({
    ...factor,
    severity: getRiskFactorSeverity(factor.contribution),
  }));

  const estimatedScore = Math.min(
    100,
    factors.reduce((sum, factor) => sum + factor.contribution, 0),
  );

  return {
    factors,
    estimatedScore,
    context,
  };
};

const buildFallbackRiskContext = (health: SecurityHealthData): SecurityRiskBreakdownContext => ({
  totalUsers: 0,
  sampledUsers: 0,
  estimatedFromSample: false,
  usersWithoutMfa: 0,
  agedApiKeys: 0,
  failedLogins24h: toNonNegativeInt(health.failed_logins_24h),
  suspiciousActivity: toNonNegativeInt(health.suspicious_activity),
});

const loadRiskBreakdownContext = async (health: SecurityHealthData): Promise<SecurityRiskBreakdownContext> => {
  const mfaAdoptionRate = Math.min(100, Math.max(0, Number(health.mfa_adoption_rate ?? 0)));
  const failedLogins24h = toNonNegativeInt(health.failed_logins_24h);
  const suspiciousActivity = toNonNegativeInt(health.suspicious_activity);

  let totalUsers = 0;
  const sampledUsers: Array<{ id: number }> = [];

  let page = 1;
  let pages = 1;

  while (page <= pages && sampledUsers.length < MAX_USERS_FOR_KEY_SCAN) {
    const usersPage = await api.getUsersPage({
      page: String(page),
      limit: String(USER_PAGE_LIMIT),
    });

    totalUsers = Math.max(totalUsers, Number(usersPage.total ?? 0));
    pages = usersPage.pages > 0
      ? usersPage.pages
      : Math.max(1, Math.ceil((usersPage.total ?? 0) / Math.max(usersPage.limit || USER_PAGE_LIMIT, 1)));

    usersPage.items.forEach((user) => {
      if (sampledUsers.length < MAX_USERS_FOR_KEY_SCAN) {
        sampledUsers.push({ id: user.id });
      }
    });

    if (usersPage.items.length === 0) break;
    page += 1;
  }

  if (totalUsers === 0) {
    totalUsers = sampledUsers.length;
  }

  const usersWithoutMfa = totalUsers > 0
    ? Math.round(totalUsers * ((100 - mfaAdoptionRate) / 100))
    : 0;

  const keyResults = await Promise.allSettled(
    sampledUsers.map(async (user) => {
      const response = await api.getUserApiKeys(String(user.id), { include_revoked: true });
      return normalizeApiKeyList(response);
    })
  );

  const agedDetected = keyResults.reduce((count, result) => {
    if (result.status !== 'fulfilled') return count;
    return count + countAgedActiveKeys(result.value);
  }, 0);

  const sampledCount = sampledUsers.length;
  const estimatedFromSample = totalUsers > sampledCount && sampledCount > 0;
  const agedApiKeys = estimatedFromSample
    ? Math.round((agedDetected / sampledCount) * totalUsers)
    : agedDetected;

  return {
    totalUsers,
    sampledUsers: sampledCount,
    estimatedFromSample,
    usersWithoutMfa,
    agedApiKeys: Math.max(0, agedApiKeys),
    failedLogins24h,
    suspiciousActivity,
  };
};

export default function SecurityPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [securityHealth, setSecurityHealth] = useState<SecurityHealthData | null>(null);
  const [alertStatus, setAlertStatus] = useState<SecurityAlertStatus | null>(null);
  const [riskBreakdown, setRiskBreakdown] = useState<SecurityRiskBreakdown | null>(null);
  const [riskBreakdownLoading, setRiskBreakdownLoading] = useState(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      let nextError: string | null = null;
      const [healthResult, alertsResult] = await Promise.allSettled([
        api.getSecurityHealth(),
        api.getSecurityAlertStatus(),
      ]);

      if (healthResult.status === 'fulfilled' && isSecurityHealthData(healthResult.value)) {
        setSecurityHealth(healthResult.value);
        setRiskBreakdownLoading(true);
        void (async () => {
          try {
            const context = await loadRiskBreakdownContext(healthResult.value);
            setRiskBreakdown(buildRiskBreakdown(context));
          } catch {
            setRiskBreakdown(buildRiskBreakdown(buildFallbackRiskContext(healthResult.value)));
          } finally {
            setRiskBreakdownLoading(false);
          }
        })();
      } else {
        setSecurityHealth(null);
        setRiskBreakdown(null);
        setRiskBreakdownLoading(false);
        nextError = 'Security health data is unavailable. Risk score and summary metrics cannot be calculated.';
      }

      if (alertsResult.status === 'fulfilled' && isSecurityAlertStatus(alertsResult.value)) {
        setAlertStatus(alertsResult.value);
      } else {
        setAlertStatus({
          total_alerts: 0,
          critical_alerts: 0,
          warning_alerts: 0,
          unacknowledged: 0,
          recent_alerts: [],
        });
      }

      if (healthResult.status === 'rejected' && alertsResult.status === 'rejected') {
        nextError = 'Failed to load security health and alert data. Some endpoints may not be available.';
      }
      setError(nextError);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const hasSecurityHealth = securityHealth !== null;
  const riskScore = hasSecurityHealth ? (securityHealth?.risk_score ?? 0) : null;
  const riskScoreClamped = riskScore !== null
    ? Math.min(Math.max(riskScore, 0), 100)
    : null;
  const riskLevel = riskScore !== null ? getRiskLevelLabel(riskScore) : null;
  const riskColor = riskScore !== null ? getRiskLevelColor(riskScore) : '';
  const failedLogins24h = securityHealth?.failed_logins_24h ?? null;

  const renderSecurityMetric = (value: number | null | undefined, suffix = '') =>
    hasSecurityHealth && value !== null && value !== undefined ? `${value}${suffix}` : '—';

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold flex items-center gap-2">
                <ShieldAlert className="h-8 w-8" />
                Security Dashboard
              </h1>
              <p className="text-muted-foreground">
                Monitor security posture, alerts, and authentication activity
              </p>
            </div>
            <Button variant="outline" onClick={() => { void loadData(); }} loading={loading} loadingText="Refreshing...">
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>

          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Risk Score Card */}
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {riskScore !== null && riskScore < 40 ? (
                  <ShieldCheck className="h-5 w-5 text-green-500" />
                ) : (
                  <ShieldAlert className="h-5 w-5 text-yellow-500" />
                )}
                Security Risk Assessment
              </CardTitle>
              <CardDescription>
                {hasSecurityHealth ? 'Overall security posture score' : 'Security health endpoint unavailable'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!hasSecurityHealth ? (
                <div className="space-y-3" data-testid="security-risk-unavailable">
                  <Badge variant="outline">Unavailable</Badge>
                  <p className="text-sm text-muted-foreground">Risk score unavailable.</p>
                  <p className="text-sm text-muted-foreground">
                    Security health data is unavailable. Risk score and summary metrics cannot be calculated.
                  </p>
                </div>
              ) : (
                <div className="flex items-center gap-8">
                  <div>
                    <div className={`text-6xl font-bold ${riskColor}`}>
                      {riskScoreClamped}
                    </div>
                    {riskLevel ? (
                      <Badge variant={riskLevel.variant} className="mt-2">
                        {riskLevel.label}
                      </Badge>
                    ) : null}
                  </div>
                  <div className="flex-1">
                    <div
                      className="h-4 bg-muted rounded-full overflow-hidden"
                      role="progressbar"
                      aria-valuenow={riskScoreClamped ?? 0}
                      aria-valuemin={0}
                      aria-valuemax={100}
                      aria-label={`Security risk score: ${riskScoreClamped ?? 0}`}
                    >
                      <div
                        className={`h-full transition-all ${getRiskBarColor(riskScore ?? 0)}`}
                        style={{ width: `${riskScoreClamped ?? 0}%` }}
                      />
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      {securityHealth?.last_security_scan
                        ? `Last scan: ${formatTimestamp(securityHealth.last_security_scan)}`
                        : 'Risk score based on recent security events and configuration'}
                    </p>

                    <div className="mt-4 rounded-md border p-3">
                      <div className="mb-3 flex items-center justify-between">
                        <h3 className="text-sm font-semibold">Risk factor breakdown</h3>
                        <Badge variant="outline" data-testid="risk-breakdown-estimated-score">
                          Estimated {riskBreakdown?.estimatedScore ?? 0}/100
                        </Badge>
                      </div>
                      <div className="mb-3 flex flex-wrap gap-3 text-xs text-muted-foreground" data-testid="risk-weight-legend">
                        <span>Weight legend:</span>
                        <span title="Each user without MFA adds 3 points (cap 40)">MFA = <strong className="text-foreground">3</strong></span>
                        <span title="Each aged API key adds 2 points (cap 25)">Keys = <strong className="text-foreground">2</strong></span>
                        <span title="Each failed login adds 1 point (cap 20)">Failed logins = <strong className="text-foreground">1</strong></span>
                        <span title="Each suspicious event adds 4 points (cap 20)">Suspicious = <strong className="text-foreground">4</strong></span>
                      </div>
                      {riskBreakdownLoading ? (
                        <div className="text-sm text-muted-foreground">Loading risk factor details...</div>
                      ) : riskBreakdown ? (
                        <>
                          <div className="overflow-x-auto rounded-md border">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Factor</TableHead>
                                  <TableHead>Value</TableHead>
                                  <TableHead>Weight</TableHead>
                                  <TableHead>Contribution</TableHead>
                                  <TableHead>Severity</TableHead>
                                  <TableHead>Remediation</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {riskBreakdown.factors.map((factor) => (
                                  <TableRow key={factor.key} data-testid={`risk-factor-${factor.key}`}>
                                    <TableCell>
                                      <div className="font-medium">{factor.label}</div>
                                      <div className="text-xs text-muted-foreground">{factor.description}</div>
                                    </TableCell>
                                    <TableCell>{factor.value}</TableCell>
                                    <TableCell>{factor.weight}</TableCell>
                                    <TableCell>
                                      <div>{factor.contribution}</div>
                                      <div className="text-xs text-muted-foreground">
                                        min({factor.cap}, {factor.value} x {factor.weight})
                                      </div>
                                    </TableCell>
                                    <TableCell>
                                      <Badge variant={getRiskFactorSeverityVariant(factor.severity)}>
                                        {factor.severity}
                                      </Badge>
                                    </TableCell>
                                    <TableCell>
                                      <Link href={factor.remediationHref} className="text-primary underline text-sm">
                                        {factor.remediationLabel}
                                      </Link>
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </div>
                          <p className="mt-2 text-xs text-muted-foreground">
                            Calculation: {riskBreakdown.factors.map((factor) => factor.contribution).join(' + ')} = {riskBreakdown.estimatedScore}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {riskBreakdown.context.estimatedFromSample
                              ? `API key-age factor estimated from ${riskBreakdown.context.sampledUsers} sampled users out of ${riskBreakdown.context.totalUsers}.`
                              : `API key-age factor calculated from ${riskBreakdown.context.sampledUsers} users.`}
                          </p>
                        </>
                      ) : (
                        <div className="text-sm text-muted-foreground">Risk factor data is unavailable.</div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Stats Grid */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4" />
                  Security Events (24h)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {renderSecurityMetric(securityHealth?.recent_security_events)}
                </div>
                <p className="text-xs text-muted-foreground">
                  {hasSecurityHealth
                    ? `${securityHealth?.suspicious_activity ?? 0} suspicious`
                    : 'Unavailable'}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Lock className="h-4 w-4" />
                  Failed Logins (24h)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-3xl font-bold ${
                  hasSecurityHealth && (failedLogins24h ?? 0) > 10 ? 'text-red-500' : ''
                }`}>
                  {renderSecurityMetric(failedLogins24h)}
                </div>
                <p className="text-xs text-muted-foreground">
                  {hasSecurityHealth ? 'Potential brute force attempts' : 'Unavailable'}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Users className="h-4 w-4" />
                  MFA Adoption
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {renderSecurityMetric(securityHealth?.mfa_adoption_rate, '%')}
                </div>
                <p className="text-xs text-muted-foreground">
                  {hasSecurityHealth ? 'Users with MFA enabled' : 'Unavailable'}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Key className="h-4 w-4" />
                  Active Sessions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {renderSecurityMetric(securityHealth?.active_sessions)}
                </div>
                <p className="text-xs text-muted-foreground">
                  {hasSecurityHealth
                    ? `${securityHealth?.api_keys_active ?? 0} API keys in use`
                    : 'Unavailable'}
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            {/* Alert Summary */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  Security Alerts
                </CardTitle>
                <CardDescription>
                  {alertStatus?.unacknowledged ?? 0} unacknowledged alerts
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="text-center p-3 rounded-lg bg-red-50 dark:bg-red-900/20">
                    <div className="text-2xl font-bold text-red-500">
                      {alertStatus?.critical_alerts ?? 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Critical</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-yellow-50 dark:bg-yellow-900/20">
                    <div className="text-2xl font-bold text-yellow-500">
                      {alertStatus?.warning_alerts ?? 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Warning</div>
                  </div>
                  <div className="text-center p-3 rounded-lg bg-muted">
                    <div className="text-2xl font-bold">
                      {alertStatus?.total_alerts ?? 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Total</div>
                  </div>
                </div>
                <Link href="/monitoring">
                  <Button variant="outline" className="w-full">
                    View All Alerts
                  </Button>
                </Link>
              </CardContent>
            </Card>

            {/* Recent Security Events */}
            <Card>
              <CardHeader>
                <CardTitle>Recent Security Events</CardTitle>
                <CardDescription>Latest security-related activity</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <TableSkeleton rows={3} columns={3} />
                ) : !alertStatus?.recent_alerts?.length ? (
                  <div className="text-center text-muted-foreground py-8">
                    <ShieldCheck className="h-12 w-12 mx-auto mb-2 text-green-500" />
                    <p>No recent security events</p>
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Severity</TableHead>
                        <TableHead>Event</TableHead>
                        <TableHead>Time</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {alertStatus.recent_alerts.slice(0, 5).map((alert) => (
                        <TableRow key={alert.id}>
                          <TableCell>
                            <Badge
                              variant={
                                alert.severity === 'critical'
                                  ? 'destructive'
                                  : alert.severity === 'warning'
                                    ? 'secondary'
                                    : 'outline'
                              }
                            >
                              {alert.severity}
                            </Badge>
                          </TableCell>
                          <TableCell className="max-w-[200px] truncate" title={alert.message}>
                            {alert.message}
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatTimestamp(alert.timestamp)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
                <Link href="/audit?filter=security" className="block mt-4">
                  <Button variant="ghost" className="w-full">
                    View Security Audit Logs
                  </Button>
                </Link>
              </CardContent>
            </Card>
          </div>

          {/* Quick Actions */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>Security Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-4">
                <Button variant="outline" onClick={() => router.push('/users')}>
                  <Users className="mr-2 h-4 w-4" />
                  Manage Users
                </Button>
                <Button variant="outline" onClick={() => router.push('/api-keys')}>
                  <Key className="mr-2 h-4 w-4" />
                  API Key Management
                </Button>
                <Button variant="outline" onClick={() => router.push('/audit')}>
                  Audit Logs
                </Button>
                <Button variant="outline" onClick={() => router.push('/roles')}>
                  Roles & Permissions
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
