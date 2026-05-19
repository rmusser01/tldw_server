'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Select } from '@/components/ui/select';
import { EmptyState } from '@/components/ui/empty-state';
import { TableSkeleton } from '@/components/ui/skeleton';
import { PlanBadge } from '@/components/PlanBadge';
import { useToast } from '@/components/ui/toast';
import { api } from '@/lib/api-client';
import { isBillingEnabled } from '@/lib/billing';
import Link from 'next/link';
import { ExportMenu } from '@/components/ui/export-menu';
import { exportSubscriptions, type ExportFormat } from '@/lib/export';
import { formatDate } from '@/lib/formatters';
import type { Subscription, SubscriptionStatus } from '@/types';

const STATUS_VARIANTS: Record<SubscriptionStatus, 'default' | 'outline' | 'destructive' | 'secondary'> = {
  active: 'default',
  trialing: 'outline',
  past_due: 'destructive',
  canceled: 'secondary',
  incomplete: 'secondary',
};

const STATUS_OPTIONS: { value: string; label: string }[] = [
  { value: 'all', label: 'All Statuses' },
  { value: 'active', label: 'Active' },
  { value: 'trialing', label: 'Trialing' },
  { value: 'past_due', label: 'Past Due' },
  { value: 'canceled', label: 'Canceled' },
];

function AtRiskBadges({ sub }: { sub: Subscription }) {
  const reasons = sub.at_risk_reasons ?? [];
  if (!sub.at_risk || reasons.length === 0) return null;

  return (
    <span className="inline-flex flex-wrap gap-1 ml-2">
      {reasons.includes('past_due_extended') && (
        <Badge variant="destructive" data-testid="badge-past-due">
          Past Due {sub.days_past_due ? `${sub.days_past_due}d` : ''}
        </Badge>
      )}
      {reasons.includes('cancelling') && (
        <Badge variant="outline" className="border-orange-500 text-orange-700 dark:text-orange-400" data-testid="badge-cancelling">
          Cancelling
        </Badge>
      )}
      {reasons.includes('canceled') && (
        <Badge variant="secondary" data-testid="badge-canceled">
          Canceled
        </Badge>
      )}
    </span>
  );
}

function NeedsAttentionSection({ subscriptions }: { subscriptions: Subscription[] }) {
  const atRiskSubs = useMemo(
    () => subscriptions.filter((s) => s.at_risk),
    [subscriptions]
  );

  if (atRiskSubs.length === 0) return null;

  return (
    <Card className="border-destructive/50 bg-destructive/5 mb-4" data-testid="needs-attention-section">
      <CardHeader className="pb-2">
        <CardTitle className="text-destructive flex items-center gap-2 text-base">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-5 w-5"
            aria-hidden="true"
          >
            <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          Needs Attention ({atRiskSubs.length})
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2">
          {atRiskSubs.map((sub) => (
            <li key={sub.id} className="flex items-center gap-2 text-sm" data-testid="at-risk-item">
              <Link
                href={`/organizations/${sub.org_id}`}
                className="text-blue-600 hover:underline font-medium"
              >
                {sub.org_name ?? `Org ${sub.org_id}`}
              </Link>
              {sub.plan ? (
                <PlanBadge tier={sub.plan.tier} />
              ) : null}
              <Badge variant={STATUS_VARIANTS[sub.status] ?? 'secondary'}>
                {sub.status}
              </Badge>
              <AtRiskBadges sub={sub} />
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function OrgDisplayName({ sub }: { sub: Subscription }) {
  const name = sub.org_name;
  if (name) {
    return (
      <>
        {name}{' '}
        <span className="text-muted-foreground text-xs">({sub.org_id})</span>
      </>
    );
  }
  return <>Org {sub.org_id}</>;
}

function SubscriptionsPageContent() {
  const { error: showError } = useToast();
  const [subscriptions, setSubscriptions] = useState<Subscription[]>([]);
  const [orgNames, setOrgNames] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');

  const fetchSubscriptions = useCallback(async () => {
    setLoading(true);
    setLoadError('');
    try {
      const params: Record<string, string> = {};
      if (statusFilter !== 'all') {
        params.status = statusFilter;
      }
      const data = await api.getSubscriptions(params);
      setSubscriptions(data);
      // Batch-fetch org names for display
      try {
        const orgs = await api.getOrganizations();
        const names: Record<string, string> = {};
        (Array.isArray(orgs) ? orgs : []).forEach((o: { id: number | string; name: string }) => {
          names[String(o.id)] = o.name;
        });
        setOrgNames(names);
      } catch (err) {
        console.error('Failed to fetch organization names for subscriptions:', err);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load subscriptions';
      setLoadError(message);
      showError(message);
    } finally {
      setLoading(false);
    }
  }, [statusFilter, showError]);

  useEffect(() => {
    if (isBillingEnabled()) {
      fetchSubscriptions();
    } else {
      setLoading(false);
    }
  }, [fetchSubscriptions]);

  if (!isBillingEnabled()) {
    return (
      <ResponsiveLayout>
        <Alert>
          <AlertDescription>Billing is not enabled</AlertDescription>
        </Alert>
      </ResponsiveLayout>
    );
  }

  return (
    <ResponsiveLayout>
      {!loading && !loadError && (
        <NeedsAttentionSection subscriptions={subscriptions} />
      )}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <CardTitle>Subscriptions</CardTitle>
            <div className="flex items-center gap-2">
              <ExportMenu
                onExport={(format: ExportFormat) => exportSubscriptions(subscriptions, format)}
                disabled={subscriptions.length === 0}
              />
              <Select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="w-48"
                aria-label="Filter by status"
              >
                {STATUS_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loadError && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{loadError}</AlertDescription>
            </Alert>
          )}

          {!loading && (() => {
            const atRisk = subscriptions.filter(s => s.status === 'past_due' || s.status === 'incomplete');
            if (atRisk.length === 0) return null;
            return (
              <Alert className="mb-4 border-red-200 bg-red-50">
                <AlertDescription className="text-red-900">
                  <strong>{atRisk.length} subscription{atRisk.length !== 1 ? 's' : ''} need attention</strong> — {atRisk.filter(s => s.status === 'past_due').length} past due, {atRisk.filter(s => s.status === 'incomplete').length} incomplete.
                </AlertDescription>
              </Alert>
            );
          })()}

          {loading ? (
            <TableSkeleton rows={5} columns={6} />
          ) : subscriptions.length === 0 ? (
            <EmptyState title="No subscriptions found" />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Organization</TableHead>
                  <TableHead>Plan</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Indicators</TableHead>
                  <TableHead>Current Period</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {subscriptions.map((sub) => (
                  <TableRow key={sub.id} className={sub.at_risk ? 'bg-destructive/5' : undefined}>
                    <TableCell>
                      <Link
                        href={`/organizations/${sub.org_id}`}
                        className="text-blue-600 hover:underline"
                      >
                        <OrgDisplayName sub={sub} />
                      </Link>
                    </TableCell>
                    <TableCell>
                      {sub.plan ? (
                        <PlanBadge tier={sub.plan.tier} />
                      ) : (
                        sub.plan_id
                      )}
                    </TableCell>
                    <TableCell>
                      <Badge variant={STATUS_VARIANTS[sub.status] ?? 'secondary'}>
                        {sub.status}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <AtRiskBadges sub={sub} />
                    </TableCell>
                    <TableCell>
                      {formatDate(sub.current_period_start)} &ndash; {formatDate(sub.current_period_end)}
                    </TableCell>
                    <TableCell>{formatDate(sub.created_at)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </ResponsiveLayout>
  );
}

export default function SubscriptionsPage() {
  return (
    <PermissionGuard role={['admin', 'super_admin', 'owner']}>
      <SubscriptionsPageContent />
    </PermissionGuard>
  );
}
