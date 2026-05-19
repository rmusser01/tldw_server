'use client';

import { useCallback, useEffect, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { CardSkeleton } from '@/components/ui/skeleton';
import { useToast } from '@/components/ui/toast';
import { api } from '@/lib/api-client';
import { isBillingEnabled } from '@/lib/billing';
import type { BillingAnalytics } from '@/types';
import { DollarSign, Users, TrendingUp, AlertTriangle } from 'lucide-react';

function formatCentsToCurrency(cents: number): string {
  return `$${(cents / 100).toFixed(2)}`;
}

function AnalyticsContent() {
  const billingActive = isBillingEnabled();
  const { error: showError } = useToast();
  const [analytics, setAnalytics] = useState<BillingAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');

  const fetchAnalytics = useCallback(async () => {
    setLoading(true);
    setLoadError('');
    try {
      const data = await api.getBillingAnalytics();
      setAnalytics(data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load billing analytics';
      setLoadError(message);
      showError(message);
    } finally {
      setLoading(false);
    }
  }, [showError]);

  useEffect(() => {
    if (billingActive) fetchAnalytics();
  }, [billingActive, fetchAnalytics]);

  if (!billingActive) {
    return (
      <Alert>
        <AlertDescription>
          Billing is not enabled. Set <code>NEXT_PUBLIC_BILLING_ENABLED=true</code> and configure
          the billing backend to view revenue analytics.
        </AlertDescription>
      </Alert>
    );
  }

  if (loading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    );
  }

  if (loadError) {
    return (
      <Alert variant="destructive">
        <AlertDescription>{loadError}</AlertDescription>
      </Alert>
    );
  }

  if (!analytics) {
    return (
      <Alert>
        <AlertDescription>No analytics data available.</AlertDescription>
      </Alert>
    );
  }

  const maxPlanCount = Math.max(
    ...analytics.plan_distribution.map((p) => p.count),
    1,
  );

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Monthly Recurring Revenue</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCentsToCurrency(analytics.mrr_cents)}/mo
            </div>
            <p className="text-xs text-muted-foreground">
              From {analytics.active_count} active subscription{analytics.active_count !== 1 ? 's' : ''}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Subscribers</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{analytics.subscriber_count}</div>
            <p className="text-xs text-muted-foreground">
              {analytics.active_count} active, {analytics.trialing_count} trialing
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Trial Conversion</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {analytics.trial_conversion_rate_pct}%
            </div>
            <p className="text-xs text-muted-foreground">
              Of trials that completed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Past Due</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{analytics.past_due_count}</div>
            <p className="text-xs text-muted-foreground">
              {analytics.canceled_count} canceled
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Plan Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Plan Distribution</CardTitle>
          <CardDescription>
            Subscriber breakdown by plan
          </CardDescription>
        </CardHeader>
        <CardContent>
          {analytics.plan_distribution.length === 0 ? (
            <p className="text-sm text-muted-foreground">No subscriptions yet.</p>
          ) : (
            <div className="space-y-3">
              {analytics.plan_distribution.map((entry) => (
                <div key={entry.plan_name} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{entry.plan_name}</span>
                    <span className="text-muted-foreground">
                      {entry.count} subscriber{entry.count !== 1 ? 's' : ''}
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary transition-all"
                      style={{ width: `${Math.round((entry.count / maxPlanCount) * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function BillingAnalyticsPage() {
  return (
    <PermissionGuard role={['admin', 'super_admin', 'owner']}>
      <ResponsiveLayout>
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Revenue Analytics</h1>
            <p className="text-muted-foreground">
              Billing metrics and subscription overview
            </p>
          </div>
          <AnalyticsContent />
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
