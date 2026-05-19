'use client';

import type { Invoice, OrgUsageSummary, PlanTier, Subscription } from '@/types';
import { formatDate } from '@/lib/format';
import { logger } from '@/lib/logger';

/**
 * Returns true when the SaaS billing surface is enabled.
 * Self-hosted deployments set NEXT_PUBLIC_BILLING_ENABLED=false (or omit it).
 */
export function isBillingEnabled(): boolean {
  return process.env.NEXT_PUBLIC_BILLING_ENABLED === 'true';
}

export const EMPTY_BILLING_CELL_PLACEHOLDER = '—';

const BILLING_DATE_LOCALE = 'en-CA';

type WarnFn = (message?: unknown, ...optionalParams: unknown[]) => void;

const defaultWarn: WarnFn = (message, ...rest) => {
  logger.warn(String(message ?? ''), { component: 'billing', detail: rest.length ? rest.map(String).join(' ') : undefined });
};

export type OrganizationPlanMap = Record<number, { tier: PlanTier; status: string }>;

export type DashboardBillingStats = {
  active_subscriptions: number;
  past_due_count: number;
  plan_distribution: Record<string, number>;
};

export type OrganizationBillingSnapshot = {
  subscription: Subscription | null;
  usageSummary: OrgUsageSummary | null;
  invoices: Invoice[];
};

export function formatBillingDate(value?: string | null): string {
  return formatDate(value, {
    fallback: EMPTY_BILLING_CELL_PLACEHOLDER,
    locale: BILLING_DATE_LOCALE,
  });
}

export function normalizeInvoices(
  invoices: unknown,
  warn: WarnFn = defaultWarn
): Invoice[] {
  if (Array.isArray(invoices)) {
    return invoices as Invoice[];
  }
  if (invoices !== undefined && invoices !== null) {
    warn('Unexpected organization invoice payload:', typeof invoices);
  }
  return [];
}

export function buildOrganizationPlanMap(
  subscriptions: unknown
): OrganizationPlanMap {
  const planMap: OrganizationPlanMap = {};
  if (!Array.isArray(subscriptions)) {
    return planMap;
  }

  subscriptions.forEach((subscription) => {
    const typedSubscription = subscription as Subscription;
    planMap[typedSubscription.org_id] = {
      tier: (typedSubscription.plan?.tier ?? 'free') as PlanTier,
      status: typedSubscription.status,
    };
  });

  return planMap;
}

export async function fetchOrganizationPlanMap(
  loadSubscriptions: () => Promise<unknown>,
  warn: WarnFn = defaultWarn
): Promise<OrganizationPlanMap> {
  try {
    return buildOrganizationPlanMap(await loadSubscriptions());
  } catch (error) {
    warn('Failed to fetch subscription plans:', error);
    return {};
  }
}

export function buildDashboardBillingStats(
  subscriptions: unknown
): DashboardBillingStats | null {
  if (!Array.isArray(subscriptions)) {
    return null;
  }

  const activeSubscriptions = subscriptions.filter(
    (subscription) =>
      (subscription as Subscription).status === 'active'
  ).length;
  const pastDueCount = subscriptions.filter(
    (subscription) =>
      (subscription as Subscription).status === 'past_due'
  ).length;
  const planDistribution: Record<string, number> = {};

  subscriptions.forEach((subscription) => {
    const typedSubscription = subscription as Subscription;
    const tier = typedSubscription.plan?.tier ?? 'free';
    planDistribution[tier] = (planDistribution[tier] ?? 0) + 1;
  });

  return {
    active_subscriptions: activeSubscriptions,
    past_due_count: pastDueCount,
    plan_distribution: planDistribution,
  };
}

export async function fetchDashboardBillingStats(
  loadSubscriptions: () => Promise<unknown>,
  warn: WarnFn = defaultWarn
): Promise<DashboardBillingStats | null> {
  try {
    return buildDashboardBillingStats(await loadSubscriptions());
  } catch (error) {
    warn('Failed to fetch billing stats:', error);
    return null;
  }
}

export function resolveOrganizationBillingSnapshot(
  subscriptionResult: PromiseSettledResult<unknown>,
  usageResult: PromiseSettledResult<unknown>,
  invoiceResult: PromiseSettledResult<unknown>,
  warn: WarnFn = defaultWarn
): OrganizationBillingSnapshot {
  const subscription =
    subscriptionResult.status === 'fulfilled'
      ? (subscriptionResult.value as Subscription)
      : null;
  const usageSummary =
    usageResult.status === 'fulfilled'
      ? (usageResult.value as OrgUsageSummary)
      : null;
  const invoices =
    invoiceResult.status === 'fulfilled'
      ? normalizeInvoices(invoiceResult.value, warn)
      : [];

  return {
    subscription,
    usageSummary,
    invoices,
  };
}
