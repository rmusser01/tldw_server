'use client';

import { useCallback, useEffect, useMemo, useRef, useState, Suspense } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Pagination } from '@/components/ui/pagination';
import { TableSkeleton } from '@/components/ui/skeleton';
import { EmptyState } from '@/components/ui/empty-state';
import { ExportMenu } from '@/components/ui/export-menu';
import { RefreshCw, Wallet, Pencil } from 'lucide-react';
import { exportBudgets, type ExportFormat } from '@/lib/export';
import { api } from '@/lib/api-client';
import { formatDateTime } from '@/lib/format';
import { useOrgContext } from '@/components/OrgContextSwitcher';
import { useUrlMultiState } from '@/lib/use-url-state';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { useToast } from '@/components/ui/toast';
import Link from 'next/link';

type BudgetAlertThresholds = {
  global?: number[];
  per_metric?: Record<string, number[]>;
};

type BudgetEnforcementMode = {
  global?: 'none' | 'soft' | 'hard';
  per_metric?: Record<string, 'none' | 'soft' | 'hard'>;
};

type ProviderBudget = {
  month_usd?: number | null;
  day_usd?: number | null;
};

type BudgetSettings = {
  budget_day_usd?: number | null;
  budget_month_usd?: number | null;
  budget_day_tokens?: number | null;
  budget_month_tokens?: number | null;
  alert_thresholds?: BudgetAlertThresholds | null;
  enforcement_mode?: BudgetEnforcementMode | null;
  provider_budgets?: Record<string, ProviderBudget | null> | null;
};

type BudgetMetricKey =
  | 'budget_day_usd'
  | 'budget_month_usd'
  | 'budget_day_tokens'
  | 'budget_month_tokens';

type EnforcementMode = 'none' | 'soft' | 'hard';

type BudgetThresholdDraft = {
  warning: string;
  critical: string;
};

type BudgetEditFormState = {
  budget_day_usd: string;
  budget_month_usd: string;
  budget_day_tokens: string;
  budget_month_tokens: string;
  thresholds: Record<BudgetMetricKey, BudgetThresholdDraft>;
  enforcement: Record<BudgetMetricKey, EnforcementMode>;
};

type BudgetEditValidationErrors = Partial<
  Record<BudgetMetricKey, string>
>;

type BudgetEditErrorsState = {
  caps: BudgetEditValidationErrors;
  thresholds: BudgetEditValidationErrors;
};

type ProviderBudgetDraft = {
  month_usd: string;
  day_usd: string;
};

type NotificationChannelSummary = {
  configuredChannels: string[];
  minSeverity: 'info' | 'warning' | 'error' | 'critical';
};

type BudgetSpend = {
  spend_day_usd?: number | null;
  spend_month_usd?: number | null;
  spend_day_tokens?: number | null;
  spend_month_tokens?: number | null;
};

type OrgBudgetItem = {
  org_id: number;
  org_name: string;
  org_slug?: string | null;
  plan_name: string;
  plan_display_name: string;
  budgets: BudgetSettings;
  spend?: BudgetSpend;
  updated_at?: string | null;
  period_start?: string | null;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const isNumberArray = (value: unknown): value is number[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'number');

const parseAlertThresholds = (value: unknown): BudgetAlertThresholds | null | undefined => {
  if (value === null) return null;
  if (!isRecord(value)) return undefined;

  const global = isNumberArray(value.global) ? value.global : undefined;
  let perMetric: Record<string, number[]> | undefined;
  if (isRecord(value.per_metric)) {
    const entries = Object.entries(value.per_metric).reduce<Record<string, number[]>>((acc, [key, entry]) => {
      if (isNumberArray(entry)) {
        acc[key] = entry;
      }
      return acc;
    }, {});
    if (Object.keys(entries).length > 0) {
      perMetric = entries;
    }
  }

  if (!global && !perMetric) return undefined;
  return {
    ...(global ? { global } : {}),
    ...(perMetric ? { per_metric: perMetric } : {}),
  };
};

const isEnforcementModeValue = (value: unknown): value is 'none' | 'soft' | 'hard' =>
  value === 'none' || value === 'soft' || value === 'hard';

const parseEnforcementMode = (value: unknown): BudgetEnforcementMode | null | undefined => {
  if (value === null) return null;
  if (!isRecord(value)) return undefined;

  const global = isEnforcementModeValue(value.global) ? value.global : undefined;
  let perMetric: Record<string, 'none' | 'soft' | 'hard'> | undefined;
  if (isRecord(value.per_metric)) {
    const entries = Object.entries(value.per_metric).reduce<Record<string, 'none' | 'soft' | 'hard'>>(
      (acc, [key, entry]) => {
        if (isEnforcementModeValue(entry)) {
          acc[key] = entry;
        }
        return acc;
      },
      {},
    );
    if (Object.keys(entries).length > 0) {
      perMetric = entries;
    }
  }

  if (!global && !perMetric) return undefined;
  return {
    ...(global ? { global } : {}),
    ...(perMetric ? { per_metric: perMetric } : {}),
  };
};

const parseBudgetSettings = (value: unknown): BudgetSettings => {
  if (!isRecord(value)) return {};

  return {
    budget_day_usd: typeof value.budget_day_usd === 'number' || value.budget_day_usd === null
      ? value.budget_day_usd
      : undefined,
    budget_month_usd: typeof value.budget_month_usd === 'number' || value.budget_month_usd === null
      ? value.budget_month_usd
      : undefined,
    budget_day_tokens: typeof value.budget_day_tokens === 'number' || value.budget_day_tokens === null
      ? value.budget_day_tokens
      : undefined,
    budget_month_tokens: typeof value.budget_month_tokens === 'number' || value.budget_month_tokens === null
      ? value.budget_month_tokens
      : undefined,
    alert_thresholds: parseAlertThresholds(value.alert_thresholds),
    enforcement_mode: parseEnforcementMode(value.enforcement_mode),
  };
};

const parseOrgBudgetItems = (value: unknown): OrgBudgetItem[] => {
  if (!Array.isArray(value)) return [];

  return value.reduce<OrgBudgetItem[]>((acc, item) => {
    if (!isRecord(item)) return acc;

    const orgId = item.org_id;
    const orgName = item.org_name;
    const planName = item.plan_name;
    const planDisplayName = item.plan_display_name;

    if (
      typeof orgId !== 'number'
      || typeof orgName !== 'string'
      || typeof planName !== 'string'
      || typeof planDisplayName !== 'string'
    ) {
      return acc;
    }

    const orgSlug = typeof item.org_slug === 'string' || item.org_slug === null ? item.org_slug : undefined;
    const updatedAt = typeof item.updated_at === 'string' || item.updated_at === null ? item.updated_at : undefined;
    const periodStart = typeof item.period_start === 'string' || item.period_start === null ? item.period_start : undefined;
    const budgets = parseBudgetSettings(item.budgets);

    // Spend data may be nested under item.spend or flattened into the item itself
    const spendRaw = isRecord(item.spend) ? item.spend : (isRecord(item) ? item : null);
    const spend: BudgetSpend = {
      spend_day_usd: typeof spendRaw?.spend_day_usd === 'number' ? spendRaw.spend_day_usd : undefined,
      spend_month_usd: typeof spendRaw?.spend_month_usd === 'number' ? spendRaw.spend_month_usd : undefined,
      spend_day_tokens: typeof spendRaw?.spend_day_tokens === 'number' ? spendRaw.spend_day_tokens : undefined,
      spend_month_tokens: typeof spendRaw?.spend_month_tokens === 'number' ? spendRaw.spend_month_tokens : undefined,
    };

    acc.push({
      org_id: orgId,
      org_name: orgName,
      org_slug: orgSlug,
      plan_name: planName,
      plan_display_name: planDisplayName,
      budgets,
      spend,
      updated_at: updatedAt,
      period_start: periodStart,
    });

    return acc;
  }, []);
};

const BUDGET_METRICS: Array<{ key: BudgetMetricKey; label: string }> = [
  { key: 'budget_day_usd', label: 'Daily USD' },
  { key: 'budget_month_usd', label: 'Monthly USD' },
  { key: 'budget_day_tokens', label: 'Daily tokens' },
  { key: 'budget_month_tokens', label: 'Monthly tokens' },
];

const ENFORCEMENT_OPTIONS: Array<{ value: EnforcementMode; label: string }> = [
  { value: 'none', label: 'None' },
  { value: 'soft', label: 'Soft (log + alert)' },
  { value: 'hard', label: 'Hard (block + alert)' },
];

const DEFAULT_WARNING_THRESHOLD = 80;
const DEFAULT_CRITICAL_THRESHOLD = 95;

const getMetricThresholdPair = (
  thresholds: BudgetAlertThresholds | null | undefined,
  metricKey: BudgetMetricKey
): [number, number] => {
  const perMetric = thresholds?.per_metric?.[metricKey];
  const global = thresholds?.global;
  const source = Array.isArray(perMetric) && perMetric.length > 0
    ? perMetric
    : (Array.isArray(global) && global.length > 0 ? global : [DEFAULT_WARNING_THRESHOLD, DEFAULT_CRITICAL_THRESHOLD]);
  const warning = Number.isFinite(source[0]) ? Number(source[0]) : DEFAULT_WARNING_THRESHOLD;
  const critical = Number.isFinite(source[1]) ? Number(source[1]) : Math.max(warning + 1, DEFAULT_CRITICAL_THRESHOLD);
  return [warning, critical];
};

const getMetricEnforcementMode = (
  mode: BudgetEnforcementMode | null | undefined,
  metricKey: BudgetMetricKey
): EnforcementMode => {
  const perMetricValue = mode?.per_metric?.[metricKey];
  if (perMetricValue === 'none' || perMetricValue === 'soft' || perMetricValue === 'hard') {
    return perMetricValue;
  }
  if (mode?.global === 'none' || mode?.global === 'soft' || mode?.global === 'hard') {
    return mode.global;
  }
  return 'none';
};

const buildBudgetEditFormState = (settings: BudgetSettings): BudgetEditFormState => {
  const thresholds = BUDGET_METRICS.reduce<Record<BudgetMetricKey, BudgetThresholdDraft>>((acc, metric) => {
    const [warning, critical] = getMetricThresholdPair(settings.alert_thresholds, metric.key);
    acc[metric.key] = {
      warning: String(warning),
      critical: String(critical),
    };
    return acc;
  }, {
    budget_day_usd: { warning: String(DEFAULT_WARNING_THRESHOLD), critical: String(DEFAULT_CRITICAL_THRESHOLD) },
    budget_month_usd: { warning: String(DEFAULT_WARNING_THRESHOLD), critical: String(DEFAULT_CRITICAL_THRESHOLD) },
    budget_day_tokens: { warning: String(DEFAULT_WARNING_THRESHOLD), critical: String(DEFAULT_CRITICAL_THRESHOLD) },
    budget_month_tokens: { warning: String(DEFAULT_WARNING_THRESHOLD), critical: String(DEFAULT_CRITICAL_THRESHOLD) },
  });

  const enforcement = BUDGET_METRICS.reduce<Record<BudgetMetricKey, EnforcementMode>>((acc, metric) => {
    acc[metric.key] = getMetricEnforcementMode(settings.enforcement_mode, metric.key);
    return acc;
  }, {
    budget_day_usd: 'none',
    budget_month_usd: 'none',
    budget_day_tokens: 'none',
    budget_month_tokens: 'none',
  });

  return {
    budget_day_usd: settings.budget_day_usd === null || settings.budget_day_usd === undefined
      ? ''
      : String(settings.budget_day_usd),
    budget_month_usd: settings.budget_month_usd === null || settings.budget_month_usd === undefined
      ? ''
      : String(settings.budget_month_usd),
    budget_day_tokens: settings.budget_day_tokens === null || settings.budget_day_tokens === undefined
      ? ''
      : String(settings.budget_day_tokens),
    budget_month_tokens: settings.budget_month_tokens === null || settings.budget_month_tokens === undefined
      ? ''
      : String(settings.budget_month_tokens),
    thresholds,
    enforcement,
  };
};

const parsePositiveNumber = (value: string): number | null => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return parsed;
};

const parsePositiveInteger = (value: string): number | null => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  if (!Number.isInteger(parsed) || parsed <= 0) return null;
  return parsed;
};

const validateBudgetEditForm = (form: BudgetEditFormState): {
  valid: boolean;
  errors: BudgetEditErrorsState;
} => {
  const errors: BudgetEditErrorsState = {
    caps: {},
    thresholds: {},
  };

  if (form.budget_day_usd.trim() && parsePositiveNumber(form.budget_day_usd) === null) {
    errors.caps.budget_day_usd = 'Daily USD cap must be a positive number.';
  }
  if (form.budget_month_usd.trim() && parsePositiveNumber(form.budget_month_usd) === null) {
    errors.caps.budget_month_usd = 'Monthly USD cap must be a positive number.';
  }
  if (form.budget_day_tokens.trim() && parsePositiveInteger(form.budget_day_tokens) === null) {
    errors.caps.budget_day_tokens = 'Daily token cap must be a positive integer.';
  }
  if (form.budget_month_tokens.trim() && parsePositiveInteger(form.budget_month_tokens) === null) {
    errors.caps.budget_month_tokens = 'Monthly token cap must be a positive integer.';
  }

  BUDGET_METRICS.forEach((metric) => {
    const warning = Number(form.thresholds[metric.key]?.warning);
    const critical = Number(form.thresholds[metric.key]?.critical);
    if (!Number.isFinite(warning) || warning < 1 || warning > 100) {
      errors.thresholds[metric.key] = `${metric.label} warning threshold must be between 1 and 100.`;
      return;
    }
    if (!Number.isFinite(critical) || critical < 1 || critical > 100) {
      errors.thresholds[metric.key] = `${metric.label} critical threshold must be between 1 and 100.`;
      return;
    }
    if (warning >= critical) {
      errors.thresholds[metric.key] = `${metric.label} warning threshold must be lower than critical.`;
    }
  });

  return {
    valid: Object.keys(errors.caps).length === 0 && Object.keys(errors.thresholds).length === 0,
    errors,
  };
};

const buildBudgetUpdatePayload = (form: BudgetEditFormState): BudgetSettings => ({
  budget_day_usd: form.budget_day_usd.trim() ? Number(form.budget_day_usd) : null,
  budget_month_usd: form.budget_month_usd.trim() ? Number(form.budget_month_usd) : null,
  budget_day_tokens: form.budget_day_tokens.trim() ? Number(form.budget_day_tokens) : null,
  budget_month_tokens: form.budget_month_tokens.trim() ? Number(form.budget_month_tokens) : null,
  alert_thresholds: {
    per_metric: BUDGET_METRICS.reduce<Record<string, number[]>>((acc, metric) => {
      const warning = Math.round(Number(form.thresholds[metric.key].warning));
      const critical = Math.round(Number(form.thresholds[metric.key].critical));
      acc[metric.key] = [warning, critical];
      return acc;
    }, {}),
  },
  enforcement_mode: {
    per_metric: BUDGET_METRICS.reduce<Record<string, EnforcementMode>>((acc, metric) => {
      acc[metric.key] = form.enforcement[metric.key];
      return acc;
    }, {}),
  },
});

const summarizeHardModeChanges = (
  previous: BudgetSettings,
  next: BudgetEditFormState
): string[] => {
  const changed: string[] = [];
  BUDGET_METRICS.forEach((metric) => {
    const previousMode = getMetricEnforcementMode(previous.enforcement_mode, metric.key);
    const nextMode = next.enforcement[metric.key];
    if (previousMode !== 'hard' && nextMode === 'hard') {
      changed.push(metric.label);
    }
  });
  return changed;
};

const normalizeNotificationSummary = (value: unknown): NotificationChannelSummary | null => {
  if (!isRecord(value)) return null;
  const configured = new Set<string>();
  if (Array.isArray(value.channels)) {
    value.channels.forEach((channel) => {
      if (!isRecord(channel)) return;
      const enabled = channel.enabled === true;
      const type = typeof channel.type === 'string' ? channel.type : null;
      if (enabled && type) {
        configured.add(type);
      }
    });
  } else {
    if (value.webhook_url) configured.add('webhook');
    if (value.email_to) configured.add('email');
    if (value.smtp_host) configured.add('email');
  }
  const minSeverityRaw = typeof value.alert_threshold === 'string'
    ? value.alert_threshold
    : (typeof value.min_severity === 'string' ? value.min_severity : 'warning');
  const normalizedSeverity = minSeverityRaw.toLowerCase();
  const minSeverity = ['info', 'warning', 'error', 'critical'].includes(normalizedSeverity)
    ? (normalizedSeverity as NotificationChannelSummary['minSeverity'])
    : 'warning';
  return {
    configuredChannels: [...configured],
    minSeverity,
  };
};

const formatCurrency = (value?: number | null) => {
  if (value === null || value === undefined) return '—';
  return `$${value.toFixed(2)}`;
};

const formatTokens = (value?: number | null) => {
  if (value === null || value === undefined) return '—';
  return value.toLocaleString();
};

const formatBudgetDate = (value?: string | null) => formatDateTime(value, {
  fallback: '—',
  locale: 'en-US',
  options: {
    dateStyle: 'medium',
    timeStyle: 'short',
  },
});

const formatPercentList = (values: number[]) =>
  values.map((value) => `${value}%`).join(', ');

const formatThresholds = (thresholds?: BudgetAlertThresholds | null) => {
  if (!thresholds) return '—';
  const parts: string[] = [];
  if (thresholds.global && thresholds.global.length > 0) {
    parts.push(`global: ${formatPercentList(thresholds.global)}`);
  }
  const perMetricKeys = thresholds.per_metric ? Object.keys(thresholds.per_metric) : [];
  if (perMetricKeys.length > 0) {
    parts.push(`per metric: ${perMetricKeys.join(', ')}`);
  }
  return parts.length > 0 ? parts.join(' | ') : '—';
};

const formatEnforcement = (mode?: BudgetEnforcementMode | null) => {
  if (!mode) return '—';
  const parts: string[] = [];
  if (mode.global) {
    parts.push(`global: ${mode.global}`);
  }
  const perMetricKeys = mode.per_metric ? Object.keys(mode.per_metric) : [];
  if (perMetricKeys.length > 0) {
    parts.push(`per metric: ${perMetricKeys.length}`);
  }
  return parts.length > 0 ? parts.join(' | ') : '—';
};

const computeExhaustionLabel = (
  currentSpend: number,
  cap: number,
  periodDays: number,
  periodStart?: string | null
): string => {
  if (cap <= 0 || currentSpend <= 0) return 'On track';
  if (currentSpend >= cap) return 'Exhausted';

  const startDate = periodStart ? new Date(periodStart) : new Date();
  const now = new Date();
  const elapsedMs = Math.max(now.getTime() - startDate.getTime(), 1);
  const elapsedDays = elapsedMs / (1000 * 60 * 60 * 24);
  if (elapsedDays < 0.01) return 'On track';

  const burnRate = currentSpend / elapsedDays;
  if (burnRate <= 0) return 'On track';

  const remaining = cap - currentSpend;
  const daysRemaining = remaining / burnRate;
  if (daysRemaining > periodDays * 2) return 'On track';

  const exhaustionDate = new Date(now.getTime() + daysRemaining * 24 * 60 * 60 * 1000);
  const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `Projected: ${monthNames[exhaustionDate.getMonth()]} ${exhaustionDate.getDate()}`;
};

const SpendProgressBar = ({
  label,
  cap,
  spend,
  formatValue,
  periodDays,
  periodStart,
}: {
  label: string;
  cap: number | null | undefined;
  spend: number | null | undefined;
  formatValue: (v?: number | null) => string;
  periodDays: number;
  periodStart?: string | null;
}) => {
  if (cap === null || cap === undefined) {
    return (
      <div className="flex items-center justify-between gap-3">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-muted-foreground">--</span>
      </div>
    );
  }

  const currentSpend = typeof spend === 'number' ? spend : 0;
  const hasSpendData = typeof spend === 'number';
  const pct = cap > 0 ? Math.min((currentSpend / cap) * 100, 100) : 0;

  const barColor =
    pct >= 95 ? 'bg-red-500' :
    pct >= 80 ? 'bg-yellow-500' :
    'bg-green-500';

  const exhaustionLabel = hasSpendData
    ? computeExhaustionLabel(currentSpend, cap, periodDays, periodStart)
    : null;

  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between gap-3">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">
          {hasSpendData ? `${formatValue(currentSpend)} / ` : ''}{formatValue(cap)}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <div
          className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted"
          role="progressbar"
          aria-label={`${label} usage`}
          aria-valuenow={Math.round(pct)}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className={`h-full rounded-full transition-all ${barColor}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className={`text-[10px] font-medium ${
          pct >= 95 ? 'text-red-600' :
          pct >= 80 ? 'text-yellow-600' :
          'text-green-600'
        }`}>
          {Math.round(pct)}%
        </span>
      </div>
      {exhaustionLabel && (
        <div className={`text-[10px] ${
          exhaustionLabel === 'Exhausted' ? 'text-red-600 font-medium' :
          exhaustionLabel === 'On track' ? 'text-muted-foreground' :
          'text-yellow-600'
        }`}>
          {exhaustionLabel}
        </div>
      )}
    </div>
  );
};

const renderBudgetCaps = (item: OrgBudgetItem) => {
  const settings = item.budgets || {};
  const spend = item.spend || {};
  const hasAny = [
    settings.budget_day_usd,
    settings.budget_month_usd,
    settings.budget_day_tokens,
    settings.budget_month_tokens,
  ].some((value) => value !== null && value !== undefined);
  if (!hasAny) {
    return <span className="text-muted-foreground text-sm">No caps set</span>;
  }
  return (
    <div className="space-y-2 text-xs min-w-[180px]">
      <SpendProgressBar
        label="Daily USD"
        cap={settings.budget_day_usd}
        spend={spend.spend_day_usd}
        formatValue={formatCurrency}
        periodDays={1}
        periodStart={item.period_start}
      />
      <SpendProgressBar
        label="Monthly USD"
        cap={settings.budget_month_usd}
        spend={spend.spend_month_usd}
        formatValue={formatCurrency}
        periodDays={30}
        periodStart={item.period_start}
      />
      <SpendProgressBar
        label="Daily tokens"
        cap={settings.budget_day_tokens}
        spend={spend.spend_day_tokens}
        formatValue={formatTokens}
        periodDays={1}
        periodStart={item.period_start}
      />
      <SpendProgressBar
        label="Monthly tokens"
        cap={settings.budget_month_tokens}
        spend={spend.spend_month_tokens}
        formatValue={formatTokens}
        periodDays={30}
        periodStart={item.period_start}
      />
      {settings.provider_budgets && Object.keys(settings.provider_budgets).length > 0 && (
        <>
          <div className="border-t my-1 pt-1">
            <span className="text-muted-foreground text-[10px] uppercase tracking-wide">Per-Provider</span>
          </div>
          {Object.entries(settings.provider_budgets).map(([provider, budget]) => (
            <div key={provider} className="flex items-center justify-between gap-3">
              <span className="text-muted-foreground">{provider}</span>
              <span className="font-mono">
                {budget?.month_usd != null ? `$${budget.month_usd}/mo` : ''}
                {budget?.month_usd != null && budget?.day_usd != null ? ' · ' : ''}
                {budget?.day_usd != null ? `$${budget.day_usd}/day` : ''}
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  );
};

function BudgetsPageContent() {
  const promptPrivilegedAction = usePrivilegedActionDialog();
  const { success, error: showError } = useToast();
  const { selectedOrg } = useOrgContext();
  const defaultPage = 1;
  const defaultPageSize = 20;
  const [{ page: rawPage, pageSize: rawPageSize }, setPaginationValues] = useUrlMultiState({
    page: defaultPage,
    pageSize: defaultPageSize,
  });
  const setPaginationValuesRef = useRef(setPaginationValues);
  useEffect(() => {
    setPaginationValuesRef.current = setPaginationValues;
  }, [setPaginationValues]);
  const page = Number.isFinite(rawPage) && rawPage > 0 ? rawPage : defaultPage;
  const pageSize = Number.isFinite(rawPageSize) && rawPageSize > 0 ? rawPageSize : defaultPageSize;
  const setPage = useCallback((nextPage: number) => {
    setPaginationValues({ page: nextPage });
  }, [setPaginationValues]);
  const [budgets, setBudgets] = useState<OrgBudgetItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notificationSummary, setNotificationSummary] = useState<NotificationChannelSummary | null>(null);
  const [editingBudget, setEditingBudget] = useState<OrgBudgetItem | null>(null);
  const [editForm, setEditForm] = useState<BudgetEditFormState | null>(null);
  const [editErrors, setEditErrors] = useState<BudgetEditErrorsState>({ caps: {}, thresholds: {} });
  const [savingBudget, setSavingBudget] = useState(false);
  const [providerBudgets, setProviderBudgets] = useState<Record<string, ProviderBudgetDraft>>({});
  const [newProviderName, setNewProviderName] = useState('');
  const [showAddProviderInput, setShowAddProviderInput] = useState(false);

  const budgetParams = useMemo(() => {
    const params: Record<string, string> = {
      page: String(page),
      limit: String(pageSize),
    };
    if (selectedOrg) {
      params.org_id = String(selectedOrg.id);
    }
    return params;
  }, [page, pageSize, selectedOrg]);

  const loadBudgets = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      const [budgetsResult, notificationsResult] = await Promise.allSettled([
        api.getBudgets(budgetParams),
        api.getNotificationSettings(),
      ]);

      if (budgetsResult.status === 'fulfilled') {
        const data = budgetsResult.value;
        if (isRecord(data)) {
          const items = parseOrgBudgetItems(data.items);
          setBudgets(items);
          const totalValue = typeof data.total === 'number' ? data.total : items.length;
          setTotal(totalValue);
        } else {
          setBudgets([]);
          setTotal(0);
        }
      } else {
        const message = budgetsResult.reason instanceof Error && budgetsResult.reason.message
          ? budgetsResult.reason.message
          : 'Failed to load budgets';
        setError(message);
        setBudgets([]);
        setTotal(0);
      }

      if (notificationsResult.status === 'fulfilled') {
        setNotificationSummary(normalizeNotificationSummary(notificationsResult.value));
      } else {
        setNotificationSummary(null);
      }
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to load budgets';
      setError(message);
      setBudgets([]);
      setTotal(0);
      setNotificationSummary(null);
    } finally {
      setLoading(false);
    }
  }, [budgetParams]);

  useEffect(() => {
    void loadBudgets();
  }, [loadBudgets]);

  useEffect(() => {
    setPaginationValuesRef.current({ page: defaultPage });
  }, [selectedOrg, defaultPage]);

  const totalItems = total || budgets.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));

  const openEditDialog = (item: OrgBudgetItem) => {
    setEditingBudget(item);
    setEditForm(buildBudgetEditFormState(item.budgets || {}));
    setEditErrors({ caps: {}, thresholds: {} });
    const existing = (item.budgets || {}).provider_budgets || {};
    const draft: Record<string, ProviderBudgetDraft> = {};
    for (const [provider, budget] of Object.entries(existing)) {
      draft[provider] = {
        month_usd: budget?.month_usd != null ? String(budget.month_usd) : '',
        day_usd: budget?.day_usd != null ? String(budget.day_usd) : '',
      };
    }
    setProviderBudgets(draft);
    setNewProviderName('');
    setShowAddProviderInput(false);
  };

  const closeEditDialog = () => {
    setEditingBudget(null);
    setEditForm(null);
    setEditErrors({ caps: {}, thresholds: {} });
    setSavingBudget(false);
    setProviderBudgets({});
    setNewProviderName('');
    setShowAddProviderInput(false);
  };

  const updateEditCap = (key: keyof Pick<BudgetEditFormState, 'budget_day_usd' | 'budget_month_usd' | 'budget_day_tokens' | 'budget_month_tokens'>, value: string) => {
    setEditForm((previous) => {
      if (!previous) return previous;
      return {
        ...previous,
        [key]: value,
      };
    });
    setEditErrors((previous) => ({
      ...previous,
      caps: {
        ...previous.caps,
        [key]: undefined,
      },
    }));
  };

  const updateEditThreshold = (metric: BudgetMetricKey, field: keyof BudgetThresholdDraft, value: string) => {
    setEditForm((previous) => {
      if (!previous) return previous;
      return {
        ...previous,
        thresholds: {
          ...previous.thresholds,
          [metric]: {
            ...previous.thresholds[metric],
            [field]: value,
          },
        },
      };
    });
    setEditErrors((previous) => ({
      ...previous,
      thresholds: {
        ...previous.thresholds,
        [metric]: undefined,
      },
    }));
  };

  const updateEditEnforcement = (metric: BudgetMetricKey, value: EnforcementMode) => {
    setEditForm((previous) => {
      if (!previous) return previous;
      return {
        ...previous,
        enforcement: {
          ...previous.enforcement,
          [metric]: value,
        },
      };
    });
  };

  const saveBudgetEdits = async () => {
    if (!editingBudget || !editForm) return;
    const validation = validateBudgetEditForm(editForm);
    if (!validation.valid) {
      setEditErrors(validation.errors);
      return;
    }

    const hardModeChanges = summarizeHardModeChanges(editingBudget.budgets || {}, editForm);
    if (hardModeChanges.length > 0) {
      const result = await promptPrivilegedAction({
        title: 'Enable hard enforcement?',
        message: `Hard enforcement can block requests when budgets are reached for: ${hardModeChanges.join(', ')}.`,
        confirmText: 'Enable hard enforcement',
        requirePassword: false,
      });
      if (!result) return;
    }

    try {
      setSavingBudget(true);
      const budgetsPayload = buildBudgetUpdatePayload(editForm);
      // Include provider budgets in the payload
      const providerBudgetsPayload: Record<string, ProviderBudget | null> = {};
      for (const [provider, draft] of Object.entries(providerBudgets)) {
        const monthValue = draft.month_usd.trim();
        const dayValue = draft.day_usd.trim();
        const monthRaw = monthValue ? Number(monthValue) : null;
        const dayRaw = dayValue ? Number(dayValue) : null;
        const monthUsd = monthRaw != null && Number.isFinite(monthRaw) ? monthRaw : null;
        const dayUsd = dayRaw != null && Number.isFinite(dayRaw) ? dayRaw : null;
        if (monthUsd === null && dayUsd === null) {
          providerBudgetsPayload[provider] = null; // Remove provider
        } else {
          providerBudgetsPayload[provider] = { month_usd: monthUsd, day_usd: dayUsd };
        }
      }
      if (Object.keys(providerBudgetsPayload).length > 0) {
        budgetsPayload.provider_budgets = providerBudgetsPayload;
      }
      await api.updateBudget(String(editingBudget.org_id), {
        budgets: budgetsPayload,
      });
      success('Budget updated', `Budget settings saved for ${editingBudget.org_name}.`);
      closeEditDialog();
      await loadBudgets();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message
        ? err.message
        : 'Failed to update budget settings.';
      showError('Budget update failed', message);
      setError(message);
    } finally {
      setSavingBudget(false);
    }
  };

  const addProviderBudgetDraft = () => {
    const providerName = newProviderName.trim().toLowerCase();
    if (!providerName) return;
    setProviderBudgets((prev) => ({
      ...prev,
      [providerName]: prev[providerName] ?? { month_usd: '', day_usd: '' },
    }));
    setNewProviderName('');
    setShowAddProviderInput(false);
  };

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h1 className="text-3xl font-bold">Budgets</h1>
              <p className="text-muted-foreground">
                Review per-organization caps, alert thresholds, and enforcement settings.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <ExportMenu
                onExport={(format: ExportFormat) => exportBudgets(budgets, format)}
                disabled={budgets.length === 0}
              />
              <Button variant="outline" onClick={loadBudgets} disabled={loading}>
                <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </div>

          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wallet className="h-5 w-5" />
                Budget overview
              </CardTitle>
              <CardDescription>
                Update caps, thresholds, and enforcement for each organization.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 pt-0">
              {notificationSummary && notificationSummary.configuredChannels.length > 0 ? (
                <Alert className="border-blue-200 bg-blue-50">
                  <AlertDescription className="text-blue-900">
                    Budget threshold alerts are wired to monitoring notification channels:{' '}
                    <span className="font-medium">{notificationSummary.configuredChannels.join(', ')}</span>
                    {' '} (minimum severity: {notificationSummary.minSeverity}).
                  </AlertDescription>
                </Alert>
              ) : (
                <Alert className="border-yellow-200 bg-yellow-50">
                  <AlertDescription className="text-yellow-900">
                    No monitoring notification channels are configured. Budget thresholds will save, but
                    no delivery channel is currently available. Configure channels in{' '}
                    <Link href="/monitoring" className="underline">
                      Monitoring
                    </Link>.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Organization budgets</CardTitle>
              <CardDescription>
                {totalItems} record{totalItems !== 1 ? 's' : ''} found
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="py-4" data-testid="table-skeleton">
                  <TableSkeleton rows={5} columns={7} />
                </div>
              ) : budgets.length === 0 ? (
                <EmptyState
                  title="No budgets found for the selected scope."
                  description="Try adjusting your filters or create budgets for additional organizations."
                />
              ) : (
                <>
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Organization</TableHead>
                          <TableHead>Plan</TableHead>
                          <TableHead>Budget caps & spend</TableHead>
                          <TableHead>Alert thresholds</TableHead>
                          <TableHead>Enforcement</TableHead>
                          <TableHead>Updated</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {budgets.map((item) => (
                          <TableRow key={item.org_id}>
                            <TableCell>
                              <div className="space-y-1">
                                <div className="font-medium">{item.org_name}</div>
                                {item.org_slug ? (
                                  <div className="text-xs text-muted-foreground">{item.org_slug}</div>
                                ) : null}
                                <div className="text-xs text-muted-foreground">ID {item.org_id}</div>
                              </div>
                            </TableCell>
                            <TableCell>
                              <div className="space-y-1">
                                <div className="font-medium">{item.plan_display_name}</div>
                                <div className="text-xs text-muted-foreground">{item.plan_name}</div>
                              </div>
                            </TableCell>
                            <TableCell>{renderBudgetCaps(item)}</TableCell>
                            <TableCell className="text-sm">{formatThresholds(item.budgets?.alert_thresholds)}</TableCell>
                            <TableCell className="text-sm">{formatEnforcement(item.budgets?.enforcement_mode)}</TableCell>
                            <TableCell className="text-sm">{formatBudgetDate(item.updated_at)}</TableCell>
                            <TableCell className="text-right">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                  openEditDialog(item);
                                }}
                                data-testid={`budget-edit-${item.org_id}`}
                              >
                                <Pencil className="mr-2 h-4 w-4" />
                                Edit
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>

                  <Pagination
                    currentPage={page}
                    totalPages={totalPages}
                    totalItems={totalItems}
                    pageSize={pageSize}
                    onPageChange={setPage}
                    onPageSizeChange={(size) => {
                      setPaginationValues({ pageSize: size, page: defaultPage });
                    }}
                  />
                </>
              )}
            </CardContent>
          </Card>

          <Dialog open={!!editingBudget && !!editForm} onOpenChange={(open) => {
            if (!open) closeEditDialog();
          }}>
            <DialogContent className="max-w-4xl">
              <DialogHeader>
                <DialogTitle>
                  Edit Budget: {editingBudget?.org_name || 'Organization'}
                </DialogTitle>
                <DialogDescription>
                  Update budget caps, threshold percentages, and enforcement behavior.
                </DialogDescription>
              </DialogHeader>

              {editForm && (
                <div className="space-y-6">
                  <div>
                    <h3 className="mb-3 text-sm font-medium">Budget Caps</h3>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-1">
                        <Label htmlFor="budget_day_usd">Daily USD cap</Label>
                        <Input
                          id="budget_day_usd"
                          type="number"
                          min={0}
                          step="0.01"
                          value={editForm.budget_day_usd}
                          onChange={(event) => {
                            updateEditCap('budget_day_usd', event.target.value);
                          }}
                        />
                        {editErrors.caps.budget_day_usd && (
                          <p className="text-xs text-destructive">{editErrors.caps.budget_day_usd}</p>
                        )}
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor="budget_month_usd">Monthly USD cap</Label>
                        <Input
                          id="budget_month_usd"
                          type="number"
                          min={0}
                          step="0.01"
                          value={editForm.budget_month_usd}
                          onChange={(event) => {
                            updateEditCap('budget_month_usd', event.target.value);
                          }}
                        />
                        {editErrors.caps.budget_month_usd && (
                          <p className="text-xs text-destructive">{editErrors.caps.budget_month_usd}</p>
                        )}
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor="budget_day_tokens">Daily token cap</Label>
                        <Input
                          id="budget_day_tokens"
                          type="number"
                          min={0}
                          step="1"
                          value={editForm.budget_day_tokens}
                          onChange={(event) => {
                            updateEditCap('budget_day_tokens', event.target.value);
                          }}
                        />
                        {editErrors.caps.budget_day_tokens && (
                          <p className="text-xs text-destructive">{editErrors.caps.budget_day_tokens}</p>
                        )}
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor="budget_month_tokens">Monthly token cap</Label>
                        <Input
                          id="budget_month_tokens"
                          type="number"
                          min={0}
                          step="1"
                          value={editForm.budget_month_tokens}
                          onChange={(event) => {
                            updateEditCap('budget_month_tokens', event.target.value);
                          }}
                        />
                        {editErrors.caps.budget_month_tokens && (
                          <p className="text-xs text-destructive">{editErrors.caps.budget_month_tokens}</p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Provider Budgets */}
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-medium">Per-Provider Budgets</h3>
                      {showAddProviderInput ? (
                        <div className="flex items-center gap-2">
                          <Input
                            autoFocus
                            value={newProviderName}
                            onChange={(e) => setNewProviderName(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') {
                                e.preventDefault();
                                addProviderBudgetDraft();
                              }
                              if (e.key === 'Escape') {
                                e.preventDefault();
                                setNewProviderName('');
                                setShowAddProviderInput(false);
                              }
                            }}
                            placeholder="openai"
                            className="w-40"
                          />
                          <Button
                            variant="outline"
                            size="sm"
                            type="button"
                            onClick={addProviderBudgetDraft}
                            disabled={!newProviderName.trim()}
                          >
                            Add
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            type="button"
                            onClick={() => {
                              setNewProviderName('');
                              setShowAddProviderInput(false);
                            }}
                          >
                            Cancel
                          </Button>
                        </div>
                      ) : (
                        <Button
                          variant="outline"
                          size="sm"
                          type="button"
                          onClick={() => setShowAddProviderInput(true)}
                        >
                          + Add Provider
                        </Button>
                      )}
                    </div>
                    {Object.keys(providerBudgets).length === 0 ? (
                      <p className="text-xs text-muted-foreground">No per-provider budgets configured.</p>
                    ) : (
                      <div className="space-y-3">
                        {Object.entries(providerBudgets).map(([provider, draft]) => (
                          <div key={provider} className="flex items-end gap-3 rounded-md border p-3">
                            <div className="min-w-[100px]">
                              <Label className="text-xs text-muted-foreground">{provider}</Label>
                            </div>
                            <div className="space-y-1 flex-1">
                              <Label htmlFor={`pb-month-${provider}`} className="text-xs">Monthly USD</Label>
                              <Input
                                id={`pb-month-${provider}`}
                                type="number"
                                min={0}
                                step="0.01"
                                placeholder="No limit"
                                value={draft.month_usd}
                                onChange={(e) => setProviderBudgets((prev) => ({
                                  ...prev,
                                  [provider]: { ...prev[provider], month_usd: e.target.value },
                                }))}
                              />
                            </div>
                            <div className="space-y-1 flex-1">
                              <Label htmlFor={`pb-day-${provider}`} className="text-xs">Daily USD</Label>
                              <Input
                                id={`pb-day-${provider}`}
                                type="number"
                                min={0}
                                step="0.01"
                                placeholder="No limit"
                                value={draft.day_usd}
                                onChange={(e) => setProviderBudgets((prev) => ({
                                  ...prev,
                                  [provider]: { ...prev[provider], day_usd: e.target.value },
                                }))}
                              />
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              type="button"
                              className="text-destructive"
                              onClick={() => setProviderBudgets((prev) => {
                                const next = { ...prev };
                                delete next[provider];
                                return next;
                              })}
                            >
                              Remove
                            </Button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <div>
                    <h3 className="mb-3 text-sm font-medium">Thresholds & Enforcement</h3>
                    <div className="overflow-x-auto rounded-md border">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Metric</TableHead>
                            <TableHead>Warning %</TableHead>
                            <TableHead>Critical %</TableHead>
                            <TableHead>Enforcement</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {BUDGET_METRICS.map((metric) => (
                            <TableRow key={metric.key}>
                              <TableCell className="font-medium">{metric.label}</TableCell>
                              <TableCell>
                                <Input
                                  type="number"
                                  min={1}
                                  max={100}
                                  step="1"
                                  value={editForm.thresholds[metric.key].warning}
                                  onChange={(event) => {
                                    updateEditThreshold(metric.key, 'warning', event.target.value);
                                  }}
                                  data-testid={`budget-threshold-warning-${metric.key}`}
                                />
                              </TableCell>
                              <TableCell>
                                <Input
                                  type="number"
                                  min={1}
                                  max={100}
                                  step="1"
                                  value={editForm.thresholds[metric.key].critical}
                                  onChange={(event) => {
                                    updateEditThreshold(metric.key, 'critical', event.target.value);
                                  }}
                                  data-testid={`budget-threshold-critical-${metric.key}`}
                                />
                              </TableCell>
                              <TableCell>
                                <Select
                                  value={editForm.enforcement[metric.key]}
                                  onChange={(event) => {
                                    updateEditEnforcement(metric.key, event.target.value as EnforcementMode);
                                  }}
                                  data-testid={`budget-enforcement-${metric.key}`}
                                >
                                  {ENFORCEMENT_OPTIONS.map((option) => (
                                    <option key={option.value} value={option.value}>
                                      {option.label}
                                    </option>
                                  ))}
                                </Select>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                    {BUDGET_METRICS.map((metric) => editErrors.thresholds[metric.key]).filter(Boolean).map((message, index) => (
                      <p key={`budget-edit-error-${index}`} className="mt-2 text-xs text-destructive">{message}</p>
                    ))}
                  </div>
                </div>
              )}

              <DialogFooter>
                <Button variant="outline" onClick={closeEditDialog} disabled={savingBudget}>
                  Cancel
                </Button>
                <Button onClick={() => { void saveBudgetEdits(); }} loading={savingBudget} loadingText="Saving...">
                  Save Budget
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

// Wrap with Suspense for useSearchParams
export default function BudgetsPage() {
  return (
    <Suspense fallback={
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8">
              <div className="h-8 w-32 bg-muted rounded animate-pulse mb-2" />
              <div className="h-4 w-64 bg-muted rounded animate-pulse" />
            </div>
            <div className="h-96 bg-muted rounded animate-pulse" />
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    }>
      <BudgetsPageContent />
    </Suspense>
  );
}
