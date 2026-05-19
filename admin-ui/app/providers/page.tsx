'use client';

import Link from 'next/link';
import { Fragment, useCallback, useEffect, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/toast';
import { Checkbox } from '@/components/ui/checkbox';
import { Cpu, RefreshCw, CheckCircle, XCircle, Key, ExternalLink, Plus, Trash2, Search, Building2, User, Settings, Activity, ChevronRight, ChevronDown } from 'lucide-react';
import { api } from '@/lib/api-client';
import { getDeprecatedModelNotice } from '@/lib/deprecated-models';
import { buildProviderTokenTrendMap, buildSparklinePoints } from '@/lib/provider-token-trends';
import { TableSkeleton } from '@/components/ui/skeleton';
import { LLMProvider, LLMProviderOverride, User as UserType, Organization } from '@/types';
import { CardSkeleton } from '@/components/ui/skeleton';
import { logger } from '@/lib/logger';

interface ByokKey {
  id?: string;
  provider: string;
  key_hint?: string;
  created_at?: string;
}

interface ProviderConfig {
  enabled?: boolean;
  models?: string[];
  default_model?: string;
  override?: LLMProviderOverride;
  [key: string]: unknown;
}

interface OverrideDialogState {
  isOpen: boolean;
  provider: LLMProvider | null;
  override: LLMProviderOverride | null;
  enabled: boolean;
  allowedModels: string;
  defaultModel: string;
  baseUrl: string;
  apiKey: string;
  clearApiKey: boolean;
  isSaving: boolean;
  isDeleting: boolean;
}

interface ByokState {
  users: UserType[];
  organizations: Organization[];
  selectedUser: UserType | null;
  selectedOrg: Organization | null;
  userKeys: ByokKey[];
  orgKeys: ByokKey[];
  isLoading: boolean;
  userSearch: string;
  userLimit: number;
}

interface AddByokDialogState {
  isOpen: boolean;
  mode: 'user' | 'org';
  provider: string;
  customProviderName: string;
  apiKey: string;
  isAdding: boolean;
}

interface DeprecatedModelDialogState {
  isOpen: boolean;
  providerName: string;
  modelName: string;
  replacement: string;
  requestsLast7d: number | null;
  isLoading: boolean;
}

interface LlmUsageSummaryRow {
  group_value: string;
  group_value_secondary?: string | null;
  requests: number;
  errors: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  total_cost_usd: number;
  latency_avg_ms?: number | null;
}

interface LlmUsageLogRow {
  model?: string | null;
  status?: number | null;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  total_cost_usd?: number | null;
  latency_ms?: number | null;
}

interface ProviderUsageMetrics {
  requests: number;
  errors: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  totalCostUsd: number;
  avgLatencyMs: number | null;
}

interface ProviderModelUsageMetrics {
  model: string;
  requests: number;
  errors: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  totalCostUsd: number;
  avgLatencyMs: number | null;
}

interface ProviderModelUsageState {
  isLoading: boolean;
  isLoaded: boolean;
  error: string;
  items: ProviderModelUsageMetrics[];
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const getStringValue = (value: unknown): string =>
  typeof value === 'string' ? value : '';

const USAGE_WINDOW_DAYS = 7;
const TREND_DAYS = 7;
const compactNumberFormatter = new Intl.NumberFormat('en-US', {
  notation: 'compact',
  maximumFractionDigits: 1,
});
const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const percentFormatter = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const getUsageWindow = () => {
  const end = new Date();
  const start = new Date(end.getTime() - USAGE_WINDOW_DAYS * 24 * 60 * 60 * 1000);
  return { start: start.toISOString(), end: end.toISOString() };
};

const normalizeLlmSummaryRows = (payload: unknown): LlmUsageSummaryRow[] => {
  if (Array.isArray(payload)) return payload as LlmUsageSummaryRow[];
  if (isRecord(payload) && Array.isArray(payload.items)) {
    return payload.items as LlmUsageSummaryRow[];
  }
  return [];
};

const normalizeLlmUsageLogRows = (payload: unknown): LlmUsageLogRow[] => {
  if (Array.isArray(payload)) return payload as LlmUsageLogRow[];
  if (isRecord(payload) && Array.isArray(payload.items)) {
    return payload.items as LlmUsageLogRow[];
  }
  return [];
};

const formatCompactNumber = (value: number) => compactNumberFormatter.format(value);

const formatCurrency = (value: number) => currencyFormatter.format(value);

const formatErrorRate = (errors: number, requests: number) =>
  requests > 0 ? percentFormatter.format(errors / requests) : '0.0%';

const formatLatency = (latencyMs: number | null) => {
  if (latencyMs === null || !Number.isFinite(latencyMs)) return '—';
  return `${Math.round(latencyMs)} ms`;
};

interface ProviderTrendSparklineProps {
  providerKey: string;
  providerLabel: string;
  series: number[];
}

function ProviderTrendSparkline({ providerKey, providerLabel, series }: ProviderTrendSparklineProps) {
  const normalizedSeries = series.length > 0
    ? series
    : Array.from({ length: TREND_DAYS }, () => 0);
  const points = buildSparklinePoints(normalizedSeries);
  const totalTokens = normalizedSeries.reduce((sum, value) => sum + value, 0);
  const hasUsage = totalTokens > 0;

  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-border/80 bg-background px-2 py-1.5">
      <div className="min-w-0">
        <div className="truncate text-xs font-medium">{providerLabel}</div>
        <div className="text-[11px] text-muted-foreground">
          {hasUsage ? `${formatCompactNumber(totalTokens)} tokens` : 'No usage'}
        </div>
      </div>
      <svg
        data-testid={`provider-token-sparkline-${providerKey}`}
        role="img"
        aria-label={`${providerLabel} token trend`}
        viewBox="0 0 84 24"
        className="h-6 w-20 shrink-0"
      >
        <polyline
          fill="none"
          stroke={hasUsage ? 'currentColor' : '#94a3b8'}
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />
      </svg>
    </div>
  );
}

interface ByokKeysTableProps {
  keys: ByokKey[];
  onDelete: (provider: string) => void;
  formatProviderName: (name: string) => string;
  deletingProvider?: string | null;
}

function ByokKeysTable({ keys, onDelete, formatProviderName, deletingProvider }: ByokKeysTableProps) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Provider</TableHead>
          <TableHead>Key Hint</TableHead>
          <TableHead>Created</TableHead>
          <TableHead className="text-right">Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {keys.map((key) => {
          const isDeleting = deletingProvider === key.provider;
          return (
          <TableRow key={key.provider}>
            <TableCell>
              <div className="font-medium">{formatProviderName(key.provider)}</div>
            </TableCell>
            <TableCell>
              <code className="text-xs bg-muted px-2 py-1 rounded">
                {key.key_hint || '****...****'}
              </code>
            </TableCell>
            <TableCell className="text-sm text-muted-foreground">
              {key.created_at
                ? new Date(key.created_at).toLocaleDateString()
                : '-'}
            </TableCell>
            <TableCell className="text-right">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onDelete(key.provider)}
                disabled={isDeleting}
                title={isDeleting ? 'Deleting key' : 'Delete key'}
                aria-label={isDeleting ? 'Deleting key' : 'Delete key'}
                loading={isDeleting}
                className="text-red-500 hover:text-red-500"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </TableCell>
          </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

export default function ProvidersPage() {
  const confirm = useConfirm();
  const { success: toastSuccess, error: toastError } = useToast();
  const [providers, setProviders] = useState<LLMProvider[]>([]);
  const [providerOverrides, setProviderOverrides] = useState<Record<string, LLMProviderOverride>>({});
  const [providerUsageSummary, setProviderUsageSummary] = useState<Record<string, ProviderUsageMetrics>>({});
  const [providerUsageUnavailable, setProviderUsageUnavailable] = useState(false);
  const [providerTokenTrends, setProviderTokenTrends] = useState<Record<string, number[]>>({});
  const [providerTokenTrendUnavailable, setProviderTokenTrendUnavailable] = useState(false);
  const [expandedProviderUsageRows, setExpandedProviderUsageRows] = useState<Record<string, boolean>>({});
  const [providerModelUsage, setProviderModelUsage] = useState<Record<string, ProviderModelUsageState>>({});
  const [loading, setLoading] = useState(true);
  const [overrideDialog, setOverrideDialog] = useState<OverrideDialogState>({
    isOpen: false,
    provider: null,
    override: null,
    enabled: true,
    allowedModels: '',
    defaultModel: '',
    baseUrl: '',
    apiKey: '',
    clearApiKey: false,
    isSaving: false,
    isDeleting: false,
  });
  const [testingProvider, setTestingProvider] = useState<string | null>(null);
  const [deletingUserByokProvider, setDeletingUserByokProvider] = useState<string | null>(null);
  const [deletingOrgByokProvider, setDeletingOrgByokProvider] = useState<string | null>(null);

  // BYOK management state
  const [byokState, setByokState] = useState<ByokState>({
    users: [],
    organizations: [],
    selectedUser: null,
    selectedOrg: null,
    userKeys: [],
    orgKeys: [],
    isLoading: false,
    userSearch: '',
    userLimit: 20,
  });

  // Add BYOK dialog
  const [addByokDialog, setAddByokDialog] = useState<AddByokDialogState>({
    isOpen: false,
    mode: 'user',
    provider: '',
    customProviderName: '',
    apiKey: '',
    isAdding: false,
  });
  const [deprecatedModelDialog, setDeprecatedModelDialog] = useState<DeprecatedModelDialogState>({
    isOpen: false,
    providerName: '',
    modelName: '',
    replacement: '',
    requestsLast7d: null,
    isLoading: false,
  });

  const updateOverrideDialog = (updates: Partial<OverrideDialogState>) => {
    setOverrideDialog((prev) => ({ ...prev, ...updates }));
  };

  const updateByokState = (updates: Partial<ByokState>) => {
    setByokState((prev) => ({ ...prev, ...updates }));
  };

  const updateAddByokDialog = (updates: Partial<AddByokDialogState>) => {
    setAddByokDialog((prev) => ({ ...prev, ...updates }));
  };

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const usageWindow = getUsageWindow();

      const [providersData, usersData, orgsData, overridesData, usageSummaryData, usageTrendData] = await Promise.allSettled([
        api.getLLMProviders(),
        api.getUsers(),
        api.getOrganizations(),
        api.getLLMProviderOverrides(),
        api.getLlmUsageSummary({
          group_by: 'provider',
          start: usageWindow.start,
          end: usageWindow.end,
        }),
        api.getLlmUsageSummary({
          group_by: ['provider', 'day'],
          start: usageWindow.start,
          end: usageWindow.end,
        }),
      ]);

      setExpandedProviderUsageRows({});
      setProviderModelUsage({});

      if (providersData.status === 'fulfilled') {
        let providersArray: LLMProvider[] = [];
        const payload = providersData.value;
        if (Array.isArray(payload)) {
          providersArray = payload;
        } else if (payload && typeof payload === 'object' && Array.isArray((payload as { providers?: unknown }).providers)) {
          providersArray = (payload as { providers: LLMProvider[] }).providers;
        } else if (payload && typeof payload === 'object') {
          providersArray = Object.entries(payload as Record<string, ProviderConfig>).map(([name, value]) => ({
            ...value,
            name,
            enabled: value.enabled ?? true,
            models: value.models || [],
          }));
        }
        setProviders(providersArray);
      }

      if (usersData.status === 'fulfilled') {
        const users = Array.isArray(usersData.value) ? usersData.value : [];
        setByokState((prev) => ({ ...prev, users }));
      }

      if (orgsData.status === 'fulfilled') {
        const organizations = Array.isArray(orgsData.value) ? orgsData.value : [];
        setByokState((prev) => ({ ...prev, organizations }));
      }

      if (overridesData.status === 'fulfilled') {
        const items = (overridesData.value && typeof overridesData.value === 'object' && Array.isArray((overridesData.value as { items?: unknown }).items))
          ? (overridesData.value as { items: LLMProviderOverride[] }).items
          : [];
        const map: Record<string, LLMProviderOverride> = {};
        items.forEach((item) => {
          if (item?.provider) {
            map[item.provider.toLowerCase()] = item;
          }
        });
        setProviderOverrides(map);
      }

      if (usageSummaryData.status === 'fulfilled') {
        const rows = normalizeLlmSummaryRows(usageSummaryData.value);
        const nextSummary: Record<string, ProviderUsageMetrics> = {};
        rows.forEach((row) => {
          const key = row.group_value?.trim().toLowerCase();
          if (!key) return;
          nextSummary[key] = {
            requests: Number.isFinite(row.requests) ? row.requests : 0,
            errors: Number.isFinite(row.errors) ? row.errors : 0,
            inputTokens: Number.isFinite(row.input_tokens) ? row.input_tokens : 0,
            outputTokens: Number.isFinite(row.output_tokens) ? row.output_tokens : 0,
            totalTokens: Number.isFinite(row.total_tokens) ? row.total_tokens : 0,
            totalCostUsd: Number.isFinite(row.total_cost_usd) ? row.total_cost_usd : 0,
            avgLatencyMs: Number.isFinite(row.latency_avg_ms ?? NaN) ? (row.latency_avg_ms ?? null) : null,
          };
        });
        setProviderUsageSummary(nextSummary);
        setProviderUsageUnavailable(false);
      } else {
        setProviderUsageSummary({});
        setProviderUsageUnavailable(true);
      }

      if (usageTrendData.status === 'fulfilled') {
        const rows = normalizeLlmSummaryRows(usageTrendData.value);
        const trendMap = buildProviderTokenTrendMap(rows, {
          days: TREND_DAYS,
          endDate: new Date(usageWindow.end),
        });
        setProviderTokenTrends(trendMap);
        setProviderTokenTrendUnavailable(false);
      } else {
        setProviderTokenTrends({});
        setProviderTokenTrendUnavailable(true);
      }
    } catch (err: unknown) {
      logger.error('Failed to load data', { component: 'ProvidersPage', error: err instanceof Error ? err.message : String(err) });
      setProviderUsageSummary({});
      setProviderUsageUnavailable(true);
      setProviderTokenTrends({});
      setProviderTokenTrendUnavailable(true);
      toastError('Failed to load providers', err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [toastError]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const loadUserByokKeys = async (user: UserType) => {
    updateByokState({ selectedUser: user, isLoading: true });
    try {
      const keys = await api.getUserByokKeys(user.id.toString());
      updateByokState({ userKeys: Array.isArray(keys) ? keys : [] });
    } catch (err: unknown) {
      logger.error('Failed to load user BYOK keys', { component: 'ProvidersPage', error: err instanceof Error ? err.message : String(err) });
      updateByokState({ userKeys: [] });
      toastError('Failed to load user keys', err instanceof Error ? err.message : 'Unable to load BYOK keys');
    } finally {
      updateByokState({ isLoading: false });
    }
  };

  const loadOrgByokKeys = async (org: Organization) => {
    updateByokState({ selectedOrg: org, isLoading: true });
    try {
      const keys = await api.getOrgByokKeys(org.id.toString());
      updateByokState({ orgKeys: Array.isArray(keys) ? keys : [] });
    } catch (err: unknown) {
      logger.error('Failed to load org BYOK keys', { component: 'ProvidersPage', error: err instanceof Error ? err.message : String(err) });
      updateByokState({ orgKeys: [] });
      toastError('Failed to load org keys', err instanceof Error ? err.message : 'Unable to load BYOK keys');
    } finally {
      updateByokState({ isLoading: false });
    }
  };

  const handleAddByok = async () => {
    const providerName = addByokDialog.provider === 'other'
      ? addByokDialog.customProviderName.trim()
      : addByokDialog.provider.trim();
    if (!canSubmitByok) {
      toastError('Missing fields', byokValidationError || 'Provider and API key are required');
      return;
    }

    updateAddByokDialog({ isAdding: true });
    try {
      const data = {
        provider: providerName.trim().toLowerCase(),
        api_key: addByokDialog.apiKey.trim(),
      };

      if (addByokDialog.mode === 'user' && byokState.selectedUser) {
        await api.createUserByokKey(byokState.selectedUser.id.toString(), data);
        await loadUserByokKeys(byokState.selectedUser);
        toastSuccess('BYOK key added', `Added for ${byokState.selectedUser.email}`);
      } else if (addByokDialog.mode === 'org' && byokState.selectedOrg) {
        await api.createOrgByokKey(byokState.selectedOrg.id.toString(), data);
        await loadOrgByokKeys(byokState.selectedOrg);
        toastSuccess('BYOK key added', `Added for ${byokState.selectedOrg.name}`);
      }

      updateAddByokDialog({
        isOpen: false,
        provider: '',
        customProviderName: '',
        apiKey: '',
      });
    } catch (err: unknown) {
      logger.error('Failed to add BYOK key', { component: 'ProvidersPage', error: err instanceof Error ? err.message : String(err) });
      toastError('Failed to add BYOK key', err instanceof Error ? err.message : 'Try again.');
    } finally {
      updateAddByokDialog({ isAdding: false });
    }
  };

  const handleDeleteUserByok = async (provider: string) => {
    if (!byokState.selectedUser) return;
    if (deletingUserByokProvider === provider) return;

    const confirmed = await confirm({
      title: 'Delete API Key',
      message: `Delete ${provider} key for ${byokState.selectedUser.email}?`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'key',
    });
    if (!confirmed) return;

    try {
      setDeletingUserByokProvider(provider);
      await api.deleteUserByokKey(byokState.selectedUser.id.toString(), provider);
      await loadUserByokKeys(byokState.selectedUser);
      toastSuccess('BYOK key deleted', `Removed ${provider} for ${byokState.selectedUser.email}`);
    } catch (err: unknown) {
      logger.error('Failed to delete BYOK key', { component: 'ProvidersPage', error: err instanceof Error ? err.message : String(err) });
      toastError('Failed to delete BYOK key', err instanceof Error ? err.message : 'Try again.');
    } finally {
      setDeletingUserByokProvider((prev) => (prev === provider ? null : prev));
    }
  };

  const handleDeleteOrgByok = async (provider: string) => {
    if (!byokState.selectedOrg) return;
    if (deletingOrgByokProvider === provider) return;

    const confirmed = await confirm({
      title: 'Delete API Key',
      message: `Delete ${provider} key for ${byokState.selectedOrg.name}?`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'key',
    });
    if (!confirmed) return;

    try {
      setDeletingOrgByokProvider(provider);
      await api.deleteOrgByokKey(byokState.selectedOrg.id.toString(), provider);
      await loadOrgByokKeys(byokState.selectedOrg);
      toastSuccess('BYOK key deleted', `Removed ${provider} for ${byokState.selectedOrg.name}`);
    } catch (err: unknown) {
      logger.error('Failed to delete BYOK key', { component: 'ProvidersPage', error: err instanceof Error ? err.message : String(err) });
      toastError('Failed to delete BYOK key', err instanceof Error ? err.message : 'Try again.');
    } finally {
      setDeletingOrgByokProvider((prev) => (prev === provider ? null : prev));
    }
  };

  const openAddByokDialog = (mode: 'user' | 'org') => {
    updateAddByokDialog({
      mode,
      provider: '',
      customProviderName: '',
      apiKey: '',
      isAdding: false,
      isOpen: true,
    });
  };

  const getProviderDocs = (name: string): string | null => {
    const docs: Record<string, string> = {
      openai: 'https://platform.openai.com/docs',
      anthropic: 'https://docs.anthropic.com',
      google: 'https://ai.google.dev/docs',
      cohere: 'https://docs.cohere.com',
      groq: 'https://console.groq.com/docs',
      mistral: 'https://docs.mistral.ai',
      deepseek: 'https://platform.deepseek.com/docs',
      ollama: 'https://ollama.com/docs',
      openrouter: 'https://openrouter.ai/docs',
    };
    return docs[name.toLowerCase()] || null;
  };

  const formatProviderName = (name: string): string => {
    const names: Record<string, string> = {
      openai: 'OpenAI',
      anthropic: 'Anthropic',
      google: 'Google AI',
      cohere: 'Cohere',
      groq: 'Groq',
      mistral: 'Mistral AI',
      deepseek: 'DeepSeek',
      ollama: 'Ollama',
      openrouter: 'OpenRouter',
      huggingface: 'HuggingFace',
      kobold: 'KoboldCpp',
      llamacpp: 'Llama.cpp',
      tabbyapi: 'TabbyAPI',
      vllm: 'vLLM',
      aphrodite: 'Aphrodite',
      custom: 'Custom OpenAI',
    };
    return names[name.toLowerCase()] || name;
  };

  const getOverrideForProvider = (provider: LLMProvider): LLMProviderOverride | undefined => {
    const key = provider.name.toLowerCase();
    return providerOverrides[key] || provider.override;
  };

  const openOverrideDialog = (provider: LLMProvider) => {
    const override = getOverrideForProvider(provider);
    setOverrideDialog({
      isOpen: true,
      provider,
      override: override || null,
      enabled: override?.is_enabled ?? provider.enabled ?? true,
      allowedModels: override?.allowed_models?.join(', ') ?? '',
      defaultModel: getStringValue(override?.config?.default_model),
      baseUrl: getStringValue(override?.credential_fields?.base_url),
      apiKey: '',
      clearApiKey: false,
      isSaving: false,
      isDeleting: false,
    });
  };

  const handleSaveOverride = async () => {
    if (!overrideDialog.provider) return;
    updateOverrideDialog({ isSaving: true });

    const allowedModels = overrideDialog.allowedModels
      .split(',')
      .map((model) => model.trim())
      .filter(Boolean);
    const defaultModelValue = overrideDialog.defaultModel.trim();
    const overrideConfig = overrideDialog.override?.config;
    const configSource = isRecord(overrideConfig) ? overrideConfig : {};
    const config: Record<string, unknown> = { ...configSource };
    if (defaultModelValue) {
      config.default_model = defaultModelValue;
    } else if ('default_model' in configSource) {
      delete config.default_model;
    }
    const credentialFields: Record<string, unknown> = {};
    const overrideCredentials = overrideDialog.override?.credential_fields;
    const credentialSource = isRecord(overrideCredentials)
      ? overrideCredentials
      : {};
    const existingBaseUrl = getStringValue(credentialSource.base_url);
    const trimmedBaseUrl = overrideDialog.baseUrl.trim();
    if (trimmedBaseUrl) {
      credentialFields.base_url = trimmedBaseUrl;
    }

    const payload: Record<string, unknown> = {
      is_enabled: overrideDialog.enabled,
      allowed_models: allowedModels,
    };

    if (defaultModelValue || 'default_model' in configSource) {
      payload.config = config;
    }

    if (Object.keys(credentialFields).length > 0) {
      payload.credential_fields = credentialFields;
    } else if (existingBaseUrl) {
      payload.credential_fields = {};
    }

    if (overrideDialog.apiKey.trim()) {
      payload.api_key = overrideDialog.apiKey.trim();
    } else if (overrideDialog.clearApiKey) {
      payload.clear_api_key = true;
    }

    try {
      await api.updateLLMProviderOverride(overrideDialog.provider.name, payload);
      toastSuccess(
        'Override updated',
        `Updated ${formatProviderName(overrideDialog.provider.name)} overrides.`
      );
      updateOverrideDialog({ isOpen: false });
      await loadData();
    } catch (err: unknown) {
      logger.error('Failed to update provider override', { component: 'ProvidersPage', error: err instanceof Error ? err.message : String(err) });
      toastError('Failed to update override', err instanceof Error ? err.message : 'Try again.');
    } finally {
      updateOverrideDialog({ isSaving: false });
    }
  };

  const handleDeleteOverride = async () => {
    if (!overrideDialog.provider) return;
    if (overrideDialog.isDeleting) return;
    const confirmed = await confirm({
      title: 'Remove Override',
      message: `Remove override for ${formatProviderName(overrideDialog.provider.name)}?`,
      confirmText: 'Remove',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;
    try {
      updateOverrideDialog({ isDeleting: true });
      await api.deleteLLMProviderOverride(overrideDialog.provider.name);
      toastSuccess(
        'Override removed',
        `Removed ${formatProviderName(overrideDialog.provider.name)} override.`
      );
      updateOverrideDialog({ isOpen: false });
      await loadData();
    } catch (err: unknown) {
      logger.error('Failed to delete provider override', { component: 'ProvidersPage', error: err instanceof Error ? err.message : String(err) });
      toastError('Failed to remove override', err instanceof Error ? err.message : 'Try again.');
    } finally {
      updateOverrideDialog({ isDeleting: false });
    }
  };

  const handleTestProvider = async (provider: LLMProvider) => {
    const override = getOverrideForProvider(provider);
    setTestingProvider(provider.name);
    try {
      const overrideDefaultModel = getStringValue(override?.config?.default_model);
      const response = await api.testLLMProvider({
        provider: provider.name,
        model: overrideDefaultModel || provider.default_model || undefined,
        use_override: true,
      }) as { model?: string | null };
      toastSuccess(
        'Connectivity check OK',
        `${formatProviderName(provider.name)} (${response?.model || 'default'})`
      );
    } catch (err: unknown) {
      logger.error('Failed to test provider', { component: 'ProvidersPage', error: err instanceof Error ? err.message : String(err) });
      toastError('Provider test failed', err instanceof Error ? err.message : 'Try again.');
    } finally {
      setTestingProvider(null);
    }
  };

  const loadProviderModelUsage = useCallback(async (providerName: string): Promise<ProviderModelUsageMetrics[] | null> => {
    const providerKey = providerName.toLowerCase();
    setProviderModelUsage((prev) => ({
      ...prev,
      [providerKey]: {
        ...(prev[providerKey] ?? {
          isLoading: false,
          isLoaded: false,
          error: '',
          items: [],
        }),
        isLoading: true,
        error: '',
      },
    }));
    try {
      const usageWindow = getUsageWindow();
      const usageLog = await api.getLlmUsage({
        provider: providerName,
        start: usageWindow.start,
        end: usageWindow.end,
        page: '1',
        limit: '500',
      });
      const rows = normalizeLlmUsageLogRows(usageLog);
      const aggregates = new Map<string, {
        requests: number;
        errors: number;
        inputTokens: number;
        outputTokens: number;
        totalTokens: number;
        totalCostUsd: number;
        latencySum: number;
        latencyCount: number;
      }>();

      rows.forEach((row) => {
        const modelName = typeof row.model === 'string' && row.model.trim().length > 0
          ? row.model
          : '(unknown)';
        const current = aggregates.get(modelName) ?? {
          requests: 0,
          errors: 0,
          inputTokens: 0,
          outputTokens: 0,
          totalTokens: 0,
          totalCostUsd: 0,
          latencySum: 0,
          latencyCount: 0,
        };
        current.requests += 1;
        if ((row.status ?? 0) >= 400) current.errors += 1;
        current.inputTokens += Number.isFinite(row.prompt_tokens ?? NaN) ? (row.prompt_tokens ?? 0) : 0;
        current.outputTokens += Number.isFinite(row.completion_tokens ?? NaN) ? (row.completion_tokens ?? 0) : 0;
        if (Number.isFinite(row.total_tokens ?? NaN)) {
          current.totalTokens += row.total_tokens ?? 0;
        } else {
          current.totalTokens += (row.prompt_tokens ?? 0) + (row.completion_tokens ?? 0);
        }
        current.totalCostUsd += Number.isFinite(row.total_cost_usd ?? NaN) ? (row.total_cost_usd ?? 0) : 0;
        if (Number.isFinite(row.latency_ms ?? NaN)) {
          current.latencySum += row.latency_ms ?? 0;
          current.latencyCount += 1;
        }
        aggregates.set(modelName, current);
      });

      const items: ProviderModelUsageMetrics[] = Array.from(aggregates.entries())
        .map(([model, data]) => ({
          model,
          requests: data.requests,
          errors: data.errors,
          inputTokens: data.inputTokens,
          outputTokens: data.outputTokens,
          totalTokens: data.totalTokens,
          totalCostUsd: data.totalCostUsd,
          avgLatencyMs: data.latencyCount > 0 ? data.latencySum / data.latencyCount : null,
        }))
        .sort((a, b) => b.requests - a.requests || b.totalCostUsd - a.totalCostUsd);

      setProviderModelUsage((prev) => ({
        ...prev,
        [providerKey]: {
          isLoading: false,
          isLoaded: true,
          error: '',
          items,
        },
      }));
      return items;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load model usage';
      setProviderModelUsage((prev) => ({
        ...prev,
        [providerKey]: {
          isLoading: false,
          isLoaded: true,
          error: message,
          items: [],
        },
      }));
      toastError('Failed to load model breakdown', message);
      return null;
    }
  }, [toastError]);

  const toggleProviderUsageRow = useCallback((providerName: string) => {
    const providerKey = providerName.toLowerCase();
    const isExpanded = expandedProviderUsageRows[providerKey] ?? false;
    setExpandedProviderUsageRows((prev) => ({
      ...prev,
      [providerKey]: !isExpanded,
    }));
    if (!isExpanded) {
      const usageState = providerModelUsage[providerKey];
      if (!usageState?.isLoaded && !usageState?.isLoading) {
        void loadProviderModelUsage(providerName);
      }
    }
  }, [expandedProviderUsageRows, loadProviderModelUsage, providerModelUsage]);

  const openDeprecatedModelDialog = useCallback(async (providerName: string, modelName: string) => {
    const notice = getDeprecatedModelNotice(modelName);
    if (!notice) return;

    setDeprecatedModelDialog({
      isOpen: true,
      providerName,
      modelName,
      replacement: notice.replacement,
      requestsLast7d: null,
      isLoading: true,
    });

    const providerKey = providerName.toLowerCase();
    const existingUsageState = providerModelUsage[providerKey];
    let usageItems = existingUsageState?.isLoaded ? (existingUsageState.items ?? []) : null;
    if (!usageItems) {
      usageItems = await loadProviderModelUsage(providerName);
    }

    const requestsLast7d = usageItems
      ? usageItems.reduce((total, usageItem) => {
        const usageNotice = getDeprecatedModelNotice(usageItem.model);
        if (usageNotice?.id === notice.id) {
          return total + usageItem.requests;
        }
        return total;
      }, 0)
      : null;

    setDeprecatedModelDialog((prev) => {
      if (!prev.isOpen || prev.providerName !== providerName || prev.modelName !== modelName) {
        return prev;
      }
      return {
        ...prev,
        requestsLast7d: requestsLast7d ?? 0,
        isLoading: false,
      };
    });
  }, [loadProviderModelUsage, providerModelUsage]);

  const filteredUsers = byokState.users.filter(
    (u) =>
      byokState.userSearch === '' ||
      u.email?.toLowerCase().includes(byokState.userSearch.toLowerCase()) ||
      u.username?.toLowerCase().includes(byokState.userSearch.toLowerCase())
  );
  const visibleUsers = filteredUsers.slice(0, byokState.userLimit);

  const enabledProviders = providers.filter((p) => p.enabled);
  const disabledProviders = providers.filter((p) => !p.enabled);

  // Common provider options for BYOK
  const commonProviders = ['openai', 'anthropic', 'google', 'cohere', 'groq', 'mistral', 'deepseek', 'openrouter'];
  const byokValidationError = (() => {
    if (addByokDialog.mode === 'user' && !byokState.selectedUser) {
      return 'Choose a user before adding a BYOK key.';
    }
    if (addByokDialog.mode === 'org' && !byokState.selectedOrg) {
      return 'Choose an organization before adding a BYOK key.';
    }
    if (!addByokDialog.provider) {
      return 'Select a provider.';
    }
    if (addByokDialog.provider === 'other' && !addByokDialog.customProviderName.trim()) {
      return 'Enter a provider name.';
    }
    if (!addByokDialog.apiKey.trim()) {
      return 'API key is required.';
    }
    return '';
  })();
  const canSubmitByok = byokValidationError === '';

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8 flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold">LLM Providers</h1>
                <p className="text-muted-foreground">
                  Manage LLM providers and BYOK secrets
                </p>
              </div>
              <Button variant="outline" onClick={loadData} disabled={loading}>
                <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>

            <Tabs defaultValue="providers" className="space-y-4">
              <TabsList>
                <TabsTrigger value="providers">
                  <Cpu className="mr-2 h-4 w-4" />
                  Providers
                </TabsTrigger>
                <TabsTrigger value="user-byok">
                  <User className="mr-2 h-4 w-4" />
                  User BYOK
                </TabsTrigger>
                <TabsTrigger value="org-byok">
                  <Building2 className="mr-2 h-4 w-4" />
                  Org BYOK
                </TabsTrigger>
              </TabsList>

              {/* Providers Tab */}
              <TabsContent value="providers">
                {/* Info Card */}
                <Card className="mb-6">
                  <CardContent className="pt-6">
                    <div className="flex items-start gap-4">
                      <Cpu className="h-8 w-8 text-primary mt-1" />
                      <div>
                        <h3 className="font-semibold">LLM Provider Configuration</h3>
                        <p className="text-sm text-muted-foreground mt-1">
                          tldw_server supports multiple LLM providers including OpenAI, Anthropic, Google, Cohere,
                          and local inference servers. Provider API keys can be configured in the server&apos;s config.txt
                          or .env file.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Summary Stats */}
                <div className="grid gap-4 md:grid-cols-3 mb-6">
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Total Providers</CardTitle>
                      <Cpu className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{providers.length}</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Enabled</CardTitle>
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-green-600">{enabledProviders.length}</div>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Disabled</CardTitle>
                      <XCircle className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-muted-foreground">{disabledProviders.length}</div>
                    </CardContent>
                  </Card>
                </div>

                {/* Providers Table */}
                <Card>
                  <CardHeader>
                    <CardTitle>Configured Providers</CardTitle>
                    <CardDescription>
                      All LLM providers available in the system, including usage and cost in the last {USAGE_WINDOW_DAYS} days
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="mb-4 space-y-2 rounded-lg border border-border/80 bg-muted/20 p-3">
                      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        Token Trend ({TREND_DAYS}d)
                      </div>
                      {providerTokenTrendUnavailable ? (
                        <p className="text-xs text-muted-foreground">
                          Token trend data is currently unavailable.
                        </p>
                      ) : (
                        <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                          {providers.map((provider) => {
                            const providerKey = provider.name.toLowerCase();
                            return (
                              <ProviderTrendSparkline
                                key={`trend-${provider.name}`}
                                providerKey={providerKey}
                                providerLabel={formatProviderName(provider.name)}
                                series={providerTokenTrends[providerKey] ?? []}
                              />
                            );
                          })}
                        </div>
                      )}
                    </div>
                    {loading ? (
                      <TableSkeleton rows={5} columns={4} />
                    ) : providers.length === 0 ? (
                      <div className="text-center text-muted-foreground py-8">
                        <Cpu className="h-12 w-12 mx-auto mb-2 opacity-50" />
                        <p>No LLM providers configured</p>
                        <p className="text-sm mt-1">Configure providers in config.txt or .env</p>
                      </div>
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow>
                          <TableHead>Provider</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead>Models</TableHead>
                          <TableHead className="text-right">Requests (7d)</TableHead>
                          <TableHead className="text-right">Tokens (7d)</TableHead>
                          <TableHead className="text-right">Cost (7d)</TableHead>
                          <TableHead className="text-right">Error Rate</TableHead>
                          <TableHead>Override</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {providers.map((provider) => {
                            const docsUrl = getProviderDocs(provider.name);
                            const override = getOverrideForProvider(provider);
                            const providerKey = provider.name.toLowerCase();
                            const usageMetrics = providerUsageSummary[providerKey];
                            const modelUsageState = providerModelUsage[providerKey];
                            const isExpanded = expandedProviderUsageRows[providerKey] ?? false;
                            const usageRequests = providerUsageUnavailable
                              ? '—'
                              : formatCompactNumber(usageMetrics?.requests ?? 0);
                            const usageTokens = providerUsageUnavailable
                              ? '—'
                              : formatCompactNumber(usageMetrics?.totalTokens ?? 0);
                            const usageCost = providerUsageUnavailable
                              ? '—'
                              : formatCurrency(usageMetrics?.totalCostUsd ?? 0);
                            const usageErrorRate = providerUsageUnavailable
                              ? '—'
                              : formatErrorRate(usageMetrics?.errors ?? 0, usageMetrics?.requests ?? 0);
                            return (
                              <Fragment key={provider.name}>
                                <TableRow>
                                  <TableCell>
                                    <div className="flex items-center gap-2">
                                      <div className="font-medium">{formatProviderName(provider.name)}</div>
                                      <code className="text-xs bg-muted px-1 rounded">{provider.name}</code>
                                    </div>
                                  </TableCell>
                                  <TableCell>
                                    {provider.enabled ? (
                                      <Badge variant="default" className="bg-green-500">
                                        <CheckCircle className="mr-1 h-3 w-3" />
                                        Enabled
                                      </Badge>
                                    ) : (
                                      <Badge variant="secondary">
                                        <XCircle className="mr-1 h-3 w-3" />
                                        Disabled
                                      </Badge>
                                    )}
                                  </TableCell>
                                  <TableCell>
                                    {provider.models && provider.models.length > 0 ? (
                                      <div className="flex flex-wrap gap-1 max-w-md">
                                        {provider.models.slice(0, 3).map((model: string) => {
                                          const deprecatedNotice = getDeprecatedModelNotice(model);
                                          return (
                                            <div key={model} className="inline-flex items-center gap-1">
                                              <Badge variant="outline" className="text-xs">
                                                {model}
                                              </Badge>
                                              {deprecatedNotice ? (
                                                <button
                                                  type="button"
                                                  className="inline-flex items-center rounded-full border border-amber-300 bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-900 hover:bg-amber-200"
                                                  onClick={() => { void openDeprecatedModelDialog(provider.name, model); }}
                                                  title={`View deprecation details for ${model}`}
                                                  aria-label={`View deprecation details for ${model}`}
                                                >
                                                  Deprecated
                                                </button>
                                              ) : null}
                                            </div>
                                          );
                                        })}
                                        {provider.models.length > 3 && (
                                          <Badge variant="outline" className="text-xs">
                                            +{provider.models.length - 3} more
                                          </Badge>
                                        )}
                                      </div>
                                    ) : (
                                      <span className="text-muted-foreground text-sm">-</span>
                                    )}
                                  </TableCell>
                                  <TableCell className="text-right tabular-nums">{usageRequests}</TableCell>
                                  <TableCell className="text-right tabular-nums">{usageTokens}</TableCell>
                                  <TableCell className="text-right tabular-nums">{usageCost}</TableCell>
                                  <TableCell className="text-right tabular-nums">{usageErrorRate}</TableCell>
                                  <TableCell>
                                    {override ? (
                                      <div className="space-y-1">
                                        <Badge variant="outline" className="text-xs">
                                          Override {override.is_enabled === false ? 'disabled' : 'active'}
                                        </Badge>
                                        {override.allowed_models?.length ? (
                                          <div className="text-xs text-muted-foreground">
                                            {override.allowed_models.length} models allowlisted
                                          </div>
                                        ) : null}
                                        {override.api_key_hint ? (
                                          <div className="text-xs text-muted-foreground">
                                            Key • ****{override.api_key_hint}
                                          </div>
                                        ) : null}
                                      </div>
                                    ) : (
                                      <span className="text-muted-foreground text-sm">-</span>
                                    )}
                                  </TableCell>
                                  <TableCell className="text-right">
                                    <div className="flex justify-end gap-2">
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        onClick={() => toggleProviderUsageRow(provider.name)}
                                        title={isExpanded ? 'Hide model breakdown' : 'Show model breakdown'}
                                        aria-label={`${isExpanded ? 'Collapse' : 'Expand'} model usage for ${provider.name}`}
                                      >
                                        {isExpanded ? (
                                          <ChevronDown className="h-4 w-4" />
                                        ) : (
                                          <ChevronRight className="h-4 w-4" />
                                        )}
                                      </Button>
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        onClick={() => openOverrideDialog(provider)}
                                        title="Manage override"
                                      >
                                        <Settings className="h-4 w-4" />
                                      </Button>
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        onClick={() => handleTestProvider(provider)}
                                        disabled={testingProvider === provider.name}
                                        loading={testingProvider === provider.name}
                                        title="Test connectivity"
                                      >
                                        <Activity className="h-4 w-4" />
                                      </Button>
                                      {docsUrl ? (
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          onClick={() => window.open(docsUrl, '_blank')}
                                          title="Open documentation"
                                        >
                                          <ExternalLink className="h-4 w-4" />
                                        </Button>
                                      ) : null}
                                      <Link
                                        href={`/usage?group_by=provider&provider=${encodeURIComponent(provider.name)}`}
                                        className="inline-flex h-9 items-center rounded-md px-2 text-xs font-medium text-primary hover:bg-accent hover:text-accent-foreground"
                                      >
                                        View usage
                                      </Link>
                                    </div>
                                  </TableCell>
                                </TableRow>
                                {isExpanded ? (
                                  <TableRow className="bg-muted/30">
                                    <TableCell colSpan={9}>
                                      <div className="space-y-3 py-2">
                                        <div className="text-sm font-medium">
                                          Per-model usage ({USAGE_WINDOW_DAYS}d)
                                        </div>
                                        {modelUsageState?.isLoading ? (
                                          <div className="text-sm text-muted-foreground" role="status" aria-live="polite">Loading model usage...</div>
                                        ) : modelUsageState?.error ? (
                                          <div className="text-sm text-red-600">{modelUsageState.error}</div>
                                        ) : (modelUsageState?.items?.length ?? 0) === 0 ? (
                                          <div className="text-sm text-muted-foreground">
                                            No model usage recorded in the last {USAGE_WINDOW_DAYS} days.
                                          </div>
                                        ) : (
                                          <Table>
                                            <TableHeader>
                                              <TableRow>
                                                <TableHead>Model</TableHead>
                                                <TableHead className="text-right">Requests</TableHead>
                                                <TableHead className="text-right">Input Tokens</TableHead>
                                                <TableHead className="text-right">Output Tokens</TableHead>
                                                <TableHead className="text-right">Total Tokens</TableHead>
                                                <TableHead className="text-right">Cost</TableHead>
                                                <TableHead className="text-right">Avg Latency</TableHead>
                                                <TableHead className="text-right">Error Rate</TableHead>
                                              </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                              {modelUsageState?.items.map((item) => {
                                                const deprecatedNotice = getDeprecatedModelNotice(item.model);
                                                return (
                                                  <TableRow key={`${provider.name}-${item.model}`}>
                                                    <TableCell>
                                                      <div className="flex flex-wrap items-center gap-2">
                                                        <code className="text-xs bg-muted px-2 py-1 rounded">{item.model}</code>
                                                        {deprecatedNotice ? (
                                                          <button
                                                            type="button"
                                                            className="inline-flex items-center rounded-full border border-amber-300 bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-900 hover:bg-amber-200"
                                                            onClick={() => { void openDeprecatedModelDialog(provider.name, item.model); }}
                                                            title={`View deprecation details for ${item.model}`}
                                                            aria-label={`View deprecation details for ${item.model}`}
                                                          >
                                                            Deprecated
                                                          </button>
                                                        ) : null}
                                                      </div>
                                                    </TableCell>
                                                    <TableCell className="text-right tabular-nums">{formatCompactNumber(item.requests)}</TableCell>
                                                    <TableCell className="text-right tabular-nums">{formatCompactNumber(item.inputTokens)}</TableCell>
                                                    <TableCell className="text-right tabular-nums">{formatCompactNumber(item.outputTokens)}</TableCell>
                                                    <TableCell className="text-right tabular-nums">{formatCompactNumber(item.totalTokens)}</TableCell>
                                                    <TableCell className="text-right tabular-nums">{formatCurrency(item.totalCostUsd)}</TableCell>
                                                    <TableCell className="text-right tabular-nums">{formatLatency(item.avgLatencyMs)}</TableCell>
                                                    <TableCell className="text-right tabular-nums">{formatErrorRate(item.errors, item.requests)}</TableCell>
                                                  </TableRow>
                                                );
                                              })}
                                            </TableBody>
                                          </Table>
                                        )}
                                      </div>
                                    </TableCell>
                                  </TableRow>
                                ) : null}
                              </Fragment>
                            );
                          })}
                        </TableBody>
                      </Table>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* User BYOK Tab */}
              <TabsContent value="user-byok">
                <div className="grid gap-6 lg:grid-cols-3">
                  {/* User Selection */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <User className="h-5 w-5" />
                        Select User
                      </CardTitle>
                      <CardDescription>Choose a user to manage their BYOK keys</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <div className="relative">
                          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                          <Input
                            placeholder="Search users..."
                            value={byokState.userSearch}
                            onChange={(e) => updateByokState({ userSearch: e.target.value, userLimit: 20 })}
                            className="pl-8"
                          />
                        </div>
                        <div className="max-h-80 overflow-y-auto space-y-1">
                          {visibleUsers.map((user) => (
                            <Button
                              key={user.id}
                              variant={byokState.selectedUser?.id === user.id ? 'default' : 'ghost'}
                              className="w-full justify-start text-left"
                              onClick={() => loadUserByokKeys(user)}
                            >
                              <User className="mr-2 h-4 w-4" />
                              <div className="truncate">
                                <div className="font-medium">{user.username || user.email}</div>
                                {user.username && (
                                  <div className="text-xs opacity-70">{user.email}</div>
                                )}
                              </div>
                            </Button>
                          ))}
                          {filteredUsers.length === 0 && (
                            <p className="text-center text-muted-foreground py-4">No users found</p>
                          )}
                          {filteredUsers.length > byokState.userLimit && (
                            <div className="space-y-2 pt-2">
                              <Button
                                variant="outline"
                                size="sm"
                                className="w-full"
                                onClick={() => updateByokState({ userLimit: byokState.userLimit + 20 })}
                              >
                                Show more
                              </Button>
                              <p className="text-center text-xs text-muted-foreground">
                                Showing {byokState.userLimit} of {filteredUsers.length} users
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* User BYOK Keys */}
                  <Card className="lg:col-span-2">
                    <CardHeader className="flex flex-row items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <Key className="h-5 w-5" />
                          User BYOK Keys
                        </CardTitle>
                        <CardDescription>
                          {byokState.selectedUser
                            ? `Managing keys for ${byokState.selectedUser.email}`
                            : 'Select a user to view their BYOK keys'}
                        </CardDescription>
                      </div>
                      {byokState.selectedUser && (
                        <Button size="sm" onClick={() => openAddByokDialog('user')}>
                          <Plus className="mr-2 h-4 w-4" />
                          Add Key
                        </Button>
                      )}
                    </CardHeader>
                    <CardContent>
                      {!byokState.selectedUser ? (
                        <div className="text-center text-muted-foreground py-8">
                          <Key className="h-12 w-12 mx-auto mb-2 opacity-50" />
                          <p>Select a user to manage their BYOK keys</p>
                        </div>
                      ) : byokState.isLoading ? (
                        <TableSkeleton rows={3} columns={3} />
                      ) : byokState.userKeys.length === 0 ? (
                        <div className="text-center text-muted-foreground py-8">
                          <Key className="h-12 w-12 mx-auto mb-2 opacity-50" />
                          <p>No BYOK keys configured for this user</p>
                          <Button size="sm" className="mt-4" onClick={() => openAddByokDialog('user')}>
                            <Plus className="mr-2 h-4 w-4" />
                            Add First Key
                          </Button>
                        </div>
                      ) : (
                        <ByokKeysTable
                          keys={byokState.userKeys}
                          onDelete={handleDeleteUserByok}
                          formatProviderName={formatProviderName}
                          deletingProvider={deletingUserByokProvider}
                        />
                      )}
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* Org BYOK Tab */}
              <TabsContent value="org-byok">
                <div className="grid gap-6 lg:grid-cols-3">
                  {/* Org Selection */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Building2 className="h-5 w-5" />
                        Select Organization
                      </CardTitle>
                      <CardDescription>Choose an organization to manage BYOK keys</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="max-h-80 overflow-y-auto space-y-1">
                        {byokState.organizations.length === 0 ? (
                          <p className="text-center text-muted-foreground py-4">No organizations found</p>
                        ) : (
                          byokState.organizations.map((org) => (
                            <Button
                              key={org.id}
                              variant={byokState.selectedOrg?.id === org.id ? 'default' : 'ghost'}
                              className="w-full justify-start text-left"
                              onClick={() => loadOrgByokKeys(org)}
                            >
                              <Building2 className="mr-2 h-4 w-4" />
                              <div className="truncate">
                                <div className="font-medium">{org.name}</div>
                                {org.description && (
                                  <div className="text-xs opacity-70 truncate">{org.description}</div>
                                )}
                              </div>
                            </Button>
                          ))
                        )}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Org BYOK Keys */}
                  <Card className="lg:col-span-2">
                    <CardHeader className="flex flex-row items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <Key className="h-5 w-5" />
                          Organization BYOK Keys
                        </CardTitle>
                        <CardDescription>
                          {byokState.selectedOrg
                            ? `Managing keys for ${byokState.selectedOrg.name}`
                            : 'Select an organization to view BYOK keys'}
                        </CardDescription>
                      </div>
                      {byokState.selectedOrg && (
                        <Button size="sm" onClick={() => openAddByokDialog('org')}>
                          <Plus className="mr-2 h-4 w-4" />
                          Add Key
                        </Button>
                      )}
                    </CardHeader>
                    <CardContent>
                      {!byokState.selectedOrg ? (
                        <div className="text-center text-muted-foreground py-8">
                          <Key className="h-12 w-12 mx-auto mb-2 opacity-50" />
                          <p>Select an organization to manage BYOK keys</p>
                        </div>
                      ) : byokState.isLoading ? (
                        <TableSkeleton rows={3} columns={3} />
                      ) : byokState.orgKeys.length === 0 ? (
                        <div className="text-center text-muted-foreground py-8">
                          <Key className="h-12 w-12 mx-auto mb-2 opacity-50" />
                          <p>No BYOK keys configured for this organization</p>
                          <Button size="sm" className="mt-4" onClick={() => openAddByokDialog('org')}>
                            <Plus className="mr-2 h-4 w-4" />
                            Add First Key
                          </Button>
                        </div>
                      ) : (
                        <ByokKeysTable
                          keys={byokState.orgKeys}
                          onDelete={handleDeleteOrgByok}
                          formatProviderName={formatProviderName}
                          deletingProvider={deletingOrgByokProvider}
                        />
                      )}
                    </CardContent>
                  </Card>
                </div>

                {/* BYOK Info */}
                <Card className="mt-6">
                  <CardContent className="pt-6">
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="p-4 rounded-lg bg-muted/50">
                        <h4 className="font-semibold mb-2">User-Level Keys</h4>
                        <p className="text-sm text-muted-foreground">
                          Individual users can configure their own API keys for personal use.
                          These keys are encrypted and stored securely. User keys take precedence
                          over organization and system keys.
                        </p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/50">
                        <h4 className="font-semibold mb-2">Organization-Level Keys</h4>
                        <p className="text-sm text-muted-foreground">
                          Organizations can set shared API keys for all members.
                          These take precedence over system defaults but can be overridden by user keys.
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

        <Dialog
          open={deprecatedModelDialog.isOpen}
          onOpenChange={(open) => {
            if (!open) {
              setDeprecatedModelDialog({
                isOpen: false,
                providerName: '',
                modelName: '',
                replacement: '',
                requestsLast7d: null,
                isLoading: false,
              });
            } else {
              setDeprecatedModelDialog((prev) => ({ ...prev, isOpen: true }));
            }
          }}
        >
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Deprecated Model Warning</DialogTitle>
              <DialogDescription>
                {deprecatedModelDialog.modelName
                  ? `${deprecatedModelDialog.modelName} (${formatProviderName(deprecatedModelDialog.providerName)})`
                  : 'Model deprecation details'}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-2 text-sm">
              {deprecatedModelDialog.isLoading ? (
                <p className="text-muted-foreground">Loading recent usage...</p>
              ) : (
                <p>
                  This model is deprecated. {deprecatedModelDialog.requestsLast7d ?? 0} requests used it in the last {USAGE_WINDOW_DAYS} days. Consider migrating to {deprecatedModelDialog.replacement}.
                </p>
              )}
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setDeprecatedModelDialog({
                  isOpen: false,
                  providerName: '',
                  modelName: '',
                  replacement: '',
                  requestsLast7d: null,
                  isLoading: false,
                })}
              >
                Close
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Provider Override Dialog */}
        <Dialog
          open={overrideDialog.isOpen}
          onOpenChange={(open) => updateOverrideDialog({ isOpen: open })}
        >
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>Provider Override</DialogTitle>
              <DialogDescription>
                {overrideDialog.provider
                  ? `Manage ${formatProviderName(overrideDialog.provider.name)} overrides`
                  : 'Manage provider overrides'}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label>Enable provider</Label>
                  <p className="text-xs text-muted-foreground">Disable to block usage across the system.</p>
                </div>
                <Checkbox
                  checked={overrideDialog.enabled}
                  onCheckedChange={(checked) => updateOverrideDialog({ enabled: Boolean(checked) })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="overrideDefaultModel">Default model</Label>
                <Input
                  id="overrideDefaultModel"
                  placeholder={overrideDialog.provider?.default_model || 'e.g. gpt-4o-mini'}
                  value={overrideDialog.defaultModel}
                  onChange={(e) => updateOverrideDialog({ defaultModel: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="overrideAllowedModels">Allowlisted models</Label>
                <Input
                  id="overrideAllowedModels"
                  placeholder="Comma-separated list (leave empty for all)"
                  value={overrideDialog.allowedModels}
                  onChange={(e) => updateOverrideDialog({ allowedModels: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="overrideBaseUrl">Base URL</Label>
                <Input
                  id="overrideBaseUrl"
                  placeholder="https://api.example.com"
                  value={overrideDialog.baseUrl}
                  onChange={(e) => updateOverrideDialog({ baseUrl: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="overrideApiKey">API key</Label>
                <Input
                  id="overrideApiKey"
                  type="password"
                  placeholder={overrideDialog.override?.api_key_hint ? `Stored (****${overrideDialog.override.api_key_hint})` : 'sk-...'}
                  value={overrideDialog.apiKey}
                  onChange={(e) => updateOverrideDialog({ apiKey: e.target.value })}
                />
                {overrideDialog.override?.api_key_hint && (
                  <p className="text-xs text-muted-foreground">
                    Stored key hint: ****{overrideDialog.override.api_key_hint}
                  </p>
                )}
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <Label>Clear stored API key</Label>
                  <p className="text-xs text-muted-foreground">Remove the stored key for this provider.</p>
                </div>
                <Checkbox
                  checked={overrideDialog.clearApiKey}
                  onCheckedChange={(checked) => updateOverrideDialog({ clearApiKey: Boolean(checked) })}
                  disabled={overrideDialog.apiKey.trim().length > 0}
                />
              </div>
            </div>
            <DialogFooter className="flex items-center justify-between sm:justify-between">
              <div className="flex gap-2">
                {overrideDialog.override && (
                  <Button
                    variant="destructive"
                    onClick={handleDeleteOverride}
                    disabled={overrideDialog.isDeleting || overrideDialog.isSaving}
                    loading={overrideDialog.isDeleting}
                    loadingText="Removing..."
                  >
                    Remove Override
                  </Button>
                )}
              </div>
              <div className="flex gap-2">
                <Button variant="outline" onClick={() => updateOverrideDialog({ isOpen: false })}>
                  Cancel
                </Button>
                <Button onClick={handleSaveOverride} disabled={overrideDialog.isSaving} loading={overrideDialog.isSaving} loadingText="Saving...">
                  Save
                </Button>
              </div>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Add BYOK Dialog */}
        <Dialog
          open={addByokDialog.isOpen}
          onOpenChange={(open) => {
            if (!open) {
              updateAddByokDialog({
                isOpen: false,
                provider: '',
                customProviderName: '',
                apiKey: '',
                isAdding: false,
              });
            } else {
              updateAddByokDialog({ isOpen: true });
            }
          }}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add BYOK Key</DialogTitle>
              <DialogDescription>
                {addByokDialog.mode === 'user' && byokState.selectedUser
                  ? `Add an API key for ${byokState.selectedUser.email}`
                  : addByokDialog.mode === 'org' && byokState.selectedOrg
                  ? `Add an API key for ${byokState.selectedOrg.name}`
                  : 'Add a new provider API key'}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="provider">Provider</Label>
                <select
                  id="provider"
                  value={addByokDialog.provider}
                  onChange={(e) => {
                    const value = e.target.value;
                    updateAddByokDialog({ provider: value });
                    if (value !== 'other') {
                      updateAddByokDialog({ customProviderName: '' });
                    }
                  }}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <option value="">Select a provider...</option>
                  {commonProviders.map((p) => (
                    <option key={p} value={p}>
                      {formatProviderName(p)}
                    </option>
                  ))}
                  <option value="other">Other (custom)</option>
                </select>
                {addByokDialog.provider === 'other' && (
                  <Input
                    placeholder="Enter provider name..."
                    value={addByokDialog.customProviderName}
                    onChange={(e) => updateAddByokDialog({ customProviderName: e.target.value })}
                    className="mt-2"
                  />
                )}
              </div>
              <div className="space-y-2">
                <Label htmlFor="apiKey">API Key</Label>
                <Input
                  id="apiKey"
                  type="password"
                  placeholder="sk-..."
                  value={addByokDialog.apiKey}
                  onChange={(e) => updateAddByokDialog({ apiKey: e.target.value })}
                />
                <p className="text-xs text-muted-foreground">
                  The key will be encrypted before storage
                </p>
              </div>
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => updateAddByokDialog({
                  isOpen: false,
                  provider: '',
                  customProviderName: '',
                  apiKey: '',
                  isAdding: false,
                })}
              >
                Cancel
              </Button>
              <Button
                onClick={handleAddByok}
                disabled={
                  addByokDialog.isAdding ||
                  !canSubmitByok
                }
                loading={addByokDialog.isAdding}
                loadingText="Adding..."
              >
                Add Key
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
