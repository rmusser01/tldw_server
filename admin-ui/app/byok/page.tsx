'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useOrgContext } from '@/components/OrgContextSwitcher';
import { useToast } from '@/components/ui/toast';
import { ApiError, api } from '@/lib/api-client';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { KeyRound, RefreshCw, Plus, Trash2, Send, Server, X } from 'lucide-react';
import { AccessibleIconButton } from '@/components/ui/accessible-icon-button';
import type { AuditLog, ByokValidationRunItem } from '@/types';

type SharedProviderKey = {
  scope_type: 'org' | 'team';
  scope_id: number;
  provider: string;
  key_hint?: string;
  last_used_at?: string;
};

type MetricSample = {
  name: string;
  labels: Record<string, string>;
  value: number;
};

type AdminByokUserKeyItem = {
  provider: string;
  key_hint?: string;
  last_used_at?: string;
  allowed?: boolean;
};

type LlmUsageLogItem = {
  user_id?: number | null;
  provider?: string | null;
  total_tokens?: number | null;
  total_cost_usd?: number | null;
};

type ByokUserUsageRow = {
  user_id: number;
  username: string;
  provider: string;
  key_hint?: string;
  last_used_at?: string;
  requests: number;
  total_tokens: number;
  total_cost_usd: number;
};

type ResolutionSummary = {
  source: string;
  count: number;
  share: string;
};

type OpenAIOAuthStatus = {
  provider: 'openai';
  connected: boolean;
  auth_source: 'api_key' | 'oauth' | 'none';
  updated_at?: string | null;
  last_used_at?: string | null;
  expires_at?: string | null;
  scope?: string | null;
};

const VALIDATION_POLL_INTERVAL_MS = 2_000;

const SOURCE_LABELS: Record<string, string> = {
  user: 'User',
  team: 'Team',
  org: 'Org',
  server_default: 'Server',
  none: 'None',
};

const unescapeLabelValue = (s: string): string => {
  let result = '';
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === '\\' && i + 1 < s.length) {
      const next = s[i + 1];
      if (next === '\\') {
        result += '\\';
        i++;
      } else if (next === '"') {
        result += '"';
        i++;
      } else if (next === 'n') {
        result += '\n';
        i++;
      } else {
        // Unknown escape, keep as-is
        result += ch + next;
        i++;
      }
    } else {
      result += ch;
    }
  }
  return result;
};

const parsePrometheusText = (text: string): MetricSample[] => {
  const samples: MetricSample[] = [];
  const lineRegex = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$/;
  const labelRegex = /(\w+)\s*=\s*"((?:\\.|[^"\\])*)"/g;
  text.split('\n').forEach((raw) => {
    const line = raw.trim();
    if (!line || line.startsWith('#')) return;
    const match = lineRegex.exec(line);
    if (!match) return;
    const [, name, , labelBlob, valueRaw] = match;
    const labels: Record<string, string> = {};
    if (labelBlob) {
      let labelMatch: RegExpExecArray | null;
      while ((labelMatch = labelRegex.exec(labelBlob))) {
        const key = labelMatch[1];
        const value = unescapeLabelValue(labelMatch[2]);
        labels[key] = value;
      }
    }
    const value = Number(valueRaw);
    if (!Number.isNaN(value)) {
      samples.push({ name, labels, value });
    }
  });
  return samples;
};

const sumValues = (values: number[]) => values.reduce((acc, val) => acc + val, 0);

const formatCount = (value: number | null) => {
  if (value === null) return '—';
  if (value < 1000) return `${value}`;
  if (value < 1000000) return `${(value / 1000).toFixed(1)}k`;
  return `${(value / 1000000).toFixed(1)}m`;
};

const formatUsd = (value: number) => new Intl.NumberFormat(undefined, {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 2,
}).format(value);

const sortValidationRuns = (runs: ByokValidationRunItem[]) =>
  [...runs].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

const isActiveValidationRun = (run: ByokValidationRunItem) =>
  run.status === 'queued' || run.status === 'running';

const formatValidationStatus = (status: ByokValidationRunItem['status']) =>
  status.charAt(0).toUpperCase() + status.slice(1);

const formatValidationCounts = (run: ByokValidationRunItem) => {
  if (run.keys_checked === null || run.keys_checked === undefined) {
    return 'Awaiting validation results.';
  }
  return `${formatCount(run.keys_checked)} checked • ${formatCount(run.valid_count ?? 0)} valid • ${formatCount(run.invalid_count ?? 0)} invalid • ${formatCount(run.error_count ?? 0)} errors`;
};

const PROVIDER_OPTIONS = [
  'openai',
  'anthropic',
  'cohere',
  'deepseek',
  'google',
  'groq',
  'huggingface',
  'mistral',
  'openrouter',
  'qwen',
  'moonshot',
  'zai',
] as const;

export default function ByokDashboardPage() {
  const { selectedOrg } = useOrgContext();
  const confirm = useConfirm();
  const promptPrivilegedAction = usePrivilegedActionDialog();
  const { success: toastSuccess, error: showError } = useToast();
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [resolutionBySource, setResolutionBySource] = useState<Record<string, number>>({});
  const [resolutionByProvider, setResolutionByProvider] = useState<Record<string, number>>({});
  const [missingByProvider, setMissingByProvider] = useState<Record<string, number>>({});
  const [auditEntries, setAuditEntries] = useState<AuditLog[]>([]);
  const [auditError, setAuditError] = useState<string | null>(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [byokUsageRows, setByokUsageRows] = useState<ByokUserUsageRow[]>([]);
  const [byokUsageLoading, setByokUsageLoading] = useState(false);
  const [byokUsageError, setByokUsageError] = useState<string | null>(null);
  const [validationRuns, setValidationRuns] = useState<ByokValidationRunItem[]>([]);
  const [validationLoading, setValidationLoading] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validationCreating, setValidationCreating] = useState(false);
  const [openAIOAuthStatus, setOpenAIOAuthStatus] = useState<OpenAIOAuthStatus | null>(null);
  const [openAIOAuthLoading, setOpenAIOAuthLoading] = useState(false);
  const [openAIOAuthError, setOpenAIOAuthError] = useState<string | null>(null);
  const [openAIOAuthAction, setOpenAIOAuthAction] = useState<string | null>(null);
  const validationToastRunIdRef = useRef<string | null>(null);

  // Shared Provider Keys
  const [sharedKeys, setSharedKeys] = useState<SharedProviderKey[]>([]);
  const [sharedKeysLoading, setSharedKeysLoading] = useState(false);
  const [sharedKeysError, setSharedKeysError] = useState<string | null>(null);
  const [showAddKey, setShowAddKey] = useState(false);
  const [newKeyProvider, setNewKeyProvider] = useState('openai');
  const [newKeyValue, setNewKeyValue] = useState('');
  const [addingKey, setAddingKey] = useState(false);
  const [testingKey, setTestingKey] = useState<string | null>(null);
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  const sharedKeysRequestIdRef = useRef(0);

  const resetAddKeyForm = useCallback(() => {
    setShowAddKey(false);
    setNewKeyValue('');
  }, []);

  const loadMetrics = useCallback(async () => {
    setMetricsLoading(true);
    setMetricsError(null);
    try {
      const raw = await api.getMetricsText();
      const samples = parsePrometheusText(raw);
      const resolutionSamples = samples.filter((sample) => sample.name === 'byok_resolution_total');
      const missingSamples = samples.filter((sample) => sample.name === 'byok_missing_credentials_total');

      const sourceTotals: Record<string, number> = {};
      const providerTotals: Record<string, number> = {};
      resolutionSamples.forEach((sample) => {
        const source = sample.labels.source || 'unknown';
        const provider = sample.labels.provider || 'unknown';
        sourceTotals[source] = (sourceTotals[source] || 0) + sample.value;
        providerTotals[provider] = (providerTotals[provider] || 0) + sample.value;
      });

      const missingProviderTotals: Record<string, number> = {};
      missingSamples.forEach((sample) => {
        const provider = sample.labels.provider || 'unknown';
        missingProviderTotals[provider] = (missingProviderTotals[provider] || 0) + sample.value;
      });

      setResolutionBySource(sourceTotals);
      setResolutionByProvider(providerTotals);
      setMissingByProvider(missingProviderTotals);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load metrics.';
      setMetricsError(message);
      showError('Metrics error', message);
    } finally {
      setMetricsLoading(false);
    }
  }, [showError]);

  const loadAudit = useCallback(async () => {
    setAuditLoading(true);
    setAuditError(null);
    try {
      const params: Record<string, string> = { limit: '200', offset: '0' };
      if (selectedOrg?.id) params.org_id = String(selectedOrg.id);
      const { entries } = await api.getAuditLogs(params);
      const byokEntries = (entries || []).filter((entry) => {
        const haystack = [
          entry.action,
          entry.resource,
          JSON.stringify(entry.details || {}),
        ]
          .join(' ')
          .toLowerCase();
        return (
          haystack.includes('byok')
          || haystack.includes('provider_secret')
          || haystack.includes('user_provider')
          || haystack.includes('org_provider')
          || haystack.includes('shared_key')
        );
      });
      setAuditEntries(byokEntries.slice(0, 12));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load audit activity.';
      setAuditError(message);
    } finally {
      setAuditLoading(false);
    }
  }, [selectedOrg?.id]);

  const loadByokUsage = useCallback(async () => {
    setByokUsageLoading(true);
    setByokUsageError(null);
    try {
      const users: Array<{ id: number; username: string }> = [];
      let page = 1;
      let pages = 1;
      do {
        const usersPage = await api.getUsersPage({
          page: String(page),
          limit: '100',
          ...(selectedOrg?.id ? { org_id: String(selectedOrg.id) } : {}),
        });
        users.push(...usersPage.items.map((user) => ({ id: user.id, username: user.username })));
        pages = usersPage.pages > 0
          ? usersPage.pages
          : Math.max(1, Math.ceil(usersPage.total / Math.max(usersPage.limit, 1)));
        if (usersPage.items.length === 0) {
          break;
        }
        page += 1;
      } while (page <= pages);

      const byokKeyResults = await Promise.allSettled(
        users.map(async (user) => ({
          userId: user.id,
          username: user.username,
          response: await api.getAdminUserByokKeys(String(user.id)) as {
            items?: AdminByokUserKeyItem[];
          },
        }))
      );

      const userProviderKeyMap = new Map<
      string,
      {
        user_id: number;
        username: string;
        provider: string;
        key_hint?: string;
        last_used_at?: string;
      }
      >();

      byokKeyResults.forEach((result) => {
        if (result.status !== 'fulfilled') {
          return;
        }
        const keyItems = Array.isArray(result.value.response.items)
          ? result.value.response.items
          : [];
        keyItems.forEach((item) => {
          const provider = (item.provider || '').trim().toLowerCase();
          if (!provider) return;
          const key = `${result.value.userId}:${provider}`;
          userProviderKeyMap.set(key, {
            user_id: result.value.userId,
            username: result.value.username,
            provider,
            key_hint: item.key_hint,
            last_used_at: item.last_used_at,
          });
        });
      });

      const usageItems: LlmUsageLogItem[] = [];
      let usagePage = 1;
      let usagePages = 1;
      do {
        const usageResponse = await api.getLlmUsage({
          page: String(usagePage),
          limit: '500',
          ...(selectedOrg?.id ? { org_id: String(selectedOrg.id) } : {}),
        }) as {
          items?: LlmUsageLogItem[];
          total?: number;
          page?: number;
          limit?: number;
        };
        const items = Array.isArray(usageResponse.items) ? usageResponse.items : [];
        usageItems.push(...items);

        const total = typeof usageResponse.total === 'number' ? usageResponse.total : items.length;
        const limit = typeof usageResponse.limit === 'number' ? usageResponse.limit : 500;
        usagePages = Math.min(10, Math.max(1, Math.ceil(total / Math.max(limit, 1))));
        if (items.length === 0) {
          break;
        }
        usagePage += 1;
      } while (usagePage <= usagePages);

      const usageByUserProvider = new Map<string, { requests: number; total_tokens: number; total_cost_usd: number }>();

      usageItems.forEach((item) => {
        if (!item.user_id || !item.provider) return;
        const key = `${item.user_id}:${item.provider.toLowerCase()}`;
        if (!userProviderKeyMap.has(key)) {
          return;
        }
        const current = usageByUserProvider.get(key) || { requests: 0, total_tokens: 0, total_cost_usd: 0 };
        current.requests += 1;
        current.total_tokens += Number(item.total_tokens || 0);
        current.total_cost_usd += Number(item.total_cost_usd || 0);
        usageByUserProvider.set(key, current);
      });

      const rows = [...userProviderKeyMap.entries()].map(([key, metadata]) => {
        const usage = usageByUserProvider.get(key) || { requests: 0, total_tokens: 0, total_cost_usd: 0 };
        return {
          ...metadata,
          requests: usage.requests,
          total_tokens: usage.total_tokens,
          total_cost_usd: usage.total_cost_usd,
        } satisfies ByokUserUsageRow;
      }).sort((a, b) => b.total_cost_usd - a.total_cost_usd);

      setByokUsageRows(rows);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load per-user BYOK usage.';
      setByokUsageError(message);
    } finally {
      setByokUsageLoading(false);
    }
  }, [selectedOrg?.id]);

  const loadValidationRuns = useCallback(async () => {
    setValidationLoading(true);
    setValidationError(null);
    try {
      const response = await api.getByokValidationRuns({ limit: 10, offset: 0 });
      setValidationRuns(sortValidationRuns(Array.isArray(response.items) ? response.items : []));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load BYOK validation history.';
      setValidationError(message);
    } finally {
      setValidationLoading(false);
    }
  }, []);

  const pollValidationRun = useCallback(async (runId: string) => {
    try {
      const run = await api.getByokValidationRun(runId);
      setValidationRuns((current) => sortValidationRuns([
        run,
        ...current.filter((item) => item.id !== run.id),
      ]));
      if (!isActiveValidationRun(run) && validationToastRunIdRef.current === runId) {
        validationToastRunIdRef.current = null;
        if (run.status === 'complete') {
          toastSuccess('Validation sweep complete', 'BYOK validation sweep finished successfully.');
        } else {
          showError('Validation sweep failed', run.error_message || 'BYOK validation sweep failed.');
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to refresh BYOK validation run.';
      setValidationError(message);
      if (validationToastRunIdRef.current === runId) {
        validationToastRunIdRef.current = null;
        showError('Validation sweep failed', message);
      }
    }
  }, [showError, toastSuccess]);

  const handleRunValidationSweep = useCallback(async () => {
    setValidationCreating(true);
    setValidationError(null);
    try {
      const run = await api.createByokValidationRun(
        selectedOrg?.id ? { org_id: selectedOrg.id } : {}
      );
      validationToastRunIdRef.current = run.id;
      setValidationRuns((current) => sortValidationRuns([
        run,
        ...current.filter((item) => item.id !== run.id),
      ]));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start BYOK validation sweep.';
      setValidationError(message);
      showError('Validation sweep failed', message);
    } finally {
      setValidationCreating(false);
    }
  }, [selectedOrg?.id, showError]);

  const loadSharedKeys = useCallback(async () => {
    const requestId = sharedKeysRequestIdRef.current + 1;
    sharedKeysRequestIdRef.current = requestId;
    setSharedKeysLoading(true);
    setSharedKeysError(null);
    setSharedKeys([]);
    try {
      const orgId = selectedOrg?.id;
      // Filter by selected org if one is selected
      const params = orgId
        ? { scope_type: 'org', scope_id: orgId }
        : undefined;
      const data = await api.getSharedProviderKeys(params);
      if (sharedKeysRequestIdRef.current !== requestId) return;
      const result = data as { keys?: SharedProviderKey[]; items?: SharedProviderKey[] };
      setSharedKeys(
        Array.isArray(result.keys) ? result.keys :
        Array.isArray(result.items) ? result.items :
        Array.isArray(result) ? result as SharedProviderKey[] : []
      );
    } catch (err) {
      if (sharedKeysRequestIdRef.current !== requestId) return;
      const message = err instanceof Error ? err.message : 'Failed to load shared keys.';
      setSharedKeysError(message);
    } finally {
      if (sharedKeysRequestIdRef.current === requestId) {
        setSharedKeysLoading(false);
      }
    }
  }, [selectedOrg?.id]);

  const loadOpenAIOAuthStatus = useCallback(async () => {
    setOpenAIOAuthLoading(true);
    setOpenAIOAuthError(null);
    try {
      const status = await api.getOpenAIOAuthStatus() as OpenAIOAuthStatus;
      setOpenAIOAuthStatus(status);
    } catch (err) {
      if (err instanceof ApiError && [403, 501].includes(err.status)) {
        setOpenAIOAuthStatus(null);
        setOpenAIOAuthError('OpenAI OAuth is not enabled in this deployment.');
        return;
      }
      const message = err instanceof Error ? err.message : 'Failed to load OpenAI OAuth status.';
      setOpenAIOAuthError(message);
    } finally {
      setOpenAIOAuthLoading(false);
    }
  }, []);

  const handleConnectOpenAIOAuth = async () => {
    try {
      setOpenAIOAuthAction('connect');
      const returnPath = typeof window !== 'undefined' && window.location.pathname
        ? window.location.pathname
        : '/byok';
      const response = await api.startOpenAIOAuth({ return_path: returnPath }) as {
        auth_url?: string;
      };
      const authUrl = typeof response?.auth_url === 'string' ? response.auth_url.trim() : '';
      if (!authUrl) {
        throw new Error('OAuth authorize response is missing auth_url.');
      }
      if (typeof window !== 'undefined') {
        const popup = window.open(authUrl, '_blank', 'noopener,noreferrer');
        if (!popup) {
          window.location.assign(authUrl);
          return;
        }
      }
      toastSuccess(
        'OAuth started',
        'Finish authorization in the opened tab, then refresh status.'
      );
      await loadOpenAIOAuthStatus();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start OpenAI OAuth flow.';
      showError('OAuth connect failed', message);
    } finally {
      setOpenAIOAuthAction(null);
    }
  };

  const handleRefreshOpenAIOAuth = async () => {
    try {
      setOpenAIOAuthAction('refresh');
      await api.refreshOpenAIOAuth();
      toastSuccess('OAuth refreshed', 'OpenAI OAuth token was refreshed.');
      await loadOpenAIOAuthStatus();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to refresh OpenAI OAuth token.';
      showError('OAuth refresh failed', message);
    } finally {
      setOpenAIOAuthAction(null);
    }
  };

  const handleSwitchOpenAICredentialSource = async (authSource: 'api_key' | 'oauth') => {
    try {
      setOpenAIOAuthAction(`switch-${authSource}`);
      await api.switchOpenAICredentialSource(authSource);
      toastSuccess(
        'Credential source updated',
        authSource === 'oauth'
          ? 'OpenAI now uses OAuth credentials.'
          : 'OpenAI now uses your API key.'
      );
      await loadOpenAIOAuthStatus();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to switch OpenAI credential source.';
      showError('Switch failed', message);
    } finally {
      setOpenAIOAuthAction(null);
    }
  };

  const handleDisconnectOpenAIOAuth = async () => {
    const result = await promptPrivilegedAction({
      title: 'Disconnect OpenAI OAuth',
      message: 'Disconnect OAuth credentials for OpenAI?',
      confirmText: 'Disconnect',
      requirePassword: false,
    });
    if (!result) return;

    try {
      setOpenAIOAuthAction('disconnect');
      await api.disconnectOpenAIOAuth();
      toastSuccess('OAuth disconnected', 'OpenAI OAuth credentials were removed.');
      await loadOpenAIOAuthStatus();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to disconnect OpenAI OAuth.';
      showError('Disconnect failed', message);
    } finally {
      setOpenAIOAuthAction(null);
    }
  };

  const handleAddSharedKey = async () => {
    if (!newKeyValue.trim()) {
      showError('API key required', 'Please enter the API key value.');
      return;
    }

    if (!selectedOrg?.id) {
      showError('Organization required', 'Please select an organization to add a shared key.');
      return;
    }

    try {
      setAddingKey(true);
      await api.createSharedProviderKey({
        scope_type: 'org',
        scope_id: selectedOrg.id,
        provider: newKeyProvider,
        api_key: newKeyValue.trim(),
      });
      toastSuccess('Key added', `Shared key for ${newKeyProvider} has been added.`);
      setShowAddKey(false);
      setNewKeyProvider('openai');
      setNewKeyValue('');
      void loadSharedKeys();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to add shared key.';
      showError('Add failed', message);
    } finally {
      setAddingKey(false);
    }
  };

  const handleDeleteSharedKey = async (key: SharedProviderKey) => {
    const keyId = `${key.scope_type}:${key.scope_id}:${key.provider}`;
    const result = await promptPrivilegedAction({
      title: 'Delete Shared Key',
      message: `Delete the shared key for "${key.provider}"? Users and orgs will fall back to their own keys.`,
      confirmText: 'Delete',
      requirePassword: true,
    });
    if (!result) return;

    try {
      setDeletingKey(keyId);
      await api.deleteSharedProviderKey(key.scope_type, key.scope_id, key.provider);
      toastSuccess('Key deleted', `Shared key for ${key.provider} has been removed.`);
      void loadSharedKeys();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete shared key.';
      showError('Delete failed', message);
    } finally {
      setDeletingKey(null);
    }
  };

  const handleTestSharedKey = async (key: SharedProviderKey) => {
    const keyId = `${key.scope_type}:${key.scope_id}:${key.provider}`;
    try {
      setTestingKey(keyId);
      await api.testSharedProviderKey({
        scope_type: key.scope_type,
        scope_id: key.scope_id,
        provider: key.provider,
      });
      toastSuccess('Test passed', `Shared key for ${key.provider} is valid.`);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Key test failed.';
      showError('Test failed', message);
    } finally {
      setTestingKey(null);
    }
  };

  useEffect(() => {
    loadMetrics();
    loadAudit();
    loadSharedKeys();
    loadByokUsage();
    loadValidationRuns();
    loadOpenAIOAuthStatus();
  }, [loadMetrics, loadAudit, loadSharedKeys, loadByokUsage, loadValidationRuns, loadOpenAIOAuthStatus]);

  const summaryCards = useMemo(() => {
    const byokTotal = sumValues(
      Object.entries(resolutionBySource)
        .filter(([source]) => ['user', 'team', 'org'].includes(source))
        .map(([, value]) => value)
    );
    const missingTotal = sumValues(Object.values(missingByProvider));
    const providersWithByok = Object.keys(resolutionByProvider).length || null;
    const topProvider = Object.entries(resolutionByProvider)
      .sort((a, b) => b[1] - a[1])[0]?.[0];

    return [
      { title: 'BYOK Requests', value: formatCount(byokTotal), detail: 'Resolved via user/team/org keys' },
      { title: 'Missing Credentials', value: formatCount(missingTotal), detail: 'Missing key events' },
      { title: 'Providers Used', value: providersWithByok ? String(providersWithByok) : '—', detail: 'Providers with BYOK traffic' },
      { title: 'Top Provider', value: topProvider || '—', detail: 'Highest BYOK volume' },
    ];
  }, [resolutionByProvider, resolutionBySource, missingByProvider]);

  const resolutionMix: ResolutionSummary[] = useMemo(() => {
    const total = sumValues(Object.values(resolutionBySource));
    const sources = Object.keys(resolutionBySource);
    const fallbackSources = ['user', 'team', 'org', 'server_default'];
    const list = (sources.length ? sources : fallbackSources).map((source) => {
      const count = resolutionBySource[source] || 0;
      const share = total > 0 ? `${Math.round((count / total) * 100)}%` : '—';
      return {
        source,
        count,
        share,
      };
    });
    return list;
  }, [resolutionBySource]);

  const missingTopProviders = useMemo(() => {
    return Object.entries(missingByProvider)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
  }, [missingByProvider]);

  const latestValidationRun = useMemo(
    () => validationRuns[0] ?? null,
    [validationRuns]
  );

  const activeValidationRun = useMemo(
    () => validationRuns.find((run) => isActiveValidationRun(run)) ?? null,
    [validationRuns]
  );

  const openAIOAuthStatusLabel = useMemo(() => {
    if (!openAIOAuthStatus) return 'Not connected';
    if (openAIOAuthStatus.auth_source === 'oauth') return 'Connected (OAuth)';
    if (openAIOAuthStatus.auth_source === 'api_key') return 'API Key';
    if (openAIOAuthStatus.connected) return 'Connected';
    return 'Not connected';
  }, [openAIOAuthStatus]);

  useEffect(() => {
    if (!activeValidationRun) {
      return undefined;
    }
    const timer = window.setTimeout(() => {
      void pollValidationRun(activeValidationRun.id);
    }, VALIDATION_POLL_INTERVAL_MS);
    return () => {
      window.clearTimeout(timer);
    };
  }, [activeValidationRun, pollValidationRun]);

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="space-y-6 p-4 lg:p-6">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <KeyRound className="h-5 w-5 text-primary" />
                <h1 className="text-2xl font-semibold">BYOK Dashboards</h1>
              </div>
              <p className="text-sm text-muted-foreground">
                Operational dashboards for BYOK adoption, validation sweeps, and recent admin key activity.
              </p>
              {selectedOrg && (
                <div className="mt-1 text-xs text-muted-foreground">
                  Active scope: {selectedOrg.name}
                </div>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">Metrics</Badge>
              <Button variant="outline" size="sm" onClick={loadMetrics} disabled={metricsLoading}>
                <RefreshCw className="mr-2 h-4 w-4" />
                {metricsLoading ? 'Refreshing…' : 'Refresh metrics'}
              </Button>
            </div>
          </div>

          <Alert>
            <AlertDescription>
              {metricsError
                ? `Metrics unavailable: ${metricsError}`
                : 'Metrics are sourced from /api/v1/metrics/text and audit logs where available.'}
            </AlertDescription>
          </Alert>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {summaryCards.map((card) => (
              <Card key={card.title}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base">{card.title}</CardTitle>
                  <CardDescription>{card.detail}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-semibold">{card.value}</div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Per-User BYOK Usage</CardTitle>
              <CardDescription>
                Aggregated from recent LLM usage logs and cross-referenced with configured user BYOK providers.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {byokUsageError && (
                <Alert variant="destructive" className="mb-3">
                  <AlertDescription>{byokUsageError}</AlertDescription>
                </Alert>
              )}
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>User</TableHead>
                    <TableHead>Provider</TableHead>
                    <TableHead>Key Hint</TableHead>
                    <TableHead>Requests</TableHead>
                    <TableHead>Tokens</TableHead>
                    <TableHead>Cost (USD)</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {byokUsageLoading && (
                    <TableRow>
                      <TableCell colSpan={6} className="text-muted-foreground">
                        Loading per-user BYOK usage...
                      </TableCell>
                    </TableRow>
                  )}
                  {!byokUsageLoading && byokUsageRows.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={6} className="text-muted-foreground">
                        No user BYOK usage data found.
                      </TableCell>
                    </TableRow>
                  )}
                  {byokUsageRows.map((row) => (
                    <TableRow key={`${row.user_id}:${row.provider}`}>
                      <TableCell>{row.username}</TableCell>
                      <TableCell className="capitalize">{row.provider}</TableCell>
                      <TableCell className="text-muted-foreground">{row.key_hint || '—'}</TableCell>
                      <TableCell>{formatCount(row.requests)}</TableCell>
                      <TableCell>{formatCount(row.total_tokens)}</TableCell>
                      <TableCell>{formatUsd(row.total_cost_usd)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <div className="grid gap-4 lg:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Resolution Mix</CardTitle>
                <CardDescription>Share of requests by credential source.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {resolutionMix.map((row) => (
                  <div key={row.source} className="flex items-center justify-between rounded-md border px-3 py-2 text-sm">
                    <div>
                      <div className="font-medium">{SOURCE_LABELS[row.source] || row.source}</div>
                      <div className="text-xs text-muted-foreground">{formatCount(row.count)} requests</div>
                    </div>
                    <div className="text-xs font-semibold text-muted-foreground">{row.share}</div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Missing Credentials</CardTitle>
                <CardDescription>Top providers with missing key events.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                {missingTopProviders.length === 0 && (
                  <div className="rounded-md border px-3 py-2 text-sm text-muted-foreground">
                    No missing credential events yet.
                  </div>
                )}
                {missingTopProviders.map(([provider, count]) => (
                  <div key={provider} className="flex items-center justify-between rounded-md border px-3 py-2">
                    <span>{provider}</span>
                    <span>{formatCount(count)}</span>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <CardTitle className="text-base">Key Validation</CardTitle>
                    <CardDescription>Run an authoritative validation sweep and review recent validation history.</CardDescription>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={loadValidationRuns}
                      disabled={validationLoading || validationCreating}
                    >
                      <RefreshCw className="mr-2 h-4 w-4" />
                      {validationLoading ? 'Refreshing…' : 'Refresh history'}
                    </Button>
                    <Button
                      size="sm"
                      onClick={handleRunValidationSweep}
                      disabled={validationLoading || validationCreating}
                    >
                      <Send className="mr-2 h-4 w-4" />
                      {validationCreating ? 'Starting…' : 'Run validation sweep'}
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                {validationError && (
                  <Alert variant="destructive">
                    <AlertDescription>{validationError}</AlertDescription>
                  </Alert>
                )}
                {latestValidationRun ? (
                  <div className="rounded-md border px-3 py-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="font-medium text-foreground">{latestValidationRun.scope_summary}</div>
                      <Badge variant={latestValidationRun.status === 'complete' ? 'default' : 'outline'}>
                        {formatValidationStatus(latestValidationRun.status)}
                      </Badge>
                    </div>
                    {latestValidationRun.status === 'complete' && latestValidationRun.keys_checked != null ? (
                      <div className="mt-2 flex flex-wrap gap-2">
                        <Badge variant="secondary">{latestValidationRun.keys_checked} checked</Badge>
                        <Badge className="bg-green-600 text-white">{latestValidationRun.valid_count ?? 0} valid</Badge>
                        {(latestValidationRun.invalid_count ?? 0) > 0 && (
                          <Badge variant="destructive">{latestValidationRun.invalid_count} invalid</Badge>
                        )}
                        {(latestValidationRun.error_count ?? 0) > 0 && (
                          <Badge className="bg-yellow-500 text-black">{latestValidationRun.error_count} errors</Badge>
                        )}
                        {(latestValidationRun.keys_checked ?? 0) > 0 && (latestValidationRun.invalid_count ?? 0) === 0 && (latestValidationRun.error_count ?? 0) === 0 && (
                          <span className="text-xs text-green-600 font-medium">All keys passed validation</span>
                        )}
                      </div>
                    ) : (
                      <div className="mt-1 text-xs text-muted-foreground">
                        {formatValidationCounts(latestValidationRun)}
                      </div>
                    )}
                    <div className="mt-1 text-xs text-muted-foreground">
                      Requested by {latestValidationRun.requested_by_label || 'Unknown'} • {new Date(latestValidationRun.created_at).toLocaleString()}
                    </div>
                    {latestValidationRun.error_message && (
                      <div className="mt-1 text-xs text-red-600">{latestValidationRun.error_message}</div>
                    )}
                  </div>
                ) : (
                  <div className="rounded-md border px-3 py-2">No validation runs yet.</div>
                )}
                {validationRuns.slice(1, 4).map((run) => (
                  <div key={run.id} className="rounded-md border px-3 py-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="font-medium text-foreground">{run.scope_summary}</span>
                      <Badge variant="outline">{formatValidationStatus(run.status)}</Badge>
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">{formatValidationCounts(run)}</div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <CardTitle className="text-base">OpenAI OAuth (Personal)</CardTitle>
                  <CardDescription>
                    Connect your own OpenAI subscription for BYOK usage and switch between OAuth/API key sources.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={openAIOAuthStatus?.auth_source === 'oauth' ? 'default' : 'outline'}>
                    {openAIOAuthStatusLabel}
                  </Badge>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={loadOpenAIOAuthStatus}
                    disabled={openAIOAuthLoading || openAIOAuthAction !== null}
                  >
                    <RefreshCw className="mr-2 h-4 w-4" />
                    {openAIOAuthLoading ? 'Refreshing…' : 'Refresh status'}
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {openAIOAuthError && (
                <Alert variant="destructive">
                  <AlertDescription>{openAIOAuthError}</AlertDescription>
                </Alert>
              )}

              <div className="grid gap-2 rounded-md border px-3 py-2 text-sm text-muted-foreground sm:grid-cols-2">
                <div>
                  <div className="font-medium text-foreground">Source</div>
                  <div>{openAIOAuthStatus?.auth_source || 'none'}</div>
                </div>
                <div>
                  <div className="font-medium text-foreground">Expires</div>
                  <div>{openAIOAuthStatus?.expires_at ? new Date(openAIOAuthStatus.expires_at).toLocaleString() : '—'}</div>
                </div>
                <div>
                  <div className="font-medium text-foreground">Scope</div>
                  <div>{openAIOAuthStatus?.scope || '—'}</div>
                </div>
                <div>
                  <div className="font-medium text-foreground">Last used</div>
                  <div>{openAIOAuthStatus?.last_used_at ? new Date(openAIOAuthStatus.last_used_at).toLocaleString() : '—'}</div>
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                <Button
                  onClick={handleConnectOpenAIOAuth}
                  disabled={openAIOAuthAction !== null || !!openAIOAuthError}
                  loading={openAIOAuthAction === 'connect'}
                  loadingText="Starting..."
                >
                  Connect OpenAI
                </Button>
                <Button
                  variant="outline"
                  onClick={handleRefreshOpenAIOAuth}
                  disabled={openAIOAuthAction !== null || !openAIOAuthStatus?.connected}
                >
                  Refresh OAuth
                </Button>
                <Button
                  variant="outline"
                  onClick={() => handleSwitchOpenAICredentialSource('oauth')}
                  disabled={
                    openAIOAuthAction !== null
                    || openAIOAuthStatus?.auth_source === 'oauth'
                    || !openAIOAuthStatus?.connected
                  }
                >
                  Use OAuth
                </Button>
                <Button
                  variant="outline"
                  onClick={() => handleSwitchOpenAICredentialSource('api_key')}
                  disabled={
                    openAIOAuthAction !== null
                    || openAIOAuthStatus?.auth_source !== 'oauth'
                  }
                >
                  Use API Key Instead
                </Button>
                <Button
                  variant="destructive"
                  onClick={handleDisconnectOpenAIOAuth}
                  disabled={openAIOAuthAction !== null || !openAIOAuthStatus?.connected}
                >
                  Disconnect
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                OAuth opens in a new tab. After completing the provider flow, refresh status here.
              </p>
            </CardContent>
          </Card>

          {/* Shared Provider Keys (Organization-Level) */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Server className="h-5 w-5" />
                    Organization Provider Keys
                  </CardTitle>
                  <CardDescription>
                    Shared API keys for the selected organization. Users in the org can use these when they don&apos;t have their own keys.
                    <br />
                    <span className="text-xs">Resolution order: User → Organization</span>
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowAddKey(true)}
                  disabled={showAddKey || !selectedOrg?.id}
                  title={!selectedOrg?.id ? 'Select an organization first' : undefined}
                >
                  <Plus className="mr-2 h-4 w-4" />
                  Add Key
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {sharedKeysError && (
                <Alert variant="destructive">
                  <AlertDescription>{sharedKeysError}</AlertDescription>
                </Alert>
              )}

              {/* Add Key Form */}
              {showAddKey && (
                <div className="p-4 border rounded-lg space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">Add Organization Provider Key</span>
                    <AccessibleIconButton
                      icon={X}
                      label="Close form"
                      variant="ghost"
                      onClick={resetAddKeyForm}
                    />
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-1">
                      <Label htmlFor="key-provider">Provider</Label>
                      <Select
                        id="key-provider"
                        value={newKeyProvider}
                        onChange={(e) => setNewKeyProvider(e.target.value)}
                      >
                        {PROVIDER_OPTIONS.map((provider) => (
                          <option key={provider} value={provider}>
                            {provider.charAt(0).toUpperCase() + provider.slice(1)}
                          </option>
                        ))}
                      </Select>
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="key-value">API Key</Label>
                      <Input
                        id="key-value"
                        type="password"
                        placeholder="sk-..."
                        value={newKeyValue}
                        onChange={(e) => setNewKeyValue(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      onClick={handleAddSharedKey}
                      disabled={addingKey || !newKeyValue.trim()}
                      loading={addingKey}
                      loadingText="Adding..."
                    >
                      Add Key
                    </Button>
                    <Button variant="outline" onClick={resetAddKeyForm}>
                      Cancel
                    </Button>
                  </div>
                </div>
              )}

              {/* Keys List */}
              {sharedKeysLoading ? (
                <div className="text-center text-muted-foreground py-4">Loading organization keys...</div>
              ) : sharedKeys.length === 0 ? (
                <div className="text-center text-muted-foreground py-8 border rounded-lg">
                  No organization-level provider keys configured.
                  <br />
                  <span className="text-xs">Users must provide their own keys or select an organization.</span>
                </div>
              ) : (
                <div className="space-y-2">
                  {sharedKeys.map((key) => {
                    const keyId = `${key.scope_type}:${key.scope_id}:${key.provider}`;
                    return (
                    <div
                      key={keyId}
                      className="flex items-center justify-between p-3 rounded-lg border bg-background"
                    >
                      <div className="flex items-center gap-3">
                        <KeyRound className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <div className="font-medium capitalize">{key.provider}</div>
                          <div className="text-xs text-muted-foreground">
                            {key.key_hint ? `Key: ${key.key_hint}` : `${key.scope_type} key`}
                            {key.last_used_at && ` • Last used: ${new Date(key.last_used_at).toLocaleDateString()}`}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="capitalize">{key.scope_type}</Badge>
                        <AccessibleIconButton
                          icon={Send}
                          label="Test key"
                          variant="ghost"
                          size="sm"
                          onClick={() => handleTestSharedKey(key)}
                          disabled={testingKey === keyId}
                          iconClassName={testingKey === keyId ? 'animate-pulse' : ''}
                        />
                        <AccessibleIconButton
                          icon={Trash2}
                          label="Delete key"
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteSharedKey(key)}
                          disabled={deletingKey === keyId}
                          iconClassName="text-red-500"
                        />
                      </div>
                    </div>
                  );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">BYOK Audit Activity</CardTitle>
              <CardDescription>Recent BYOK-related audit events captured by the backend.</CardDescription>
            </CardHeader>
            <CardContent>
              {auditError && (
                <div className="mb-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {auditError}
                </div>
              )}
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>When</TableHead>
                    <TableHead>Actor</TableHead>
                    <TableHead>Action</TableHead>
                    <TableHead>Scope</TableHead>
                    <TableHead>Provider</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {auditLoading && (
                    <TableRow>
                      <TableCell colSpan={5} className="text-muted-foreground">
                        Loading audit activity…
                      </TableCell>
                    </TableRow>
                  )}
                  {!auditLoading && auditEntries.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={5} className="text-muted-foreground">
                        No BYOK audit events found yet.
                      </TableCell>
                    </TableRow>
                  )}
                  {auditEntries.map((entry) => {
                    const details = entry.details || {};
                    const provider = typeof details.provider === 'string' ? details.provider : '—';
                    const scope =
                      typeof details.scope_type === 'string'
                        ? details.scope_type
                        : typeof details.scope === 'string'
                          ? details.scope
                          : '—';
                    return (
                      <TableRow key={entry.id}>
                        <TableCell className="text-muted-foreground">{entry.timestamp || '—'}</TableCell>
                        <TableCell>{entry.username || entry.user_id || '—'}</TableCell>
                        <TableCell>{entry.action || '—'}</TableCell>
                        <TableCell>{scope}</TableCell>
                        <TableCell>{provider}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
