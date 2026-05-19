'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/ui/empty-state';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { useToast } from '@/components/ui/toast';
import { api } from '@/lib/api-client';
import { formatDateTime } from '@/lib/format';
import { RefreshCw, Trash2 } from 'lucide-react';

type MaintenanceState = {
  enabled: boolean;
  message?: string | null;
  allowlist_user_ids?: number[];
  allowlist_emails?: string[];
  updated_at?: string | null;
  updated_by?: string | null;
};

type FeatureFlagItem = {
  id?: number | string | null;
  key: string;
  scope: 'global' | 'org' | 'user';
  enabled: boolean;
  description?: string | null;
  org_id?: number | null;
  user_id?: number | null;
  target_user_ids?: number[];
  rollout_percent?: number;
  variant_value?: string | null;
  updated_at?: string | null;
  updated_by?: string | null;
  history?: {
    timestamp: string;
    enabled: boolean;
    actor?: string | null;
    note?: string | null;
    before?: {
      scope?: 'global' | 'org' | 'user';
      enabled?: boolean;
      org_id?: number | null;
      user_id?: number | null;
      target_user_ids?: number[];
      rollout_percent?: number;
      variant_value?: string | null;
    } | null;
    after?: {
      scope?: 'global' | 'org' | 'user';
      enabled?: boolean;
      org_id?: number | null;
      user_id?: number | null;
      target_user_ids?: number[];
      rollout_percent?: number;
      variant_value?: string | null;
    } | null;
  }[];
};

type FlagsResponse = {
  items?: FeatureFlagItem[];
};

const FLAG_SCOPES = ['global', 'org', 'user'] as const;

type ParsedList<T> = {
  values: T[];
  invalid: string[];
};

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const formatInvalidValues = (values: string[]) => {
  const sample = values.slice(0, 5).join(', ');
  return values.length > 5 ? `${sample} (+${values.length - 5} more)` : sample;
};

const parsePositiveInt = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  if (!Number.isInteger(parsed) || parsed <= 0) return null;
  return parsed;
};

const parseRolloutPercent = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  if (!Number.isInteger(parsed) || parsed < 0 || parsed > 100) return null;
  return parsed;
};

const parsePositiveIntList = (value: string): ParsedList<number> => {
  const values: number[] = [];
  const invalid: string[] = [];

  value.split(',').forEach((item) => {
    const trimmed = item.trim();
    if (!trimmed) return;
    const parsed = Number(trimmed);
    if (Number.isInteger(parsed) && parsed > 0) {
      values.push(parsed);
    } else {
      invalid.push(trimmed);
    }
  });

  return { values, invalid };
};

const parseEmailList = (value: string): ParsedList<string> => {
  const values: string[] = [];
  const invalid: string[] = [];

  value.split(',').forEach((item) => {
    const trimmed = item.trim();
    if (!trimmed) return;
    if (EMAIL_REGEX.test(trimmed)) {
      values.push(trimmed);
    } else {
      invalid.push(trimmed);
    }
  });

  return { values, invalid };
};

const formatFlagDate = (value?: string | null) =>
  formatDateTime(value, { fallback: '—' });

const formatVariant = (value?: string | null) => {
  const trimmed = value?.trim();
  return trimmed ? trimmed : '—';
};

const formatTargetUsers = (values?: number[]) => {
  if (!Array.isArray(values) || values.length === 0) {
    return 'All users in scope';
  }
  return values.join(', ');
};

const getFlagTargetLabel = (flag: FeatureFlagItem) => {
  if (flag.scope === 'global') return 'Global';
  if (flag.scope === 'org') return `Org ${flag.org_id ?? '—'}`;
  return `User ${flag.user_id ?? '—'}`;
};

const getDiffRows = (
  historyEntry: NonNullable<FeatureFlagItem['history']>[number]
): Array<{ field: string; before: string; after: string }> => {
  const before = historyEntry.before;
  const after = historyEntry.after;
  if (!before || !after) return [];

  const rows: Array<{ field: string; before: string; after: string }> = [];
  const pushDiff = (field: string, beforeValue: string, afterValue: string) => {
    if (beforeValue === afterValue) return;
    rows.push({ field, before: beforeValue, after: afterValue });
  };

  pushDiff('Scope', before.scope ?? '—', after.scope ?? '—');
  pushDiff(
    'Target users',
    formatTargetUsers(before.target_user_ids),
    formatTargetUsers(after.target_user_ids)
  );
  pushDiff(
    'Rollout %',
    `${before.rollout_percent ?? 100}%`,
    `${after.rollout_percent ?? 100}%`
  );
  pushDiff('Variant', formatVariant(before.variant_value), formatVariant(after.variant_value));
  pushDiff('Enabled', before.enabled ? 'true' : 'false', after.enabled ? 'true' : 'false');

  return rows;
};

const getFlagId = (flag: FeatureFlagItem) =>
  `${flag.key}:${flag.scope}:${flag.org_id ?? ''}:${flag.user_id ?? ''}`;

export default function FlagsPage() {
  const promptPrivilegedAction = usePrivilegedActionDialog();
  const { success, error: showError, warning } = useToast();

  const [maintenance, setMaintenance] = useState<MaintenanceState | null>(null);
  const [maintenanceEnabled, setMaintenanceEnabled] = useState(false);
  const [maintenanceMessage, setMaintenanceMessage] = useState('');
  const [maintenanceAllowUserIds, setMaintenanceAllowUserIds] = useState('');
  const [maintenanceAllowEmails, setMaintenanceAllowEmails] = useState('');
  const [maintenanceLoading, setMaintenanceLoading] = useState(true);
  const [maintenanceSaving, setMaintenanceSaving] = useState(false);

  const [flags, setFlags] = useState<FeatureFlagItem[]>([]);
  const [flagError, setFlagError] = useState('');
  const [flagLoading, setFlagLoading] = useState(true);
  const [flagScopeFilter, setFlagScopeFilter] = useState('');
  const [flagOrgFilter, setFlagOrgFilter] = useState('');
  const [flagUserFilter, setFlagUserFilter] = useState('');

  const [flagKey, setFlagKey] = useState('');
  const [flagScope, setFlagScope] = useState<'global' | 'org' | 'user'>('global');
  const [flagEnabled, setFlagEnabled] = useState(true);
  const [flagDescription, setFlagDescription] = useState('');
  const [flagOrgId, setFlagOrgId] = useState('');
  const [flagUserId, setFlagUserId] = useState('');
  const [flagTargetUsers, setFlagTargetUsers] = useState('');
  const [flagRolloutPercent, setFlagRolloutPercent] = useState('100');
  const [flagVariantValue, setFlagVariantValue] = useState('');
  const [flagNote, setFlagNote] = useState('');
  const [flagSaving, setFlagSaving] = useState(false);
  const [deletingFlagId, setDeletingFlagId] = useState<string | null>(null);
  const [togglingFlagId, setTogglingFlagId] = useState<string | null>(null);
  const [togglingFlags, setTogglingFlags] = useState<Record<string, boolean>>({});
  const togglingFlagsRef = useRef<Set<string>>(new Set());

  const flagParams = useMemo(() => {
    const params: Record<string, string> = {};
    if (flagScopeFilter) params.scope = flagScopeFilter;
    if (flagOrgFilter) params.org_id = flagOrgFilter;
    if (flagUserFilter) params.user_id = flagUserFilter;
    return params;
  }, [flagOrgFilter, flagScopeFilter, flagUserFilter]);

  const loadMaintenance = useCallback(async (signal?: AbortSignal) => {
    try {
      setMaintenanceLoading(true);
      const data = (await api.getMaintenanceMode({ signal })) as MaintenanceState;
      setMaintenance(data);
      setMaintenanceEnabled(Boolean(data?.enabled));
      setMaintenanceMessage(data?.message || '');
      setMaintenanceAllowUserIds((data?.allowlist_user_ids || []).join(', '));
      setMaintenanceAllowEmails((data?.allowlist_emails || []).join(', '));
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const message = err instanceof Error && err.message ? err.message : 'Failed to load maintenance mode';
      showError(message);
    } finally {
      setMaintenanceLoading(false);
    }
  }, [showError]);

  const loadFlags = useCallback(async (signal?: AbortSignal) => {
    try {
      setFlagLoading(true);
      setFlagError('');
      const data = (await api.getFeatureFlags(flagParams, { signal })) as FlagsResponse;
      const items = Array.isArray(data.items) ? data.items : [];
      setFlags(items);
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const message = err instanceof Error && err.message ? err.message : 'Failed to load feature flags';
      setFlagError(message);
      setFlags([]);
    } finally {
      setFlagLoading(false);
    }
  }, [flagParams]);

  useEffect(() => {
    const controller = new AbortController();
    void loadMaintenance(controller.signal);
    return () => controller.abort();
  }, [loadMaintenance]);

  useEffect(() => {
    const controller = new AbortController();
    void loadFlags(controller.signal);
    return () => controller.abort();
  }, [loadFlags]);

  const handleRefresh = useCallback(() => {
    void Promise.allSettled([loadMaintenance(), loadFlags()]);
  }, [loadFlags, loadMaintenance]);

  const handleSaveMaintenance = async () => {
    if (!maintenance) return;
    const changed = maintenanceEnabled !== maintenance.enabled;
    if (changed) {
      const result = await promptPrivilegedAction({
        title: maintenanceEnabled ? 'Enable maintenance mode?' : 'Disable maintenance mode?',
        message: maintenanceEnabled
          ? 'This will block non-allowlisted users.'
          : 'Service traffic will resume for all users.',
        confirmText: maintenanceEnabled ? 'Enable' : 'Disable',
        requirePassword: false,
      });
      if (!result) return;
    }
    try {
      setMaintenanceSaving(true);
      const allowlistUserIds = parsePositiveIntList(maintenanceAllowUserIds);
      const allowlistEmails = parseEmailList(maintenanceAllowEmails);
      if (allowlistUserIds.invalid.length > 0) {
        warning(
          'Some allowlist user IDs were ignored',
          `Invalid values: ${formatInvalidValues(allowlistUserIds.invalid)}`
        );
      }
      if (allowlistEmails.invalid.length > 0) {
        warning(
          'Some allowlist emails were ignored',
          `Invalid values: ${formatInvalidValues(allowlistEmails.invalid)}`
        );
      }
      const payload = {
        enabled: maintenanceEnabled,
        message: maintenanceMessage,
        allowlist_user_ids: allowlistUserIds.values,
        allowlist_emails: allowlistEmails.values,
      };
      const updated = await api.updateMaintenanceMode(payload);
      const updatedState = updated as MaintenanceState;
      setMaintenance(updatedState);
      setMaintenanceEnabled(Boolean(updatedState.enabled));
      setMaintenanceMessage(updatedState.message || '');
      setMaintenanceAllowUserIds((updatedState.allowlist_user_ids || []).join(', '));
      setMaintenanceAllowEmails((updatedState.allowlist_emails || []).join(', '));
      success('Maintenance mode updated');
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to update maintenance mode';
      showError(message);
    } finally {
      setMaintenanceSaving(false);
    }
  };

  const handleUpsertFlag = async () => {
    const key = flagKey.trim();
    if (!key) {
      showError('Flag key is required');
      return;
    }
    if (flagScope === 'org' && !flagOrgId.trim()) {
      showError('Org ID is required for org-scoped flags');
      return;
    }
    if (flagScope === 'user' && !flagUserId.trim()) {
      showError('User ID is required for user-scoped flags');
      return;
    }
    const parsedOrgId = flagOrgId.trim() ? parsePositiveInt(flagOrgId) : undefined;
    const parsedUserId = flagUserId.trim() ? parsePositiveInt(flagUserId) : undefined;
    const parsedRolloutPercent = parseRolloutPercent(flagRolloutPercent);
    const parsedTargetUsers = parsePositiveIntList(flagTargetUsers);
    if (flagScope === 'org' && !parsedOrgId) {
      showError('Org ID must be a positive integer');
      return;
    }
    if (flagScope === 'user' && !parsedUserId) {
      showError('User ID must be a positive integer');
      return;
    }
    if (parsedRolloutPercent === null) {
      showError('Rollout % must be an integer between 0 and 100');
      return;
    }
    if (parsedTargetUsers.invalid.length > 0) {
      showError(
        'Target user IDs must be positive integers',
        `Invalid values: ${formatInvalidValues(parsedTargetUsers.invalid)}`
      );
      return;
    }
    try {
      setFlagSaving(true);
      await api.upsertFeatureFlag(key, {
        scope: flagScope,
        enabled: flagEnabled,
        description: flagDescription,
        org_id: flagScope === 'org' ? parsedOrgId : undefined,
        user_id: flagScope === 'user' ? parsedUserId : undefined,
        target_user_ids: Array.from(new Set(parsedTargetUsers.values)),
        rollout_percent: parsedRolloutPercent,
        variant_value: flagVariantValue.trim() || undefined,
        note: flagNote,
      });
      success('Feature flag saved');
      setFlagKey('');
      setFlagDescription('');
      setFlagOrgId('');
      setFlagUserId('');
      setFlagTargetUsers('');
      setFlagRolloutPercent('100');
      setFlagVariantValue('');
      setFlagNote('');
      await loadFlags();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to save feature flag';
      showError(message);
    } finally {
      setFlagSaving(false);
    }
  };

  const handleToggleFlag = async (flag: FeatureFlagItem) => {
    const flagId = getFlagId(flag);
    if (togglingFlagId === flagId || togglingFlagsRef.current.has(flagId)) return;
    const previousEnabled = flag.enabled;
    const nextEnabled = !previousEnabled;

    // Optimistic update: toggle immediately in UI
    togglingFlagsRef.current.add(flagId);
    setTogglingFlags((prev) => ({ ...prev, [flagId]: true }));
    setFlags((prev) =>
      prev.map((f) => (getFlagId(f) === flagId ? { ...f, enabled: nextEnabled } : f))
    );
    setTogglingFlagId(flagId);

    try {
      await api.upsertFeatureFlag(flag.key, {
        scope: flag.scope,
        enabled: nextEnabled,
        org_id: flag.org_id ?? undefined,
        user_id: flag.user_id ?? undefined,
        description: flag.description ?? undefined,
        target_user_ids: flag.target_user_ids ?? undefined,
        rollout_percent: flag.rollout_percent ?? undefined,
        variant_value: flag.variant_value ?? undefined,
        note: `Toggled ${nextEnabled ? 'on' : 'off'} via quick toggle`,
      });
      success(`Flag "${flag.key}" ${nextEnabled ? 'enabled' : 'disabled'}`);
    } catch (err: unknown) {
      // Revert on error
      setFlags((prev) =>
        prev.map((f) => (getFlagId(f) === flagId ? { ...f, enabled: previousEnabled } : f))
      );
      const message = err instanceof Error && err.message ? err.message : 'Failed to toggle flag';
      showError(message);
    } finally {
      setTogglingFlagId((prev) => (prev === flagId ? null : prev));
      togglingFlagsRef.current.delete(flagId);
      setTogglingFlags((prev) => {
        const { [flagId]: _removed, ...rest } = prev;
        return rest;
      });
    }
  };

  const handleDeleteFlag = async (flag: FeatureFlagItem) => {
    const flagId = getFlagId(flag);
    if (deletingFlagId === flagId) return;
    const result = await promptPrivilegedAction({
      title: `Delete flag ${flag.key}?`,
      message: 'This removes the flag override for the selected scope.',
      confirmText: 'Delete',
      requirePassword: false,
    });
    if (!result) return;
    try {
      setDeletingFlagId(flagId);
      const params: Record<string, string> = { scope: flag.scope };
      if (flag.org_id !== null && flag.org_id !== undefined) {
        params.org_id = String(flag.org_id);
      }
      if (flag.user_id !== null && flag.user_id !== undefined) {
        params.user_id = String(flag.user_id);
      }
      await api.deleteFeatureFlag(flag.key, params);
      success('Feature flag deleted');
      await loadFlags();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to delete feature flag';
      showError(message);
    } finally {
      setDeletingFlagId((prev) => (prev === flagId ? null : prev));
    }
  };

  const isRefreshing = maintenanceLoading || flagLoading;

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="flex flex-col gap-6 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Flags & Maintenance</h1>
              <p className="text-muted-foreground">Control runtime switches and maintenance mode.</p>
            </div>
            <Button variant="outline" onClick={handleRefresh} loading={isRefreshing} loadingText="Refreshing...">
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Maintenance Mode</CardTitle>
              <CardDescription>Block non-allowlisted users with a custom message.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <div className="flex items-center gap-2">
                <Checkbox
                  checked={maintenanceEnabled}
                  onCheckedChange={(checked) => setMaintenanceEnabled(Boolean(checked))}
                  id="maintenance-enabled"
                />
                <Label htmlFor="maintenance-enabled">Enabled</Label>
                {maintenance?.updated_at && (
                  <Badge variant="outline" className="ml-2">
                    Updated {formatFlagDate(maintenance.updated_at)}
                  </Badge>
                )}
              </div>
              <div className="space-y-1 md:col-span-2">
                <Label htmlFor="maintenance-message">Message</Label>
                <Input
                  id="maintenance-message"
                  placeholder="Maintenance in progress"
                  value={maintenanceMessage}
                  onChange={(e) => setMaintenanceMessage(e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="maintenance-users">Allowlist User IDs</Label>
                <Input
                  id="maintenance-users"
                  placeholder="1, 2, 3"
                  value={maintenanceAllowUserIds}
                  onChange={(e) => setMaintenanceAllowUserIds(e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="maintenance-emails">Allowlist Emails</Label>
                <Input
                  id="maintenance-emails"
                  placeholder="admin@example.com"
                  value={maintenanceAllowEmails}
                  onChange={(e) => setMaintenanceAllowEmails(e.target.value)}
                />
              </div>
              <div>
                <Button
                  onClick={handleSaveMaintenance}
                  disabled={maintenanceSaving || maintenanceLoading || !maintenance}
                  loading={maintenanceSaving}
                  loadingText="Saving..."
                >
                  Save Maintenance
                </Button>
              </div>
            </CardContent>
          </Card>

          <hr className="my-2 border-border" />
          <Card>
            <CardHeader>
              <CardTitle>Feature Flags</CardTitle>
              <CardDescription>
                Manage feature overrides by scope. This is a feature flag system, not an A/B testing platform — for experiments, use a dedicated experimentation tool.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-1">
                  <Label htmlFor="flag-scope-filter">Scope</Label>
                  <Select
                    id="flag-scope-filter"
                    value={flagScopeFilter}
                    onChange={(e) => setFlagScopeFilter(e.target.value)}
                  >
                    <option value="">All</option>
                    {FLAG_SCOPES.map((scope) => (
                      <option key={scope} value={scope}>
                        {scope}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="flag-org-filter">Org ID</Label>
                  <Input
                    id="flag-org-filter"
                    placeholder="optional"
                    value={flagOrgFilter}
                    onChange={(e) => setFlagOrgFilter(e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="flag-user-filter">User ID</Label>
                  <Input
                    id="flag-user-filter"
                    placeholder="optional"
                    value={flagUserFilter}
                    onChange={(e) => setFlagUserFilter(e.target.value)}
                  />
                </div>
              </div>

              {flagError && (
                <Alert variant="destructive">
                  <AlertDescription>{flagError}</AlertDescription>
                </Alert>
              )}

              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-1">
                  <Label htmlFor="flag-key">Flag Key</Label>
                  <Input
                    id="flag-key"
                    placeholder="claims.monitoring"
                    value={flagKey}
                    onChange={(e) => setFlagKey(e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="flag-scope">Scope</Label>
                  <Select
                    id="flag-scope"
                    value={flagScope}
                    onChange={(e) => setFlagScope(e.target.value as typeof flagScope)}
                  >
                    {FLAG_SCOPES.map((scope) => (
                      <option key={scope} value={scope}>
                        {scope}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="flex items-center gap-2 pt-6">
                  <Checkbox
                    checked={flagEnabled}
                    onCheckedChange={(checked) => setFlagEnabled(Boolean(checked))}
                    id="flag-enabled"
                  />
                  <Label htmlFor="flag-enabled">Enabled</Label>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="flag-org">Org ID (org scope)</Label>
                  <Input
                    id="flag-org"
                    placeholder="optional"
                    value={flagOrgId}
                    onChange={(e) => setFlagOrgId(e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="flag-user">User ID (user scope)</Label>
                  <Input
                    id="flag-user"
                    placeholder="optional"
                    value={flagUserId}
                    onChange={(e) => setFlagUserId(e.target.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="flag-rollout">Rollout %</Label>
                  <Input
                    id="flag-rollout"
                    type="number"
                    min={0}
                    max={100}
                    value={flagRolloutPercent}
                    onChange={(e) => setFlagRolloutPercent(e.target.value)}
                  />
                </div>
                <div className="space-y-1 md:col-span-3">
                  <Label htmlFor="flag-target-users">Target User IDs (comma-separated)</Label>
                  <Input
                    id="flag-target-users"
                    placeholder="e.g. 12, 42, 77"
                    value={flagTargetUsers}
                    onChange={(e) => setFlagTargetUsers(e.target.value)}
                  />
                </div>
                <div className="space-y-1 md:col-span-3">
                  <Label htmlFor="flag-variant">Variant Value (optional)</Label>
                  <Input
                    id="flag-variant"
                    placeholder="e.g. experiment_a"
                    value={flagVariantValue}
                    onChange={(e) => setFlagVariantValue(e.target.value)}
                  />
                </div>
                <div className="space-y-1 md:col-span-3">
                  <Label htmlFor="flag-description">Description</Label>
                  <Input
                    id="flag-description"
                    placeholder="optional"
                    value={flagDescription}
                    onChange={(e) => setFlagDescription(e.target.value)}
                  />
                </div>
                <div className="space-y-1 md:col-span-3">
                  <Label htmlFor="flag-note">Change Note</Label>
                  <Input
                    id="flag-note"
                    placeholder="optional"
                    value={flagNote}
                    onChange={(e) => setFlagNote(e.target.value)}
                  />
                </div>
                <div>
                  <Button onClick={handleUpsertFlag} loading={flagSaving} loadingText="Saving...">
                    Save Flag
                  </Button>
                </div>
              </div>

              {flagLoading ? (
                <div className="py-8 text-center text-muted-foreground" role="status" aria-live="polite">Loading flags...</div>
              ) : flags.length === 0 ? (
                <EmptyState
                  title="No flags found."
                  description="Create a feature flag to control rollout behavior."
                />
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Key</TableHead>
                      <TableHead>Scope</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Target</TableHead>
                      <TableHead>Rollout</TableHead>
                      <TableHead>Variant</TableHead>
                      <TableHead>Updated</TableHead>
                      <TableHead>History</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {flags.map((flag, index) => {
                      const hasStableFields =
                        Boolean(flag.key) ||
                        Boolean(flag.scope) ||
                        flag.org_id !== undefined ||
                        flag.user_id !== undefined;
                      const rowKey =
                        flag.id !== null && flag.id !== undefined
                          ? `flag-${flag.id}`
                          : hasStableFields
                            ? JSON.stringify([
                                flag.key,
                                flag.scope,
                                flag.org_id ?? 'NULL',
                                flag.user_id ?? 'NULL',
                              ])
                            : `flag-index-${index}`;
                      const flagId = getFlagId(flag);
                      const isDeleting = deletingFlagId === flagId;
                      const rolloutPercent = Math.min(
                        100,
                        Math.max(0, Math.trunc(Number(flag.rollout_percent ?? 100)))
                      );
                      return (
                        <TableRow key={rowKey}>
                          <TableCell className="font-medium">{flag.key}</TableCell>
                          <TableCell>{flag.scope}</TableCell>
                          <TableCell>
                            <button
                              type="button"
                              onClick={() => void handleToggleFlag(flag)}
                              disabled={togglingFlagId === flagId || Boolean(togglingFlags[flagId])}
                              className="cursor-pointer border-0 bg-transparent p-0 hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-50"
                              title={`Click to ${flag.enabled ? 'disable' : 'enable'} flag`}
                              aria-label={`Toggle flag ${flag.key} ${flag.enabled ? 'off' : 'on'}`}
                              aria-pressed={flag.enabled}
                            >
                              <Badge variant={flag.enabled ? 'default' : 'outline'}>
                                {togglingFlagId === flagId ? 'Toggling...' : (flag.enabled ? 'Enabled' : 'Disabled')}
                              </Badge>
                            </button>
                          </TableCell>
                          <TableCell>
                            <div className="space-y-1">
                              <div>{getFlagTargetLabel(flag)}</div>
                              <div className="text-xs text-muted-foreground">
                                Users: {formatTargetUsers(flag.target_user_ids)}
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="w-32 space-y-1">
                              <div className="text-xs text-muted-foreground">
                                {rolloutPercent}% rollout
                              </div>
                              <div
                                role="progressbar"
                                aria-label={`Rollout ${rolloutPercent}%`}
                                aria-valuemin={0}
                                aria-valuemax={100}
                                aria-valuenow={rolloutPercent}
                                className="h-2 overflow-hidden rounded bg-muted"
                              >
                                <div
                                  className="h-full bg-primary"
                                  style={{ width: `${rolloutPercent}%` }}
                                />
                              </div>
                            </div>
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {formatVariant(flag.variant_value)}
                          </TableCell>
                          <TableCell>{formatFlagDate(flag.updated_at)}</TableCell>
                          <TableCell>
                            <details className="text-xs text-muted-foreground group">
                              <summary className="cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden rounded px-1.5 py-0.5 hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                                <span className="inline-flex items-center gap-1">
                                  <span className="transition-transform group-open:rotate-90" aria-hidden="true">&#9654;</span>
                                  {flag.history?.length || 0} changes
                                </span>
                              </summary>
                              <div className="mt-2 space-y-2">
                                {(flag.history || []).map((entry, entryIndex) => {
                                  const diffRows = getDiffRows(entry);
                                  return (
                                    <div key={`${flag.key}-history-${entryIndex}`}>
                                      <div className="font-medium text-foreground">
                                        {entry.enabled ? 'Enabled' : 'Disabled'}
                                      </div>
                                      <div>
                                        {formatFlagDate(entry.timestamp)}{' '}
                                        {entry.actor ? `· ${entry.actor}` : ''}
                                      </div>
                                      {entry.note ? <div>Note: {entry.note}</div> : null}
                                      {diffRows.length > 0 ? (
                                        <div className="space-y-1 rounded border border-border/60 p-2">
                                          {diffRows.map((row) => (
                                            <div key={`${entry.timestamp}-${row.field}`}>
                                              <span className="font-medium text-foreground">{row.field}:</span>{' '}
                                              <span className="line-through opacity-70">{row.before}</span>{' '}
                                              <span aria-hidden="true">→</span> <span>{row.after}</span>
                                            </div>
                                          ))}
                                        </div>
                                      ) : null}
                                    </div>
                                  );
                                })}
                              </div>
                            </details>
                          </TableCell>
                          <TableCell>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleDeleteFlag(flag)}
                              title={isDeleting ? 'Deleting flag' : 'Delete flag'}
                              aria-label={isDeleting ? 'Deleting flag' : 'Delete flag'}
                              disabled={isDeleting}
                              loading={isDeleting}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
