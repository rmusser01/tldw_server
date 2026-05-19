'use client';

import { Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { EmptyState } from '@/components/ui/empty-state';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Pagination } from '@/components/ui/pagination';
import { Select } from '@/components/ui/select';
import { CardSkeleton, TableSkeleton } from '@/components/ui/skeleton';
import { useOrgContext } from '@/components/OrgContextSwitcher';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/toast';
import { UnifiedApiKeysTable } from '@/components/api-keys/UnifiedApiKeysTable';
import {
  buildKeyHygieneSummary,
  buildUnifiedApiKeyRows,
  filterByHygiene,
  filterUnifiedApiKeyRows,
  getKeyExpiryIndicator,
  type ApiKeyMetadataLike,
  type ApiKeyUsageLike,
  type HygieneFilter,
  type UnifiedApiKeyStatus,
} from '@/lib/api-keys-hub';
import { api } from '@/lib/api-client';
import { useUrlPagination, useUrlState } from '@/lib/use-url-state';
import type { UserWithKeyCount } from '@/types';
import { AlertTriangle, Copy, FileText, Key, Plus, RotateCw, Search, ShieldOff } from 'lucide-react';
import { ExportMenu } from '@/components/ui/export-menu';
import { exportUnifiedApiKeys, ExportFormat } from '@/lib/export';
import { logger } from '@/lib/logger';

const USER_PAGE_LIMIT = 100;

function ApiKeysPageContent() {
  const { selectedOrg } = useOrgContext();
  const confirm = useConfirm();
  const { success: toastSuccess, error: toastError } = useToast();

  const [availableUsers, setAvailableUsers] = useState<UserWithKeyCount[]>([]);
  const [rows, setRows] = useState<ReturnType<typeof buildUnifiedApiKeyRows>>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [partialLoadWarning, setPartialLoadWarning] = useState('');
  const [selectedRowIds, setSelectedRowIds] = useState<Set<string>>(new Set());
  const [bulkRotating, setBulkRotating] = useState(false);
  const [bulkRevoking, setBulkRevoking] = useState(false);
  const [bulkMutating, setBulkMutating] = useState(false);

  // Create Key dialog state
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [createUserId, setCreateUserId] = useState('');
  const [createKeyName, setCreateKeyName] = useState('');
  const [createKeyScope, setCreateKeyScope] = useState('read');
  const [createKeyExpiresDays, setCreateKeyExpiresDays] = useState('90');
  const [creatingKey, setCreatingKey] = useState(false);
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useUrlState<string>('q', { defaultValue: '' });
  const [ownerFilter, setOwnerFilter] = useUrlState<string>('owner', { defaultValue: '' });
  const [statusFilter, setStatusFilter] = useUrlState<string>('status', { defaultValue: 'all' });
  const [createdBefore, setCreatedBefore] = useUrlState<string>('created_before', { defaultValue: '' });
  const [hygieneFilter, setHygieneFilter] = useState<HygieneFilter>('none');

  const {
    page: currentPage,
    pageSize,
    setPage: setCurrentPage,
    setPageSize,
    resetPagination,
  } = useUrlPagination();

  const loadUnifiedApiKeys = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      setPartialLoadWarning('');

      const loadedUsers: UserWithKeyCount[] = [];
      const userQuery: Record<string, string> = {
        page: '1',
        limit: String(USER_PAGE_LIMIT),
      };
      if (selectedOrg) {
        userQuery.org_id = String(selectedOrg.id);
      }

      let page = 1;
      let pages = 1;
      while (page <= pages) {
        userQuery.page = String(page);
        const usersPage = await api.getUsersPage(userQuery);
        loadedUsers.push(...usersPage.items);

        pages = usersPage.pages > 0
          ? usersPage.pages
          : Math.max(1, Math.ceil(usersPage.total / Math.max(usersPage.limit, 1)));

        if (usersPage.items.length === 0) {
          break;
        }

        page += 1;
      }

      const keyResults = await Promise.allSettled(
        loadedUsers.map(async (user) => ({
          userId: user.id,
          username: user.username,
          keys: await api.getUserApiKeys(String(user.id), { include_revoked: true }) as ApiKeyMetadataLike[],
        }))
      );

      const keysByUserId: Record<number, ApiKeyMetadataLike[]> = {};
      const failedUsers: string[] = [];

      keyResults.forEach((result, index) => {
        const fallbackUser = loadedUsers[index];
        if (result.status === 'fulfilled') {
          keysByUserId[result.value.userId] = Array.isArray(result.value.keys)
            ? result.value.keys
            : [];
          return;
        }

        if (fallbackUser) {
          keysByUserId[fallbackUser.id] = [];
          failedUsers.push(fallbackUser.username);
        }
      });

      // Fetch per-key usage attribution (best-effort; does not block key listing)
      let usageByKeyId: Record<string, ApiKeyUsageLike> | undefined;
      try {
        const topUsage = await api.getTopApiKeyUsage(200);
        if (topUsage?.items?.length) {
          usageByKeyId = {};
          for (const item of topUsage.items) {
            usageByKeyId[String(item.key_id)] = item;
          }
        }
      } catch {
        // Usage data is non-critical; proceed without it
      }

      setAvailableUsers(loadedUsers);
      setRows(buildUnifiedApiKeyRows(loadedUsers, keysByUserId, usageByKeyId));

      if (failedUsers.length > 0) {
        setPartialLoadWarning(
          `Some key lists could not be loaded (${failedUsers.slice(0, 5).join(', ')}${failedUsers.length > 5 ? ', ...' : ''}).`
        );
      }
    } catch (err: unknown) {
      logger.error('Failed to load unified API keys', { component: 'ApiKeysPage', error: err instanceof Error ? err.message : String(err) });
      const message = err instanceof Error && err.message ? err.message : 'Failed to load API keys';
      setError(message);
      setAvailableUsers([]);
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, [selectedOrg]);

  useEffect(() => {
    void loadUnifiedApiKeys();
  }, [loadUnifiedApiKeys]);

  const ownerOptions = useMemo(() => {
    const byOwner = new Map<number, { id: number; username: string }>();
    availableUsers.forEach((user) => {
      if (!byOwner.has(user.id)) {
        byOwner.set(user.id, { id: user.id, username: user.username });
      }
    });
    return [...byOwner.values()].sort((a, b) => a.username.localeCompare(b.username));
  }, [availableUsers]);

  const normalizedStatusFilter: 'all' | UnifiedApiKeyStatus =
    statusFilter === 'active' || statusFilter === 'revoked' || statusFilter === 'expired'
      ? statusFilter
      : 'all';

  const filteredRows = useMemo(
    () => {
      const baseFiltered = filterUnifiedApiKeyRows(rows, {
        search: searchQuery || '',
        ownerUserId: ownerFilter ? Number(ownerFilter) : null,
        status: normalizedStatusFilter,
        createdBefore: createdBefore || '',
      });
      return filterByHygiene(baseFiltered, hygieneFilter);
    },
    [rows, searchQuery, ownerFilter, normalizedStatusFilter, createdBefore, hygieneFilter]
  );
  const hygieneSummary = useMemo(() => buildKeyHygieneSummary(rows), [rows]);

  // Count keys expiring within 7 days for the urgent warning banner
  const keysExpiringIn7Days = useMemo(() => {
    const now = new Date();
    return rows.filter((row) => {
      if (row.status !== 'active') return false;
      const indicator = getKeyExpiryIndicator(row.expiresAt, now);
      return indicator !== null && indicator.daysRemaining <= 7;
    }).length;
  }, [rows]);

  const rowById = useMemo(() => {
    return new Map(rows.map((row) => [`${row.ownerUserId}:${row.keyId}`, row]));
  }, [rows]);

  const totalItems = filteredRows.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  const startIndex = (currentPage - 1) * pageSize;
  const paginatedRows = filteredRows.slice(startIndex, startIndex + pageSize);

  const updateFilter = (setter: (value: string | undefined) => void, value: string) => {
    setter(value || undefined);
    resetPagination();
  };

  const clearFilters = () => {
    setSearchQuery(undefined);
    setOwnerFilter(undefined);
    setStatusFilter(undefined);
    setCreatedBefore(undefined);
    setHygieneFilter('none');
    resetPagination();
  };

  const toggleHygieneFilter = (filter: HygieneFilter) => {
    setHygieneFilter((prev) => (prev === filter ? 'none' : filter));
    resetPagination();
  };

  const hasActiveFilters = Boolean(searchQuery || ownerFilter || statusFilter !== 'all' || createdBefore || hygieneFilter !== 'none');
  const selectedRows = [...selectedRowIds]
    .map((rowId) => rowById.get(rowId))
    .filter((row): row is NonNullable<typeof row> => Boolean(row));

  const handleToggleRowSelection = (rowId: string, checked: boolean) => {
    setSelectedRowIds((previous) => {
      const next = new Set(previous);
      if (checked) {
        next.add(rowId);
      } else {
        next.delete(rowId);
      }
      return next;
    });
  };

  const handleToggleAllSelection = (rowIds: string[], checked: boolean) => {
    setSelectedRowIds((previous) => {
      const next = new Set(previous);
      rowIds.forEach((rowId) => {
        if (checked) {
          next.add(rowId);
        } else {
          next.delete(rowId);
        }
      });
      return next;
    });
  };

  const clearSelection = () => {
    setSelectedRowIds(new Set());
  };

  const handleRotateSelected = async () => {
    if (selectedRows.length === 0 || bulkMutating) return;

    const confirmed = await confirm({
      title: 'Rotate selected keys',
      message: `Rotate ${selectedRows.length} key(s)? Existing keys will stop working immediately.`,
      confirmText: 'Rotate',
      variant: 'warning',
      icon: 'rotate',
    });
    if (!confirmed) return;

    try {
      setBulkMutating(true);
      setBulkRotating(true);
      const results = await Promise.allSettled(
        selectedRows.map((row) => api.rotateApiKey(String(row.ownerUserId), row.keyId))
      );
      const successCount = results.filter((result) => result.status === 'fulfilled').length;
      const failureCount = results.length - successCount;

      if (successCount > 0) {
        toastSuccess('Bulk rotation complete', `${successCount} key(s) rotated successfully.`);
      }
      if (failureCount > 0) {
        toastError('Some rotations failed', `${failureCount} key(s) failed to rotate.`);
      }

      clearSelection();
      await loadUnifiedApiKeys();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Bulk rotation failed';
      toastError('Bulk rotation failed', message);
    } finally {
      setBulkRotating(false);
      setBulkMutating(false);
    }
  };

  const handleRevokeSelected = async () => {
    const revokeableRows = selectedRows.filter((row) => row.status === 'active');
    if (revokeableRows.length === 0 || bulkRevoking) return;

    const confirmed = await confirm({
      title: 'Revoke selected keys',
      message: `Permanently revoke ${revokeableRows.length} active key(s)? This action cannot be undone.`,
      confirmText: 'Revoke',
      variant: 'danger',
      icon: 'warning',
    });
    if (!confirmed) return;

    try {
      setBulkMutating(true);
      setBulkRevoking(true);
      const results = await Promise.allSettled(
        revokeableRows.map((row) => api.revokeApiKey(String(row.ownerUserId), row.keyId))
      );
      const successCount = results.filter((result) => result.status === 'fulfilled').length;
      const failureCount = results.length - successCount;

      if (successCount > 0) {
        toastSuccess('Bulk revocation complete', `${successCount} key(s) revoked successfully.`);
      }
      if (failureCount > 0) {
        toastError('Some revocations failed', `${failureCount} key(s) failed to revoke.`);
      }

      clearSelection();
      await loadUnifiedApiKeys();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Bulk revocation failed';
      toastError('Bulk revocation failed', message);
    } finally {
      setBulkRevoking(false);
      setBulkMutating(false);
    }
  };

  const handleOpenCreateDialog = () => {
    setCreateUserId('');
    setCreateKeyName('');
    setCreateKeyScope('read');
    setCreateKeyExpiresDays('90');
    setNewKeyValue(null);
    setCreateDialogOpen(true);
  };

  const handleCreateKey = async () => {
    if (!createUserId || !createKeyName.trim()) return;
    try {
      setCreatingKey(true);
      const result = await api.createApiKey(createUserId, {
        name: createKeyName.trim(),
        scope: createKeyScope,
        expires_in_days: parseInt(createKeyExpiresDays) || 90,
      });
      if (result.key) {
        setNewKeyValue(result.key);
      }
      toastSuccess('API key created', `Key created for ${ownerOptions.find((o) => String(o.id) === createUserId)?.username || 'user'}.`);
      await loadUnifiedApiKeys();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to create API key';
      toastError('Create key failed', message);
    } finally {
      setCreatingKey(false);
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toastSuccess('Copied', 'API key copied to clipboard.');
    } catch {
      toastError('Copy failed', 'Could not copy to clipboard.');
    }
  };

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <div className="mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">API Keys</h1>
              <p className="text-muted-foreground">Unified API key inventory across all users</p>
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={handleOpenCreateDialog}>
                <Plus className="mr-2 h-4 w-4" />
                Create Key
              </Button>
              <ExportMenu
                onExport={(format: ExportFormat) => exportUnifiedApiKeys(filteredRows, format)}
                disabled={filteredRows.length === 0}
              />
            </div>
          </div>

          {keysExpiringIn7Days > 0 && (
            <Alert className="mb-6 border-yellow-300 bg-yellow-50">
              <AlertTriangle className="h-4 w-4 text-yellow-600" />
              <AlertDescription className="ml-2 flex items-center justify-between">
                <span className="text-yellow-800">
                  {keysExpiringIn7Days} key{keysExpiringIn7Days !== 1 ? 's' : ''} expire{keysExpiringIn7Days === 1 ? 's' : ''} within 7 days.
                </span>
                <Button
                  variant="link"
                  size="sm"
                  className="text-yellow-800 underline p-0 h-auto"
                  onClick={() => toggleHygieneFilter('expiring-soon')}
                >
                  View expiring keys
                </Button>
              </AlertDescription>
            </Alert>
          )}

          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {partialLoadWarning && (
            <Alert className="mb-6">
              <AlertDescription>{partialLoadWarning}</AlertDescription>
            </Alert>
          )}

          <Card className="mb-6">
            <CardContent className="pt-6">
              <div className="flex items-start gap-4">
                <Key className="mt-1 h-8 w-8 text-primary" />
                <div>
                  <h3 className="font-semibold">Unified Key Management</h3>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Per-key token consumption and estimated cost appear automatically when usage data is available.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="mb-6 grid gap-4 md:grid-cols-4">
            <Card
              className={`cursor-pointer transition-colors hover:border-primary ${hygieneFilter === 'needs-rotation' ? 'border-primary ring-1 ring-primary' : ''}`}
              onClick={() => toggleHygieneFilter('needs-rotation')}
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Keys Needing Rotation</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{hygieneSummary.keysNeedingRotation}</p>
                <p className="text-xs text-muted-foreground">&gt;180 days old</p>
              </CardContent>
            </Card>
            <Card
              className={`cursor-pointer transition-colors hover:border-primary ${hygieneFilter === 'expiring-soon' ? 'border-primary ring-1 ring-primary' : ''}`}
              onClick={() => toggleHygieneFilter('expiring-soon')}
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Expiring Soon</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{hygieneSummary.keysExpiringSoon}</p>
                <p className="text-xs text-muted-foreground">Within 30 days</p>
              </CardContent>
            </Card>
            <Card
              className={`cursor-pointer transition-colors hover:border-primary ${hygieneFilter === 'inactive' ? 'border-primary ring-1 ring-primary' : ''}`}
              onClick={() => toggleHygieneFilter('inactive')}
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Inactive Keys</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{hygieneSummary.keysInactive}</p>
                <p className="text-xs text-muted-foreground">No use in 30+ days</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">Hygiene Score</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{hygieneSummary.hygieneScore}%</p>
                <p className="text-xs text-muted-foreground">Higher is healthier</p>
              </CardContent>
            </Card>
          </div>

          <div className="mb-6 flex items-center gap-2 text-sm">
            <FileText className="h-4 w-4 text-muted-foreground" />
            <Link
              href="/audit?resource=api_key"
              className="text-primary hover:underline"
            >
              View key activity in audit log
            </Link>
          </div>

          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Filters</CardTitle>
              <CardDescription>Search by key prefix or owner username, then narrow by owner, status, and age.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 md:grid-cols-5">
                <div className="relative md:col-span-2">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    placeholder="Search key prefix, key ID, or owner"
                    value={searchQuery || ''}
                    onChange={(event) => updateFilter(setSearchQuery, event.target.value)}
                    className="pl-10"
                  />
                </div>

                <Select
                  value={ownerFilter || ''}
                  onChange={(event) => updateFilter(setOwnerFilter, event.target.value)}
                >
                  <option value="">All owners</option>
                  {ownerOptions.map((owner) => (
                    <option key={owner.id} value={String(owner.id)}>
                      {owner.username}
                    </option>
                  ))}
                </Select>

                <Select
                  value={normalizedStatusFilter}
                  onChange={(event) => updateFilter(setStatusFilter, event.target.value)}
                >
                  <option value="all">All statuses</option>
                  <option value="active">Active</option>
                  <option value="revoked">Revoked</option>
                  <option value="expired">Expired</option>
                </Select>

                <div className="flex items-center gap-2">
                  <Input
                    type="date"
                    value={createdBefore || ''}
                    onChange={(event) => updateFilter(setCreatedBefore, event.target.value)}
                  />
                  <Button variant="outline" onClick={clearFilters} disabled={!hasActiveFilters}>
                    Clear
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <CardTitle>All API Keys</CardTitle>
                  <CardDescription>
                    {totalItems} key{totalItems !== 1 ? 's' : ''} matched
                  </CardDescription>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <Button variant="outline" onClick={clearSelection} disabled={selectedRowIds.size === 0}>
                    Clear Selection
                  </Button>
                  <Button
                    onClick={handleRotateSelected}
                    disabled={selectedRowIds.size === 0 || bulkRotating || bulkRevoking || bulkMutating}
                    loading={bulkRotating}
                    loadingText="Rotating..."
                  >
                    <RotateCw className="mr-2 h-4 w-4" />
                    Rotate Selected ({selectedRowIds.size})
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={handleRevokeSelected}
                    disabled={selectedRowIds.size === 0 || bulkRevoking || bulkRotating || bulkMutating}
                    loading={bulkRevoking}
                    loadingText="Revoking..."
                  >
                    <ShieldOff className="mr-2 h-4 w-4" />
                    Revoke Selected ({selectedRowIds.size})
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="py-4">
                  <TableSkeleton rows={8} columns={9} />
                </div>
              ) : totalItems === 0 ? (
                <EmptyState
                  icon={Key}
                  title="No API keys match the current filters."
                  description={
                    hasActiveFilters
                      ? 'Try clearing filters to view all keys.'
                      : 'No keys are currently available for the selected scope.'
                  }
                  actions={[
                    hasActiveFilters
                      ? {
                          label: 'Clear filters',
                          onClick: clearFilters,
                        }
                      : {
                          label: 'Refresh inventory',
                          onClick: () => {
                            void loadUnifiedApiKeys();
                          },
                        },
                  ]}
                />
              ) : (
                <>
                  <UnifiedApiKeysTable
                    rows={paginatedRows}
                    selectedRowIds={selectedRowIds}
                    onToggleRowSelection={handleToggleRowSelection}
                    onToggleAllSelection={handleToggleAllSelection}
                  />
                  {totalItems > 0 && (
                    <Pagination
                      currentPage={currentPage}
                      totalPages={totalPages}
                      totalItems={totalItems}
                      pageSize={pageSize}
                      onPageChange={setCurrentPage}
                      onPageSizeChange={(size) => {
                        setPageSize(size);
                        resetPagination();
                      }}
                    />
                  )}
                </>
              )}
            </CardContent>
          </Card>

          {/* Create Key Dialog */}
          <Dialog open={createDialogOpen} onOpenChange={(open) => {
            if (!open && newKeyValue) {
              // Don't allow silent dismissal when secret is displayed — force confirmation via Done button
              return;
            }
            setCreateDialogOpen(open);
          }}>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Create API Key</DialogTitle>
                <DialogDescription>Select a user and configure the new API key.</DialogDescription>
              </DialogHeader>
              {newKeyValue ? (
                <div className="space-y-4 py-4">
                  <Alert className="border-yellow-300 bg-yellow-50">
                    <AlertDescription>
                      <p className="mb-2 font-semibold text-yellow-800">
                        Save this API key now — it will not be shown again.
                      </p>
                      <div className="flex items-center gap-2">
                        <code className="flex-1 rounded bg-yellow-100 p-2 font-mono text-sm break-all">
                          {newKeyValue}
                        </code>
                        <Button variant="outline" size="sm" onClick={() => copyToClipboard(newKeyValue)}>
                          <Copy className="h-4 w-4" />
                        </Button>
                      </div>
                    </AlertDescription>
                  </Alert>
                  <div className="flex justify-end">
                    <Button onClick={async () => {
                      const confirmed = await confirm({
                        title: 'Dismiss API key',
                        message: 'Have you saved this API key? It will not be shown again.',
                        confirmText: 'Yes, I saved it',
                        variant: 'warning',
                      });
                      if (confirmed) {
                        setCreateDialogOpen(false);
                      }
                    }}>Done</Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="create-key-user">User</Label>
                    <Select
                      id="create-key-user"
                      value={createUserId}
                      onChange={(e) => setCreateUserId(e.target.value)}
                    >
                      <option value="">Select a user...</option>
                      {ownerOptions.map((owner) => (
                        <option key={owner.id} value={String(owner.id)}>
                          {owner.username}
                        </option>
                      ))}
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="create-key-name">Key Name</Label>
                    <Input
                      id="create-key-name"
                      placeholder="e.g., Production API"
                      value={createKeyName}
                      onChange={(e) => setCreateKeyName(e.target.value)}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="create-key-scope">Scope</Label>
                      <Select
                        id="create-key-scope"
                        value={createKeyScope}
                        onChange={(e) => setCreateKeyScope(e.target.value)}
                      >
                        <option value="read">Read Only</option>
                        <option value="write">Read & Write</option>
                        <option value="admin">Admin</option>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="create-key-expires">Expires In (days)</Label>
                      <Input
                        id="create-key-expires"
                        type="number"
                        min="1"
                        max="365"
                        value={createKeyExpiresDays}
                        onChange={(e) => setCreateKeyExpiresDays(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button variant="outline" onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
                    <Button
                      onClick={handleCreateKey}
                      disabled={!createUserId || !createKeyName.trim() || creatingKey}
                      loading={creatingKey}
                      loadingText="Creating..."
                    >
                      Create Key
                    </Button>
                  </div>
                </div>
              )}
            </DialogContent>
          </Dialog>

          {/* Recent Key Activity */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Recent Key Activity</CardTitle>
              <CardDescription>Latest API key create, rotate, and revoke events from audit log</CardDescription>
            </CardHeader>
            <CardContent>
              <KeyActivitySection />
            </CardContent>
          </Card>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

function KeyActivitySection() {
  const [logs, setLogs] = useState<Array<{ id: number; timestamp: string; user_id: number; action: string; resource: string; details?: string }>>([]);
  const [loadingLogs, setLoadingLogs] = useState(true);

  useEffect(() => {
    let cancelled = false;

    api.getAuditLogs({ resource: 'api_key', limit: '10' })
      .then((data) => {
        if (cancelled) return;
        const items = Array.isArray(data) ? data : (data as { items?: unknown[] })?.items ?? [];
        setLogs(items as typeof logs);
      })
      .catch(() => {
        if (!cancelled) {
          setLogs([]);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingLogs(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  if (loadingLogs) return <CardSkeleton />;
  if (logs.length === 0) return <p className="text-sm text-muted-foreground">No recent key activity.</p>;

  return (
    <div className="space-y-2 max-h-48 overflow-y-auto">
      {logs.map((log) => (
        <div key={log.id} className="flex items-center justify-between p-2 rounded bg-muted/30 text-sm">
          <div>
            <span className="font-medium">{log.action}</span>
            <span className="text-muted-foreground ml-2">by User {log.user_id}</span>
          </div>
          <span className="text-xs text-muted-foreground">
            {new Date(log.timestamp).toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function ApiKeysPage() {
  return (
    <Suspense
      fallback={
        <PermissionGuard variant="route" requireAuth role="admin">
          <ResponsiveLayout>
            <div className="p-4 lg:p-8">
              <div className="mb-8">
                <div className="mb-2 h-8 w-32 animate-pulse rounded bg-muted" />
                <div className="h-4 w-64 animate-pulse rounded bg-muted" />
              </div>
              <div className="h-96 animate-pulse rounded bg-muted" />
            </div>
          </ResponsiveLayout>
        </PermissionGuard>
      }
    >
      <ApiKeysPageContent />
    </Suspense>
  );
}
