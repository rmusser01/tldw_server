'use client';

import { useEffect, useRef, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { ArrowLeft, Plus, Copy, KeyRound, ShieldCheck, Trash2, X } from 'lucide-react';
import { AccessibleIconButton } from '@/components/ui/accessible-icon-button';
import { ToggleBadgeGroup } from '@/components/ui/toggle-badge-group';
import { api } from '@/lib/api-client';
import { ApiKeyCreateForm, ApiKeyFormData } from '@/components/users/ApiKeyCreateForm';
import { ApiKeysTable } from '@/components/users/ApiKeysTable';
import { useUserApiKeys } from '@/lib/use-user-api-keys';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { formatDateTime } from '@/lib/format';
import { PageHeaderSkeleton, TableSkeleton, CardSkeleton } from '@/components/ui/skeleton';
import { logger } from '@/lib/logger';

type VirtualApiKey = {
  id: string;
  name: string;
  scopes: string[];
  created_at?: string;
  last_used?: string;
  expires_at?: string;
};

const AVAILABLE_SCOPES = [
  'read:media',
  'write:media',
  'read:chat',
  'write:chat',
  'read:notes',
  'write:notes',
  'read:rag',
  'write:rag',
  'read:embeddings',
  'admin:users',
  'admin:org',
] as const;

const formatDate = (dateStr?: string) => formatDateTime(dateStr, { fallback: '—' });

export default function UserApiKeysPage() {
  const params = useParams();
  const router = useRouter();
  const confirm = useConfirm();
  const userId = params.id as string;

  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newKeyValue, setNewKeyValue] = useState<string | null>(null);
  const [creatingKey, setCreatingKey] = useState(false);
  const successTimerRef = useRef<number | null>(null);
  const isMountedRef = useRef(true);

  // Virtual API Keys
  const [virtualKeys, setVirtualKeys] = useState<VirtualApiKey[]>([]);
  const [virtualKeysLoading, setVirtualKeysLoading] = useState(false);
  const [showVirtualKeyForm, setShowVirtualKeyForm] = useState(false);
  const [virtualKeyName, setVirtualKeyName] = useState('');
  const [selectedScopes, setSelectedScopes] = useState<string[]>([]);
  const [creatingVirtualKey, setCreatingVirtualKey] = useState(false);
  const [newVirtualKeyValue, setNewVirtualKeyValue] = useState<string | null>(null);
  const [deletingVirtualKeyId, setDeletingVirtualKeyId] = useState<string | null>(null);

  const { user, apiKeys, loading, reload } = useUserApiKeys(userId, { onError: setError });

  const loadVirtualKeys = async () => {
    try {
      setVirtualKeysLoading(true);
      const data = await api.getUserVirtualKeys(userId);
      const result = data as { keys?: VirtualApiKey[]; items?: VirtualApiKey[] };
      setVirtualKeys(
        Array.isArray(result.keys) ? result.keys :
        Array.isArray(result.items) ? result.items :
        Array.isArray(result) ? result as VirtualApiKey[] : []
      );
    } catch (err: unknown) {
      logger.error('Failed to load virtual keys', { component: 'UserApiKeysPage', error: err instanceof Error ? err.message : String(err) });
      // Don't set error - virtual keys may not be available
    } finally {
      setVirtualKeysLoading(false);
    }
  };

  const handleDeleteVirtualKey = async (keyId: string, keyName: string) => {
    const confirmed = await confirm({
      title: 'Delete virtual key',
      message: `Delete virtual key "${keyName}"? This action cannot be undone.`,
      confirmText: 'Delete',
      variant: 'danger',
    });
    if (!confirmed) return;
    try {
      setDeletingVirtualKeyId(keyId);
      await api.deleteUserVirtualKey(userId, keyId);
      setSuccess('Virtual key deleted');
      await loadVirtualKeys();
    } catch (err: unknown) {
      logger.error('Failed to delete virtual key', { component: 'UserApiKeysPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to delete virtual key');
    } finally {
      setDeletingVirtualKeyId(null);
    }
  };

  useEffect(() => {
    loadVirtualKeys();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (successTimerRef.current !== null) {
        window.clearTimeout(successTimerRef.current);
        successTimerRef.current = null;
      }
    };
  }, []);

  const handleCreate = async (data: ApiKeyFormData) => {
    try {
      setCreatingKey(true);
      setError('');
      setSuccess('');
      const result = await api.createApiKey(userId, {
        name: data.name,
        scope: data.scope,
        expires_in_days: data.expires_days,
      });

      // The API returns the full key value only on creation
      if (result.key) {
        setNewKeyValue(result.key);
      }

      setSuccess('API key created successfully');
      setShowCreateForm(false);
      void reload();
    } catch (err: unknown) {
      logger.error('Failed to create API key', { component: 'UserApiKeysPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to create API key');
    } finally {
      setCreatingKey(false);
    }
  };

  const handleRotate = async (keyId: string) => {
    const confirmed = await confirm({
      title: 'Rotate API Key',
      message: 'Are you sure you want to rotate this key? The old key will stop working immediately.',
      confirmText: 'Rotate',
      variant: 'warning',
      icon: 'rotate',
    });
    if (!confirmed) return;

    try {
      setError('');
      setSuccess('');
      const result = await api.rotateApiKey(userId, keyId);
      if (result.key) {
        setNewKeyValue(result.key);
      }
      setSuccess('API key rotated successfully');
      void reload();
    } catch (err: unknown) {
      logger.error('Failed to rotate API key', { component: 'UserApiKeysPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to rotate API key');
    }
  };

  const handleRevoke = async (keyId: string, keyName: string) => {
    const confirmed = await confirm({
      title: 'Revoke API Key',
      message: `Are you sure you want to revoke "${keyName || keyId}"? This action cannot be undone.`,
      confirmText: 'Revoke',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;

    try {
      setError('');
      setSuccess('');
      await api.revokeApiKey(userId, keyId);
      setSuccess('API key revoked successfully');
      void reload();
    } catch (err: unknown) {
      logger.error('Failed to revoke API key', { component: 'UserApiKeysPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to revoke API key');
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      if (!isMountedRef.current) {
        return;
      }
      setError('');
      setSuccess('Copied to clipboard!');
      if (successTimerRef.current !== null) {
        window.clearTimeout(successTimerRef.current);
      }
      successTimerRef.current = window.setTimeout(() => {
        if (!isMountedRef.current) {
          return;
        }
        setSuccess('');
        successTimerRef.current = null;
      }, 2000);
    } catch (err: unknown) {
      logger.error('Failed to copy to clipboard', { component: 'UserApiKeysPage', error: err instanceof Error ? err.message : String(err) });
      if (!isMountedRef.current) {
        return;
      }
      setError('Failed to copy to clipboard. Please copy manually or check browser permissions.');
    }
  };

  const handleCreateVirtualKey = async () => {
    if (!virtualKeyName.trim()) {
      setError('Key name is required');
      return;
    }
    if (selectedScopes.length === 0) {
      setError('At least one scope is required');
      return;
    }

    try {
      setCreatingVirtualKey(true);
      setError('');
      const result = await api.createUserVirtualKey(userId, {
        name: virtualKeyName.trim(),
        scopes: selectedScopes,
      });

      // Handle the key value if returned
      const keyResult = result as { key?: string; virtual_key?: string };
      if (keyResult.key || keyResult.virtual_key) {
        setNewVirtualKeyValue(keyResult.key || keyResult.virtual_key || null);
      }

      setSuccess('Virtual API key created successfully');
      setShowVirtualKeyForm(false);
      setVirtualKeyName('');
      setSelectedScopes([]);
      void loadVirtualKeys();
    } catch (err: unknown) {
      logger.error('Failed to create virtual API key', { component: 'UserApiKeysPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to create virtual API key');
    } finally {
      setCreatingVirtualKey(false);
    }
  };

  if (loading) {
    return (
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <PageHeaderSkeleton />
            <TableSkeleton rows={3} columns={4} />
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    );
  }

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            {/* Header */}
            <div className="mb-8 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <AccessibleIconButton
                  icon={ArrowLeft}
                  label="Go back to user details"
                  variant="ghost"
                  onClick={() => router.push(`/users/${userId}`)}
                />
                <div>
                  <h1 className="text-3xl font-bold">API Keys</h1>
                  <p className="text-muted-foreground">
                    Manage API keys for {user?.username || 'user'}
                  </p>
                </div>
              </div>
              <Button onClick={() => setShowCreateForm(!showCreateForm)}>
                <Plus className="mr-2 h-4 w-4" />
                Create API Key
              </Button>
            </div>

            {error && (
              <Alert variant="destructive" className="mb-6">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {success && (
              <Alert className="mb-6 bg-green-50 border-green-200">
                <AlertDescription className="text-green-800">{success}</AlertDescription>
              </Alert>
            )}

            {/* New Key Display */}
            {newKeyValue && (
              <Alert className="mb-6 bg-yellow-50 border-yellow-200">
                <AlertDescription>
                  <div className="space-y-2">
                    <p className="font-semibold text-yellow-800">
                      Save this API key now - it won&apos;t be shown again!
                    </p>
                    <div className="flex items-center gap-2">
                      <code className="flex-1 bg-yellow-100 p-2 rounded font-mono text-sm break-all">
                        {newKeyValue}
                      </code>
                      <AccessibleIconButton
                        icon={Copy}
                        label="Copy API key to clipboard"
                        variant="outline"
                        size="sm"
                        onClick={() => copyToClipboard(newKeyValue)}
                      />
                    </div>
                    <DismissKeyButton key={newKeyValue} onDismiss={() => setNewKeyValue(null)} />
                  </div>
                </AlertDescription>
              </Alert>
            )}

            {/* Create Form */}
            {showCreateForm && (
              <ApiKeyCreateForm
                onSubmit={handleCreate}
                onCancel={() => setShowCreateForm(false)}
                isSubmitting={creatingKey}
              />
            )}

            {/* API Keys Table */}
            <Card>
              <CardHeader>
                <CardTitle>API Keys</CardTitle>
                <CardDescription>
                  {apiKeys.length} key{apiKeys.length !== 1 ? 's' : ''} total
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ApiKeysTable
                  apiKeys={apiKeys}
                  onRotate={handleRotate}
                  onRevoke={handleRevoke}
                />
              </CardContent>
            </Card>

            {/* Virtual API Keys */}
            <Card className="mt-6">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <ShieldCheck className="h-5 w-5" />
                      Virtual API Keys
                    </CardTitle>
                    <CardDescription>
                      Scoped keys with limited permissions for specific use cases
                    </CardDescription>
                  </div>
                  <Button
                    variant="outline"
                    onClick={() => setShowVirtualKeyForm(!showVirtualKeyForm)}
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Create Virtual Key
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {/* New Virtual Key Display */}
                {newVirtualKeyValue && (
                  <Alert className="mb-6 bg-yellow-50 border-yellow-200">
                    <AlertDescription>
                      <div className="space-y-2">
                        <p className="font-semibold text-yellow-800">
                          Save this virtual API key now - it won&apos;t be shown again!
                        </p>
                        <div className="flex items-center gap-2">
                          <code className="flex-1 bg-yellow-100 p-2 rounded font-mono text-sm break-all">
                            {newVirtualKeyValue}
                          </code>
                          <AccessibleIconButton
                            icon={Copy}
                            label="Copy virtual API key to clipboard"
                            variant="outline"
                            size="sm"
                            onClick={() => copyToClipboard(newVirtualKeyValue)}
                          />
                        </div>
                        <DismissKeyButton key={newVirtualKeyValue} onDismiss={() => setNewVirtualKeyValue(null)} />
                      </div>
                    </AlertDescription>
                  </Alert>
                )}

                {/* Create Virtual Key Form */}
                {showVirtualKeyForm && (
                  <div className="mb-6 p-4 border rounded-lg space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Create Virtual API Key</span>
                      <AccessibleIconButton
                        icon={X}
                        label="Close form"
                        variant="ghost"
                        onClick={() => setShowVirtualKeyForm(false)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="virtual-key-name">Key Name</Label>
                      <Input
                        id="virtual-key-name"
                        placeholder="e.g., Chat-only Access"
                        value={virtualKeyName}
                        onChange={(e) => setVirtualKeyName(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Scopes (select permissions)</Label>
                      <ToggleBadgeGroup
                        options={[...AVAILABLE_SCOPES]}
                        selected={selectedScopes}
                        onChange={setSelectedScopes}
                        label="Select API key permission scopes"
                      />
                      {selectedScopes.length > 0 && (
                        <p className="text-xs text-muted-foreground">
                          Selected: {selectedScopes.join(', ')}
                        </p>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={handleCreateVirtualKey}
                        disabled={creatingVirtualKey || !virtualKeyName.trim() || selectedScopes.length === 0}
                        loading={creatingVirtualKey}
                        loadingText="Creating..."
                      >
                        Create Virtual Key
                      </Button>
                      <Button variant="outline" onClick={() => setShowVirtualKeyForm(false)}>
                        Cancel
                      </Button>
                    </div>
                  </div>
                )}

                {/* Virtual Keys List */}
                {virtualKeysLoading ? (
                  <TableSkeleton rows={3} columns={5} />
                ) : virtualKeys.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    <KeyRound className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>No virtual keys created yet.</p>
                    <p className="text-xs">Virtual keys have limited scopes for specific use cases.</p>
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Scopes</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead>Last Used</TableHead>
                        <TableHead>Expires</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {virtualKeys.map((key) => (
                        <TableRow key={key.id}>
                          <TableCell className="font-medium">
                            <div className="flex items-center gap-2">
                              <ShieldCheck className="h-4 w-4 text-muted-foreground" />
                              {key.name}
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="flex flex-wrap gap-1">
                              {key.scopes.slice(0, 3).map((scope) => (
                                <Badge key={scope} variant="secondary" className="text-xs">
                                  {scope}
                                </Badge>
                              ))}
                              {key.scopes.length > 3 && (
                                <Badge variant="outline" className="text-xs">
                                  +{key.scopes.length - 3} more
                                </Badge>
                              )}
                            </div>
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatDate(key.created_at)}
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatDate(key.last_used)}
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatDate(key.expires_at)}
                          </TableCell>
                          <TableCell className="text-right">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                void handleDeleteVirtualKey(key.id, key.name);
                              }}
                              disabled={deletingVirtualKeyId === key.id}
                              loading={deletingVirtualKeyId === key.id}
                              aria-label={`Delete virtual key ${key.name}`}
                              title="Delete virtual key"
                              className="text-destructive hover:text-destructive"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
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

function DismissKeyButton({ onDismiss }: { onDismiss: () => void }) {
  const [confirming, setConfirming] = useState(false);

  if (confirming) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs text-yellow-800">Have you saved this key?</span>
        <Button variant="destructive" size="sm" onClick={onDismiss}>
          Yes, dismiss
        </Button>
        <Button variant="ghost" size="sm" onClick={() => setConfirming(false)} className="text-yellow-700">
          Cancel
        </Button>
      </div>
    );
  }

  return (
    <Button variant="ghost" size="sm" onClick={() => setConfirming(true)} className="text-yellow-700">
      Dismiss
    </Button>
  );
}
