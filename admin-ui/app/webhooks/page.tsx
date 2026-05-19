'use client';

import { Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/ui/empty-state';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { useToast } from '@/components/ui/toast';
import { api } from '@/lib/api-client';
import { formatDateTime } from '@/lib/format';
import type { AdminWebhook, AdminWebhookDeliveryLogEntry } from '@/types';
import { Activity, ChevronDown, ChevronRight, Copy, Play, Plus, RefreshCw, Trash2, Webhook } from 'lucide-react';


const AVAILABLE_EVENTS = [
  'user.created',
  'user.deleted',
  'incident.created',
  'incident.updated',
  'incident.resolved',
] as const;

type DeliveryHistoryItem = {
  id: number;
  event_type: string;
  status_code: number | null;
  response_time_ms: number | null;
  success: boolean;
  error: string | null;
  attempted_at: string | null;
};

const toDeliveryHistoryItem = (delivery: AdminWebhookDeliveryLogEntry): DeliveryHistoryItem => ({
  id: delivery.id,
  event_type: delivery.event_type,
  status_code: delivery.status_code,
  response_time_ms: delivery.latency_ms,
  success: typeof delivery.status_code === 'number' && delivery.status_code >= 200 && delivery.status_code < 300,
  error: delivery.error_message,
  attempted_at: delivery.delivered_at ?? delivery.created_at,
});

function DeliveryStatusBadge({ success }: { success: boolean }) {
  return (
    <Badge variant={success ? 'default' : 'destructive'} className="text-xs">
      {success ? 'Success' : 'Failed'}
    </Badge>
  );
}

function DeliveryHistory({
  webhookId,
  visible,
  refreshKey,
}: {
  webhookId: number;
  visible: boolean;
  refreshKey?: number;
}) {
  const [deliveries, setDeliveries] = useState<DeliveryHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchDeliveries = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const response = await api.getWebhookDeliveries(webhookId, { limit: 50 });
      setDeliveries((response.items ?? []).map(toDeliveryHistoryItem));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load deliveries';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [webhookId]);

  useEffect(() => {
    if (visible) {
      fetchDeliveries();
    }
  }, [visible, fetchDeliveries, refreshKey]);

  if (!visible) return null;

  return (
    <div className="px-4 pb-4">
      <div className="rounded-md border bg-muted/30 p-3">
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-sm font-medium flex items-center gap-1">
            <Activity className="h-3.5 w-3.5" />
            Delivery History
          </h4>
          <Button variant="ghost" size="sm" onClick={fetchDeliveries} disabled={loading} aria-label="Refresh delivery history">
            <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
        {error && (
          <Alert variant="destructive" className="mb-2">
            <AlertDescription className="text-xs">{error}</AlertDescription>
          </Alert>
        )}
        {!loading && deliveries.length === 0 && !error && (
          <p className="text-xs text-muted-foreground py-2">No deliveries recorded yet.</p>
        )}
        {deliveries.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Event</TableHead>
                <TableHead className="text-xs">Status</TableHead>
                <TableHead className="text-xs">HTTP Code</TableHead>
                <TableHead className="text-xs">Response Time</TableHead>
                <TableHead className="text-xs">Time</TableHead>
                <TableHead className="text-xs">Error</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {deliveries.map((delivery) => (
                <TableRow key={delivery.id}>
                  <TableCell className="text-xs font-mono">{delivery.event_type}</TableCell>
                  <TableCell>
                    <DeliveryStatusBadge success={delivery.success} />
                  </TableCell>
                  <TableCell className="text-xs">
                    {delivery.status_code ?? '\u2014'}
                  </TableCell>
                  <TableCell className="text-xs">
                    {delivery.response_time_ms != null ? `${delivery.response_time_ms}ms` : '\u2014'}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {formatDateTime(delivery.attempted_at, { fallback: '\u2014' })}
                  </TableCell>
                  <TableCell className="text-xs text-destructive max-w-[200px] truncate">
                    {delivery.error ?? ''}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  );
}

function WebhooksPageContent() {
  const promptPrivileged = usePrivilegedActionDialog();
  const { success, error: showError } = useToast();

  const [webhooks, setWebhooks] = useState<AdminWebhook[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Track which webhook's deliveries are expanded
  const [expandedWebhookId, setExpandedWebhookId] = useState<number | null>(null);

  // Track which webhooks have a test in-flight
  const [testingWebhookIds, setTestingWebhookIds] = useState<Set<number>>(new Set());
  // Bump to refresh delivery history after a test webhook
  const [deliveryRefreshKey, setDeliveryRefreshKey] = useState(0);

  // Create dialog state
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [createUrl, setCreateUrl] = useState('');
  const [createEvents, setCreateEvents] = useState<string[]>([]);
  const [creating, setCreating] = useState(false);

  // Secret display state (shown once after creation)
  const [showSecretDialog, setShowSecretDialog] = useState(false);
  const [createdSecret, setCreatedSecret] = useState('');
  const [secretCopied, setSecretCopied] = useState(false);

  const fetchWebhooks = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const response = await api.getWebhooks();
      setWebhooks(response.items ?? []);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load webhooks';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWebhooks();
  }, [fetchWebhooks]);

  const handleCreate = async () => {
    if (!createUrl.trim()) return;
    if (createEvents.length === 0) return;
    setCreating(true);
    try {
      const result = await api.createWebhook({
        url: createUrl.trim(),
        event_types: createEvents,
        active: true,
      });
      const returnedSecret =
        'secret' in result && typeof result.secret === 'string' ? result.secret : '';
      success('Webhook created');
      setShowCreateDialog(false);
      setCreateUrl('');
      setCreateEvents([]);
      if (returnedSecret) {
        setCreatedSecret(returnedSecret);
        setSecretCopied(false);
        setShowSecretDialog(true);
      }
      await fetchWebhooks();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create webhook';
      showError(message);
    } finally {
      setCreating(false);
    }
  };

  const handleToggleEnabled = async (webhook: AdminWebhook) => {
    try {
      await api.updateWebhook(webhook.id, { active: !webhook.active });
      success(webhook.active ? 'Webhook disabled' : 'Webhook enabled');
      await fetchWebhooks();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to update webhook';
      showError(message);
    }
  };

  const handleDelete = async (webhook: AdminWebhook) => {
    const result = await promptPrivileged({
      title: 'Delete Webhook',
      message: `Delete the webhook for ${webhook.url}? This cannot be undone.`,
      confirmText: 'Delete Webhook',
    });
    if (!result) return;
    try {
      await api.deleteWebhook(webhook.id);
      success('Webhook deleted');
      if (expandedWebhookId === webhook.id) {
        setExpandedWebhookId(null);
      }
      await fetchWebhooks();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete webhook';
      showError(message);
    }
  };

  const handleTestWebhook = async (webhook: AdminWebhook) => {
    setTestingWebhookIds((prev) => new Set(prev).add(webhook.id));
    try {
      const delivery = await api.testWebhook(webhook.id);
      if (delivery.success) {
        success(`Test delivery succeeded (HTTP ${delivery.status_code})`);
      } else {
        showError(`Test delivery failed: ${delivery.error || `HTTP ${delivery.status_code}`}`);
      }
      // Expand deliveries and refresh to show the new test result
      setExpandedWebhookId(webhook.id);
      setDeliveryRefreshKey((k) => k + 1);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send test';
      showError(message);
    } finally {
      setTestingWebhookIds((prev) => {
        const next = new Set(prev);
        next.delete(webhook.id);
        return next;
      });
    }
  };

  const handleCopySecret = async () => {
    try {
      await navigator.clipboard.writeText(createdSecret);
      setSecretCopied(true);
    } catch {
      // Fallback: select input text
    }
  };

  const toggleCreateEvent = (event: string) => {
    setCreateEvents((prev) =>
      prev.includes(event) ? prev.filter((e) => e !== event) : [...prev, event]
    );
  };

  const toggleDeliveries = (webhookId: number) => {
    setExpandedWebhookId((prev) => (prev === webhookId ? null : webhookId));
  };

  return (
    <ResponsiveLayout>
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Webhooks</CardTitle>
              <CardDescription>
                Configure outgoing webhooks for event notifications
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={fetchWebhooks} disabled={loading}>
                <RefreshCw className={`h-4 w-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              <Button size="sm" onClick={() => setShowCreateDialog(true)}>
                <Plus className="h-4 w-4 mr-1" />
                Add Webhook
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {!loading && webhooks.length === 0 && !error && (
            <EmptyState
              icon={Webhook}
              title="No webhooks configured"
              description="Add a webhook to receive event notifications at your URL."
            />
          )}

          {webhooks.length > 0 && (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-8"></TableHead>
                    <TableHead>URL</TableHead>
                    <TableHead>Events</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {webhooks.map((webhook) => (
                    <>
                      <TableRow key={webhook.id}>
                        <TableCell className="w-8 pr-0">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 w-6 p-0"
                            onClick={() => toggleDeliveries(webhook.id)}
                            aria-label="Toggle deliveries"
                          >
                            {expandedWebhookId === webhook.id ? (
                              <ChevronDown className="h-4 w-4" />
                            ) : (
                              <ChevronRight className="h-4 w-4" />
                            )}
                          </Button>
                        </TableCell>
                        <TableCell className="font-mono text-sm max-w-[300px] truncate">
                          {webhook.url}
                        </TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {webhook.event_types.map((event) => (
                              <Badge key={event} variant="secondary" className="text-xs">
                                {event}
                              </Badge>
                            ))}
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant={webhook.active ? 'default' : 'outline'}>
                            {webhook.active ? 'Enabled' : 'Disabled'}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {formatDateTime(webhook.created_at, { fallback: '\u2014' })}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-1">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleTestWebhook(webhook)}
                              disabled={testingWebhookIds.has(webhook.id)}
                              title="Send test delivery"
                            >
                              <Play className={`h-4 w-4 mr-1 ${testingWebhookIds.has(webhook.id) ? 'animate-pulse' : ''}`} />
                              {testingWebhookIds.has(webhook.id) ? 'Testing...' : 'Test'}
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleToggleEnabled(webhook)}
                            >
                              {webhook.active ? 'Disable' : 'Enable'}
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleDelete(webhook)}
                              aria-label="Delete webhook"
                            >
                              <Trash2 className="h-4 w-4 text-destructive" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                      {expandedWebhookId === webhook.id && (
                        <TableRow key={`${webhook.id}-deliveries`}>
                          <TableCell colSpan={6} className="p-0">
                            <DeliveryHistory
                              webhookId={webhook.id}
                              visible={expandedWebhookId === webhook.id}
                              refreshKey={deliveryRefreshKey}
                            />
                          </TableCell>
                        </TableRow>
                      )}
                    </>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Create Webhook Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Webhook</DialogTitle>
            <DialogDescription>
              Configure a URL to receive event notifications. A signing secret will be generated automatically.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="webhook-url">Endpoint URL</Label>
              <Input
                id="webhook-url"
                type="url"
                placeholder="https://example.com/webhook"
                value={createUrl}
                onChange={(e) => setCreateUrl(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Events</Label>
              <div className="space-y-2">
                {AVAILABLE_EVENTS.map((event) => (
                  <label key={event} className="flex items-center gap-2 text-sm">
                    <Checkbox
                      checked={createEvents.includes(event)}
                      onCheckedChange={() => toggleCreateEvent(event)}
                    />
                    {event}
                  </label>
                ))}
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreate}
              disabled={creating || !createUrl.trim() || createEvents.length === 0}
            >
              {creating ? 'Creating...' : 'Create Webhook'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Secret Display Dialog (shown once) */}
      <Dialog open={showSecretDialog} onOpenChange={setShowSecretDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Webhook Secret</DialogTitle>
            <DialogDescription>
              Copy your webhook signing secret now. It will not be shown again.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Input
                readOnly
                value={createdSecret}
                className="font-mono text-sm"
                data-testid="webhook-secret-value"
              />
              <Button variant="outline" size="sm" onClick={handleCopySecret} aria-label="Copy webhook secret">
                <Copy className="h-4 w-4" />
              </Button>
            </div>
            {secretCopied && (
              <p className="text-sm text-green-600">Copied to clipboard</p>
            )}
            <Alert>
              <AlertDescription>
                Use this secret to verify webhook signatures using HMAC-SHA256.
              </AlertDescription>
            </Alert>
          </div>
          <DialogFooter>
            <Button onClick={() => setShowSecretDialog(false)}>
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </ResponsiveLayout>
  );
}

export default function WebhooksPage() {
  return (
    <PermissionGuard role={['admin', 'super_admin', 'owner']}>
      <WebhooksPageContent />
    </PermissionGuard>
  );
}
