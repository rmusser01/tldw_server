'use client';

import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState, Suspense } from 'react';
import { flushSync } from 'react-dom';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { EmptyState } from '@/components/ui/empty-state';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Checkbox } from '@/components/ui/checkbox';
import { Pagination } from '@/components/ui/pagination';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/toast';
import { api } from '@/lib/api-client';
import { generateIdempotencyKey } from '@/lib/idempotent-command';
import { ApiError } from '@/lib/http';
import { formatDateTime } from '@/lib/format';
import {
  addIncidentActionItem,
  buildPostmortemTimelineMessage,
  ensureIncidentWorkflowState,
  incidentAssignmentWorkflowStateFromIncident,
  mergeIncidentWorkflowWithIncidents,
  removeIncidentActionItem,
  replaceIncidentWorkflowState,
  updateIncidentActionItem,
  upsertIncidentWorkflowState,
  type IncidentWorkflowMap,
} from '@/lib/incident-workflow';
import { useUrlPagination } from '@/lib/use-url-state';
import { usePagedResource } from '@/lib/use-paged-resource';
import type { IncidentItem } from '@/types/incidents';
import { AlertTriangle, Bell, ExternalLink, Mail, RefreshCw, Trash2 } from 'lucide-react';
import { ExportMenu } from '@/components/ui/export-menu';
import { exportIncidents, ExportFormat } from '@/lib/export';
import type {
  IncidentNotifyResponse,
  IncidentWebhookNotifyResponse,
} from '@/types/incidents';

const STATUSES = ['open', 'investigating', 'mitigating', 'resolved'] as const;
const SEVERITIES = ['low', 'medium', 'high', 'critical'] as const;

const formatIncidentDate = (value?: string | null) =>
  formatDateTime(value, { fallback: '—' });

const normalizeSafeRunbookUrl = (value?: string | null): string | null => {
  const normalized = value?.trim();
  if (!normalized) return null;

  try {
    const parsed = new URL(normalized);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return null;
    }
    return parsed.toString();
  } catch {
    return null;
  }
};

const formatMinutes = (minutes: number | null | undefined): string => {
  if (minutes == null) return '—';
  const rounded = Math.round(minutes);
  if (rounded < 1) return '< 1m';
  if (rounded < 60) return `${rounded}m`;
  if (rounded < 1440) {
    const h = Math.floor(rounded / 60);
    const m = rounded % 60;
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
  }
  const d = Math.floor(rounded / 1440);
  const h = Math.floor((rounded % 1440) / 60);
  return h > 0 ? `${d}d ${h}h` : `${d}d`;
};

const formatDuration = (ms: number): string => {
  const hours = Math.floor(ms / 3_600_000);
  const minutes = Math.floor((ms % 3_600_000) / 60_000);
  if (hours > 24) return `${Math.floor(hours / 24)}d ${hours % 24}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
};

type IncidentAssignableUser = {
  id: string;
  label: string;
};

type PendingIncidentWebhookCommand = {
  incidentId: string;
  narrative: string | null;
  idempotencyKey: string;
};

const normalizeAssignableUsers = (payload: unknown): IncidentAssignableUser[] => {
  if (!Array.isArray(payload)) return [];
  return payload
    .map((entry): IncidentAssignableUser | null => {
      if (!entry || typeof entry !== 'object') return null;
      const row = entry as Record<string, unknown>;
      const idValue = row.id ?? row.user_id;
      if (idValue === undefined || idValue === null) return null;
      const id = String(idValue);
      const label = (
        (typeof row.username === 'string' && row.username.trim() && row.username) ||
        (typeof row.email === 'string' && row.email.trim() && row.email) ||
        (typeof row.name === 'string' && row.name.trim() && row.name) ||
        `User ${id}`
      ) as string;
      return { id, label };
    })
    .filter((entry): entry is IncidentAssignableUser => entry !== null);
};

function IncidentsPageContent() {
  const confirm = useConfirm();
  const { success, error: showError } = useToast();
  const { page, pageSize, setPage, setPageSize, resetPagination } = useUrlPagination();

  const [statusFilter, setStatusFilter] = useState('');
  const [severityFilter, setSeverityFilter] = useState('');
  const [tagFilterInput, setTagFilterInput] = useState('');
  const deferredTagFilter = useDeferredValue(tagFilterInput);

  const [title, setTitle] = useState('');
  const [status, setStatus] = useState<typeof STATUSES[number]>('open');
  const [severity, setSeverity] = useState<typeof SEVERITIES[number]>('medium');
  const [summary, setSummary] = useState('');
  const [tags, setTags] = useState('');
  const [runbookUrl, setRunbookUrl] = useState('');
  const [creating, setCreating] = useState(false);

  const [updateNotes, setUpdateNotes] = useState<Record<string, string>>({});
  const [updatingIncidents, setUpdatingIncidents] = useState<Set<string>>(new Set());
  const [optimisticStatuses, setOptimisticStatuses] = useState<Record<string, IncidentItem['status']>>({});
  const [assignableUsers, setAssignableUsers] = useState<IncidentAssignableUser[]>([]);
  const [incidentWorkflow, setIncidentWorkflow] = useState<IncidentWorkflowMap>({});
  const [slaMetrics, setSlaMetrics] = useState<{
    avg_mtta_minutes: number | null;
    avg_mttr_minutes: number | null;
    p95_mttr_minutes: number | null;
    resolved_count: number;
    total_incidents: number;
  } | null>(null);

  // Notify dialog state
  const [notifyIncidentId, setNotifyIncidentId] = useState<string | null>(null);
  const [notifyRecipients, setNotifyRecipients] = useState('');
  const [notifyMessage, setNotifyMessage] = useState('');
  const [notifying, setNotifying] = useState(false);
  const [notifyResults, setNotifyResults] = useState<IncidentNotifyResponse | null>(null);
  const [webhookNotifyIncident, setWebhookNotifyIncident] = useState<IncidentItem | null>(null);
  const [webhookNarrative, setWebhookNarrative] = useState('');
  const [webhookNarrativeReviewed, setWebhookNarrativeReviewed] = useState(false);
  const [webhookNotifying, setWebhookNotifying] = useState(false);
  const [webhookNotifyResult, setWebhookNotifyResult] = useState<IncidentWebhookNotifyResponse | null>(null);
  const [webhookNotifyMessage, setWebhookNotifyMessage] = useState('');
  const [webhookRetryAvailable, setWebhookRetryAvailable] = useState(false);
  const pendingWebhookCommandRef = useRef<PendingIncidentWebhookCommand | null>(null);
  const activeWebhookCommandRef = useRef<symbol | null>(null);
  const webhookCommandGenerationRef = useRef(0);

  const params = useMemo(() => {
    const offset = Math.max(0, (page - 1) * pageSize);
    const payload: Record<string, string> = {
      limit: String(pageSize),
      offset: String(offset),
    };
    if (statusFilter) payload.status = statusFilter;
    if (severityFilter) payload.severity = severityFilter;
    if (deferredTagFilter) payload.tag = deferredTagFilter;
    return payload;
  }, [deferredTagFilter, page, pageSize, severityFilter, statusFilter]);

  const loadIncidents = useCallback(({ signal }: { signal?: AbortSignal } = {}) =>
    api.getIncidents(params, signal ? { signal } : undefined), [params]);

  const {
    items: incidents,
    total,
    loading,
    error,
    reload,
  } = usePagedResource<IncidentItem>({
    load: loadIncidents,
    defaultError: 'Failed to load incidents',
  });

  useEffect(() => {
    resetPagination();
  }, [deferredTagFilter, resetPagination]);

  useEffect(() => {
    setIncidentWorkflow((prev) => mergeIncidentWorkflowWithIncidents(incidents, prev));
  }, [incidents]);

  useEffect(() => {
    api.getIncidentSlaMetrics().then(setSlaMetrics).catch(() => {});
  }, [incidents]);

  useEffect(() => {
    let active = true;
    const loadAssignableUsers = async () => {
      try {
        const payload = await api.getUsers({ limit: '100', admin_capable: 'true' });
        if (!active) return;
        setAssignableUsers(normalizeAssignableUsers(payload));
      } catch (err: unknown) {
        if (!active) return;
        const message = err instanceof Error && err.message ? err.message : 'Failed to load users';
        showError(message);
        setAssignableUsers([]);
      }
    };
    void loadAssignableUsers();
    return () => {
      active = false;
    };
  }, [showError]);

  useEffect(() => {
    const clearWebhookCommand = () => {
      webhookCommandGenerationRef.current += 1;
      pendingWebhookCommandRef.current = null;
      activeWebhookCommandRef.current = null;
      flushSync(() => {
        setWebhookNotifyIncident(null);
        setWebhookNarrative('');
        setWebhookNarrativeReviewed(false);
        setWebhookNotifying(false);
        setWebhookNotifyResult(null);
        setWebhookNotifyMessage('');
        setWebhookRetryAvailable(false);
      });
    };
    const handlePageShow = (event: PageTransitionEvent) => {
      if (event.persisted) clearWebhookCommand();
    };
    window.addEventListener('pagehide', clearWebhookCommand);
    window.addEventListener('pageshow', handlePageShow);
    return () => {
      window.removeEventListener('pagehide', clearWebhookCommand);
      window.removeEventListener('pageshow', handlePageShow);
      webhookCommandGenerationRef.current += 1;
      pendingWebhookCommandRef.current = null;
      activeWebhookCommandRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!webhookRetryAvailable) return;
    const preserveAmbiguousCommand = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', preserveAmbiguousCommand);
    return () => window.removeEventListener('beforeunload', preserveAmbiguousCommand);
  }, [webhookRetryAvailable]);

  const setIncidentUpdating = useCallback((incidentId: string, isUpdating: boolean) => {
    setUpdatingIncidents((prev) => {
      const next = new Set(prev);
      if (isUpdating) {
        next.add(incidentId);
      } else {
        next.delete(incidentId);
      }
      return next;
    });
  }, []);

  const updateIncidentWorkflow = useCallback((
    incidentId: string,
    nextState: Parameters<typeof upsertIncidentWorkflowState>[2]
  ) => {
    setIncidentWorkflow((prev) => upsertIncidentWorkflowState(prev, incidentId, nextState));
  }, []);

  const getAssigneeLabel = useCallback((userId?: string, fallbackLabel?: string) => {
    if (!userId) return 'Unassigned';
    return (
      assignableUsers.find((user) => user.id === userId)?.label
      ?? fallbackLabel
      ?? `User ${userId}`
    );
  }, [assignableUsers]);

  const handleCreateIncident = async () => {
    if (!title.trim()) {
      showError('Incident title is required');
      return;
    }
    const normalizedRunbookUrl = runbookUrl.trim() ? normalizeSafeRunbookUrl(runbookUrl) : null;
    if (runbookUrl.trim() && !normalizedRunbookUrl) {
      showError('Runbook URL must start with http:// or https://');
      return;
    }
    try {
      setCreating(true);
      await api.createIncident({
        title,
        status,
        severity,
        summary,
        tags: tags ? tags.split(',').map((item) => item.trim()).filter(Boolean) : [],
        ...(normalizedRunbookUrl ? { runbook_url: normalizedRunbookUrl } : {}),
      });
      success('Incident created');
      setTitle('');
      setSummary('');
      setTags('');
      setRunbookUrl('');
      resetPagination();
      await reload();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to create incident';
      showError(message);
    } finally {
      setCreating(false);
    }
  };

  const handleStatusChange = async (incidentId: string, nextStatus: IncidentItem['status']) => {
    // Optimistic update: show the new status immediately
    setOptimisticStatuses((prev) => ({ ...prev, [incidentId]: nextStatus }));

    try {
      setIncidentUpdating(incidentId, true);
      await api.updateIncident(incidentId, {
        status: nextStatus,
        update_message: `Status changed to ${nextStatus}`,
      });
      // Clear optimistic override before reload (reload will bring fresh data)
      setOptimisticStatuses((prev) => {
        const next = { ...prev };
        delete next[incidentId];
        return next;
      });
      await reload();
    } catch (err: unknown) {
      // Revert on error
      setOptimisticStatuses((prev) => {
        const next = { ...prev };
        delete next[incidentId];
        return next;
      });
      const message = err instanceof Error && err.message ? err.message : 'Failed to update status';
      showError(message);
    } finally {
      setIncidentUpdating(incidentId, false);
    }
  };

  const handleSeverityChange = async (incidentId: string, nextSeverity: IncidentItem['severity']) => {
    try {
      setIncidentUpdating(incidentId, true);
      await api.updateIncident(incidentId, {
        severity: nextSeverity,
        update_message: `Severity set to ${nextSeverity}`,
      });
      await reload();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to update severity';
      showError(message);
    } finally {
      setIncidentUpdating(incidentId, false);
    }
  };

  const handleAddUpdate = async (incidentId: string) => {
    const note = (updateNotes[incidentId] || '').trim();
    if (!note) {
      showError('Update message is required');
      return;
    }
    try {
      setIncidentUpdating(incidentId, true);
      await api.addIncidentEvent(incidentId, { message: note });
      setUpdateNotes((prev) => ({ ...prev, [incidentId]: '' }));
      await reload();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to add update';
      showError(message);
    } finally {
      setIncidentUpdating(incidentId, false);
    }
  };

  const handleAssignmentChange = async (incident: IncidentItem, assignedTo: string) => {
    const assigneeLabel = getAssigneeLabel(assignedTo || undefined);
    try {
      setIncidentUpdating(incident.id, true);
      const updated = await api.updateIncident(incident.id, {
        assigned_to_user_id: assignedTo ? Number(assignedTo) : null,
        update_message: assignedTo ? `Assigned to ${assigneeLabel}` : 'Assignment cleared',
      });
      setIncidentWorkflow((prev) => upsertIncidentWorkflowState(
        prev,
        incident.id,
        incidentAssignmentWorkflowStateFromIncident(updated),
      ));
      await reload();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to update assignment';
      showError(message);
      setIncidentWorkflow((prev) => upsertIncidentWorkflowState(
        prev,
        incident.id,
        incidentAssignmentWorkflowStateFromIncident(incident),
      ));
    } finally {
      setIncidentUpdating(incident.id, false);
    }
  };

  const handleSavePostmortem = async (incident: IncidentItem) => {
    const state = ensureIncidentWorkflowState(incidentWorkflow, incident);
    try {
      setIncidentUpdating(incident.id, true);
      const updated = await api.updateIncident(incident.id, {
        root_cause: state.rootCause.trim() || null,
        impact: state.impact.trim() || null,
        runbook_url: state.runbookUrl?.trim() || null,
        action_items: state.actionItems,
        update_message: buildPostmortemTimelineMessage(state),
      });
      setIncidentWorkflow((prev) => replaceIncidentWorkflowState(prev, updated));
      success('Post-mortem saved');
      await reload();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to save post-mortem';
      showError(message);
    } finally {
      setIncidentUpdating(incident.id, false);
    }
  };

  const handleDeleteIncident = async (incidentId: string) => {
    const confirmed = await confirm({
      title: 'Delete incident?',
      message: 'This removes the incident from the timeline.',
      confirmText: 'Delete',
      variant: 'danger',
    });
    if (!confirmed) return;
    try {
      setIncidentUpdating(incidentId, true);
      await api.deleteIncident(incidentId);
      success('Incident deleted');
      await reload();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to delete incident';
      showError(message);
    } finally {
      setIncidentUpdating(incidentId, false);
    }
  };

  const handleNotifyStakeholders = async () => {
    if (!notifyIncidentId) return;
    const emails = notifyRecipients
      .split(',')
      .map((e) => e.trim())
      .filter(Boolean);
    if (emails.length === 0) {
      showError('At least one recipient email is required');
      return;
    }
    try {
      setNotifying(true);
      const result = await api.notifyIncidentStakeholders(notifyIncidentId, {
        recipients: emails,
        ...(notifyMessage.trim() ? { message: notifyMessage.trim() } : {}),
      });
      setNotifyResults(result);
      const sentCount = result.notifications.filter((n) => n.status === 'sent').length;
      success(`Notification sent to ${sentCount}/${result.notifications.length} recipient(s)`);
      await reload();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to send notifications';
      showError(message);
    } finally {
      setNotifying(false);
    }
  };

  const openNotifyDialog = (incidentId: string) => {
    setNotifyIncidentId(incidentId);
    setNotifyRecipients('');
    setNotifyMessage('');
    setNotifyResults(null);
  };

  const closeNotifyDialog = () => {
    setNotifyIncidentId(null);
    setNotifyRecipients('');
    setNotifyMessage('');
    setNotifyResults(null);
  };

  const openWebhookNotifyDialog = (incident: IncidentItem) => {
    webhookCommandGenerationRef.current += 1;
    pendingWebhookCommandRef.current = null;
    activeWebhookCommandRef.current = null;
    setWebhookNotifyIncident(incident);
    setWebhookNarrative('');
    setWebhookNarrativeReviewed(false);
    setWebhookNotifyResult(null);
    setWebhookNotifyMessage('');
    setWebhookRetryAvailable(false);
  };

  const closeWebhookNotifyDialog = () => {
    if (webhookNotifying || webhookRetryAvailable) return;
    webhookCommandGenerationRef.current += 1;
    pendingWebhookCommandRef.current = null;
    activeWebhookCommandRef.current = null;
    setWebhookNotifyIncident(null);
    setWebhookNarrative('');
    setWebhookNarrativeReviewed(false);
    setWebhookNotifyResult(null);
    setWebhookNotifyMessage('');
    setWebhookRetryAvailable(false);
  };

  const runWebhookNotifyCommand = async (pending: PendingIncidentWebhookCommand) => {
    const generation = webhookCommandGenerationRef.current;
    const token = Symbol('incident-webhook-command');
    activeWebhookCommandRef.current = token;
    pendingWebhookCommandRef.current = pending;
    setWebhookNotifying(true);
    setWebhookRetryAvailable(false);
    setWebhookNotifyMessage('');
    try {
      const result = await api.notifyIncidentWebhooks(
        pending.incidentId,
        { narrative: pending.narrative },
        pending.idempotencyKey,
      );
      if (
        webhookCommandGenerationRef.current !== generation
        || activeWebhookCommandRef.current !== token
      ) return;
      pendingWebhookCommandRef.current = null;
      setWebhookNotifyResult(result);
      setWebhookNotifyMessage(
        result.replayed
          ? 'Webhook command accepted from the original idempotent submission.'
          : 'Webhook command accepted for durable delivery.',
      );
      success('Webhook notification command accepted');
    } catch (err: unknown) {
      if (
        webhookCommandGenerationRef.current !== generation
        || activeWebhookCommandRef.current !== token
      ) return;
      const ambiguous = err instanceof TypeError
        || (err instanceof ApiError && [502, 504].includes(err.status))
        || (err instanceof Error && ['AbortError', 'NetworkError'].includes(err.name));
      if (ambiguous) {
        setWebhookRetryAvailable(true);
        setWebhookNotifyMessage(
          'The command result is unknown. Retry the same command to recover its durable acceptance.',
        );
      } else {
        pendingWebhookCommandRef.current = null;
        const message = err instanceof Error && err.message
          ? err.message
          : 'Failed to notify webhook receivers';
        showError(message);
      }
    } finally {
      if (
        webhookCommandGenerationRef.current === generation
        && activeWebhookCommandRef.current === token
      ) {
        activeWebhookCommandRef.current = null;
        setWebhookNotifying(false);
      }
    }
  };

  const handleNotifyIncidentWebhooks = async () => {
    if (!webhookNotifyIncident || !webhookNarrativeReviewed) return;
    const narrative = webhookNarrative.trim() || null;
    await runWebhookNotifyCommand({
      incidentId: webhookNotifyIncident.id,
      narrative,
      idempotencyKey: generateIdempotencyKey(),
    });
  };

  const retryIncidentWebhookCommand = async () => {
    const pending = pendingWebhookCommandRef.current;
    if (pending) await runWebhookNotifyCommand(pending);
  };

  const webhookNarrativePayload = webhookNarrative.trim() || null;

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="flex flex-col gap-6 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">Incidents</h1>
              <p className="text-muted-foreground">Track operational events and updates.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <ExportMenu
                onExport={(format: ExportFormat) => exportIncidents(incidents, format)}
                disabled={incidents.length === 0}
              />
              <Button
                variant="outline"
                onClick={() => {
                  void reload();
                }}
                disabled={loading}
              >
                <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Create Incident</CardTitle>
              <CardDescription>Log a new operational event.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <div className="space-y-1 md:col-span-2">
                <Label htmlFor="incident-title">Title</Label>
                <Input
                  id="incident-title"
                  placeholder="Brief incident title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="incident-status">Status</Label>
                <Select
                  id="incident-status"
                  value={status}
                  onChange={(e) => setStatus(e.target.value as IncidentItem['status'])}
                >
                  {STATUSES.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="space-y-1">
                <Label htmlFor="incident-severity">Severity</Label>
                <Select
                  id="incident-severity"
                  value={severity}
                  onChange={(e) => setSeverity(e.target.value as IncidentItem['severity'])}
                >
                  {SEVERITIES.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="space-y-1 md:col-span-2">
                <Label htmlFor="incident-summary">Summary</Label>
                <Input
                  id="incident-summary"
                  placeholder="What happened?"
                  value={summary}
                  onChange={(e) => setSummary(e.target.value)}
                />
              </div>
              <div className="space-y-1 md:col-span-2">
                <Label htmlFor="incident-tags">Tags</Label>
                <Input
                  id="incident-tags"
                  placeholder="comma separated"
                  value={tags}
                  onChange={(e) => setTags(e.target.value)}
                />
              </div>
              <div className="space-y-1 md:col-span-2">
                <Label htmlFor="incident-runbook-url">Runbook URL (optional)</Label>
                <Input
                  id="incident-runbook-url"
                  type="url"
                  placeholder="https://wiki.example.com/runbooks/..."
                  value={runbookUrl}
                  onChange={(e) => setRunbookUrl(e.target.value)}
                />
              </div>
              <div>
                <Button
                  onClick={() => {
                    void handleCreateIncident();
                  }}
                  disabled={creating}
                  loading={creating}
                  loadingText="Creating..."
                >
                  Create Incident
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Filters</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-3">
              <div className="space-y-1">
                <Label htmlFor="filter-status">Status</Label>
                <Select
                  id="filter-status"
                  value={statusFilter}
                  onChange={(e) => {
                    setStatusFilter(e.target.value);
                    resetPagination();
                  }}
                >
                  <option value="">All</option>
                  {STATUSES.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="space-y-1">
                <Label htmlFor="filter-severity">Severity</Label>
                <Select
                  id="filter-severity"
                  value={severityFilter}
                  onChange={(e) => {
                    setSeverityFilter(e.target.value);
                    resetPagination();
                  }}
                >
                  <option value="">All</option>
                  {SEVERITIES.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="space-y-1">
                <Label htmlFor="filter-tag">Tag</Label>
                <Input
                  id="filter-tag"
                  placeholder="tag"
                  value={tagFilterInput}
                  onChange={(e) => {
                    setTagFilterInput(e.target.value);
                  }}
                />
              </div>
            </CardContent>
          </Card>

          {slaMetrics && (
            <div className="grid gap-4 md:grid-cols-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Avg. Time to Acknowledge</CardDescription>
                  <CardTitle className="text-lg">
                    {formatMinutes(slaMetrics.avg_mtta_minutes)}
                  </CardTitle>
                </CardHeader>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Avg. Time to Resolve</CardDescription>
                  <CardTitle className="text-lg">
                    {formatMinutes(slaMetrics.avg_mttr_minutes)}
                  </CardTitle>
                </CardHeader>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>P95 Resolution Time</CardDescription>
                  <CardTitle className="text-lg">
                    {formatMinutes(slaMetrics.p95_mttr_minutes)}
                  </CardTitle>
                </CardHeader>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardDescription>Resolved</CardDescription>
                  <CardTitle className="text-lg">
                    {slaMetrics.resolved_count} / {slaMetrics.total_incidents}
                  </CardTitle>
                </CardHeader>
              </Card>
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {loading ? (
            <div className="py-10 text-center text-muted-foreground" role="status" aria-live="polite">Loading incidents...</div>
          ) : incidents.length === 0 ? (
            <EmptyState
              icon={AlertTriangle}
              title="No incidents found."
              description="Try changing filters or create a new incident."
              actions={[
                {
                  label: 'Clear filters',
                  onClick: () => {
                    setStatusFilter('');
                    setSeverityFilter('');
                    setTagFilterInput('');
                    resetPagination();
                  },
                },
              ]}
              className="py-10"
            />
          ) : (
            <div className="grid gap-4">
              {incidents.map((incident) => {
                const isUpdating = updatingIncidents.has(incident.id);
                const displayStatus = optimisticStatuses[incident.id] ?? incident.status;
                const workflowState = ensureIncidentWorkflowState(incidentWorkflow, incident);
                const currentAssigneeId = workflowState.assignedTo ?? '';
                const currentAssigneeInOptions = currentAssigneeId
                  ? assignableUsers.some((user) => user.id === currentAssigneeId)
                  : true;
                const currentAssigneeLabel = currentAssigneeId
                  ? getAssigneeLabel(currentAssigneeId, workflowState.assignedToLabel)
                  : 'Unassigned';
                return (
                  <Card key={incident.id}>
                  <CardHeader className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        {incident.title}
                        {workflowState.runbookUrl && (
                          <a href={workflowState.runbookUrl} target="_blank" rel="noopener noreferrer"
                            className="text-primary hover:underline text-sm font-normal" title="Open runbook">
                            <ExternalLink className="h-3.5 w-3.5 inline" />
                          </a>
                        )}
                        <Badge variant={displayStatus === 'resolved' ? 'secondary' : 'outline'}>
                          {displayStatus}
                        </Badge>
                        <Badge variant={incident.severity === 'critical' ? 'destructive' : 'outline'}>
                          {incident.severity}
                        </Badge>
                      </CardTitle>
                      <CardDescription>
                        Updated {formatIncidentDate(incident.updated_at)} · Created {formatIncidentDate(incident.created_at)}
                        {incident.mtta_minutes != null && (
                          <span className="ml-2 text-xs text-muted-foreground">
                            MTTA: {formatMinutes(incident.mtta_minutes)}
                          </span>
                        )}
                        {incident.mttr_minutes != null && (
                          <span className="ml-2 text-xs text-muted-foreground">
                            MTTR: {formatMinutes(incident.mttr_minutes)}
                          </span>
                        )}
                        {incident.time_to_acknowledge_seconds != null && (
                          <span className="ml-2">· TTA: {formatDuration(incident.time_to_acknowledge_seconds * 1000)}</span>
                        )}
                        {incident.time_to_resolve_seconds != null && (
                          <span className="ml-2">· TTR: {formatDuration(incident.time_to_resolve_seconds * 1000)}</span>
                        )}
                      </CardDescription>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => openNotifyDialog(incident.id)}
                        aria-label={`Notify stakeholders for incident ${incident.id}`}
                        title="Notify stakeholders"
                        disabled={isUpdating}
                        data-testid={`incident-notify-${incident.id}`}
                      >
                        <Mail className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => openWebhookNotifyDialog(incident)}
                        aria-label={`Notify webhook receivers for incident ${incident.id}`}
                        title="Queue webhook notification"
                        disabled={isUpdating}
                        data-testid={`incident-webhook-notify-${incident.id}`}
                      >
                        <Bell className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => {
                          void handleDeleteIncident(incident.id);
                        }}
                        aria-label={`Delete incident ${incident.id}`}
                        title={`Delete incident ${incident.id}`}
                        disabled={isUpdating}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="grid gap-4">
                    <div className="text-sm text-muted-foreground">
                      {incident.summary || 'No summary provided.'}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {(incident.tags || []).map((tag, index) => (
                        <Badge key={`${tag}-${index}`} variant="outline">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                    {normalizeSafeRunbookUrl(incident.runbook_url) && (
                      <div className="flex items-center gap-1 text-sm">
                        <ExternalLink className="h-3.5 w-3.5 text-muted-foreground" />
                        <a
                          href={normalizeSafeRunbookUrl(incident.runbook_url) ?? undefined}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-primary hover:underline"
                        >
                          Runbook
                        </a>
                      </div>
                    )}
                    <div className="grid gap-3 md:grid-cols-4">
                      <div className="space-y-1">
                        <Label htmlFor={`status-${incident.id}`}>Status</Label>
                        <Select
                          id={`status-${incident.id}`}
                          value={displayStatus}
                          onChange={(e) => {
                            void handleStatusChange(incident.id, e.target.value as IncidentItem['status']);
                          }}
                          disabled={isUpdating}
                        >
                          {STATUSES.map((value) => (
                            <option key={value} value={value}>
                              {value}
                            </option>
                          ))}
                        </Select>
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor={`severity-${incident.id}`}>Severity</Label>
                        <Select
                          id={`severity-${incident.id}`}
                          value={incident.severity}
                          onChange={(e) => {
                            void handleSeverityChange(incident.id, e.target.value as IncidentItem['severity']);
                          }}
                          disabled={isUpdating}
                        >
                          {SEVERITIES.map((value) => (
                            <option key={value} value={value}>
                              {value}
                            </option>
                          ))}
                        </Select>
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor={`resolved-${incident.id}`}>Resolved</Label>
                        <div id={`resolved-${incident.id}`} className="text-sm text-muted-foreground">
                          {incident.resolved_at ? formatIncidentDate(incident.resolved_at) : 'Not resolved'}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor={`assigned-${incident.id}`}>Assigned To</Label>
                        <Select
                          id={`assigned-${incident.id}`}
                          value={currentAssigneeId}
                          onChange={(event) => {
                            void handleAssignmentChange(incident, event.target.value);
                          }}
                          disabled={isUpdating}
                          data-testid={`incident-assigned-to-${incident.id}`}
                        >
                          <option value="">Unassigned</option>
                          {!currentAssigneeInOptions && currentAssigneeId ? (
                            <option value={currentAssigneeId}>{currentAssigneeLabel}</option>
                          ) : null}
                          {assignableUsers.map((user) => (
                            <option key={user.id} value={user.id}>
                              {user.label}
                            </option>
                          ))}
                        </Select>
                      </div>
                    </div>
                    {displayStatus === 'resolved' && (
                      <div
                        className="space-y-3 rounded-md border p-3"
                        data-testid={`incident-postmortem-${incident.id}`}
                      >
                        <div className="text-sm font-medium">Post-mortem</div>
                        <div className="grid gap-3 md:grid-cols-2">
                          <div className="space-y-1">
                            <Label htmlFor={`root-cause-${incident.id}`}>Root Cause</Label>
                            <textarea
                              id={`root-cause-${incident.id}`}
                              className="flex min-h-24 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                              placeholder="Describe the primary root cause"
                              value={workflowState.rootCause}
                              onChange={(event) => {
                                updateIncidentWorkflow(incident.id, { rootCause: event.target.value });
                              }}
                              data-testid={`incident-root-cause-${incident.id}`}
                            />
                          </div>
                          <div className="space-y-1">
                            <Label htmlFor={`impact-${incident.id}`}>Impact</Label>
                            <textarea
                              id={`impact-${incident.id}`}
                              className="flex min-h-24 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                              placeholder="Describe user/business impact"
                              value={workflowState.impact}
                              onChange={(event) => {
                                updateIncidentWorkflow(incident.id, { impact: event.target.value });
                              }}
                              data-testid={`incident-impact-${incident.id}`}
                            />
                          </div>
                        </div>
                        <div className="space-y-1">
                          <Label htmlFor={`runbook-${incident.id}`}>Runbook URL</Label>
                          <Input
                            id={`runbook-${incident.id}`}
                            type="url"
                            placeholder="https://wiki.example.com/runbooks/..."
                            value={workflowState.runbookUrl || ''}
                            onChange={(e) => updateIncidentWorkflow(incident.id, { runbookUrl: e.target.value })}
                          />
                        </div>
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <Label>Action Items</Label>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                setIncidentWorkflow((prev) => addIncidentActionItem(prev, incident.id));
                              }}
                            >
                              Add Action Item
                            </Button>
                          </div>
                          {workflowState.actionItems.length === 0 ? (
                            <p className="text-xs text-muted-foreground">
                              No action items added yet.
                            </p>
                          ) : (
                            workflowState.actionItems.map((item) => (
                              <div key={item.id} className="grid gap-2 md:grid-cols-[auto_1fr_auto] md:items-center">
                                <Checkbox
                                  checked={item.done}
                                  onCheckedChange={(checked) => {
                                    setIncidentWorkflow((prev) =>
                                      updateIncidentActionItem(prev, incident.id, item.id, {
                                        done: Boolean(checked),
                                      })
                                    );
                                  }}
                                  aria-label={`Toggle action item ${item.id}`}
                                />
                                <Input
                                  value={item.text}
                                  onChange={(event) => {
                                    setIncidentWorkflow((prev) =>
                                      updateIncidentActionItem(prev, incident.id, item.id, {
                                        text: event.target.value,
                                      })
                                    );
                                  }}
                                  placeholder="Describe follow-up action"
                                  data-testid={`incident-action-item-${incident.id}-${item.id}`}
                                />
                                <Button
                                  type="button"
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => {
                                    setIncidentWorkflow((prev) =>
                                      removeIncidentActionItem(prev, incident.id, item.id)
                                    );
                                  }}
                                >
                                  Remove
                                </Button>
                              </div>
                            ))
                          )}
                        </div>
                        <div className="flex justify-end">
                          <Button
                            type="button"
                            onClick={() => {
                              void handleSavePostmortem(incident);
                            }}
                            loading={isUpdating}
                            loadingText="Saving..."
                            data-testid={`incident-save-postmortem-${incident.id}`}
                          >
                            Save Post-mortem
                          </Button>
                        </div>
                      </div>
                    )}
                    <div className="grid gap-2 md:grid-cols-[1fr_auto]">
                      <Input
                        placeholder="Add update..."
                        value={updateNotes[incident.id] || ''}
                        onChange={(e) =>
                          setUpdateNotes((prev) => ({ ...prev, [incident.id]: e.target.value }))
                        }
                        disabled={isUpdating}
                      />
                      <Button
                        onClick={() => {
                          void handleAddUpdate(incident.id);
                        }}
                        disabled={isUpdating}
                        loading={isUpdating}
                        loadingText="Updating..."
                      >
                        Add Update
                      </Button>
                    </div>
                    <details className="text-sm text-muted-foreground" open={displayStatus !== 'resolved'}>
                      <summary className="cursor-pointer">Timeline ({incident.timeline?.length || 0})</summary>
                      <div className="mt-2 space-y-2">
                        {(incident.timeline || []).map((event) => (
                          <div key={event.id} className="rounded-md border p-2">
                            <div className="font-medium text-foreground">{event.message}</div>
                            <div className="text-xs text-muted-foreground">
                              {formatIncidentDate(event.created_at)} {event.actor ? `· ${event.actor}` : ''}
                            </div>
                          </div>
                        ))}
                      </div>
                    </details>
                  </CardContent>
                  </Card>
                );
              })}
            </div>
          )}

          <Pagination
            currentPage={page}
            totalPages={totalPages}
            totalItems={total}
            pageSize={pageSize}
            onPageChange={setPage}
            onPageSizeChange={setPageSize}
          />

          {/* Notify Stakeholders Dialog */}
          <Dialog open={!!notifyIncidentId} onOpenChange={(open) => { if (!open) closeNotifyDialog(); }}>
            <DialogContent data-testid="notify-dialog-overlay">
              <DialogHeader>
                <DialogTitle>Notify Stakeholders</DialogTitle>
                <DialogDescription>
                  Send an email update to the selected incident stakeholders.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div className="space-y-1">
                  <Label htmlFor="notify-recipients">Recipients (comma-separated emails)</Label>
                  <Input
                    id="notify-recipients"
                    data-testid="notify-recipients-input"
                    placeholder="alice@example.com, bob@example.com"
                    value={notifyRecipients}
                    onChange={(e) => setNotifyRecipients(e.target.value)}
                    disabled={notifying}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="notify-message">Message (optional)</Label>
                  <textarea
                    id="notify-message"
                    data-testid="notify-message-input"
                    className="flex min-h-20 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    placeholder="Optional custom message for stakeholders"
                    value={notifyMessage}
                    onChange={(e) => setNotifyMessage(e.target.value)}
                    disabled={notifying}
                  />
                </div>
                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={closeNotifyDialog} disabled={notifying}>
                    Cancel
                  </Button>
                  <Button
                    onClick={() => {
                      void handleNotifyStakeholders();
                    }}
                    disabled={notifying}
                    loading={notifying}
                    loadingText="Sending..."
                    data-testid="notify-send-button"
                  >
                    <Mail className="mr-2 h-4 w-4" />
                    Send Notification
                  </Button>
                </div>
                {notifyResults && (
                  <div className="space-y-2 rounded-md border p-3" data-testid="notify-results">
                    <div className="text-sm font-medium">Delivery Results</div>
                    {notifyResults.notifications.map((result, idx) => (
                      <div key={idx} className="flex items-center justify-between text-sm">
                        <span className="truncate mr-2">{result.email}</span>
                        <Badge variant={result.status === 'sent' ? 'secondary' : 'destructive'}>
                          {result.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </DialogContent>
          </Dialog>

          <Dialog
            open={!!webhookNotifyIncident}
            onOpenChange={(open) => { if (!open) closeWebhookNotifyDialog(); }}
          >
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Notify webhook receivers</DialogTitle>
                <DialogDescription>
                  Review the exact public event data before creating a durable notification.
                </DialogDescription>
              </DialogHeader>
              {webhookNotifyIncident ? (
                <div className="space-y-4">
                  <div className="space-y-1">
                    <Label htmlFor="incident-webhook-narrative">Receiver narrative (optional)</Label>
                    <textarea
                      id="incident-webhook-narrative"
                      className="flex min-h-24 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={webhookNarrative}
                      onChange={(event) => {
                        setWebhookNarrative(event.target.value);
                        setWebhookNarrativeReviewed(false);
                      }}
                      maxLength={4096}
                      disabled={webhookNotifying || webhookNotifyResult !== null}
                    />
                    <p className="text-xs text-muted-foreground">
                      This text is delivered to configured receivers. Do not include secrets.
                    </p>
                  </div>
                  <section className="space-y-2" aria-label="Outbound event preview">
                    <div className="flex flex-wrap items-baseline justify-between gap-2">
                      <h3 className="text-sm font-semibold">Outbound event preview</h3>
                      <span className="text-xs text-muted-foreground">
                        Envelope timestamp assigned at command acceptance
                      </span>
                    </div>
                    <dl
                      className="grid grid-cols-[minmax(8rem,auto)_1fr] gap-x-4 gap-y-2 border-y py-3 text-sm"
                      data-testid="incident-webhook-preview"
                    >
                      <dt className="text-muted-foreground">type</dt>
                      <dd className="font-mono">incident.notify</dd>
                      <dt className="text-muted-foreground">incident_id</dt>
                      <dd className="break-all font-mono">{webhookNotifyIncident.id}</dd>
                      <dt className="text-muted-foreground">state</dt>
                      <dd className="font-mono">{webhookNotifyIncident.status}</dd>
                      <dt className="text-muted-foreground">severity</dt>
                      <dd className="font-mono">{webhookNotifyIncident.severity}</dd>
                      <dt className="text-muted-foreground">resource_version</dt>
                      <dd className="font-mono">{webhookNotifyIncident.version}</dd>
                      <dt className="text-muted-foreground">created_at</dt>
                      <dd className="break-all font-mono">{webhookNotifyIncident.created_at}</dd>
                      <dt className="text-muted-foreground">updated_at</dt>
                      <dd className="break-all font-mono">{webhookNotifyIncident.updated_at}</dd>
                      <dt className="text-muted-foreground">resolved_at</dt>
                      <dd className="break-all font-mono">
                        {webhookNotifyIncident.resolved_at ?? 'null'}
                      </dd>
                      <dt className="text-muted-foreground">narrative</dt>
                      <dd className="break-all whitespace-pre-wrap font-mono">
                        {JSON.stringify(webhookNarrativePayload)}
                      </dd>
                    </dl>
                  </section>
                  <div className="flex items-start gap-2">
                    <Checkbox
                      id="incident-webhook-reviewed"
                      checked={webhookNarrativeReviewed}
                      onCheckedChange={(checked) => setWebhookNarrativeReviewed(Boolean(checked))}
                      disabled={webhookNotifying || webhookNotifyResult !== null}
                    />
                    <Label htmlFor="incident-webhook-reviewed" className="font-normal leading-5">
                      I reviewed this narrative and the incident details above.
                    </Label>
                  </div>
                  {webhookNotifyMessage ? (
                    <Alert variant={webhookRetryAvailable ? 'destructive' : 'default'}>
                      <AlertDescription className="space-y-1">
                        <p>{webhookNotifyMessage}</p>
                        {webhookNotifyResult ? (
                          <p className="break-all font-mono text-xs">
                            Event {webhookNotifyResult.event_id}; command {webhookNotifyResult.command_id}
                          </p>
                        ) : null}
                      </AlertDescription>
                    </Alert>
                  ) : null}
                  <div className="flex flex-wrap justify-end gap-2">
                    <Button
                      variant="outline"
                      onClick={closeWebhookNotifyDialog}
                      disabled={webhookNotifying || webhookRetryAvailable}
                    >
                      {webhookNotifyResult ? 'Close' : 'Cancel'}
                    </Button>
                    {webhookRetryAvailable ? (
                      <Button
                        onClick={() => { void retryIncidentWebhookCommand(); }}
                        loading={webhookNotifying}
                        loadingText="Retrying..."
                      >
                        Retry same command
                      </Button>
                    ) : webhookNotifyResult === null ? (
                      <Button
                        onClick={() => { void handleNotifyIncidentWebhooks(); }}
                        disabled={!webhookNarrativeReviewed || webhookNotifying}
                        loading={webhookNotifying}
                        loadingText="Sending..."
                      >
                        <Bell className="mr-2 h-4 w-4" />
                        Send webhook notification
                      </Button>
                    ) : null}
                  </div>
                </div>
              ) : null}
            </DialogContent>
          </Dialog>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

// Wrap with Suspense for useSearchParams
export default function IncidentsPage() {
  return (
    <Suspense fallback={
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="flex flex-col gap-6 p-6">
            <div className="mb-8">
              <div className="h-8 w-32 bg-muted rounded animate-pulse mb-2" />
              <div className="h-4 w-64 bg-muted rounded animate-pulse" />
            </div>
            <div className="h-96 bg-muted rounded animate-pulse" />
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    }>
      <IncidentsPageContent />
    </Suspense>
  );
}
