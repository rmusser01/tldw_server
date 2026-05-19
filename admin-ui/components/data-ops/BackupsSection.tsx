'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Pagination } from '@/components/ui/pagination';
import { useToast } from '@/components/ui/toast';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { api } from '@/lib/api-client';
import { formatBytes, formatDateTime, formatDuration } from '@/lib/format';
import { useUrlPagination } from '@/lib/use-url-state';
import { usePagedResource, type LoadOptions } from '@/lib/use-paged-resource';
import type { BackupItem, BackupScheduleItem, User } from '@/types';
import {
  AlertCircle,
  CheckCircle2,
  Database,
  Loader2,
  Pause,
  Pencil,
  Play,
  Trash2,
  XCircle,
} from 'lucide-react';
import { Field } from '@/components/data-ops/Field';
import { logger } from '@/lib/logger';

type BackupsSectionProps = {
  refreshSignal: number;
};

type BackupsTab = 'backups' | 'schedule';
type BackupHistoryStatusFilter = 'all' | 'success' | 'failed' | 'in_progress';
type BackupScheduleFrequency = 'daily' | 'weekly' | 'monthly';

type BackupListItem = BackupItem & {
  duration_seconds?: number | null;
  duration_ms?: number | null;
  started_at?: string | null;
  completed_at?: string | null;
  error_message?: string | null;
  failure_reason?: string | null;
  status_message?: string | null;
};

type BackupTrendPoint = {
  id: string;
  createdAt: string;
  createdAtMs: number;
  sizeBytes: number;
};

type BackupDatasetTrend = {
  dataset: string;
  points: BackupTrendPoint[];
  maxSizeBytes: number;
  growthRateMbPerMonth: number;
};

const DATASET_OPTIONS = [
  { value: 'media', label: 'Media DB' },
  { value: 'chacha', label: 'Notes/Chats DB' },
  { value: 'prompts', label: 'Prompts DB' },
  { value: 'evaluations', label: 'Evaluations DB' },
  { value: 'audit', label: 'Unified Audit DB' },
  { value: 'authnz', label: 'AuthNZ Users DB' },
];

const PER_USER_BACKUP_DATASETS = new Set(['media', 'chacha', 'prompts', 'evaluations', 'audit']);

const BACKUP_TYPES = [
  { value: 'full', label: 'Full' },
  { value: 'incremental', label: 'Incremental' },
];

const SCHEDULE_FREQUENCY_OPTIONS: Array<{ value: BackupScheduleFrequency; label: string }> = [
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
  { value: 'monthly', label: 'Monthly' },
];

const formatBackupDate = (value?: string | null) => formatDateTime(value, {
  fallback: '—',
  options: {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  },
});

const isBackupScheduleFrequency = (value: string): value is BackupScheduleFrequency =>
  value === 'daily' || value === 'weekly' || value === 'monthly';

const toPositiveInteger = (value: string): number | null => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) return null;
  return parsed;
};

const parseBackupItems = (value: unknown): BackupListItem[] => {
  if (Array.isArray(value)) {
    return value as BackupListItem[];
  }
  if (value && typeof value === 'object') {
    const payload = value as { items?: unknown };
    if (Array.isArray(payload.items)) {
      return payload.items as BackupListItem[];
    }
  }
  return [];
};

const normalizeBackupStatus = (status?: string | null): BackupHistoryStatusFilter => {
  const normalized = String(status || '').trim().toLowerCase();
  if (['ready', 'success', 'succeeded', 'completed', 'complete'].includes(normalized)) {
    return 'success';
  }
  if (['failed', 'error', 'errored', 'aborted'].includes(normalized)) {
    return 'failed';
  }
  if (['in_progress', 'in-progress', 'running', 'processing', 'queued', 'pending', 'started'].includes(normalized)) {
    return 'in_progress';
  }
  return 'success';
};

const resolveBackupDurationSeconds = (item: BackupListItem): number | null => {
  if (typeof item.duration_seconds === 'number' && Number.isFinite(item.duration_seconds) && item.duration_seconds >= 0) {
    return item.duration_seconds;
  }
  if (typeof item.duration_ms === 'number' && Number.isFinite(item.duration_ms) && item.duration_ms >= 0) {
    return item.duration_ms / 1000;
  }
  if (item.started_at && item.completed_at) {
    const startedMs = Date.parse(item.started_at);
    const completedMs = Date.parse(item.completed_at);
    if (Number.isFinite(startedMs) && Number.isFinite(completedMs) && completedMs >= startedMs) {
      return (completedMs - startedMs) / 1000;
    }
  }
  return null;
};

const resolveBackupError = (item: BackupListItem): string | null => {
  const candidates = [item.error_message, item.failure_reason, item.status_message]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0);
  if (candidates.length > 0) {
    return candidates[0];
  }
  return null;
};

const buildDatasetTrends = (items: BackupListItem[]): BackupDatasetTrend[] => {
  const byDataset = new Map<string, BackupTrendPoint[]>();

  items.forEach((item) => {
    if (!item.dataset) return;
    const createdAtMs = Date.parse(item.created_at || '');
    if (!Number.isFinite(createdAtMs)) return;
    if (!Number.isFinite(item.size_bytes) || item.size_bytes < 0) return;
    const entry: BackupTrendPoint = {
      id: item.id,
      createdAt: item.created_at,
      createdAtMs,
      sizeBytes: item.size_bytes,
    };
    const existing = byDataset.get(item.dataset) ?? [];
    existing.push(entry);
    byDataset.set(item.dataset, existing);
  });

  const monthMs = 30 * 24 * 60 * 60 * 1000;

  return Array.from(byDataset.entries())
    .map(([dataset, points]): BackupDatasetTrend | null => {
      const sorted = points.slice().sort((a, b) => a.createdAtMs - b.createdAtMs);
      const latestTen = sorted.slice(-10);
      if (latestTen.length === 0) return null;

      const first = latestTen[0];
      const last = latestTen[latestTen.length - 1];
      const elapsedMs = last.createdAtMs - first.createdAtMs;
      const growthRateMbPerMonth = elapsedMs > 0
        ? ((last.sizeBytes - first.sizeBytes) / (1024 * 1024)) * (monthMs / elapsedMs)
        : 0;
      const maxSizeBytes = Math.max(1, ...latestTen.map((point) => point.sizeBytes));

      return {
        dataset,
        points: latestTen,
        maxSizeBytes,
        growthRateMbPerMonth: Number.isFinite(growthRateMbPerMonth) ? growthRateMbPerMonth : 0,
      };
    })
    .filter((trend): trend is BackupDatasetTrend => trend !== null)
    .sort((a, b) => a.dataset.localeCompare(b.dataset));
};

const requiresScheduleTargetUser = (dataset: string) => PER_USER_BACKUP_DATASETS.has(dataset);

const formatScheduleExecutionStatus = (status?: string | null) => {
  const normalized = String(status || '').trim();
  if (!normalized) return 'No runs yet';
  return normalized
    .split('_')
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(' ');
};

const BackupStatusBadge = ({ status }: { status?: string | null }) => {
  const normalizedStatus = normalizeBackupStatus(status);
  if (normalizedStatus === 'failed') {
    return (
      <span className="inline-flex items-center gap-2" data-testid="backup-status-failed">
        <XCircle className="h-4 w-4 text-red-600" />
        <Badge variant="destructive">Failed</Badge>
      </span>
    );
  }
  if (normalizedStatus === 'in_progress') {
    return (
      <span className="inline-flex items-center gap-2" data-testid="backup-status-in-progress">
        <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
        <Badge variant="secondary">In progress</Badge>
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-2" data-testid="backup-status-success">
      <CheckCircle2 className="h-4 w-4 text-emerald-600" />
      <Badge variant="default">Success</Badge>
    </span>
  );
};

export const BackupsSection = ({ refreshSignal }: BackupsSectionProps) => {
  const { page, pageSize, setPage, setPageSize, resetPagination } = useUrlPagination();
  const { success, error: showError } = useToast();
  const confirm = useConfirm();

  const [activeTab, setActiveTab] = useState<BackupsTab>('backups');

  const [listDataset, setListDataset] = useState('');
  const [listUserId, setListUserId] = useState('');

  const [createDataset, setCreateDataset] = useState('media');
  const [createUserId, setCreateUserId] = useState('');
  const [backupType, setBackupType] = useState('full');
  const [maxBackups, setMaxBackups] = useState('');
  const [creatingBackup, setCreatingBackup] = useState(false);
  const [restoreBusyId, setRestoreBusyId] = useState<string | null>(null);

  const [historyItems, setHistoryItems] = useState<BackupListItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState('');
  const [historyDatasetFilter, setHistoryDatasetFilter] = useState('');
  const [historyStatusFilter, setHistoryStatusFilter] = useState<BackupHistoryStatusFilter>('all');

  const [schedules, setSchedules] = useState<BackupScheduleItem[]>([]);
  const [scheduleUsers, setScheduleUsers] = useState<User[]>([]);
  const [scheduleLoading, setScheduleLoading] = useState(true);
  const [scheduleUsersLoading, setScheduleUsersLoading] = useState(true);
  const [scheduleSubmitting, setScheduleSubmitting] = useState(false);
  const [scheduleActionBusyId, setScheduleActionBusyId] = useState<string | null>(null);
  const [scheduleDataset, setScheduleDataset] = useState('media');
  const [scheduleTargetUserId, setScheduleTargetUserId] = useState('');
  const [scheduleFrequency, setScheduleFrequency] = useState('');
  const [scheduleTimeOfDay, setScheduleTimeOfDay] = useState('');
  const [scheduleRetentionCount, setScheduleRetentionCount] = useState('30');
  const [scheduleError, setScheduleError] = useState('');
  const [editingScheduleId, setEditingScheduleId] = useState<string | null>(null);

  const backupParams = useMemo(() => {
    const params: Record<string, string> = {
      limit: String(pageSize),
      offset: String(Math.max(0, (page - 1) * pageSize)),
    };
    if (listDataset) params.dataset = listDataset;
    if (listUserId) params.user_id = listUserId;
    return params;
  }, [listDataset, listUserId, page, pageSize]);

  const loadBackups = useCallback(({ signal }: LoadOptions = {}) =>
    api.getBackups(backupParams, signal ? { signal } : undefined), [backupParams]);

  const {
    items: backups,
    total: backupTotal,
    loading: backupLoading,
    error: backupError,
    reload,
  } = usePagedResource<BackupListItem>({
    load: loadBackups,
    deps: [refreshSignal],
    defaultError: 'Failed to load backups',
  });

  const loadHistory = useCallback(async () => {
    try {
      setHistoryLoading(true);
      setHistoryError('');
      const result = await api.getBackups({
        limit: '200',
        offset: '0',
      });
      const parsed = parseBackupItems(result)
        .slice()
        .sort((a, b) => Date.parse(b.created_at || '') - Date.parse(a.created_at || ''));
      setHistoryItems(parsed);
    } catch (err: unknown) {
      logger.error('Failed to load backup history', { component: 'BackupsSection', error: err instanceof Error ? err.message : String(err) });
      setHistoryError(err instanceof Error && err.message ? err.message : 'Failed to load backup history');
      setHistoryItems([]);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory, refreshSignal]);

  const loadSchedules = useCallback(async () => {
    try {
      setScheduleLoading(true);
      const response = await api.listBackupSchedules({
        limit: '100',
        offset: '0',
      });
      setSchedules(response.items.filter((item) => !item.deleted_at));
      setScheduleError('');
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to load backup schedules';
      setScheduleError(message);
      setSchedules([]);
    } finally {
      setScheduleLoading(false);
    }
  }, []);

  const loadScheduleUsers = useCallback(async () => {
    try {
      setScheduleUsersLoading(true);
      const response = await api.getUsers({ limit: '100' });
      setScheduleUsers(response);
    } catch (err: unknown) {
      logger.error('Failed to load backup schedule users', { component: 'BackupsSection', error: err instanceof Error ? err.message : String(err) });
      setScheduleUsers([]);
    } finally {
      setScheduleUsersLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSchedules();
    void loadScheduleUsers();
  }, [loadScheduleUsers, loadSchedules, refreshSignal]);

  const upsertSchedule = useCallback((item: BackupScheduleItem) => {
    setSchedules((current) => {
      const remaining = current.filter((schedule) => schedule.id !== item.id);
      if (item.deleted_at) {
        return remaining;
      }
      return [item, ...remaining]
        .slice()
        .sort((a, b) => Date.parse(b.created_at || '') - Date.parse(a.created_at || ''));
    });
  }, []);

  const scheduleUsersById = useMemo(() => {
    const next = new Map<number, User>();
    scheduleUsers.forEach((user) => {
      next.set(user.id, user);
    });
    return next;
  }, [scheduleUsers]);

  const scheduleRequiresTargetUser = requiresScheduleTargetUser(scheduleDataset);

  const resolveScheduleTargetLabel = useCallback((schedule: BackupScheduleItem) => {
    if (!schedule.target_user_id) {
      return 'Platform';
    }
    const user = scheduleUsersById.get(schedule.target_user_id);
    if (!user) {
      return `User #${schedule.target_user_id}`;
    }
    const identity = user.email?.trim() || user.username?.trim() || `User #${user.id}`;
    return `${identity} (#${user.id})`;
  }, [scheduleUsersById]);

  const handleBackupFilterChange = (key: 'dataset' | 'user', value: string) => {
    if (key === 'dataset') {
      setListDataset(value);
    } else {
      setListUserId(value);
    }
    resetPagination();
  };

  const handleCreateBackup = async () => {
    const payload: Record<string, unknown> = {
      dataset: createDataset,
      backup_type: backupType,
    };
    if (createUserId.trim()) {
      const parsed = Number(createUserId);
      if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
        showError('Invalid user id', 'User id must be a positive integer.');
        return;
      }
      payload.user_id = parsed;
    }
    if (maxBackups.trim()) {
      const parsed = Number(maxBackups);
      if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
        showError('Invalid max backups', 'Max backups must be a positive integer.');
        return;
      }
      payload.max_backups = parsed;
    }

    try {
      setCreatingBackup(true);
      await api.createBackup(payload);
      success('Backup created', 'Snapshot created successfully.');
      await reload();
      await loadHistory();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to create backup';
      showError('Backup failed', message);
    } finally {
      setCreatingBackup(false);
    }
  };

  const handleRestoreBackup = async (backup: BackupListItem) => {
    const accepted = await confirm({
      title: 'Restore backup?',
      message: `Restore ${backup.dataset} from ${backup.id}? This overwrites the current dataset.`,
      confirmText: 'Restore',
      variant: 'danger',
      icon: 'warning',
    });
    if (!accepted) return;

    try {
      setRestoreBusyId(backup.id);
      await api.restoreBackup(backup.id, {
        dataset: backup.dataset,
        user_id: backup.user_id ?? undefined,
        confirm: true,
      });
      success('Restore complete', 'Backup restored successfully.');
      await reload();
      await loadHistory();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to restore backup';
      showError('Restore failed', message);
    } finally {
      setRestoreBusyId(null);
    }
  };

  const validateScheduleForm = () => {
    if (scheduleRequiresTargetUser) {
      const targetUserId = toPositiveInteger(scheduleTargetUserId);
      if (targetUserId === null) {
        return 'Select a target user.';
      }
    }
    if (!scheduleFrequency || !isBackupScheduleFrequency(scheduleFrequency)) {
      return 'Frequency is required.';
    }
    if (!scheduleTimeOfDay || !/^\d{2}:\d{2}$/.test(scheduleTimeOfDay)) {
      return 'Time of day is required.';
    }
    const retentionCount = toPositiveInteger(scheduleRetentionCount);
    if (retentionCount === null) {
      return 'Retention count must be a positive integer.';
    }
    return '';
  };

  const resetScheduleForm = () => {
    setScheduleDataset('media');
    setScheduleTargetUserId('');
    setScheduleFrequency('');
    setScheduleTimeOfDay('');
    setScheduleRetentionCount('30');
    setScheduleError('');
    setEditingScheduleId(null);
  };

  const handleSubmitSchedule = async () => {
    const validationError = validateScheduleForm();
    if (validationError) {
      setScheduleError(validationError);
      return;
    }

    const retentionCount = toPositiveInteger(scheduleRetentionCount) as number;
    const targetUserId = scheduleRequiresTargetUser ? toPositiveInteger(scheduleTargetUserId) : null;

    try {
      setScheduleSubmitting(true);
      setScheduleError('');

      if (editingScheduleId) {
        const response = await api.updateBackupSchedule(editingScheduleId, {
          frequency: scheduleFrequency,
          time_of_day: scheduleTimeOfDay,
          retention_count: retentionCount,
        });
        upsertSchedule(response.item);
        success('Schedule updated', 'Backup schedule updated.');
      } else {
        const payload: Record<string, unknown> = {
          dataset: scheduleDataset,
          frequency: scheduleFrequency,
          time_of_day: scheduleTimeOfDay,
          retention_count: retentionCount,
        };
        if (targetUserId !== null) {
          payload.target_user_id = targetUserId;
        }
        const response = await api.createBackupSchedule(payload);
        upsertSchedule(response.item);
        success('Schedule created', 'Backup schedule created.');
      }

      resetScheduleForm();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to save backup schedule';
      setScheduleError(message);
      showError('Schedule failed', message);
    } finally {
      setScheduleSubmitting(false);
    }
  };

  const handleEditSchedule = (schedule: BackupScheduleItem) => {
    setEditingScheduleId(schedule.id);
    setScheduleDataset(schedule.dataset);
    setScheduleTargetUserId(schedule.target_user_id ? String(schedule.target_user_id) : '');
    setScheduleFrequency(schedule.frequency);
    setScheduleTimeOfDay(schedule.time_of_day);
    setScheduleRetentionCount(String(schedule.retention_count));
    setScheduleError('');
  };

  const handleToggleSchedulePause = async (schedule: BackupScheduleItem) => {
    try {
      setScheduleActionBusyId(schedule.id);
      const response = schedule.is_paused
        ? await api.resumeBackupSchedule(schedule.id)
        : await api.pauseBackupSchedule(schedule.id);
      upsertSchedule(response.item);
      success(
        schedule.is_paused ? 'Schedule resumed' : 'Schedule paused',
        schedule.is_paused ? 'Backup schedule resumed.' : 'Backup schedule paused.'
      );
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to update backup schedule';
      setScheduleError(message);
      showError('Schedule update failed', message);
    } finally {
      setScheduleActionBusyId(null);
    }
  };

  const handleDeleteSchedule = async (scheduleId: string) => {
    const accepted = await confirm({
      title: 'Delete backup schedule?',
      message: 'This removes the shared backup schedule configuration.',
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'delete',
    });
    if (!accepted) return;

    try {
      setScheduleActionBusyId(scheduleId);
      await api.deleteBackupSchedule(scheduleId);
      setSchedules((current) => current.filter((schedule) => schedule.id !== scheduleId));
      if (editingScheduleId === scheduleId) {
        resetScheduleForm();
      }
      success('Schedule deleted', 'Backup schedule removed.');
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to delete backup schedule';
      setScheduleError(message);
      showError('Schedule delete failed', message);
    } finally {
      setScheduleActionBusyId(null);
    }
  };

  const filteredHistoryItems = useMemo(() => {
    return historyItems
      .filter((item) => (historyDatasetFilter ? item.dataset === historyDatasetFilter : true))
      .filter((item) => (historyStatusFilter === 'all' ? true : normalizeBackupStatus(item.status) === historyStatusFilter))
      .slice(0, 20);
  }, [historyDatasetFilter, historyItems, historyStatusFilter]);
  const datasetTrends = useMemo(() => buildDatasetTrends(historyItems), [historyItems]);

  const totalPages = Math.max(1, Math.ceil(backupTotal / pageSize));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Database className="h-5 w-5" />
          Backups
        </CardTitle>
        <CardDescription>Create, browse, schedule, and restore backup snapshots.</CardDescription>
        <div className="flex flex-wrap gap-2 pt-1">
          <Button
            variant={activeTab === 'backups' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setActiveTab('backups')}
          >
            Backups
          </Button>
          <Button
            variant={activeTab === 'schedule' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setActiveTab('schedule')}
          >
            Schedule
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {activeTab === 'schedule' ? (
          <>
            <Alert>
              <AlertDescription>
                Backup schedules are shared platform policy. Scheduled runs enqueue backup jobs and retain their latest run state here.
              </AlertDescription>
            </Alert>

            <div className="grid gap-3 md:grid-cols-6">
              <Field id="backup-schedule-dataset" label="Dataset">
                <Select
                  id="backup-schedule-dataset"
                  value={scheduleDataset}
                  onChange={(event) => {
                    setScheduleDataset(event.target.value);
                    setScheduleError('');
                  }}
                  disabled={Boolean(editingScheduleId)}
                >
                  {DATASET_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              </Field>
              {scheduleRequiresTargetUser && (
                <Field id="backup-schedule-target-user" label="Target user">
                  <Select
                    id="backup-schedule-target-user"
                    value={scheduleTargetUserId}
                    onChange={(event) => {
                      setScheduleTargetUserId(event.target.value);
                      setScheduleError('');
                    }}
                    disabled={Boolean(editingScheduleId) || scheduleUsersLoading}
                  >
                    <option value="">
                      {scheduleUsersLoading ? 'Loading users…' : 'Select a user'}
                    </option>
                    {scheduleUsers.map((user) => (
                      <option key={user.id} value={String(user.id)}>
                        {(user.email?.trim() || user.username?.trim() || `User #${user.id}`)} ({user.id})
                      </option>
                    ))}
                  </Select>
                </Field>
              )}
              <Field id="backup-schedule-frequency" label="Frequency">
                <Select
                  id="backup-schedule-frequency"
                  value={scheduleFrequency}
                  onChange={(event) => {
                    setScheduleFrequency(event.target.value);
                    setScheduleError('');
                  }}
                >
                  <option value="">Select frequency</option>
                  {SCHEDULE_FREQUENCY_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field id="backup-schedule-time" label="Time of day">
                <Input
                  id="backup-schedule-time"
                  type="time"
                  value={scheduleTimeOfDay}
                  onChange={(event) => {
                    setScheduleTimeOfDay(event.target.value);
                    setScheduleError('');
                  }}
                />
              </Field>
              <Field id="backup-schedule-retention" label="Retention count">
                <Input
                  id="backup-schedule-retention"
                  value={scheduleRetentionCount}
                  onChange={(event) => {
                    setScheduleRetentionCount(event.target.value);
                    setScheduleError('');
                  }}
                />
              </Field>
              <div className="flex items-end gap-2">
                <Button onClick={() => { void handleSubmitSchedule(); }} disabled={scheduleSubmitting}>
                  {editingScheduleId ? 'Update schedule' : 'Create schedule'}
                </Button>
                {editingScheduleId && (
                  <Button variant="outline" onClick={resetScheduleForm} disabled={scheduleSubmitting}>
                    Cancel
                  </Button>
                )}
              </div>
            </div>

            {scheduleError && (
              <Alert variant="destructive">
                <AlertDescription>{scheduleError}</AlertDescription>
              </Alert>
            )}

            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Dataset</TableHead>
                  <TableHead>Target</TableHead>
                  <TableHead>Schedule</TableHead>
                  <TableHead>Next run</TableHead>
                  <TableHead>Last run</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {scheduleLoading ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-muted-foreground">
                      Loading backup schedules…
                    </TableCell>
                  </TableRow>
                ) : schedules.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-muted-foreground">
                      No backup schedules configured.
                    </TableCell>
                  </TableRow>
                ) : (
                  schedules.map((schedule) => (
                    <TableRow key={schedule.id} data-testid={`backup-schedule-row-${schedule.id}`}>
                      <TableCell>{schedule.dataset}</TableCell>
                      <TableCell>{resolveScheduleTargetLabel(schedule)}</TableCell>
                      <TableCell>
                        <div className="space-y-1">
                          <div>{schedule.schedule_description}</div>
                          <div className="text-xs text-muted-foreground">
                            Retain {schedule.retention_count} snapshots
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>{formatBackupDate(schedule.next_run_at)}</TableCell>
                      <TableCell>{formatBackupDate(schedule.last_run_at)}</TableCell>
                      <TableCell>
                        <div className="space-y-1">
                          <Badge variant={schedule.is_paused ? 'secondary' : 'default'}>
                            {schedule.is_paused ? 'Paused' : 'Active'}
                          </Badge>
                          {(!schedule.is_paused || String(schedule.last_status || '').trim().toLowerCase() !== 'paused') ? (
                            <div className="text-xs text-muted-foreground">
                              {formatScheduleExecutionStatus(schedule.last_status)}
                            </div>
                          ) : null}
                          {schedule.last_error ? (
                            <div className="text-xs text-red-600">{schedule.last_error}</div>
                          ) : null}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleEditSchedule(schedule)}
                            aria-label="Edit schedule"
                            title="Edit schedule"
                            disabled={scheduleActionBusyId === schedule.id || scheduleSubmitting}
                          >
                            <Pencil className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => { void handleToggleSchedulePause(schedule); }}
                            aria-label={schedule.is_paused ? 'Resume schedule' : 'Pause schedule'}
                            title={schedule.is_paused ? 'Resume schedule' : 'Pause schedule'}
                            data-testid={`backup-schedule-toggle-${schedule.id}`}
                            disabled={scheduleActionBusyId === schedule.id || scheduleSubmitting}
                          >
                            {schedule.is_paused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => { void handleDeleteSchedule(schedule.id); }}
                            aria-label="Delete schedule"
                            title="Delete schedule"
                            disabled={scheduleActionBusyId === schedule.id || scheduleSubmitting}
                          >
                            <Trash2 className="h-4 w-4 text-destructive" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </>
        ) : (
          <>
            <div className="grid gap-3 md:grid-cols-4">
              <Field id="backup-filter-dataset" label="Dataset filter">
                <Select
                  id="backup-filter-dataset"
                  value={listDataset}
                  onChange={(event) => handleBackupFilterChange('dataset', event.target.value)}
                >
                  <option value="">All datasets</option>
                  {DATASET_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field id="backup-filter-user" label="User ID filter">
                <Input
                  id="backup-filter-user"
                  placeholder="Optional"
                  value={listUserId}
                  onChange={(event) => handleBackupFilterChange('user', event.target.value)}
                />
              </Field>
              <div className="flex items-end">
                <Button variant="outline" onClick={reload} disabled={backupLoading}>
                  Refresh list
                </Button>
              </div>
            </div>

            <div className="grid gap-3 md:grid-cols-5">
              <Field id="backup-create-dataset" label="Dataset">
                <Select
                  id="backup-create-dataset"
                  value={createDataset}
                  onChange={(event) => setCreateDataset(event.target.value)}
                >
                  {DATASET_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field id="backup-create-user" label="User ID">
                <Input
                  id="backup-create-user"
                  placeholder="Optional"
                  value={createUserId}
                  onChange={(event) => setCreateUserId(event.target.value)}
                />
              </Field>
              <Field id="backup-create-type" label="Type">
                <Select
                  id="backup-create-type"
                  value={backupType}
                  onChange={(event) => setBackupType(event.target.value)}
                >
                  {BACKUP_TYPES.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field id="backup-create-max" label="Max backups">
                <Input
                  id="backup-create-max"
                  placeholder="Optional"
                  value={maxBackups}
                  onChange={(event) => setMaxBackups(event.target.value)}
                />
              </Field>
              <div className="flex items-end">
                <Button onClick={handleCreateBackup} disabled={creatingBackup} loading={creatingBackup} loadingText="Creating...">
                  Create backup
                </Button>
              </div>
            </div>

            {backupError && (
              <Alert variant="destructive">
                <AlertDescription>{backupError}</AlertDescription>
              </Alert>
            )}

            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead>Dataset</TableHead>
                  <TableHead>User</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {backupLoading ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-muted-foreground">
                      Loading backups...
                    </TableCell>
                  </TableRow>
                ) : backups.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-muted-foreground">
                      No backups found.
                    </TableCell>
                  </TableRow>
                ) : (
                  backups.map((backup) => {
                    const statusKind = normalizeBackupStatus(backup.status);
                    const backupErrorMessage = resolveBackupError(backup);
                    return (
                      <TableRow key={backup.id}>
                        <TableCell className="font-mono text-xs">{backup.id}</TableCell>
                        <TableCell>{backup.dataset}</TableCell>
                        <TableCell>{backup.user_id ?? '—'}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <BackupStatusBadge status={backup.status} />
                            {statusKind === 'failed' && backupErrorMessage && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => showError('Backup failed', backupErrorMessage)}
                                title={backupErrorMessage}
                                aria-label="View backup error"
                              >
                                <AlertCircle className="h-4 w-4 text-destructive" />
                              </Button>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>{formatBytes(backup.size_bytes, { fallback: '—' })}</TableCell>
                        <TableCell>{formatBackupDate(backup.created_at)}</TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => { void handleRestoreBackup(backup); }}
                            disabled={restoreBusyId === backup.id}
                          >
                            {restoreBusyId === backup.id ? 'Restoring...' : 'Restore'}
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })
                )}
              </TableBody>
            </Table>

            <Pagination
              currentPage={page}
              totalPages={totalPages}
              totalItems={backupTotal}
              pageSize={pageSize}
              onPageChange={setPage}
              onPageSizeChange={setPageSize}
            />

            <div className="space-y-4 rounded-lg border p-4" data-testid="backup-history-section">
              <div className="space-y-1">
                <h3 className="text-lg font-semibold">Backup History</h3>
                <p className="text-sm text-muted-foreground">
                  Last 20 backups across datasets with status and duration.
                </p>
              </div>

              <div className="grid gap-3 md:grid-cols-3">
                <Field id="backup-history-dataset" label="Dataset">
                  <Select
                    id="backup-history-dataset"
                    value={historyDatasetFilter}
                    onChange={(event) => setHistoryDatasetFilter(event.target.value)}
                  >
                    <option value="">All datasets</option>
                    {DATASET_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field id="backup-history-status" label="Status">
                  <Select
                    id="backup-history-status"
                    value={historyStatusFilter}
                    onChange={(event) => setHistoryStatusFilter(event.target.value as BackupHistoryStatusFilter)}
                  >
                    <option value="all">All statuses</option>
                    <option value="success">Success</option>
                    <option value="failed">Failed</option>
                    <option value="in_progress">In progress</option>
                  </Select>
                </Field>
                <div className="flex items-end">
                  <Button variant="outline" onClick={() => { void loadHistory(); }} disabled={historyLoading}>
                    Refresh history
                  </Button>
                </div>
              </div>

              {historyError && (
                <Alert variant="destructive">
                  <AlertDescription>{historyError}</AlertDescription>
                </Alert>
              )}

              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>Dataset</TableHead>
                    <TableHead>Size</TableHead>
                    <TableHead>Duration</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {historyLoading ? (
                    <TableRow>
                      <TableCell colSpan={5} className="text-muted-foreground">Loading history...</TableCell>
                    </TableRow>
                  ) : filteredHistoryItems.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={5} className="text-muted-foreground">No backup history found.</TableCell>
                    </TableRow>
                  ) : (
                    filteredHistoryItems.map((item) => {
                      const statusKind = normalizeBackupStatus(item.status);
                      const backupErrorMessage = resolveBackupError(item);
                      const durationSeconds = resolveBackupDurationSeconds(item);
                      return (
                        <TableRow key={`${item.id}-${item.created_at}`} data-testid="backup-history-row">
                          <TableCell>{formatBackupDate(item.created_at)}</TableCell>
                          <TableCell>{item.dataset}</TableCell>
                          <TableCell>{formatBytes(item.size_bytes, { fallback: '—' })}</TableCell>
                          <TableCell>{formatDuration(durationSeconds, { fallback: '—' })}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-2">
                              <BackupStatusBadge status={item.status} />
                              {statusKind === 'failed' && backupErrorMessage && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => showError('Backup failed', backupErrorMessage)}
                                  title={backupErrorMessage}
                                  aria-label="View backup history error"
                                >
                                  <AlertCircle className="h-4 w-4 text-destructive" />
                                </Button>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                      );
                    })
                  )}
                </TableBody>
              </Table>

              <div className="space-y-3 rounded-lg border p-4" data-testid="backup-storage-trending">
                <div className="space-y-1">
                  <h4 className="text-base font-semibold">Storage Trending</h4>
                  <p className="text-sm text-muted-foreground">
                    Backup size over time (last 10 snapshots per dataset).
                  </p>
                </div>
                {datasetTrends.length === 0 ? (
                  <p className="text-sm text-muted-foreground">Not enough backup history to render trends.</p>
                ) : (
                  <div className="grid gap-3 md:grid-cols-2">
                    {datasetTrends.map((trend) => (
                      <div key={trend.dataset} className="rounded-md border p-3 space-y-2" data-testid={`backup-trend-card-${trend.dataset}`}>
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium">{trend.dataset}</p>
                          <p className="text-xs text-muted-foreground">{trend.points.length} snapshots</p>
                        </div>
                        <div className="flex h-20 items-end gap-1" data-testid={`backup-trend-series-${trend.dataset}`}>
                          {trend.points.map((point, index) => {
                            const ratio = trend.maxSizeBytes > 0 ? point.sizeBytes / trend.maxSizeBytes : 0;
                            const barHeight = Math.max(8, Math.round(ratio * 64));
                            return (
                              <div
                                key={`${point.id}-${point.createdAt}-${index}`}
                                className="flex-1 rounded-sm bg-emerald-500/70"
                                style={{ height: `${barHeight}px` }}
                                title={`${formatBackupDate(point.createdAt)} • ${formatBytes(point.sizeBytes, { fallback: '0 B' })}`}
                              />
                            );
                          })}
                        </div>
                        <p className="text-xs text-muted-foreground" data-testid={`backup-trend-growth-${trend.dataset}`}>
                          Storage growing at {Math.max(0, trend.growthRateMbPerMonth).toFixed(1)} MB/month.
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};
