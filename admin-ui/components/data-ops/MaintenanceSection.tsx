'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/toast';
import { api } from '@/lib/api-client';
import type {
  MaintenanceRotationRunCreateRequest,
  MaintenanceRotationRunItem,
} from '@/types';
import { Settings, Trash2, FileText, Database, Key, RefreshCw } from 'lucide-react';
import { logger } from '@/lib/logger';
import { CardSkeleton } from '@/components/ui/skeleton';

type CleanupSettings = {
  auto_cleanup_enabled: boolean;
  retention_days: number;
  cleanup_schedule?: string;
  last_cleanup?: string;
};

type NotesTitleSettings = {
  auto_generate_titles: boolean;
  title_format?: string;
  max_title_length: number;
};

type MaintenanceSectionProps = {
  refreshSignal: number;
};

const formatTimestamp = (value?: string | null) => {
  if (!value) return 'Never';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return 'Never';
  return parsed.toLocaleString();
};

export function MaintenanceSection({ refreshSignal }: MaintenanceSectionProps) {
  const confirm = useConfirm();
  const { success, error: showError } = useToast();

  const [cleanupSettings, setCleanupSettings] = useState<CleanupSettings | null>(null);
  const [cleanupLoading, setCleanupLoading] = useState(false);
  const [cleanupSaving, setCleanupSaving] = useState(false);
  const [editRetentionDays, setEditRetentionDays] = useState('');
  const [editAutoCleanup, setEditAutoCleanup] = useState(false);

  const [, setNotesSettings] = useState<NotesTitleSettings | null>(null);
  const [notesLoading, setNotesLoading] = useState(false);
  const [notesSaving, setNotesSaving] = useState(false);
  const [editAutoTitles, setEditAutoTitles] = useState(false);
  const [editMaxTitleLength, setEditMaxTitleLength] = useState('');

  const [ftsMaintRunning, setFtsMaintRunning] = useState(false);

  const [rotationSubmitting, setRotationSubmitting] = useState(false);
  const [rotationHistoryLoading, setRotationHistoryLoading] = useState(false);
  const [rotationHistory, setRotationHistory] = useState<MaintenanceRotationRunItem[]>([]);
  const [currentRotationRun, setCurrentRotationRun] = useState<MaintenanceRotationRunItem | null>(null);
  const [rotationMode, setRotationMode] = useState<'dry_run' | 'execute'>('dry_run');
  const [rotationDomain, setRotationDomain] = useState('jobs');
  const [rotationQueue, setRotationQueue] = useState('default');
  const [rotationJobType, setRotationJobType] = useState('encryption_rotation');
  const [rotationLimit, setRotationLimit] = useState('1000');
  const [rotatePayloadField, setRotatePayloadField] = useState(true);
  const [rotateResultField, setRotateResultField] = useState(true);
  const [rotationConfirmed, setRotationConfirmed] = useState(false);

  const previousRotationStatusRef = useRef<MaintenanceRotationRunItem['status'] | null>(null);

  const loadCleanupSettings = useCallback(async () => {
    try {
      setCleanupLoading(true);
      const data = await api.getCleanupSettings();
      const settings = data as CleanupSettings;
      setCleanupSettings(settings);
      setEditRetentionDays(settings?.retention_days?.toString() || '30');
      setEditAutoCleanup(settings?.auto_cleanup_enabled ?? false);
    } catch (err: unknown) {
      logger.warn('Failed to load cleanup settings', { component: 'MaintenanceSection', error: err instanceof Error ? err.message : String(err) });
      setCleanupSettings({
        auto_cleanup_enabled: false,
        retention_days: 30,
      });
      setEditRetentionDays('30');
      setEditAutoCleanup(false);
    } finally {
      setCleanupLoading(false);
    }
  }, []);

  const loadNotesSettings = useCallback(async () => {
    try {
      setNotesLoading(true);
      const data = await api.getNotesTitleSettings();
      const settings = data as NotesTitleSettings;
      setNotesSettings(settings);
      setEditAutoTitles(settings?.auto_generate_titles ?? true);
      setEditMaxTitleLength(settings?.max_title_length?.toString() || '100');
    } catch (err: unknown) {
      logger.warn('Failed to load notes title settings', { component: 'MaintenanceSection', error: err instanceof Error ? err.message : String(err) });
      setNotesSettings({
        auto_generate_titles: true,
        max_title_length: 100,
      });
      setEditAutoTitles(true);
      setEditMaxTitleLength('100');
    } finally {
      setNotesLoading(false);
    }
  }, []);

  const loadRotationRuns = useCallback(async (preferredRunId?: string) => {
    try {
      setRotationHistoryLoading(true);
      const response = await api.getMaintenanceRotationRuns({ limit: 10, offset: 0 });
      const items = response.items ?? [];
      setRotationHistory(items);
      setCurrentRotationRun((previous) => {
        if (preferredRunId) {
          return items.find((item) => item.id === preferredRunId) ?? previous;
        }
        if (previous) {
          return items.find((item) => item.id === previous.id) ?? items[0] ?? null;
        }
        return items[0] ?? null;
      });
    } catch (err) {
      logger.warn('Failed to load maintenance rotation runs', { component: 'MaintenanceSection', error: err instanceof Error ? err.message : String(err) });
    } finally {
      setRotationHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadCleanupSettings();
    void loadNotesSettings();
    void loadRotationRuns();
  }, [loadCleanupSettings, loadNotesSettings, loadRotationRuns, refreshSignal]);

  useEffect(() => {
    if (!currentRotationRun || !['queued', 'running'].includes(currentRotationRun.status)) {
      return;
    }

    let cancelled = false;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const updated = await api.getMaintenanceRotationRun(currentRotationRun.id);
          if (cancelled) return;
          setCurrentRotationRun(updated);
          setRotationHistory((previous) => {
            const deduped = previous.filter((item) => item.id !== updated.id);
            return [updated, ...deduped].slice(0, 10);
          });
          if (!['queued', 'running'].includes(updated.status)) {
            void loadRotationRuns(updated.id);
          }
        } catch (err) {
          if (cancelled) return;
          logger.warn('Failed to poll maintenance rotation run', { component: 'MaintenanceSection', error: err instanceof Error ? err.message : String(err) });
        }
      })();
    }, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [currentRotationRun, loadRotationRuns]);

  useEffect(() => {
    const currentStatus = currentRotationRun?.status ?? null;
    const previousStatus = previousRotationStatusRef.current;

    if (
      previousStatus
      && ['queued', 'running'].includes(previousStatus)
      && currentStatus
      && !['queued', 'running'].includes(currentStatus)
      && currentRotationRun
    ) {
      if (currentStatus === 'complete') {
        success(
          'Rotation complete',
          `Processed ${currentRotationRun.affected_count ?? 0} records.`,
        );
      } else if (currentStatus === 'failed') {
        showError(
          'Rotation failed',
          currentRotationRun.error_message ?? 'Maintenance rotation failed.',
        );
      }
    }

    previousRotationStatusRef.current = currentStatus;
  }, [currentRotationRun, showError, success]);

  const handleSaveCleanupSettings = async () => {
    try {
      setCleanupSaving(true);
      const retentionDays = parseInt(editRetentionDays, 10);
      if (Number.isNaN(retentionDays) || retentionDays < 1) {
        showError('Invalid retention', 'Retention days must be at least 1');
        return;
      }
      await api.updateCleanupSettings({
        auto_cleanup_enabled: editAutoCleanup,
        retention_days: retentionDays,
      });
      setCleanupSettings((previous) => ({
        ...(previous ?? { auto_cleanup_enabled: editAutoCleanup, retention_days: retentionDays }),
        auto_cleanup_enabled: editAutoCleanup,
        retention_days: retentionDays,
      }));
      success('Settings saved', 'Cleanup settings updated');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to save cleanup settings';
      showError('Save failed', message);
    } finally {
      setCleanupSaving(false);
    }
  };

  const handleSaveNotesSettings = async () => {
    try {
      setNotesSaving(true);
      const maxLength = parseInt(editMaxTitleLength, 10);
      if (Number.isNaN(maxLength) || maxLength < 10) {
        showError('Invalid length', 'Max title length must be at least 10');
        return;
      }
      await api.updateNotesTitleSettings({
        auto_generate_titles: editAutoTitles,
        max_title_length: maxLength,
      });
      setNotesSettings((previous) => ({
        ...(previous ?? { auto_generate_titles: editAutoTitles, max_title_length: maxLength }),
        auto_generate_titles: editAutoTitles,
        max_title_length: maxLength,
      }));
      success('Settings saved', 'Notes title settings updated');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to save notes settings';
      showError('Save failed', message);
    } finally {
      setNotesSaving(false);
    }
  };

  const handleRunFtsMaintenance = async () => {
    const confirmed = await confirm({
      title: 'Run FTS Maintenance',
      message: 'This will optimize the full-text search indexes. The operation may take a few minutes depending on data size.',
      confirmText: 'Run Maintenance',
      variant: 'warning',
    });
    if (!confirmed) return;

    try {
      setFtsMaintRunning(true);
      await api.runKanbanFtsMaintenance();
      success('Maintenance complete', 'FTS indexes have been optimized');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'FTS maintenance failed';
      showError('Maintenance failed', message);
    } finally {
      setFtsMaintRunning(false);
    }
  };

  const handleSubmitRotation = async () => {
    const selectedFields = [
      rotatePayloadField ? 'payload' : null,
      rotateResultField ? 'result' : null,
    ].filter((field): field is string => field !== null);

    if (selectedFields.length === 0) {
      showError('Invalid scope', 'Select at least one field to rotate.');
      return;
    }

    const parsedLimit = parseInt(rotationLimit, 10);
    if (Number.isNaN(parsedLimit) || parsedLimit < 1) {
      showError('Invalid limit', 'Limit must be at least 1.');
      return;
    }

    if (rotationMode === 'execute' && !rotationConfirmed) {
      showError('Confirmation required', 'Confirm the live rotation before submitting.');
      return;
    }

    const payload: MaintenanceRotationRunCreateRequest = {
      mode: rotationMode,
      domain: rotationDomain.trim() || undefined,
      queue: rotationQueue.trim() || undefined,
      job_type: rotationJobType.trim() || undefined,
      fields: selectedFields,
      limit: parsedLimit,
      confirmed: rotationMode === 'execute' ? rotationConfirmed : false,
    };

    try {
      setRotationSubmitting(true);
      const response = await api.createMaintenanceRotationRun(payload);
      setCurrentRotationRun(response.item);
      setRotationConfirmed(false);
      await loadRotationRuns(response.item.id);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to submit maintenance rotation request.';
      showError('Rotation failed', message);
    } finally {
      setRotationSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Trash2 className="h-5 w-5" />
              Cleanup Settings
            </CardTitle>
            <CardDescription>Configure automatic data cleanup and retention</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {cleanupLoading ? (
              <CardSkeleton />
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="auto-cleanup-checkbox">Auto Cleanup</Label>
                    <p className="text-xs text-muted-foreground">Automatically delete old data</p>
                  </div>
                  <input
                    id="auto-cleanup-checkbox"
                    type="checkbox"
                    checked={editAutoCleanup}
                    onChange={(event) => setEditAutoCleanup(event.target.checked)}
                    className="h-4 w-4"
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="retention-days">Retention Period (days)</Label>
                  <Input
                    id="retention-days"
                    type="number"
                    min="1"
                    value={editRetentionDays}
                    onChange={(event) => setEditRetentionDays(event.target.value)}
                  />
                </div>
                {cleanupSettings?.last_cleanup && (
                  <p className="text-xs text-muted-foreground">
                    Last cleanup: {formatTimestamp(cleanupSettings.last_cleanup)}
                  </p>
                )}
                <Button onClick={handleSaveCleanupSettings} disabled={cleanupSaving} loading={cleanupSaving} loadingText="Saving...">
                  Save Settings
                </Button>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Notes Title Settings
            </CardTitle>
            <CardDescription>Configure automatic note title generation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {notesLoading ? (
              <CardSkeleton />
            ) : (
              <>
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="auto-generate-titles">Auto-generate Titles</Label>
                    <p className="text-xs text-muted-foreground">Generate titles from content</p>
                  </div>
                  <input
                    id="auto-generate-titles"
                    type="checkbox"
                    checked={editAutoTitles}
                    onChange={(event) => setEditAutoTitles(event.target.checked)}
                    className="h-4 w-4"
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="max-title-length">Max Title Length</Label>
                  <Input
                    id="max-title-length"
                    type="number"
                    min="10"
                    max="500"
                    value={editMaxTitleLength}
                    onChange={(event) => setEditMaxTitleLength(event.target.value)}
                  />
                </div>
                <Button onClick={handleSaveNotesSettings} disabled={notesSaving} loading={notesSaving} loadingText="Saving...">
                  Save Settings
                </Button>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Maintenance Operations
          </CardTitle>
          <CardDescription>Database maintenance and security operations</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-3 rounded-lg border p-4">
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                <span className="font-medium">FTS Index Maintenance</span>
              </div>
              <p className="text-sm text-muted-foreground">
                Optimize full-text search indexes for better performance.
              </p>
              <Button
                variant="outline"
                onClick={handleRunFtsMaintenance}
                disabled={ftsMaintRunning}
                loading={ftsMaintRunning}
                loadingText="Running..."
              >
                <RefreshCw className="mr-2 h-4 w-4" />
                Run Maintenance
              </Button>
            </div>

            <div className="space-y-4 rounded-lg border p-4" data-testid="maintenance-rotation-form">
              <div className="flex items-center gap-2">
                <Key className="h-5 w-5" />
                <span className="font-medium">Encryption Key Rotation</span>
                <Badge variant="destructive">Authoritative</Badge>
              </div>
              <p className="text-sm text-muted-foreground">
                Submit a scoped dry-run or execute request and monitor the shared backend run history.
              </p>

              <div className="grid gap-3">
                <div className="space-y-1">
                  <Label htmlFor="rotation-mode">Mode</Label>
                  <select
                    id="rotation-mode"
                    value={rotationMode}
                    onChange={(event) => setRotationMode(event.target.value as 'dry_run' | 'execute')}
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                  >
                    <option value="dry_run">Dry Run</option>
                    <option value="execute">Execute</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <Label htmlFor="rotation-domain">Domain</Label>
                  <Input
                    id="rotation-domain"
                    value={rotationDomain}
                    onChange={(event) => setRotationDomain(event.target.value)}
                  />
                </div>

                <div className="space-y-1">
                  <Label htmlFor="rotation-queue">Queue</Label>
                  <Input
                    id="rotation-queue"
                    value={rotationQueue}
                    onChange={(event) => setRotationQueue(event.target.value)}
                  />
                </div>

                <div className="space-y-1">
                  <Label htmlFor="rotation-job-type">Job type</Label>
                  <Input
                    id="rotation-job-type"
                    value={rotationJobType}
                    onChange={(event) => setRotationJobType(event.target.value)}
                  />
                </div>

                <div className="space-y-1">
                  <Label htmlFor="rotation-limit">Limit</Label>
                  <Input
                    id="rotation-limit"
                    type="number"
                    min="1"
                    value={rotationLimit}
                    onChange={(event) => setRotationLimit(event.target.value)}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Fields</Label>
                  <label htmlFor="rotate-payload-field" className="flex items-center gap-2 text-sm">
                    <Checkbox
                      id="rotate-payload-field"
                      checked={rotatePayloadField}
                      onCheckedChange={setRotatePayloadField}
                    />
                    <span>Rotate payload field</span>
                  </label>
                  <label htmlFor="rotate-result-field" className="flex items-center gap-2 text-sm">
                    <Checkbox
                      id="rotate-result-field"
                      checked={rotateResultField}
                      onCheckedChange={setRotateResultField}
                    />
                    <span>Rotate result field</span>
                  </label>
                </div>

                {rotationMode === 'execute' && (
                  <label htmlFor="rotation-confirmed" className="flex items-start gap-2 text-sm text-red-700">
                    <Checkbox
                      id="rotation-confirmed"
                      checked={rotationConfirmed}
                      onCheckedChange={setRotationConfirmed}
                    />
                    <span>I confirm this will execute live key rotation.</span>
                  </label>
                )}
              </div>

              <Button
                onClick={() => {
                  void handleSubmitRotation();
                }}
                disabled={rotationSubmitting}
                loading={rotationSubmitting}
                loadingText="Submitting..."
              >
                Submit rotation request
              </Button>
            </div>
          </div>

          {currentRotationRun && (
            <div className="space-y-2 rounded-lg border p-4" data-testid="maintenance-rotation-current">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium">Current rotation run</p>
                  <p className="text-xs text-muted-foreground">{currentRotationRun.scope_summary}</p>
                </div>
                <Badge data-testid="maintenance-rotation-status">{currentRotationRun.status}</Badge>
              </div>
              <div className="grid gap-2 text-sm md:grid-cols-2">
                <p>Mode: {currentRotationRun.mode}</p>
                <p>Requested by: {currentRotationRun.requested_by_label ?? 'Unknown'}</p>
                <p>Created: {formatTimestamp(currentRotationRun.created_at)}</p>
                <p>Started: {formatTimestamp(currentRotationRun.started_at)}</p>
                <p>Completed: {formatTimestamp(currentRotationRun.completed_at)}</p>
                <p>Affected count: {currentRotationRun.affected_count ?? 0}</p>
              </div>
              {currentRotationRun.error_message && (
                <p className="text-sm text-red-700">{currentRotationRun.error_message}</p>
              )}
              {['queued', 'running'].includes(currentRotationRun.status) && (
                <p className="text-xs text-muted-foreground">
                  Polling authoritative run status until this request reaches a terminal state.
                </p>
              )}
            </div>
          )}

          <div className="space-y-2 border-t pt-4" data-testid="maintenance-rotation-history">
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm font-medium">Rotation history</p>
              {rotationHistoryLoading && (
                <span className="text-xs text-muted-foreground">Refreshing...</span>
              )}
            </div>
            {rotationHistory.length === 0 ? (
              <p className="text-xs text-muted-foreground">No rotation history yet.</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Status</TableHead>
                    <TableHead>Requested</TableHead>
                    <TableHead>Affected</TableHead>
                    <TableHead>Initiated by</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rotationHistory.map((entry) => (
                    <TableRow key={entry.id} data-testid="maintenance-rotation-history-row">
                      <TableCell>{entry.status}</TableCell>
                      <TableCell>{formatTimestamp(entry.created_at)}</TableCell>
                      <TableCell>{entry.affected_count ?? 0}</TableCell>
                      <TableCell>{entry.requested_by_label ?? 'Unknown'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
