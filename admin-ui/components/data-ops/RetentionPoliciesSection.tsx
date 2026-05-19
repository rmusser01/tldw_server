'use client';

import { Fragment, useCallback, useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useToast } from '@/components/ui/toast';
import { api } from '@/lib/api-client';
import type { RetentionPolicy, RetentionPolicyPreviewResponse } from '@/types';
import { AlertTriangle, ShieldCheck } from 'lucide-react';

type RetentionPoliciesSectionProps = {
  refreshSignal: number;
};

type RetentionImpactCounts = {
  auditLogEntries: number;
  jobRecords: number;
  backupFiles: number;
};

type RetentionImpactPreview = {
  currentDays: number;
  newDays: number;
  counts: RetentionImpactCounts;
  previewSignature: string;
  notes: string[];
};

const toNonNegativeNumber = (value: unknown): number | null => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return null;
  return Math.floor(value);
};

const parsePositiveDays = (raw: string): number | null => {
  if (!raw.trim()) return null;
  const value = Number(raw.trim());
  if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) return null;
  return value;
};

const parseImpactCounts = (payload: unknown): RetentionImpactCounts => {
  if (!payload || typeof payload !== 'object') {
    return { auditLogEntries: 0, jobRecords: 0, backupFiles: 0 };
  }
  const root = payload as Record<string, unknown>;
  const candidates: Array<Record<string, unknown>> = [
    root,
    typeof root.estimate === 'object' && root.estimate ? root.estimate as Record<string, unknown> : {},
    typeof root.estimated === 'object' && root.estimated ? root.estimated as Record<string, unknown> : {},
    typeof root.preview === 'object' && root.preview ? root.preview as Record<string, unknown> : {},
    typeof root.impact === 'object' && root.impact ? root.impact as Record<string, unknown> : {},
    typeof root.counts === 'object' && root.counts ? root.counts as Record<string, unknown> : {},
  ];

  const readCount = (keys: string[]): number => {
    for (const candidate of candidates) {
      for (const key of keys) {
        const value = toNonNegativeNumber(candidate[key]);
        if (value !== null) return value;
      }
    }
    return 0;
  };

  return {
    auditLogEntries: readCount(['audit_log_entries', 'audit_logs', 'audit_entries', 'audit_count']),
    jobRecords: readCount(['job_records', 'jobs', 'job_count']),
    backupFiles: readCount(['backup_files', 'backups', 'backup_count']),
  };
};

export const RetentionPoliciesSection = ({ refreshSignal }: RetentionPoliciesSectionProps) => {
  const { success, error: showError } = useToast();

  const [policies, setPolicies] = useState<RetentionPolicy[]>([]);
  const [policyError, setPolicyError] = useState('');
  const [policyLoading, setPolicyLoading] = useState(true);
  const [policyEdits, setPolicyEdits] = useState<Record<string, string>>({});
  const [policyPreviewLoading, setPolicyPreviewLoading] = useState<Record<string, boolean>>({});
  const [policyPreviews, setPolicyPreviews] = useState<Record<string, RetentionImpactPreview>>({});
  const [policyPreviewAcknowledged, setPolicyPreviewAcknowledged] = useState<Record<string, boolean>>({});
  const [policySaving, setPolicySaving] = useState<Record<string, boolean>>({});

  const loadPolicies = useCallback(async () => {
    try {
      setPolicyLoading(true);
      setPolicyError('');
      const data = await api.getRetentionPolicies();
      setPolicies(data.policies);
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to load retention policies';
      setPolicyError(message);
      setPolicies([]);
    } finally {
      setPolicyLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPolicies();
  }, [loadPolicies, refreshSignal]);

  const clearPolicyPreview = (policyKey: string) => {
    setPolicyPreviews((prev) => {
      if (!prev[policyKey]) return prev;
      const next = { ...prev };
      delete next[policyKey];
      return next;
    });
    setPolicyPreviewAcknowledged((prev) => ({ ...prev, [policyKey]: false }));
  };

  const handlePreviewImpact = async (policy: RetentionPolicy) => {
    const raw = policyEdits[policy.key] ?? (policy.days?.toString() ?? '');
    const days = parsePositiveDays(raw);
    if (days === null) {
      showError('Invalid value', 'Retention days must be a positive whole number before preview.');
      return;
    }
    const currentDays = typeof policy.days === 'number' && policy.days > 0
      ? policy.days
      : days;

    setPolicyPreviewLoading((prev) => ({ ...prev, [policy.key]: true }));
    try {
      const previewResponse = await api.previewRetentionPolicyImpact(policy.key, {
        current_days: currentDays,
        days,
      });
      const typedPreview = previewResponse as RetentionPolicyPreviewResponse;
      const counts = parseImpactCounts(typedPreview.counts);
      if (!typedPreview.preview_signature?.trim()) {
        throw new Error('Backend retention preview did not return a preview signature.');
      }
      setPolicyPreviews((prev) => ({
        ...prev,
        [policy.key]: {
          currentDays: typedPreview.current_days,
          newDays: typedPreview.new_days,
          counts,
          previewSignature: typedPreview.preview_signature,
          notes: Array.isArray(typedPreview.notes) ? typedPreview.notes : [],
        },
      }));
      setPolicyPreviewAcknowledged((prev) => ({ ...prev, [policy.key]: false }));
    } catch (err: unknown) {
      clearPolicyPreview(policy.key);
      const message = err instanceof Error && err.message
        ? err.message
        : 'Failed to preview retention policy impact';
      showError('Preview failed', message);
    } finally {
      setPolicyPreviewLoading((prev) => ({ ...prev, [policy.key]: false }));
    }
  };

  const handlePolicyUpdate = async (policy: RetentionPolicy) => {
    const raw = policyEdits[policy.key] ?? (policy.days?.toString() ?? '');
    const value = parsePositiveDays(raw);
    if (value === null) {
      showError('Invalid value', 'Retention days must be a positive whole number.');
      return;
    }
    const preview = policyPreviews[policy.key];
    if (!preview || preview.newDays !== value) {
      showError('Preview required', 'Run impact preview for the latest value before saving.');
      return;
    }
    if (!policyPreviewAcknowledged[policy.key]) {
      showError('Confirmation required', 'Check "I understand" before saving this retention change.');
      return;
    }

    setPolicySaving((prev) => ({ ...prev, [policy.key]: true }));
    try {
      await api.updateRetentionPolicy(policy.key, {
        days: value,
        preview_signature: preview.previewSignature,
      });
      success('Retention updated', `${policy.key} set to ${value} days.`);
      setPolicyEdits((prev) => {
        const next = { ...prev };
        delete next[policy.key];
        return next;
      });
      clearPolicyPreview(policy.key);
      await loadPolicies();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to update retention policy';
      showError('Update failed', message);
    } finally {
      setPolicySaving((prev) => ({ ...prev, [policy.key]: false }));
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5" />
          Retention Policies
        </CardTitle>
        <CardDescription>Adjust cleanup windows for system datasets.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {policyError && (
          <Alert variant="destructive">
            <AlertDescription>{policyError}</AlertDescription>
          </Alert>
        )}
        <Alert className="bg-yellow-50 border-yellow-200">
          <AlertTriangle className="h-4 w-4 text-yellow-600" />
          <AlertDescription className="text-yellow-800">
            Retention policy changes apply immediately and persist across restarts. Lower values can delete data
            sooner than expected, so review carefully before saving.
          </AlertDescription>
        </Alert>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Policy</TableHead>
              <TableHead>Description</TableHead>
              <TableHead>Days</TableHead>
              <TableHead className="text-right">Action</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {policyLoading ? (
              <TableRow>
                <TableCell colSpan={4} className="text-muted-foreground">
                  Loading retention policies...
                </TableCell>
              </TableRow>
            ) : policies.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} className="text-muted-foreground">
                  No retention policies found.
                </TableCell>
              </TableRow>
            ) : (
              policies.map((policy) => {
                const draft = policyEdits[policy.key];
                const value = draft ?? (policy.days?.toString() ?? '');
                const parsedDays = parsePositiveDays(value);
                const preview = policyPreviews[policy.key];
                const previewCurrent = preview && preview.newDays === parsedDays ? preview : null;
                const saveEnabled = Boolean(
                  parsedDays !== null
                  && previewCurrent
                  && policyPreviewAcknowledged[policy.key]
                  && !policySaving[policy.key]
                );
                return (
                  <Fragment key={policy.key}>
                    <TableRow>
                      <TableCell className="font-mono text-xs">{policy.key}</TableCell>
                      <TableCell>{policy.description || '—'}</TableCell>
                      <TableCell className="w-40">
                        <Input
                          value={value}
                          onChange={(event) => {
                            setPolicyEdits((prev) => ({ ...prev, [policy.key]: event.target.value }));
                            clearPolicyPreview(policy.key);
                          }}
                        />
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => { void handlePreviewImpact(policy); }}
                            disabled={policyPreviewLoading[policy.key]}
                            loading={policyPreviewLoading[policy.key]}
                            loadingText="Previewing..."
                          >
                            Preview impact
                          </Button>
                          <Button
                            size="sm"
                            onClick={() => { void handlePolicyUpdate(policy); }}
                            disabled={!saveEnabled}
                            loading={policySaving[policy.key]}
                            loadingText="Saving..."
                          >
                            Save
                          </Button>
                        </div>
                        {!saveEnabled && (
                          <p className="mt-1 text-xs text-muted-foreground">
                            Run preview and confirm impact to enable save.
                          </p>
                        )}
                      </TableCell>
                    </TableRow>

                    {previewCurrent && (() => {
                      const totalAffected = previewCurrent.counts.auditLogEntries + previewCurrent.counts.jobRecords + previewCurrent.counts.backupFiles;
                      const severityClass = totalAffected > 1000
                        ? 'border-red-400 bg-red-50'
                        : totalAffected > 100
                          ? 'border-yellow-400 bg-yellow-50'
                          : 'bg-muted/40';
                      const textClass = totalAffected > 1000
                        ? 'text-red-700 font-semibold'
                        : totalAffected > 100
                          ? 'text-yellow-700'
                          : '';
                      return (
                      <TableRow data-testid={`retention-preview-row-${policy.key}`}>
                        <TableCell colSpan={4}>
                          <div className={`rounded-md border p-3 space-y-2 ${severityClass}`}>
                            <p className={`text-sm ${textClass}`} data-testid={`retention-preview-text-${policy.key}`}>
                              {totalAffected > 1000 && <AlertTriangle className="inline h-4 w-4 mr-1 text-red-600" />}
                              Changing from {previewCurrent.currentDays} to {previewCurrent.newDays} days will delete approximately{' '}
                              {previewCurrent.counts.auditLogEntries} audit log entries, {previewCurrent.counts.jobRecords}{' '}
                              job records, {previewCurrent.counts.backupFiles} backup files.
                            </p>
                            {previewCurrent.notes.map((note) => (
                              <p key={note} className="text-xs text-muted-foreground">
                                {note}
                              </p>
                            ))}
                            <label
                              htmlFor={`retention-preview-ack-${policy.key}`}
                              className="flex items-start gap-2 text-sm"
                            >
                              <Checkbox
                                id={`retention-preview-ack-${policy.key}`}
                                checked={Boolean(policyPreviewAcknowledged[policy.key])}
                                onCheckedChange={(checked) =>
                                  setPolicyPreviewAcknowledged((prev) => ({ ...prev, [policy.key]: checked === true }))
                                }
                              />
                              <span>I understand this change can permanently delete historical data.</span>
                            </label>
                          </div>
                        </TableCell>
                      </TableRow>
                      );
                    })()}
                  </Fragment>
                );
              })
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
};
