'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { CardSkeleton } from '@/components/ui/skeleton';
import { useToast } from '@/components/ui/toast';
import { api } from '@/lib/api-client';
import { AccessibleIconButton } from '@/components/ui/accessible-icon-button';
import type { CompliancePosture, ComplianceReportSchedule } from '@/types';
import { Shield, ShieldCheck, Key, Users, FileText, Database, RefreshCw, ExternalLink, Plus, Trash2, Send, Calendar } from 'lucide-react';

type ComplianceGrade = 'A' | 'B' | 'C' | 'D' | 'F';

function getGrade(score: number): ComplianceGrade {
  if (score >= 90) return 'A';
  if (score >= 80) return 'B';
  if (score >= 70) return 'C';
  if (score >= 60) return 'D';
  return 'F';
}

function getGradeColor(grade: ComplianceGrade): string {
  switch (grade) {
    case 'A': return 'text-green-600 dark:text-green-400';
    case 'B': return 'text-blue-600 dark:text-blue-400';
    case 'C': return 'text-yellow-600 dark:text-yellow-400';
    case 'D': return 'text-orange-600 dark:text-orange-400';
    case 'F': return 'text-red-600 dark:text-red-400';
  }
}

function getGradeBadgeVariant(grade: ComplianceGrade): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (grade) {
    case 'A':
    case 'B':
      return 'default';
    case 'C':
      return 'secondary';
    case 'D':
    case 'F':
      return 'destructive';
  }
}

function getPercentColor(pct: number): string {
  if (pct >= 90) return 'text-green-600 dark:text-green-400';
  if (pct >= 70) return 'text-yellow-600 dark:text-yellow-400';
  return 'text-red-600 dark:text-red-400';
}

function ProgressBar({ value, className }: { value: number; className?: string }) {
  const clamped = Math.max(0, Math.min(100, value));
  let barColor = 'bg-red-500';
  if (clamped >= 90) barColor = 'bg-green-500';
  else if (clamped >= 70) barColor = 'bg-yellow-500';
  else if (clamped >= 50) barColor = 'bg-orange-500';

  return (
    <div className={`h-2 w-full rounded-full bg-muted ${className ?? ''}`}>
      <div
        className={`h-full rounded-full transition-all duration-500 ${barColor}`}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}

function CompliancePageContent() {
  const { error: showError } = useToast();
  const [posture, setPosture] = useState<CompliancePosture | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Report Schedules state
  const [schedules, setSchedules] = useState<ComplianceReportSchedule[]>([]);
  const [schedulesLoading, setSchedulesLoading] = useState(true);
  const [schedulesError, setSchedulesError] = useState<string | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [createFrequency, setCreateFrequency] = useState<string>('weekly');
  const [createFormat, setCreateFormat] = useState<string>('html');
  const [createRecipients, setCreateRecipients] = useState('');
  const [createBusy, setCreateBusy] = useState(false);
  const [createError, setCreateError] = useState('');
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());
  const [sendingIds, setSendingIds] = useState<Set<string>>(new Set());
  const [sendResult, setSendResult] = useState<string | null>(null);

  const fetchPosture = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getCompliancePosture();
      setPosture(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to load compliance posture';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchSchedules = useCallback(async () => {
    setSchedulesLoading(true);
    setSchedulesError(null);
    try {
      const data = await api.getReportSchedules();
      setSchedules(Array.isArray(data?.items) ? data.items : []);
    } catch (err: unknown) {
      setSchedulesError(err instanceof Error ? err.message : 'Failed to load schedules');
    } finally {
      setSchedulesLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPosture();
    fetchSchedules();
  }, [fetchPosture, fetchSchedules]);

  const handleCreateSchedule = async () => {
    setCreateError('');
    const recipientList = createRecipients
      .split(/[,;\n]+/)
      .map((e) => e.trim())
      .filter(Boolean);
    if (recipientList.length === 0) {
      setCreateError('At least one recipient email is required.');
      return;
    }
    try {
      setCreateBusy(true);
      await api.createReportSchedule({
        frequency: createFrequency,
        recipients: recipientList,
        format: createFormat,
        enabled: true,
      });
      setShowCreateDialog(false);
      setCreateRecipients('');
      void fetchSchedules();
    } catch (err: unknown) {
      setCreateError(err instanceof Error ? err.message : 'Failed to create schedule');
    } finally {
      setCreateBusy(false);
    }
  };

  const handleDeleteSchedule = async (id: string) => {
    try {
      setDeletingIds((prev) => new Set(prev).add(id));
      await api.deleteReportSchedule(id);
      void fetchSchedules();
    } catch (err: unknown) {
      showError(err instanceof Error ? err.message : 'Failed to delete schedule');
    } finally {
      setDeletingIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  };

  const handleSendNow = async (id: string) => {
    try {
      setSendingIds((prev) => new Set(prev).add(id));
      setSendResult(null);
      const result = await api.sendReportNow(id);
      setSendResult(
        `Report sent to ${result.sent_count}/${result.total_recipients} recipients.` +
        (result.errors.length > 0 ? ` Errors: ${result.errors.join('; ')}` : ''),
      );
      void fetchSchedules();
    } catch (err: unknown) {
      setSendResult(err instanceof Error ? err.message : 'Failed to send report');
    } finally {
      setSendingIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  };

  const handleToggleEnabled = async (schedule: ComplianceReportSchedule) => {
    try {
      await api.updateReportSchedule(schedule.id, { enabled: !schedule.enabled });
      void fetchSchedules();
    } catch (err: unknown) {
      showError(err instanceof Error ? err.message : 'Failed to toggle schedule');
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Compliance Posture</h1>
            <p className="text-muted-foreground">Loading compliance metrics...</p>
          </div>
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <CardHeader className="pb-2">
                <div className="h-4 w-24 animate-pulse rounded bg-muted" />
              </CardHeader>
              <CardContent>
                <div className="h-8 w-16 animate-pulse rounded bg-muted" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  const postureAvailable = !error && posture;
  const grade = postureAvailable ? getGrade(posture.overall_score) : null;
  const gradeColor = grade ? getGradeColor(grade) : '';
  const gradeBadge = grade ? getGradeBadgeVariant(grade) : 'secondary';
  const mfaColor = postureAvailable ? getPercentColor(posture.mfa_adoption_pct) : '';
  const keyColor = postureAvailable ? getPercentColor(posture.key_rotation_compliance_pct) : '';
  const usersWithoutMfa = postureAvailable ? posture.total_users - posture.mfa_enabled_count : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Compliance Posture</h1>
          <p className="text-muted-foreground">Security and compliance overview</p>
        </div>
        <Button variant="outline" size="sm" onClick={fetchPosture}>
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>
            Unable to load compliance posture: {error}
            <Button variant="outline" size="sm" className="ml-2" onClick={fetchPosture}>
              <RefreshCw className="mr-1 h-3 w-3" /> Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Cards grid */}
      {postureAvailable && <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* Overall Score */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Overall Score</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2">
              <span className={`text-3xl font-bold ${gradeColor}`}>
                {posture.overall_score.toFixed(0)}
              </span>
              <Badge variant={gradeBadge} data-testid="compliance-grade">{grade}</Badge>
            </div>
            <ProgressBar value={posture.overall_score} className="mt-3" />
            <p className="mt-2 text-xs text-muted-foreground">
              Weighted: MFA 40% + Key rotation 40% + Audit 20%
            </p>
          </CardContent>
        </Card>

        {/* MFA Adoption */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">MFA Adoption</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2">
              <span className={`text-3xl font-bold ${mfaColor}`}>
                {posture.mfa_adoption_pct.toFixed(0)}%
              </span>
            </div>
            <ProgressBar value={posture.mfa_adoption_pct} className="mt-3" />
            <p className="mt-2 text-xs text-muted-foreground">
              {posture.mfa_enabled_count} of {posture.total_users} users enabled
            </p>
            {usersWithoutMfa > 0 && (
              <Link
                href="/users?is_active=true"
                className="mt-2 inline-flex items-center gap-1 text-xs text-blue-600 hover:underline dark:text-blue-400"
              >
                {usersWithoutMfa} without MFA
                <ExternalLink className="h-3 w-3" />
              </Link>
            )}
          </CardContent>
        </Card>

        {/* Key Rotation */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Key Rotation</CardTitle>
            <Key className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2">
              <span className={`text-3xl font-bold ${keyColor}`}>
                {posture.key_rotation_compliance_pct.toFixed(0)}%
              </span>
            </div>
            <ProgressBar value={posture.key_rotation_compliance_pct} className="mt-3" />
            <p className="mt-2 text-xs text-muted-foreground">
              {posture.keys_total - posture.keys_needing_rotation} of {posture.keys_total} keys compliant
              ({posture.rotation_threshold_days}d threshold)
            </p>
            {posture.keys_needing_rotation > 0 && (
              <Link
                href="/api-keys"
                className="mt-2 inline-flex items-center gap-1 text-xs text-blue-600 hover:underline dark:text-blue-400"
              >
                {posture.keys_needing_rotation} need rotation
                <ExternalLink className="h-3 w-3" />
              </Link>
            )}
          </CardContent>
        </Card>

        {/* Audit Logging */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Audit Logging</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-2">
              {posture.audit_logging_enabled ? (
                <ShieldCheck className="h-8 w-8 text-green-600 dark:text-green-400" />
              ) : (
                <Shield className="h-8 w-8 text-red-600 dark:text-red-400" />
              )}
              <span className="text-lg font-semibold">
                {posture.audit_logging_enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              {posture.audit_logging_enabled
                ? 'All admin actions are being recorded'
                : 'Audit logging is not active'}
            </p>
            <Link
              href="/audit"
              className="mt-2 inline-flex items-center gap-1 text-xs text-blue-600 hover:underline dark:text-blue-400"
            >
              View audit logs
              <ExternalLink className="h-3 w-3" />
            </Link>
          </CardContent>
        </Card>
      </div>}

      {/* Report Schedules */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <div>
            <CardTitle>Report Schedules</CardTitle>
            <CardDescription>
              Configure automated compliance report delivery to recipients.
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={() => setShowCreateDialog(true)}>
            <Plus className="mr-2 h-4 w-4" />
            New Schedule
          </Button>
        </CardHeader>
        <CardContent className="space-y-4">
          {schedulesError && (
            <Alert variant="destructive">
              <AlertDescription>{schedulesError}</AlertDescription>
            </Alert>
          )}
          {sendResult && (
            <Alert>
              <AlertDescription>{sendResult}</AlertDescription>
            </Alert>
          )}
          {schedulesLoading ? (
            <p className="text-sm text-muted-foreground">Loading schedules...</p>
          ) : schedules.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No report schedules configured. Create one to start receiving automated compliance reports.
            </p>
          ) : (
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Frequency</TableHead>
                    <TableHead>Recipients</TableHead>
                    <TableHead>Format</TableHead>
                    <TableHead>Enabled</TableHead>
                    <TableHead>Last Sent</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {schedules.map((sched) => (
                    <TableRow key={sched.id} data-testid={`schedule-row-${sched.id}`}>
                      <TableCell className="capitalize">{sched.frequency}</TableCell>
                      <TableCell>
                        <span className="text-sm" title={sched.recipients.join(', ')}>
                          {sched.recipients.length} recipient{sched.recipients.length !== 1 ? 's' : ''}
                        </span>
                      </TableCell>
                      <TableCell className="uppercase text-xs">{sched.format}</TableCell>
                      <TableCell>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-auto px-2 py-0.5"
                          onClick={() => handleToggleEnabled(sched)}
                          aria-label={`Toggle schedule ${sched.frequency} ${sched.enabled ? 'off' : 'on'}`}
                        >
                          <Badge variant={sched.enabled ? 'default' : 'secondary'}>
                            {sched.enabled ? 'Active' : 'Paused'}
                          </Badge>
                        </Button>
                      </TableCell>
                      <TableCell>
                        {sched.last_sent_at
                          ? new Date(sched.last_sent_at).toLocaleDateString()
                          : 'Never'}
                      </TableCell>
                      <TableCell className="text-right">
                        <span className="inline-flex gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleSendNow(sched.id)}
                            disabled={sendingIds.has(sched.id)}
                            title="Send report now"
                            aria-label="Send report now"
                          >
                            <Send className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleDeleteSchedule(sched.id)}
                            disabled={deletingIds.has(sched.id)}
                            className="text-destructive hover:text-destructive"
                            title="Delete schedule"
                            aria-label="Delete schedule"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </span>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Create Schedule Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>New Report Schedule</DialogTitle>
            <DialogDescription>
              Configure a recurring compliance report to be sent to specified recipients.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label htmlFor="schedule-frequency">Frequency</Label>
              <select
                id="schedule-frequency"
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                value={createFrequency}
                onChange={(e) => setCreateFrequency(e.target.value)}
              >
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="schedule-format">Format</Label>
              <select
                id="schedule-format"
                className="w-full rounded-md border bg-background px-3 py-2 text-sm"
                value={createFormat}
                onChange={(e) => setCreateFormat(e.target.value)}
              >
                <option value="html">HTML</option>
                <option value="json">JSON</option>
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="schedule-recipients">Recipients (comma-separated emails)</Label>
              <Input
                id="schedule-recipients"
                placeholder="admin@example.com, security@example.com"
                value={createRecipients}
                onChange={(e) => setCreateRecipients(e.target.value)}
              />
            </div>
            {createError && (
              <Alert variant="destructive">
                <AlertDescription>{createError}</AlertDescription>
              </Alert>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateSchedule} disabled={createBusy}>
              {createBusy ? 'Creating...' : 'Create Schedule'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default function CompliancePage() {
  return (
    <PermissionGuard variant="route" requireAuth role={['admin', 'super_admin', 'owner']}>
      <ResponsiveLayout>
        <CompliancePageContent />
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
