'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { EmptyState } from '@/components/ui/empty-state';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/toast';
import { RefreshCw, Briefcase, Filter, AlertTriangle, Eye, RotateCcw, XCircle, Repeat, Clock, Plus, X, Paperclip, Trash2 } from 'lucide-react';
import { ExportMenu } from '@/components/ui/export-menu';
import { exportJobs, ExportFormat } from '@/lib/export';
import { AccessibleIconButton } from '@/components/ui/accessible-icon-button';
import { api, ApiError } from '@/lib/api-client';
import { formatBytes, formatDateTime, formatDuration } from '@/lib/format';
import {
  normalizeMonitoringMetricsPayload,
} from '@/lib/monitoring-metrics';

interface SlaPolicy {
  id: string;
  name: string;
  job_type?: string;
  domain?: string;
  queue?: string;
  max_processing_time_seconds: number;
  max_wait_time_seconds: number;
  max_queue_latency_seconds?: number;
  max_duration_seconds?: number;
  priority_boost?: number;
  enabled: boolean;
}

interface SlaBreach {
  job_id: number;
  domain: string;
  queue: string;
  job_type: string;
  status: string;
  breach_kinds: string[];
  wait_seconds?: number;
  max_wait_seconds?: number;
  processing_seconds?: number;
  max_processing_seconds?: number;
}

interface JobAttachment {
  id: string;
  name: string;
  content_type: string;
  size_bytes: number;
  created_at?: string;
}

interface JobItem {
  id: number;
  uuid?: string | null;
  domain: string;
  queue: string;
  job_type: string;
  status: string;
  priority?: number | null;
  retry_count?: number | null;
  max_retries?: number | null;
  available_at?: string | null;
  created_at?: string | null;
  acquired_at?: string | null;
  started_at?: string | null;
  leased_until?: string | null;
  completed_at?: string | null;
  parent_id?: number | string | null;
  parent_job_id?: number | string | null;
  parent_job_uuid?: string | null;
  depends_on_job_uuid?: string | null;
  depends_on?: unknown;
  child_job_ids?: unknown;
  child_job_uuids?: unknown;
}

interface JobDetail extends JobItem {
  payload?: unknown;
  result?: unknown;
  archived?: boolean;
  error_message?: string | null;
  last_error?: string | null;
  progress_percent?: number | null;
  progress_message?: string | null;
}

interface QueueStats {
  domain: string;
  queue: string;
  job_type: string;
  queued: number;
  scheduled: number;
  processing: number;
  quarantined: number;
}

interface StaleGroup {
  domain: string;
  queue: string;
  count: number;
}

interface QueueDepthPoint {
  timestamp: string;
  label: string;
  depth: number;
}

interface QueueThroughputSummary {
  completedJobs24h: number;
  jobsCompletedPerHour: number;
  averageProcessingSeconds: number;
}

type JobRelationship = {
  parent: JobItem;
  child: JobItem;
  source: 'parent_ref' | 'child_ref';
};

const parseTimestampMs = (value?: string | null): number | null => {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const buildQueueThroughputSummary = (
  jobs: JobItem[],
  now: Date = new Date(),
): QueueThroughputSummary => {
  const windowEnd = now.getTime();
  const windowStart = windowEnd - (24 * 60 * 60 * 1000);
  let completedJobs24h = 0;
  let totalProcessingSeconds = 0;
  let durationSamples = 0;

  jobs.forEach((job) => {
    const completedAtMs = parseTimestampMs(job.completed_at);
    if (completedAtMs === null || completedAtMs < windowStart || completedAtMs > windowEnd) {
      return;
    }
    completedJobs24h += 1;
    const startedAtMs = parseTimestampMs(job.started_at);
    if (startedAtMs === null || startedAtMs > completedAtMs) return;
    totalProcessingSeconds += (completedAtMs - startedAtMs) / 1000;
    durationSamples += 1;
  });

  return {
    completedJobs24h,
    jobsCompletedPerHour: completedJobs24h / 24,
    averageProcessingSeconds: durationSamples > 0 ? totalProcessingSeconds / durationSamples : 0,
  };
};

const formatJobDateTime = (value?: string | null) =>
  formatDateTime(value, { fallback: '—' });

const formatDurationDisplay = (value?: number | null) =>
  formatDuration(value, { fallback: '—' });

const statusBadge = (status: string) => {
  const normalized = status.toLowerCase();
  if (normalized === 'completed') return 'bg-green-500';
  if (normalized === 'processing') return 'bg-blue-500';
  if (normalized === 'queued' || normalized === 'scheduled') return 'bg-yellow-500';
  if (normalized === 'failed' || normalized === 'cancelled' || normalized === 'quarantined') {
    return 'bg-red-500';
  }
  return 'bg-muted text-muted-foreground';
};

const clampLimit = (value: string) => {
  if (!value.trim()) return '';
  const parsed = Number(value);
  if (Number.isNaN(parsed)) return '';
  const clamped = Math.min(500, Math.max(1, Math.floor(parsed)));
  return String(clamped);
};

const toRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object'
    ? (value as Record<string, unknown>)
    : null;

const toReference = (value: unknown): string | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return String(Math.trunc(value));
  }
  if (typeof value === 'string' && value.trim()) {
    return value.trim();
  }
  return null;
};

const collectReferenceValues = (value: unknown, out: Set<string>) => {
  if (value === null || value === undefined) return;
  if (Array.isArray(value)) {
    value.forEach((entry) => {
      collectReferenceValues(entry, out);
    });
    return;
  }
  const direct = toReference(value);
  if (direct) {
    out.add(direct);
    return;
  }
  const record = toRecord(value);
  if (!record) return;
  const idLike = toReference(record.id ?? record.job_id ?? record.uuid ?? record.job_uuid);
  if (idLike) {
    out.add(idLike);
  }
};

const extractParentReferences = (job: Record<string, unknown>): string[] => {
  const refs = new Set<string>();
  [
    job.parent_id,
    job.parent_job_id,
    job.parent_job_uuid,
    job.depends_on_job_uuid,
    job.depends_on,
  ].forEach((candidate) => {
    collectReferenceValues(candidate, refs);
  });
  const payload = toRecord(job.payload);
  if (payload) {
    [
      payload.parent_id,
      payload.parent_job_id,
      payload.parent_job_uuid,
      payload.depends_on_job_uuid,
      payload.depends_on,
    ].forEach((candidate) => {
      collectReferenceValues(candidate, refs);
    });
  }
  return [...refs];
};

const extractChildReferences = (job: Record<string, unknown>): string[] => {
  const refs = new Set<string>();
  [job.child_job_ids, job.child_job_uuids].forEach((candidate) => {
    collectReferenceValues(candidate, refs);
  });
  const payload = toRecord(job.payload);
  if (payload) {
    [payload.child_job_ids, payload.child_job_uuids, payload.children].forEach((candidate) => {
      collectReferenceValues(candidate, refs);
    });
  }
  return [...refs];
};

export default function JobsPage() {
  const confirm = useConfirm();
  const { success, error: showError } = useToast();
  const [jobs, setJobs] = useState<JobItem[]>([]);
  const [stats, setStats] = useState<QueueStats[]>([]);
  const [staleGroups, setStaleGroups] = useState<StaleGroup[]>([]);
  const [queueDepthHistory, setQueueDepthHistory] = useState<QueueDepthPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [routesUnavailable, setRoutesUnavailable] = useState(false);
  const [selectedJob, setSelectedJob] = useState<JobItem | null>(null);
  const [jobDetail, setJobDetail] = useState<JobDetail | null>(null);
  const [jobDetailLoading, setJobDetailLoading] = useState(false);
  const [jobDetailError, setJobDetailError] = useState('');

  // SLA Policies
  const [slaPolicies, setSlaPolicies] = useState<SlaPolicy[]>([]);
  const [slaPoliciesLoading, setSlaPoliciesLoading] = useState(false);
  const [slaPoliciesError, setSlaPoliciesError] = useState<string | null>(null);
  const [showSlaForm, setShowSlaForm] = useState(false);
  const [slaFormSaving, setSlaFormSaving] = useState(false);
  const [slaFormName, setSlaFormName] = useState('');
  const [slaFormJobType, setSlaFormJobType] = useState('');
  const [slaFormMaxProcessing, setSlaFormMaxProcessing] = useState('3600');
  const [slaFormMaxWait, setSlaFormMaxWait] = useState('300');
  const [slaFormEnabled, setSlaFormEnabled] = useState(true);
  const [slaFormDomain, setSlaFormDomain] = useState('');
  const [slaFormQueue, setSlaFormQueue] = useState('');

  // SLA Breaches
  const [slaBreaches, setSlaBreaches] = useState<SlaBreach[]>([]);

  // Job Attachments
  const [jobAttachments, setJobAttachments] = useState<JobAttachment[]>([]);
  const [attachmentsLoading, setAttachmentsLoading] = useState(false);

  const [filters, setFilters] = useState({
    domain: '',
    queue: '',
    status: '',
    job_type: '',
    limit: '100',
  });
  const clearJobFilters = useCallback(() => {
    setFilters({
      domain: '',
      queue: '',
      status: '',
      job_type: '',
      limit: '100',
    });
  }, []);

  const statsParams = useMemo(() => {
    const params: Record<string, string> = {};
    if (filters.domain) params.domain = filters.domain;
    if (filters.queue) params.queue = filters.queue;
    if (filters.job_type) params.job_type = filters.job_type;
    return params;
  }, [filters.domain, filters.queue, filters.job_type]);

  const listParams = useMemo(() => {
    const params: Record<string, string> = {};
    if (filters.domain) params.domain = filters.domain;
    if (filters.queue) params.queue = filters.queue;
    if (filters.status) params.status = filters.status;
    if (filters.job_type) params.job_type = filters.job_type;
    const normalizedLimit = clampLimit(filters.limit);
    if (normalizedLimit) params.limit = normalizedLimit;
    return params;
  }, [filters.domain, filters.queue, filters.status, filters.job_type, filters.limit]);

  const isNotFoundError = (err: unknown): boolean => {
    if (err instanceof ApiError) {
      return err.status === 404;
    }
    if (typeof err === 'object' && err !== null && 'status' in err) {
      return (err as { status?: number }).status === 404;
    }
    if (err instanceof Error) {
      return /not found|404/i.test(err.message);
    }
    return false;
  };

  const loadData = useCallback(async () => {
    setLoading(true);
    setSlaPoliciesLoading(true);
    setSlaPoliciesError(null);
    setError('');
    setRoutesUnavailable(false);

    const queueHistoryEnd = new Date();
    const queueHistoryStart = new Date(queueHistoryEnd.getTime() - (24 * 60 * 60 * 1000));
    const queueHistoryParams = {
      start: queueHistoryStart.toISOString(),
      end: queueHistoryEnd.toISOString(),
      granularity: '1h',
    };

    const [statsResult, jobsResult, staleResult, slaResult, queueHistoryResult, breachesResult] = await Promise.allSettled([
      api.getJobsStats(statsParams),
      api.getJobs(listParams),
      api.getJobsStale(statsParams),
      api.getJobSlaPolicies(),
      api.getMonitoringMetrics(queueHistoryParams),
      api.getJobSlaBreaches(statsParams),
    ]);

    const sawNotFound = [statsResult, jobsResult, staleResult].some(
      (result) => result.status === 'rejected' && isNotFoundError(result.reason)
    );
    setRoutesUnavailable(sawNotFound);

    if (statsResult.status === 'fulfilled') {
      setStats(Array.isArray(statsResult.value) ? statsResult.value : []);
    } else {
      setStats([]);
    }

    if (jobsResult.status === 'fulfilled') {
      setJobs(Array.isArray(jobsResult.value) ? jobsResult.value : []);
    } else {
      setJobs([]);
    }

    if (staleResult.status === 'fulfilled') {
      setStaleGroups(Array.isArray(staleResult.value) ? staleResult.value : []);
    } else {
      setStaleGroups([]);
    }

    if (slaResult.status === 'fulfilled') {
      const data = slaResult.value as { policies?: SlaPolicy[]; items?: SlaPolicy[] };
      setSlaPolicies(
        Array.isArray(data.policies) ? data.policies :
        Array.isArray(data.items) ? data.items :
        Array.isArray(data) ? data as SlaPolicy[] : []
      );
    } else {
      setSlaPolicies([]);
      setSlaPoliciesError(
        isNotFoundError(slaResult.reason)
          ? 'SLA policies endpoint is unavailable.'
          : 'Failed to load SLA policies.'
      );
    }

    if (breachesResult.status === 'fulfilled') {
      setSlaBreaches(Array.isArray(breachesResult.value) ? breachesResult.value as SlaBreach[] : []);
    } else {
      setSlaBreaches([]);
    }

    let nextQueueHistory: QueueDepthPoint[] = [];
    if (queueHistoryResult.status === 'fulfilled') {
      nextQueueHistory = normalizeMonitoringMetricsPayload(
        queueHistoryResult.value,
        queueHistoryParams.end,
      ).map((point) => ({
        timestamp: point.timestamp,
        label: point.label,
        depth: point.queueDepth,
      }));
    }

    setQueueDepthHistory(nextQueueHistory);

    setSlaPoliciesLoading(false);
    setLoading(false);
  }, [listParams, statsParams]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const breachedJobIds = useMemo(() => {
    const ids = new Set<number>();
    slaBreaches.forEach((b) => ids.add(b.job_id));
    return ids;
  }, [slaBreaches]);

  const breachByJobId = useMemo(() => {
    const map = new Map<number, SlaBreach>();
    slaBreaches.forEach((b) => map.set(b.job_id, b));
    return map;
  }, [slaBreaches]);

  const jobLookupByRef = useMemo(() => {
    const lookup = new Map<string, JobItem>();
    jobs.forEach((job) => {
      lookup.set(String(job.id), job);
      if (job.uuid) {
        lookup.set(job.uuid, job);
      }
    });
    return lookup;
  }, [jobs]);

  const dependencyRelationships = useMemo<JobRelationship[]>(() => {
    const links: JobRelationship[] = [];
    const seen = new Set<string>();

    const addLink = (
      parent: JobItem,
      child: JobItem,
      source: JobRelationship['source']
    ) => {
      if (parent.id === child.id) return;
      const key = `${parent.id}->${child.id}`;
      if (seen.has(key)) return;
      seen.add(key);
      links.push({ parent, child, source });
    };

    jobs.forEach((job) => {
      const record = job as unknown as Record<string, unknown>;
      extractParentReferences(record).forEach((parentRef) => {
        const parent = jobLookupByRef.get(parentRef);
        if (parent) {
          addLink(parent, job, 'parent_ref');
        }
      });
      extractChildReferences(record).forEach((childRef) => {
        const child = jobLookupByRef.get(childRef);
        if (child) {
          addLink(job, child, 'child_ref');
        }
      });
    });

    return links;
  }, [jobLookupByRef, jobs]);

  const selectedJobRelationships = useMemo(() => {
    const empty = {
      parentJobs: [] as JobItem[],
      childJobs: [] as JobItem[],
      unresolvedParentRefs: [] as string[],
      unresolvedChildRefs: [] as string[],
    };
    if (!selectedJob) return empty;

    const parentJobsById = new Map<number, JobItem>();
    const childJobsById = new Map<number, JobItem>();

    dependencyRelationships.forEach((relation) => {
      if (relation.child.id === selectedJob.id) {
        parentJobsById.set(relation.parent.id, relation.parent);
      }
      if (relation.parent.id === selectedJob.id) {
        childJobsById.set(relation.child.id, relation.child);
      }
    });

    const unresolvedParentRefs = new Set<string>();
    const unresolvedChildRefs = new Set<string>();
    const detailRecord = toRecord(jobDetail) ?? (selectedJob as unknown as Record<string, unknown>);

    extractParentReferences(detailRecord).forEach((parentRef) => {
      const match = jobLookupByRef.get(parentRef);
      if (match) {
        parentJobsById.set(match.id, match);
      } else {
        unresolvedParentRefs.add(parentRef);
      }
    });

    extractChildReferences(detailRecord).forEach((childRef) => {
      const match = jobLookupByRef.get(childRef);
      if (match) {
        childJobsById.set(match.id, match);
      } else {
        unresolvedChildRefs.add(childRef);
      }
    });

    return {
      parentJobs: [...parentJobsById.values()],
      childJobs: [...childJobsById.values()],
      unresolvedParentRefs: [...unresolvedParentRefs],
      unresolvedChildRefs: [...unresolvedChildRefs],
    };
  }, [dependencyRelationships, jobDetail, jobLookupByRef, selectedJob]);

  const handleReset = () => {
    setFilters({
      domain: '',
      queue: '',
      status: '',
      job_type: '',
      limit: '100',
    });
  };

  const handleOpenDetail = async (job: JobItem) => {
    setSelectedJob(job);
    setJobDetail(null);
    setJobDetailError('');
    setJobDetailLoading(true);
    setJobAttachments([]);
    setAttachmentsLoading(true);
    try {
      const [detail, attachments] = await Promise.allSettled([
        api.getJobDetail(job.id, { domain: job.domain }),
        api.getJobAttachments(String(job.id), { domain: job.domain }),
      ]);
      if (detail.status === 'fulfilled') {
        setJobDetail(detail.value as JobDetail);
      } else {
        const message = detail.reason instanceof Error ? detail.reason.message : 'Failed to load job detail';
        setJobDetailError(message);
      }
      if (attachments.status === 'fulfilled') {
        const data = attachments.value as { attachments?: JobAttachment[]; items?: JobAttachment[] };
        setJobAttachments(
          Array.isArray(data.attachments) ? data.attachments :
          Array.isArray(data.items) ? data.items :
          Array.isArray(data) ? data as JobAttachment[] : []
        );
      } else {
        const message = attachments.reason instanceof Error
          ? attachments.reason.message
          : 'Failed to load attachments';
        showError('Attachments unavailable', message);
      }
    } finally {
      setJobDetailLoading(false);
      setAttachmentsLoading(false);
    }
  };

  const handleCreateSlaPolicy = async () => {
    if (!slaFormName.trim()) {
      showError('Name required', 'Please enter a policy name');
      return;
    }
    if (!slaFormJobType.trim()) {
      showError('Job type required', 'Please enter a backend job type');
      return;
    }
    const maxProcessing = parseInt(slaFormMaxProcessing, 10);
    const maxWait = parseInt(slaFormMaxWait, 10);
    if (Number.isNaN(maxProcessing) || maxProcessing < 1) {
      showError('Invalid value', 'Max processing time must be at least 1 second');
      return;
    }
    if (Number.isNaN(maxWait) || maxWait < 1) {
      showError('Invalid value', 'Max wait time must be at least 1 second');
      return;
    }

    try {
      setSlaFormSaving(true);
      await api.createJobSlaPolicy({
        name: slaFormName.trim(),
        domain: slaFormDomain.trim() || 'default',
        queue: slaFormQueue.trim() || 'default',
        job_type: slaFormJobType.trim(),
        max_queue_latency_seconds: maxWait,
        max_duration_seconds: maxProcessing,
        enabled: slaFormEnabled,
      });
      success('SLA policy created', `Policy "${slaFormName}" has been created`);
      setShowSlaForm(false);
      setSlaFormName('');
      setSlaFormJobType('');
      setSlaFormDomain('');
      setSlaFormQueue('');
      setSlaFormMaxProcessing('3600');
      setSlaFormMaxWait('300');
      setSlaFormEnabled(true);
      void loadData();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to create SLA policy';
      showError('Create failed', message);
    } finally {
      setSlaFormSaving(false);
    }
  };

  const handleDeleteSlaPolicy = async (policy: SlaPolicy) => {
    const policyDomain = policy.domain || 'default';
    const policyQueue = policy.queue || 'default';
    const policyJobType = policy.job_type || policy.name || '';
    const confirmed = await confirm({
      title: 'Delete SLA policy',
      message: `Delete SLA policy for ${policyJobType} (${policyDomain}/${policyQueue})?`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'warning',
    });
    if (!confirmed) return;

    try {
      await api.deleteJobSlaPolicy({
        domain: policyDomain,
        queue: policyQueue,
        job_type: policyJobType,
      });
      success('Policy deleted', `SLA policy for "${policyJobType}" has been deleted`);
      void loadData();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete SLA policy';
      showError('Delete failed', message);
    }
  };

  const handleCloseDetail = () => {
    setSelectedJob(null);
    setJobDetail(null);
    setJobDetailError('');
    setJobDetailLoading(false);
  };

  const handleCancelJob = async (job: JobItem) => {
    const confirmed = await confirm({
      title: 'Cancel job',
      message: `Cancel job ${job.id}? This stops any processing immediately.`,
      confirmText: 'Cancel job',
      variant: 'danger',
      icon: 'warning',
    });
    if (!confirmed) return;

    try {
      await api.cancelJobs({ domain: job.domain, job_id: job.id, dry_run: false });
      success('Job cancelled', `Job ${job.id} has been cancelled.`);
      handleCloseDetail();
      void loadData();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to cancel job';
      showError('Cancel failed', message);
    }
  };

  const handleRetryJob = async (job: JobItem) => {
    const confirmed = await confirm({
      title: 'Retry job',
      message: `Retry job ${job.id}? The job will re-enter the queue immediately.`,
      confirmText: 'Retry job',
      variant: 'default',
    });
    if (!confirmed) return;

    try {
      await api.retryJobsNow({ domain: job.domain, job_id: job.id, only_failed: true, dry_run: false });
      success('Job retried', `Job ${job.id} has been re-queued.`);
      handleCloseDetail();
      void loadData();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to retry job';
      showError('Retry failed', message);
    }
  };

  const handleRequeueJob = async (job: JobItem) => {
    const confirmed = await confirm({
      title: 'Requeue job',
      message: `Requeue job ${job.id}? The job will move out of quarantine.`,
      confirmText: 'Requeue job',
      variant: 'default',
    });
    if (!confirmed) return;

    try {
      await api.requeueQuarantinedJobs({ domain: job.domain, job_id: job.id, dry_run: false });
      success('Job requeued', `Job ${job.id} moved back to queued.`);
      handleCloseDetail();
      void loadData();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to requeue job';
      showError('Requeue failed', message);
    }
  };

  const formatJson = (value: unknown) => {
    if (value === undefined || value === null) {
      return '—';
    }
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  };

  const hasSelectedJobRelationships =
    selectedJobRelationships.parentJobs.length > 0
    || selectedJobRelationships.childJobs.length > 0
    || selectedJobRelationships.unresolvedParentRefs.length > 0
    || selectedJobRelationships.unresolvedChildRefs.length > 0;

  const queueThroughputSummary = useMemo(
    () => buildQueueThroughputSummary(jobs),
    [jobs],
  );
  const queueTotals = useMemo(
    () => stats.reduce(
      (acc, row) => ({
        queued: acc.queued + row.queued,
        processing: acc.processing + row.processing,
        quarantined: acc.quarantined + row.quarantined,
      }),
      { queued: 0, processing: 0, quarantined: 0 },
    ),
    [stats],
  );

  const queueDepthChartPoints = useMemo(
    () => queueDepthHistory.slice(-25),
    [queueDepthHistory],
  );
  const maxQueueDepth = useMemo(
    () => Math.max(1, ...queueDepthChartPoints.map((point) => point.depth)),
    [queueDepthChartPoints],
  );
  const queueDepthPeak = useMemo(
    () => Math.max(0, ...queueDepthHistory.map((point) => point.depth)),
    [queueDepthHistory],
  );
  const queueDepthCurrent = queueDepthHistory.length > 0
    ? queueDepthHistory[queueDepthHistory.length - 1].depth
    : 0;
  const queueStatsLiveSummary = useMemo(() => {
    if (loading) {
      return 'Queue statistics loading.';
    }
    if (stats.length === 0) {
      return 'No queue statistics available.';
    }
    return `Queue statistics updated. ${queueTotals.queued} queued, ${queueTotals.processing} processing, ` +
      `${queueTotals.quarantined} quarantined. Current queue depth ${queueDepthCurrent.toFixed(0)}.`;
  }, [loading, queueDepthCurrent, queueTotals.processing, queueTotals.queued, queueTotals.quarantined, stats.length]);

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h1 className="text-3xl font-bold">Jobs</h1>
              <p className="text-muted-foreground">Inspect queues, job health, and recent activity</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <ExportMenu
                onExport={(format: ExportFormat) => exportJobs(jobs, format)}
                disabled={jobs.length === 0}
              />
              <Button variant="outline" onClick={loadData} disabled={loading}>
                <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </div>

          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {routesUnavailable && (
            <Alert className="mb-6 border-yellow-200 bg-yellow-50">
              <AlertTriangle className="h-4 w-4 text-yellow-600" />
              <AlertDescription className="text-yellow-800">
                Jobs admin endpoints are disabled or unavailable. Enable the <span className="font-medium">jobs</span>{' '}
                route in server settings and restart the API to load queue data.
              </AlertDescription>
            </Alert>
          )}

          {staleGroups.length > 0 && (
            <Alert className="mb-6 bg-yellow-50 border-yellow-200">
              <AlertTriangle className="h-4 w-4 text-yellow-600" />
              <AlertDescription className="text-yellow-800">
                {staleGroups.length} stale processing group{staleGroups.length !== 1 ? 's' : ''} detected. Review queues for
                hung jobs.
              </AlertDescription>
            </Alert>
          )}

          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Filter className="h-5 w-5" />
                Filters
              </CardTitle>
              <CardDescription>Limit jobs by domain, queue, status, or type.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                <div className="space-y-2">
                  <Label htmlFor="jobs-domain">Domain</Label>
                  <Input
                    id="jobs-domain"
                    value={filters.domain}
                    onChange={(event) => setFilters((prev) => ({ ...prev, domain: event.target.value }))}
                    placeholder="e.g. chatbooks"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="jobs-queue">Queue</Label>
                  <Input
                    id="jobs-queue"
                    value={filters.queue}
                    onChange={(event) => setFilters((prev) => ({ ...prev, queue: event.target.value }))}
                    placeholder="default"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="jobs-status">Status</Label>
                  <Select
                    id="jobs-status"
                    value={filters.status}
                    onChange={(event) => setFilters((prev) => ({ ...prev, status: event.target.value }))}
                  >
                    <option value="">All</option>
                    <option value="queued">Queued</option>
                    <option value="scheduled">Scheduled</option>
                    <option value="processing">Processing</option>
                    <option value="completed">Completed</option>
                    <option value="failed">Failed</option>
                    <option value="cancelled">Cancelled</option>
                    <option value="quarantined">Quarantined</option>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="jobs-type">Job type</Label>
                  <Input
                    id="jobs-type"
                    value={filters.job_type}
                    onChange={(event) => setFilters((prev) => ({ ...prev, job_type: event.target.value }))}
                    placeholder="export"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="jobs-limit">Limit</Label>
                  <Input
                    id="jobs-limit"
                    type="number"
                    min={1}
                    max={500}
                    value={filters.limit}
                    onChange={(event) => setFilters((prev) => ({ ...prev, limit: event.target.value }))}
                    onBlur={(event) => {
                      const next = clampLimit(event.target.value);
                      setFilters((prev) => ({ ...prev, limit: next }));
                    }}
                  />
                </div>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button onClick={loadData} disabled={loading}>
                  Apply filters
                </Button>
                <Button variant="outline" onClick={handleReset} disabled={loading}>
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-6 lg:grid-cols-3 mb-6">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Briefcase className="h-5 w-5" />
                  Queue Stats
                </CardTitle>
                <CardDescription>Current queue snapshot by domain/queue/type.</CardDescription>
              </CardHeader>
              <CardContent>
                <div
                  role="status"
                  aria-live="polite"
                  aria-atomic="true"
                  data-testid="jobs-queue-stats-live-region"
                >
                  <p className="sr-only">{queueStatsLiveSummary}</p>
                  {loading ? (
                    <div className="text-muted-foreground">Loading stats…</div>
                  ) : stats.length === 0 ? (
                    <div className="text-muted-foreground">No queue stats available.</div>
                  ) : (
                    <Table caption={`Queue stats table with ${stats.length} rows.`}>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Domain</TableHead>
                          <TableHead>Queue</TableHead>
                          <TableHead>Job type</TableHead>
                          <TableHead className="text-right">Queued</TableHead>
                          <TableHead className="text-right">Scheduled</TableHead>
                          <TableHead className="text-right">Processing</TableHead>
                          <TableHead className="text-right">Quarantined</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {stats.map((row) => (
                          <TableRow key={`${row.domain}-${row.queue}-${row.job_type}`}>
                            <TableCell>{row.domain}</TableCell>
                            <TableCell>{row.queue}</TableCell>
                            <TableCell>{row.job_type}</TableCell>
                            <TableCell className="text-right">{row.queued}</TableCell>
                            <TableCell className="text-right">{row.scheduled}</TableCell>
                            <TableCell className="text-right">{row.processing}</TableCell>
                            <TableCell className="text-right">{row.quarantined}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Stale processing</CardTitle>
                <CardDescription>Queues with expired leases.</CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-muted-foreground">Loading stale jobs…</div>
                ) : staleGroups.length === 0 ? (
                  <div className="text-muted-foreground">No stale jobs detected.</div>
                ) : (
                  <div className="space-y-3">
                    {staleGroups.map((group) => (
                      <div key={`${group.domain}-${group.queue}`} className="rounded-lg border p-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium">{group.domain}</p>
                            <p className="text-xs text-muted-foreground">Queue {group.queue}</p>
                          </div>
                          <Badge variant="destructive">{group.count} stale</Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2 mb-6">
            <Card>
              <CardHeader>
                <CardTitle>Queue Depth (24h)</CardTitle>
                <CardDescription>
                  Time-series for queued and processing workload over the last 24 hours.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="text-muted-foreground">Loading queue depth history…</div>
                ) : queueDepthChartPoints.length === 0 ? (
                  <div className="text-muted-foreground">Queue depth history unavailable.</div>
                ) : (
                  <div className="space-y-3">
                    <div
                      className="flex h-40 items-end gap-1 rounded-md border bg-muted/20 p-2"
                      data-testid="queue-depth-chart"
                    >
                      {queueDepthChartPoints.map((point, index) => {
                        const heightPercent = Math.max(4, (point.depth / maxQueueDepth) * 100);
                        return (
                          <div
                            key={`${point.timestamp}-${index}`}
                            className="flex-1 rounded-sm bg-primary/80"
                            style={{ height: `${heightPercent}%` }}
                            title={`${point.label}: ${point.depth.toFixed(0)}`}
                            aria-label={`${point.label}: queue depth ${point.depth.toFixed(0)}`}
                          />
                        );
                      })}
                    </div>
                    <div className="grid gap-3 sm:grid-cols-3 text-sm">
                      <div>
                        <p className="text-muted-foreground">Current</p>
                        <p className="font-semibold" data-testid="queue-depth-current">{queueDepthCurrent.toFixed(0)}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Peak</p>
                        <p className="font-semibold" data-testid="queue-depth-peak">{queueDepthPeak.toFixed(0)}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Data points</p>
                        <p className="font-semibold">{queueDepthChartPoints.length}</p>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Queue Throughput (24h)</CardTitle>
                <CardDescription>
                  Completed jobs per hour and average processing time for recent completed work.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-lg border p-3">
                    <p className="text-xs uppercase text-muted-foreground">Jobs/Hour</p>
                    <p className="text-xl font-semibold" data-testid="queue-throughput-jobs-per-hour">
                      {queueThroughputSummary.jobsCompletedPerHour.toFixed(2)}
                    </p>
                  </div>
                  <div className="rounded-lg border p-3">
                    <p className="text-xs uppercase text-muted-foreground">Avg Processing</p>
                    <p className="text-xl font-semibold" data-testid="queue-throughput-avg-processing">
                      {formatDurationDisplay(queueThroughputSummary.averageProcessingSeconds)}
                    </p>
                  </div>
                  <div className="rounded-lg border p-3">
                    <p className="text-xs uppercase text-muted-foreground">Completed (24h)</p>
                    <p className="text-xl font-semibold" data-testid="queue-throughput-completed">
                      {queueThroughputSummary.completedJobs24h}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* SLA Policies */}
          <Card className="mb-6">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="h-5 w-5" />
                    SLA Policies
                    {slaBreaches.length > 0 && (
                      <Badge variant="destructive" data-testid="sla-breach-count">
                        {slaBreaches.length} breach{slaBreaches.length !== 1 ? 'es' : ''}
                      </Badge>
                    )}
                  </CardTitle>
                  <CardDescription>Service level agreements for job processing times</CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowSlaForm(!showSlaForm)}
                >
                  {showSlaForm ? (
                    <>
                      <X className="mr-2 h-4 w-4" />
                      Cancel
                    </>
                  ) : (
                    <>
                      <Plus className="mr-2 h-4 w-4" />
                      New Policy
                    </>
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {/* Create SLA Policy Form */}
              {showSlaForm && (
                <div className="mb-4 p-4 border rounded-lg space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">New SLA Policy</span>
                    <AccessibleIconButton
                      icon={X}
                      label="Close form"
                      variant="ghost"
                      onClick={() => setShowSlaForm(false)}
                    />
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    <div className="space-y-1">
                      <Label htmlFor="sla-name">Policy Name</Label>
                      <Input
                        id="sla-name"
                        placeholder="e.g., Standard Export"
                        value={slaFormName}
                        onChange={(e) => setSlaFormName(e.target.value)}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="sla-domain">Domain</Label>
                      <Input
                        id="sla-domain"
                        placeholder="default"
                        value={slaFormDomain}
                        onChange={(e) => setSlaFormDomain(e.target.value)}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="sla-queue">Queue</Label>
                      <Input
                        id="sla-queue"
                        placeholder="default"
                        value={slaFormQueue}
                        onChange={(e) => setSlaFormQueue(e.target.value)}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="sla-job-type">Job Type</Label>
                      <Input
                        id="sla-job-type"
                        placeholder="e.g., export"
                        value={slaFormJobType}
                        onChange={(e) => setSlaFormJobType(e.target.value)}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="sla-max-processing">Max Processing (sec)</Label>
                      <Input
                        id="sla-max-processing"
                        type="number"
                        min="1"
                        value={slaFormMaxProcessing}
                        onChange={(e) => setSlaFormMaxProcessing(e.target.value)}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="sla-max-wait">Max Wait (sec)</Label>
                      <Input
                        id="sla-max-wait"
                        type="number"
                        min="1"
                        value={slaFormMaxWait}
                        onChange={(e) => setSlaFormMaxWait(e.target.value)}
                      />
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="sla-enabled"
                        checked={slaFormEnabled}
                        onCheckedChange={(checked) => setSlaFormEnabled(checked === true)}
                      />
                      <Label htmlFor="sla-enabled">Enabled</Label>
                    </div>
                    <Button onClick={handleCreateSlaPolicy} disabled={slaFormSaving} loading={slaFormSaving} loadingText="Creating...">
                      Create Policy
                    </Button>
                  </div>
                </div>
              )}

              {/* SLA Policies List */}
              {slaPoliciesError ? (
                <Alert variant="destructive" className="mb-4">
                  <AlertDescription>{slaPoliciesError}</AlertDescription>
                </Alert>
              ) : slaPoliciesLoading ? (
                <div className="text-muted-foreground">Loading SLA policies...</div>
              ) : slaPolicies.length === 0 ? (
                <div className="text-muted-foreground text-center py-4">
                  No SLA policies configured. Create one to define processing time expectations.
                </div>
              ) : (
                <Table caption={`SLA policies table with ${slaPolicies.length} rows.`}>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Domain</TableHead>
                      <TableHead>Queue</TableHead>
                      <TableHead>Job Type</TableHead>
                      <TableHead className="text-right">Max Processing</TableHead>
                      <TableHead className="text-right">Max Wait</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {slaPolicies.map((policy, idx) => (
                      <TableRow key={policy.id || `sla-${idx}`}>
                        <TableCell>{policy.domain || '—'}</TableCell>
                        <TableCell>{policy.queue || '—'}</TableCell>
                        <TableCell className="font-medium">{policy.job_type || policy.name || 'All'}</TableCell>
                        <TableCell className="text-right">
                          {formatDurationDisplay(policy.max_duration_seconds ?? policy.max_processing_time_seconds)}
                        </TableCell>
                        <TableCell className="text-right">
                          {formatDurationDisplay(policy.max_queue_latency_seconds ?? policy.max_wait_time_seconds)}
                        </TableCell>
                        <TableCell>
                          <Badge variant={policy.enabled ? 'default' : 'secondary'}>
                            {policy.enabled ? 'Enabled' : 'Disabled'}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <AccessibleIconButton
                            icon={Trash2}
                            label="Delete policy"
                            variant="outline"
                            onClick={() => handleDeleteSlaPolicy(policy)}
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Job Dependencies</CardTitle>
              <CardDescription>
                Parent/child relationships detected in the currently loaded jobs.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-muted-foreground">Loading dependency graph…</div>
              ) : dependencyRelationships.length === 0 ? (
                <div className="text-muted-foreground">
                  No related jobs found in the current result set.
                </div>
              ) : (
                <Table caption={`Job dependency table with ${dependencyRelationships.length} rows.`}>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Parent</TableHead>
                      <TableHead>Child</TableHead>
                      <TableHead>Relationship</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {dependencyRelationships.map((relation) => (
                      <TableRow
                        key={`${relation.parent.id}-${relation.child.id}`}
                        data-testid={`job-dependency-row-${relation.parent.id}-${relation.child.id}`}
                      >
                        <TableCell>
                          Job {relation.parent.id}
                          <span className="ml-2 text-xs text-muted-foreground">
                            {relation.parent.job_type}
                          </span>
                        </TableCell>
                        <TableCell>
                          Job {relation.child.id}
                          <span className="ml-2 text-xs text-muted-foreground">
                            {relation.child.job_type}
                          </span>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">
                            {relation.source === 'parent_ref' ? 'Child references parent' : 'Parent lists child'}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent jobs</CardTitle>
              <CardDescription>Latest jobs in the selected scope.</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-muted-foreground">Loading jobs…</div>
              ) : jobs.length === 0 ? (
                <EmptyState
                  icon={Briefcase}
                  title="No jobs found."
                  description="Try broadening filters or refresh to load recent jobs."
                  actions={[
                    {
                      label: 'Clear filters',
                      onClick: clearJobFilters,
                    },
                  ]}
                  className="py-8"
                />
              ) : (
                <>
                  <Table caption={`Jobs table with ${jobs.length} rows.`}>
                    <TableHeader>
                      <TableRow>
                        <TableHead>ID</TableHead>
                        <TableHead>Domain</TableHead>
                        <TableHead>Queue</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead className="text-right">Retries</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead>Started</TableHead>
                        <TableHead>Completed</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {jobs.map((job) => {
                        const normalizedStatus = job.status.toLowerCase();
                        const canCancel = normalizedStatus === 'queued' || normalizedStatus === 'processing';
                        const canRetry = normalizedStatus === 'failed';
                        const canRequeue = normalizedStatus === 'quarantined';
                        const isBreaching = breachedJobIds.has(job.id);
                        const breach = breachByJobId.get(job.id);

                        // SLA breach detection
                        const matchingSla =
                          slaPolicies.find((policy) => policy.enabled && policy.job_type === job.job_type)
                          ?? slaPolicies.find((policy) => policy.enabled && !policy.job_type);
                        const nowMs = Date.now();
                        const queuedAtMs = parseTimestampMs(job.available_at ?? job.created_at);
                        const startedAtMs = parseTimestampMs(job.started_at);
                        const completedAtMs = parseTimestampMs(job.completed_at);
                        const processingDurationSec = startedAtMs !== null
                          ? ((completedAtMs ?? nowMs) - startedAtMs) / 1000
                          : null;
                        const waitDurationSec = queuedAtMs !== null
                          ? (((startedAtMs ?? completedAtMs) ?? nowMs) - queuedAtMs) / 1000
                          : null;
                        const isSlaBreach = Boolean(
                          matchingSla && (
                            (
                              processingDurationSec !== null
                              && processingDurationSec > matchingSla.max_processing_time_seconds
                            )
                            || (
                              waitDurationSec !== null
                              && waitDurationSec > matchingSla.max_wait_time_seconds
                            )
                          )
                        );

                        return (
                          <TableRow key={job.id} className={isBreaching ? 'bg-red-50 dark:bg-red-950/30' : ''}>
                            <TableCell className="font-medium">
                              {job.id}
                              {isBreaching && (
                                <Badge
                                  variant="destructive"
                                  className="ml-2"
                                  title={breach ? breach.breach_kinds.join(', ') : 'SLA breach'}
                                  data-testid={`sla-breach-badge-${job.id}`}
                                >
                                  SLA
                                </Badge>
                              )}
                            </TableCell>
                            <TableCell>{job.domain}</TableCell>
                            <TableCell>{job.queue}</TableCell>
                            <TableCell>{job.job_type}</TableCell>
                            <TableCell>
                              <Badge className={statusBadge(job.status)}>{job.status}</Badge>
                            </TableCell>
                            <TableCell className="text-right">
                              {job.retry_count ?? 0}/{job.max_retries ?? 0}
                            </TableCell>
                            <TableCell>{formatJobDateTime(job.created_at)}</TableCell>
                            <TableCell>{formatJobDateTime(job.started_at)}</TableCell>
                            <TableCell>{formatJobDateTime(job.completed_at)}</TableCell>
                            <TableCell className="text-right">
                              <div className="flex justify-end gap-2">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => handleOpenDetail(job)}
                                >
                                  <Eye className="mr-2 h-4 w-4" />
                                  View
                                </Button>
                                <AccessibleIconButton
                                  icon={RotateCcw}
                                  label={canRetry ? 'Retry job' : 'Retry available for failed jobs'}
                                  variant="outline"
                                  onClick={() => handleRetryJob(job)}
                                  disabled={!canRetry}
                                />
                                <AccessibleIconButton
                                  icon={Repeat}
                                  label={canRequeue ? 'Requeue job' : 'Requeue available for quarantined jobs'}
                                  variant="outline"
                                  onClick={() => handleRequeueJob(job)}
                                  disabled={!canRequeue}
                                />
                                <AccessibleIconButton
                                  icon={XCircle}
                                  label={canCancel ? 'Cancel job' : 'Cancel available for queued or processing jobs'}
                                  variant="outline"
                                  onClick={() => handleCancelJob(job)}
                                  disabled={!canCancel}
                                />
                              </div>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </>
              )}
            </CardContent>
          </Card>

          <Dialog open={!!selectedJob} onOpenChange={(open) => !open && handleCloseDetail()}>
            <DialogContent className="max-w-4xl">
              <DialogHeader>
                <DialogTitle>Job {selectedJob?.id}</DialogTitle>
                <DialogDescription>
                  {jobDetail?.archived ? 'Archived job details' : 'Job details and payload'}
                </DialogDescription>
              </DialogHeader>
              {jobDetailLoading && (
                <div className="text-sm text-muted-foreground">Loading job details…</div>
              )}
              {jobDetailError && (
                <Alert variant="destructive">
                  <AlertDescription>{jobDetailError}</AlertDescription>
                </Alert>
              )}
              {jobDetail && (
                <div className="space-y-4">
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-1">
                      <Label>ID</Label>
                      <div className="text-sm font-mono">{jobDetail.id}</div>
                    </div>
                    <div className="space-y-1">
                      <Label>Status</Label>
                      <Badge className={statusBadge(jobDetail.status)}>{jobDetail.status}</Badge>
                    </div>
                    <div className="space-y-1">
                      <Label>Domain</Label>
                      <div className="text-sm">{jobDetail.domain}</div>
                    </div>
                    <div className="space-y-1">
                      <Label>Queue</Label>
                      <div className="text-sm">{jobDetail.queue}</div>
                    </div>
                    <div className="space-y-1">
                      <Label>Job type</Label>
                      <div className="text-sm">{jobDetail.job_type}</div>
                    </div>
                    <div className="space-y-1">
                      <Label>Retries</Label>
                      <div className="text-sm">
                        {jobDetail.retry_count ?? 0}/{jobDetail.max_retries ?? 0}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <Label>Created</Label>
                      <div className="text-sm">{formatJobDateTime(jobDetail.created_at)}</div>
                    </div>
                    <div className="space-y-1">
                      <Label>Started</Label>
                      <div className="text-sm">{formatJobDateTime(jobDetail.started_at)}</div>
                    </div>
                    <div className="space-y-1">
                      <Label>Completed</Label>
                      <div className="text-sm">{formatJobDateTime(jobDetail.completed_at)}</div>
                    </div>
                    <div className="space-y-1">
                      <Label>Archived</Label>
                      <div className="text-sm">{jobDetail.archived ? 'Yes' : 'No'}</div>
                    </div>
                  </div>

                  {hasSelectedJobRelationships && (
                    <div className="space-y-2" data-testid="job-related-jobs">
                      <Label>Related Jobs</Label>
                      <div className="rounded-md border p-3 text-sm">
                        {selectedJobRelationships.parentJobs.length > 0 && (
                          <div className="mb-3">
                            <p className="mb-1 font-medium">Parent job(s)</p>
                            <div className="flex flex-wrap gap-2">
                              {selectedJobRelationships.parentJobs.map((parentJob) => (
                                <Button
                                  key={`parent-${parentJob.id}`}
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    void handleOpenDetail(parentJob);
                                  }}
                                >
                                  Job {parentJob.id}
                                </Button>
                              ))}
                            </div>
                          </div>
                        )}

                        {selectedJobRelationships.unresolvedParentRefs.length > 0 && (
                          <div className="mb-3">
                            <p className="mb-1 font-medium">Parent reference(s)</p>
                            <div className="flex flex-wrap gap-2">
                              {selectedJobRelationships.unresolvedParentRefs.map((ref) => (
                                <Badge key={`missing-parent-${ref}`} variant="outline">
                                  {ref}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}

                        {selectedJobRelationships.childJobs.length > 0 && (
                          <div className="mb-3">
                            <p className="mb-1 font-medium">Child job(s)</p>
                            <div className="flex flex-wrap gap-2">
                              {selectedJobRelationships.childJobs.map((childJob) => (
                                <Button
                                  key={`child-${childJob.id}`}
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    void handleOpenDetail(childJob);
                                  }}
                                >
                                  Job {childJob.id}
                                </Button>
                              ))}
                            </div>
                          </div>
                        )}

                        {selectedJobRelationships.unresolvedChildRefs.length > 0 && (
                          <div>
                            <p className="mb-1 font-medium">Child reference(s)</p>
                            <div className="flex flex-wrap gap-2">
                              {selectedJobRelationships.unresolvedChildRefs.map((ref) => (
                                <Badge key={`missing-child-${ref}`} variant="outline">
                                  {ref}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {jobDetail.error_message && (
                    <Alert variant="destructive">
                      <AlertDescription>{jobDetail.error_message}</AlertDescription>
                    </Alert>
                  )}

                  <div className="space-y-2">
                    <Label>Payload</Label>
                    <pre className="max-h-64 overflow-auto rounded-md bg-muted p-3 text-xs">
                      {formatJson(jobDetail.payload)}
                    </pre>
                  </div>
                  <div className="space-y-2">
                    <Label>Result</Label>
                    <pre className="max-h-64 overflow-auto rounded-md bg-muted p-3 text-xs">
                      {formatJson(jobDetail.result)}
                    </pre>
                  </div>

                  {/* Job Attachments */}
                  {attachmentsLoading ? (
                    <div className="text-sm text-muted-foreground">Loading attachments...</div>
                  ) : jobAttachments.length > 0 && (
                    <div className="space-y-2">
                      <Label className="flex items-center gap-2">
                        <Paperclip className="h-4 w-4" />
                        Attachments ({jobAttachments.length})
                      </Label>
                      <div className="rounded-md border">
                        <Table caption={`Job attachments table with ${jobAttachments.length} rows.`}>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Name</TableHead>
                              <TableHead>Type</TableHead>
                              <TableHead className="text-right">Size</TableHead>
                              <TableHead>Created</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {jobAttachments.map((attachment) => (
                              <TableRow key={attachment.id}>
                                <TableCell className="font-medium">{attachment.name}</TableCell>
                                <TableCell className="text-muted-foreground">{attachment.content_type}</TableCell>
                                <TableCell className="text-right">
                                  {formatBytes(attachment.size_bytes, {
                                    fallback: '—',
                                    precision: attachment.size_bytes >= 1024 ? 1 : 0,
                                  })}
                                </TableCell>
                                <TableCell className="text-muted-foreground">
                                  {formatJobDateTime(attachment.created_at)}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </DialogContent>
          </Dialog>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
