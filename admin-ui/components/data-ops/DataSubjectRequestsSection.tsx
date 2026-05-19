'use client';

import { useEffect, useMemo, useState } from 'react';
import { ShieldAlert } from 'lucide-react';

import { Field } from '@/components/data-ops/Field';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useToast } from '@/components/ui/toast';
import { api } from '@/lib/api-client';
import { formatDateTime } from '@/lib/format';

type DataSubjectRequestsSectionProps = {
  refreshSignal: number;
};

type DataSubjectRequestType = 'export' | 'erasure' | 'access';
type DataSubjectRequestStatus = 'recorded' | 'completed' | 'failed' | 'rejected' | 'in_review';

type DataCategoryCount = {
  key: string;
  label: string;
  count: number;
};

type DataSubjectRequestLogItem = {
  id: string;
  requester_identifier: string;
  request_type: DataSubjectRequestType;
  status: DataSubjectRequestStatus;
  requested_at: string;
  selected_categories: string[];
  preview_summary: DataCategoryCount[];
};

const CATEGORY_DEFS: Array<{ key: string; label: string }> = [
  { key: 'media_records', label: 'Media records' },
  { key: 'chat_messages', label: 'Chat sessions/messages' },
  { key: 'notes', label: 'Notes' },
  { key: 'audit_events', label: 'Audit log events' },
];

const REQUEST_TYPE_OPTIONS: Array<{ value: DataSubjectRequestType; label: string }> = [
  { value: 'export', label: 'Export' },
  { value: 'erasure', label: 'Erasure' },
  { value: 'access', label: 'Access' },
];

const parseNonNegativeInteger = (value: unknown): number | null => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return null;
  return Math.floor(value);
};

const parseSummaryArray = (value: unknown): DataCategoryCount[] | null => {
  if (!Array.isArray(value)) return null;
  const parsed = value
    .map((entry) => {
      if (!entry || typeof entry !== 'object') return null;
      const record = entry as Record<string, unknown>;
      const key = typeof record.key === 'string' ? record.key : null;
      const label = typeof record.label === 'string' ? record.label : null;
      const count = parseNonNegativeInteger(record.count);
      if (!key || !label || count === null) return null;
      return { key, label, count };
    })
    .filter((entry): entry is DataCategoryCount => entry !== null);
  return parsed.length > 0 ? parsed : null;
};

const parseCategorySummary = (value: unknown): DataCategoryCount[] | null => {
  const fromArray = parseSummaryArray(value);
  if (fromArray) return fromArray;
  if (!value || typeof value !== 'object') return null;

  const root = value as Record<string, unknown>;
  const nestedArray = parseSummaryArray(root.summary);
  if (nestedArray) return nestedArray;

  const sourceSummary = (() => {
    if (root.counts && typeof root.counts === 'object') return root.counts as Record<string, unknown>;
    if (root.summary && typeof root.summary === 'object' && !Array.isArray(root.summary)) {
      return root.summary as Record<string, unknown>;
    }
    if (root.categories && typeof root.categories === 'object') return root.categories as Record<string, unknown>;
    return root;
  })();

  const mapped = CATEGORY_DEFS.map((def) => {
    const aliases: Record<string, string[]> = {
      media_records: ['media_records', 'media', 'media_items', 'content_items'],
      chat_messages: ['chat_messages', 'chats', 'chat_sessions'],
      notes: ['notes', 'note_items'],
      audit_events: ['audit_events', 'audit_logs', 'audit_log_entries'],
    };
    const keys = aliases[def.key] ?? [def.key];
    for (const key of keys) {
      const parsed = parseNonNegativeInteger(sourceSummary[key]);
      if (parsed !== null) {
        return { key: def.key, label: def.label, count: parsed };
      }
    }
    return { key: def.key, label: def.label, count: 0 };
  });

  return mapped;
};

const parseRequestLogItem = (value: unknown): DataSubjectRequestLogItem | null => {
  if (!value || typeof value !== 'object') return null;
  const record = value as Record<string, unknown>;

  const rawId = record.client_request_id ?? record.id;
  const id = typeof rawId === 'string' || typeof rawId === 'number' ? String(rawId) : '';
  const requesterIdentifier = typeof record.requester_identifier === 'string'
    ? record.requester_identifier
    : typeof record.requester === 'string'
      ? record.requester
      : '';
  const requestType = typeof record.request_type === 'string'
    ? record.request_type
    : '';
  const status = typeof record.status === 'string'
    ? record.status
    : '';
  const requestedAt = typeof record.requested_at === 'string'
    ? record.requested_at
    : '';

  if (!id || !requesterIdentifier || !requestedAt) return null;
  if (!['export', 'erasure', 'access'].includes(requestType)) return null;
  if (!['recorded', 'completed', 'failed', 'rejected', 'in_review'].includes(status)) return null;

  const previewSummary = parseCategorySummary(record.preview_summary) ?? [];
  const selectedCategories = Array.isArray(record.selected_categories)
    ? record.selected_categories.filter((entry): entry is string => typeof entry === 'string')
    : [];

  return {
    id,
    requester_identifier: requesterIdentifier,
    request_type: requestType as DataSubjectRequestType,
    status: status as DataSubjectRequestStatus,
    requested_at: requestedAt,
    selected_categories: selectedCategories,
    preview_summary: previewSummary,
  };
};

const parseRequestLogResponse = (value: unknown): DataSubjectRequestLogItem[] => {
  if (!value || typeof value !== 'object') return [];
  const root = value as Record<string, unknown>;
  const items = Array.isArray(root.items) ? root.items : [];
  return items
    .map(parseRequestLogItem)
    .filter((entry): entry is DataSubjectRequestLogItem => entry !== null);
};

const badgeVariantForStatus = (status: DataSubjectRequestStatus) => {
  if (status === 'failed' || status === 'rejected') return 'destructive';
  if (status === 'recorded' || status === 'in_review') return 'secondary';
  return 'default';
};

export const DataSubjectRequestsSection = ({ refreshSignal }: DataSubjectRequestsSectionProps) => {
  const { success, error: showError } = useToast();

  const [requesterIdentifier, setRequesterIdentifier] = useState('');
  const [requestType, setRequestType] = useState<DataSubjectRequestType>('access');
  const [submitting, setSubmitting] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [requestLogLoading, setRequestLogLoading] = useState(true);
  const [formError, setFormError] = useState('');
  const [requestLogError, setRequestLogError] = useState('');

  const [accessSummary, setAccessSummary] = useState<DataCategoryCount[] | null>(null);
  const [erasurePreview, setErasurePreview] = useState<DataCategoryCount[] | null>(null);
  const [selectedErasureCategories, setSelectedErasureCategories] = useState<Record<string, boolean>>({});
  const [erasureConfirmed, setErasureConfirmed] = useState(false);
  const [requestLog, setRequestLog] = useState<DataSubjectRequestLogItem[]>([]);

  const selectedErasureKeys = useMemo(() => {
    return Object.entries(selectedErasureCategories)
      .filter(([, selected]) => selected)
      .map(([key]) => key);
  }, [selectedErasureCategories]);

  const loadRequestLog = async () => {
    setRequestLogLoading(true);
    setRequestLogError('');
    try {
      const response = await api.listDataSubjectRequests({ limit: '50', offset: '0' });
      setRequestLog(parseRequestLogResponse(response));
    } catch (error: unknown) {
      const message = error instanceof Error && error.message
        ? error.message
        : 'Failed to load data subject requests';
      setRequestLog([]);
      setRequestLogError(message);
      showError('Request log unavailable', message);
    } finally {
      setRequestLogLoading(false);
    }
  };

  useEffect(() => {
    let cancelled = false;

    const loadInitialRequestLog = async () => {
      setRequestLogLoading(true);
      setRequestLogError('');
      try {
        const response = await api.listDataSubjectRequests({ limit: '50', offset: '0' });
        if (cancelled) return;
        setRequestLog(parseRequestLogResponse(response));
      } catch (error: unknown) {
        if (cancelled) return;
        const message = error instanceof Error && error.message
          ? error.message
          : 'Failed to load data subject requests';
        setRequestLog([]);
        setRequestLogError(message);
        showError('Request log unavailable', message);
      } finally {
        if (!cancelled) {
          setRequestLogLoading(false);
        }
      }
    };

    setRequesterIdentifier('');
    setRequestType('access');
    setFormError('');
    setAccessSummary(null);
    setErasurePreview(null);
    setSelectedErasureCategories({});
    setErasureConfirmed(false);
    void loadInitialRequestLog();

    return () => {
      cancelled = true;
    };
  }, [refreshSignal, showError]);

  const resolveCategorySummary = async (
    requester: string,
    request: { requestType?: DataSubjectRequestType; categories?: string[] } = {},
  ) => {
    const response = await api.previewDataSubjectRequest({
      requester_identifier: requester,
      request_type: request.requestType,
      categories: request.categories,
    });
    const parsed = parseCategorySummary(response);
    if (!parsed) {
      throw new Error('Preview response did not include a valid category summary');
    }
    return parsed;
  };

  const handlePreviewErasure = async () => {
    const normalizedRequester = requesterIdentifier.trim();
    if (!normalizedRequester) {
      setFormError('User identifier (email or user ID) is required.');
      return;
    }

    setPreviewLoading(true);
    setFormError('');
    try {
      const summary = await resolveCategorySummary(normalizedRequester, {
        requestType: 'erasure',
        categories: CATEGORY_DEFS.map((entry) => entry.key),
      });
      setErasurePreview(summary);
      setSelectedErasureCategories({});
      setErasureConfirmed(false);
    } catch (error: unknown) {
      const message = error instanceof Error && error.message ? error.message : 'Failed to preview user data';
      showError('Preview failed', message);
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleSubmitRequest = async () => {
    const normalizedRequester = requesterIdentifier.trim();
    if (!normalizedRequester) {
      setFormError('User identifier (email or user ID) is required.');
      return;
    }

    if (requestType === 'erasure' && !erasurePreview) {
      setFormError('Preview user data before submitting an erasure request.');
      return;
    }
    if (requestType === 'erasure' && selectedErasureKeys.length === 0) {
      setFormError('Select at least one data category to erase.');
      return;
    }
    if (requestType === 'erasure' && !erasureConfirmed) {
      setFormError('Confirm that this action cannot be undone before submitting erasure.');
      return;
    }

    setSubmitting(true);
    setFormError('');

    const requestId = `dsr-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;

    try {
      const response = await api.createDataSubjectRequest({
        client_request_id: requestId,
        requester_identifier: normalizedRequester,
        request_type: requestType,
        categories: requestType === 'erasure' ? selectedErasureKeys : undefined,
      });
      const createdItem = parseRequestLogItem(
        response && typeof response === 'object'
          ? (response as Record<string, unknown>).item
          : null,
      );
      const createdSummary = createdItem?.preview_summary ?? [];

      if (requestType === 'access' && createdSummary.length > 0) {
        setAccessSummary(createdSummary);
        success('Request recorded', 'The access request was recorded and the authoritative summary is shown below.');
      } else {
        success(
          'Request recorded',
          'The request was recorded for review. Export and erasure are not executed automatically in this release.',
        );
      }

      await loadRequestLog();
    } catch (error: unknown) {
      const message = error instanceof Error && error.message ? error.message : 'Failed to process data subject request';
      showError('Request failed', message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ShieldAlert className="h-5 w-5" />
          Data Subject Requests
        </CardTitle>
        <CardDescription>Record GDPR-style access, export, and erasure requests with authoritative backend data.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Alert className="border-amber-300 bg-amber-50">
          <AlertDescription className="flex items-center gap-2 text-amber-800">
            <Badge variant="outline" className="border-amber-500 text-amber-700 font-semibold shrink-0" data-testid="dsr-record-only-badge">
              Record-only mode
            </Badge>
            <span>
              Erasure and export are not executed automatically. Requests are recorded server-side for manual review.
            </span>
          </AlertDescription>
        </Alert>

        <div className="grid gap-3 md:grid-cols-3">
          <Field id="dsr-requester" label="User identifier (email or user ID)">
            <Input
              id="dsr-requester"
              value={requesterIdentifier}
              onChange={(event) => setRequesterIdentifier(event.target.value)}
              placeholder="user@example.com or 42"
            />
          </Field>
          <Field id="dsr-request-type" label="Request type">
            <Select
              id="dsr-request-type"
              value={requestType}
              onChange={(event) => {
                const nextValue = event.target.value as DataSubjectRequestType;
                setRequestType(nextValue);
                setFormError('');
                setAccessSummary(null);
                if (nextValue !== 'erasure') {
                  setErasurePreview(null);
                  setSelectedErasureCategories({});
                  setErasureConfirmed(false);
                }
              }}
            >
              {REQUEST_TYPE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </Select>
          </Field>
          <div className="flex items-end gap-2">
            {requestType === 'erasure' && (
              <Button
                variant="outline"
                onClick={() => { void handlePreviewErasure(); }}
                disabled={previewLoading}
                loading={previewLoading}
                loadingText="Previewing..."
              >
                Preview user data
              </Button>
            )}
            <Button
              onClick={() => { void handleSubmitRequest(); }}
              disabled={submitting}
              loading={submitting}
              loadingText="Submitting..."
            >
              Submit request
            </Button>
          </div>
        </div>

        {formError && (
          <Alert variant="destructive">
            <AlertDescription>{formError}</AlertDescription>
          </Alert>
        )}

        {requestLogError && (
          <Alert variant="destructive">
            <AlertDescription>{requestLogError}</AlertDescription>
          </Alert>
        )}

        {requestType === 'erasure' && erasurePreview && (
          <div className="space-y-3 rounded-lg border p-4" data-testid="dsr-erasure-preview">
            <p className="text-sm font-medium">Select categories to erase</p>
            <div className="grid gap-2 md:grid-cols-2">
              {erasurePreview.map((entry) => (
                <label key={entry.key} className="flex items-center justify-between gap-2 rounded border p-2 text-sm">
                  <span className="flex items-center gap-2">
                    <Checkbox
                      checked={Boolean(selectedErasureCategories[entry.key])}
                      onCheckedChange={(checked) =>
                        setSelectedErasureCategories((prev) => ({
                          ...prev,
                          [entry.key]: checked,
                        }))
                      }
                    />
                    <span>{entry.label}</span>
                  </span>
                  <Badge variant="secondary">{entry.count}</Badge>
                </label>
              ))}
            </div>
            <Alert variant="destructive">
              <AlertDescription>This action cannot be undone.</AlertDescription>
            </Alert>
            <label htmlFor="dsr-erasure-confirm" className="flex items-center gap-2 text-sm">
              <Checkbox
                id="dsr-erasure-confirm"
                checked={erasureConfirmed}
                onCheckedChange={setErasureConfirmed}
              />
              <span>I understand this action cannot be undone.</span>
            </label>
          </div>
        )}

        {requestType === 'access' && accessSummary && (
          <div className="space-y-2 rounded-lg border p-4" data-testid="dsr-access-summary">
            <p className="text-sm font-medium">Access summary</p>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Category</TableHead>
                  <TableHead className="text-right">Records</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {accessSummary.map((entry) => (
                  <TableRow key={entry.key}>
                    <TableCell>{entry.label}</TableCell>
                    <TableCell className="text-right">{entry.count}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}

        <div className="space-y-2" data-testid="dsr-request-log">
          <p className="text-sm font-medium">Request log</p>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Requested</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Requester</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {requestLogLoading ? (
                <TableRow>
                  <TableCell colSpan={4} className="text-muted-foreground">Loading requests…</TableCell>
                </TableRow>
              ) : requestLog.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={4} className="text-muted-foreground">No recorded requests yet.</TableCell>
                </TableRow>
              ) : (
                requestLog.map((entry) => (
                  <TableRow key={entry.id} data-testid="dsr-request-log-row">
                    <TableCell>{formatDateTime(entry.requested_at, { fallback: '—' })}</TableCell>
                    <TableCell className="capitalize">{entry.request_type}</TableCell>
                    <TableCell>{entry.requester_identifier}</TableCell>
                    <TableCell>
                      <Badge variant={badgeVariantForStatus(entry.status)}>
                        {entry.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
};
