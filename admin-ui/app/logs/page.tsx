'use client';

import { useCallback, useEffect, useMemo, useRef, useState, Suspense } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { EmptyState } from '@/components/ui/empty-state';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Pagination } from '@/components/ui/pagination';
import { api } from '@/lib/api-client';
import { useSearchParams } from 'next/navigation';
import { useUrlPagination } from '@/lib/use-url-state';
import { formatDateTime } from '@/lib/format';
import { ExportMenu } from '@/components/ui/export-menu';
import { exportData, type ExportFormat } from '@/lib/export';
import { RefreshCw } from 'lucide-react';

type SystemLogEntry = {
  timestamp?: string | null;
  level?: string | null;
  message?: string | null;
  logger?: string | null;
  module?: string | null;
  function?: string | null;
  line?: number | null;
  request_id?: string | null;
  org_id?: number | null;
  user_id?: number | null;
};

type SystemLogsResponse = {
  items?: SystemLogEntry[];
  total?: number;
};

type LogFilters = {
  start: string;
  end: string;
  level: string;
  service: string;
  query: string;
  regexMode: boolean;
  requestId: string;
  orgId: string;
  userId: string;
};

const LOG_LEVELS = ['', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

const getLevelBadgeProps = (level?: string | null) => {
  const normalized = (level || '').toUpperCase();
  if (normalized === 'ERROR' || normalized === 'CRITICAL') {
    return { variant: 'destructive' as const };
  }
  if (normalized === 'WARNING') {
    return { variant: 'outline' as const, className: 'border-yellow-300 bg-yellow-50 text-yellow-900' };
  }
  if (normalized === 'INFO') {
    return { variant: 'secondary' as const };
  }
  return { variant: 'outline' as const };
};

const formatLogDateTime = (value?: string | null) =>
  formatDateTime(value, { fallback: '—' });

const toIsoIfSet = (value: string) => {
  if (!value) return '';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toISOString();
};

const parsePositiveInt = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  if (!Number.isInteger(parsed) || parsed <= 0) return null;
  return parsed;
};

const toRegexValidation = (enabled: boolean, pattern: string): {
  enabled: boolean;
  valid: boolean;
  message: string;
} => {
  if (!enabled) {
    return { enabled: false, valid: true, message: '' };
  }
  if (!pattern.trim()) {
    return { enabled: true, valid: true, message: 'Regex enabled. Enter a pattern to match log entries.' };
  }
  try {
    // Validate pattern early so we can provide immediate feedback and avoid failing fetch cycles.
    new RegExp(pattern);
    return { enabled: true, valid: true, message: 'Valid regex pattern.' };
  } catch (err: unknown) {
    const reason = err instanceof Error && err.message ? err.message : 'Invalid regular expression.';
    return { enabled: true, valid: false, message: reason };
  }
};

const buildSearchableLogText = (entry: SystemLogEntry): string =>
  [
    entry.message,
    entry.logger,
    entry.module,
    entry.function,
    entry.request_id,
  ]
    .filter((value): value is string => typeof value === 'string' && value.length > 0)
    .join(' ');

const areFiltersEqual = (left: LogFilters, right: LogFilters) => (
  left.start === right.start
  && left.end === right.end
  && left.level === right.level
  && left.service === right.service
  && left.query === right.query
  && left.regexMode === right.regexMode
  && left.requestId === right.requestId
  && left.orgId === right.orgId
  && left.userId === right.userId
);

function LogsPageContent() {
  const { page, pageSize, setPage, setPageSize, resetPagination } = useUrlPagination();
  const searchParams = useSearchParams();
  const [logs, setLogs] = useState<SystemLogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshSignal, setRefreshSignal] = useState(0);
  const timezoneLabel =
    typeof Intl !== 'undefined'
      ? Intl.DateTimeFormat().resolvedOptions().timeZone || 'local time'
      : 'local time';

  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [level, setLevel] = useState('');
  const [service, setService] = useState('');
  const [query, setQuery] = useState('');
  const [regexMode, setRegexMode] = useState(false);
  const [requestId, setRequestId] = useState(() => searchParams.get('request_id') ?? '');
  const [orgId, setOrgId] = useState('');
  const [userId, setUserId] = useState('');
  const requestIdParam = searchParams.get('request_id') ?? '';
  const clearLogFilters = useCallback(() => {
    setStart('');
    setEnd('');
    setLevel('');
    setService('');
    setQuery('');
    setRegexMode(false);
    setRequestId('');
    setOrgId('');
    setUserId('');
    resetPagination();
  }, [resetPagination]);

  const filters = useMemo<LogFilters>(() => ({
    start,
    end,
    level,
    service,
    query,
    regexMode,
    requestId,
    orgId,
    userId,
  }), [end, level, orgId, query, regexMode, requestId, service, start, userId]);
  const [debouncedFilters, setDebouncedFilters] = useState<LogFilters>(filters);
  const debouncedFiltersRef = useRef<LogFilters>(filters);

  const regexValidation = useMemo(
    () => toRegexValidation(debouncedFilters.regexMode, debouncedFilters.query),
    [debouncedFilters.query, debouncedFilters.regexMode]
  );

  useEffect(() => {
    const handle = window.setTimeout(() => {
      if (areFiltersEqual(debouncedFiltersRef.current, filters)) {
        return;
      }
      debouncedFiltersRef.current = filters;
      setDebouncedFilters(filters);
      resetPagination();
    }, 300);
    return () => window.clearTimeout(handle);
  }, [filters, resetPagination]);

  useEffect(() => {
    setRequestId(requestIdParam);
  }, [requestIdParam]);

  const validationError = useMemo(() => {
    const issues: string[] = [];
    if (debouncedFilters.orgId.trim() && parsePositiveInt(debouncedFilters.orgId) === null) {
      issues.push('Org ID must be a positive integer.');
    }
    if (debouncedFilters.userId.trim() && parsePositiveInt(debouncedFilters.userId) === null) {
      issues.push('User ID must be a positive integer.');
    }
    if (!regexValidation.valid) {
      issues.push(`Regex pattern is invalid: ${regexValidation.message}`);
    }
    return issues.join(' ');
  }, [debouncedFilters.orgId, debouncedFilters.userId, regexValidation.message, regexValidation.valid]);

  const params = useMemo(() => {
    const offset = Math.max(0, (page - 1) * pageSize);
    const payload: Record<string, string> = {
      limit: String(pageSize),
      offset: String(offset),
    };
    const isoStart = toIsoIfSet(debouncedFilters.start);
    const isoEnd = toIsoIfSet(debouncedFilters.end);
    if (isoStart) payload.start = isoStart;
    if (isoEnd) payload.end = isoEnd;
    if (debouncedFilters.level) payload.level = debouncedFilters.level;
    if (debouncedFilters.service) payload.service = debouncedFilters.service;
    if (debouncedFilters.query) payload.query = debouncedFilters.query;
    if (debouncedFilters.regexMode) payload.query_mode = 'regex';
    if (debouncedFilters.requestId) payload.request_id = debouncedFilters.requestId;
    if (debouncedFilters.orgId) payload.org_id = debouncedFilters.orgId;
    if (debouncedFilters.userId) payload.user_id = debouncedFilters.userId;
    return payload;
  }, [debouncedFilters, page, pageSize]);

  const loadLogs = useCallback(async (signal?: AbortSignal) => {
    try {
      setLoading(true);
      setError('');
      if (validationError) {
        setError(validationError);
        return;
      }
      const data = (await api.getSystemLogs(params, { signal })) as SystemLogsResponse;
      const items = Array.isArray(data?.items) ? data.items : [];
      const hasRequestIdFilter = debouncedFilters.requestId.trim().length > 0;
      const hasRegexFilter = debouncedFilters.regexMode && debouncedFilters.query.trim().length > 0;

      let filteredItems = items;
      if (hasRequestIdFilter) {
        const normalizedRequestId = debouncedFilters.requestId.trim();
        filteredItems = filteredItems.filter((entry) => entry.request_id === normalizedRequestId);
      }

      if (hasRegexFilter) {
        const regex = new RegExp(debouncedFilters.query);
        filteredItems = filteredItems.filter((entry) => regex.test(buildSearchableLogText(entry)));
      }

      setLogs(filteredItems);
      const isClientFiltered = hasRequestIdFilter || hasRegexFilter;
      setTotal(
        isClientFiltered
          ? filteredItems.length
          : (typeof data?.total === 'number' ? data.total : filteredItems.length)
      );
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const message = err instanceof Error && err.message ? err.message : 'Failed to load logs';
      setError(message);
      setLogs([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [debouncedFilters.query, debouncedFilters.regexMode, debouncedFilters.requestId, params, validationError]);

  useEffect(() => {
    const controller = new AbortController();
    void loadLogs(controller.signal);
    return () => controller.abort();
  }, [loadLogs, refreshSignal]);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const applyRequestCorrelationFilter = (nextRequestId: string) => {
    if (!nextRequestId.trim()) return;
    setRequestId(nextRequestId.trim());
    resetPagination();
  };

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="flex flex-col gap-6 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold">System Logs</h1>
              <p className="text-muted-foreground">Query recent logs aggregated across workers.</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <ExportMenu
                onExport={(format: ExportFormat) => {
                  exportData({
                    data: logs as Record<string, unknown>[],
                    filename: 'system-logs',
                    format,
                  });
                }}
                disabled={logs.length === 0}
              />
              <Button
                variant="outline"
                onClick={() => {
                  setRefreshSignal((prev) => prev + 1);
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
              <CardTitle>Filters</CardTitle>
              <CardDescription>
                Restrict logs by time range, level, or metadata. Times use {timezoneLabel} and
                filters are sent to the server as UTC.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-3">
              <div className="space-y-1">
                <Label htmlFor="start">Start</Label>
                <Input
                  id="start"
                  type="datetime-local"
                  value={start}
                  onChange={(e) => {
                    setStart(e.target.value);
                  }}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="end">End</Label>
                <Input
                  id="end"
                  type="datetime-local"
                  value={end}
                  onChange={(e) => {
                    setEnd(e.target.value);
                  }}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="level">Level</Label>
                <Select
                  id="level"
                  value={level}
                  onChange={(e) => {
                    setLevel(e.target.value);
                  }}
                >
                  {LOG_LEVELS.map((item) => (
                    <option key={item || 'all'} value={item}>
                      {item || 'All'}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="space-y-1">
                <Label htmlFor="service">Service</Label>
                <Input
                  id="service"
                  placeholder="logger or module"
                  value={service}
                  onChange={(e) => {
                    setService(e.target.value);
                  }}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="query">Search</Label>
                <Input
                  id="query"
                  placeholder="message contains..."
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value);
                  }}
                />
                <div className="flex items-center gap-2 pt-1">
                  <Checkbox
                    id="queryRegexMode"
                    checked={regexMode}
                    onCheckedChange={(checked) => {
                      setRegexMode(checked === true);
                    }}
                  />
                  <Label htmlFor="queryRegexMode" className="text-sm font-normal">
                    Treat search as regex
                  </Label>
                </div>
                {regexValidation.enabled && (
                  <p
                    className={`text-xs ${
                      regexValidation.valid ? 'text-muted-foreground' : 'text-destructive'
                    }`}
                    role={regexValidation.valid ? 'status' : 'alert'}
                  >
                    {regexValidation.message}
                  </p>
                )}
              </div>
              <div className="space-y-1">
                <Label htmlFor="requestId">Request ID</Label>
                <Input
                  id="requestId"
                  placeholder="Correlate by request id"
                  value={requestId}
                  onChange={(e) => {
                    setRequestId(e.target.value);
                  }}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="orgId">Org ID</Label>
                <Input
                  id="orgId"
                  placeholder="optional"
                  value={orgId}
                  onChange={(e) => {
                    setOrgId(e.target.value);
                  }}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="userId">User ID</Label>
                <Input
                  id="userId"
                  placeholder="optional"
                  value={userId}
                  onChange={(e) => {
                    setUserId(e.target.value);
                  }}
                />
              </div>
            </CardContent>
          </Card>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {debouncedFilters.requestId.trim() && (
            <Alert className="border-blue-200 bg-blue-50">
              <AlertDescription className="flex flex-wrap items-center justify-between gap-2 text-blue-900">
                <span>
                  Showing correlated entries for request ID{' '}
                  <span className="font-mono">{debouncedFilters.requestId.trim()}</span>.
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setRequestId('');
                  }}
                >
                  Clear correlation filter
                </Button>
              </AlertDescription>
            </Alert>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Entries</CardTitle>
              <CardDescription>Newest entries appear first.</CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="py-12 text-center text-muted-foreground">Loading logs...</div>
              ) : logs.length === 0 ? (
                <EmptyState
                  icon={RefreshCw}
                  title="No logs match the filters."
                  description="Broaden time range or clear filters to see recent entries."
                  actions={[
                    {
                      label: 'Clear filters',
                      onClick: clearLogFilters,
                    },
                  ]}
                  className="py-12"
                />
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Time</TableHead>
                      <TableHead>Level</TableHead>
                      <TableHead>Message</TableHead>
                      <TableHead>Logger</TableHead>
                      <TableHead>Request ID</TableHead>
                      <TableHead>Org/User</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {logs.map((entry, idx) => (
                      <TableRow
                        key={
                          (entry.request_id
                            ? `${entry.request_id}-${idx}`
                            : '') ||
                          `${entry.timestamp || 'log'}-${entry.logger || 'unknown'}-${
                            entry.org_id || 'org'
                          }-${entry.user_id || 'user'}-${idx}`
                        }
                      >
                        <TableCell className="whitespace-nowrap">
                          {formatLogDateTime(entry.timestamp)}
                        </TableCell>
                        <TableCell>
                          <Badge {...getLevelBadgeProps(entry.level)}>
                            {entry.level || '—'}
                          </Badge>
                        </TableCell>
                        <TableCell className="max-w-[420px]">
                          {entry.message ? (
                            <details>
                              <summary className="cursor-pointer truncate" title={entry.message}>
                                {entry.message}
                              </summary>
                              <div className="mt-2 whitespace-pre-wrap break-words text-sm text-muted-foreground">
                                {entry.message}
                              </div>
                            </details>
                          ) : (
                            '—'
                          )}
                        </TableCell>
                        <TableCell className="max-w-[240px] truncate">
                          {entry.logger || entry.module || '—'}
                        </TableCell>
                        <TableCell className="max-w-[200px] truncate">
                          {entry.request_id ? (
                            <Button
                              variant="link"
                              className="h-auto p-0 font-mono text-xs"
                              onClick={() => {
                                applyRequestCorrelationFilter(entry.request_id ?? '');
                              }}
                            >
                              {entry.request_id}
                            </Button>
                          ) : (
                            '—'
                          )}
                        </TableCell>
                        <TableCell>
                          {entry.org_id ? `Org ${entry.org_id}` : '—'}{' '}
                          {entry.user_id ? `• User ${entry.user_id}` : ''}
                        </TableCell>
                        <TableCell className="text-right">
                          {entry.request_id ? (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                applyRequestCorrelationFilter(entry.request_id ?? '');
                              }}
                            >
                              View correlated logs
                            </Button>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          <Pagination
            currentPage={page}
            totalPages={totalPages}
            totalItems={total}
            pageSize={pageSize}
            onPageChange={setPage}
            onPageSizeChange={setPageSize}
          />
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

// Wrap with Suspense for useSearchParams
export default function LogsPage() {
  return (
    <Suspense fallback={
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="flex flex-col gap-6 p-6">
            <div className="mb-8">
              <div className="h-8 w-40 bg-muted rounded animate-pulse mb-2" />
              <div className="h-4 w-64 bg-muted rounded animate-pulse" />
            </div>
            <div className="h-96 bg-muted rounded animate-pulse" />
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    }>
      <LogsPageContent />
    </Suspense>
  );
}
