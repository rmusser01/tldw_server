'use client';

import { useCallback, useEffect, useMemo, useRef, useState, Suspense } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Pagination } from '@/components/ui/pagination';
import { FileText, RefreshCw, Filter, Eye, Clipboard, Trash2, Bell, ExternalLink, ShieldAlert } from 'lucide-react';
import { api } from '@/lib/api-client';
import type { AuditLog, UserWithKeyCount } from '@/types';
import { ExportMenu } from '@/components/ui/export-menu';
import { exportAuditLogs, ExportFormat } from '@/lib/export';
import { Skeleton, TableSkeleton } from '@/components/ui/skeleton';
import { useUrlMultiState, useUrlPagination } from '@/lib/use-url-state';
import { useOrgContext } from '@/components/OrgContextSwitcher';
import { useToast } from '@/components/ui/toast';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { getScopedItem, setScopedItem } from '@/lib/scoped-storage';
import { logger } from '@/lib/logger';

type AuditFilters = {
  user: string;
  action: string;
  actionPrefix: string;
  resource: string;
  start: string;
  end: string;
};

type SavedAuditSearch = {
  id: string;
  name: string;
  filters: AuditFilters;
  alertOnPattern: boolean;
  createdAt: string;
  updatedAt: string;
  lastCheckedAt?: string;
  lastMatchedEventId?: string;
  lastMatchCount?: number;
};

type ComplianceReportType = 'activity_summary' | 'access_review' | 'data_access';

type ComplianceReportSummary = {
  categoryCounts: Array<{ category: string; count: number }>;
  userCounts: Array<{ userLabel: string; count: number }>;
  anomalies: string[];
};

type AdminUserCandidate = UserWithKeyCount & {
  is_superuser?: boolean;
};

const SAVED_SEARCHES_STORAGE_KEY = 'admin.audit.saved-searches.v1';
const ALERT_CHECK_INTERVAL_MS = 60_000;
const COMPLIANCE_REPORT_LIMIT = 5000;

/**
 * Action prefix used for the "Admin Actions Only" quick filter.
 * Matches dotted-namespace admin actions: maintenance.*, monitoring.*,
 * backup.*, config.*, incident.*, webhook.*, feature_flag.*, etc.
 * The trailing `*` triggers prefix matching on the backend.
 */
const ADMIN_ACTIONS_FILTER_PREFIXES = [
  'maintenance.',
  'monitoring.',
  'backup.',
  'backup_schedule.',
  'config.',
  'incident.',
  'webhook.',
  'feature_flag.',
  'data_subject_request.',
  'admin.',
] as const;

/**
 * Client-side predicate to check whether an audit log entry looks like an
 * admin-originated action, based on well-known action name prefixes.
 */
const ADMIN_ACTIONS_GLOB = ADMIN_ACTIONS_FILTER_PREFIXES.map((p) => `${p}*`).join(',');

const isAdminAction = (action: string): boolean => {
  const lower = action.toLowerCase();
  return ADMIN_ACTIONS_FILTER_PREFIXES.some((prefix) => lower.startsWith(prefix));
};

const COMPLIANCE_REPORT_TYPE_LABELS: Record<ComplianceReportType, string> = {
  activity_summary: 'Activity Summary',
  access_review: 'Access Review',
  data_access: 'Data Access',
};

const parseDateInput = (value: string) => {
  if (!value) return null;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const date = new Date(`${value}T00:00:00`);
  return Number.isNaN(date.getTime()) ? null : date;
};

const getDateRangeError = (start: string, end: string) => {
  if (!start && !end) return '';
  const startDate = parseDateInput(start);
  const endDate = parseDateInput(end);
  if (start && !startDate) return 'Start date is invalid.';
  if (end && !endDate) return 'End date is invalid.';
  if (startDate && endDate && startDate > endDate) {
    return 'Start date must be on or before end date.';
  }
  return '';
};

const parseUserFilter = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  return /^\d+$/.test(trimmed) ? trimmed : null;
};

const normalizeAuditFilters = (filters: AuditFilters): AuditFilters => ({
  user: filters.user.trim(),
  action: filters.action.trim(),
  actionPrefix: filters.actionPrefix.trim(),
  resource: filters.resource.trim(),
  start: filters.start.trim(),
  end: filters.end.trim(),
});

const hasAnyFilterValue = (filters: AuditFilters) =>
  Boolean(filters.user || filters.action || filters.actionPrefix || filters.resource || filters.start || filters.end);

const matchesActionPrefix = (action: string, actionPrefix: string) => {
  const normalizedAction = action.trim().toLowerCase();
  const normalizedPrefix = actionPrefix.trim().toLowerCase();
  if (!normalizedAction || !normalizedPrefix) return true;
  if (normalizedPrefix === 'destructive') {
    return ['delete', 'revoke', 'disable', 'reset'].some((keyword) => normalizedAction.includes(keyword));
  }
  return normalizedAction.startsWith(normalizedPrefix) || normalizedAction.includes(`.${normalizedPrefix}`);
};

const escapeHtml = (value: string) =>
  value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');

const summarizeComplianceReport = (entries: AuditLog[]): ComplianceReportSummary => {
  const categoryMap = new Map<string, number>();
  const userMap = new Map<string, number>();
  let destructiveEventCount = 0;
  let failedAuthCount = 0;

  entries.forEach((entry) => {
    const normalizedAction = entry.action?.trim().toLowerCase() || 'unknown';
    const category = normalizedAction.includes('.')
      ? normalizedAction.split('.')[0]
      : normalizedAction || 'unknown';
    categoryMap.set(category, (categoryMap.get(category) || 0) + 1);

    const userLabel = entry.username
      ? `${entry.username} (${entry.user_id})`
      : entry.user_id
        ? `User ${entry.user_id}`
        : 'System/Unknown';
    userMap.set(userLabel, (userMap.get(userLabel) || 0) + 1);

    if (
      normalizedAction.includes('delete')
      || normalizedAction.includes('revoke')
      || normalizedAction.includes('disable')
      || normalizedAction.includes('reset')
    ) {
      destructiveEventCount += 1;
    }
    if (normalizedAction.includes('failed') || normalizedAction.includes('denied')) {
      failedAuthCount += 1;
    }
  });

  const categoryCounts = Array.from(categoryMap.entries())
    .map(([category, count]) => ({ category, count }))
    .sort((a, b) => b.count - a.count);
  const userCounts = Array.from(userMap.entries())
    .map(([userLabel, count]) => ({ userLabel, count }))
    .sort((a, b) => b.count - a.count);

  const anomalies: string[] = [];
  if (destructiveEventCount > 0) {
    anomalies.push(`${destructiveEventCount} high-impact modification events detected.`);
  }
  if (failedAuthCount > 0) {
    anomalies.push(`${failedAuthCount} failed or denied actions detected.`);
  }
  const mostActiveUser = userCounts[0];
  if (mostActiveUser && mostActiveUser.count >= 20) {
    anomalies.push(`High concentration: ${mostActiveUser.userLabel} performed ${mostActiveUser.count} actions.`);
  }
  if (anomalies.length === 0) {
    anomalies.push('No anomaly signals detected for the selected period.');
  }

  return { categoryCounts, userCounts, anomalies };
};

const buildComplianceReportHtml = ({
  reportType,
  reportPeriodLabel,
  selectedOrgId,
  generatedAt,
  totalEvents,
  categoryCounts,
  userCounts,
  anomalies,
}: {
  reportType: ComplianceReportType;
  reportPeriodLabel: string;
  selectedOrgId?: number;
  generatedAt: string;
  totalEvents: number;
  categoryCounts: Array<{ category: string; count: number }>;
  userCounts: Array<{ userLabel: string; count: number }>;
  anomalies: string[];
}) => {
  const categoryRows = categoryCounts.length > 0
    ? categoryCounts
      .map((entry) => `<tr><td>${escapeHtml(entry.category)}</td><td>${entry.count}</td></tr>`)
      .join('')
    : '<tr><td colspan="2">No events found.</td></tr>';
  const userRows = userCounts.length > 0
    ? userCounts
      .slice(0, 20)
      .map((entry) => `<tr><td>${escapeHtml(entry.userLabel)}</td><td>${entry.count}</td></tr>`)
      .join('')
    : '<tr><td colspan="2">No user activity found.</td></tr>';
  const anomalyItems = anomalies
    .map((entry) => `<li>${escapeHtml(entry)}</li>`)
    .join('');

  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Compliance Report - ${escapeHtml(COMPLIANCE_REPORT_TYPE_LABELS[reportType])}</title>
    <style>
      body { font-family: "IBM Plex Sans", Arial, sans-serif; margin: 24px; color: #111827; }
      h1, h2 { margin: 0 0 12px; }
      .meta { margin: 0 0 16px; font-size: 14px; color: #374151; }
      .card { border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
      table { width: 100%; border-collapse: collapse; margin-top: 8px; }
      th, td { border: 1px solid #d1d5db; padding: 8px; text-align: left; font-size: 14px; }
      th { background: #f3f4f6; }
      ul { margin: 8px 0 0 20px; }
    </style>
  </head>
  <body>
    <h1>Compliance Report</h1>
    <p class="meta"><strong>Type:</strong> ${escapeHtml(COMPLIANCE_REPORT_TYPE_LABELS[reportType])}</p>
    <p class="meta"><strong>Period:</strong> ${escapeHtml(reportPeriodLabel)}</p>
    <p class="meta"><strong>Generated:</strong> ${escapeHtml(generatedAt)}</p>
    <p class="meta"><strong>Organization:</strong> ${selectedOrgId ? `Org ${selectedOrgId}` : 'All organizations in scope'}</p>

    <section class="card">
      <h2>Summary</h2>
      <p>Total events: <strong>${totalEvents}</strong></p>
    </section>

    <section class="card">
      <h2>Events by Category</h2>
      <table>
        <thead><tr><th>Category</th><th>Events</th></tr></thead>
        <tbody>${categoryRows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>User Activity Summary</h2>
      <table>
        <thead><tr><th>User</th><th>Events</th></tr></thead>
        <tbody>${userRows}</tbody>
      </table>
    </section>

    <section class="card">
      <h2>Anomaly Highlights</h2>
      <ul>${anomalyItems}</ul>
    </section>
  </body>
</html>`;
};

const buildAuditParams = (
  activeFilters: AuditFilters,
  page: number,
  size: number,
  selectedOrgId?: number,
): Record<string, string> => {
  const normalized = normalizeAuditFilters(activeFilters);
  const params: Record<string, string> = {
    limit: String(size),
    offset: String((page - 1) * size),
  };

  if (selectedOrgId) {
    params.org_id = String(selectedOrgId);
  }

  const userIdParam = parseUserFilter(normalized.user);
  if (userIdParam) params.user_id = userIdParam;
  if (normalized.action && !normalized.actionPrefix) params.action = normalized.action;
  if (normalized.resource) params.resource = normalized.resource;
  if (normalized.start) params.start = normalized.start;
  if (normalized.end) params.end = normalized.end;

  return params;
};

const generateSavedSearchId = () => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const parseSavedSearches = (value: string | null): SavedAuditSearch[] => {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) return [];
    return parsed.reduce<SavedAuditSearch[]>((acc, entry) => {
      if (typeof entry !== 'object' || entry === null) return acc;
      const candidate = entry as Partial<SavedAuditSearch> & { filters?: Partial<AuditFilters> };
      if (typeof candidate.id !== 'string' || typeof candidate.name !== 'string') return acc;
      const filters: AuditFilters = {
        user: String(candidate.filters?.user ?? ''),
        action: String(candidate.filters?.action ?? ''),
        actionPrefix: String(candidate.filters?.actionPrefix ?? ''),
        resource: String(candidate.filters?.resource ?? ''),
        start: String(candidate.filters?.start ?? ''),
        end: String(candidate.filters?.end ?? ''),
      };
      acc.push({
        id: candidate.id,
        name: candidate.name,
        filters: normalizeAuditFilters(filters),
        alertOnPattern: Boolean(candidate.alertOnPattern),
        createdAt: String(candidate.createdAt ?? new Date().toISOString()),
        updatedAt: String(candidate.updatedAt ?? candidate.createdAt ?? new Date().toISOString()),
        lastCheckedAt: candidate.lastCheckedAt ? String(candidate.lastCheckedAt) : undefined,
        lastMatchedEventId: candidate.lastMatchedEventId ? String(candidate.lastMatchedEventId) : undefined,
        lastMatchCount: candidate.lastMatchCount !== undefined ? Number(candidate.lastMatchCount) : undefined,
      });
      return acc;
    }, []);
  } catch {
    return [];
  }
};

function AuditPageContent() {
  const { selectedOrg } = useOrgContext();
  const { success, error: showError } = useToast();
  const router = useRouter();
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [totalItems, setTotalItems] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null);
  const [savedSearches, setSavedSearches] = useState<SavedAuditSearch[]>([]);
  const [savedSearchName, setSavedSearchName] = useState('');
  const [activeSavedSearchId, setActiveSavedSearchId] = useState<string | null>(null);
  const [reportType, setReportType] = useState<ComplianceReportType>('activity_summary');
  const [reportStartDate, setReportStartDate] = useState(() => {
    const start = new Date();
    start.setDate(start.getDate() - 29);
    return start.toISOString().slice(0, 10);
  });
  const [reportEndDate, setReportEndDate] = useState(() => new Date().toISOString().slice(0, 10));
  const [reportGenerating, setReportGenerating] = useState(false);
  const [adminUserIds, setAdminUserIds] = useState<Set<number> | null>(null);
  const [adminUsersFilterActive, setAdminUsersFilterActive] = useState(false);
  const [adminUsersLoading, setAdminUsersLoading] = useState(false);
  const storageHydratedRef = useRef(false);
  const savedSearchesRef = useRef<SavedAuditSearch[]>([]);

  // URL state for filters
  const [filters, setFilters, clearFilters] = useUrlMultiState<AuditFilters>({
    user: '',
    action: '',
    actionPrefix: '',
    resource: '',
    start: '',
    end: '',
  });
  const [debouncedFilters, setDebouncedFilters] = useState<AuditFilters>(filters);

  // URL state for pagination
  const { page: currentPage, pageSize, setPage: setCurrentPage, setPageSize, resetPagination } = useUrlPagination();

  useEffect(() => {
    const handle = window.setTimeout(() => {
      setDebouncedFilters((prev) => {
        if (
          prev.user === filters.user
          && prev.action === filters.action
          && prev.actionPrefix === filters.actionPrefix
          && prev.resource === filters.resource
          && prev.start === filters.start
          && prev.end === filters.end
        ) {
          return prev;
        }
        return filters;
      });
    }, 300);
    return () => window.clearTimeout(handle);
  }, [filters]);

  useEffect(() => {
    const parsed = parseSavedSearches(getScopedItem(SAVED_SEARCHES_STORAGE_KEY));
    setSavedSearches(parsed);
    savedSearchesRef.current = parsed;
    storageHydratedRef.current = true;
  }, []);

  useEffect(() => {
    if (!storageHydratedRef.current) return;
    savedSearchesRef.current = savedSearches;
    setScopedItem(SAVED_SEARCHES_STORAGE_KEY, JSON.stringify(savedSearches));
  }, [savedSearches]);

  const loadLogs = useCallback(async (activeFilters: AuditFilters, page: number, size: number) => {
    try {
      setLoading(true);
      setError('');

      const rangeError = getDateRangeError(activeFilters.start, activeFilters.end);
      if (rangeError) {
        setLogs([]);
        setTotalItems(0);
        return;
      }

      const normalizedFilters = normalizeAuditFilters(activeFilters);
      const prefixFiltering = Boolean(normalizedFilters.actionPrefix);
      const params = buildAuditParams(
        activeFilters,
        prefixFiltering ? 1 : page,
        prefixFiltering ? COMPLIANCE_REPORT_LIMIT : size,
        selectedOrg?.id
      );
      const data = await api.getAuditLogs(params);
      const rawItems = Array.isArray(data) ? data : data.entries ?? [];
      if (prefixFiltering) {
        const matchingItems = rawItems.filter((item) => matchesActionPrefix(item.action ?? '', normalizedFilters.actionPrefix));
        const startIndex = (page - 1) * size;
        setLogs(matchingItems.slice(startIndex, startIndex + size));
        setTotalItems(matchingItems.length);
      } else {
        setLogs(rawItems);
        setTotalItems(Number(data.total ?? rawItems.length ?? 0));
      }
    } catch (err: unknown) {
      logger.error('Failed to load audit logs', { component: 'AuditPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to load audit logs');
      setLogs([]);
      setTotalItems(0);
    } finally {
      setLoading(false);
    }
  }, [selectedOrg]);

  useEffect(() => {
    void loadLogs(debouncedFilters, currentPage, pageSize);
  }, [currentPage, debouncedFilters, loadLogs, pageSize]);

  const handleFilterChange = (updates: Partial<AuditFilters>) => {
    setFilters(updates);
    resetPagination();
  };

  const handleClearFilters = () => {
    clearFilters();
    setActiveSavedSearchId(null);
    setAdminUsersFilterActive(false);
    resetPagination();
  };

  const loadAdminUserIds = useCallback(async (): Promise<Set<number>> => {
    if (adminUserIds !== null) return adminUserIds;
    try {
      setAdminUsersLoading(true);
      const response = await api.getUsersPage({ limit: '200' });
      const items: AdminUserCandidate[] = Array.isArray(response?.items) ? response.items : [];
      const ids = new Set<number>();
      for (const user of items) {
        const role = user.role.toLowerCase();
        const roles = Array.isArray(user.roles) ? user.roles.map((r) => String(r).toLowerCase()) : [];
        if (
          role === 'admin' || role === 'owner' || role === 'super_admin'
          || roles.includes('admin') || roles.includes('owner') || roles.includes('super_admin')
          || user.is_superuser === true
        ) {
          const id = Number(user.id);
          if (Number.isFinite(id)) ids.add(id);
        }
      }
      setAdminUserIds(ids);
      return ids;
    } catch {
      return new Set();
    } finally {
      setAdminUsersLoading(false);
    }
  }, [adminUserIds]);

  const handleToggleAdminUsersFilter = useCallback(async () => {
    if (adminUsersFilterActive) {
      setAdminUsersFilterActive(false);
      return;
    }
    await loadAdminUserIds();
    setAdminUsersFilterActive(true);
  }, [adminUsersFilterActive, loadAdminUserIds]);

  const displayedLogs = useMemo(() => {
    if (!adminUsersFilterActive || !adminUserIds) return logs;
    return logs.filter((log) => adminUserIds.has(Number(log.user_id)));
  }, [logs, adminUsersFilterActive, adminUserIds]);

  const dateRangeError = useMemo(() => getDateRangeError(filters.start, filters.end), [filters.end, filters.start]);
  const complianceDateRangeError = useMemo(
    () => getDateRangeError(reportStartDate, reportEndDate),
    [reportEndDate, reportStartDate]
  );

  const handleSaveCurrentSearch = () => {
    const trimmedName = savedSearchName.trim();
    if (!trimmedName) {
      showError('Saved search name required', 'Enter a name before saving.');
      return;
    }

    const normalizedFilters = normalizeAuditFilters(filters);
    if (!hasAnyFilterValue(normalizedFilters)) {
      showError('No filters to save', 'Set at least one filter value before saving.');
      return;
    }

    const nowIso = new Date().toISOString();
    setSavedSearches((previous) => {
      const existing = previous.find((entry) => entry.name.toLowerCase() === trimmedName.toLowerCase());
      if (existing) {
        return previous.map((entry) =>
          entry.id === existing.id
            ? {
                ...entry,
                name: trimmedName,
                filters: normalizedFilters,
                updatedAt: nowIso,
              }
            : entry
        );
      }

      return [
        {
          id: generateSavedSearchId(),
          name: trimmedName,
          filters: normalizedFilters,
          alertOnPattern: false,
          createdAt: nowIso,
          updatedAt: nowIso,
        },
        ...previous,
      ];
    });

    setSavedSearchName('');
    success('Saved search stored');
  };

  const handleApplySavedSearch = (search: SavedAuditSearch) => {
    setFilters(search.filters);
    setDebouncedFilters(search.filters);
    setActiveSavedSearchId(search.id);
    resetPagination();
  };

  const handleDeleteSavedSearch = (searchId: string) => {
    setSavedSearches((previous) => previous.filter((entry) => entry.id !== searchId));
    if (activeSavedSearchId === searchId) {
      setActiveSavedSearchId(null);
    }
  };

  const handleAlertToggle = (searchId: string, enabled: boolean) => {
    setSavedSearches((previous) =>
      previous.map((entry) =>
        entry.id === searchId
          ? {
              ...entry,
              alertOnPattern: enabled,
              // Reset baseline when enabling so first check doesn't immediately alert.
              lastMatchedEventId: enabled ? undefined : entry.lastMatchedEventId,
              updatedAt: new Date().toISOString(),
            }
          : entry
      )
    );
  };

  const alertSearchDependencyKey = useMemo(
    () =>
      JSON.stringify(
        savedSearches.map((entry) => ({
          id: entry.id,
          alertOnPattern: entry.alertOnPattern,
          lastMatchedEventId: entry.lastMatchedEventId ?? null,
          filters: normalizeAuditFilters(entry.filters),
        }))
      ),
    [savedSearches]
  );

  useEffect(() => {
    let cancelled = false;

    const checkSavedSearches = async () => {
      const enabledSearches = savedSearchesRef.current.filter((entry) => entry.alertOnPattern);
      if (enabledSearches.length === 0) return;

      const updates = new Map<string, Partial<SavedAuditSearch>>();

      await Promise.all(enabledSearches.map(async (search) => {
        try {
          const params = buildAuditParams(search.filters, 1, 1, selectedOrg?.id);
          const result = await api.getAuditLogs(params);
          const entries = Array.isArray(result) ? result : result.entries ?? [];
          const latest = entries[0];
          const latestId = latest?.id ? String(latest.id) : undefined;
          const totalMatches = Number((result as { total?: number }).total ?? entries.length ?? 0);
          const checkedAt = new Date().toISOString();

          if (!latestId) {
            updates.set(search.id, {
              lastCheckedAt: checkedAt,
              lastMatchCount: totalMatches,
            });
            return;
          }

          if (!search.lastMatchedEventId) {
            updates.set(search.id, {
              lastMatchedEventId: latestId,
              lastCheckedAt: checkedAt,
              lastMatchCount: totalMatches,
            });
            return;
          }

          if (search.lastMatchedEventId !== latestId) {
            success(
              'Audit pattern matched',
              `${search.name} found new matching events (${totalMatches}).`
            );
            void api.testNotification().catch(() => {
              // Notification channels may not be configured; UI toast is the fallback signal.
            });
            updates.set(search.id, {
              lastMatchedEventId: latestId,
              lastCheckedAt: checkedAt,
              lastMatchCount: totalMatches,
            });
            return;
          }

          updates.set(search.id, {
            lastCheckedAt: checkedAt,
            lastMatchCount: totalMatches,
          });
        } catch {
          // Keep polling even if an individual query fails.
        }
      }));

      if (cancelled || updates.size === 0) return;

      setSavedSearches((previous) =>
        previous.map((entry) => {
          const update = updates.get(entry.id);
          if (!update) return entry;
          return {
            ...entry,
            ...update,
            updatedAt: new Date().toISOString(),
          };
        })
      );
    };

    void checkSavedSearches();
    const intervalId = window.setInterval(() => {
      void checkSavedSearches();
    }, ALERT_CHECK_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [alertSearchDependencyKey, selectedOrg, success]);

  const handleExport = (format: ExportFormat) => {
    exportAuditLogs(logs, format);
  };

  const handleGenerateComplianceReport = async () => {
    if (complianceDateRangeError) {
      showError('Invalid compliance report date range', complianceDateRangeError);
      return;
    }
    try {
      setReportGenerating(true);
      const params: Record<string, string> = {
        limit: String(COMPLIANCE_REPORT_LIMIT),
        offset: '0',
      };
      if (selectedOrg?.id) {
        params.org_id = String(selectedOrg.id);
      }
      if (reportStartDate) {
        params.start = reportStartDate;
      }
      if (reportEndDate) {
        params.end = reportEndDate;
      }
      if (!reportStartDate && !reportEndDate) {
        params.days = '30';
      }

      const response = await api.getAuditLogs(params);
      const entries = Array.isArray(response) ? response : response.entries ?? [];
      const summary = summarizeComplianceReport(entries);
      const generatedAt = new Date().toISOString();
      const periodLabel = reportStartDate && reportEndDate
        ? `${reportStartDate} to ${reportEndDate}`
        : 'Last 30 days';
      const reportHtml = buildComplianceReportHtml({
        reportType,
        reportPeriodLabel: periodLabel,
        selectedOrgId: selectedOrg?.id,
        generatedAt,
        totalEvents: entries.length,
        categoryCounts: summary.categoryCounts,
        userCounts: summary.userCounts,
        anomalies: summary.anomalies,
      });
      const blob = new Blob([reportHtml], { type: 'text/html;charset=utf-8' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      const suffix = new Date().toISOString().slice(0, 10);
      link.href = url;
      link.download = `compliance-report-${reportType.replaceAll('_', '-')}-${suffix}.html`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      if (entries.length >= COMPLIANCE_REPORT_LIMIT) {
        showError(
          'Report may be incomplete',
          `The report reached the ${COMPLIANCE_REPORT_LIMIT.toLocaleString()} event limit. Narrow the date range for a complete report.`
        );
      }
      success(
        'Compliance report generated',
        `${COMPLIANCE_REPORT_TYPE_LABELS[reportType]} exported with ${entries.length} events.`
      );
    } catch (err: unknown) {
      logger.error('Failed to generate compliance report', { component: 'AuditPage', error: err instanceof Error ? err.message : String(err) });
      showError(
        'Compliance report generation failed',
        err instanceof Error && err.message ? err.message : 'Unable to build compliance report.'
      );
    } finally {
      setReportGenerating(false);
    }
  };

  const handleCopyRaw = async () => {
    if (!selectedLog) return;
    const rawPayload = selectedLog.raw ?? {
      id: selectedLog.id,
      timestamp: selectedLog.timestamp,
      user_id: selectedLog.user_id,
      action: selectedLog.action,
      resource: selectedLog.resource,
      details: selectedLog.details,
      ip_address: selectedLog.ip_address,
      username: selectedLog.username,
    };
    try {
      if (!navigator.clipboard) {
        showError('Copy failed', 'Clipboard API not available. Ensure you are using HTTPS.');
        return;
      }
      await navigator.clipboard.writeText(JSON.stringify(rawPayload, null, 2));
      success('Copied audit event');
    } catch (err) {
      logger.error('Failed to copy audit log', { component: 'AuditPage', error: err instanceof Error ? err.message : String(err) });
      showError('Copy failed', 'Unable to copy audit event details.');
    }
  };

  // Pagination
  const totalPages = Math.ceil(totalItems / pageSize);

  const formatTimestamp = (ts: string) => {
    return new Date(ts).toLocaleString();
  };

  const getActionBadgeVariant = (action: string): 'default' | 'secondary' | 'destructive' | 'outline' => {
    const lowerAction = action.toLowerCase();
    if (lowerAction.includes('delete') || lowerAction.includes('revoke')) return 'destructive';
    if (lowerAction.includes('create') || lowerAction.includes('add')) return 'default';
    if (lowerAction.includes('update') || lowerAction.includes('modify')) return 'secondary';
    return 'outline';
  };

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8 flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold">Audit Logs</h1>
                <p className="text-muted-foreground">
                  Track all system activities and user actions
                </p>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => loadLogs(debouncedFilters, currentPage, pageSize)}
                  disabled={Boolean(dateRangeError)}
                  loading={loading}
                  loadingText="Refreshing..."
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Refresh
                </Button>
                <ExportMenu
                  onExport={handleExport}
                  disabled={logs.length === 0}
                />
              </div>
            </div>

            {error && (
              <Alert variant="destructive" className="mb-6">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Info Card */}
            <Card className="mb-6">
              <CardContent className="pt-6">
                <div className="flex items-start gap-4">
                  <FileText className="h-8 w-8 text-primary mt-1" />
                  <div>
                    <h3 className="font-semibold">Audit Trail</h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      View detailed records of all administrative actions, user activities, and system events.
                      Use filters to narrow down results by user, action type, or date range.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="mb-6">
              <CardHeader>
                <CardTitle>Compliance Reports</CardTitle>
                <CardDescription>
                  Generate a downloadable HTML report with activity summary, category distribution, user activity, and anomaly highlights.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
                  <div className="space-y-2">
                    <Label htmlFor="complianceReportType">Report Type</Label>
                    <Select
                      id="complianceReportType"
                      value={reportType}
                      onChange={(event) => setReportType(event.target.value as ComplianceReportType)}
                    >
                      <option value="activity_summary">Activity Summary</option>
                      <option value="access_review">Access Review</option>
                      <option value="data_access">Data Access</option>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="complianceReportStartDate">Report Start Date</Label>
                    <Input
                      id="complianceReportStartDate"
                      type="date"
                      value={reportStartDate}
                      onChange={(event) => setReportStartDate(event.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="complianceReportEndDate">Report End Date</Label>
                    <Input
                      id="complianceReportEndDate"
                      type="date"
                      value={reportEndDate}
                      onChange={(event) => setReportEndDate(event.target.value)}
                    />
                  </div>
                  <div className="flex items-end">
                    <Button
                      onClick={handleGenerateComplianceReport}
                      loading={reportGenerating}
                      loadingText="Generating..."
                      disabled={Boolean(complianceDateRangeError)}
                    >
                      Generate Compliance Report
                    </Button>
                  </div>
                </div>
                {complianceDateRangeError ? (
                  <Alert variant="destructive" className="mt-4">
                    <AlertDescription>{complianceDateRangeError}</AlertDescription>
                  </Alert>
                ) : null}
              </CardContent>
            </Card>

            {/* Filters */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Filter className="h-5 w-5" />
                  Filters & Saved Searches
                </CardTitle>
                <CardDescription>
                  Save common filter combinations and optionally alert on new matching events.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-6 space-y-3 rounded-md border p-4">
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-[minmax(240px,1fr)_auto]">
                    <div className="space-y-2">
                      <Label htmlFor="savedSearchName">Saved search name</Label>
                      <Input
                        id="savedSearchName"
                        placeholder="e.g., Failed logins (today)"
                        value={savedSearchName}
                        onChange={(event) => setSavedSearchName(event.target.value)}
                      />
                    </div>
                    <div className="flex items-end">
                      <Button
                        onClick={handleSaveCurrentSearch}
                        variant="outline"
                        data-testid="save-audit-search"
                      >
                        Save Current Filters
                      </Button>
                    </div>
                  </div>

                  {savedSearches.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                      No saved searches yet.
                    </p>
                  ) : (
                    <div className="flex flex-wrap gap-2">
                      {savedSearches.map((search) => (
                        <div
                          key={search.id}
                          className="flex flex-wrap items-center gap-2 rounded-full border px-3 py-2"
                          data-testid={`saved-search-${search.id}`}
                        >
                          <Button
                            type="button"
                            size="sm"
                            variant={activeSavedSearchId === search.id ? 'default' : 'outline'}
                            className="rounded-full"
                            onClick={() => handleApplySavedSearch(search)}
                          >
                            {search.name}
                          </Button>
                          <label className="flex items-center gap-2 text-xs text-muted-foreground">
                            <Checkbox
                              checked={search.alertOnPattern}
                              onCheckedChange={(checked) => handleAlertToggle(search.id, checked)}
                              aria-label={`Enable alert on pattern for ${search.name}`}
                            />
                            <Bell className="h-3 w-3" />
                            Alert on pattern
                          </label>
                          {typeof search.lastMatchCount === 'number' ? (
                            <Badge variant="secondary">
                              {search.lastMatchCount} match{search.lastMatchCount === 1 ? '' : 'es'}
                            </Badge>
                          ) : null}
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            onClick={() => handleDeleteSavedSearch(search.id)}
                            aria-label={`Delete saved search ${search.name}`}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Alert checks run every 60 seconds and trigger in-app notifications (plus monitoring notification test when configured).
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="userFilter">User ID (exact)</Label>
                    <Input
                      id="userFilter"
                      placeholder="e.g., 42"
                      value={filters.user}
                      onChange={(e) => handleFilterChange({ user: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="actionFilter">Action (exact or prefix*)</Label>
                    <Input
                      id="actionFilter"
                      placeholder="e.g., user.create or admin*"
                      value={filters.action}
                      onChange={(e) => handleFilterChange({ action: e.target.value, actionPrefix: '' })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="resourceFilter">Resource</Label>
                    <Input
                      id="resourceFilter"
                      placeholder="e.g., user, api_key..."
                      value={filters.resource}
                      onChange={(e) => handleFilterChange({ resource: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="startDate">Start Date</Label>
                    <Input
                      id="startDate"
                      type="date"
                      value={filters.start}
                      onChange={(e) => handleFilterChange({ start: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="endDate">End Date</Label>
                    <Input
                      id="endDate"
                      type="date"
                      value={filters.end}
                      onChange={(e) => handleFilterChange({ end: e.target.value })}
                    />
                  </div>
                </div>
                {dateRangeError && (
                  <Alert variant="destructive" className="mt-4">
                    <AlertTitle>Invalid date range</AlertTitle>
                    <AlertDescription>{dateRangeError}</AlertDescription>
                  </Alert>
                )}
                <div className="flex flex-wrap gap-2 mt-4">
                  <Button variant="outline" onClick={handleClearFilters}>
                    Clear Filters
                  </Button>
                  <Button
                    variant={filters.action === ADMIN_ACTIONS_GLOB ? 'default' : 'outline'}
                    onClick={() => {
                      if (filters.action === ADMIN_ACTIONS_GLOB) {
                        handleFilterChange({ action: '' });
                      } else {
                        handleFilterChange({ action: ADMIN_ACTIONS_GLOB });
                      }
                    }}
                    data-testid="admin-actions-filter"
                  >
                    <ShieldAlert className="mr-1.5 h-4 w-4" />
                    Admin Actions Only
                  </Button>
                  <Button
                    variant={adminUsersFilterActive ? 'default' : 'outline'}
                    onClick={() => void handleToggleAdminUsersFilter()}
                    disabled={adminUsersLoading}
                    data-testid="admin-users-filter"
                  >
                    <Filter className="mr-1.5 h-4 w-4" />
                    {adminUsersLoading ? 'Loading...' : 'Admin Users'}
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => handleFilterChange({ action: '', actionPrefix: 'destructive' })}
                  >
                    Destructive Actions
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => handleFilterChange({ action: '', actionPrefix: 'login' })}
                  >
                    Login Events
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => handleFilterChange({ resource: 'api_key' })}
                  >
                    API Key Activity
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Dialog open={!!selectedLog} onOpenChange={(open) => !open && setSelectedLog(null)}>
              <DialogContent className="max-w-3xl">
                <DialogHeader>
                  <DialogTitle>Audit Event</DialogTitle>
                  <DialogDescription>
                    {selectedLog ? `${selectedLog.action} · ${formatTimestamp(selectedLog.timestamp)}` : 'Event details'}
                  </DialogDescription>
                </DialogHeader>
                {selectedLog && (
                  <div className="space-y-4">
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-1">
                        <Label>Event ID</Label>
                        <div className="text-sm font-mono">{selectedLog.id}</div>
                      </div>
                      <div className="space-y-1">
                        <Label>Resource</Label>
                        <div className="text-sm font-mono">{selectedLog.resource || '—'}</div>
                      </div>
                      <div className="space-y-1">
                        <Label>User</Label>
                        <div className="text-sm">
                          <div className="font-mono">{selectedLog.user_id || '—'}</div>
                          {selectedLog.username && (
                            <div className="text-xs text-muted-foreground">{selectedLog.username}</div>
                          )}
                          {selectedLog.user_id ? (
                            <Link href={`/users/${selectedLog.user_id}`} className="text-xs text-primary underline">
                              View user
                            </Link>
                          ) : null}
                        </div>
                      </div>
                      <div className="space-y-1">
                        <Label>IP Address</Label>
                        <div className="text-sm font-mono">{selectedLog.ip_address || '—'}</div>
                      </div>
                      {selectedLog.request_id && (
                        <div className="space-y-1">
                          <Label>Request ID</Label>
                          <Button
                            variant="link"
                            className="h-auto p-0 font-mono text-xs"
                            onClick={() => {
                              router.push(`/logs?request_id=${encodeURIComponent(selectedLog.request_id!)}`);
                            }}
                          >
                            {selectedLog.request_id}
                          </Button>
                        </div>
                      )}
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Details</Label>
                      </div>
                      {selectedLog.details ? (
                        <pre className="text-xs bg-muted p-3 rounded max-h-64 overflow-auto">
                          {JSON.stringify(selectedLog.details, null, 2)}
                        </pre>
                      ) : (
                        <div className="text-sm text-muted-foreground">No details captured.</div>
                      )}
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label>Raw Event</Label>
                        <Button variant="outline" size="sm" onClick={handleCopyRaw}>
                          <Clipboard className="mr-2 h-4 w-4" />
                          Copy JSON
                        </Button>
                      </div>
                      <pre className="text-xs bg-muted p-3 rounded max-h-64 overflow-auto">
                        {JSON.stringify(
                          selectedLog.raw ?? selectedLog,
                          null,
                          2
                        )}
                      </pre>
                    </div>
                  </div>
                )}
              </DialogContent>
            </Dialog>

            {/* Logs Table */}
            <Card>
              <CardHeader>
                <CardTitle>Activity Log</CardTitle>
                <CardDescription>
                  {totalItems} record{totalItems !== 1 ? 's' : ''} found
                  {adminUsersFilterActive ? ` (showing ${displayedLogs.length} from admin users)` : ''}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="py-4">
                    <TableSkeleton rows={5} columns={5} />
                  </div>
                ) : displayedLogs.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    {adminUsersFilterActive && logs.length > 0
                      ? 'No audit logs from admin-role users in current results.'
                      : 'No audit logs found for the selected filters.'}
                  </div>
                ) : (
                  <>
                    <div className="overflow-x-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Timestamp</TableHead>
                            <TableHead>User</TableHead>
                            <TableHead>Action</TableHead>
                            <TableHead>Resource</TableHead>
                            <TableHead>Details</TableHead>
                            <TableHead className="text-right">Actions</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {displayedLogs.map((log) => (
                            <TableRow key={log.id}>
                              <TableCell className="text-sm text-muted-foreground whitespace-nowrap">
                                {formatTimestamp(log.timestamp)}
                              </TableCell>
                              <TableCell className="font-mono text-sm">
                                <div className="flex flex-col">
                                  <span>{log.user_id}</span>
                                  {log.username && (
                                    <span className="text-xs text-muted-foreground">{log.username}</span>
                                  )}
                                </div>
                              </TableCell>
                              <TableCell>
                                <Badge variant={getActionBadgeVariant(log.action)}>
                                  {log.action}
                                </Badge>
                              </TableCell>
                              <TableCell className="font-mono text-sm">
                                {log.resource}
                              </TableCell>
                              <TableCell className="max-w-xs">
                                {log.details ? (
                                  <code className="text-xs bg-muted p-1 rounded block truncate">
                                    {JSON.stringify(log.details)}
                                  </code>
                                ) : (
                                  <span className="text-muted-foreground">-</span>
                                )}
                              </TableCell>
                              <TableCell className="text-right">
                                <div className="flex justify-end gap-1">
                                  <Button variant="outline" size="sm" onClick={() => setSelectedLog(log)}>
                                    <Eye className="mr-2 h-4 w-4" />
                                    View
                                  </Button>
                                  {log.request_id && (
                                    <Button
                                      variant="outline"
                                      size="sm"
                                      onClick={() => router.push(`/logs?request_id=${encodeURIComponent(log.request_id!)}`)}
                                    >
                                      <ExternalLink className="mr-2 h-4 w-4" />
                                      View Logs
                                    </Button>
                                  )}
                                </div>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>

                    <Pagination
                      currentPage={currentPage}
                      totalPages={totalPages}
                      totalItems={totalItems}
                      pageSize={pageSize}
                      onPageChange={setCurrentPage}
                      onPageSizeChange={(size) => {
                        setPageSize(size);
                        resetPagination();
                      }}
                    />
                  </>
                )}
              </CardContent>
            </Card>
          </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

// Wrap with Suspense for useSearchParams
export default function AuditPage() {
  return (
    <Suspense fallback={
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8">
              <Skeleton className="h-8 w-32 mb-2" />
              <Skeleton className="h-4 w-64" />
            </div>
            <TableSkeleton rows={5} columns={5} />
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    }>
      <AuditPageContent />
    </Suspense>
  );
}
