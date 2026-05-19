/**
 * Export utilities for downloading data as CSV or JSON
 */

export type ExportFormat = 'csv' | 'json';

interface ExportOptions<T> {
  /** Data to export */
  data: T[];
  /** Filename without extension */
  filename: string;
  /** Export format */
  format: ExportFormat;
  /** Column definitions for CSV (order and headers) */
  columns?: {
    key: keyof T | string;
    header: string;
    /** Transform function for the value */
    transform?: (value: unknown, row: T) => string;
  }[];
}

/**
 * Escape a value for CSV format
 */
function escapeCSV(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }

  const str = typeof value === 'object' ? JSON.stringify(value) : String(value);

  // If the value contains comma, newline, or quote, wrap in quotes and escape internal quotes
  if (str.includes(',') || str.includes('\n') || str.includes('"')) {
    return `"${str.replace(/"/g, '""')}"`;
  }

  return str;
}

/**
 * Get nested property from object using dot notation
 */
function getNestedValue(obj: Record<string, unknown> | null | undefined, path: string): unknown {
  return path.split('.').reduce<unknown>((current, key) => {
    if (current && typeof current === 'object') {
      return (current as Record<string, unknown>)[key];
    }
    return undefined;
  }, obj);
}

function toIsoDateString(value: unknown, fallback = ''): string {
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (typeof value === 'string' || typeof value === 'number') {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? fallback : date.toISOString();
  }
  return fallback;
}

/**
 * Convert data array to CSV string
 */
function toCSV<T extends Record<string, unknown>>(data: T[], columns: ExportOptions<T>['columns']): string {
  if (!columns || columns.length === 0) {
    // Auto-generate columns from first item
    if (data.length === 0) return '';
    const keys = Object.keys(data[0] as object);
    columns = keys.map(key => ({ key, header: key }));
  }

  const headers = columns.map(col => escapeCSV(col.header));
  const rows = data.map(row =>
    columns!.map(col => {
      const value = getNestedValue(row, col.key as string);
      const transformed = col.transform ? col.transform(value, row) : value;
      return escapeCSV(transformed);
    }).join(',')
  );

  return [headers.join(','), ...rows].join('\n');
}

/**
 * Convert data array to formatted JSON string
 */
function toJSON<T>(data: T[]): string {
  return JSON.stringify(data, null, 2);
}

/**
 * Trigger file download in browser
 */
function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

/**
 * Export data to CSV or JSON file
 */
export function exportData<T extends Record<string, unknown>>(options: ExportOptions<T>): void {
  const { data, filename, format, columns } = options;
  const timestamp = new Date().toISOString().split('T')[0];

  if (format === 'csv') {
    const content = toCSV(data, columns);
    downloadFile(content, `${filename}-${timestamp}.csv`, 'text/csv;charset=utf-8');
  } else {
    const content = toJSON(data);
    downloadFile(content, `${filename}-${timestamp}.json`, 'application/json');
  }
}

/**
 * Pre-configured export for audit logs
 */
export function exportAuditLogs<T extends object>(logs: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: logs as unknown as Record<string, unknown>[],
    filename: 'audit-logs',
    format,
    columns: [
      { key: 'timestamp', header: 'Timestamp', transform: (v) => toIsoDateString(v) },
      { key: 'user_id', header: 'User ID' },
      { key: 'action', header: 'Action' },
      { key: 'resource', header: 'Resource' },
      { key: 'resource_id', header: 'Resource ID' },
      { key: 'ip_address', header: 'IP Address' },
      { key: 'details', header: 'Details', transform: (v) => v ? JSON.stringify(v) : '' },
    ],
  });
}

/**
 * Pre-configured export for users
 */
export function exportUsers<T extends object>(users: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: users as Record<string, unknown>[],
    filename: 'users',
    format,
    columns: [
      { key: 'id', header: 'ID' },
      { key: 'username', header: 'Username' },
      { key: 'email', header: 'Email' },
      { key: 'role', header: 'Role' },
      { key: 'is_active', header: 'Active', transform: (v) => v ? 'Yes' : 'No' },
      { key: 'is_verified', header: 'Verified', transform: (v) => v ? 'Yes' : 'No' },
      { key: 'storage_used_mb', header: 'Storage Used (MB)' },
      { key: 'storage_quota_mb', header: 'Storage Quota (MB)' },
      { key: 'created_at', header: 'Created At', transform: (v) => toIsoDateString(v) },
      { key: 'last_login', header: 'Last Login', transform: (v) => toIsoDateString(v) },
    ],
  });
}

/**
 * Pre-configured export for API keys
 */
export function exportApiKeys<T extends object>(keys: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: keys as Record<string, unknown>[],
    filename: 'api-keys',
    format,
    columns: [
      { key: 'id', header: 'ID' },
      { key: 'name', header: 'Name' },
      { key: 'prefix', header: 'Key Prefix' },
      { key: 'user_id', header: 'User ID' },
      { key: 'is_active', header: 'Active', transform: (v) => v ? 'Yes' : 'No' },
      { key: 'created_at', header: 'Created At', transform: (v) => toIsoDateString(v) },
      { key: 'expires_at', header: 'Expires At', transform: (v) => toIsoDateString(v, 'Never') },
      { key: 'last_used_at', header: 'Last Used', transform: (v) => toIsoDateString(v, 'Never') },
    ],
  });
}

type UnifiedApiKeyExportRow = {
  keyId: string;
  keyPrefix: string;
  ownerUserId: number;
  ownerUsername: string;
  ownerEmail: string;
  createdAt: string | null;
  expiresAt: string | null;
  lastUsedAt: string | null;
  status: string;
  totalTokens: number | null;
  estimatedCostUsd: number | null;
};

/**
 * Pre-configured export for unified admin API key rows.
 */
export function exportUnifiedApiKeys<T extends UnifiedApiKeyExportRow>(rows: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: rows as unknown as Record<string, unknown>[],
    filename: 'api-keys',
    format,
    columns: [
      { key: 'keyId', header: 'Key ID' },
      { key: 'keyPrefix', header: 'Key Prefix' },
      { key: 'ownerUserId', header: 'User ID' },
      { key: 'ownerUsername', header: 'Owner Username' },
      { key: 'ownerEmail', header: 'Owner Email' },
      { key: 'status', header: 'Status' },
      { key: 'createdAt', header: 'Created At', transform: (v) => toIsoDateString(v) },
      { key: 'expiresAt', header: 'Expires At', transform: (v) => toIsoDateString(v, 'Never') },
      { key: 'lastUsedAt', header: 'Last Used', transform: (v) => toIsoDateString(v, 'Never') },
      { key: 'totalTokens', header: 'Total Tokens' },
      {
        key: 'estimatedCostUsd',
        header: 'Estimated Cost (USD)',
        transform: (v) => typeof v === 'number' ? v.toFixed(v < 0.01 ? 4 : 2) : '',
      },
    ],
  });
}

/**
 * Pre-configured export for organizations
 */
export function exportOrganizations<T extends object>(orgs: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: orgs as Record<string, unknown>[],
    filename: 'organizations',
    format,
    columns: [
      { key: 'id', header: 'ID' },
      { key: 'name', header: 'Name' },
      { key: 'slug', header: 'Slug' },
      { key: 'created_at', header: 'Created At', transform: (v) => toIsoDateString(v) },
    ],
  });
}

/**
 * Pre-configured export for incidents
 */
export function exportIncidents<T extends object>(incidents: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: incidents as Record<string, unknown>[],
    filename: 'incidents',
    format,
    columns: [
      { key: 'id', header: 'ID' },
      { key: 'title', header: 'Title' },
      { key: 'status', header: 'Status' },
      { key: 'severity', header: 'Severity' },
      { key: 'summary', header: 'Summary' },
      { key: 'tags', header: 'Tags', transform: (v) => Array.isArray(v) ? v.join(', ') : '' },
      { key: 'assigned_to_user_id', header: 'Assigned To' },
      { key: 'created_at', header: 'Created At', transform: (v) => toIsoDateString(v) },
      { key: 'resolved_at', header: 'Resolved At', transform: (v) => toIsoDateString(v, '') },
      { key: 'mtta_minutes', header: 'MTTA (min)' },
      { key: 'mttr_minutes', header: 'MTTR (min)' },
    ],
  });
}

/**
 * Pre-configured export for jobs
 */
export function exportJobs<T extends object>(jobs: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: jobs as Record<string, unknown>[],
    filename: 'jobs',
    format,
    columns: [
      { key: 'id', header: 'ID' },
      { key: 'uuid', header: 'UUID' },
      { key: 'domain', header: 'Domain' },
      { key: 'queue', header: 'Queue' },
      { key: 'job_type', header: 'Job Type' },
      { key: 'status', header: 'Status' },
      { key: 'priority', header: 'Priority' },
      { key: 'retry_count', header: 'Retries' },
      { key: 'created_at', header: 'Created At', transform: (v) => toIsoDateString(v) },
      { key: 'started_at', header: 'Started At', transform: (v) => toIsoDateString(v) },
      { key: 'completed_at', header: 'Completed At', transform: (v) => toIsoDateString(v) },
    ],
  });
}

/**
 * Pre-configured export for subscriptions
 */
export function exportSubscriptions<T extends object>(subs: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: subs as Record<string, unknown>[],
    filename: 'subscriptions',
    format,
    columns: [
      { key: 'id', header: 'ID' },
      { key: 'org_id', header: 'Org ID' },
      { key: 'plan_id', header: 'Plan ID' },
      { key: 'status', header: 'Status' },
      { key: 'current_period_start', header: 'Period Start', transform: (v) => toIsoDateString(v) },
      { key: 'current_period_end', header: 'Period End', transform: (v) => toIsoDateString(v) },
      { key: 'created_at', header: 'Created At', transform: (v) => toIsoDateString(v) },
    ],
  });
}

/**
 * Pre-configured export for teams
 */
export function exportTeams<T extends object>(teams: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: teams as Record<string, unknown>[],
    filename: 'teams',
    format,
    columns: [
      { key: 'id', header: 'ID' },
      { key: 'name', header: 'Name' },
      { key: 'description', header: 'Description' },
      { key: 'created_at', header: 'Created At', transform: (v) => toIsoDateString(v) },
    ],
  });
}

/**
 * Pre-configured export for budgets
 */
export function exportBudgets<T extends object>(budgets: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: budgets as Record<string, unknown>[],
    filename: 'budgets',
    format,
    columns: [
      { key: 'org_id', header: 'Org ID' },
      { key: 'org_name', header: 'Org Name' },
      { key: 'plan_name', header: 'Plan' },
      { key: 'budgets.budget_day_usd', header: 'Daily USD Cap' },
      { key: 'budgets.budget_month_usd', header: 'Monthly USD Cap' },
      { key: 'budgets.budget_day_tokens', header: 'Daily Token Cap' },
      { key: 'budgets.budget_month_tokens', header: 'Monthly Token Cap' },
      { key: 'updated_at', header: 'Updated At', transform: (v) => toIsoDateString(v) },
    ],
  });
}

/**
 * Pre-configured export for system logs
 */
export function exportLogs<T extends object>(logs: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: logs as Record<string, unknown>[],
    filename: 'system-logs',
    format,
    columns: [
      { key: 'timestamp', header: 'Timestamp', transform: (v) => toIsoDateString(v) },
      { key: 'level', header: 'Level' },
      { key: 'message', header: 'Message' },
      { key: 'logger', header: 'Logger' },
      { key: 'module', header: 'Module' },
      { key: 'request_id', header: 'Request ID' },
      { key: 'org_id', header: 'Org ID' },
      { key: 'user_id', header: 'User ID' },
    ],
  });
}

/**
 * Pre-configured export for voice commands
 */
export function exportVoiceCommands<T extends object>(commands: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: commands as Record<string, unknown>[],
    filename: 'voice-commands',
    format,
    columns: [
      { key: 'id', header: 'ID' },
      { key: 'name', header: 'Name' },
      { key: 'phrases', header: 'Phrases', transform: (v) => Array.isArray(v) ? v.join(', ') : '' },
      { key: 'action_type', header: 'Action Type' },
      { key: 'description', header: 'Description' },
      { key: 'priority', header: 'Priority' },
      { key: 'enabled', header: 'Enabled', transform: (v) => v ? 'Yes' : 'No' },
      { key: 'requires_confirmation', header: 'Requires Confirmation', transform: (v) => v ? 'Yes' : 'No' },
    ],
  });
}

/**
 * Pre-configured export for ACP sessions
 */
export function exportACPSessions<T extends object>(sessions: T[], format: ExportFormat = 'csv'): void {
  exportData({
    data: sessions as Record<string, unknown>[],
    filename: 'acp-sessions',
    format,
    columns: [
      { key: 'session_id', header: 'Session ID' },
      { key: 'user_id', header: 'User ID' },
      { key: 'agent_type', header: 'Agent Type' },
      { key: 'name', header: 'Name' },
      { key: 'status', header: 'Status' },
      { key: 'message_count', header: 'Messages' },
      { key: 'usage.total_tokens', header: 'Total Tokens' },
      { key: 'estimated_cost_usd', header: 'Est. Cost (USD)' },
      { key: 'model', header: 'Model' },
      { key: 'has_websocket', header: 'WebSocket', transform: (v) => v ? 'Yes' : 'No' },
      { key: 'created_at', header: 'Created At', transform: (v) => toIsoDateString(v) },
      { key: 'last_activity_at', header: 'Last Activity', transform: (v) => toIsoDateString(v) },
    ],
  });
}
