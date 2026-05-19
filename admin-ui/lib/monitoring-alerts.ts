import type {
  AlertAssignableUser,
  AlertHistoryAction,
  AlertHistoryEntry,
  AlertRule,
  AlertRuleDraft,
  AlertRuleDurationMinutes,
  AlertRuleMetric,
  AlertRuleOperator,
  AlertRuleValidationErrors,
  AlertSeverity,
  SnoozeDurationOption,
  SystemAlert,
} from '@/app/monitoring/types';
import { getScopedItem, setScopedItem } from '@/lib/scoped-storage';

type AlertRuleValidationResult = {
  valid: boolean;
  errors: AlertRuleValidationErrors;
};

const ALERT_RULES_STORAGE_KEY = 'admin.monitoring.alert-rules.v1';
const ALERT_HISTORY_STORAGE_KEY = 'admin.monitoring.alert-history.v1';

const PERCENT_METRICS = new Set<AlertRuleMetric>(['cpu', 'memory', 'diskUsage']);
const ALLOWED_OPERATORS: AlertRuleOperator[] = ['>', '<', '=='];
const ALLOWED_SEVERITIES: AlertSeverity[] = ['info', 'warning', 'error', 'critical'];
const ALLOWED_RULE_DURATIONS: AlertRuleDurationMinutes[] = [1, 5, 10, 15, 30, 60, 240, 1440];

const SNOOZE_MINUTES: Record<SnoozeDurationOption, number> = {
  '15m': 15,
  '1h': 60,
  '4h': 240,
  '24h': 1440,
};

export const ALERT_RULE_METRIC_OPTIONS: Array<{ value: AlertRuleMetric; label: string }> = [
  { value: 'cpu', label: 'CPU' },
  { value: 'memory', label: 'Memory' },
  { value: 'diskUsage', label: 'Disk Usage' },
  { value: 'throughput', label: 'Request Throughput' },
  { value: 'activeConnections', label: 'Active Connections' },
  { value: 'queueDepth', label: 'Queue Depth' },
];

export const ALERT_RULE_DURATION_OPTIONS: Array<{ value: AlertRuleDurationMinutes; label: string }> = [
  { value: 1, label: '1 minute' },
  { value: 5, label: '5 minutes' },
  { value: 10, label: '10 minutes' },
  { value: 15, label: '15 minutes' },
  { value: 30, label: '30 minutes' },
  { value: 60, label: '1 hour' },
  { value: 240, label: '4 hours' },
  { value: 1440, label: '24 hours' },
];

export const ALERT_RULE_OPERATOR_OPTIONS: AlertRuleOperator[] = ['>', '<', '=='];
export const ALERT_SEVERITY_OPTIONS: AlertSeverity[] = ['warning', 'critical', 'error', 'info'];
export const ALERT_SNOOZE_OPTIONS: Array<{ value: SnoozeDurationOption; label: string }> = [
  { value: '15m', label: '15m' },
  { value: '1h', label: '1h' },
  { value: '4h', label: '4h' },
  { value: '24h', label: '24h' },
];

export const DEFAULT_ALERT_RULE_DRAFT: AlertRuleDraft = {
  metric: 'cpu',
  operator: '>',
  threshold: '85',
  durationMinutes: '5',
  severity: 'warning',
};

const toObject = (value: unknown): Record<string, unknown> | null =>
  typeof value === 'object' && value !== null ? (value as Record<string, unknown>) : null;

const toFiniteNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value)
    ? value
    : (typeof value === 'string' && value.trim() && Number.isFinite(Number(value))
      ? Number(value)
      : null);

const coerceMetric = (value: unknown): AlertRuleMetric | null => {
  if (typeof value !== 'string') return null;
  return ALERT_RULE_METRIC_OPTIONS.some((option) => option.value === value)
    ? (value as AlertRuleMetric)
    : null;
};

const coerceOperator = (value: unknown): AlertRuleOperator | null => {
  if (typeof value !== 'string') return null;
  return ALLOWED_OPERATORS.includes(value as AlertRuleOperator)
    ? (value as AlertRuleOperator)
    : null;
};

const coerceSeverity = (value: unknown, fallback: AlertSeverity = 'warning'): AlertSeverity => {
  if (typeof value !== 'string') return fallback;
  return ALLOWED_SEVERITIES.includes(value as AlertSeverity)
    ? (value as AlertSeverity)
    : fallback;
};

const coerceString = (value: unknown): string => (typeof value === 'string' ? value : '');

const coerceRuleDuration = (value: unknown): AlertRuleDurationMinutes | null => {
  const parsed = toFiniteNumber(value);
  if (parsed === null) return null;
  const duration = Math.round(parsed);
  return ALLOWED_RULE_DURATIONS.includes(duration as AlertRuleDurationMinutes)
    ? (duration as AlertRuleDurationMinutes)
    : null;
};

const parseRulesArray = (raw: unknown): AlertRule[] => {
  if (!Array.isArray(raw)) return [];
  return raw.map((entry): AlertRule | null => {
    const obj = toObject(entry);
    if (!obj) return null;
    const metric = coerceMetric(obj.metric);
    const operator = coerceOperator(obj.operator);
    const threshold = toFiniteNumber(obj.threshold);
    const durationMinutes = coerceRuleDuration(obj.durationMinutes);
    if (!metric || !operator || threshold === null || !durationMinutes) return null;
    const id = coerceString(obj.id) || `rule-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const createdAtRaw = coerceString(obj.createdAt);
    const createdAt = createdAtRaw && !Number.isNaN(Date.parse(createdAtRaw))
      ? new Date(createdAtRaw).toISOString()
      : new Date().toISOString();
    return {
      id,
      metric,
      operator,
      threshold,
      durationMinutes,
      severity: coerceSeverity(obj.severity, 'warning'),
      createdAt,
    };
  }).filter((entry): entry is AlertRule => entry !== null);
};

const parseHistoryArray = (raw: unknown): AlertHistoryEntry[] => {
  if (!Array.isArray(raw)) return [];
  return raw.map((entry): AlertHistoryEntry | null => {
    const obj = toObject(entry);
    if (!obj) return null;
    const id = coerceString(obj.id);
    const alertId = coerceString(obj.alertId);
    const timestampRaw = coerceString(obj.timestamp);
    const actionRaw = coerceString(obj.action);
    if (!id || !alertId || !timestampRaw || !actionRaw) return null;
    const timestamp = !Number.isNaN(Date.parse(timestampRaw))
      ? new Date(timestampRaw).toISOString()
      : new Date().toISOString();
    const action = actionRaw as AlertHistoryAction;
    return {
      id,
      alertId,
      timestamp,
      action,
      actor: coerceString(obj.actor) || undefined,
      details: coerceString(obj.details) || undefined,
    };
  }).filter((entry): entry is AlertHistoryEntry => entry !== null);
};

const getBrowserStorage = (): Storage | null => {
  if (typeof window === 'undefined') return null;
  return window.localStorage;
};

const storageRead = (key: string, fallback: unknown, storage?: Storage | null): unknown => {
  try {
    // When an explicit Storage handle is provided (e.g. tests), use it directly.
    // Otherwise delegate to the scoped-storage helper for tenant isolation.
    const raw = storage ? storage.getItem(key) : getScopedItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
};

const storageWrite = (key: string, value: unknown, storage?: Storage | null): void => {
  try {
    if (storage) {
      storage.setItem(key, JSON.stringify(value));
    } else {
      setScopedItem(key, JSON.stringify(value));
    }
  } catch {
    // no-op when storage is unavailable or quota is exceeded
  }
};

export const validateAlertRuleDraft = (draft: AlertRuleDraft): AlertRuleValidationResult => {
  const errors: AlertRuleValidationErrors = {};

  if (!coerceMetric(draft.metric)) {
    errors.metric = 'Select a metric.';
  }

  if (!coerceOperator(draft.operator)) {
    errors.operator = 'Select an operator.';
  }

  const threshold = toFiniteNumber(draft.threshold);
  if (threshold === null) {
    errors.threshold = 'Threshold must be a number.';
  } else if (PERCENT_METRICS.has(draft.metric) && (threshold < 0 || threshold > 100)) {
    errors.threshold = 'Threshold for utilization metrics must be between 0 and 100.';
  } else if (!PERCENT_METRICS.has(draft.metric) && threshold < 0) {
    errors.threshold = 'Threshold must be zero or greater.';
  }

  const duration = coerceRuleDuration(draft.durationMinutes);
  if (!duration) {
    errors.durationMinutes = 'Select a valid duration.';
  }

  if (!ALLOWED_SEVERITIES.includes(draft.severity)) {
    errors.severity = 'Select a severity.';
  }

  return {
    valid: Object.keys(errors).length === 0,
    errors,
  };
};

export const buildAlertRuleFromDraft = (
  draft: AlertRuleDraft,
  now: Date = new Date()
): AlertRule => ({
  id: `rule-${now.getTime()}-${Math.random().toString(36).slice(2, 8)}`,
  metric: draft.metric,
  operator: draft.operator,
  threshold: Number(draft.threshold),
  durationMinutes: Number(draft.durationMinutes) as AlertRuleDurationMinutes,
  severity: draft.severity,
  createdAt: now.toISOString(),
});

export const readStoredAlertRules = (storage?: Storage | null): AlertRule[] =>
  parseRulesArray(storageRead(ALERT_RULES_STORAGE_KEY, [], storage));

export const writeStoredAlertRules = (rules: AlertRule[], storage?: Storage | null): void => {
  storageWrite(ALERT_RULES_STORAGE_KEY, rules, storage);
};

export const readStoredAlertHistory = (storage?: Storage | null): AlertHistoryEntry[] =>
  parseHistoryArray(storageRead(ALERT_HISTORY_STORAGE_KEY, [], storage));

export const writeStoredAlertHistory = (history: AlertHistoryEntry[], storage?: Storage | null): void => {
  storageWrite(ALERT_HISTORY_STORAGE_KEY, history, storage);
};

export const formatSnoozeCountdown = (snoozedUntil: string, now: Date = new Date()): string => {
  const end = new Date(snoozedUntil);
  if (Number.isNaN(end.valueOf())) {
    return 'Invalid';
  }
  const diffMs = end.getTime() - now.getTime();
  if (diffMs <= 0) {
    return 'Expired';
  }
  const remainingMinutes = Math.ceil(diffMs / (60 * 1000));
  if (remainingMinutes < 60) {
    return `${remainingMinutes}m remaining`;
  }
  const hours = Math.floor(remainingMinutes / 60);
  const minutes = remainingMinutes % 60;
  if (minutes === 0) {
    return `${hours}h remaining`;
  }
  return `${hours}h ${minutes}m remaining`;
};

const formatAdminAlertHistoryDetails = (
  action: AlertHistoryAction,
  details: Record<string, unknown> | null
): string => {
  switch (action) {
    case 'assigned': {
      const assignee = toFiniteNumber(details?.assigned_to_user_id);
      return assignee !== null ? `Assigned to user ${assignee}` : 'Alert assigned';
    }
    case 'unassigned':
      return 'Alert unassigned';
    case 'snoozed': {
      const snoozedUntil = coerceString(details?.snoozed_until);
      return snoozedUntil ? `Snoozed until ${snoozedUntil}` : 'Alert snoozed';
    }
    case 'escalated': {
      const severity = coerceString(details?.severity) || 'critical';
      return `Severity escalated to ${severity}`;
    }
    case 'acknowledged':
      return 'Alert acknowledged';
    case 'dismissed':
      return 'Alert dismissed';
    case 'triggered':
      return coerceString(details?.message) || 'Alert triggered';
    default:
      return 'Alert updated';
  }
};

export const resolveSnoozedUntil = (
  option: SnoozeDurationOption,
  now: Date = new Date()
): string => {
  const minutes = SNOOZE_MINUTES[option];
  return new Date(now.getTime() + (minutes * 60 * 1000)).toISOString();
};

export const isAlertSnoozed = (alert: SystemAlert, now: Date = new Date()): boolean => {
  if (!alert.snoozed_until) return false;
  const snoozedUntil = new Date(alert.snoozed_until);
  if (Number.isNaN(snoozedUntil.valueOf())) return false;
  return snoozedUntil.getTime() > now.getTime();
};

export const normalizeMonitoringAlert = (value: unknown): SystemAlert | null => {
  const obj = toObject(value);
  if (!obj) return null;

  const idRaw = obj.id ?? obj.alert_id;
  if (idRaw === undefined || idRaw === null) return null;
  const id = String(idRaw);
  const alertIdentity = coerceString(obj.alert_identity) || `alert:${id}`;

  const severity = coerceSeverity(
    obj.escalated_severity ?? obj.severity ?? obj.rule_severity,
    'warning'
  );
  const message =
    coerceString(obj.message) ||
    coerceString(obj.text_snippet) ||
    coerceString(obj.pattern) ||
    'Alert';
  const source = coerceString(obj.source) || undefined;
  const timestampRaw = coerceString(obj.timestamp) || coerceString(obj.created_at);
  const timestamp = !Number.isNaN(Date.parse(timestampRaw))
    ? new Date(timestampRaw).toISOString()
    : new Date().toISOString();
  const acknowledged = Boolean(obj.acknowledged ?? obj.is_read ?? obj.read_at);
  const acknowledgedAtRaw = coerceString(obj.acknowledged_at) || coerceString(obj.read_at);
  const acknowledgedAt = acknowledgedAtRaw && !Number.isNaN(Date.parse(acknowledgedAtRaw))
    ? new Date(acknowledgedAtRaw).toISOString()
    : undefined;
  const dismissedAtRaw = coerceString(obj.dismissed_at);
  const dismissedAt = dismissedAtRaw && !Number.isNaN(Date.parse(dismissedAtRaw))
    ? new Date(dismissedAtRaw).toISOString()
    : undefined;
  const acknowledgedBy = coerceString(obj.acknowledged_by) || undefined;
  const assignedTo =
    coerceString(obj.assigned_to)
    || (toFiniteNumber(obj.assigned_to_user_id) !== null
      ? String(toFiniteNumber(obj.assigned_to_user_id))
      : undefined);
  const snoozedUntilRaw = coerceString(obj.snoozed_until);
  const snoozedUntil = snoozedUntilRaw && !Number.isNaN(Date.parse(snoozedUntilRaw))
    ? new Date(snoozedUntilRaw).toISOString()
    : undefined;
  const metadataRaw = toObject(obj.metadata);
  const metadata = metadataRaw ?? undefined;

  return {
    id,
    alert_identity: alertIdentity,
    severity,
    message,
    source,
    timestamp,
    acknowledged,
    acknowledged_at: acknowledgedAt,
    dismissed_at: dismissedAt,
    acknowledged_by: acknowledgedBy,
    assigned_to: assignedTo,
    snoozed_until: snoozedUntil,
    escalated_severity: coerceSeverity(obj.escalated_severity, severity),
    metadata,
  };
};

export const normalizeMonitoringAlertsPayload = (payload: unknown): SystemAlert[] => {
  if (Array.isArray(payload)) {
    return payload
      .map((entry) => normalizeMonitoringAlert(entry))
      .filter((entry): entry is SystemAlert => entry !== null);
  }
  const payloadObj = toObject(payload);
  if (!payloadObj) return [];
  const items = payloadObj.items ?? payloadObj.alerts;
  if (!Array.isArray(items)) return [];
  return items
    .map((entry) => normalizeMonitoringAlert(entry))
    .filter((entry): entry is SystemAlert => entry !== null);
};

export const normalizeAdminAlertHistoryPayload = (payload: unknown): AlertHistoryEntry[] => {
  const payloadObj = toObject(payload);
  const items = Array.isArray(payload)
    ? payload
    : Array.isArray(payloadObj?.items)
      ? payloadObj.items
      : [];

  return sortAlertHistoryEntries(
    items
      .map((entry): AlertHistoryEntry | null => {
        const obj = toObject(entry);
        if (!obj) return null;
        const idValue = obj.id;
        const actionValue = coerceString(obj.action) as AlertHistoryAction;
        const alertId = coerceString(obj.alert_identity);
        const timestampValue = coerceString(obj.created_at);
        if ((idValue === undefined || idValue === null) || !actionValue || !alertId || !timestampValue) {
          return null;
        }
        const details = toObject(obj.details);
        const actorId = toFiniteNumber(obj.actor_user_id);
        return {
          id: String(idValue),
          alertId,
          action: actionValue,
          actor: actorId !== null ? `User ${actorId}` : undefined,
          details: formatAdminAlertHistoryDetails(actionValue, details),
          timestamp: !Number.isNaN(Date.parse(timestampValue))
            ? new Date(timestampValue).toISOString()
            : new Date().toISOString(),
        };
      })
      .filter((entry): entry is AlertHistoryEntry => entry !== null)
  );
};

export const buildAlertHistoryEntry = (
  alertId: string,
  action: AlertHistoryAction,
  details: string,
  options?: { actor?: string; timestamp?: string }
): AlertHistoryEntry => {
  const timestamp = options?.timestamp && !Number.isNaN(Date.parse(options.timestamp))
    ? new Date(options.timestamp).toISOString()
    : new Date().toISOString();
  return {
    id: `history-${alertId}-${timestamp}-${Math.random().toString(36).slice(2, 8)}`,
    alertId,
    action,
    details,
    actor: options?.actor,
    timestamp,
  };
};

export const ensureTriggeredHistoryEntries = (
  history: AlertHistoryEntry[],
  alerts: SystemAlert[]
): AlertHistoryEntry[] => {
  const existing = new Set(
    history
      .filter((entry) => entry.action === 'triggered')
      .map((entry) => `${entry.alertId}:${entry.action}`)
  );
  const additions = alerts
    .filter((alert) => !existing.has(`${alert.id}:triggered`))
    .map((alert) =>
      buildAlertHistoryEntry(
        alert.id,
        'triggered',
        alert.message,
        { timestamp: alert.timestamp }
      )
    );
  return sortAlertHistoryEntries([...history, ...additions]);
};

export const sortAlertHistoryEntries = (history: AlertHistoryEntry[]): AlertHistoryEntry[] =>
  [...history].sort(
    (a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)
  );

export const buildAssignableUsers = (payload: unknown): AlertAssignableUser[] => {
  if (!Array.isArray(payload)) return [];
  const mapped = payload
    .map((entry): AlertAssignableUser | null => {
      const obj = toObject(entry);
      if (!obj) return null;
      const idValue = obj.id ?? obj.user_id;
      if (idValue === undefined || idValue === null) return null;
      const id = String(idValue);
      const label =
        coerceString(obj.username) ||
        coerceString(obj.email) ||
        coerceString(obj.name) ||
        `User ${id}`;
      return { id, label };
    })
    .filter((entry): entry is AlertAssignableUser => entry !== null);

  const deduped = new Map<string, AlertAssignableUser>();
  mapped.forEach((entry) => {
    if (!deduped.has(entry.id)) {
      deduped.set(entry.id, entry);
    }
  });
  return Array.from(deduped.values());
};

export const formatAlertHistoryActionLabel = (action: AlertHistoryAction): string => {
  switch (action) {
    case 'triggered':
      return 'Triggered';
    case 'acknowledged':
      return 'Acknowledged';
    case 'dismissed':
      return 'Dismissed';
    case 'assigned':
      return 'Assigned';
    case 'unassigned':
      return 'Unassigned';
    case 'snoozed':
      return 'Snoozed';
    case 'escalated':
      return 'Escalated';
    default:
      return action;
  }
};
