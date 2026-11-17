import { describe, expect, it } from 'vitest';
import {
  DEFAULT_ALERT_RULE_DRAFT,
  ensureTriggeredHistoryEntries,
  formatSnoozeCountdown,
  normalizeMonitoringAlert,
  normalizeAdminAlertHistoryPayload,
  validateAlertRuleDraft,
} from './monitoring-alerts';

describe('validateAlertRuleDraft', () => {
  it('rejects invalid threshold and duration values', () => {
    const result = validateAlertRuleDraft({
      ...DEFAULT_ALERT_RULE_DRAFT,
      threshold: '120',
      durationMinutes: '2',
    });
    expect(result.valid).toBe(false);
    expect(result.errors.threshold).toBe(
      'Threshold for utilization metrics must be between 0 and 100.'
    );
    expect(result.errors.durationMinutes).toBe('Select a valid duration.');
  });

  it('accepts valid rule drafts', () => {
    const result = validateAlertRuleDraft({
      ...DEFAULT_ALERT_RULE_DRAFT,
      metric: 'throughput',
      threshold: '150',
      durationMinutes: '15',
      severity: 'critical',
    });
    expect(result.valid).toBe(true);
    expect(result.errors).toEqual({});
  });
});

describe('formatSnoozeCountdown', () => {
  it('formats minute-based countdowns', () => {
    const now = new Date('2026-02-17T12:00:00.000Z');
    expect(formatSnoozeCountdown('2026-02-17T12:15:00.000Z', now)).toBe('15m remaining');
  });

  it('formats hour+minute countdowns', () => {
    const now = new Date('2026-02-17T12:00:00.000Z');
    expect(formatSnoozeCountdown('2026-02-17T13:30:00.000Z', now)).toBe('1h 30m remaining');
  });

  it('returns expired for completed snoozes', () => {
    const now = new Date('2026-02-17T12:00:00.000Z');
    expect(formatSnoozeCountdown('2026-02-17T11:59:00.000Z', now)).toBe('Expired');
  });
});

describe('ensureTriggeredHistoryEntries', () => {
  it('adds one triggered event per alert id if missing', () => {
    const history = ensureTriggeredHistoryEntries([], [
      {
        id: 'alert-1',
        severity: 'warning',
        message: 'CPU high',
        timestamp: '2026-02-17T10:00:00.000Z',
        acknowledged: false,
      },
      {
        id: 'alert-2',
        severity: 'critical',
        message: 'Queue depth rising',
        timestamp: '2026-02-17T10:05:00.000Z',
        acknowledged: false,
      },
    ]);
    expect(history).toHaveLength(2);
    expect(history.every((entry) => entry.action === 'triggered')).toBe(true);
  });
});

describe('normalizeAdminAlertHistoryPayload', () => {
  it('renders unassigned history entries truthfully', () => {
    const history = normalizeAdminAlertHistoryPayload({
      items: [
        {
          id: 1,
          alert_identity: 'alert:1',
          action: 'unassigned',
          actor_user_id: 7,
          details: { assigned_to_user_id: null },
          created_at: '2026-02-17T12:00:00.000Z',
        },
      ],
    });

    expect(history).toEqual([
      {
        id: '1',
        alertId: 'alert:1',
        action: 'unassigned',
        actor: 'User 7',
        details: 'Alert unassigned',
        timestamp: '2026-02-17T12:00:00.000Z',
      },
    ]);
  });
});

describe('normalizeMonitoringAlert', () => {
  it('preserves numeric assigned_to_user_id values for alert assignee rendering', () => {
    const alert = normalizeMonitoringAlert({
      id: 1,
      alert_identity: 'alert:1',
      severity: 'warning',
      text_snippet: 'CPU high',
      source: 'system',
      created_at: '2026-03-12T06:54:08.142506+00:00',
      assigned_to_user_id: 1,
    });

    expect(alert).toMatchObject({
      id: '1',
      alert_identity: 'alert:1',
      assigned_to: '1',
    });
  });
});

describe('acknowledgment state', () => {
  it.each([
    [{ is_read: true, read_at: '2026-02-17T10:00:00Z' }, undefined],
    [{ acknowledged: true, acknowledged_at: 'invalid' }, undefined],
    [{ is_read: false, acknowledged_at: '2026-02-17T10:00:00Z' }, '2026-02-17T10:00:00.000Z'],
  ] as const)('reflects the recorded acknowledgment time', (fields, timestamp) => {
    expect(normalizeMonitoringAlert({ id: 8, ...fields })).toMatchObject({
      acknowledged: Boolean(timestamp),
      acknowledged_at: timestamp,
    });
  });
});
