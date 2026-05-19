import { useCallback } from 'react';
import { resolveSnoozedUntil } from '@/lib/monitoring-alerts';
import {
  escalateAlertSeverity,
  markAlertAcknowledged,
  removeAlertById,
  setAlertAssignment,
  setAlertSnoozeUntil,
} from './alert-state-utils';
import { logger } from '@/lib/logger';
import type { SnoozeDurationOption, SystemAlert } from './types';

type ConfirmVariant = 'danger' | 'warning' | 'default';
type ConfirmIcon = 'delete' | 'warning' | 'rotate' | 'remove-user' | 'key';

type ConfirmOptions = {
  title: string;
  message: string;
  confirmText?: string;
  variant?: ConfirmVariant;
  icon?: ConfirmIcon;
};

type ConfirmFn = (options: ConfirmOptions) => Promise<boolean>;

type StateSetter<T> = (next: T | ((prev: T) => T)) => void;

export type AlertActionsApiClient = {
  acknowledgeAlert: (alertId: string) => Promise<unknown>;
  dismissAlert: (alertId: string) => Promise<unknown>;
  assignAdminAlert: (alertIdentity: string, data: Record<string, unknown>) => Promise<unknown>;
  snoozeAdminAlert: (alertIdentity: string, data: Record<string, unknown>) => Promise<unknown>;
  escalateAdminAlert: (alertIdentity: string, data: Record<string, unknown>) => Promise<unknown>;
};

type UseAlertActionsArgs = {
  apiClient: AlertActionsApiClient;
  confirm: ConfirmFn;
  setAlerts: StateSetter<SystemAlert[]>;
  setError: (message: string) => void;
  setSuccess: (message: string) => void;
  onReloadRequested: () => void | Promise<void>;
};

const getAlertIdentity = (alert: SystemAlert): string => alert.alert_identity ?? `alert:${alert.id}`;

export const useAlertActions = ({
  apiClient,
  confirm,
  setAlerts,
  setError,
  setSuccess,
  onReloadRequested,
}: UseAlertActionsArgs) => {
  const handleAcknowledgeAlert = useCallback(async (alert: SystemAlert) => {
    setError('');
    // Optimistic: update UI immediately
    const acknowledgedAt = new Date().toISOString();
    setAlerts((prev) => markAlertAcknowledged(prev, alert.id, acknowledgedAt));
    try {
      await apiClient.acknowledgeAlert(alert.id);
      setSuccess('Alert acknowledged');
    } catch (err: unknown) {
      // Rollback on failure
      setAlerts((prev) =>
        prev.map((a) =>
          a.id === alert.id ? { ...a, acknowledged: false, acknowledged_at: undefined } : a,
        ),
      );
      logger.error('Failed to acknowledge alert', { component: 'useAlertActions', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to acknowledge alert');
      return;
    }

    try {
      await Promise.resolve(onReloadRequested());
    } catch (err: unknown) {
      console.error('Failed to reload alerts after acknowledgement:', err);
      setError('Alert acknowledged, but refresh failed');
    }
  }, [apiClient, onReloadRequested, setAlerts, setError, setSuccess]);

  const handleDismissAlert = useCallback(async (alert: SystemAlert) => {
    const confirmed = await confirm({
      title: 'Dismiss Alert',
      message: 'Dismiss this alert?',
      confirmText: 'Dismiss',
      variant: 'warning',
      icon: 'warning',
    });
    if (!confirmed) return;

    try {
      setError('');
      await apiClient.dismissAlert(alert.id);
      setAlerts((prev) => removeAlertById(prev, alert.id));
      setSuccess('Alert dismissed');
      await Promise.resolve(onReloadRequested());
    } catch (err: unknown) {
      logger.error('Failed to dismiss alert', { component: 'useAlertActions', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to dismiss alert');
    }
  }, [apiClient, confirm, onReloadRequested, setAlerts, setError, setSuccess]);

  const handleAssignAlert = useCallback(async (alert: SystemAlert, userId: string) => {
    try {
      setError('');
      await apiClient.assignAdminAlert(getAlertIdentity(alert), {
        assigned_to_user_id: userId ? Number(userId) : null,
      });
      setAlerts((prev) => setAlertAssignment(prev, alert.id, userId || undefined));
      setSuccess(userId ? 'Alert assigned' : 'Alert unassigned');
      await Promise.resolve(onReloadRequested());
    } catch (err: unknown) {
      logger.error('Failed to assign alert', { component: 'useAlertActions', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to assign alert');
    }
  }, [apiClient, onReloadRequested, setAlerts, setError, setSuccess]);

  const handleSnoozeAlert = useCallback(async (alert: SystemAlert, duration: SnoozeDurationOption) => {
    try {
      setError('');
      const snoozedUntil = resolveSnoozedUntil(duration);
      await apiClient.snoozeAdminAlert(getAlertIdentity(alert), {
        snoozed_until: snoozedUntil,
      });
      setAlerts((prev) => setAlertSnoozeUntil(prev, alert.id, snoozedUntil));
      setSuccess(`Alert snoozed for ${duration}`);
      await Promise.resolve(onReloadRequested());
    } catch (err: unknown) {
      logger.error('Failed to snooze alert', { component: 'useAlertActions', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to snooze alert');
    }
  }, [apiClient, onReloadRequested, setAlerts, setError, setSuccess]);

  const handleEscalateAlert = useCallback(async (alert: SystemAlert) => {
    if (alert.severity === 'critical') {
      return;
    }
    try {
      setError('');
      await apiClient.escalateAdminAlert(getAlertIdentity(alert), { severity: 'critical' });
      setAlerts((prev) => escalateAlertSeverity(prev, alert.id));
      setSuccess('Alert escalated to critical');
      await Promise.resolve(onReloadRequested());
    } catch (err: unknown) {
      logger.error('Failed to escalate alert', { component: 'useAlertActions', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to escalate alert');
    }
  }, [apiClient, onReloadRequested, setAlerts, setError, setSuccess]);

  return {
    handleAcknowledgeAlert,
    handleDismissAlert,
    handleAssignAlert,
    handleSnoozeAlert,
    handleEscalateAlert,
  };
};
