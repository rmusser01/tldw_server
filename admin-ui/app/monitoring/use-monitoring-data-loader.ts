import { useCallback, type MutableRefObject } from 'react';
import { api } from '@/lib/api-client';
import { logger } from '@/lib/logger';
import { measureTimedEndpoint } from '@/lib/monitoring-health';
import type { MonitoringTimeRangeOption } from '@/lib/monitoring-metrics';
import {
  fetchMonitoringSettledResults,
  monitoringLoadResultEntries,
  resolveMonitoringLoadState,
} from './load-state-utils';
import type { MonitoringNotificationSettingsStatus } from './use-monitoring-notification-state';
import type {
  AlertAssignableUser,
  AlertHistoryEntry,
  Metric,
  NotificationSettings,
  RecentNotification,
  SystemAlert,
  SystemStatusItem,
  Watchlist,
} from './types';

type UseMonitoringDataLoaderArgs = {
  alertsRef: MutableRefObject<SystemAlert[]>;
  alertHistoryRef: MutableRefObject<AlertHistoryEntry[]>;
  customRangeStart: string;
  customRangeEnd: string;
  timeRange: MonitoringTimeRangeOption;
  loadMetricsHistoryForRange: (
    timeRange: MonitoringTimeRangeOption,
    customRangeStart?: string,
    customRangeEnd?: string
  ) => Promise<boolean> | boolean | void;
  markMonitoringDataUpdated: () => void;
  setLoading: (value: boolean) => void;
  setError: (value: string) => void;
  setNotificationSettingsStatus: (value: MonitoringNotificationSettingsStatus) => void;
  setNotificationSettings: (value: NotificationSettings | null) => void;
  setRecentNotifications: (value: RecentNotification[]) => void;
  setMetrics: (value: Metric[]) => void;
  setWatchlists: (value: Watchlist[]) => void;
  setAlerts: (value: SystemAlert[]) => void;
  setAlertHistory: (value: AlertHistoryEntry[]) => void;
  setAssignableUsers: (value: AlertAssignableUser[]) => void;
  setSystemStatus: (value: SystemStatusItem[]) => void;
  metricWarningThreshold: number;
  metricCriticalThreshold: number;
};

export const useMonitoringDataLoader = ({
  alertsRef,
  alertHistoryRef,
  customRangeStart,
  customRangeEnd,
  timeRange,
  loadMetricsHistoryForRange,
  markMonitoringDataUpdated,
  setLoading,
  setError,
  setNotificationSettingsStatus,
  setNotificationSettings,
  setRecentNotifications,
  setMetrics,
  setWatchlists,
  setAlerts,
  setAlertHistory,
  setAssignableUsers,
  setSystemStatus,
  metricWarningThreshold,
  metricCriticalThreshold,
}: UseMonitoringDataLoaderArgs) => {
  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      setNotificationSettingsStatus('pending');

      const settledResults = await fetchMonitoringSettledResults({
        apiClient: api,
        measureTimedRequest: measureTimedEndpoint,
      });

      monitoringLoadResultEntries(settledResults).forEach(({ name, result }) => {
        if (result.status === 'rejected') {
          logger.warn(`Failed to load ${name}`, { component: 'useMonitoringDataLoader', error: result.reason instanceof Error ? result.reason.message : String(result.reason) });
        }
      });

      const resolvedState = resolveMonitoringLoadState({
        settledResults,
        previousAlerts: alertsRef.current,
        metricWarningThreshold,
        metricCriticalThreshold,
      });

      setNotificationSettingsStatus(resolvedState.notificationSettingsStatus);
      setNotificationSettings(resolvedState.notificationSettings);
      setRecentNotifications(resolvedState.recentNotifications);
      if (resolvedState.metrics) {
        setMetrics(resolvedState.metrics);
      }
      if (resolvedState.watchlists) {
        setWatchlists(resolvedState.watchlists);
      }
      if (resolvedState.alerts) {
        alertsRef.current = resolvedState.alerts;
        setAlerts(resolvedState.alerts);
      }
      if (resolvedState.alertHistory) {
        alertHistoryRef.current = resolvedState.alertHistory;
        setAlertHistory(resolvedState.alertHistory);
      }
      setAssignableUsers(resolvedState.assignableUsers);
      setSystemStatus(resolvedState.systemStatus);

      void loadMetricsHistoryForRange(timeRange, customRangeStart, customRangeEnd);
      markMonitoringDataUpdated();
    } catch (err: unknown) {
      logger.error('Failed to load monitoring data', { component: 'useMonitoringDataLoader', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to load monitoring data');
      setNotificationSettingsStatus('rejected');
      setNotificationSettings(null);
    } finally {
      setLoading(false);
    }
  }, [
    alertHistoryRef,
    alertsRef,
    customRangeEnd,
    customRangeStart,
    loadMetricsHistoryForRange,
    markMonitoringDataUpdated,
    metricCriticalThreshold,
    metricWarningThreshold,
    setAlertHistory,
    setAlerts,
    setAssignableUsers,
    setError,
    setLoading,
    setMetrics,
    setNotificationSettings,
    setNotificationSettingsStatus,
    setRecentNotifications,
    setSystemStatus,
    setWatchlists,
    timeRange,
  ]);

  return {
    loadData,
  };
};
