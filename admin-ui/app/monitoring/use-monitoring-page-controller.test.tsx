/* @vitest-environment jsdom */
import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useMonitoringPageController } from './use-monitoring-page-controller';

const {
  useConfirmMock,
  useMonitoringDashboardStateMock,
  useMonitoringNotificationStateMock,
  useMonitoringAlertStateMock,
  useMonitoringMetricsHistoryMock,
  useMonitoringMessagesMock,
  useMonitoringDataLoaderMock,
  useWatchlistActionsMock,
  useAlertActionsMock,
  useNotificationActionsMock,
  useAlertRulesMock,
} = vi.hoisted(() => ({
  useConfirmMock: vi.fn(),
  useMonitoringDashboardStateMock: vi.fn(),
  useMonitoringNotificationStateMock: vi.fn(),
  useMonitoringAlertStateMock: vi.fn(),
  useMonitoringMetricsHistoryMock: vi.fn(),
  useMonitoringMessagesMock: vi.fn(),
  useMonitoringDataLoaderMock: vi.fn(),
  useWatchlistActionsMock: vi.fn(),
  useAlertActionsMock: vi.fn(),
  useNotificationActionsMock: vi.fn(),
  useAlertRulesMock: vi.fn(),
}));

vi.mock('next/navigation', () => ({
  usePathname: () => '/monitoring',
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

vi.mock('@/components/ui/confirm-dialog', () => ({
  useConfirm: () => useConfirmMock(),
}));

vi.mock('./use-monitoring-dashboard-state', () => ({
  useMonitoringDashboardState: (...args: unknown[]) => useMonitoringDashboardStateMock(...args),
}));

vi.mock('./use-monitoring-notification-state', () => ({
  useMonitoringNotificationState: (...args: unknown[]) => useMonitoringNotificationStateMock(...args),
}));

vi.mock('./use-monitoring-alert-state', () => ({
  useMonitoringAlertState: (...args: unknown[]) => useMonitoringAlertStateMock(...args),
}));

vi.mock('./use-monitoring-metrics-history', () => ({
  useMonitoringMetricsHistory: (...args: unknown[]) => useMonitoringMetricsHistoryMock(...args),
}));

vi.mock('./use-monitoring-messages', () => ({
  useMonitoringMessages: (...args: unknown[]) => useMonitoringMessagesMock(...args),
}));

vi.mock('./use-monitoring-data-loader', () => ({
  useMonitoringDataLoader: (...args: unknown[]) => useMonitoringDataLoaderMock(...args),
}));

vi.mock('./use-watchlist-actions', () => ({
  useWatchlistActions: (...args: unknown[]) => useWatchlistActionsMock(...args),
}));

vi.mock('./use-alert-actions', () => ({
  useAlertActions: (...args: unknown[]) => useAlertActionsMock(...args),
}));

vi.mock('./use-notification-actions', () => ({
  useNotificationActions: (...args: unknown[]) => useNotificationActionsMock(...args),
}));

vi.mock('./use-alert-rules', () => ({
  useAlertRules: (...args: unknown[]) => useAlertRulesMock(...args),
}));

describe('useMonitoringPageController', () => {
  const confirmFn = vi.fn();
  const setShowSnoozedAlerts = vi.fn();
  const loadData = vi.fn();

  beforeEach(() => {
    loadData.mockReset();
    loadData.mockResolvedValue(undefined);
    setShowSnoozedAlerts.mockReset();

    useConfirmMock.mockReturnValue(confirmFn);

    useMonitoringDashboardStateMock.mockReturnValue({
      metrics: [{ name: 'cpu', value: 80, unit: '%', status: 'warning' }],
      setMetrics: vi.fn(),
      watchlists: [
        {
          id: 'watchlist-1',
          name: 'Core',
          target: '/api/v1/chat',
          type: 'endpoint',
          status: 'healthy',
        },
      ],
      setWatchlists: vi.fn(),
      systemStatus: [{ key: 'api', label: 'API', status: 'healthy', detail: 'Operational' }],
      setSystemStatus: vi.fn(),
      loading: false,
      setLoading: vi.fn(),
      lastUpdated: new Date('2026-03-01T12:00:00.000Z'),
      markMonitoringDataUpdated: vi.fn(),
    });

    useMonitoringNotificationStateMock.mockReturnValue({
      notificationSettings: null,
      setNotificationSettings: vi.fn(),
      recentNotifications: [
        {
          id: 'notification-1',
          channel: 'email',
          recipient: 'ops@example.com',
          severity: 'warning',
          status: 'sent',
          message: 'CPU warning',
          sentAt: '2026-03-01T12:00:00.000Z',
        },
      ],
      setRecentNotifications: vi.fn(),
      setNotificationSettingsStatus: vi.fn(),
      canSaveNotificationSettings: true,
    });

    useMonitoringAlertStateMock.mockReturnValue({
      alerts: [
        {
          id: 'alert-active',
          severity: 'warning',
          message: 'CPU high',
          timestamp: '2026-03-01T12:00:00.000Z',
          acknowledged: false,
        },
        {
          id: 'alert-ack',
          severity: 'warning',
          message: 'Memory high',
          timestamp: '2026-03-01T12:00:00.000Z',
          acknowledged: true,
        },
        {
          id: 'alert-snoozed',
          severity: 'warning',
          message: 'Disk high',
          timestamp: '2026-03-01T12:00:00.000Z',
          acknowledged: false,
          snoozed_until: '2999-01-01T00:00:00.000Z',
        },
      ],
      setAlerts: vi.fn(),
      alertsRef: { current: [] },
      assignableUsers: [{ id: 'user-1', label: 'Alice' }],
      setAssignableUsers: vi.fn(),
      showSnoozedAlerts: false,
      setShowSnoozedAlerts,
      alertHistory: [
        {
          id: 'history-1',
          alertId: 'alert-active',
          action: 'triggered',
          timestamp: '2026-03-01T12:00:00.000Z',
        },
      ],
      setAlertHistory: vi.fn(),
      alertHistoryRef: { current: [] },
    });

    useMonitoringMetricsHistoryMock.mockReturnValue({
      metricsHistory: [],
      timeRange: '24h',
      customRangeStart: '',
      customRangeEnd: '',
      rangeValidationError: '',
      activeRangeLabel: '24h',
      setCustomRangeStart: vi.fn(),
      setCustomRangeEnd: vi.fn(),
      loadMetricsHistoryForRange: vi.fn(),
      handleSelectTimeRange: vi.fn(),
      handleApplyCustomTimeRange: vi.fn(),
    });

    useMonitoringMessagesMock.mockReturnValue({
      error: '',
      setError: vi.fn(),
      success: 'Saved',
      setSuccess: vi.fn(),
    });

    useMonitoringDataLoaderMock.mockReturnValue({
      loadData,
    });

    useWatchlistActionsMock.mockReturnValue({
      showCreateWatchlist: false,
      setShowCreateWatchlist: vi.fn(),
      newWatchlist: {
        name: '',
        description: '',
        target: '',
        type: 'resource',
        threshold: 80,
      },
      setNewWatchlist: vi.fn(),
      deletingWatchlistId: null,
      handleCreateWatchlist: vi.fn(),
      handleDeleteWatchlist: vi.fn(),
    });

    useAlertActionsMock.mockReturnValue({
      handleAcknowledgeAlert: vi.fn(),
      handleDismissAlert: vi.fn(),
      handleAssignAlert: vi.fn(),
      handleSnoozeAlert: vi.fn(),
      handleEscalateAlert: vi.fn(),
    });

    useNotificationActionsMock.mockReturnValue({
      notificationsSaving: false,
      handleSaveNotificationSettings: vi.fn(),
      handleTestNotification: vi.fn(),
    });

    useAlertRulesMock.mockReturnValue({
      alertRules: [
        {
          id: 'rule-1',
          metric: 'cpu',
          operator: '>',
          threshold: 90,
          durationMinutes: 5,
          severity: 'critical',
          createdAt: '2026-03-01T12:00:00.000Z',
        },
      ],
      alertRuleDraft: {
        metric: 'cpu',
        operator: '>',
        threshold: '90',
        durationMinutes: '5',
        severity: 'critical',
      },
      alertRuleValidationErrors: {},
      alertRulesSaving: false,
      handleAlertRuleDraftChange: vi.fn(),
      handleCreateAlertRule: vi.fn(),
      handleDeleteAlertRule: vi.fn(),
    });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('composes section props from child hooks and triggers initial load', async () => {
    const { result } = renderHook(() => useMonitoringPageController());

    await waitFor(() => {
      expect(loadData).toHaveBeenCalled();
    });
    const initialLoadCalls = loadData.mock.calls.length;
    expect(initialLoadCalls).toBeGreaterThanOrEqual(1);

    expect(result.current.headerProps.loading).toBe(false);
    expect(result.current.feedbackBannersProps.success).toBe('Saved');
    expect(result.current.feedbackBannersProps.activeAlertsCount).toBe(1);
    expect(result.current.metricsSectionProps.timeRangeControlsProps.options).toHaveLength(6);
    expect(
      result.current.managementPanelsProps.alertRulesPanelProps.rules
    ).toHaveLength(1);
    expect(
      result.current.managementPanelsProps.watchlistsPanelProps.watchlists
    ).toHaveLength(1);

    await act(async () => {
      await result.current.headerProps.onRefresh();
    });
    expect(loadData).toHaveBeenCalledTimes(initialLoadCalls + 1);

    const watchlistActionsArgs = useWatchlistActionsMock.mock.calls[0]?.[0];
    expect(watchlistActionsArgs).toEqual(
      expect.objectContaining({
        confirm: confirmFn,
        onReloadRequested: expect.any(Function),
      })
    );

    const alertActionsArgs = useAlertActionsMock.mock.calls[0]?.[0];
    expect(alertActionsArgs).toEqual(
      expect.objectContaining({
        confirm: confirmFn,
        onReloadRequested: expect.any(Function),
      })
    );
    expect(useNotificationActionsMock).toHaveBeenCalledWith(
      expect.objectContaining({
        canSave: true,
      })
    );
  });

  it('forwards snoozed toggle through state-updater setter', () => {
    const { result } = renderHook(() => useMonitoringPageController());

    act(() => {
      result.current.managementPanelsProps.alertsPanelProps.onToggleShowSnoozed();
    });

    expect(setShowSnoozedAlerts).toHaveBeenCalledTimes(1);
    const updater = setShowSnoozedAlerts.mock.calls[0][0] as (prev: boolean) => boolean;
    expect(updater(false)).toBe(true);
    expect(updater(true)).toBe(false);
  });

  it('keeps top-level section prop references stable across rerender when dependencies are unchanged', async () => {
    const { result, rerender } = renderHook(() => useMonitoringPageController());

    await waitFor(() => {
      expect(loadData).toHaveBeenCalled();
    });
    const initialLoadCalls = loadData.mock.calls.length;

    const headerProps = result.current.headerProps;
    const feedbackBannersProps = result.current.feedbackBannersProps;
    const metricsSectionProps = result.current.metricsSectionProps;
    const managementPanelsProps = result.current.managementPanelsProps;
    const toggleSnoozed = result.current.managementPanelsProps.alertsPanelProps.onToggleShowSnoozed;

    rerender();

    expect(loadData).toHaveBeenCalledTimes(initialLoadCalls);
    expect(result.current.headerProps).toBe(headerProps);
    expect(result.current.feedbackBannersProps).toBe(feedbackBannersProps);
    expect(result.current.metricsSectionProps).toBe(metricsSectionProps);
    expect(result.current.managementPanelsProps).toBe(managementPanelsProps);
    expect(result.current.managementPanelsProps.alertsPanelProps.onToggleShowSnoozed).toBe(
      toggleSnoozed
    );
  });

  it('does not start overlapping auto-refresh loads while a request is still in flight', async () => {
    vi.useFakeTimers();

    let resolveLoad: (() => void) | null = null;
    loadData.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveLoad = resolve;
        })
    );

    renderHook(() => useMonitoringPageController());

    await act(async () => {
      await Promise.resolve();
    });
    expect(loadData).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(120_000);
    });

    expect(loadData).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveLoad?.();
      await Promise.resolve();
    });

    act(() => {
      vi.advanceTimersByTime(60_000);
    });

    expect(loadData).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });
});
