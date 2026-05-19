/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import MonitoringPage from '../page';
import { normalizeHealthStatus } from '../status-utils';
import { api } from '@/lib/api-client';
import { formatAxeViolations, getCriticalAndSeriousAxeViolations } from '@/test-utils/axe';

const confirmMock = vi.hoisted(() => vi.fn());

vi.mock('next/navigation', () => ({
  usePathname: () => '/monitoring',
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/ui/confirm-dialog', () => ({
  useConfirm: () => confirmMock,
}));

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => (
    <div data-testid="chart">{children}</div>
  ),
  LineChart: ({ data, children }: { data: unknown; children: ReactNode }) => (
    <div data-testid="line-chart" data-points={JSON.stringify(data)}>
      {children}
    </div>
  ),
  Line: () => null,
  XAxis: () => null,
  YAxis: ({ yAxisId }: { yAxisId?: string }) => (
    <div data-testid={`y-axis-${yAxisId || 'default'}`} />
  ),
  CartesianGrid: () => null,
  Tooltip: () => null,
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getMetrics: vi.fn(),
    getWatchlists: vi.fn(),
    getAlerts: vi.fn(),
    getAdminAlertHistory: vi.fn(),
    getAdminAlertRules: vi.fn(),
    getUsers: vi.fn(),
    getHealth: vi.fn(),
    getHealthMetrics: vi.fn(),
    getLlmHealth: vi.fn(),
    getRagHealth: vi.fn(),
    getTtsHealth: vi.fn(),
    getSttHealth: vi.fn(),
    getEmbeddingsHealth: vi.fn(),
    getMetricsText: vi.fn(),
    getNotificationSettings: vi.fn(),
    getRecentNotifications: vi.fn(),
    createWatchlist: vi.fn(),
    deleteWatchlist: vi.fn(),
    acknowledgeAlert: vi.fn(),
    dismissAlert: vi.fn(),
    createAdminAlertRule: vi.fn(),
    deleteAdminAlertRule: vi.fn(),
    assignAdminAlert: vi.fn(),
    snoozeAdminAlert: vi.fn(),
    escalateAdminAlert: vi.fn(),
    getMonitoringMetrics: vi.fn(),
    updateNotificationSettings: vi.fn(),
    testNotification: vi.fn(),
  },
}));

const apiMock = vi.mocked(api);

const setDefaultApiMocks = () => {
  apiMock.getMetrics.mockResolvedValue({ cpu_usage: 45, memory_usage: 67 });
  apiMock.getWatchlists.mockResolvedValue([]);
  apiMock.getAlerts.mockResolvedValue([]);
  apiMock.getAdminAlertHistory.mockResolvedValue({ items: [] });
  apiMock.getAdminAlertRules.mockResolvedValue({ items: [] });
  apiMock.getUsers.mockResolvedValue([
    { id: '1', email: 'admin@example.com', username: 'admin' },
  ]);
  apiMock.getHealth.mockResolvedValue({
    status: 'ok',
    checks: { database: { status: 'ok' } },
  });
  apiMock.getLlmHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getRagHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getTtsHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getSttHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getEmbeddingsHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getMetricsText.mockResolvedValue('queue_depth 3');
  apiMock.getNotificationSettings.mockResolvedValue({
    channels: [],
    alert_threshold: 'warning',
    digest_enabled: false,
    digest_frequency: 'daily',
  });
  apiMock.getRecentNotifications.mockResolvedValue([]);
  apiMock.getHealthMetrics.mockResolvedValue({
    cpu: { percent: 10 },
    memory: { percent: 20 },
  });
  apiMock.getMonitoringMetrics.mockResolvedValue([
    {
      timestamp: '2026-02-17T00:00:00.000Z',
      cpu: 10,
      memory: 20,
      disk_usage: 33,
      throughput: 42,
      active_connections: 7,
      queue_depth: 3,
    },
  ]);
  apiMock.createWatchlist.mockResolvedValue({});
  apiMock.acknowledgeAlert.mockResolvedValue({});
  apiMock.dismissAlert.mockResolvedValue({});
  apiMock.createAdminAlertRule.mockResolvedValue({ item: null });
  apiMock.deleteAdminAlertRule.mockResolvedValue({ status: 'deleted', id: 1 });
  apiMock.assignAdminAlert.mockResolvedValue({ item: {} });
  apiMock.snoozeAdminAlert.mockResolvedValue({ item: {} });
  apiMock.escalateAdminAlert.mockResolvedValue({ item: {} });
  apiMock.updateNotificationSettings.mockResolvedValue({});
  apiMock.testNotification.mockResolvedValue({});
};

const expectLoadDataCalls = () => {
  expect(apiMock.getMetrics).toHaveBeenCalled();
  expect(apiMock.getWatchlists).toHaveBeenCalled();
  expect(apiMock.getAlerts).toHaveBeenCalled();
  expect(apiMock.getUsers).toHaveBeenCalled();
  expect(apiMock.getHealth).toHaveBeenCalled();
  expect(apiMock.getLlmHealth).toHaveBeenCalled();
  expect(apiMock.getRagHealth).toHaveBeenCalled();
  expect(apiMock.getTtsHealth).toHaveBeenCalled();
  expect(apiMock.getSttHealth).toHaveBeenCalled();
  expect(apiMock.getEmbeddingsHealth).toHaveBeenCalled();
  expect(apiMock.getMetricsText).toHaveBeenCalled();
  expect(apiMock.getNotificationSettings).toHaveBeenCalled();
  expect(apiMock.getRecentNotifications).toHaveBeenCalled();
};

const createDeferred = <T,>() => {
  let resolve: (value: T) => void = () => {};
  let reject: (reason?: unknown) => void = () => {};
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

const flushPromises = async () => {
  await Promise.resolve();
  await Promise.resolve();
};

beforeEach(() => {
  localStorage.clear();
  confirmMock.mockResolvedValue(true);
  setDefaultApiMocks();
});

afterEach(() => {
  localStorage.clear();
  cleanup();
  vi.useRealTimers();
  vi.resetAllMocks();
});

describe('normalizeHealthStatus', () => {
  it('maps known statuses to normalized values', () => {
    expect(normalizeHealthStatus('OK')).toBe('healthy');
    expect(normalizeHealthStatus('healthy')).toBe('healthy');
    expect(normalizeHealthStatus('READY')).toBe('healthy');
    expect(normalizeHealthStatus('alive')).toBe('healthy');
    expect(normalizeHealthStatus('degraded')).toBe('warning');
    expect(normalizeHealthStatus('warning')).toBe('warning');
    expect(normalizeHealthStatus('critical')).toBe('critical');
    expect(normalizeHealthStatus('error')).toBe('critical');
    expect(normalizeHealthStatus('not_ready')).toBe('critical');
  });

  it('returns unknown for empty or unexpected values', () => {
    expect(normalizeHealthStatus()).toBe('unknown');
    expect(normalizeHealthStatus('mystery')).toBe('unknown');
  });
});

describe('MonitoringPage', () => {
  it('has no critical/serious axe violations in the default state', async () => {
    const { container } = render(<MonitoringPage />);
    await screen.findByRole('heading', { name: 'Monitoring' });

    const violations = await getCriticalAndSeriousAxeViolations(container);
    expect(violations, formatAxeViolations(violations)).toEqual([]);
  });

  it('renders loading state while data is fetching', async () => {
    const metricsDeferred = createDeferred<Record<string, unknown>>();
    const watchlistsDeferred = createDeferred<unknown[]>();
    const alertsDeferred = createDeferred<unknown[]>();
    const healthDeferred = createDeferred<Record<string, unknown>>();
    const llmDeferred = createDeferred<Record<string, unknown>>();
    const ragDeferred = createDeferred<Record<string, unknown>>();

    apiMock.getMetrics.mockReturnValue(metricsDeferred.promise);
    apiMock.getWatchlists.mockReturnValue(watchlistsDeferred.promise);
    apiMock.getAlerts.mockReturnValue(alertsDeferred.promise);
    apiMock.getHealth.mockReturnValue(healthDeferred.promise);
    apiMock.getLlmHealth.mockReturnValue(llmDeferred.promise);
    apiMock.getRagHealth.mockReturnValue(ragDeferred.promise);

    render(<MonitoringPage />);

    expect(screen.getByText('Loading metrics...')).toBeTruthy();
    // Skeleton loading placeholders replace bare "Loading..." text
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThan(0);

    metricsDeferred.resolve({});
    watchlistsDeferred.resolve([]);
    alertsDeferred.resolve([]);
    healthDeferred.resolve({ status: 'ok', checks: { database: { status: 'ok' } } });
    llmDeferred.resolve({ status: 'ok' });
    ragDeferred.resolve({ status: 'ok' });

    await flushPromises();
  });

  it('renders empty state when no data is returned', async () => {
    apiMock.getMetrics.mockResolvedValue({});
    apiMock.getWatchlists.mockResolvedValue([]);
    apiMock.getAlerts.mockResolvedValue([]);

    render(<MonitoringPage />);

    expect(await screen.findByText('No metrics available. The server may not expose metrics.')).toBeTruthy();
    expect(screen.getByTestId('monitoring-alert-count-live').textContent).toContain('0 active alerts');
    const statusRegion = screen.getByTestId('system-status-live-region');
    expect(statusRegion.getAttribute('role')).toBe('status');
    expect(statusRegion.getAttribute('aria-live')).toBe('polite');
    expect(screen.getByText('No watchlists configured')).toBeTruthy();
    expect(screen.getByText('No alerts - system is healthy')).toBeTruthy();

    expectLoadDataCalls();
  });

  it('renders success state with metrics and watchlists', async () => {
    apiMock.getMetrics.mockResolvedValue({ cpu_usage: 55, active_users: 12 });
    apiMock.getWatchlists.mockResolvedValue([
      {
        id: 'watch-1',
        name: 'API Response Time',
        description: 'Latency watch',
        target: '/api/v1/chat',
        type: 'resource',
        threshold: 80,
        status: 'healthy',
        last_checked: '2024-05-01T12:00:00Z',
        created_at: '2024-05-01T12:00:00Z',
      },
    ]);
    apiMock.getAlerts.mockResolvedValue([
      {
        id: 'alert-1',
        severity: 'warning',
        message: 'High CPU usage',
        source: 'system',
        timestamp: '2024-05-01T12:00:00Z',
        acknowledged: true,
        acknowledged_at: '2024-05-01T12:05:00Z',
        acknowledged_by: 'admin',
      },
    ]);

    render(<MonitoringPage />);

    expect(await screen.findByText('Cpu Usage')).toBeTruthy();
    expect(screen.getByText('API Response Time')).toBeTruthy();
    expect(screen.getAllByText('High CPU usage').length).toBeGreaterThan(0);

    expectLoadDataCalls();
  });

  it('renders error state when initial load fails', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    apiMock.getMetrics.mockImplementation(() => {
      throw new Error('Load failed');
    });

    render(<MonitoringPage />);

    expect(await screen.findByText('Load failed')).toBeTruthy();
    consoleError.mockRestore();
  });

  it('refreshes metrics history on interval polling', async () => {
    vi.useFakeTimers();
    apiMock.getMonitoringMetrics.mockResolvedValue([
      { timestamp: '2026-02-17T00:00:00.000Z', cpu: 5, memory: 10, disk_usage: 20, throughput: 2, active_connections: 1, queue_depth: 0 },
    ]);

    render(<MonitoringPage />);
    await flushPromises();
    await flushPromises();
    const baselineCalls = apiMock.getMonitoringMetrics.mock.calls.length;
    expect(baselineCalls).toBeGreaterThanOrEqual(2);

    // Advance past the 60s auto-refresh interval; both the 60s full-page
    // auto-refresh and the 5-minute metrics-history poll may fire.
    await vi.advanceTimersByTimeAsync(5 * 60 * 1000);
    await flushPromises();

    expect(apiMock.getMonitoringMetrics.mock.calls.length).toBeGreaterThan(baselineCalls);

    expectLoadDataCalls();
  });

  it('creates a watchlist and reloads data', async () => {
    const user = userEvent.setup();
    apiMock.getWatchlists
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        {
          id: 'watch-2',
          name: 'CPU Usage',
          description: 'CPU threshold',
          target: 'cpu_usage',
          type: 'metric',
          threshold: 85,
          status: 'warning',
          last_checked: '2024-05-01T12:10:00Z',
          created_at: '2024-05-01T12:00:00Z',
        },
      ]);

    render(<MonitoringPage />);

    expect(await screen.findByText('No watchlists configured')).toBeTruthy();
    expectLoadDataCalls();

    await user.click(screen.getByRole('button', { name: 'Add' }));
    await user.type(screen.getByLabelText('Name'), 'CPU Usage');
    await user.type(screen.getByLabelText('Description'), 'CPU threshold');
    await user.type(screen.getByLabelText('Target'), 'cpu_usage');
    await user.click(screen.getByRole('button', { name: 'Create Watchlist' }));

    await waitFor(() => {
      expect(apiMock.createWatchlist).toHaveBeenCalledWith({
        name: 'CPU Usage',
        description: 'CPU threshold',
        target: 'cpu_usage',
        type: 'resource',
        threshold: 80,
      });
    });

    await waitFor(() => {
      expect(apiMock.getWatchlists).toHaveBeenCalledTimes(2);
      expect(apiMock.getMetrics).toHaveBeenCalledTimes(2);
    });

    expect(screen.getByText('CPU Usage')).toBeTruthy();
    expect(screen.queryByText('Configure a new resource or metric to monitor')).toBeNull();
  });

  it('acknowledges alerts and reloads data', async () => {
    const user = userEvent.setup();
    apiMock.getAlerts
      .mockResolvedValueOnce([
        {
          id: 'alert-2',
          severity: 'warning',
          message: 'High memory usage',
          source: 'system',
          timestamp: '2024-05-01T12:00:00Z',
          acknowledged: false,
        },
      ])
      .mockResolvedValueOnce([
        {
          id: 'alert-2',
          severity: 'warning',
          message: 'High memory usage',
          source: 'system',
          timestamp: '2024-05-01T12:00:00Z',
          acknowledged: true,
          acknowledged_at: '2024-05-01T12:05:00Z',
          acknowledged_by: 'admin',
        },
      ]);

    render(<MonitoringPage />);

    expect((await screen.findAllByText('High memory usage')).length).toBeGreaterThan(0);
    expectLoadDataCalls();

    await user.click(screen.getByTitle('Acknowledge'));

    await waitFor(() => {
      expect(apiMock.acknowledgeAlert).toHaveBeenCalledWith('alert-2');
      expect(apiMock.getAlerts).toHaveBeenCalledTimes(2);
    });

    expect(screen.getAllByText('Acknowledged').length).toBeGreaterThan(0);
  });

  it('dismisses alerts after confirmation', async () => {
    const user = userEvent.setup();
    apiMock.getAlerts
      .mockResolvedValueOnce([
        {
          id: 'alert-3',
          severity: 'critical',
          message: 'Service unavailable',
          source: 'rag',
          timestamp: '2024-05-01T12:00:00Z',
          acknowledged: false,
        },
      ])
      .mockResolvedValueOnce([]);

    render(<MonitoringPage />);

    expect((await screen.findAllByText('Service unavailable')).length).toBeGreaterThan(0);
    expectLoadDataCalls();

    await user.click(screen.getByTitle('Dismiss'));

    await waitFor(() => {
      expect(apiMock.dismissAlert).toHaveBeenCalledWith('alert-3');
      expect(apiMock.getAlerts).toHaveBeenCalledTimes(2);
    });

    expect(screen.getByText('No alerts - system is healthy')).toBeTruthy();
  });

  it('shows error alert when watchlist creation fails', async () => {
    const user = userEvent.setup();
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    apiMock.createWatchlist.mockRejectedValue(new Error('Create failed'));

    render(<MonitoringPage />);

    await user.click(screen.getByRole('button', { name: 'Add' }));
    await user.type(screen.getByLabelText('Name'), 'Disk Usage');
    await user.type(screen.getByLabelText('Target'), 'disk_usage');
    await user.click(screen.getByRole('button', { name: 'Create Watchlist' }));

    expect(await screen.findByText('Create failed')).toBeTruthy();
    consoleError.mockRestore();
  });

  it('updates metrics query parameters when selecting a different time range', async () => {
    const user = userEvent.setup();
    render(<MonitoringPage />);

    await screen.findByText('Monitoring');
    await user.click(screen.getByTestId('monitoring-time-range-7d'));

    await waitFor(() => {
      expect(apiMock.getMonitoringMetrics).toHaveBeenLastCalledWith(
        expect.objectContaining({ granularity: '1h' })
      );
    });
  });

  it('validates custom ranges require start before end', async () => {
    const user = userEvent.setup();
    render(<MonitoringPage />);
    await screen.findByText('Monitoring');

    await user.click(screen.getByTestId('monitoring-time-range-custom'));
    const startInput = screen.getByLabelText('Custom Start');
    const endInput = screen.getByLabelText('Custom End');

    await user.clear(startInput);
    await user.type(startInput, '2026-02-17T12:00');
    await user.clear(endInput);
    await user.type(endInput, '2026-02-17T08:00');
    await user.click(screen.getByTestId('monitoring-time-range-apply-custom'));

    expect(await screen.findByText('Custom range start must be before end.')).toBeTruthy();
  });

  it('toggles chart metric series visibility from legend controls', async () => {
    const user = userEvent.setup();
    render(<MonitoringPage />);
    await screen.findByText('Monitoring');

    const cpuToggle = await screen.findByTestId('metric-series-toggle-cpu');
    expect(cpuToggle.getAttribute('aria-pressed')).toBe('true');
    await user.click(cpuToggle);
    expect(cpuToggle.getAttribute('aria-pressed')).toBe('false');
  });

  it('renders separate y-axes for percent and count metrics', async () => {
    render(<MonitoringPage />);
    await screen.findByText('Monitoring');

    expect(await screen.findByTestId('y-axis-percent')).toBeTruthy();
    expect(await screen.findByTestId('y-axis-count')).toBeTruthy();
  });

  it('renders expanded subsystem status cards with timing metadata', async () => {
    render(<MonitoringPage />);
    await screen.findByText('Monitoring');

    const subsystemKeys = [
      'api',
      'database',
      'llm',
      'rag',
      'tts',
      'stt',
      'embeddings',
      'cache',
      'queue',
    ] as const;

    subsystemKeys.forEach((key) => {
      expect(screen.getByTestId(`system-status-card-${key}`)).toBeTruthy();
    });

    await waitFor(() => {
      const apiResponse = screen.getByTestId('system-status-response-api').textContent || '';
      expect(apiResponse).toContain('Response:');
      expect(apiResponse).toContain('ms');
    });
  });

  it('falls back to metrics for subsystem status when endpoint checks fail', async () => {
    apiMock.getTtsHealth.mockRejectedValue(new Error('TTS endpoint unavailable'));
    apiMock.getSttHealth.mockRejectedValue(new Error('STT endpoint unavailable'));
    apiMock.getEmbeddingsHealth.mockRejectedValue(new Error('Embeddings endpoint unavailable'));
    apiMock.getMetricsText.mockResolvedValue(
      [
        'tts_requests_total 3',
        'stt_transcriptions_total 2',
        'embeddings_requests_total 8',
        'jobs_queue_depth 91',
      ].join('\n')
    );

    render(<MonitoringPage />);
    await screen.findByText('Monitoring');

    await waitFor(() => {
      expect(screen.getByTestId('system-status-card-tts').textContent).toContain('metrics fallback');
      expect(screen.getByTestId('system-status-card-stt').textContent).toContain('metrics fallback');
      expect(screen.getByTestId('system-status-card-embeddings').textContent).toContain('metrics fallback');
      expect(screen.getByTestId('system-status-card-queue').textContent).toContain('Depth 91');
    });
  });

  it('renders notification delivery dashboard and retries failed notifications', async () => {
    const user = userEvent.setup();
    apiMock.getRecentNotifications
      .mockResolvedValueOnce({
        items: [
          {
            id: 'n-1',
            channel: 'email',
            message: 'Digest sent',
            status: 'sent',
            timestamp: '2026-02-17T10:00:00Z',
          },
          {
            id: 'n-2',
            channel: 'webhook',
            message: 'Webhook delivery failed',
            status: 'failed',
            timestamp: '2026-02-17T10:05:00Z',
            error: 'Timeout',
          },
          {
            id: 'n-3',
            channel: 'slack',
            message: 'Queued notification',
            status: 'pending',
            timestamp: '2026-02-17T10:06:00Z',
          },
        ],
      })
      .mockResolvedValueOnce({ items: [] });

    render(<MonitoringPage />);

    expect(await screen.findByTestId('notification-delivery-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('notification-delivery-total').textContent).toContain('3');
    expect(screen.getByTestId('notification-delivery-rate').textContent).toContain('33.3%');
    expect(screen.getByTestId('notification-failure-rate').textContent).toContain('33.3%');
    expect(screen.getByTestId('notification-channel-email').textContent).toContain('1');
    expect(screen.getByTestId('notification-channel-webhook').textContent).toContain('1');
    expect(screen.getByTestId('notification-error-n-2').textContent).toContain('Timeout');

    await user.click(screen.getByRole('button', { name: 'Retry' }));

    await waitFor(() => {
      expect(apiMock.testNotification).toHaveBeenCalledWith(
        expect.objectContaining({
          message: 'Webhook delivery failed',
          severity: 'error',
        }),
      );
    });
  });
});
