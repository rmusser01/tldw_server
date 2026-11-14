/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import DashboardPage from '../page';
import { api } from '@/lib/api-client';

const toast = vi.hoisted(() => ({ success: vi.fn(), warning: vi.fn(), error: vi.fn() }));

const createDeferred = <T,>() => {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

vi.mock('next/link', () => ({
  default: ({ href, children, ...props }: { href: string; children: ReactNode }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/OrgContextSwitcher', () => ({
  useOrgContext: () => ({ selectedOrg: null }),
}));

vi.mock('@/components/ui/confirm-dialog', () => ({
  useConfirm: () => vi.fn().mockResolvedValue(true),
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => toast,
}));

vi.mock('@/components/ui/accessible-icon-button', () => ({
  AccessibleIconButton: ({ label, onClick }: { label: string; onClick: () => void }) => (
    <button type="button" onClick={onClick}>
      {label}
    </button>
  ),
}));

vi.mock('@/components/dashboard/StatsGrid', () => ({
  StatsGrid: () => <section aria-label="Dashboard stats">Stats</section>,
}));

vi.mock('@/components/dashboard/ActivitySection', () => ({
  ActivitySection: () => <section aria-label="Dashboard activity">Activity</section>,
}));

vi.mock('@/components/dashboard/RecentActivityCard', () => ({
  RecentActivityCard: () => <section aria-label="Recent activity">Recent activity</section>,
}));

vi.mock('@/components/dashboard/QuickActionsCard', () => ({
  QuickActionsCard: () => <section aria-label="Quick actions">Quick actions</section>,
}));

vi.mock('@/components/dashboard/CreateOrganizationDialog', () => ({
  CreateOrganizationDialog: () => <button type="button">Create organization</button>,
}));

vi.mock('@/components/dashboard/CreateRegistrationCodeDialog', () => ({
  CreateRegistrationCodeDialog: () => <button type="button">Create registration code</button>,
}));

vi.mock('@/components/dashboard/CreateUserDialog', () => ({
  CreateUserDialog: () => <button type="button">Create user</button>,
}));

vi.mock('@/lib/billing', () => ({
  fetchDashboardBillingStats: vi.fn(),
  isBillingEnabled: () => false,
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getDashboardStats: vi.fn(),
    getUsers: vi.fn(),
    getOrganizations: vi.fn(),
    getLLMProviders: vi.fn(),
    getAuditLogs: vi.fn(),
    getAlerts: vi.fn(),
    acknowledgeAlert: vi.fn(),
    getRealtimeStats: vi.fn(),
    getDashboardActivity: vi.fn(),
    getUsageDaily: vi.fn(),
    getLlmUsageSummary: vi.fn(),
    getJobsStats: vi.fn(),
    getMetricsText: vi.fn(),
    getRegistrationSettings: vi.fn(),
    getRegistrationCodes: vi.fn(),
    getHealth: vi.fn(),
    getLlmHealth: vi.fn(),
    getRagHealth: vi.fn(),
    getTtsHealth: vi.fn(),
    getSttHealth: vi.fn(),
    getEmbeddingsHealth: vi.fn(),
    getSecurityHealth: vi.fn(),
    getIncidents: vi.fn(),
    createRegistrationCode: vi.fn(),
    deleteRegistrationCode: vi.fn(),
    updateRegistrationSettings: vi.fn(),
    createUser: vi.fn(),
    createOrganization: vi.fn(),
  },
}));

type ApiMock = {
  getDashboardStats: ReturnType<typeof vi.fn>;
  getUsers: ReturnType<typeof vi.fn>;
  getOrganizations: ReturnType<typeof vi.fn>;
  getLLMProviders: ReturnType<typeof vi.fn>;
  getAuditLogs: ReturnType<typeof vi.fn>;
  getAlerts: ReturnType<typeof vi.fn>;
  getDashboardActivity: ReturnType<typeof vi.fn>;
  getUsageDaily: ReturnType<typeof vi.fn>;
  getLlmUsageSummary: ReturnType<typeof vi.fn>;
  getJobsStats: ReturnType<typeof vi.fn>;
  getMetricsText: ReturnType<typeof vi.fn>;
  getRegistrationSettings: ReturnType<typeof vi.fn>;
  getRegistrationCodes: ReturnType<typeof vi.fn>;
  getHealth: ReturnType<typeof vi.fn>;
  getLlmHealth: ReturnType<typeof vi.fn>;
  getRagHealth: ReturnType<typeof vi.fn>;
  getTtsHealth: ReturnType<typeof vi.fn>;
  getSttHealth: ReturnType<typeof vi.fn>;
  getEmbeddingsHealth: ReturnType<typeof vi.fn>;
  getSecurityHealth: ReturnType<typeof vi.fn>;
  getIncidents: ReturnType<typeof vi.fn>;
  createRegistrationCode: ReturnType<typeof vi.fn>;
  deleteRegistrationCode: ReturnType<typeof vi.fn>;
  updateRegistrationSettings: ReturnType<typeof vi.fn>;
  createUser: ReturnType<typeof vi.fn>;
  createOrganization: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

beforeEach(() => {
  apiMock.getDashboardStats.mockResolvedValue({});
  apiMock.getUsers.mockResolvedValue([]);
  apiMock.getOrganizations.mockResolvedValue([]);
  apiMock.getLLMProviders.mockResolvedValue([]);
  apiMock.getAuditLogs.mockResolvedValue({ entries: [] });
  apiMock.getAlerts.mockResolvedValue([]);
  apiMock.getDashboardActivity.mockResolvedValue([]);
  apiMock.getUsageDaily.mockResolvedValue([]);
  apiMock.getLlmUsageSummary.mockResolvedValue([]);
  apiMock.getJobsStats.mockResolvedValue([]);
  apiMock.getMetricsText.mockResolvedValue('');
  apiMock.getRegistrationSettings.mockResolvedValue({
    enable_registration: false,
    require_registration_code: false,
    self_registration_allowed: true,
  });
  apiMock.getRegistrationCodes.mockResolvedValue([]);
  apiMock.getHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getLlmHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getRagHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getTtsHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getSttHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getEmbeddingsHealth.mockResolvedValue({ status: 'ok' });
  apiMock.getSecurityHealth.mockResolvedValue({});
  apiMock.getIncidents.mockResolvedValue([]);
  apiMock.createRegistrationCode.mockResolvedValue({});
  apiMock.deleteRegistrationCode.mockResolvedValue({});
  apiMock.updateRegistrationSettings.mockResolvedValue({});
  apiMock.createUser.mockResolvedValue({});
  apiMock.createOrganization.mockResolvedValue({});
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.clearAllTimers();
  vi.resetAllMocks();
});

describe('DashboardPage refresh behavior', () => {
  it('does not start overlapping auto-refresh loads while a request is still in flight', async () => {
    vi.useFakeTimers();
    const statsDeferred = createDeferred<Record<string, unknown>>();
    apiMock.getDashboardStats.mockReturnValue(statsDeferred.promise);

    render(<DashboardPage />);

    await act(async () => {
      await Promise.resolve();
    });
    expect(apiMock.getDashboardStats).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(60_000);
    });

    expect(apiMock.getDashboardStats).toHaveBeenCalledTimes(1);

    await act(async () => {
      statsDeferred.resolve({});
      await Promise.resolve();
    });

    await act(async () => {
      vi.advanceTimersByTime(60_000);
      await Promise.resolve();
    });

    expect(apiMock.getDashboardStats).toHaveBeenCalledTimes(2);
  });
});

describe('Dashboard bulk acknowledgment', () => {
  const stamp = '2026-02-17T10:00:00.000Z';
  const receipt = (id: number, acknowledged_at: string | null) => ({
    status: 'ok', id, item: { id, acknowledged_at },
  });

  it.each([
    ['both confirmed', receipt(2, stamp), 'success'],
    ['unconfirmed response', receipt(2, null), 'warning'],
    ['different alert', receipt(9, stamp), 'warning'],
    ['request failed', new Error('Denied'), 'warning'],
  ] as const)('reports %s after checking the server response', async (_, second, level) => {
    apiMock.getAlerts.mockResolvedValue({ items: [
      { id: 1, is_read: true }, { id: 2 },
      { id: 3, acknowledged_at: stamp }, { id: 4, dismissed_at: stamp },
      { id: 5, snoozed_until: '2999-01-01T00:00:00Z' },
    ] });
    const acknowledge = vi.mocked(api.acknowledgeAlert);
    acknowledge.mockResolvedValueOnce(receipt(1, stamp));
    if (second instanceof Error) acknowledge.mockRejectedValueOnce(second);
    else acknowledge.mockResolvedValueOnce(second);

    render(<DashboardPage />);
    expect(await screen.findByText(/2 active alerts require attention/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Acknowledge All' }));
    await waitFor(() => expect(toast[level]).toHaveBeenCalledOnce());
    expect(acknowledge.mock.calls).toEqual([['1'], ['2']]);
    expect(apiMock.getAlerts).toHaveBeenCalledTimes(2);
    if (level === 'warning') {
      expect(toast.warning.mock.calls[0][1]).toContain('1 of 2');
      expect(toast.success).not.toHaveBeenCalled();
    }
  });

  it('reports total failure when all writes reject', async () => {
    apiMock.getAlerts.mockResolvedValue([{ id: 10 }]);
    vi.mocked(api.acknowledgeAlert).mockRejectedValue(new Error('Denied'));
    render(<DashboardPage />);
    fireEvent.click(await screen.findByRole('button', { name: 'Acknowledge All' }));
    await waitFor(() => expect(toast.error).toHaveBeenCalledOnce());
    expect(toast.success).not.toHaveBeenCalled();
  });

  it('only sends one acknowledgment when clicked twice before a render', async () => {
    const pending = createDeferred<ReturnType<typeof receipt>>();
    const acknowledge = vi.mocked(api.acknowledgeAlert).mockReturnValue(pending.promise);
    apiMock.getAlerts.mockResolvedValueOnce([{ id: 1 }]);

    render(<DashboardPage />);
    const button = await screen.findByRole('button', { name: 'Acknowledge All' });
    act(() => {
      button.click();
      button.click();
    });
    expect(acknowledge.mock.calls).toEqual([['1']]);
    expect(button).toBeDisabled();

    await act(async () => pending.resolve(receipt(1, stamp)));
    expect(screen.queryByRole('button', { name: 'Acknowledge All' })).toBeNull();
  });

  it('keeps confirmed alerts off the banner when an older refresh completes', async () => {
    const alert = { id: 1, message: 'Confirmed alert', is_read: true };
    const pendingStats = createDeferred<Record<string, unknown>>();
    apiMock.getDashboardStats.mockResolvedValueOnce({}).mockReturnValueOnce(pendingStats.promise);
    apiMock.getAlerts.mockResolvedValueOnce([alert]).mockResolvedValueOnce([
      alert, { ...alert, id: 2, message: 'Another alert' },
    ]);
    vi.mocked(api.acknowledgeAlert).mockResolvedValueOnce(receipt(1, stamp));

    render(<DashboardPage />);
    const button = await screen.findByRole('button', { name: 'Acknowledge All' });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    });
    fireEvent.click(button);
    await waitFor(() => expect(toast.success).toHaveBeenCalledOnce());

    await act(async () => pendingStats.resolve({}));
    expect(screen.getByTestId('dashboard-alert-count-live').textContent).toBe('1 open alert on the dashboard.');
    expect(screen.getByText(/Another alert/)).toBeInTheDocument();
    expect(screen.queryByText(/Confirmed alert/)).toBeNull();
  });

  it('keeps only the unconfirmed alert actionable when the follow-up refresh fails', async () => {
    apiMock.getAlerts.mockResolvedValueOnce([
      { id: 1, message: 'Confirmed alert', is_read: true }, { id: 2, message: 'Denied alert' },
    ]).mockRejectedValueOnce(new Error('Unavailable'));
    const acknowledge = vi.mocked(api.acknowledgeAlert)
      .mockResolvedValueOnce(receipt(1, stamp))
      .mockRejectedValueOnce(new Error('Denied'));

    render(<DashboardPage />);
    fireEvent.click(await screen.findByRole('button', { name: 'Acknowledge All' }));
    expect(await screen.findByText('Some dashboard data failed to load: alerts')).toBeInTheDocument();
    expect(toast.warning).toHaveBeenCalledWith('Alerts partially acknowledged', '1 of 2 visible alerts confirmed acknowledged.');
    expect(screen.getByTestId('dashboard-alert-count-live').textContent).toBe('1 open alert on the dashboard.');
    expect(screen.getByText(/Denied alert/)).toBeInTheDocument();
    expect(screen.queryByText(/Confirmed alert/)).toBeNull();

    acknowledge.mockResolvedValueOnce(receipt(2, stamp));
    fireEvent.click(screen.getByRole('button', { name: 'Acknowledge All' }));
    await waitFor(() => expect(toast.success).toHaveBeenCalledOnce());
    expect(acknowledge.mock.calls).toEqual([['1'], ['2'], ['2']]);
  });
});
