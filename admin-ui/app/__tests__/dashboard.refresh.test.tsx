/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render } from '@testing-library/react';
import DashboardPage from '../page';
import { api } from '@/lib/api-client';

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
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
  }),
}));

vi.mock('@/components/ui/accessible-icon-button', () => ({
  AccessibleIconButton: ({ label, onClick }: { label: string; onClick: () => void }) => (
    <button type="button" onClick={onClick}>
      {label}
    </button>
  ),
}));

vi.mock('@/components/dashboard/DashboardHeader', () => ({
  DashboardHeader: () => <section aria-label="Dashboard header">Header</section>,
}));

vi.mock('@/components/dashboard/AlertsBanner', () => ({
  AlertsBanner: () => <section aria-label="Dashboard alerts">Alerts</section>,
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
    getRealtimeStats: vi.fn(),
    getUsers: vi.fn(),
    getOrganizations: vi.fn(),
    getLLMProviders: vi.fn(),
    getAuditLogs: vi.fn(),
    getAlerts: vi.fn(),
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
  getRealtimeStats: ReturnType<typeof vi.fn>;
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
  apiMock.getRealtimeStats.mockResolvedValue({
    active_sessions: 0,
    tokens_today: { prompt: 0, completion: 0, total: 0 },
  });
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
